#include "scar/copula/multivariate/correlation/factor.hpp"
#include "scar/copula/multivariate/rosenblatt.hpp"
#include "scar/copula/multivariate/student/rosenblatt.hpp"
#include "scar/detail/parallel.hpp"

#include <algorithm>
#include <atomic>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace {

constexpr std::size_t dimension = 40;
constexpr std::size_t fixture_rows = 65;
constexpr std::size_t failed_row = 17;

void require(bool value, const char* message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}

class RuntimeScope {
public:
    RuntimeScope() { scar_internal::shutdown_parallel_runtime(); }
    ~RuntimeScope() { scar_internal::shutdown_parallel_runtime(); }
};

scar::ObservationView view(const std::vector<double>& values) {
    return {values.data(), values.size() / dimension, static_cast<int>(dimension)};
}

scar::DoubleView flat(const std::vector<double>& values) {
    return {values.data(), values.size()};
}

std::vector<double> make_loadings(std::size_t rank, bool high, bool zero) {
    std::vector<double> values(dimension * rank, 0.0);
    if (zero) {
        return values;
    }
    for (std::size_t row = 0; row < dimension; ++row) {
        double norm_squared = 0.0;
        for (std::size_t column = 0; column < rank; ++column) {
            const double value = static_cast<double>((row * 3 + column * 7) % 17) - 8.5;
            values[row * rank + column] = value;
            norm_squared += value * value;
        }
        const double scale = (high ? 0.97 : 0.23) / std::sqrt(norm_squared);
        for (std::size_t column = 0; column < rank; ++column) {
            values[row * rank + column] *= scale;
        }
    }
    return values;
}

std::vector<double> make_observations(std::size_t rows, bool tails) {
    std::vector<double> values(rows * dimension);
    for (std::size_t index = 0; index < values.size(); ++index) {
        values[index] = 0.02 + 0.96 * static_cast<double>((index * 37) % 997) / 996.0;
        if (tails && index % 23 == 0) {
            values[index] = index % 2 == 0 ? 0.0 : 1.0;
        }
    }
    return values;
}

scar::MultivariateRosenblattResult evaluate(
    const scar::FactorCorrelationOperator& factor,
    const std::vector<double>& values,
    const std::vector<double>& df,
    int request,
    bool student) {
    return student
        ? scar::student_rosenblatt_factor(factor, view(values), flat(df), request)
        : scar::gaussian_rosenblatt_factor(factor, view(values), request);
}

void exact_output(const scar::MultivariateRosenblattResult& actual,
                  const scar::MultivariateRosenblattResult& reference) {
    require(actual.status == reference.status
            && actual.failure.index == reference.failure.index
            && actual.failure.coordinate == reference.failure.coordinate
            && actual.n_rows == reference.n_rows
            && actual.dimension == reference.dimension
            && actual.residuals.size() == reference.residuals.size()
            && std::memcmp(actual.residuals.data(), reference.residuals.data(),
                           actual.residuals.size() * sizeof(double)) == 0,
            "factor Rosenblatt output or failure changed across workers");
}

void successful_calls() {
    RuntimeScope runtime;
    struct Case {
        std::size_t rows;
        std::size_t rank;
        bool tails;
        bool high;
        bool zero;
        bool path_df;
    };
    const Case cases[] = {
        {65, 1, false, false, false, false},
        {257, 4, false, false, false, true},
        {65, 31, false, false, false, false},
        {65, 32, false, false, false, true},
        {65, 33, false, false, false, false},
        {65, 4, true, true, false, true},
        {65, 1, true, false, true, false},
    };
    for (const auto& item : cases) {
        const scar::FactorCorrelationOperator factor(
            make_loadings(item.rank, item.high, item.zero), dimension, item.rank, 1e-8);
        const auto values = make_observations(item.rows, item.tails);
        std::vector<double> df(item.path_df ? item.rows : 1, 5.7);
        if (item.path_df) {
            for (std::size_t row = 0; row < item.rows; ++row) {
                df[row] = row % 2 == 0 ? 3.25 : 1200.0;
            }
        }
        for (bool student : {false, true}) {
            scar_internal::shutdown_parallel_runtime();
            const auto reference = evaluate(factor, values, df, 1, student);
            require(reference.is_ok(), "serial factor Rosenblatt reference failed");
            require(!scar_internal::parallel_runtime_info().initialized,
                    "serial factor Rosenblatt initialized the pool");
            for (int request : {4, 17, 32}) {
                const auto before = scar_internal::parallel_runtime_info();
                const auto actual = evaluate(factor, values, df, request, student);
                const auto after = scar_internal::parallel_runtime_info();
                exact_output(actual, reference);
                const std::size_t blocks = std::min(
                    static_cast<std::size_t>(request), (item.rows + 15) / 16);
                require(after.batches_submitted - before.batches_submitted == 1
                        && after.tasks_submitted - before.tasks_submitted == blocks,
                        "factor Rosenblatt did not use one bounded row batch");
                require(actual.parallel_blocks == request,
                        "legacy factor Rosenblatt diagnostic changed meaning");
            }
        }
    }
}

void independent_dense_reference() {
    const scar::FactorCorrelationOperator factor(
        make_loadings(4, false, false), dimension, 4, 1e-8);
    const auto values = make_observations(65, false);
    std::vector<double> matrix(dimension * dimension, 0.0);
    for (std::size_t row = 0; row < dimension; ++row) {
        for (std::size_t column = 0; column < dimension; ++column) {
            double value = row == column ? 1.0 : 0.0;
            if (row != column) {
                for (std::size_t component = 0; component < 4; ++component) {
                    value += factor.loadings()[row * 4 + component]
                        * factor.loadings()[column * 4 + component];
                }
            }
            matrix[row * dimension + column] = value;
        }
    }
    const std::vector<double> df{5.7};
    for (bool student : {false, true}) {
        const auto actual = evaluate(factor, values, df, 4, student);
        const auto dense = student
            ? scar::student_rosenblatt_dense(flat(matrix), static_cast<int>(dimension),
                                            view(values), flat(df), 1)
            : scar::gaussian_rosenblatt_dense(flat(matrix), static_cast<int>(dimension),
                                             view(values), 1);
        require(actual.is_ok() && dense.is_ok(), "dense Rosenblatt reference failed");
        for (std::size_t index = 0; index < values.size(); ++index) {
            require(std::abs(actual.residuals[index] - dense.residuals[index]) < 3e-11,
                    "factor Rosenblatt disagrees with independent dense algorithm");
        }
    }
}

std::vector<double> numerical_values(std::size_t coordinate) {
    std::vector<double> values(fixture_rows * dimension, 0.5);
    values[failed_row * dimension + coordinate] = 0.7;
    return values;
}

std::vector<double> numerical_df() {
    std::vector<double> degrees(fixture_rows, 5.7);
    degrees[failed_row] = 1e308;
    return degrees;
}

std::size_t block_end_for_row(std::size_t row, std::size_t blocks) {
    const std::size_t quotient = fixture_rows / blocks;
    const std::size_t remainder = fixture_rows % blocks;
    std::size_t end = 0;
    for (std::size_t block = 0; block < blocks; ++block) {
        end += quotient + (block < remainder ? 1 : 0);
        if (row < end) {
            return end;
        }
    }
    throw std::runtime_error("fixture row is outside all blocks");
}

void numerical_pattern(const scar::MultivariateRosenblattResult& result,
                       std::size_t coordinate, std::size_t blocks) {
    require(result.status == scar::Status::NumericalFailure
            && result.failure.index == static_cast<std::int64_t>(failed_row)
            && result.failure.coordinate == static_cast<int>(coordinate),
            "real numerical failure lost its row/coordinate");
    const std::size_t stopped_end = block_end_for_row(failed_row, blocks);
    for (std::size_t row = 0; row < fixture_rows; ++row) {
        for (std::size_t column = 0; column < dimension; ++column) {
            const double expected = column < coordinate
                || (column == coordinate && (row < failed_row || row >= stopped_end))
                ? 0.5 : 0.0;
            require(result.residuals[row * dimension + column] == expected,
                    "numerical failure changed original row-block output prefix");
        }
    }
}

void real_numerical_failure_and_nested() {
    RuntimeScope runtime;
    const scar::FactorCorrelationOperator factor(
        std::vector<double>(dimension, 0.0), dimension, 1, 1e-8);
    const auto values = numerical_values(3);
    const auto df = numerical_df();
    for (int request : {1, 4, 17, 32}) {
        const auto result = evaluate(factor, values, df, request, true);
        numerical_pattern(result, 3, request == 1 ? 1 : std::min(
            static_cast<std::size_t>(request), (fixture_rows + 15) / 16));
    }
    const auto clean_values = make_observations(65, false);
    const std::vector<double> clean_df{5.7};
    const auto reference = evaluate(factor, clean_values, clean_df, 1, true);
    std::atomic<int> completed{0};
    scar_internal::parallel_for_blocks(0, 2, 1, 2,
        [&](std::int64_t, std::int64_t, std::size_t) {
            const auto before = scar_internal::parallel_runtime_info();
            numerical_pattern(evaluate(factor, values, df, 4, true), 3, 1);
            exact_output(evaluate(factor, clean_values, clean_df, 4, true), reference);
            const auto after = scar_internal::parallel_runtime_info();
            require(before.tasks_submitted == after.tasks_submitted
                    && before.batches_submitted == after.batches_submitted,
                    "nested factor Rosenblatt submitted inner work");
            completed.fetch_add(1);
        });
    require(completed.load() == 2, "nested factor Rosenblatt calls did not complete");
}

class Event {
public:
    void signal() noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        signalled_ = true;
        ready_.notify_all();
    }
    bool wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        return ready_.wait_for(lock, std::chrono::seconds(10), [&] { return signalled_; });
    }
private:
    std::mutex mutex_;
    std::condition_variable ready_;
    bool signalled_ = false;
};

struct FirstError : std::exception {
    const char* what() const noexcept override { return "first recorded Rosenblatt error"; }
};
struct LaterError : std::exception {
    const char* what() const noexcept override { return "later recorded Rosenblatt error"; }
};

enum class FaultCase {
    NumericalThenException,
    ExceptionThenNumerical,
    SameCoordinate,
    RegistrationOrder,
    Preparation,
    Update,
    NumericalThenUpdate,
    EnvironmentApply,
    PreparationBeforeEnvironmentApply,
};

using Operation = scar::rosenblatt_testing::Operation;

struct FaultState {
    explicit FaultState(FaultCase selected) : selected(selected) {
        output.residuals.resize(fixture_rows * dimension, -99.0);
    }
    FaultCase selected;
    Event registered;
    Event numerical_entered;
    std::atomic<std::size_t> first_registration{std::numeric_limits<std::size_t>::max()};
    std::atomic<std::size_t> later_registration{std::numeric_limits<std::size_t>::max()};
    std::atomic<int> thrown_hooks{0};
    std::atomic<int> block_two_apply_attempts{0};
    bool shared_gaussian_preparation = false;
    scar::MultivariateRosenblattResult output;
    bool observed = false;
    bool observation_shape_ok = false;
};

std::atomic<FaultState*> fault_state{nullptr};

void record_error(std::size_t, Operation, std::size_t block,
                  std::size_t registration) noexcept {
    auto* state = fault_state.load();
    if (block == 2) {
        state->first_registration.store(registration);
        state->registered.signal();
    } else if (block == 1) {
        state->later_registration.store(registration);
    }
}

void observe_output(const scar::MultivariateRosenblattResult& result) noexcept {
    auto* state = fault_state.load();
    state->observed = true;
    state->observation_shape_ok = result.residuals.size() == state->output.residuals.size();
    if (state->observation_shape_ok) {
        std::copy(result.residuals.begin(), result.residuals.end(), state->output.residuals.begin());
    }
    state->output.status = result.status;
    state->output.failure = result.failure;
    state->output.n_rows = result.n_rows;
    state->output.dimension = result.dimension;
}

void inject_fault(std::size_t coordinate, Operation operation,
                  std::size_t block, std::int64_t row) {
    auto& state = *fault_state.load();
    if (state.selected == FaultCase::PreparationBeforeEnvironmentApply
        && state.shared_gaussian_preparation && block == 0 && coordinate == 0
        && operation == Operation::Preparation) {
        state.thrown_hooks.fetch_add(1);
        throw FirstError{};
    }
    if (state.selected == FaultCase::PreparationBeforeEnvironmentApply
        && !state.shared_gaussian_preparation
        && block == 1 && coordinate == 0 && operation == Operation::Preparation) {
        require(state.registered.wait(), "environment failure did not register first");
        state.thrown_hooks.fetch_add(1);
        throw FirstError{};
    }
    const bool numerical = operation == Operation::Rows && coordinate == 3
        && block == 1 && row == static_cast<std::int64_t>(failed_row);
    if (numerical && (state.selected == FaultCase::NumericalThenException
                      || state.selected == FaultCase::SameCoordinate
                      || state.selected == FaultCase::NumericalThenUpdate)) {
        require(state.registered.wait(), "later-coordinate exception did not register");
    }
    if (numerical && state.selected == FaultCase::ExceptionThenNumerical) {
        state.numerical_entered.signal();
    }
    bool first = false;
    if (block == 2) {
        switch (state.selected) {
        case FaultCase::NumericalThenException:
            first = operation == Operation::Rows && coordinate == 4 && row == 35;
            break;
        case FaultCase::ExceptionThenNumerical:
            if (operation == Operation::Rows && coordinate == 2 && row == 35) {
                require(state.numerical_entered.wait(), "later numerical row was not attempted");
                first = true;
            }
            break;
        case FaultCase::SameCoordinate:
            first = operation == Operation::Rows && coordinate == 3 && row == 35;
            break;
        case FaultCase::RegistrationOrder:
            first = operation == Operation::Rows && coordinate == 2 && row == 35;
            break;
        case FaultCase::Preparation:
            first = operation == Operation::Preparation && coordinate == 6;
            break;
        case FaultCase::Update:
        case FaultCase::NumericalThenUpdate:
            first = operation == Operation::Update && coordinate == 3;
            break;
        case FaultCase::EnvironmentApply:
        case FaultCase::PreparationBeforeEnvironmentApply:
            break;
        }
    }
    if (first) {
        state.thrown_hooks.fetch_add(1);
        throw FirstError{};
    }
    if (state.selected == FaultCase::RegistrationOrder && block == 1
        && operation == Operation::Rows && coordinate == 2 && row == 18) {
        require(state.registered.wait(), "first exception was not recorded before the second");
        state.thrown_hooks.fetch_add(1);
        throw LaterError{};
    }
}

class HookScope {
public:
    explicit HookScope(FaultState& state) {
        fault_state.store(&state);
        scar::rosenblatt_testing::set_hooks(inject_fault, observe_output, record_error);
    }
    ~HookScope() {
        scar::rosenblatt_testing::set_hooks(nullptr, nullptr);
        fault_state.store(nullptr);
    }
    HookScope(const HookScope&) = delete;
    HookScope& operator=(const HookScope&) = delete;
};

void check_exception_output(const FaultState& state, std::size_t coordinate,
                            bool entire_coordinate_zero, bool entire_coordinate_done,
                            bool second_failed_block) {
    require(state.observed && state.observation_shape_ok,
            "exception observer did not receive the joined, repaired result");
    for (std::size_t row = 0; row < fixture_rows; ++row) {
        for (std::size_t column = 0; column < dimension; ++column) {
            double expected = column < coordinate ? 0.5 : 0.0;
            if (column == coordinate && !entire_coordinate_zero) {
                const bool stopped = !entire_coordinate_done
                    && ((row >= 35 && row < 49)
                        || (second_failed_block && row >= 17 && row < 33));
                expected = stopped ? 0.0 : 0.5;
                if (state.selected == FaultCase::RegistrationOrder && row == 17) {
                    expected = 0.5;
                }
            }
            require(state.output.residuals[row * dimension + column] == expected,
                    "exception output did not preserve the original coordinate/block prefix");
        }
    }
}

void ordered_outcomes() {
    RuntimeScope runtime;
    const scar::FactorCorrelationOperator factor(
        std::vector<double>(dimension, 0.0), dimension, 1, 1e-8);
    for (FaultCase selected : {
             FaultCase::NumericalThenException, FaultCase::ExceptionThenNumerical,
             FaultCase::SameCoordinate, FaultCase::RegistrationOrder,
             FaultCase::Preparation, FaultCase::Update, FaultCase::NumericalThenUpdate}) {
        FaultState state(selected);
        const bool numerical = selected == FaultCase::NumericalThenException
            || selected == FaultCase::ExceptionThenNumerical
            || selected == FaultCase::SameCoordinate
            || selected == FaultCase::NumericalThenUpdate;
        const auto values = numerical ? numerical_values(3)
            : std::vector<double>(fixture_rows * dimension, 0.5);
        const auto df = numerical ? numerical_df() : std::vector<double>{5.7};
        bool first_caught = false;
        scar::MultivariateRosenblattResult result;
        {
            HookScope hooks(state);
            try {
                result = evaluate(factor, values, df, 4, true);
            } catch (const FirstError&) {
                first_caught = true;
            }
        }
        const bool numeric_wins = selected == FaultCase::NumericalThenException
            || selected == FaultCase::NumericalThenUpdate;
        require(first_caught != numeric_wins,
                "Rosenblatt selected the wrong numerical/exception outcome");
        require(state.thrown_hooks.load() == (selected == FaultCase::RegistrationOrder ? 2 : 1),
                "mixed-outcome test did not exercise every requested exception");
        if (numeric_wins) {
            numerical_pattern(result, 3, 4);
            require(state.observed && state.observation_shape_ok,
                    "numerical result was not observed after repair");
            numerical_pattern(state.output, 3, 4);
        } else if (selected == FaultCase::Preparation) {
            check_exception_output(state, 6, true, false, false);
        } else if (selected == FaultCase::Update) {
            check_exception_output(state, 3, false, true, false);
        } else {
            const std::size_t coordinate = selected == FaultCase::SameCoordinate ? 3 : 2;
            check_exception_output(state, coordinate, false, false,
                selected == FaultCase::SameCoordinate || selected == FaultCase::RegistrationOrder);
        }
        if (selected == FaultCase::RegistrationOrder) {
            require(state.first_registration.load() < state.later_registration.load(),
                    "exception registration ordering was not exercised");
            require(state.output.failure.index == 35,
                    "lower block ID displaced the first registered exception");
        }
    }
}

bool fail_environment_apply_block_two(std::size_t block, bool restoring) noexcept {
    if (!restoring && block == 2) {
        fault_state.load()->block_two_apply_attempts.fetch_add(1);
    }
    return !restoring && block == 2;
}

class EnvironmentHookScope {
public:
    EnvironmentHookScope() {
        scar_internal::parallel_testing::set_environment_failure_hook(
            fail_environment_apply_block_two);
    }
    ~EnvironmentHookScope() {
        scar_internal::parallel_testing::set_environment_failure_hook(nullptr);
    }
};

void environment_failure_outcomes() {
    RuntimeScope runtime;
    const scar::FactorCorrelationOperator factor(
        std::vector<double>(dimension * 16, 0.0), dimension, 16, 1e-8);
    const std::vector<double> values(fixture_rows * dimension, 0.5);
    const std::vector<double> df{5.7};
    for (bool student : {false, true}) {
        for (bool preparation : {false, true}) {
            FaultState state(preparation ? FaultCase::PreparationBeforeEnvironmentApply
                                         : FaultCase::EnvironmentApply);
            state.shared_gaussian_preparation = preparation && !student;
            bool preparation_caught = false;
            bool application_caught = false;
            const auto before = scar_internal::parallel_runtime_info();
            {
                HookScope hooks(state);
                EnvironmentHookScope environment_hook;
                try {
                    (void)evaluate(factor, values, df, 4, student);
                } catch (const FirstError&) {
                    preparation_caught = true;
                } catch (const std::runtime_error& error) {
                    application_caught = std::strstr(
                        error.what(), "apply caller floating-point environment") != nullptr;
                    require(application_caught, "unexpected environment outcome exception");
                }
            }
            require(preparation_caught == preparation && application_caught != preparation,
                    "environment failure bypassed coordinate/operation selection");
            require(state.observed && state.observation_shape_ok
                    && state.output.status == scar::Status::NumericalFailure
                    && state.output.failure.coordinate == 0
                    && state.output.failure.index == (preparation ? -1 : 33),
                    "environment failure lost its first-coordinate row context");
            require(state.thrown_hooks.load() == (preparation ? 1 : 0),
                    "environment case did not exercise its requested preparation hook");
            if (state.shared_gaussian_preparation) {
                const auto after = scar_internal::parallel_runtime_info();
                require(state.block_two_apply_attempts.load() == 0
                        && before.batches_submitted == after.batches_submitted
                        && before.tasks_submitted == after.tasks_submitted,
                        "coordinate-zero Gaussian preparation failure published row work");
            } else if (preparation) {
                require(state.first_registration.load() < state.later_registration.load(),
                        "preparation priority was not tested after an earlier apply record");
            }
            // The original 65-row/four-block geometry is [0,17), [17,33),
            // [33,49), [49,65). A failed application skips all of block two.
            for (std::size_t row = 0; row < fixture_rows; ++row) {
                for (std::size_t coordinate = 0; coordinate < dimension; ++coordinate) {
                    const double expected = !preparation && coordinate == 0
                        && (row < 33 || row >= 49) ? 0.5 : 0.0;
                    require(state.output.residuals[row * dimension + coordinate] == expected,
                            "environment failure changed the repaired raw output");
                }
            }
        }
    }
}

class EnvironmentScope {
public:
    EnvironmentScope() { require(std::fegetenv(&saved_) == 0, "cannot capture test fenv"); }
    ~EnvironmentScope() { (void)std::fesetenv(&saved_); }
private:
    std::fenv_t saved_{};
};

void caller_environment() {
    RuntimeScope runtime;
    const scar::FactorCorrelationOperator factor(
        make_loadings(4, false, false), dimension, 4, 1e-8);
    const auto values = make_observations(65, false);
    const std::vector<double> df{5.7};
    EnvironmentScope environment;
    for (bool student : {false, true}) {
        require(std::fesetround(FE_DOWNWARD) == 0
                && std::feclearexcept(FE_ALL_EXCEPT) == 0
                && std::feraiseexcept(FE_DIVBYZERO) == 0,
                "cannot install test rounding/flags");
        const int flags = std::fetestexcept(FE_ALL_EXCEPT);
        const auto result = evaluate(factor, values, df, 4, student);
        const int after_flags = std::fetestexcept(FE_ALL_EXCEPT);
        const int after_rounding = std::fegetround();
        require(result.is_ok(), "non-default rounding failed factor Rosenblatt");
        require(after_rounding == FE_DOWNWARD && after_flags == flags,
                "queued factor Rosenblatt changed the waiting caller's fenv");
        require(std::fesetround(FE_DOWNWARD) == 0
                && std::feclearexcept(FE_ALL_EXCEPT) == 0
                && std::feraiseexcept(FE_DIVBYZERO) == 0,
                "cannot restore serial reference environment");
        exact_output(evaluate(factor, values, df, 1, student), result);
    }
}

void validation_before_input_access() {
    RuntimeScope runtime;
    const scar::FactorCorrelationOperator factor(
        std::vector<double>(dimension, 0.0), dimension, 1, 1e-8);
    const double dummy = 0.5;
    const double degrees = 5.7;
    const scar::ObservationView too_large{
        &dummy, std::numeric_limits<std::size_t>::max() / sizeof(double) / dimension + 1,
        static_cast<int>(dimension)};
    const auto gaussian = scar::gaussian_rosenblatt_factor(factor, too_large, 4);
    const auto student = scar::student_rosenblatt_factor(factor, too_large, {&degrees, 1}, 4);
    require(gaussian.status == scar::Status::InvalidSize && gaussian.residuals.empty()
            && student.status == scar::Status::InvalidSize && student.residuals.empty(),
            "Rosenblatt byte overflow was not rejected before reading input");
    const std::vector<double> values(16 * dimension, 0.5);
    const auto null_df = scar::student_rosenblatt_factor(factor, view(values), {nullptr, 1}, 4);
    require(null_df.status == scar::Status::InvalidSize && null_df.residuals.empty(),
            "nonempty null df view was not rejected");
    for (bool is_student : {false, true}) {
        const auto empty = evaluate(factor, {}, {degrees}, 4, is_student);
        const auto small = evaluate(factor, values, {degrees}, 4, is_student);
        require(empty.is_ok() && empty.residuals.empty() && small.is_ok(),
                "empty or small Rosenblatt direct call failed");
    }
    require(!scar_internal::parallel_runtime_info().initialized,
            "invalid, empty, or small Rosenblatt call initialized the pool");
}

}  // namespace

int run_factor_rosenblatt_parallel_tests() {
    try {
        successful_calls();
        independent_dense_reference();
        real_numerical_failure_and_nested();
        ordered_outcomes();
        environment_failure_outcomes();
        caller_environment();
        validation_before_input_access();
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "factor Rosenblatt parallel regression: %s\n", error.what());
        return 1;
    }
}
