#include "scar/copula/multivariate/correlation/factor.hpp"
#include "scar/copula/multivariate/rosenblatt.hpp"
#include "scar/detail/parallel.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cfenv>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <initializer_list>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace {
constexpr std::size_t rows = 65;
constexpr std::size_t dimension = 40;
using Operation = scar::rosenblatt_testing::Operation;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

struct PreparationError : std::exception {
    const char* what() const noexcept override { return "Gaussian preparation error"; }
};
struct FirstRowError : std::exception {
    const char* what() const noexcept override { return "first Gaussian row error"; }
};
struct LaterRowError : std::exception {
    const char* what() const noexcept override { return "later Gaussian row error"; }
};

enum class Scenario {
    CountPreparation,
    PreparationOnly,
    PreparationAndRow,
    UpdateOnly,
    UpdateAndRow,
    EarlierRowException,
    SameCoordinateFirstRow,
    CrossBlockRegistration,
    EarlierInjectedNumerical,
    EarlierExceptionAfterInjected,
    EnvironmentMutation,
    GuardApplyFailure,
    GuardRestoreFailure,
    PreparationAndRestoreFailure,
};

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

struct State {
    explicit State(Scenario scenario)
        : scenario(scenario), values(rows * dimension, 0.5), caller(std::this_thread::get_id()) {
        output.residuals.resize(values.size(), -1.0);
        for (auto& value : preparation_calls) value.store(0);
        for (auto& value : update_calls) value.store(0);
        for (auto& value : row_visits) value.store(0);
    }
    Scenario scenario;
    std::vector<double> values;
    std::thread::id caller;
    std::array<std::atomic<int>, dimension> preparation_calls;
    std::array<std::atomic<int>, dimension> update_calls;
    std::array<std::atomic<int>, rows * dimension> row_visits;
    std::atomic<int> preparation_wrong_context{0};
    std::atomic<int> rows_on_caller{0};
    std::atomic<int> rows_wrong_environment{0};
    std::atomic<int> injected_values{0};
    std::atomic<std::size_t> first_registration{std::numeric_limits<std::size_t>::max()};
    std::atomic<std::size_t> later_registration{std::numeric_limits<std::size_t>::max()};
    std::atomic<bool> global_preparation_recorded{false};
    Event first_recorded;
    scar::MultivariateRosenblattResult output;
    bool observed = false;
    bool valid_observation_shape = false;
};

std::atomic<State*> active_state{nullptr};

void record_exception(std::size_t, Operation operation, std::size_t block,
                      std::size_t registration) noexcept {
    auto& state = *active_state.load();
    if (block == 0 && operation != Operation::Rows) {
        state.global_preparation_recorded.store(true);
    }
    if (block == 2) {
        state.first_registration.store(registration);
        state.first_recorded.signal();
    } else if (block == 1) {
        state.later_registration.store(registration);
    }
}

void observe(const scar::MultivariateRosenblattResult& result) noexcept {
    auto& state = *active_state.load();
    state.observed = true;
    state.valid_observation_shape = result.residuals.size() == state.output.residuals.size();
    if (state.valid_observation_shape) {
        std::copy(result.residuals.begin(), result.residuals.end(), state.output.residuals.begin());
    }
    state.output.status = result.status;
    state.output.failure = result.failure;
}

void hook(std::size_t coordinate, Operation operation, std::size_t block,
          std::int64_t row_index) {
    auto& state = *active_state.load();
    const Scenario scenario = state.scenario;
    if (operation != Operation::Rows) {
        if (block != 0 || std::this_thread::get_id() != state.caller) {
            state.preparation_wrong_context.fetch_add(1);
        }
        auto& counts = operation == Operation::Preparation
            ? state.preparation_calls : state.update_calls;
        counts[coordinate].fetch_add(1);
        if (operation == Operation::Preparation && coordinate == 0
            && (scenario == Scenario::EnvironmentMutation
                || scenario == Scenario::PreparationAndRestoreFailure)) {
            require(std::fesetround(FE_UPWARD) == 0
                    && std::feclearexcept(FE_ALL_EXCEPT) == 0
                    && std::feraiseexcept(FE_INVALID) == 0,
                    "cannot mutate Gaussian preparation environment");
        }
        if (coordinate == 6
            && ((operation == Operation::Preparation
                 && (scenario == Scenario::PreparationOnly
                     || scenario == Scenario::PreparationAndRow
                     || scenario == Scenario::PreparationAndRestoreFailure))
                || (operation == Operation::Update
                    && (scenario == Scenario::UpdateOnly
                        || scenario == Scenario::UpdateAndRow)))) {
            throw PreparationError{};
        }
        return;
    }

    const auto row = static_cast<std::size_t>(row_index);
    state.row_visits[row * dimension + coordinate].fetch_add(1);
    if (std::this_thread::get_id() == state.caller) {
        state.rows_on_caller.fetch_add(1);
    }
    if (scenario == Scenario::EnvironmentMutation
        && (std::fegetround() != FE_DOWNWARD
            || (std::fetestexcept(FE_DIVBYZERO) & FE_DIVBYZERO) == 0)) {
        state.rows_wrong_environment.fetch_add(1);
    }
    if ((scenario == Scenario::PreparationAndRow || scenario == Scenario::UpdateAndRow)
        && block == 1 && row == 18 && coordinate == 2) {
        throw FirstRowError{};
    }
    // Block one starts at row17. These competing failures straddle the
    // four-row tiles [17,21) and [21,25), so both events must execute.
    if ((scenario == Scenario::EarlierRowException
         || scenario == Scenario::EarlierInjectedNumerical)
        && block == 1 && row == 17 && coordinate == 7) {
        throw LaterRowError{};
    }
    if ((scenario == Scenario::EarlierRowException
         || scenario == Scenario::EarlierExceptionAfterInjected)
        && block == 1 && row == 21 && coordinate == 2) {
        throw FirstRowError{};
    }
    if (scenario == Scenario::SameCoordinateFirstRow
        && block == 1 && coordinate == 3 && (row == 17 || row == 18)) {
        if (row == 17) throw FirstRowError{};
        throw LaterRowError{};
    }
    if (scenario == Scenario::CrossBlockRegistration && coordinate == 3) {
        if (block == 2 && row == 35) throw FirstRowError{};
        if (block == 1 && row == 18) {
            require(state.first_recorded.wait(), "first Gaussian block error did not register");
            throw LaterRowError{};
        }
    }
    const bool inject_earlier = scenario == Scenario::EarlierInjectedNumerical
        && block == 1 && row == 21 && coordinate == 2;
    const bool inject_later = scenario == Scenario::EarlierExceptionAfterInjected
        && block == 1 && row == 17 && coordinate == 7;
    if (inject_earlier || inject_later) {
        // Deliberate post-validation input injection, not a valid-input failure.
        // Only this block reads its own row, and validation has already ended.
        state.values[row * dimension + coordinate] = std::numeric_limits<double>::quiet_NaN();
        state.injected_values.fetch_add(1);
    }
}

bool environment_fault(std::size_t block, bool restoring) noexcept {
    const auto scenario = active_state.load()->scenario;
    if (scenario == Scenario::GuardApplyFailure) return !restoring && block == 0;
    return restoring && (scenario == Scenario::GuardRestoreFailure
                         || scenario == Scenario::PreparationAndRestoreFailure);
}

class Hooks {
public:
    explicit Hooks(State& state) {
        active_state.store(&state);
        scar::rosenblatt_testing::set_hooks(hook, observe, record_exception);
        scar_internal::parallel_testing::set_environment_failure_hook(environment_fault);
    }
    ~Hooks() {
        scar_internal::parallel_testing::set_environment_failure_hook(nullptr);
        scar::rosenblatt_testing::set_hooks(nullptr, nullptr);
        active_state.store(nullptr);
    }
};

class Environment {
public:
    Environment() { require(std::fegetenv(&saved_) == 0, "cannot save Gaussian test environment"); }
    ~Environment() { (void)std::fesetenv(&saved_); }
private:
    std::fenv_t saved_{};
};

class Runtime {
public:
    Runtime() { scar_internal::shutdown_parallel_runtime(); }
    ~Runtime() { scar_internal::shutdown_parallel_runtime(); }
};

void pattern(const State& state, std::size_t coordinate, std::int64_t failed_row,
             bool empty_coordinate,
             std::initializer_list<std::pair<std::size_t, std::size_t>> stopped) {
    require(state.observed && state.valid_observation_shape
            && state.output.status == scar::Status::NumericalFailure
            && state.output.failure.coordinate == static_cast<int>(coordinate)
            && state.output.failure.index == failed_row,
            "Gaussian observer lost the selected error context");
    for (std::size_t row = 0; row < rows; ++row) {
        bool stopped_row = empty_coordinate;
        for (const auto& range : stopped) {
            stopped_row = stopped_row || (row >= range.first && row < range.second);
        }
        for (std::size_t column = 0; column < dimension; ++column) {
            const double expected = column < coordinate
                || (column == coordinate && !stopped_row) ? 0.5 : 0.0;
            require(state.output.residuals[row * dimension + column] == expected,
                    "Gaussian prepared rows changed the original raw output prefix");
        }
    }
}

void require_strict_prefix(const State& state, std::size_t first_later_row,
                           std::size_t end_row, std::size_t coordinate) {
    for (std::size_t row = first_later_row; row < end_row; ++row) {
        for (std::size_t column = 0; column < coordinate; ++column) {
            require(state.row_visits[row * dimension + column].load() == 1,
                    "later Gaussian row did not evaluate the necessary earlier prefix");
        }
        for (std::size_t column = coordinate; column < dimension; ++column) {
            require(state.row_visits[row * dimension + column].load() == 0,
                    "later Gaussian row evaluated the block's failed coordinate or suffix");
        }
    }
}

enum class Caught { None, Preparation, FirstRow, LaterRow, Apply, Restore, Composite };

Caught execute(State& state, const scar::FactorCorrelationOperator& factor) {
    Hooks hooks(state);
    try {
        (void)scar::gaussian_rosenblatt_factor(factor,
            {state.values.data(), rows, static_cast<int>(dimension)}, 4);
    } catch (const scar_internal::ParallelEnvironmentRestoreError& error) {
        bool primary_ok = false;
        bool restore_ok = false;
        try { error.rethrow_primary(); }
        catch (const PreparationError&) { primary_ok = true; }
        try { std::rethrow_exception(error.restore_exception()); }
        catch (const std::runtime_error& restored) {
            restore_ok = std::strstr(restored.what(), "restore entry floating-point environment") != nullptr;
        }
        require(primary_ok && restore_ok, "Gaussian preparation/restore double failure lost a cause");
        return Caught::Composite;
    } catch (const PreparationError&) { return Caught::Preparation; }
    catch (const FirstRowError&) { return Caught::FirstRow; }
    catch (const LaterRowError&) { return Caught::LaterRow; }
    catch (const std::runtime_error& error) {
        if (std::strstr(error.what(), "apply caller floating-point environment")) return Caught::Apply;
        if (std::strstr(error.what(), "restore entry floating-point environment")) return Caught::Restore;
        throw;
    }
    return Caught::None;
}

void prepared_outcomes() {
    Runtime runtime;
    Environment environment;
    const scar::FactorCorrelationOperator factor(
        std::vector<double>(dimension * 16, 0.0), dimension, 16, 1e-8);
    for (Scenario scenario : {
        Scenario::CountPreparation, Scenario::PreparationOnly, Scenario::PreparationAndRow,
        Scenario::UpdateOnly, Scenario::UpdateAndRow, Scenario::EarlierRowException,
        Scenario::SameCoordinateFirstRow, Scenario::CrossBlockRegistration,
        Scenario::EarlierInjectedNumerical, Scenario::EarlierExceptionAfterInjected,
        Scenario::EnvironmentMutation, Scenario::GuardApplyFailure,
        Scenario::GuardRestoreFailure, Scenario::PreparationAndRestoreFailure}) {
        State state(scenario);
        require(std::fesetround(FE_DOWNWARD) == 0 && std::feclearexcept(FE_ALL_EXCEPT) == 0
                && std::feraiseexcept(FE_DIVBYZERO) == 0,
                "cannot set Gaussian caller environment");
        const int before_flags = std::fetestexcept(FE_ALL_EXCEPT);
        const auto before = scar_internal::parallel_runtime_info();
        const Caught caught = execute(state, factor);
        const int after_flags = std::fetestexcept(FE_ALL_EXCEPT);
        const int after_rounding = std::fegetround();
        const auto after = scar_internal::parallel_runtime_info();
        if (scenario == Scenario::PreparationAndRow || scenario == Scenario::UpdateAndRow) {
            require(state.global_preparation_recorded.load(),
                    "mixed Gaussian outcome did not record its later preparation/update error");
        }
        if (scenario == Scenario::EarlierRowException
            || scenario == Scenario::EarlierInjectedNumerical) {
            require(state.row_visits[17 * dimension + 7].load() == 1,
                    "mixed Gaussian outcome did not reach its later-coordinate row exception");
        }
        const bool guard_failure = scenario == Scenario::GuardApplyFailure
            || scenario == Scenario::GuardRestoreFailure
            || scenario == Scenario::PreparationAndRestoreFailure;
        require(after.batches_submitted - before.batches_submitted == (guard_failure ? 0u : 1u)
                && after.tasks_submitted - before.tasks_submitted == (guard_failure ? 0u : 4u),
                "Gaussian preparation guard used an unexpected row publication path");
        require(state.preparation_wrong_context.load() == 0 && state.rows_on_caller.load() == 0,
                "Gaussian shared preparation or queued rows ran in the wrong context");
        if (!guard_failure) {
            require(after_rounding == FE_DOWNWARD && after_flags == before_flags,
                    "Gaussian shared preparation did not restore caller rounding/flags");
        }
        switch (scenario) {
        case Scenario::CountPreparation:
        case Scenario::EnvironmentMutation:
            require(caught == Caught::None && state.observed && state.output.status == scar::Status::Ok,
                    "Gaussian shared preparation success failed");
            for (std::size_t coordinate = 0; coordinate < dimension; ++coordinate) {
                require(state.preparation_calls[coordinate].load() == 1
                        && state.update_calls[coordinate].load() == 1,
                        "Gaussian covariance preparation was repeated per block");
            }
            for (const auto& visited : state.row_visits) {
                require(visited.load() == 1, "Gaussian row/coordinate coverage is not exact");
            }
            require(state.rows_wrong_environment.load() == 0,
                    "Gaussian row execution inherited temporary preparation environment");
            require(std::all_of(state.output.residuals.begin(), state.output.residuals.end(),
                                [](double value) { return value == 0.5; }),
                    "Gaussian zero-loading center changed during shared preparation");
            break;
        case Scenario::PreparationOnly:
            require(caught == Caught::Preparation && state.global_preparation_recorded.load(),
                    "global Gaussian preparation failure was overwritten by row progress");
            pattern(state, 6, -1, true, {});
            require_strict_prefix(state, 0, rows, 6);
            break;
        case Scenario::UpdateOnly:
            require(caught == Caught::Preparation && state.global_preparation_recorded.load(),
                    "global Gaussian update failure was overwritten by row progress");
            pattern(state, 6, -1, false, {});
            require_strict_prefix(state, 0, rows, 7);
            break;
        case Scenario::PreparationAndRow:
        case Scenario::UpdateAndRow:
            require(caught == Caught::FirstRow, "earlier Gaussian row exception did not win");
            pattern(state, 2, 18, false, {{18, 33}});
            require_strict_prefix(state, 19, 33, 2);
            break;
        case Scenario::EarlierRowException:
        case Scenario::EarlierExceptionAfterInjected:
            require(caught == Caught::FirstRow, "earlier Gaussian row exception did not win");
            pattern(state, 2, 21, false, {{21, 33}});
            require_strict_prefix(state, 22, 33, 2);
            break;
        case Scenario::SameCoordinateFirstRow:
            require(caught == Caught::FirstRow, "later row replaced the first same-coordinate exception");
            pattern(state, 3, 17, false, {{17, 33}});
            require_strict_prefix(state, 18, 33, 3);
            break;
        case Scenario::CrossBlockRegistration:
            require(caught == Caught::FirstRow
                    && state.first_registration.load() < state.later_registration.load(),
                    "Gaussian block ID displaced the first registered exception");
            pattern(state, 3, 35, false, {{18, 33}, {35, 49}});
            break;
        case Scenario::EarlierInjectedNumerical:
            require(caught == Caught::None, "late Gaussian exception displaced earlier injected numerical failure");
            pattern(state, 2, 21, false, {{21, 33}});
            require_strict_prefix(state, 22, 33, 2);
            break;
        case Scenario::GuardApplyFailure:
            require(caught == Caught::Apply, "Gaussian preparation apply error was lost");
            break;
        case Scenario::GuardRestoreFailure:
            require(caught == Caught::Restore, "Gaussian preparation restore error was lost");
            break;
        case Scenario::PreparationAndRestoreFailure:
            require(caught == Caught::Composite, "Gaussian double failure was not transported explicitly");
            break;
        }
        if (scenario == Scenario::EarlierInjectedNumerical
            || scenario == Scenario::EarlierExceptionAfterInjected) {
            require(state.injected_values.load() == 1, "post-validation numerical injection did not execute");
        }
        if (guard_failure) {
            for (const auto& visited : state.row_visits) {
                require(visited.load() == 0, "unsafe rows ran after preparation environment failure");
            }
        }
    }
}
}  // namespace

int run_factor_rosenblatt_gaussian_plan_tests() {
    try {
        prepared_outcomes();
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Gaussian Rosenblatt preparation regression: %s\n", error.what());
        return 1;
    }
}
