#include "scar/copula/multivariate/correlation/factor.hpp"
#include "scar/copula/multivariate/student/factor_density.hpp"
#include "scar/copula/multivariate/correlation/factor_solve.hpp"
#include "scar/detail/parallel.hpp"

#include <algorithm>
#include <array>
#include <atomic>
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
#include <thread>
#include <vector>

namespace {

constexpr std::size_t dimension = 5;
constexpr std::size_t rank = 2;
using scar::FactorStudentJointResult;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Function>
void require_invalid(Function&& function) {
    bool caught = false;
    try {
        function();
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    require(caught, "invalid factor workspace input was accepted");
}

class RuntimeScope {
public:
    RuntimeScope() { scar_internal::shutdown_parallel_runtime(); }
    ~RuntimeScope() { scar_internal::shutdown_parallel_runtime(); }
    RuntimeScope(const RuntimeScope&) = delete;
    RuntimeScope& operator=(const RuntimeScope&) = delete;
};

scar::FactorCorrelationOperator make_factor() {
    return scar::FactorCorrelationOperator(
        {0.17, -0.04, 0.08, 0.21, -0.13, 0.09, 0.03, -0.16, 0.14, 0.12},
        dimension, rank, 1e-8);
}

std::vector<double> observations(std::size_t rows, std::size_t offset = 0) {
    std::vector<double> values(rows * dimension);
    for (std::size_t index = 0; index < values.size(); ++index) {
        values[index] = 0.03 + 0.94 * static_cast<double>(
            (index * 37 + offset * 13) % 997) / 996.0;
    }
    return values;
}

// Solve the dense correlation by pivoted elimination, independently of the
// factor operator's Woodbury formula and rank-sized Cholesky solve.
std::array<double, dimension> dense_solution(
    const scar::FactorCorrelationOperator& factor,
    std::array<double, dimension> rhs) {

    std::array<double, dimension * dimension> matrix{};
    for (std::size_t row = 0; row < dimension; ++row) {
        for (std::size_t column = 0; column < dimension; ++column) {
            double value = row == column ? 1.0 : 0.0;
            if (row != column) {
                for (std::size_t component = 0; component < rank; ++component) {
                    value += factor.loadings()[row * rank + component]
                        * factor.loadings()[column * rank + component];
                }
            }
            matrix[row * dimension + column] = value;
        }
    }
    for (std::size_t column = 0; column < dimension; ++column) {
        std::size_t pivot = column;
        for (std::size_t row = column + 1; row < dimension; ++row) {
            if (std::abs(matrix[row * dimension + column])
                > std::abs(matrix[pivot * dimension + column])) {
                pivot = row;
            }
        }
        for (std::size_t index = 0; index < dimension; ++index) {
            std::swap(matrix[column * dimension + index],
                      matrix[pivot * dimension + index]);
        }
        std::swap(rhs[column], rhs[pivot]);
        for (std::size_t row = column + 1; row < dimension; ++row) {
            const double multiplier = matrix[row * dimension + column]
                / matrix[column * dimension + column];
            for (std::size_t index = column; index < dimension; ++index) {
                matrix[row * dimension + index] -=
                    multiplier * matrix[column * dimension + index];
            }
            rhs[row] -= multiplier * rhs[column];
        }
    }
    for (std::size_t row = dimension; row-- > 0;) {
        for (std::size_t column = row + 1; column < dimension; ++column) {
            rhs[row] -= matrix[row * dimension + column] * rhs[column];
        }
        rhs[row] /= matrix[row * dimension + row];
    }
    return rhs;
}

void row_workspace_contract() {
    const auto factor = make_factor();
    std::array<double, rank + 2> workspace{};
    std::array<double, dimension> output{};
    const std::array<std::array<double, dimension>, 3> rows{{
        {{0.0, 0.0, 0.0, 0.0, 0.0}},
        {{-2.25, 0.125, 3.5, -0.75, 0.5}},
        {{1e-4, -1.5, 0.0, 2.75, -0.0625}},
    }};
    for (const auto& input : rows) {
        workspace.fill(std::numeric_limits<double>::quiet_NaN());
        workspace.front() = 12345.0;
        workspace.back() = -6789.0;
        scar_internal::factor_solve_row_with_workspace(
            factor, input.data(), output.data(), workspace.data() + 1, rank);
        const auto reference = dense_solution(factor, input);
        for (std::size_t index = 0; index < dimension; ++index) {
            require(std::isfinite(output[index])
                    && std::abs(output[index] - reference[index]) < 2e-14,
                    "workspace solve disagrees with independent dense solve");
        }
        require(workspace.front() == 12345.0 && workspace.back() == -6789.0,
                "workspace solve wrote outside its rank-sized slice");
        auto inplace = input;
        workspace[1] = 1e200;
        workspace[2] = -1e200;
        scar_internal::factor_solve_row_with_workspace(
            factor, inplace.data(), inplace.data(), workspace.data() + 1, rank);
        require(std::memcmp(inplace.data(), output.data(), sizeof(output)) == 0,
                "in-place solve or dirty workspace changed result bits");
        std::array<double, dimension> public_output{};
        factor.solve_rows(input.data(), 1, public_output.data(), 1);
        require(std::memcmp(public_output.data(), output.data(), sizeof(output)) == 0,
                "public row solve and workspace solve disagree");
    }
    const auto input = rows[1];
    output.fill(999.0);
    for (double bad : {std::numeric_limits<double>::infinity(),
                       -std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::quiet_NaN()}) {
        auto invalid = input;
        invalid.back() = bad;
        require_invalid([&] {
            scar_internal::factor_solve_row_with_workspace(
                factor, invalid.data(), output.data(), workspace.data() + 1, rank);
        });
        require(std::all_of(output.begin(), output.end(),
                           [](double value) { return value == 999.0; }),
                "non-finite input changed solve output before rejection");
    }
    require_invalid([&] {
        scar_internal::factor_solve_row_with_workspace(
            factor, nullptr, output.data(), workspace.data(), rank);
    });
    require_invalid([&] {
        scar_internal::factor_solve_row_with_workspace(
            factor, input.data(), nullptr, workspace.data(), rank);
    });
    require_invalid([&] {
        scar_internal::factor_solve_row_with_workspace(
            factor, input.data(), output.data(), nullptr, rank);
    });
    require_invalid([&] {
        scar_internal::factor_solve_row_with_workspace(
            factor, input.data(), output.data(), workspace.data(), rank - 1);
    });
    require(std::all_of(output.begin(), output.end(),
                       [](double value) { return value == 999.0; }),
            "invalid workspace changed solve output before rejection");
}

void exact_joint(const FactorStudentJointResult& actual,
                 const FactorStudentJointResult& reference) {
    require(actual.status == reference.status
            && actual.failure.index == reference.failure.index
            && actual.reduction_blocks == reference.reduction_blocks
            && actual.parallel_blocks == reference.parallel_blocks,
            "joint status or logical partials changed");
    require(std::memcmp(&actual.log_likelihood, &reference.log_likelihood,
                        sizeof(double)) == 0
            && std::memcmp(&actual.dlog_likelihood_ddf,
                           &reference.dlog_likelihood_ddf, sizeof(double)) == 0
            && actual.dlog_likelihood_dloadings.size()
                == reference.dlog_likelihood_dloadings.size()
            && std::memcmp(actual.dlog_likelihood_dloadings.data(),
                           reference.dlog_likelihood_dloadings.data(),
                           actual.dlog_likelihood_dloadings.size() * sizeof(double)) == 0,
            "joint value or gradient bits changed across execution choices");
}

void planned_workspace(const FactorStudentJointResult& result,
                       std::size_t rows, std::size_t slots) {
    require(result.planned_worker_slots == slots
            && result.planned_worker_workspace_bytes
                == slots * (3 * dimension + rank) * sizeof(double)
            && result.worker_workspace_peak_bytes
                == (rows == 0 ? 0 : (3 * dimension + rank) * sizeof(double))
            && result.reduction_workspace_bytes
                == std::min(rows, std::size_t{64}) * dimension * rank * sizeof(double),
            "joint workspace diagnostics do not describe logical slots");
}

void joint_partitions_and_history() {
    RuntimeScope runtime;
    const auto factor = make_factor();
    for (std::size_t count : {std::size_t{0}, std::size_t{1}, std::size_t{16},
                             std::size_t{63}, std::size_t{64}, std::size_t{65},
                             std::size_t{257}}) {
        const auto values = observations(count);
        const auto reference = scar::factor_student_joint_likelihood_gradient(
            factor, values.data(), count, 5.7, 1);
        require(reference.is_ok(), "serial joint reference failed");
        for (int request : {1, 4, 17, 32}) {
            const auto actual = scar::factor_student_joint_likelihood_gradient(
                factor, values.data(), count, 5.7, request);
            exact_joint(actual, reference);
            const std::size_t slots = count == 0 ? 0 : std::min(
                static_cast<std::size_t>(request),
                std::max(std::size_t{1}, std::min(count, std::size_t{64}) / 4));
            planned_workspace(actual, count, slots);
        }
    }
    scar_internal::parallel_for_blocks(0, 32, 1, 32,
        [](std::int64_t, std::int64_t, std::size_t) {});
    const auto values = observations(257);
    const auto reference = scar::factor_student_joint_likelihood_gradient(
        factor, values.data(), 257, 6.25, 1);
    const auto before = scar_internal::parallel_runtime_info();
    const auto actual = scar::factor_student_joint_likelihood_gradient(
        factor, values.data(), 257, 6.25, 4);
    const auto after = scar_internal::parallel_runtime_info();
    exact_joint(actual, reference);
    planned_workspace(actual, 257, 4);
    require(after.worker_count == 32 && before.worker_count == 32
            && after.tasks_submitted - before.tasks_submitted == 4
            && after.batches_submitted - before.batches_submitted == 1,
            "resident pool size leaked into joint execution budget");

    auto invalid_values = values;
    invalid_values[17 * dimension + 4] = std::numeric_limits<double>::quiet_NaN();
    invalid_values[241 * dimension] = std::numeric_limits<double>::infinity();
    const auto invalid_reference = scar::factor_student_joint_likelihood_gradient(
        factor, invalid_values.data(), 257, 6.25, 1);
    require(invalid_reference.status == scar::Status::NumericalFailure
            && invalid_reference.failure.index == 17,
            "joint reference did not select first invalid row");
    exact_joint(scar::factor_student_joint_likelihood_gradient(
        factor, invalid_values.data(), 257, 6.25, 4), invalid_reference);
    exact_joint(scar::factor_student_joint_likelihood_gradient(
        factor, values.data(), 257, 6.25, 4), reference);
}

void nested_joint() {
    RuntimeScope runtime;
    const auto factor = make_factor();
    const auto values = observations(65);
    const auto reference = scar::factor_student_joint_likelihood_gradient(
        factor, values.data(), 65, 5.7, 1);
    std::atomic<int> visited{0};
    scar_internal::parallel_for_blocks(0, 4, 1, 4,
        [&](std::int64_t, std::int64_t, std::size_t) {
            const auto before = scar_internal::parallel_runtime_info();
            const auto actual = scar::factor_student_joint_likelihood_gradient(
                factor, values.data(), 65, 5.7, 4);
            const auto after = scar_internal::parallel_runtime_info();
            exact_joint(actual, reference);
            planned_workspace(actual, 65, 1);
            require(after.tasks_submitted == before.tasks_submitted
                    && after.batches_submitted == before.batches_submitted,
                    "nested joint submitted a batch");
            visited.fetch_add(1);
        });
    require(visited.load() == 4, "nested joint did not complete every call");
}

class ConcurrentEntry {
public:
    bool arrive() noexcept {
        try {
            std::unique_lock<std::mutex> lock(mutex_);
            ++arrived_;
            ready_.notify_all();
            return ready_.wait_for(lock, std::chrono::seconds(10),
                                   [&] { return arrived_ == 2; });
        } catch (...) {
            return false;
        }
    }
private:
    std::mutex mutex_;
    std::condition_variable ready_;
    int arrived_ = 0;
};

std::atomic<ConcurrentEntry*> concurrent_entry{nullptr};

bool hold_first_portion(std::size_t block, bool restoring) noexcept {
    if (block != 0 || restoring) {
        return false;
    }
    auto* entry = concurrent_entry.load();
    return entry == nullptr || !entry->arrive();
}

void concurrent_joint() {
    RuntimeScope runtime;
    const auto factor = make_factor();
    const auto first = observations(257, 1);
    const auto second = observations(129, 2);
    const auto first_reference = scar::factor_student_joint_likelihood_gradient(
        factor, first.data(), 257, 5.7, 1);
    const auto second_reference = scar::factor_student_joint_likelihood_gradient(
        factor, second.data(), 129, 8.25, 1);
    scar_internal::parallel_for_blocks(0, 8, 1, 8,
        [](std::int64_t, std::int64_t, std::size_t) {});
    ConcurrentEntry entry;
    concurrent_entry.store(&entry);
    std::array<FactorStudentJointResult, 2> results;
    std::array<std::exception_ptr, 2> failures{};
    auto call = [&](std::size_t index) {
        scar_internal::parallel_testing::set_environment_failure_hook(hold_first_portion);
        try {
            results[index] = scar::factor_student_joint_likelihood_gradient(
                factor, index == 0 ? first.data() : second.data(),
                index == 0 ? 257 : 129, index == 0 ? 5.7 : 8.25, 4);
        } catch (...) {
            failures[index] = std::current_exception();
        }
        scar_internal::parallel_testing::set_environment_failure_hook(nullptr);
    };
    std::thread first_caller(call, 0);
    try {
        std::thread second_caller(call, 1);
        first_caller.join();
        second_caller.join();
    } catch (...) {
        first_caller.join();
        concurrent_entry.store(nullptr);
        throw;
    }
    concurrent_entry.store(nullptr);
    for (const auto& failure : failures) {
        if (failure) {
            std::rethrow_exception(failure);
        }
    }
    exact_joint(results[0], first_reference);
    exact_joint(results[1], second_reference);
    planned_workspace(results[0], 257, 4);
    planned_workspace(results[1], 129, 4);
}

void joint_validation() {
    RuntimeScope runtime;
    const auto factor = make_factor();
    const double dummy = 0.5;
    for (int request : {0, 257}) {
        require_invalid([&] {
            (void)scar::factor_student_joint_likelihood_gradient(
                factor, &dummy, 1, 5.7, request);
        });
    }
    for (double df : {2.0, std::numeric_limits<double>::infinity(),
                      std::numeric_limits<double>::quiet_NaN()}) {
        require_invalid([&] {
            (void)scar::factor_student_joint_likelihood_gradient(
                factor, &dummy, 1, df, 4);
        });
    }
    require_invalid([&] {
        (void)scar::factor_student_joint_likelihood_gradient(
            factor, nullptr, 1, 5.7, 4);
    });
    for (std::size_t rows : {
             std::numeric_limits<std::size_t>::max() / dimension + 1,
             std::numeric_limits<std::size_t>::max() / sizeof(double) / dimension + 1}) {
        require_invalid([&] {
            (void)scar::factor_student_joint_likelihood_gradient(
                factor, &dummy, rows, 5.7, 4);
        });
    }
    require(!scar_internal::parallel_runtime_info().initialized,
            "invalid joint input initialized the pool");
}

}  // namespace

int run_factor_joint_workspace_tests() {
    try {
        row_workspace_contract();
        joint_partitions_and_history();
        nested_joint();
        concurrent_joint();
        joint_validation();
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "factor workspace regression: %s\n", error.what());
        return 1;
    }
}
