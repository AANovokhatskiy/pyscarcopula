#include "scar/copula/multivariate/rosenblatt.hpp"

#include "scar/copula/multivariate/correlation/factor.hpp"
#include "scar/copula/multivariate/student/distribution.hpp"
#include "scar/copula/multivariate/student/quantile.hpp"
#include "scar/core/checked_arithmetic.hpp"
#include "scar/detail/linalg.hpp"
#include "scar/detail/parallel.hpp"
#include "scar/detail/safety.hpp"
#include "scar/math/gamma.hpp"
#include "scar/math/normal.hpp"

#include <algorithm>
#include <atomic>
#include <exception>
#include <utility>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace scar {
namespace {

constexpr double kCorrelationTolerance = 1e-12;
constexpr std::size_t kFactorRowAlignment = 128;
constexpr std::size_t kFactorRowAlignmentValues = kFactorRowAlignment / sizeof(double);
constexpr std::size_t kFactorRowsPerTile = 4;

bool validate_observations(
    ObservationView observations,
    std::size_t dimension,
    std::size_t& values) {

    if (observations.dim != static_cast<int>(dimension)
        || !core::checked_size_mul(
            observations.n_obs, dimension, values)
        || (values != 0 && observations.data() == nullptr)) {
        return false;
    }
    for (std::size_t index = 0; index < values; ++index) {
        const double value = observations.data()[index];
        if (!std::isfinite(value) || value < 0.0 || value > 1.0) {
            return false;
        }
    }
    return true;
}

bool prepare_dense_correlation(
    DoubleView correlation,
    std::size_t dimension,
    std::vector<double>& lower,
    int& failure_coordinate) {

    std::size_t square = 0;
    if (!core::checked_size_mul(dimension, dimension, square)
        || correlation.size() != square
        || correlation.data() == nullptr) {
        return false;
    }
    for (std::size_t row = 0; row < dimension; ++row) {
        const double diagonal = correlation[row * dimension + row];
        if (!std::isfinite(diagonal)
            || std::abs(diagonal - 1.0) > kCorrelationTolerance) {
            failure_coordinate = static_cast<int>(row);
            return false;
        }
        for (std::size_t column = 0; column < row; ++column) {
            const double left = correlation[row * dimension + column];
            const double right = correlation[column * dimension + row];
            const double scale = std::max(
                {1.0, std::abs(left), std::abs(right)});
            if (!std::isfinite(left)
                || !std::isfinite(right)
                || std::abs(left - right)
                    > kCorrelationTolerance * scale) {
                failure_coordinate = static_cast<int>(row);
                return false;
            }
        }
    }
    std::size_t failed = dimension;
    if (!scar_internal::linalg::cholesky_symmetric(
            correlation.data(),
            dimension,
            lower,
            0.0,
            &failed)) {
        failure_coordinate = failed < dimension
            ? static_cast<int>(failed)
            : -1;
        return false;
    }
    return true;
}

int rosenblatt_workers(
    int n_threads,
    std::size_t rows,
    std::size_t dimension) {

    return n_threads > 1
        && scar_internal::grid_parallel_worthwhile(
            rows, dimension, 16, 2048)
        ? n_threads
        : 1;
}

enum class RosenblattOperation { Preparation, Rows, Update };

struct RosenblattOutcome {
    std::size_t coordinate = 0;
    RosenblattOperation operation = RosenblattOperation::Preparation;
    std::int64_t row = -1;
    bool failed = false;
    std::exception_ptr exception;
    std::size_t registration = 0;
};

struct RosenblattFailures {
    RosenblattOutcome* outcomes;
    const scar_internal::ParallelExecutionPlan* plan;
    std::atomic<std::size_t> registrations{0};
#ifdef SCAR_PARALLEL_TESTING
    rosenblatt_testing::RecordedObserver recorded_observer = nullptr;
#endif

    void record_outcome(RosenblattOutcome& outcome, std::size_t block,
                        std::exception_ptr exception) noexcept {
        (void)block;  // Used only by the test registration observer.
        outcome.registration = registrations.fetch_add(1, std::memory_order_relaxed);
        outcome.exception = std::move(exception);
        outcome.failed = true;
#ifdef SCAR_PARALLEL_TESTING
        if (recorded_observer != nullptr) {
            recorded_observer(
                outcome.coordinate,
                static_cast<rosenblatt_testing::Operation>(outcome.operation),
                block, outcome.registration);
        }
#endif
    }

    void record(std::size_t block, std::exception_ptr exception) noexcept {
        record_outcome(outcomes[block], block, std::move(exception));
    }

    static void from_runtime(
        void* state, std::size_t block,
        scar_internal::ParallelFailureOrigin origin,
        std::exception_ptr exception) noexcept {

        auto& failures = *static_cast<RosenblattFailures*>(state);
        auto& outcome = failures.outcomes[block];
        if (origin == scar_internal::ParallelFailureOrigin::EnvironmentApply) {
            // No row callback has run: this is entry to coordinate zero.
            outcome.coordinate = 0;
            outcome.operation = RosenblattOperation::Rows;
            outcome.row = failures.plan->bounds()[block];
        }
        // Escaped callback errors retain the current coordinate/operation.
        failures.record(block, std::move(exception));
    }
};

bool earlier_outcome(const RosenblattOutcome& left, const RosenblattOutcome& right) {
    if (left.coordinate != right.coordinate) {
        return left.coordinate < right.coordinate;
    }
    if (left.operation != right.operation) {
        return left.operation < right.operation;
    }
    if (static_cast<bool>(left.exception) != static_cast<bool>(right.exception)) {
        return static_cast<bool>(left.exception);
    }
    return left.exception ? left.registration < right.registration : left.row < right.row;
}

#ifdef SCAR_PARALLEL_TESTING
thread_local rosenblatt_testing::FaultHook factor_fault_hook = nullptr;
thread_local rosenblatt_testing::OutputObserver factor_output_observer = nullptr;
thread_local rosenblatt_testing::RecordedObserver factor_recorded_observer = nullptr;
#endif

std::exception_ptr prepared_gaussian_rosenblatt(
    const FactorCorrelationOperator& correlation,
    ObservationView u,
    const scar_internal::ParallelExecutionPlan& plan,
    std::size_t coefficient_values,
    std::size_t row_state_values,
    std::size_t row_state_stride,
    MultivariateRosenblattResult& out,
    std::vector<RosenblattOutcome>& outcomes,
    RosenblattFailures& failures) {

    const std::size_t dimension = correlation.dimension();
    const std::size_t rank = correlation.rank();
    const std::size_t square = rank * rank;  // Checked by the caller.
    std::vector<double> coefficients(coefficient_values, 0.0);
    std::vector<double> row_states(row_state_values, 0.0);
    // Each repeatedly updated state occupies separate cache lines, including
    // machines with 128-byte lines. The allocation includes alignment slack.
    const std::size_t misalignment =
        reinterpret_cast<std::uintptr_t>(row_states.data()) % kFactorRowAlignment;
    double* aligned_row_states = row_states.data()
        + ((kFactorRowAlignment - misalignment) % kFactorRowAlignment) / sizeof(double);
    double* variances = coefficients.data() + dimension * rank;
    double* standard_deviations = variances + dimension;
    auto& preparation = outcomes.back();
    const auto preparation_plan = scar_internal::make_parallel_execution_plan(
        0, static_cast<std::int64_t>(dimension), 1, 1);
#ifdef SCAR_PARALLEL_TESTING
    const auto fault_hook = factor_fault_hook;
#endif
    {
        // All numerical storage is allocated before preparation. Row storage
        // coexists with covariance here; covariance is freed before row work.
        std::vector<double> covariance(square, 0.0);
        for (std::size_t diagonal = 0; diagonal < rank; ++diagonal) {
            covariance[diagonal * rank + diagonal] = 1.0;
        }
        try {
            scar_internal::execute_parallel_plan(
                preparation_plan,
                [&](std::int64_t, std::int64_t,
                    const scar_internal::ParallelBlockContext&) {
                for (std::size_t coordinate = 0; coordinate < dimension; ++coordinate) {
                    preparation.coordinate = coordinate;
                    preparation.operation = RosenblattOperation::Preparation;
                    const double* loading = correlation.loadings().data() + coordinate * rank;
                    double* h = coefficients.data() + coordinate * rank;
                    try {
#ifdef SCAR_PARALLEL_TESTING
                        if (fault_hook != nullptr) {
                            fault_hook(coordinate, rosenblatt_testing::Operation::Preparation, 0, -1);
                        }
#endif
                        // These helpers and the update allocate no storage and
                        // cannot throw in production after the preallocation.
                        scar_internal::linalg::row_major_matvec(
                            covariance.data(), rank, rank, loading, h);
                        double variance = correlation.uniqueness()[coordinate];
                        variance += scar_internal::linalg::dot(loading, h, rank);
                        variances[coordinate] = variance;
                        if (!std::isfinite(variance) || !(variance > 0.0)) {
                            preparation.failed = true;
                            return;
                        }
                        standard_deviations[coordinate] = std::sqrt(variance);
                        preparation.operation = RosenblattOperation::Update;
#ifdef SCAR_PARALLEL_TESTING
                        if (fault_hook != nullptr) {
                            fault_hook(coordinate, rosenblatt_testing::Operation::Update, 0, -1);
                        }
#endif
                        for (std::size_t left = 0; left < rank; ++left) {
                            for (std::size_t right = 0; right < rank; ++right) {
                                covariance[left * rank + right] -=
                                    h[left] * h[right] / variance;
                            }
                        }
                    } catch (...) {
                        failures.record_outcome(preparation, 0, std::current_exception());
                        return;
                    }
                }
            });
        } catch (...) {
            // A guard failure prevents publication. Preserve a deferred
            // preparation exception as well if restoring the environment fails.
            if (preparation.exception) {
                throw scar_internal::ParallelEnvironmentRestoreError(
                    preparation.exception, std::current_exception());
            }
            throw;
        }
    }
    const std::size_t prepared_coordinates = !preparation.failed ? dimension
        : preparation.coordinate
            + (preparation.operation == RosenblattOperation::Update ? 1 : 0);
    if (prepared_coordinates == 0) {
        return {};
    }

    return scar_internal::execute_parallel_plan_deferred(
        plan,
        [&](std::int64_t begin, std::int64_t end,
            const scar_internal::ParallelBlockContext& context) {
        auto& outcome = outcomes[context.block_id];
        double* tile_states = aligned_row_states + context.worker_slot * row_state_stride;
        for (std::int64_t tile_begin = begin; tile_begin < end;) {
            const std::int64_t tile_end = tile_begin + std::min(
                static_cast<std::int64_t>(kFactorRowsPerTile), end - tile_begin);
            std::fill(tile_states,
                      tile_states + static_cast<std::size_t>(tile_end - tile_begin) * rank, 0.0);
            // The original block stopped at its first failed row in column j.
            // Later rows still need earlier columns, but must not execute j.
            const std::size_t limit = outcome.failed ? outcome.coordinate : prepared_coordinates;
            for (std::size_t coordinate = 0; coordinate < limit; ++coordinate) {
                const double* loading = correlation.loadings().data() + coordinate * rank;
                const double* h = coefficients.data() + coordinate * rank;
                const double variance = variances[coordinate];
                const double standard_deviation = standard_deviations[coordinate];
                for (std::int64_t row_index = tile_begin; row_index < tile_end; ++row_index) {
                    double* row_state = tile_states
                        + static_cast<std::size_t>(row_index - tile_begin) * rank;
                    try {
    #ifdef SCAR_PARALLEL_TESTING
                        if (fault_hook != nullptr) {
                            fault_hook(coordinate, rosenblatt_testing::Operation::Rows,
                                       context.block_id, row_index);
                        }
    #endif
                        const std::size_t row = static_cast<std::size_t>(row_index);
                        const double probability = scar_internal::clip_pseudo_observation(
                            u.row(row)[coordinate]);
                        const double latent = math::normal_quantile_refined(probability);
                        const double conditional_mean = scar_internal::linalg::dot(
                            row_state, loading, rank);
                        const double innovation = latent - conditional_mean;
                        const double residual = math::normal_cdf(
                            innovation / standard_deviation);
                        for (std::size_t factor = 0; factor < rank; ++factor) {
                            row_state[factor] += innovation / variance * h[factor];
                        }
                        if (!std::isfinite(latent) || !std::isfinite(residual)) {
                            outcome = {};
                            outcome.coordinate = coordinate;
                            outcome.operation = RosenblattOperation::Rows;
                            outcome.row = row_index;
                            outcome.failed = true;
                            break;
                        }
                        out.residuals[row * dimension + coordinate] =
                            scar_internal::clip_pseudo_observation(residual);
                    } catch (...) {
                        outcome = {};
                        outcome.coordinate = coordinate;
                        outcome.operation = RosenblattOperation::Rows;
                        outcome.row = row_index;
                        failures.record(context.block_id, std::current_exception());
                        break;
                    }
                }
                if (outcome.failed && outcome.coordinate == coordinate) {
                    break;
                }
            }
            tile_begin = tile_end;
        }
    }, {&failures, &RosenblattFailures::from_runtime});
}

MultivariateRosenblattResult factor_rosenblatt(
    const FactorCorrelationOperator& correlation,
    ObservationView u,
    DoubleView df,
    int n_threads,
    bool student) {

    MultivariateRosenblattResult out;
    out.n_rows = static_cast<std::int64_t>(u.n_obs);
    out.dimension = static_cast<int>(correlation.dimension());
    out.n_threads_requested = n_threads;
    std::size_t values = 0;
    std::size_t input_bytes = 0;
    if (!scar_internal::valid_thread_count(n_threads)) {
        out.status = Status::InvalidParameter;
        return out;
    }
    if (u.n_obs > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())
        || correlation.dimension() > static_cast<std::size_t>(std::numeric_limits<int>::max())
        || !core::checked_size_mul(u.n_obs, correlation.dimension(), values)
        || !core::checked_size_mul(values, sizeof(double), input_bytes)
        || !validate_observations(u, correlation.dimension(), values)) {
        out.status = Status::InvalidSize;
        return out;
    }
    if (student && ((df.size() != 1 && df.size() != u.n_obs)
                    || (!df.empty() && df.data() == nullptr))) {
        out.status = Status::InvalidSize;
        return out;
    }
    if (student) {
        for (std::size_t row = 0; row < df.size(); ++row) {
            if (!std::isfinite(df[row]) || !(df[row] > 2.0)) {
                out.status = Status::InvalidParameter;
                out.failure.index = df.size() == 1
                    ? -1
                    : static_cast<std::int64_t>(row);
                return out;
            }
        }
    }

    const std::size_t rows = u.n_obs;
    const std::size_t dimension = correlation.dimension();
    const std::size_t rank = correlation.rank();
    if (rows == 0) {
        return out;
    }

    const int workers = rosenblatt_workers(n_threads, rows, dimension);
    out.parallel_blocks = workers;  // Preserve the legacy policy diagnostic.
    const std::size_t grain_blocks = rows / 16 + (rows % 16 != 0 ? 1 : 0);
    const std::size_t blocks = workers == 1 ? 1
        : std::min(static_cast<std::size_t>(workers), grain_blocks);
    auto plan = scar_internal::make_parallel_execution_plan(
        0, static_cast<std::int64_t>(rows), blocks, workers);
    if (blocks > 1 && scar_internal::parallel_execution_slots(plan) == 1) {
        // The old nested wrapper processes one full row block per coordinate.
        plan = scar_internal::make_parallel_execution_plan(
            0, static_cast<std::int64_t>(rows), 1, 1);
    }
    const std::size_t slots = scar_internal::parallel_execution_slots(plan);
    // Shared preparation has a fixed serial/control cost. Keep the accepted
    // coordinate-outer path for short, low-rank, few-block calls where that
    // cost does not amortize; both paths retain the same B geometry.
    const bool prepared_gaussian = !student && plan.block_count() > 1
        && (rows > 2048 || rank >= 16 || plan.block_count() >= 8);
    const std::size_t capacity = static_cast<std::size_t>(
        plan.bounds()[1] - plan.bounds()[0]);
    std::size_t square = 0;
    std::size_t rank_scratch = 0;
    std::size_t state_values = 0;
    std::size_t slot_values = 0;
    std::size_t scratch_values = 0;
    std::size_t output_bytes = 0;
    std::size_t scratch_bytes = 0;
    std::size_t outcome_bytes = 0;
    std::size_t boundary_bytes = 0;
    std::size_t execution_bytes = 0;
    std::size_t conversion_bytes = 0;
    std::size_t coefficient_values = 0;
    std::size_t coefficient_bytes = 0;
    std::size_t covariance_bytes = 0;
    std::size_t outcome_count = 0;
    std::size_t row_state_stride = 0;
    if (!core::checked_size_mul(rank, rank, square)
        || !core::checked_size_mul(values, sizeof(double), output_bytes)
        || !core::checked_size_add(plan.block_count(), prepared_gaussian ? 1 : 0, outcome_count)
        || !core::checked_size_mul(outcome_count, sizeof(RosenblattOutcome), outcome_bytes)
        || !core::checked_size_mul(plan.bounds().size(), sizeof(std::int64_t), boundary_bytes)
        || !core::checked_size_add(output_bytes, output_bytes, conversion_bytes)) {
        out.status = Status::InvalidSize;
        return out;
    }
    if (prepared_gaussian) {
        // Preparation and row storage coexist; both are gone before conversion.
        if (!core::checked_size_mul(dimension, rank, coefficient_values)
            || !core::checked_size_add(coefficient_values, dimension, coefficient_values)
            || !core::checked_size_add(coefficient_values, dimension, coefficient_values)
            || !core::checked_size_mul(coefficient_values, sizeof(double), coefficient_bytes)
            || !core::checked_size_mul(square, sizeof(double), covariance_bytes)
            || !core::checked_size_mul(rank, kFactorRowsPerTile, row_state_stride)
            || !core::checked_size_add(
                row_state_stride, kFactorRowAlignmentValues - 1, row_state_stride)
            || !core::checked_size_mul(
                row_state_stride / kFactorRowAlignmentValues,
                kFactorRowAlignmentValues, row_state_stride)
            || !core::checked_size_mul(slots, row_state_stride, scratch_values)
            || !core::checked_size_add(
                scratch_values, kFactorRowAlignmentValues - 1, scratch_values)
            || !core::checked_size_mul(scratch_values, sizeof(double), scratch_bytes)
            || !core::checked_size_add(output_bytes, coefficient_bytes, execution_bytes)
            || !core::checked_size_add(execution_bytes, covariance_bytes, execution_bytes)
            || !core::checked_size_add(execution_bytes, scratch_bytes, execution_bytes)
            || !core::checked_size_add(execution_bytes, 2 * sizeof(std::int64_t), execution_bytes)) {
            out.status = Status::InvalidSize;
            return out;
        }
    } else if (!core::checked_size_mul(rank, std::size_t{2}, rank_scratch)
        || !core::checked_size_mul(capacity, rank, state_values)
        || !core::checked_size_add(square, rank_scratch, slot_values)
        || !core::checked_size_add(slot_values, state_values, slot_values)
        || !core::checked_size_add(slot_values, student ? capacity : 0, slot_values)
        || !core::checked_size_mul(slot_values, slots, scratch_values)
        || !core::checked_size_mul(scratch_values, sizeof(double), scratch_bytes)
        || !core::checked_size_add(output_bytes, scratch_bytes, execution_bytes)) {
        out.status = Status::InvalidSize;
        return out;
    }
    if (!core::checked_size_add(execution_bytes, outcome_bytes, execution_bytes)
        || !core::checked_size_add(execution_bytes, boundary_bytes, execution_bytes)) {
        out.status = Status::InvalidSize;
        return out;
    }
    out.residuals.assign(values, 0.0);
    std::vector<RosenblattOutcome> outcomes(outcome_count);
    RosenblattFailures failures{outcomes.data(), &plan};
    std::exception_ptr restore_failure;
#ifdef SCAR_PARALLEL_TESTING
    const auto fault_hook = factor_fault_hook;
    const auto output_observer = factor_output_observer;
    failures.recorded_observer = factor_recorded_observer;
#endif
    if (prepared_gaussian) {
        restore_failure = prepared_gaussian_rosenblatt(
            correlation, u, plan, coefficient_values, scratch_values, row_state_stride,
            out, outcomes, failures);
    } else {
        std::vector<double> scratch(scratch_values, 0.0);
        const scar_internal::PreparedParallelBlockFunction evaluate_portion =
            [&](std::int64_t begin, std::int64_t end,
                const scar_internal::ParallelBlockContext& context) {
            auto& outcome = outcomes[context.block_id];
            double* covariance = scratch.data() + context.worker_slot * slot_values;
            double* covariance_loading = covariance + square;
            double* solved_projection = covariance_loading + rank;
            double* state = solved_projection + rank;
            double* diagonal_quadratic = state + state_values;
            std::fill(covariance, covariance + slot_values, 0.0);
            for (std::size_t diagonal = 0; diagonal < rank; ++diagonal) {
                covariance[diagonal * rank + diagonal] = 1.0;
            }
            // After preallocation the production matvec/dot/math and updates
            // allocate no storage. Catch at each operation also covers hooks.
            for (std::size_t coordinate = 0; coordinate < dimension; ++coordinate) {
                outcome.coordinate = coordinate;
                outcome.operation = RosenblattOperation::Preparation;
                outcome.row = -1;
                const double* loading = correlation.loadings().data() + coordinate * rank;
                double conditional_variance = 0.0;
                try {
#ifdef SCAR_PARALLEL_TESTING
                    if (fault_hook != nullptr) {
                        fault_hook(coordinate, rosenblatt_testing::Operation::Preparation,
                                   context.block_id, -1);
                    }
#endif
                    scar_internal::linalg::row_major_matvec(
                        covariance, rank, rank, loading, covariance_loading);
                    conditional_variance = correlation.uniqueness()[coordinate];
                    conditional_variance += scar_internal::linalg::dot(
                        loading, covariance_loading, rank);
                    if (!std::isfinite(conditional_variance) || !(conditional_variance > 0.0)) {
                        outcome.failed = true;
                        return;
                    }
                } catch (...) {
                    failures.record(context.block_id, std::current_exception());
                    return;
                }
                outcome.operation = RosenblattOperation::Rows;
                outcome.row = begin;
                try {
                for (std::int64_t row_index = begin;
                     row_index < end;
                     ++row_index) {
                    outcome.row = row_index;
#ifdef SCAR_PARALLEL_TESTING
                    if (fault_hook != nullptr) {
                        fault_hook(coordinate, rosenblatt_testing::Operation::Rows,
                                   context.block_id, row_index);
                    }
#endif
                    const std::size_t row =
                        static_cast<std::size_t>(row_index);
                    const double probability = scar_internal::clip_pseudo_observation(
                        u.row(row)[coordinate]);
                    const double degrees = student
                        ? df[df.size() == 1 ? 0 : row]
                        : 0.0;
                    const double latent = student
                        ? scar_internal::student_quantile_refined_value(
                            probability, degrees)
                        : math::normal_quantile_refined(probability);
                    const std::size_t local_row = static_cast<std::size_t>(row_index - begin);
                    double* row_state = state + local_row * rank;
                    double residual = 0.0;

                    if (student) {
                        if (coordinate == 0) {
                            residual = probability;
                        } else {
                            scar_internal::linalg::row_major_matvec(
                                covariance,
                                rank,
                                rank,
                                row_state,
                                solved_projection);
                            const double conditional_mean =
                                scar_internal::linalg::dot(
                                    solved_projection,
                                    loading,
                                    rank);
                            const double correction =
                                scar_internal::linalg::dot(
                                    row_state,
                                    solved_projection,
                                    rank);
                            const double quadratic = std::max(
                                diagonal_quadratic[local_row] - correction,
                                0.0);
                            const double conditional_df = degrees
                                + static_cast<double>(coordinate);
                            const double scale =
                                (degrees + quadratic) / conditional_df;
                            residual = scar_internal::student_cdf_refined_value(
                                (latent - conditional_mean)
                                    / std::sqrt(
                                        conditional_variance * scale),
                                conditional_df);
                        }
                        const double inverse_uniqueness =
                            correlation.inverse_uniqueness()[coordinate];
                        for (std::size_t factor = 0;
                             factor < rank;
                             ++factor) {
                            row_state[factor] += latent
                                * inverse_uniqueness * loading[factor];
                        }
                        diagonal_quadratic[local_row] +=
                            latent * latent * inverse_uniqueness;
                    } else {
                        const double conditional_mean =
                            scar_internal::linalg::dot(
                                row_state, loading, rank);
                        const double innovation = latent - conditional_mean;
                        residual = math::normal_cdf(
                            innovation / std::sqrt(conditional_variance));
                        for (std::size_t factor = 0;
                             factor < rank;
                             ++factor) {
                            row_state[factor] +=
                                innovation / conditional_variance
                                * covariance_loading[factor];
                        }
                    }
                    if (!std::isfinite(latent)
                        || !std::isfinite(residual)) {
                        outcome.failed = true;
                        return;
                    }
                    out.residuals[row * dimension + coordinate] =
                        scar_internal::clip_pseudo_observation(residual);
                }
                } catch (...) {
                    failures.record(context.block_id, std::current_exception());
                    return;
                }
                outcome.operation = RosenblattOperation::Update;
                outcome.row = -1;
                try {
#ifdef SCAR_PARALLEL_TESTING
                    if (fault_hook != nullptr) {
                        fault_hook(coordinate, rosenblatt_testing::Operation::Update,
                                   context.block_id, -1);
                    }
#endif
                    for (std::size_t left = 0; left < rank; ++left) {
                        for (std::size_t right = 0; right < rank; ++right) {
                            covariance[left * rank + right] -=
                                covariance_loading[left] * covariance_loading[right]
                                / conditional_variance;
                        }
                    }
                } catch (...) {
                    failures.record(context.block_id, std::current_exception());
                    return;
                }
            }
        };
        if (plan.block_count() == 1) {
            // Legacy serial and nested paths keep their direct FP/TLS effects.
            evaluate_portion(0, static_cast<std::int64_t>(rows), {0, 0});
        } else {
            restore_failure = scar_internal::execute_parallel_plan_deferred(
                plan, evaluate_portion, {&failures, &RosenblattFailures::from_runtime});
        }
    }

    const RosenblattOutcome* selected = nullptr;
    for (const auto& outcome : outcomes) {
        if (outcome.failed && (selected == nullptr || earlier_outcome(outcome, *selected))) {
            selected = &outcome;
        }
    }
    if (selected != nullptr) {
        out.status = Status::NumericalFailure;
        out.failure.coordinate = static_cast<int>(selected->coordinate);
        out.failure.index = selected->row;
        for (std::size_t block = 0; block < plan.block_count(); ++block) {
            const auto& outcome = outcomes[block];
            for (std::int64_t row = plan.bounds()[block]; row < plan.bounds()[block + 1]; ++row) {
                const std::size_t offset = static_cast<std::size_t>(row) * dimension;
                std::fill(out.residuals.begin() + offset + selected->coordinate + 1,
                          out.residuals.begin() + offset + dimension, 0.0);
                if (selected->operation == RosenblattOperation::Preparation
                    || (selected->operation == RosenblattOperation::Rows
                        && outcome.failed && outcome.coordinate == selected->coordinate
                        && outcome.operation == RosenblattOperation::Rows && row >= outcome.row)) {
                    out.residuals[offset + selected->coordinate] = 0.0;
                }
            }
        }
    }
#ifdef SCAR_PARALLEL_TESTING
    if (output_observer != nullptr) {
        output_observer(out);
    }
#endif
    if (selected != nullptr && selected->exception) {
        if (restore_failure) {
            throw scar_internal::ParallelEnvironmentRestoreError(selected->exception, restore_failure);
        }
        std::rethrow_exception(selected->exception);
    }
    if (restore_failure) {
        // Currently unreachable: direct calls bypass the prepared executor.
        // Keep restoration failure explicit if that execution choice changes.
        std::rethrow_exception(restore_failure);
    }
    return out;
}

}  // namespace

#ifdef SCAR_PARALLEL_TESTING
namespace rosenblatt_testing {
void set_hooks(
    FaultHook fault, OutputObserver observer, RecordedObserver recorded) noexcept {
    factor_fault_hook = fault;
    factor_output_observer = observer;
    factor_recorded_observer = recorded;
}
}  // namespace rosenblatt_testing
#endif

GaussianScoreCorrelationResult gaussian_score_correlation(
    ObservationView u) {

    GaussianScoreCorrelationResult out;
    out.dimension = u.dim;
    std::size_t value_count = 0;
    if (u.dim < 2
        || u.n_obs == 0
        || !validate_observations(
            u, static_cast<std::size_t>(u.dim), value_count)) {
        out.status = Status::InvalidSize;
        return out;
    }

    const std::size_t dimension = static_cast<std::size_t>(u.dim);
    std::vector<double> scores(value_count, 0.0);
    std::vector<double> means(dimension, 0.0);
    for (std::size_t row = 0; row < u.n_obs; ++row) {
        for (std::size_t column = 0; column < dimension; ++column) {
            const std::size_t index = row * dimension + column;
            const double score = math::normal_quantile_refined(
                scar_internal::clip_pseudo_observation(u.data()[index]));
            if (!std::isfinite(score)) {
                out.status = Status::NumericalFailure;
                out.failure.index = static_cast<std::int64_t>(row);
                out.failure.coordinate = static_cast<int>(column);
                return out;
            }
            scores[index] = score;
            means[column] += score;
        }
    }
    for (double& mean : means) {
        mean /= static_cast<double>(u.n_obs);
    }

    std::vector<double> centered_squares(dimension, 0.0);
    for (std::size_t row = 0; row < u.n_obs; ++row) {
        for (std::size_t column = 0; column < dimension; ++column) {
            const std::size_t index = row * dimension + column;
            scores[index] -= means[column];
            centered_squares[column] += scores[index] * scores[index];
        }
    }
    for (std::size_t column = 0; column < dimension; ++column) {
        if (!std::isfinite(centered_squares[column])
            || !(centered_squares[column] > 0.0)) {
            out.status = Status::InvalidParameter;
            out.failure.coordinate = static_cast<int>(column);
            return out;
        }
    }

    out.correlation.assign(dimension * dimension, 0.0);
    for (std::size_t row = 0; row < dimension; ++row) {
        out.correlation[row * dimension + row] = 1.0;
        for (std::size_t column = 0; column < row; ++column) {
            double cross = 0.0;
            for (std::size_t observation = 0;
                 observation < u.n_obs;
                 ++observation) {
                cross += scores[observation * dimension + row]
                    * scores[observation * dimension + column];
            }
            const double value = cross / std::sqrt(
                centered_squares[row] * centered_squares[column]);
            if (!std::isfinite(value)) {
                out.status = Status::NumericalFailure;
                out.failure.coordinate = static_cast<int>(row);
                out.correlation.clear();
                return out;
            }
            out.correlation[row * dimension + column] = value;
            out.correlation[column * dimension + row] = value;
        }
    }

    std::vector<double> lower;
    std::size_t failed = dimension;
    if (!scar_internal::linalg::cholesky_symmetric(
            out.correlation.data(),
            dimension,
            lower,
            0.0,
            &failed)) {
        out.status = Status::NumericalFailure;
        out.failure.coordinate = failed < dimension
            ? static_cast<int>(failed)
            : -1;
        out.correlation.clear();
    }
    return out;
}

MultivariateRosenblattResult gaussian_rosenblatt_dense(
    DoubleView correlation,
    int dimension,
    ObservationView u,
    int n_threads) {

    MultivariateRosenblattResult out;
    out.n_rows = static_cast<std::int64_t>(u.n_obs);
    out.dimension = dimension;
    out.n_threads_requested = n_threads;
    if (dimension < 1 || !scar_internal::valid_thread_count(n_threads)) {
        out.status = Status::InvalidParameter;
        return out;
    }
    const std::size_t width = static_cast<std::size_t>(dimension);
    std::size_t values = 0;
    if (!validate_observations(u, width, values)) {
        out.status = Status::InvalidSize;
        return out;
    }
    std::vector<double> lower;
    int failed_coordinate = -1;
    if (!prepare_dense_correlation(
            correlation, width, lower, failed_coordinate)) {
        out.status = Status::NumericalFailure;
        out.failure.coordinate = failed_coordinate;
        return out;
    }
    out.correlation_factorizations = 1;
    out.residuals.assign(values, 0.0);
    if (u.n_obs == 0) {
        return out;
    }

    const int workers = rosenblatt_workers(
        n_threads, u.n_obs, width);
    out.parallel_blocks = workers;
    scar_internal::parallel_for_blocks(
        0,
        static_cast<std::int64_t>(u.n_obs),
        16,
        workers,
        [&](std::int64_t begin,
            std::int64_t end,
            std::size_t) {
            std::vector<double> whitened(width, 0.0);
            for (std::int64_t row_index = begin;
                 row_index < end;
                 ++row_index) {
                const std::size_t row =
                    static_cast<std::size_t>(row_index);
                for (std::size_t coordinate = 0;
                     coordinate < width;
                     ++coordinate) {
                    double value = math::normal_quantile_refined(
                        scar_internal::clip_pseudo_observation(
                            u.row(row)[coordinate]));
                    value -= scar_internal::linalg::dot(
                        lower.data() + coordinate * width,
                        whitened.data(),
                        coordinate);
                    whitened[coordinate] =
                        value / lower[coordinate * width + coordinate];
                    out.residuals[row * width + coordinate] =
                        scar_internal::clip_pseudo_observation(
                            math::normal_cdf(whitened[coordinate]));
                }
            }
        });
    return out;
}

MultivariateRosenblattResult gaussian_rosenblatt_equicorrelation(
    DoubleView rho,
    ObservationView u,
    int n_threads) {

    MultivariateRosenblattResult out;
    out.n_rows = static_cast<std::int64_t>(u.n_obs);
    out.dimension = u.dim;
    out.n_threads_requested = n_threads;
    if (u.dim < 2 || !scar_internal::valid_thread_count(n_threads)) {
        out.status = Status::InvalidParameter;
        return out;
    }
    const std::size_t dimension = static_cast<std::size_t>(u.dim);
    std::size_t values = 0;
    if (!validate_observations(u, dimension, values)
        || (rho.size() != 1 && rho.size() != u.n_obs)) {
        out.status = Status::InvalidSize;
        return out;
    }
    const double lower = -1.0 / static_cast<double>(dimension - 1);
    for (std::size_t index = 0; index < rho.size(); ++index) {
        if (!std::isfinite(rho[index])
            || !(rho[index] > lower)
            || !(rho[index] < 1.0)) {
            out.status = Status::InvalidParameter;
            out.failure.index = rho.size() == 1
                ? -1
                : static_cast<std::int64_t>(index);
            return out;
        }
    }
    out.residuals.assign(values, 0.0);
    if (u.n_obs == 0) {
        return out;
    }

    const int workers = rosenblatt_workers(
        n_threads, u.n_obs, dimension);
    out.parallel_blocks = workers;
    std::vector<std::int64_t> failures(
        static_cast<std::size_t>(workers), -1);
    scar_internal::parallel_for_blocks(
        0,
        static_cast<std::int64_t>(u.n_obs),
        16,
        workers,
        [&](std::int64_t begin,
            std::int64_t end,
            std::size_t block) {
            std::vector<double> latent(dimension, 0.0);
            for (std::int64_t row_index = begin;
                 row_index < end;
                 ++row_index) {
                const std::size_t row =
                    static_cast<std::size_t>(row_index);
                const double parameter = rho[rho.size() == 1 ? 0 : row];
                double prefix_sum = 0.0;
                for (std::size_t coordinate = 0;
                     coordinate < dimension;
                     ++coordinate) {
                    latent[coordinate] = math::normal_quantile_refined(
                        scar_internal::clip_pseudo_observation(
                            u.row(row)[coordinate]));
                    double residual = u.row(row)[coordinate];
                    if (coordinate != 0) {
                        const double denominator = 1.0
                            + static_cast<double>(coordinate - 1)
                                * parameter;
                        const double conditional_mean =
                            parameter * prefix_sum / denominator;
                        const double conditional_variance = std::max(
                            1.0
                                - static_cast<double>(coordinate)
                                    * parameter * parameter / denominator,
                            1e-10);
                        residual = math::normal_cdf(
                            (latent[coordinate] - conditional_mean)
                            / std::sqrt(conditional_variance));
                    }
                    if (!std::isfinite(latent[coordinate])
                        || !std::isfinite(residual)) {
                        failures[block] = row_index;
                        return;
                    }
                    out.residuals[row * dimension + coordinate] =
                        scar_internal::clip_pseudo_observation(residual);
                    prefix_sum += latent[coordinate];
                }
            }
        });
    for (std::int64_t failure : failures) {
        if (failure >= 0
            && (out.failure.index < 0 || failure < out.failure.index)) {
            out.failure.index = failure;
        }
    }
    if (out.failure.index >= 0) {
        out.status = Status::NumericalFailure;
    }
    return out;
}

MultivariateRosenblattResult gaussian_rosenblatt_factor(
    const FactorCorrelationOperator& correlation,
    ObservationView u,
    int n_threads) {

    return factor_rosenblatt(
        correlation, u, {}, n_threads, false);
}

MultivariateRosenblattResult student_rosenblatt_factor(
    const FactorCorrelationOperator& correlation,
    ObservationView u,
    DoubleView df,
    int n_threads) {

    return factor_rosenblatt(
        correlation, u, df, n_threads, true);
}

RadialSummaryResult radial_uniform_summary(
    ObservationView residuals,
    int n_threads) {

    RadialSummaryResult out;
    out.n_rows = static_cast<std::int64_t>(residuals.n_obs);
    out.dimension = residuals.dim;
    out.n_threads_requested = n_threads;
    if (residuals.dim < 1
        || !scar_internal::valid_thread_count(n_threads)) {
        out.status = Status::InvalidParameter;
        return out;
    }
    const std::size_t dimension =
        static_cast<std::size_t>(residuals.dim);
    std::size_t values = 0;
    if (!validate_observations(residuals, dimension, values)) {
        out.status = Status::InvalidSize;
        return out;
    }
    out.values.assign(residuals.n_obs, 0.0);
    if (residuals.n_obs == 0) {
        return out;
    }
    const int workers = rosenblatt_workers(
        n_threads, residuals.n_obs, dimension);
    out.parallel_blocks = workers;
    scar_internal::parallel_for_blocks(
        0,
        static_cast<std::int64_t>(residuals.n_obs),
        16,
        workers,
        [&](std::int64_t begin,
            std::int64_t end,
            std::size_t) {
            for (std::int64_t row_index = begin;
                 row_index < end;
                 ++row_index) {
                const std::size_t row =
                    static_cast<std::size_t>(row_index);
                double quadratic = 0.0;
                for (std::size_t coordinate = 0;
                     coordinate < dimension;
                     ++coordinate) {
                    const double quantile =
                        math::normal_quantile_refined(
                            scar_internal::clip_pseudo_observation(
                                residuals.row(row)[coordinate]));
                    quadratic += quantile * quantile;
                }
                out.values[row] = math::regularized_gamma_p(
                    0.5 * static_cast<double>(dimension),
                    0.5 * quadratic);
            }
        });
    return out;
}

}  // namespace scar
