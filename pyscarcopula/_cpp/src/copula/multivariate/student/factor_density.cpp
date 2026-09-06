#include "scar/copula/multivariate/student/factor_density.hpp"

#include "scar/copula/multivariate/correlation/factor_parameterization.hpp"
#include "scar/copula/multivariate/student/density.hpp"
#include "scar/copula/multivariate/student/quantile.hpp"
#include "scar/copula/multivariate/correlation/factor_solve.hpp"
#include "scar/detail/parallel.hpp"
#include "scar/detail/safety.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace scar {
namespace {

struct WorkerResult {
    bool ran = false;
    std::int64_t failure_index = -1;
    std::size_t workspace_peak_bytes = 0;
};

struct JointBlockResult {
    bool ran = false;
    double log_likelihood = 0.0;
    double dlog_likelihood_ddf = 0.0;
    std::int64_t failure_index = -1;
};

constexpr std::size_t kJointReductionBlocks = 64;

}  // namespace

FactorStudentRowsResult factor_student_log_pdf_and_dlog_ddf(
    const FactorCorrelationOperator& correlation,
    const double* observations,
    std::size_t rows,
    const double* df,
    std::size_t df_count,
    int n_threads) {

    scar_internal::validate_thread_count(n_threads);
    if ((rows > 0 && observations == nullptr)
        || df == nullptr
        || (df_count != 1 && df_count != rows)) {
        throw std::invalid_argument(
            "df must contain one value or one value per observation row");
    }
    for (std::size_t index = 0; index < df_count; ++index) {
        if (!std::isfinite(df[index]) || df[index] <= 2.0) {
            throw std::invalid_argument(
                "Student degrees of freedom must be finite and greater than 2");
        }
    }

    const std::size_t dimension = correlation.dimension();
    std::size_t input_values = 0;
    std::size_t three_dimensions = 0;
    std::size_t workspace_values = 0;
    if (!scar_internal::checked_size_mul(
            rows, dimension, input_values)
        || !scar_internal::checked_size_mul(
            dimension, std::size_t{3}, three_dimensions)
        || !scar_internal::checked_size_add(
            three_dimensions, correlation.rank(), workspace_values)) {
        throw std::invalid_argument(
            "factor Student input shape is not representable");
    }

    FactorStudentRowsResult result;
    result.log_pdf.assign(
        rows, std::numeric_limits<double>::quiet_NaN());
    result.dlog_ddf.assign(
        rows, std::numeric_limits<double>::quiet_NaN());
    result.n_threads_requested = n_threads;

    const int threads = scar_internal::worker_count_for_items(
        n_threads, rows, 4);
    std::vector<WorkerResult> worker_results(
        static_cast<std::size_t>(threads));
    scar_internal::parallel_for_blocks(
        0,
        static_cast<std::int64_t>(rows),
        1,
        threads,
        [&](std::int64_t begin,
            std::int64_t end,
            std::size_t block) {
            WorkerResult& worker = worker_results[block];
            worker.ran = true;
            worker.workspace_peak_bytes =
                workspace_values * sizeof(double);
            std::vector<double> quantiles(dimension, 0.0);
            std::vector<double> quantile_derivatives(dimension, 0.0);
            std::vector<double> precision_times_quantiles(
                dimension, 0.0);
            for (std::int64_t row_index = begin;
                 row_index < end;
                 ++row_index) {
                const std::size_t row =
                    static_cast<std::size_t>(row_index);
                const double* observation =
                    observations + row * dimension;
                bool finite = true;
                for (std::size_t column = 0;
                     column < dimension;
                     ++column) {
                    if (!std::isfinite(observation[column])) {
                        finite = false;
                        break;
                    }
                }
                if (!finite) {
                    worker.failure_index = row_index;
                    return;
                }

                const double row_df = df[df_count == 1 ? 0 : row];
                for (std::size_t column = 0;
                     column < dimension;
                     ++column) {
                    scar_internal::student_quantile_value_and_derivative(
                        observation[column],
                        row_df,
                        quantiles[column],
                        quantile_derivatives[column]);
                }
                correlation.solve_rows(
                    quantiles.data(),
                    1,
                    precision_times_quantiles.data(),
                    1);
                double quadratic_form = 0.0;
                double quadratic_form_derivative = 0.0;
                for (std::size_t column = 0;
                     column < dimension;
                     ++column) {
                    quadratic_form +=
                        quantiles[column]
                        * precision_times_quantiles[column];
                    quadratic_form_derivative +=
                        2.0
                        * precision_times_quantiles[column]
                        * quantile_derivatives[column];
                }
                if (!scar_internal::student_log_pdf_from_quantiles(
                        quantiles.data(),
                        quantile_derivatives.data(),
                        dimension,
                        row_df,
                        correlation.logdet(),
                        quadratic_form,
                        quadratic_form_derivative,
                        result.log_pdf[row],
                        &result.dlog_ddf[row])) {
                    worker.failure_index = row_index;
                    return;
                }
            }
        });

    for (const WorkerResult& worker : worker_results) {
        if (!worker.ran) {
            continue;
        }
        ++result.row_parallel_blocks;
        result.worker_workspace_peak_bytes = std::max(
            result.worker_workspace_peak_bytes,
            worker.workspace_peak_bytes);
        if (worker.failure_index >= 0
            && (result.failure.index < 0
                || worker.failure_index < result.failure.index)) {
            result.failure.index = worker.failure_index;
        }
    }
    if (result.failure.index >= 0) {
        result.status = Status::NumericalFailure;
        result.log_likelihood =
            std::numeric_limits<double>::quiet_NaN();
        result.dlog_likelihood_ddf =
            std::numeric_limits<double>::quiet_NaN();
        result.negative_log_likelihood =
            std::numeric_limits<double>::quiet_NaN();
        result.dnegative_log_likelihood_ddf =
            std::numeric_limits<double>::quiet_NaN();
        return result;
    }
    for (std::size_t row = 0; row < rows; ++row) {
        result.log_likelihood += result.log_pdf[row];
        result.dlog_likelihood_ddf += result.dlog_ddf[row];
    }
    result.negative_log_likelihood = -result.log_likelihood;
    result.dnegative_log_likelihood_ddf =
        -result.dlog_likelihood_ddf;
    return result;
}

FactorStudentJointResult factor_student_joint_likelihood_gradient(
    const FactorCorrelationOperator& correlation,
    const double* observations,
    std::size_t rows,
    double df,
    int n_threads) {

    scar_internal::validate_thread_count(n_threads);
    if ((rows > 0 && observations == nullptr)
        || !std::isfinite(df)
        || df <= 2.0) {
        throw std::invalid_argument(
            "observations must be present and df must be finite and greater "
            "than 2");
    }

    const std::size_t dimension = correlation.dimension();
    const std::size_t rank = correlation.rank();
    std::size_t input_values = 0;
    std::size_t input_bytes = 0;
    std::size_t loading_values = 0;
    std::size_t loading_bytes = 0;
    if (rows > static_cast<std::size_t>(
            std::numeric_limits<std::int64_t>::max())
        || !scar_internal::checked_size_mul(
            rows, dimension, input_values)
        || !scar_internal::checked_size_mul(
            input_values, sizeof(double), input_bytes)
        || !scar_internal::checked_size_mul(
            dimension, rank, loading_values)
        || !scar_internal::checked_size_mul(
            loading_values, sizeof(double), loading_bytes)) {
        throw std::invalid_argument(
            "factor Student joint input shape is not representable");
    }

    const std::size_t reduction_blocks = std::min(
        rows, kJointReductionBlocks);
    std::size_t reduction_values = 0;
    std::size_t reduction_bytes = 0;
    std::size_t three_dimensions = 0;
    std::size_t worker_values = 0;
    std::size_t worker_bytes = 0;
    if (!scar_internal::checked_size_mul(
            reduction_blocks, loading_values, reduction_values)
        || !scar_internal::checked_size_mul(
            reduction_values, sizeof(double), reduction_bytes)
        || !scar_internal::checked_size_mul(
            dimension, std::size_t{3}, three_dimensions)
        || !scar_internal::checked_size_add(
            three_dimensions, rank, worker_values)
        || !scar_internal::checked_size_mul(
            worker_values, sizeof(double), worker_bytes)) {
        throw std::invalid_argument(
            "factor Student joint workspace size is not representable");
    }
    // Numerical partials and their ordered fold do not depend on this budget.
    // A large request may use fewer workers without falling back to serial.
    const int threads = scar_internal::limit_worker_count(
        n_threads, std::max(std::size_t{1}, reduction_blocks / 4));
    const auto plan = scar_internal::make_parallel_execution_plan(
        0, static_cast<std::int64_t>(reduction_blocks), reduction_blocks, threads);
    const std::size_t slots = scar_internal::parallel_execution_slots(plan);
    std::size_t scratch_values = 0;
    std::size_t scratch_bytes = 0;
    std::size_t block_bytes = 0;
    std::size_t prepared_values = 0;
    std::size_t prepared_bytes = 0;
    std::size_t rank_bytes = 0;
    std::size_t boundary_bytes = 0;
    std::size_t preparation_bytes = 0;
    std::size_t execution_bytes = 0;
    std::size_t conversion_bytes = 0;
    // Check each lifetime separately. Their maximum is the owned buffer peak;
    // borrowed inputs/operator and runtime bookkeeping are accounted elsewhere.
    if (!scar_internal::checked_size_mul(slots, worker_values, scratch_values)
        || !scar_internal::checked_size_mul(
            scratch_values, sizeof(double), scratch_bytes)
        || !scar_internal::checked_size_mul(
            reduction_blocks, sizeof(JointBlockResult), block_bytes)
        || !scar_internal::checked_size_add(
            loading_values, dimension, prepared_values)
        || !scar_internal::checked_size_mul(
            prepared_values, sizeof(double), prepared_bytes)
        || !scar_internal::checked_size_mul(rank, sizeof(double), rank_bytes)
        || !scar_internal::checked_size_mul(
            plan.bounds().size(), sizeof(std::int64_t), boundary_bytes)
        || !scar_internal::checked_size_add(
            prepared_bytes, loading_bytes, preparation_bytes)
        || !scar_internal::checked_size_add(
            preparation_bytes, boundary_bytes, preparation_bytes)
        || !scar_internal::checked_size_add(
            preparation_bytes, rank_bytes, preparation_bytes)
        || !scar_internal::checked_size_add(
            prepared_bytes, loading_bytes, execution_bytes)
        || !scar_internal::checked_size_add(
            execution_bytes, boundary_bytes, execution_bytes)
        || !scar_internal::checked_size_add(
            execution_bytes, reduction_bytes, execution_bytes)
        || !scar_internal::checked_size_add(
            execution_bytes, scratch_bytes, execution_bytes)
        || !scar_internal::checked_size_add(
            execution_bytes, block_bytes, execution_bytes)
        || !scar_internal::checked_size_add(
            loading_bytes, loading_bytes, conversion_bytes)) {
        throw std::invalid_argument(
            "factor Student joint workspace size is not representable");
    }

    FactorStudentJointResult result;
    result.dlog_likelihood_dloadings.assign(loading_values, 0.0);
    result.n_threads_requested = n_threads;
    result.reduction_blocks = static_cast<int>(reduction_blocks);
    if (rows == 0) {
        return result;
    }
    result.reduction_workspace_bytes = reduction_bytes;
    result.worker_workspace_peak_bytes = worker_bytes;
    result.planned_worker_slots = slots;
    result.planned_worker_workspace_bytes = scratch_bytes;

    std::vector<double> precision_times_loadings(loading_values, 0.0);
    std::vector<double> precision_diagonal(dimension, 0.0);
    {
        std::vector<double> small(rank, 0.0);
        const auto& weighted_loadings = correlation.weighted_loadings();
        const auto& inverse_uniqueness = correlation.inverse_uniqueness();
        for (std::size_t row = 0; row < dimension; ++row) {
            const double* weighted = weighted_loadings.data() + row * rank;
            std::copy(weighted, weighted + rank, small.begin());
            correlation.solve_core_inplace(small.data());
            double diagonal_correction = 0.0;
            for (std::size_t factor = 0; factor < rank; ++factor) {
                precision_times_loadings[row * rank + factor] = small[factor];
                diagonal_correction += weighted[factor] * small[factor];
            }
            precision_diagonal[row] =
                inverse_uniqueness[row] - diagonal_correction;
        }
    }

    std::vector<JointBlockResult> blocks(reduction_blocks);
    std::vector<double> loading_gradients(reduction_values, 0.0);
    {
        std::vector<double> scratch(scratch_values, 0.0);
        const scar_internal::PreparedParallelBlockFunction evaluate_blocks =
        [&](std::int64_t begin,
            std::int64_t end,
            const scar_internal::ParallelBlockContext& context) {
            double* quantiles = scratch.data() + context.worker_slot * worker_values;
            double* quantile_derivatives = quantiles + dimension;
            double* precision_times_quantiles = quantile_derivatives + dimension;
            double* projected = precision_times_quantiles + dimension;
            for (std::int64_t block_index = begin;
                 block_index < end;
                 ++block_index) {
                JointBlockResult& block =
                    blocks[static_cast<std::size_t>(block_index)];
                block.ran = true;
                const std::size_t block_id = static_cast<std::size_t>(block_index);
                double* block_gradient =
                    loading_gradients.data() + block_id * loading_values;
                // Same floor-based partition without overflowing rows * block_id.
                const std::size_t quotient = rows / reduction_blocks;
                const std::size_t remainder = rows % reduction_blocks;
                const std::size_t row_begin = quotient * block_id
                    + remainder * block_id / reduction_blocks;
                const std::size_t row_end = quotient * (block_id + 1)
                    + remainder * (block_id + 1) / reduction_blocks;
                for (std::size_t row = row_begin;
                     row < row_end;
                     ++row) {
                    const double* observation =
                        observations + row * dimension;
                    bool finite = true;
                    for (std::size_t column = 0;
                         column < dimension;
                         ++column) {
                        if (!std::isfinite(observation[column])) {
                            finite = false;
                            break;
                        }
                        scar_internal::
                            student_quantile_value_and_derivative(
                                observation[column],
                                df,
                                quantiles[column],
                                quantile_derivatives[column]);
                    }
                    if (!finite) {
                        block.failure_index =
                            static_cast<std::int64_t>(row);
                        break;
                    }
                    scar_internal::factor_solve_row_with_workspace(
                        correlation, quantiles, precision_times_quantiles,
                        projected, rank);
                    double quadratic_form = 0.0;
                    double quadratic_form_derivative = 0.0;
                    std::fill(projected, projected + rank, 0.0);
                    for (std::size_t column = 0;
                         column < dimension;
                         ++column) {
                        const double precision_value =
                            precision_times_quantiles[column];
                        quadratic_form +=
                            quantiles[column] * precision_value;
                        quadratic_form_derivative +=
                            2.0
                            * precision_value
                            * quantile_derivatives[column];
                        const double* loading =
                            correlation.loadings().data() + column * rank;
                        for (std::size_t factor = 0;
                             factor < rank;
                             ++factor) {
                            projected[factor] +=
                                precision_value * loading[factor];
                        }
                    }
                    double row_log_pdf = 0.0;
                    double row_df_gradient = 0.0;
                    if (!scar_internal::student_log_pdf_from_quantiles(
                            quantiles,
                            quantile_derivatives,
                            dimension,
                            df,
                            correlation.logdet(),
                            quadratic_form,
                            quadratic_form_derivative,
                            row_log_pdf,
                            &row_df_gradient)) {
                        block.failure_index =
                            static_cast<std::int64_t>(row);
                        break;
                    }
                    block.log_likelihood += row_log_pdf;
                    block.dlog_likelihood_ddf += row_df_gradient;
                    const double shape_weight =
                        (df + static_cast<double>(dimension))
                        / (2.0 * (df + quadratic_form));
                    for (std::size_t column = 0;
                         column < dimension;
                         ++column) {
                        const double precision_value =
                            precision_times_quantiles[column];
                        const double diagonal_gradient =
                            -0.5 * precision_diagonal[column]
                            + shape_weight
                                * precision_value
                                * precision_value;
                        const double* loading =
                            correlation.loadings().data() + column * rank;
                        double* gradient =
                            block_gradient + column * rank;
                        const double* precision_loading =
                            precision_times_loadings.data()
                            + column * rank;
                        for (std::size_t factor = 0;
                             factor < rank;
                             ++factor) {
                            const double matrix_gradient_times_loading =
                                -0.5 * precision_loading[factor]
                                + shape_weight
                                    * precision_value
                                    * projected[factor];
                            gradient[factor] += 2.0 * (
                                matrix_gradient_times_loading
                                - diagonal_gradient * loading[factor]);
                        }
                    }
                }
            }
        };
        if (threads == 1) {
            // Keep the legacy serial callback's caller/TLS/fenv behavior.
            evaluate_blocks(0, static_cast<std::int64_t>(reduction_blocks), {0, 0});
        } else {
            scar_internal::execute_parallel_plan(plan, evaluate_blocks);
        }
    }

    for (std::size_t block_id = 0; block_id < reduction_blocks; ++block_id) {
        const JointBlockResult& block = blocks[block_id];
        if (!block.ran) {
            continue;
        }
        ++result.parallel_blocks;
        if (block.failure_index >= 0) {
            if (
                    result.failure.index < 0
                    || block.failure_index < result.failure.index) {
                result.failure.index = block.failure_index;
            }
            continue;
        }
        result.log_likelihood += block.log_likelihood;
        result.dlog_likelihood_ddf += block.dlog_likelihood_ddf;
        for (std::size_t index = 0;
             index < loading_values;
             ++index) {
            result.dlog_likelihood_dloadings[index] +=
                loading_gradients[block_id * loading_values + index];
        }
    }
    if (result.failure.index >= 0) {
        result.status = Status::NumericalFailure;
        result.log_likelihood =
            std::numeric_limits<double>::quiet_NaN();
        result.dlog_likelihood_ddf =
            std::numeric_limits<double>::quiet_NaN();
        std::fill(
            result.dlog_likelihood_dloadings.begin(),
            result.dlog_likelihood_dloadings.end(),
            std::numeric_limits<double>::quiet_NaN());
    }
    return result;
}

FactorStudentPenalizedObjectiveResult
factor_student_penalized_parameterized_objective_gradient(
    const double* observations,
    std::size_t rows,
    double df,
    const double* parameters,
    std::size_t parameter_count,
    const double* free_rows,
    const double* free_columns,
    const double* diagonal_entries,
    std::size_t dimension,
    std::size_t rank,
    double max_norm,
    double uniqueness_min,
    double condition_max,
    double penalty,
    int n_threads) {

    if (!std::isfinite(penalty)
        || penalty < 0.0
        || !std::isfinite(condition_max)
        || !(condition_max > 0.0)) {
        throw std::invalid_argument(
            "factor Student penalty and condition gate are invalid");
    }
    const DoubleView parameter_view{parameters, parameter_count};
    const DoubleView free_row_view{free_rows, parameter_count};
    const DoubleView free_column_view{free_columns, parameter_count};
    const DoubleView diagonal_view{diagonal_entries, parameter_count};
    Result<std::vector<double>> transformed =
        factor_parameterization_loadings(
            parameter_view,
            free_row_view,
            free_column_view,
            diagonal_view,
            dimension,
            rank,
            max_norm);

    FactorStudentPenalizedObjectiveResult result;
    result.n_threads_requested = n_threads;
    if (!transformed.is_ok()) {
        result.status = transformed.status;
        result.failure = transformed.failure;
        return result;
    }
    result.loadings = std::move(transformed.value);
    FactorCorrelationOperator correlation(
        result.loadings, dimension, rank, uniqueness_min);
    result.condition_estimate = correlation.condition_estimate();
    if (result.condition_estimate > condition_max) {
        result.status = Status::InvalidParameter;
        return result;
    }
    FactorStudentJointResult joint =
        factor_student_joint_likelihood_gradient(
            correlation, observations, rows, df, n_threads);
    result.status = joint.status;
    result.failure = joint.failure;
    result.n_threads_requested = joint.n_threads_requested;
    result.reduction_blocks = joint.reduction_blocks;
    result.parallel_blocks = joint.parallel_blocks;
    result.worker_workspace_peak_bytes =
        joint.worker_workspace_peak_bytes;
    result.reduction_workspace_bytes =
        joint.reduction_workspace_bytes;
    result.planned_worker_slots = joint.planned_worker_slots;
    result.planned_worker_workspace_bytes = joint.planned_worker_workspace_bytes;
    result.log_likelihood = joint.log_likelihood;
    if (!joint.is_ok()) {
        return result;
    }

    std::vector<double> penalized_loading_gradient =
        std::move(joint.dlog_likelihood_dloadings);
    double squared_norm = 0.0;
    for (std::size_t index = 0;
         index < result.loadings.size();
         ++index) {
        squared_norm += result.loadings[index] * result.loadings[index];
        penalized_loading_gradient[index] =
            -penalized_loading_gradient[index]
            + 2.0 * penalty * result.loadings[index];
    }
    Result<std::vector<double>> pulled_back =
        factor_parameterization_pullback(
            parameter_view,
            DoubleView{
                penalized_loading_gradient.data(),
                penalized_loading_gradient.size()},
            free_row_view,
            free_column_view,
            diagonal_view,
            dimension,
            rank,
            max_norm);
    if (!pulled_back.is_ok()) {
        result.status = pulled_back.status;
        result.failure = pulled_back.failure;
        return result;
    }
    result.objective = -joint.log_likelihood + penalty * squared_norm;
    result.gradient.resize(parameter_count + 1);
    result.gradient[0] = -joint.dlog_likelihood_ddf;
    std::copy(
        pulled_back.value.begin(),
        pulled_back.value.end(),
        result.gradient.begin() + 1);
    return result;
}

}  // namespace scar
