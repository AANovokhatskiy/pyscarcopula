#include "scar/factor.hpp"

#include "scar/detail/copula.hpp"
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
    std::vector<double> loading_gradient;
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
            && (result.failure_index < 0
                || worker.failure_index < result.failure_index)) {
            result.failure_index = worker.failure_index;
        }
    }
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
    std::size_t loading_values = 0;
    if (!scar_internal::checked_size_mul(
            rows, dimension, input_values)
        || !scar_internal::checked_size_mul(
            dimension, rank, loading_values)) {
        throw std::invalid_argument(
            "factor Student joint input shape is not representable");
    }

    std::vector<double> precision_times_loadings(
        loading_values, 0.0);
    std::vector<double> precision_diagonal(dimension, 0.0);
    std::vector<double> small(rank, 0.0);
    const std::vector<double>& weighted_loadings =
        correlation.weighted_loadings();
    const std::vector<double>& inverse_uniqueness =
        correlation.inverse_uniqueness();
    for (std::size_t row = 0; row < dimension; ++row) {
        const double* weighted =
            weighted_loadings.data() + row * rank;
        std::copy(weighted, weighted + rank, small.begin());
        correlation.solve_core_inplace(small.data());
        double diagonal_correction = 0.0;
        for (std::size_t factor = 0; factor < rank; ++factor) {
            precision_times_loadings[row * rank + factor] =
                small[factor];
            diagonal_correction += weighted[factor] * small[factor];
        }
        precision_diagonal[row] =
            inverse_uniqueness[row] - diagonal_correction;
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
    FactorStudentJointResult result;
    result.dlog_likelihood_dloadings.assign(loading_values, 0.0);
    result.n_threads_requested = n_threads;
    result.reduction_blocks = static_cast<int>(reduction_blocks);
    if (rows == 0) {
        return result;
    }

    std::vector<JointBlockResult> blocks(reduction_blocks);
    for (JointBlockResult& block : blocks) {
        block.loading_gradient.assign(loading_values, 0.0);
    }
    result.reduction_workspace_bytes = reduction_bytes;
    const int threads = scar_internal::worker_count_for_items(
        n_threads, reduction_blocks, 4);
    result.worker_workspace_peak_bytes = worker_bytes;

    scar_internal::parallel_for_blocks(
        0,
        static_cast<std::int64_t>(reduction_blocks),
        1,
        threads,
        [&](std::int64_t begin,
            std::int64_t end,
            std::size_t) {
            std::vector<double> quantiles(dimension, 0.0);
            std::vector<double> quantile_derivatives(dimension, 0.0);
            std::vector<double> precision_times_quantiles(
                dimension, 0.0);
            std::vector<double> projected(rank, 0.0);
            for (std::int64_t block_index = begin;
                 block_index < end;
                 ++block_index) {
                JointBlockResult& block =
                    blocks[static_cast<std::size_t>(block_index)];
                block.ran = true;
                const std::size_t row_begin =
                    rows * static_cast<std::size_t>(block_index)
                    / reduction_blocks;
                const std::size_t row_end =
                    rows * (
                        static_cast<std::size_t>(block_index) + 1)
                    / reduction_blocks;
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
                    correlation.solve_rows(
                        quantiles.data(),
                        1,
                        precision_times_quantiles.data(),
                        1);
                    double quadratic_form = 0.0;
                    double quadratic_form_derivative = 0.0;
                    std::fill(projected.begin(), projected.end(), 0.0);
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
                            quantiles.data(),
                            quantile_derivatives.data(),
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
                            block.loading_gradient.data() + column * rank;
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
        });

    for (const JointBlockResult& block : blocks) {
        if (!block.ran) {
            continue;
        }
        ++result.parallel_blocks;
        if (block.failure_index >= 0) {
            if (
                    result.failure_index < 0
                    || block.failure_index < result.failure_index) {
                result.failure_index = block.failure_index;
            }
            continue;
        }
        result.log_likelihood += block.log_likelihood;
        result.dlog_likelihood_ddf += block.dlog_likelihood_ddf;
        for (std::size_t index = 0;
             index < loading_values;
             ++index) {
            result.dlog_likelihood_dloadings[index] +=
                block.loading_gradient[index];
        }
    }
    if (result.failure_index >= 0) {
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

}  // namespace scar
