#include "scar/copula/multivariate/sampling.hpp"

#include "scar/copula/multivariate/correlation/factor.hpp"
#include "scar/copula/multivariate/gaussian/conditional.hpp"
#include "scar/copula/multivariate/student/conditional.hpp"
#include "scar/copula/multivariate/student/distribution.hpp"
#include "scar/copula/multivariate/student/quantile.hpp"
#include "scar/core/checked_arithmetic.hpp"
#include "scar/detail/linalg.hpp"
#include "scar/detail/parallel.hpp"
#include "scar/detail/safety.hpp"
#include "scar/math/normal.hpp"
#include "scar/math/gamma.hpp"
#include "scar/numerical_constants.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace scar {
namespace {

struct SamplingBlockResult {
    bool ran = false;
    std::int64_t failure_index = -1;
};

bool checked_shape(
    std::int64_t n_rows,
    std::size_t width,
    std::size_t& rows,
    std::size_t& values) {

    return scar_internal::checked_nonnegative_size(n_rows, rows)
        && core::checked_size_mul(rows, width, values);
}

bool all_finite(DoubleView values) {
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (!std::isfinite(values[index])) {
            return false;
        }
    }
    return true;
}

double conditional_probability(double value) {
    return std::clamp(
        value,
        numerical::kConditionalSampleEps,
        1.0 - numerical::kConditionalSampleEps);
}

double df_at(DoubleView df, std::size_t row) {
    return df[df.size() == 1 ? 0 : row];
}

bool valid_degrees_of_freedom(DoubleView df, std::size_t rows) {
    if (df.size() != 1 && df.size() != rows) {
        return false;
    }
    for (std::size_t index = 0; index < df.size(); ++index) {
        const double value = df[index];
        if (!std::isfinite(value) || !(value > 2.0)) {
            return false;
        }
    }
    return true;
}

bool validate_given_indices(
    std::size_t dimension,
    const std::vector<int>& given_indices,
    std::vector<int>& free_indices) {

    if (given_indices.empty() || given_indices.size() >= dimension) {
        return false;
    }
    std::vector<bool> is_given(dimension, false);
    for (int index : given_indices) {
        if (index < 0
            || static_cast<std::size_t>(index) >= dimension
            || is_given[static_cast<std::size_t>(index)]) {
            return false;
        }
        is_given[static_cast<std::size_t>(index)] = true;
    }
    free_indices.clear();
    free_indices.reserve(dimension - given_indices.size());
    for (std::size_t index = 0; index < dimension; ++index) {
        if (!is_given[index]) {
            free_indices.push_back(static_cast<int>(index));
        }
    }
    return true;
}

bool valid_given_uniforms(
    DoubleView values,
    std::size_t rows,
    std::size_t n_given) {

    if (values.size() != n_given
        && values.size() != rows * n_given) {
        return false;
    }
    for (std::size_t index = 0; index < values.size(); ++index) {
        const double value = values[index];
        if (!std::isfinite(value) || !(value > 0.0) || !(value < 1.0)) {
            return false;
        }
    }
    return true;
}

double given_at(
    DoubleView values,
    std::size_t n_given,
    std::size_t row,
    std::size_t column) {

    return values[values.size() == n_given
        ? column
        : row * n_given + column];
}

int sampling_worker_count(int n_threads, std::size_t rows) {
    return scar_internal::worker_count_for_items(n_threads, rows, 32);
}

void finish_parallel_result(
    ConditionalSampleResult& out,
    const std::vector<SamplingBlockResult>& blocks,
    std::size_t width) {

    for (const SamplingBlockResult& block : blocks) {
        if (!block.ran) {
            continue;
        }
        ++out.parallel_blocks;
        if (block.failure_index >= 0
            && (out.failure.index < 0
                || block.failure_index < out.failure.index)) {
            out.failure.index = block.failure_index;
            out.status = Status::NumericalFailure;
        }
    }
    if (out.failure.index >= 0) {
        const std::size_t first_uncomputed =
            static_cast<std::size_t>(out.failure.index + 1) * width;
        std::fill(
            out.values.begin() + first_uncomputed,
            out.values.end(),
            0.0);
    }
}

ConditionalSampleResult dense_sample(
    DoubleView correlation,
    int dimension,
    DoubleView df,
    DoubleView normal_draws,
    DoubleView chi_square_draws,
    std::int64_t n_rows,
    int n_threads,
    bool student) {

    ConditionalSampleResult out;
    out.n_rows = n_rows;
    out.n_free = dimension;
    out.n_threads_requested = n_threads;
    std::size_t rows = 0;
    std::size_t values = 0;
    if (dimension < 2
        || !checked_shape(
            n_rows, static_cast<std::size_t>(dimension), rows, values)
        || !scar_internal::valid_thread_count(n_threads)
        || correlation.size()
            != static_cast<std::size_t>(dimension)
                * static_cast<std::size_t>(dimension)
        || normal_draws.size() != values) {
        out.status = Status::InvalidSize;
        return out;
    }
    if (!all_finite(correlation) || !all_finite(normal_draws)) {
        out.status = Status::InvalidParameter;
        return out;
    }
    if (student
        && (!valid_degrees_of_freedom(df, rows)
            || chi_square_draws.size() != rows
            || !all_finite(chi_square_draws))) {
        out.status = Status::InvalidParameter;
        return out;
    }
    for (std::size_t index = 0;
         index < chi_square_draws.size();
         ++index) {
        const double draw = chi_square_draws[index];
        if (!(draw > 0.0)) {
            out.status = Status::InvalidParameter;
            return out;
        }
    }
    if (rows == 0) {
        return out;
    }

    const std::size_t d = static_cast<std::size_t>(dimension);
    std::vector<double> lower;
    if (!scar_internal::linalg::cholesky_symmetric(
            correlation.data(), d, lower)) {
        out.status = Status::NumericalFailure;
        out.failure.index = 0;
        return out;
    }
    out.correlation_factorizations = 1;
    out.values.assign(values, 0.0);
    const int workers = sampling_worker_count(n_threads, rows);
    std::vector<SamplingBlockResult> blocks(
        static_cast<std::size_t>(workers));
    scar_internal::parallel_for_blocks(
        0,
        n_rows,
        32,
        workers,
        [&](std::int64_t begin,
            std::int64_t end,
            std::size_t block_index) {
            SamplingBlockResult& block = blocks[block_index];
            block.ran = true;
            std::vector<double> latent(d, 0.0);
            for (std::int64_t row_index = begin;
                 row_index < end;
                 ++row_index) {
                const std::size_t row =
                    static_cast<std::size_t>(row_index);
                scar_internal::linalg::lower_triangular_matvec(
                    lower.data(),
                    d,
                    normal_draws.data() + row * d,
                    latent.data());
                const double degrees = student ? df_at(df, row) : 0.0;
                const double scale = student
                    ? std::sqrt(degrees / chi_square_draws[row])
                    : 1.0;
                for (std::size_t column = 0; column < d; ++column) {
                    const double value = scale * latent[column];
                    const double probability = student
                        ? scar_internal::student_cdf_refined_value(value, degrees)
                        : math::normal_cdf(value);
                    if (!std::isfinite(probability)) {
                        block.failure_index = row_index;
                        return;
                    }
                    out.values[row * d + column] = probability;
                }
            }
        });
    finish_parallel_result(out, blocks, d);
    return out;
}

ConditionalSampleResult factor_sample(
    const FactorCorrelationOperator& correlation,
    DoubleView df,
    DoubleView factor_draws,
    DoubleView residual_draws,
    DoubleView chi_square_draws,
    std::int64_t n_rows,
    int n_threads,
    bool student) {

    ConditionalSampleResult out;
    out.n_rows = n_rows;
    out.n_free = static_cast<std::int64_t>(correlation.dimension());
    out.n_threads_requested = n_threads;
    std::size_t rows = 0;
    std::size_t residual_values = 0;
    std::size_t factor_values = 0;
    if (!checked_shape(
            n_rows, correlation.dimension(), rows, residual_values)
        || !core::checked_size_mul(
            rows, correlation.rank(), factor_values)
        || !scar_internal::valid_thread_count(n_threads)
        || factor_draws.size() != factor_values
        || residual_draws.size() != residual_values) {
        out.status = Status::InvalidSize;
        return out;
    }
    if (!all_finite(factor_draws) || !all_finite(residual_draws)) {
        out.status = Status::InvalidParameter;
        return out;
    }
    if (student
        && (!valid_degrees_of_freedom(df, rows)
            || chi_square_draws.size() != rows
            || !all_finite(chi_square_draws))) {
        out.status = Status::InvalidParameter;
        return out;
    }
    for (std::size_t index = 0;
         index < chi_square_draws.size();
         ++index) {
        const double draw = chi_square_draws[index];
        if (!(draw > 0.0)) {
            out.status = Status::InvalidParameter;
            return out;
        }
    }
    if (rows == 0) {
        return out;
    }

    std::vector<double> factors(
        factor_draws.data(), factor_draws.data() + factor_draws.size());
    out.values.assign(
        residual_draws.data(), residual_draws.data() + residual_draws.size());
    if (student) {
        for (std::size_t row = 0; row < rows; ++row) {
            const double scale = std::sqrt(
                df_at(df, row) / chi_square_draws[row]);
            for (std::size_t index = 0;
                 index < correlation.rank();
                 ++index) {
                factors[row * correlation.rank() + index] *= scale;
            }
            for (std::size_t index = 0;
                 index < correlation.dimension();
                 ++index) {
                out.values[row * correlation.dimension() + index] *= scale;
            }
        }
    }
    correlation.sample_normal_inplace(
        factors.data(), out.values.data(), rows, n_threads);

    const int workers = sampling_worker_count(n_threads, rows);
    std::vector<SamplingBlockResult> blocks(
        static_cast<std::size_t>(workers));
    scar_internal::parallel_for_blocks(
        0,
        n_rows,
        32,
        workers,
        [&](std::int64_t begin,
            std::int64_t end,
            std::size_t block_index) {
            SamplingBlockResult& block = blocks[block_index];
            block.ran = true;
            for (std::int64_t row_index = begin;
                 row_index < end;
                 ++row_index) {
                const std::size_t row =
                    static_cast<std::size_t>(row_index);
                const double degrees = student ? df_at(df, row) : 0.0;
                for (std::size_t column = 0;
                     column < correlation.dimension();
                     ++column) {
                    double& value = out.values[
                        row * correlation.dimension() + column];
                    value = student
                        ? scar_internal::student_cdf_refined_value(value, degrees)
                        : math::normal_cdf(value);
                    if (!std::isfinite(value)) {
                        block.failure_index = row_index;
                        return;
                    }
                }
            }
        });
    finish_parallel_result(out, blocks, correlation.dimension());
    return out;
}

ConditionalSampleResult gaussian_conditional_uniforms(
    DoubleView correlations,
    std::int64_t correlation_rows,
    int dimension,
    const std::vector<int>& given_indices,
    DoubleView given_uniforms,
    DoubleView normal_draws,
    std::int64_t n_rows,
    int n_threads) {

    ConditionalSampleResult invalid;
    invalid.n_rows = n_rows;
    invalid.n_free = dimension
        - static_cast<std::int64_t>(given_indices.size());
    invalid.n_threads_requested = n_threads;
    std::size_t rows = 0;
    std::size_t unused = 0;
    std::vector<int> free_indices;
    if (dimension < 2
        || !checked_shape(n_rows, 1, rows, unused)
        || !scar_internal::valid_thread_count(n_threads)
        || !validate_given_indices(
            static_cast<std::size_t>(dimension),
            given_indices,
            free_indices)
        || !valid_given_uniforms(
            given_uniforms, rows, given_indices.size())) {
        invalid.status = Status::InvalidParameter;
        return invalid;
    }
    if (rows == 0) {
        invalid.values.clear();
        return invalid;
    }

    std::vector<double> latent(given_uniforms.size(), 0.0);
    for (std::size_t index = 0; index < given_uniforms.size(); ++index) {
        latent[index] = math::normal_quantile_refined(
            given_uniforms[index]);
    }
    ConditionalSampleResult out = multivariate_gaussian_conditional(
        correlations,
        correlation_rows,
        dimension,
        given_indices,
        {latent.data(), latent.size()},
        normal_draws,
        n_rows,
        n_threads);
    if (!out.is_ok()) {
        return out;
    }
    for (double& value : out.values) {
        value = conditional_probability(math::normal_cdf(value));
    }
    return out;
}

ConditionalSampleResult student_conditional_uniforms(
    DoubleView correlations,
    std::int64_t correlation_rows,
    int dimension,
    const std::vector<int>& given_indices,
    DoubleView given_uniforms,
    DoubleView df,
    DoubleView normal_draws,
    DoubleView chi_square_draws,
    std::int64_t n_rows,
    int n_threads) {

    ConditionalSampleResult invalid;
    invalid.n_rows = n_rows;
    invalid.n_free = dimension
        - static_cast<std::int64_t>(given_indices.size());
    invalid.n_threads_requested = n_threads;
    std::size_t rows = 0;
    std::size_t unused = 0;
    std::vector<int> free_indices;
    if (dimension < 2
        || !checked_shape(n_rows, 1, rows, unused)
        || !scar_internal::valid_thread_count(n_threads)
        || !validate_given_indices(
            static_cast<std::size_t>(dimension),
            given_indices,
            free_indices)
        || !valid_given_uniforms(
            given_uniforms, rows, given_indices.size())
        || !valid_degrees_of_freedom(df, rows)) {
        invalid.status = Status::InvalidParameter;
        return invalid;
    }
    if (rows == 0) {
        invalid.values.clear();
        return invalid;
    }

    const std::size_t n_given = given_indices.size();
    std::vector<double> degrees(rows, 0.0);
    std::vector<double> latent(rows * n_given, 0.0);
    for (std::size_t row = 0; row < rows; ++row) {
        degrees[row] = df_at(df, row);
        for (std::size_t column = 0; column < n_given; ++column) {
            latent[row * n_given + column] =
                scar_internal::student_quantile_value(
                    given_at(
                        given_uniforms,
                        n_given,
                        row,
                        column),
                    degrees[row]);
        }
    }
    ConditionalSampleResult out = multivariate_student_conditional(
        correlations,
        correlation_rows,
        dimension,
        given_indices,
        {latent.data(), latent.size()},
        {degrees.data(), degrees.size()},
        normal_draws,
        chi_square_draws,
        n_rows,
        n_threads);
    if (!out.is_ok()) {
        return out;
    }
    const std::size_t n_free =
        static_cast<std::size_t>(out.n_free);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < n_free; ++column) {
            double& value = out.values[row * n_free + column];
            value = conditional_probability(
                scar_internal::student_cdf_refined_value(value, degrees[row]));
        }
    }
    return out;
}

bool prepare_factor_conditioning(
    const FactorCorrelationOperator& correlation,
    const std::vector<int>& given_indices,
    std::vector<int>& free_indices,
    std::vector<double>& lower) {

    const std::size_t dimension = correlation.dimension();
    const std::size_t rank = correlation.rank();
    if (!validate_given_indices(
            dimension, given_indices, free_indices)) {
        return false;
    }
    std::vector<double> precision(rank * rank, 0.0);
    for (std::size_t diagonal = 0; diagonal < rank; ++diagonal) {
        precision[diagonal * rank + diagonal] = 1.0;
    }
    for (int raw_index : given_indices) {
        const std::size_t index = static_cast<std::size_t>(raw_index);
        const double* loading =
            correlation.loadings().data() + index * rank;
        const double inverse_uniqueness =
            correlation.inverse_uniqueness()[index];
        for (std::size_t left = 0; left < rank; ++left) {
            for (std::size_t right = 0; right < rank; ++right) {
                precision[left * rank + right] +=
                    loading[left] * loading[right]
                    * inverse_uniqueness;
            }
        }
    }
    return scar_internal::linalg::cholesky_symmetric(
        precision.data(), rank, lower);
}

void solve_cholesky_inplace(
    const std::vector<double>& lower,
    std::size_t dimension,
    std::vector<double>& values) {

    for (std::size_t row = 0; row < dimension; ++row) {
        double value = values[row];
        for (std::size_t column = 0; column < row; ++column) {
            value -= lower[row * dimension + column] * values[column];
        }
        values[row] = value / lower[row * dimension + row];
    }
    for (std::size_t row = dimension; row-- > 0;) {
        double value = values[row];
        for (std::size_t column = row + 1;
             column < dimension;
             ++column) {
            value -= lower[column * dimension + row] * values[column];
        }
        values[row] = value / lower[row * dimension + row];
    }
}

void solve_upper_from_lower_inplace(
    const std::vector<double>& lower,
    std::size_t dimension,
    std::vector<double>& values) {

    for (std::size_t row = dimension; row-- > 0;) {
        double value = values[row];
        for (std::size_t column = row + 1;
             column < dimension;
             ++column) {
            value -= lower[column * dimension + row] * values[column];
        }
        values[row] = value / lower[row * dimension + row];
    }
}

ConditionalSampleResult factor_conditional(
    const FactorCorrelationOperator& correlation,
    const std::vector<int>& given_indices,
    DoubleView given_uniforms,
    DoubleView df,
    DoubleView factor_draws,
    DoubleView residual_draws,
    DoubleView chi_square_draws,
    std::int64_t n_rows,
    int n_threads,
    bool student) {

    ConditionalSampleResult out;
    out.n_rows = n_rows;
    out.n_threads_requested = n_threads;
    std::size_t rows = 0;
    std::size_t factor_values = 0;
    std::size_t residual_values = 0;
    std::vector<int> free_indices;
    std::vector<double> lower;
    if (!checked_shape(
            n_rows, correlation.rank(), rows, factor_values)
        || !core::checked_size_mul(
            rows, correlation.dimension(), residual_values)
        || !scar_internal::valid_thread_count(n_threads)
        || !prepare_factor_conditioning(
            correlation, given_indices, free_indices, lower)) {
        out.status = Status::InvalidParameter;
        return out;
    }
    const std::size_t n_given = given_indices.size();
    const std::size_t n_free = free_indices.size();
    out.n_free = static_cast<std::int64_t>(n_free);
    if (!valid_given_uniforms(given_uniforms, rows, n_given)
        || factor_draws.size() != factor_values
        || residual_draws.size() != residual_values
        || !all_finite(factor_draws)
        || !all_finite(residual_draws)) {
        out.status = Status::InvalidParameter;
        return out;
    }
    if (student
        && (!valid_degrees_of_freedom(df, rows)
            || chi_square_draws.size() != rows
            || !all_finite(chi_square_draws))) {
        out.status = Status::InvalidParameter;
        return out;
    }
    for (std::size_t index = 0;
         index < chi_square_draws.size();
         ++index) {
        const double draw = chi_square_draws[index];
        if (!(draw > 0.0)) {
            out.status = Status::InvalidParameter;
            return out;
        }
    }
    if (rows == 0) {
        return out;
    }

    out.correlation_factorizations = 1;
    out.values.assign(rows * n_free, 0.0);
    const std::size_t rank = correlation.rank();
    const int workers = sampling_worker_count(n_threads, rows);
    std::vector<SamplingBlockResult> blocks(
        static_cast<std::size_t>(workers));
    scar_internal::parallel_for_blocks(
        0,
        n_rows,
        32,
        workers,
        [&](std::int64_t begin,
            std::int64_t end,
            std::size_t block_index) {
            SamplingBlockResult& block = blocks[block_index];
            block.ran = true;
            std::vector<double> latent_given(n_given, 0.0);
            std::vector<double> projected(rank, 0.0);
            std::vector<double> mean(rank, 0.0);
            std::vector<double> factor_innovation(rank, 0.0);
            std::vector<double> factor_value(rank, 0.0);
            for (std::int64_t row_index = begin;
                 row_index < end;
                 ++row_index) {
                const std::size_t row =
                    static_cast<std::size_t>(row_index);
                const double degrees = student ? df_at(df, row) : 0.0;
                std::fill(projected.begin(), projected.end(), 0.0);
                double diagonal_quadratic = 0.0;
                for (std::size_t i = 0; i < n_given; ++i) {
                    const double probability = given_at(
                        given_uniforms,
                        n_given,
                        row,
                        i);
                    const double latent = student
                        ? scar_internal::student_quantile_value(
                            probability, degrees)
                        : math::normal_quantile_refined(probability);
                    latent_given[i] = latent;
                    const std::size_t index = static_cast<std::size_t>(
                        given_indices[i]);
                    const double inverse_uniqueness =
                        correlation.inverse_uniqueness()[index];
                    diagonal_quadratic +=
                        latent * latent * inverse_uniqueness;
                    const double* loading =
                        correlation.loadings().data() + index * rank;
                    for (std::size_t factor = 0;
                         factor < rank;
                         ++factor) {
                        projected[factor] +=
                            latent * inverse_uniqueness * loading[factor];
                    }
                }
                mean = projected;
                solve_cholesky_inplace(lower, rank, mean);
                for (std::size_t factor = 0;
                     factor < rank;
                     ++factor) {
                    factor_innovation[factor] = factor_draws[
                        row * rank + factor];
                }
                solve_upper_from_lower_inplace(
                    lower, rank, factor_innovation);

                double radial = 1.0;
                if (student) {
                    double correction = 0.0;
                    for (std::size_t factor = 0;
                         factor < rank;
                         ++factor) {
                        correction += projected[factor] * mean[factor];
                    }
                    const double delta = std::max(
                        diagonal_quadratic - correction, 0.0);
                    radial = std::sqrt(
                        (degrees + delta) / chi_square_draws[row]);
                }
                for (std::size_t factor = 0;
                     factor < rank;
                     ++factor) {
                    factor_value[factor] = mean[factor]
                        + radial * factor_innovation[factor];
                }
                for (std::size_t free = 0; free < n_free; ++free) {
                    const std::size_t column =
                        static_cast<std::size_t>(free_indices[free]);
                    const double* loading =
                        correlation.loadings().data() + column * rank;
                    double latent = radial
                        * std::sqrt(correlation.uniqueness()[column])
                        * residual_draws[
                            row * correlation.dimension() + column];
                    for (std::size_t factor = 0;
                         factor < rank;
                         ++factor) {
                        latent += loading[factor] * factor_value[factor];
                    }
                    const double probability = student
                        ? scar_internal::student_cdf_refined_value(latent, degrees)
                        : math::normal_cdf(latent);
                    if (!std::isfinite(probability)) {
                        block.failure_index = row_index;
                        return;
                    }
                    out.values[row * n_free + free] =
                        conditional_probability(probability);
                }
            }
        });
    finish_parallel_result(out, blocks, n_free);
    return out;
}

double equicorrelation_at(DoubleView rho, std::size_t row) {
    return rho[rho.size() == 1 ? 0 : row];
}

bool valid_equicorrelation_path(
    DoubleView rho,
    std::size_t rows,
    std::size_t dimension) {

    if ((rho.size() != 1 && rho.size() != rows) || dimension < 2) {
        return false;
    }
    const double lower = -1.0 / static_cast<double>(dimension - 1);
    for (std::size_t index = 0; index < rho.size(); ++index) {
        const double value = rho[index];
        if (!std::isfinite(value) || !(value > lower) || !(value < 1.0)) {
            return false;
        }
    }
    return true;
}

ConditionalSampleResult equicorrelation_sample(
    DoubleView rho,
    int dimension,
    DoubleView normal_draws,
    DoubleView common_draws,
    std::int64_t n_rows,
    int n_threads) {

    ConditionalSampleResult out;
    out.n_rows = n_rows;
    out.n_free = dimension;
    out.n_threads_requested = n_threads;
    std::size_t rows = 0;
    std::size_t values = 0;
    if (dimension < 2
        || !checked_shape(
            n_rows, static_cast<std::size_t>(dimension), rows, values)
        || !scar_internal::valid_thread_count(n_threads)
        || normal_draws.size() != values
        || !valid_equicorrelation_path(
            rho, rows, static_cast<std::size_t>(dimension))
        || !all_finite(normal_draws)) {
        out.status = Status::InvalidParameter;
        return out;
    }

    bool common_factor = true;
    for (std::size_t index = 0; index < rho.size(); ++index) {
        common_factor = common_factor && rho[index] >= 0.0;
    }
    if (common_draws.size() != (common_factor ? rows : 0)
        || !all_finite(common_draws)) {
        out.status = Status::InvalidSize;
        return out;
    }
    out.values.assign(values, 0.0);
    if (rows == 0) {
        return out;
    }

    const std::size_t width = static_cast<std::size_t>(dimension);
    const int workers = sampling_worker_count(n_threads, rows);
    std::vector<SamplingBlockResult> blocks(
        static_cast<std::size_t>(workers));
    scar_internal::parallel_for_blocks(
        0,
        n_rows,
        32,
        workers,
        [&](std::int64_t begin,
            std::int64_t end,
            std::size_t block_index) {
            SamplingBlockResult& block = blocks[block_index];
            block.ran = true;
            for (std::int64_t row_index = begin;
                 row_index < end;
                 ++row_index) {
                const std::size_t row =
                    static_cast<std::size_t>(row_index);
                const double parameter = equicorrelation_at(rho, row);
                const double orthogonal_scale = std::sqrt(1.0 - parameter);
                double row_mean = 0.0;
                double parallel_scale = 0.0;
                if (common_factor) {
                    parallel_scale = std::sqrt(parameter) * common_draws[row];
                } else {
                    for (std::size_t column = 0; column < width; ++column) {
                        row_mean += normal_draws[row * width + column];
                    }
                    row_mean /= static_cast<double>(width);
                    parallel_scale = std::sqrt(
                        1.0 + static_cast<double>(width - 1) * parameter);
                }
                for (std::size_t column = 0; column < width; ++column) {
                    const double draw = normal_draws[row * width + column];
                    const double latent = common_factor
                        ? orthogonal_scale * draw + parallel_scale
                        : orthogonal_scale * (draw - row_mean)
                            + parallel_scale * row_mean;
                    const double probability = math::normal_cdf(latent);
                    if (!std::isfinite(probability)) {
                        block.failure_index = row_index;
                        return;
                    }
                    out.values[row * width + column] = probability;
                }
            }
        });
    finish_parallel_result(out, blocks, width);
    return out;
}

ConditionalSampleResult equicorrelation_conditional_uniforms(
    DoubleView rho,
    int dimension,
    const std::vector<int>& given_indices,
    DoubleView given_uniforms,
    DoubleView normal_draws,
    std::int64_t n_rows,
    int n_threads) {

    ConditionalSampleResult out;
    out.n_rows = n_rows;
    out.n_free = dimension
        - static_cast<std::int64_t>(given_indices.size());
    out.n_threads_requested = n_threads;
    std::size_t rows = 0;
    std::size_t unused = 0;
    std::vector<int> free_indices;
    if (dimension < 2
        || !checked_shape(n_rows, 1, rows, unused)
        || !scar_internal::valid_thread_count(n_threads)
        || !validate_given_indices(
            static_cast<std::size_t>(dimension),
            given_indices,
            free_indices)
        || !valid_given_uniforms(
            given_uniforms, rows, given_indices.size())
        || !valid_equicorrelation_path(
            rho, rows, static_cast<std::size_t>(dimension))) {
        out.status = Status::InvalidParameter;
        return out;
    }
    const std::size_t n_given = given_indices.size();
    const std::size_t n_free = free_indices.size();
    std::size_t draw_values = 0;
    if (!core::checked_size_mul(rows, n_free, draw_values)
        || normal_draws.size() != draw_values
        || !all_finite(normal_draws)) {
        out.status = Status::InvalidSize;
        return out;
    }
    out.values.assign(draw_values, 0.0);
    if (rows == 0) {
        return out;
    }

    std::vector<double> given_sums(
        given_uniforms.size() == n_given ? 1 : rows, 0.0);
    for (std::size_t row = 0; row < given_sums.size(); ++row) {
        for (std::size_t column = 0; column < n_given; ++column) {
            given_sums[row] += math::normal_quantile_refined(
                scar_internal::clip_pseudo_observation(
                    given_at(given_uniforms, n_given, row, column)));
        }
    }

    const int workers = sampling_worker_count(n_threads, rows);
    std::vector<SamplingBlockResult> blocks(
        static_cast<std::size_t>(workers));
    scar_internal::parallel_for_blocks(
        0,
        n_rows,
        32,
        workers,
        [&](std::int64_t begin,
            std::int64_t end,
            std::size_t block_index) {
            SamplingBlockResult& block = blocks[block_index];
            block.ran = true;
            for (std::int64_t row_index = begin;
                 row_index < end;
                 ++row_index) {
                const std::size_t row =
                    static_cast<std::size_t>(row_index);
                const double parameter = equicorrelation_at(rho, row);
                const double denominator = 1.0
                    + static_cast<double>(n_given - 1) * parameter;
                const double conditional_mean = parameter / denominator
                    * given_sums[given_sums.size() == 1 ? 0 : row];
                const double orthogonal_variance = 1.0 - parameter;
                const double parallel_variance = orthogonal_variance
                    * (1.0
                        + static_cast<double>(dimension - 1) * parameter)
                    / denominator;
                if (!(parallel_variance > 0.0)
                    || !std::isfinite(parallel_variance)) {
                    block.failure_index = row_index;
                    return;
                }
                double row_mean = 0.0;
                for (std::size_t column = 0; column < n_free; ++column) {
                    row_mean += normal_draws[row * n_free + column];
                }
                row_mean /= static_cast<double>(n_free);
                const double orthogonal_scale =
                    std::sqrt(orthogonal_variance);
                const double parallel_scale = std::sqrt(parallel_variance);
                for (std::size_t column = 0; column < n_free; ++column) {
                    const double draw = normal_draws[row * n_free + column];
                    const double latent = orthogonal_scale * (draw - row_mean)
                        + parallel_scale * row_mean
                        + conditional_mean;
                    const double probability = conditional_probability(
                        math::normal_cdf(latent));
                    if (!std::isfinite(probability)) {
                        block.failure_index = row_index;
                        return;
                    }
                    out.values[row * n_free + column] = probability;
                }
            }
        });
    finish_parallel_result(out, blocks, n_free);
    return out;
}

Status chi_square_draws_from_uniforms(
    DoubleView df,
    std::size_t rows,
    std::size_t degrees_offset,
    DoubleView uniforms,
    std::vector<double>& draws) {

    if (!valid_degrees_of_freedom(df, rows) || uniforms.size() != rows) {
        return Status::InvalidSize;
    }
    draws.resize(rows);
    for (std::size_t row = 0; row < rows; ++row) {
        const double uniform = uniforms[row];
        if (!std::isfinite(uniform) || uniform < 0.0 || uniform >= 1.0) {
            draws.clear();
            return Status::InvalidParameter;
        }
        const double probability = std::max(
            uniform, std::numeric_limits<double>::min());
        double draw = math::chi_square_quantile(
            probability,
            df_at(df, row) + static_cast<double>(degrees_offset));
        if (draw == 0.0 && probability > 0.0) {
            draw = std::numeric_limits<double>::min();
        }
        if (!std::isfinite(draw) || !(draw > 0.0)) {
            draws.clear();
            return Status::NumericalFailure;
        }
        draws[row] = draw;
    }
    return Status::Ok;
}

}  // namespace

Result<std::int64_t> equicorr_gaussian_common_draw_count(
    DoubleView rho,
    int dimension,
    std::int64_t n_rows) {

    Result<std::int64_t> result;
    std::size_t rows = 0;
    std::size_t unused = 0;
    if (dimension < 2 || !checked_shape(n_rows, 1, rows, unused)) {
        result.status = Status::InvalidSize;
        return result;
    }
    if (!valid_equicorrelation_path(
            rho, rows, static_cast<std::size_t>(dimension))) {
        result.status = Status::InvalidParameter;
        return result;
    }
    for (std::size_t index = 0; index < rho.size(); ++index) {
        if (rho[index] < 0.0) {
            return result;
        }
    }
    result.value = static_cast<std::int64_t>(rows);
    return result;
}

Status validate_equicorrelation_path(
    DoubleView rho,
    int dimension,
    std::int64_t n_rows) noexcept {

    std::size_t rows = 0;
    std::size_t unused = 0;
    if (dimension < 2 || !checked_shape(n_rows, 1, rows, unused)) {
        return Status::InvalidSize;
    }
    return valid_equicorrelation_path(
        rho, rows, static_cast<std::size_t>(dimension))
        ? Status::Ok
        : Status::InvalidParameter;
}

ConditionalSampleResult multivariate_gaussian_sample_dense(
    DoubleView correlation,
    int dimension,
    DoubleView normal_draws,
    std::int64_t n_rows,
    int n_threads) {

    return dense_sample(
        correlation,
        dimension,
        {},
        normal_draws,
        {},
        n_rows,
        n_threads,
        false);
}

ConditionalSampleResult multivariate_gaussian_sample_equicorrelation(
    DoubleView rho,
    int dimension,
    DoubleView normal_draws,
    DoubleView common_draws,
    std::int64_t n_rows,
    int n_threads) {

    return equicorrelation_sample(
        rho,
        dimension,
        normal_draws,
        common_draws,
        n_rows,
        n_threads);
}

ConditionalSampleResult multivariate_student_sample_dense(
    DoubleView correlation,
    int dimension,
    DoubleView df,
    DoubleView normal_draws,
    DoubleView chi_square_draws,
    std::int64_t n_rows,
    int n_threads) {

    return dense_sample(
        correlation,
        dimension,
        df,
        normal_draws,
        chi_square_draws,
        n_rows,
        n_threads,
        true);
}

ConditionalSampleResult multivariate_student_sample_dense_from_uniforms(
    DoubleView correlation,
    int dimension,
    DoubleView df,
    DoubleView normal_draws,
    DoubleView chi_square_uniforms,
    std::int64_t n_rows,
    int n_threads) {

    ConditionalSampleResult invalid;
    invalid.n_rows = n_rows;
    invalid.n_free = dimension;
    invalid.n_threads_requested = n_threads;
    std::size_t rows = 0;
    std::size_t unused = 0;
    if (!checked_shape(n_rows, 1, rows, unused)) {
        invalid.status = Status::InvalidSize;
        return invalid;
    }
    std::vector<double> chi_square;
    const Status status = chi_square_draws_from_uniforms(
        df, rows, 0, chi_square_uniforms, chi_square);
    if (!ok(status)) {
        invalid.status = status;
        return invalid;
    }
    return multivariate_student_sample_dense(
        correlation, dimension, df, normal_draws,
        {chi_square.data(), chi_square.size()}, n_rows, n_threads);
}

ConditionalSampleResult multivariate_gaussian_sample_factor(
    const FactorCorrelationOperator& correlation,
    DoubleView factor_draws,
    DoubleView residual_draws,
    std::int64_t n_rows,
    int n_threads) {

    return factor_sample(
        correlation,
        {},
        factor_draws,
        residual_draws,
        {},
        n_rows,
        n_threads,
        false);
}

ConditionalSampleResult multivariate_student_sample_factor(
    const FactorCorrelationOperator& correlation,
    DoubleView df,
    DoubleView factor_draws,
    DoubleView residual_draws,
    DoubleView chi_square_draws,
    std::int64_t n_rows,
    int n_threads) {

    return factor_sample(
        correlation,
        df,
        factor_draws,
        residual_draws,
        chi_square_draws,
        n_rows,
        n_threads,
        true);
}

ConditionalSampleResult multivariate_student_sample_factor_from_uniforms(
    const FactorCorrelationOperator& correlation,
    DoubleView df,
    DoubleView factor_draws,
    DoubleView residual_draws,
    DoubleView chi_square_uniforms,
    std::int64_t n_rows,
    int n_threads) {

    ConditionalSampleResult invalid;
    invalid.n_rows = n_rows;
    invalid.n_free = static_cast<std::int64_t>(correlation.dimension());
    invalid.n_threads_requested = n_threads;
    std::size_t rows = 0;
    std::size_t unused = 0;
    if (!checked_shape(n_rows, 1, rows, unused)) {
        invalid.status = Status::InvalidSize;
        return invalid;
    }
    std::vector<double> chi_square;
    const Status status = chi_square_draws_from_uniforms(
        df, rows, 0, chi_square_uniforms, chi_square);
    if (!ok(status)) {
        invalid.status = status;
        return invalid;
    }
    return multivariate_student_sample_factor(
        correlation, df, factor_draws, residual_draws,
        {chi_square.data(), chi_square.size()}, n_rows, n_threads);
}

ConditionalSampleResult multivariate_gaussian_conditional_from_uniforms(
    DoubleView correlations,
    std::int64_t correlation_rows,
    int dimension,
    const std::vector<int>& given_indices,
    DoubleView given_uniforms,
    DoubleView normal_draws,
    std::int64_t n_rows,
    int n_threads) {

    return gaussian_conditional_uniforms(
        correlations,
        correlation_rows,
        dimension,
        given_indices,
        given_uniforms,
        normal_draws,
        n_rows,
        n_threads);
}

ConditionalSampleResult
multivariate_gaussian_conditional_equicorrelation_from_uniforms(
    DoubleView rho,
    int dimension,
    const std::vector<int>& given_indices,
    DoubleView given_uniforms,
    DoubleView normal_draws,
    std::int64_t n_rows,
    int n_threads) {

    return equicorrelation_conditional_uniforms(
        rho,
        dimension,
        given_indices,
        given_uniforms,
        normal_draws,
        n_rows,
        n_threads);
}

ConditionalSampleResult multivariate_student_conditional_from_uniforms(
    DoubleView correlations,
    std::int64_t correlation_rows,
    int dimension,
    const std::vector<int>& given_indices,
    DoubleView given_uniforms,
    DoubleView df,
    DoubleView normal_draws,
    DoubleView chi_square_draws,
    std::int64_t n_rows,
    int n_threads) {

    return student_conditional_uniforms(
        correlations,
        correlation_rows,
        dimension,
        given_indices,
        given_uniforms,
        df,
        normal_draws,
        chi_square_draws,
        n_rows,
        n_threads);
}

ConditionalSampleResult
multivariate_student_conditional_from_normal_uniforms(
    DoubleView correlations,
    std::int64_t correlation_rows,
    int dimension,
    const std::vector<int>& given_indices,
    DoubleView given_uniforms,
    DoubleView df,
    DoubleView normal_draws,
    DoubleView chi_square_uniforms,
    std::int64_t n_rows,
    int n_threads) {

    ConditionalSampleResult invalid;
    invalid.n_rows = n_rows;
    invalid.n_free = dimension
        - static_cast<std::int64_t>(given_indices.size());
    invalid.n_threads_requested = n_threads;
    std::size_t rows = 0;
    std::size_t unused = 0;
    if (!checked_shape(n_rows, 1, rows, unused)) {
        invalid.status = Status::InvalidSize;
        return invalid;
    }
    std::vector<double> chi_square;
    const Status status = chi_square_draws_from_uniforms(
        df, rows, given_indices.size(), chi_square_uniforms, chi_square);
    if (!ok(status)) {
        invalid.status = status;
        return invalid;
    }
    return multivariate_student_conditional_from_uniforms(
        correlations, correlation_rows, dimension, given_indices,
        given_uniforms, df, normal_draws,
        {chi_square.data(), chi_square.size()}, n_rows, n_threads);
}

ConditionalSampleResult multivariate_gaussian_conditional_factor(
    const FactorCorrelationOperator& correlation,
    const std::vector<int>& given_indices,
    DoubleView given_uniforms,
    DoubleView factor_draws,
    DoubleView residual_draws,
    std::int64_t n_rows,
    int n_threads) {

    return factor_conditional(
        correlation,
        given_indices,
        given_uniforms,
        {},
        factor_draws,
        residual_draws,
        {},
        n_rows,
        n_threads,
        false);
}

ConditionalSampleResult multivariate_student_conditional_factor(
    const FactorCorrelationOperator& correlation,
    const std::vector<int>& given_indices,
    DoubleView given_uniforms,
    DoubleView df,
    DoubleView factor_draws,
    DoubleView residual_draws,
    DoubleView chi_square_draws,
    std::int64_t n_rows,
    int n_threads) {

    return factor_conditional(
        correlation,
        given_indices,
        given_uniforms,
        df,
        factor_draws,
        residual_draws,
        chi_square_draws,
        n_rows,
        n_threads,
        true);
}

ConditionalSampleResult
multivariate_student_conditional_factor_from_normal_uniforms(
    const FactorCorrelationOperator& correlation,
    const std::vector<int>& given_indices,
    DoubleView given_uniforms,
    DoubleView df,
    DoubleView factor_draws,
    DoubleView residual_draws,
    DoubleView chi_square_uniforms,
    std::int64_t n_rows,
    int n_threads) {

    ConditionalSampleResult invalid;
    invalid.n_rows = n_rows;
    invalid.n_free = static_cast<std::int64_t>(correlation.dimension())
        - static_cast<std::int64_t>(given_indices.size());
    invalid.n_threads_requested = n_threads;
    std::size_t rows = 0;
    std::size_t unused = 0;
    if (!checked_shape(n_rows, 1, rows, unused)) {
        invalid.status = Status::InvalidSize;
        return invalid;
    }
    std::vector<double> chi_square;
    const Status status = chi_square_draws_from_uniforms(
        df, rows, given_indices.size(), chi_square_uniforms, chi_square);
    if (!ok(status)) {
        invalid.status = status;
        return invalid;
    }
    return multivariate_student_conditional_factor(
        correlation, given_indices, given_uniforms, df,
        factor_draws, residual_draws,
        {chi_square.data(), chi_square.size()}, n_rows, n_threads);
}

}  // namespace scar
