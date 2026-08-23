#include "scar/copula/multivariate/student/rosenblatt.hpp"

#include "scar/copula/multivariate/student/distribution.hpp"
#include "scar/copula/multivariate/student/quantile.hpp"
#include "scar/detail/linalg.hpp"
#include "scar/detail/parallel.hpp"
#include "scar/detail/safety.hpp"
#include "scar/status.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace scar {
namespace {

constexpr std::size_t kParallelMinRows = 16;
constexpr std::size_t kParallelMinCells = 2048;
constexpr double kCorrelationTolerance = 1e-12;

struct RosenblattBlockResult {
    bool ran = false;
    std::int64_t failure_row = -1;
    int failure_coordinate = -1;
};

double dense_student_quantile(double probability, double df) {
    const double p = scar_internal::clip_pseudo_observation(probability);
    const double initial = scar_internal::student_quantile_value(p, df);
    const double initial_cdf =
        scar_internal::student_cdf_value(initial, df);
    const double tail = p < 0.5 ? p : 1.0 - p;
    if (std::isfinite(initial)
        && std::isfinite(initial_cdf)
        && std::abs(initial_cdf - p) <= 2e-13 * tail) {
        return initial;
    }

    const bool negative = p < 0.5;
    double low = 0.0;
    double high = std::max(1.0, std::abs(initial));
    const double maximum = std::numeric_limits<double>::max() / 4.0;
    while (scar_internal::student_cdf_value(-high, df) > tail
           && high < maximum) {
        high = std::min(maximum, high * 2.0);
    }
    if (scar_internal::student_cdf_value(-high, df) > tail) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    for (int iteration = 0; iteration < 256; ++iteration) {
        const double middle = low + 0.5 * (high - low);
        const double survival =
            scar_internal::student_cdf_value(-middle, df);
        if (std::abs(survival - tail) <= 2e-13 * tail
            || high - low <= 2e-13 * std::max(1.0, middle)) {
            return negative ? -middle : middle;
        }
        if (survival > tail) {
            low = middle;
        } else {
            high = middle;
        }
    }
    const double result = low + 0.5 * (high - low);
    return negative ? -result : result;
}

bool prepare_correlation(
    DoubleView correlation,
    std::size_t dimension,
    std::vector<double>& lower,
    int& failure_coordinate,
    int& failure_status) {

    std::size_t square = 0;
    if (!scar_internal::checked_size_mul(
            dimension, dimension, square)
        || correlation.size() != square
        || correlation.data() == nullptr) {
        failure_status = SCAR_INVALID_SIZE;
        return false;
    }

    for (std::size_t row = 0; row < dimension; ++row) {
        const double diagonal = correlation[row * dimension + row];
        if (!std::isfinite(diagonal)
            || std::abs(diagonal - 1.0) > kCorrelationTolerance) {
            failure_coordinate = static_cast<int>(row);
            failure_status = SCAR_INVALID_PARAMETER;
            return false;
        }
        for (std::size_t column = 0; column < row; ++column) {
            const double lower_value =
                correlation[row * dimension + column];
            const double upper_value =
                correlation[column * dimension + row];
            const double scale = std::max(
                {1.0, std::abs(lower_value), std::abs(upper_value)});
            if (!std::isfinite(lower_value)
                || !std::isfinite(upper_value)
                || std::abs(lower_value - upper_value)
                    > kCorrelationTolerance * scale) {
                failure_coordinate = static_cast<int>(row);
                failure_status = SCAR_INVALID_PARAMETER;
                return false;
            }
        }
    }

    std::size_t cholesky_failure = dimension;
    if (!scar_internal::linalg::cholesky_symmetric(
            correlation.data(),
            dimension,
            lower,
            0.0,
            &cholesky_failure)) {
        if (cholesky_failure < dimension) {
            failure_coordinate = static_cast<int>(cholesky_failure);
        }
        failure_status = SCAR_NUMERICAL_FAILURE;
        return false;
    }
    return true;
}

}  // namespace

DenseStudentRosenblattResult student_rosenblatt_dense(
    DoubleView correlation,
    int dimension,
    ObservationView u,
    DoubleView df,
    int n_threads) {

    DenseStudentRosenblattResult out;
    out.n_threads_requested = n_threads;
    out.dimension = dimension;
    if (dimension <= 0
        || u.dim != dimension
        || u.n_obs > static_cast<std::size_t>(
            std::numeric_limits<std::int64_t>::max())) {
        out.status = SCAR_INVALID_SIZE;
        return out;
    }
    out.n_rows = static_cast<std::int64_t>(u.n_obs);

    const std::size_t width = static_cast<std::size_t>(dimension);
    std::size_t output_size = 0;
    if (!scar_internal::checked_size_mul(u.n_obs, width, output_size)
        || (output_size != 0 && u.data() == nullptr)
        || (df.size() != 1 && df.size() != u.n_obs)
        || (df.size() != 0 && df.data() == nullptr)
        || (u.n_obs != 0 && df.size() == 0)) {
        out.status = SCAR_INVALID_SIZE;
        return out;
    }
    if (!scar_internal::valid_thread_count(n_threads)) {
        out.status = SCAR_INVALID_PARAMETER;
        return out;
    }

    for (std::size_t row = 0; row < df.size(); ++row) {
        if (!std::isfinite(df[row]) || !(df[row] > 0.0)) {
            out.status = SCAR_INVALID_PARAMETER;
            out.failure_index = df.size() == 1
                ? -1
                : static_cast<std::int64_t>(row);
            return out;
        }
    }

    std::vector<double> lower;
    int correlation_failure = -1;
    int correlation_status = SCAR_OK;
    if (!prepare_correlation(
            correlation,
            width,
            lower,
            correlation_failure,
            correlation_status)) {
        out.status = correlation_status;
        out.failure_coordinate = correlation_failure;
        return out;
    }
    out.correlation_factorizations = 1;
    out.residuals.assign(output_size, 0.0);
    if (u.n_obs == 0) {
        out.status = SCAR_OK;
        return out;
    }

    const bool use_threads = n_threads > 1
        && scar_internal::grid_parallel_worthwhile(
            u.n_obs, width, kParallelMinRows, kParallelMinCells);
    std::vector<RosenblattBlockResult> block_results(
        static_cast<std::size_t>(use_threads ? n_threads : 1));
    scar_internal::parallel_for_blocks(
        0,
        out.n_rows,
        static_cast<std::int64_t>(kParallelMinRows),
        use_threads ? n_threads : 1,
        [&](std::int64_t begin,
            std::int64_t end,
            std::size_t block) {
            RosenblattBlockResult& block_result = block_results[block];
            block_result.ran = true;
            std::vector<double> quantiles(width, 0.0);
            std::vector<double> whitened(width, 0.0);

            for (std::int64_t row_index = begin;
                 row_index < end;
                 ++row_index) {
                const std::size_t row =
                    static_cast<std::size_t>(row_index);
                const std::size_t offset = row * width;
                const double row_df = df[df.size() == 1 ? 0 : row];
                double quadratic_form = 0.0;

                for (std::size_t column = 0;
                     column < width;
                     ++column) {
                    const double probability = u.data()[offset + column];
                    if (std::isnan(probability)) {
                        block_result.failure_row = row_index;
                        block_result.failure_coordinate =
                            static_cast<int>(column);
                        break;
                    }
                    quantiles[column] = dense_student_quantile(
                        probability, row_df);

                    const double conditional_mean =
                        scar_internal::linalg::dot(
                            lower.data() + column * width,
                            whitened.data(),
                            column);
                    const double diagonal =
                        lower[column * width + column];
                    whitened[column] =
                        (quantiles[column] - conditional_mean) / diagonal;

                    double residual = 0.0;
                    if (column == 0) {
                        residual = scar_internal::student_cdf_value(
                            quantiles[column], row_df);
                    } else {
                        const double conditional_df =
                            row_df + static_cast<double>(column);
                        const double scale =
                            (row_df + quadratic_form) / conditional_df;
                        residual = scar_internal::student_cdf_value(
                            whitened[column] / std::sqrt(scale),
                            conditional_df);
                    }
                    if (!std::isfinite(quantiles[column])
                        || !std::isfinite(whitened[column])
                        || !std::isfinite(residual)) {
                        block_result.failure_row = row_index;
                        block_result.failure_coordinate =
                            static_cast<int>(column);
                        break;
                    }
                    out.residuals[offset + column] =
                        scar_internal::clip_pseudo_observation(residual);
                    quadratic_form +=
                        whitened[column] * whitened[column];
                }
                if (block_result.failure_row >= 0) {
                    break;
                }
            }
        });

    for (const RosenblattBlockResult& block : block_results) {
        if (!block.ran) {
            continue;
        }
        ++out.parallel_blocks;
        if (block.failure_row >= 0
            && (out.failure_index < 0
                || block.failure_row < out.failure_index)) {
            out.failure_index = block.failure_row;
            out.failure_coordinate = block.failure_coordinate;
        }
    }
    out.status = out.failure_index >= 0
        ? SCAR_NUMERICAL_FAILURE
        : SCAR_OK;
    return out;
}

}  // namespace scar
