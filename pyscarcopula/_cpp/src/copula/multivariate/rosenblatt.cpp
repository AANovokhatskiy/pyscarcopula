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
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace scar {
namespace {

constexpr double kCorrelationTolerance = 1e-12;

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
    if (!scar_internal::valid_thread_count(n_threads)) {
        out.status = Status::InvalidParameter;
        return out;
    }
    if (!validate_observations(u, correlation.dimension(), values)) {
        out.status = Status::InvalidSize;
        return out;
    }
    if (student && df.size() != 1 && df.size() != u.n_obs) {
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
    out.residuals.assign(values, 0.0);
    if (rows == 0) {
        return out;
    }

    std::vector<double> covariance(rank * rank, 0.0);
    for (std::size_t diagonal = 0; diagonal < rank; ++diagonal) {
        covariance[diagonal * rank + diagonal] = 1.0;
    }
    std::vector<double> state(rows * rank, 0.0);
    std::vector<double> diagonal_quadratic(
        student ? rows : 0, 0.0);
    const int workers = rosenblatt_workers(
        n_threads, rows, dimension);
    out.parallel_blocks = workers;

    for (std::size_t coordinate = 0;
         coordinate < dimension;
         ++coordinate) {
        const double* loading =
            correlation.loadings().data() + coordinate * rank;
        std::vector<double> covariance_loading(rank, 0.0);
        scar_internal::linalg::row_major_matvec(
            covariance.data(),
            rank,
            rank,
            loading,
            covariance_loading.data());
        double conditional_variance =
            correlation.uniqueness()[coordinate];
        conditional_variance += scar_internal::linalg::dot(
            loading, covariance_loading.data(), rank);
        if (!std::isfinite(conditional_variance)
            || !(conditional_variance > 0.0)) {
            out.status = Status::NumericalFailure;
            out.failure.coordinate = static_cast<int>(coordinate);
            return out;
        }

        std::vector<std::int64_t> failures(
            static_cast<std::size_t>(workers), -1);
        scar_internal::parallel_for_blocks(
            0,
            static_cast<std::int64_t>(rows),
            16,
            workers,
            [&](std::int64_t begin,
                std::int64_t end,
                std::size_t block) {
                std::vector<double> solved_projection(rank, 0.0);
                for (std::int64_t row_index = begin;
                     row_index < end;
                     ++row_index) {
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
                    double* row_state = state.data() + row * rank;
                    double residual = 0.0;

                    if (student) {
                        if (coordinate == 0) {
                            residual = probability;
                        } else {
                            scar_internal::linalg::row_major_matvec(
                                covariance.data(),
                                rank,
                                rank,
                                row_state,
                                solved_projection.data());
                            const double conditional_mean =
                                scar_internal::linalg::dot(
                                    solved_projection.data(),
                                    loading,
                                    rank);
                            const double correction =
                                scar_internal::linalg::dot(
                                    row_state,
                                    solved_projection.data(),
                                    rank);
                            const double quadratic = std::max(
                                diagonal_quadratic[row] - correction,
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
                        diagonal_quadratic[row] +=
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
                        failures[block] = row_index;
                        return;
                    }
                    out.residuals[row * dimension + coordinate] =
                        scar_internal::clip_pseudo_observation(residual);
                }
            });

        for (std::int64_t failure : failures) {
            if (failure >= 0
                && (out.failure.index < 0
                    || failure < out.failure.index)) {
                out.failure.index = failure;
                out.failure.coordinate = static_cast<int>(coordinate);
            }
        }
        if (out.failure.index >= 0) {
            out.status = Status::NumericalFailure;
            return out;
        }

        for (std::size_t left = 0; left < rank; ++left) {
            for (std::size_t right = 0; right < rank; ++right) {
                covariance[left * rank + right] -=
                    covariance_loading[left]
                    * covariance_loading[right]
                    / conditional_variance;
            }
        }
    }
    return out;
}

}  // namespace

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
