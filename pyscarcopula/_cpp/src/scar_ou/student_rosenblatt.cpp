#include "scar/ou.hpp"

#include "evaluator_internal.hpp"
#include "scar/detail/copula/common.hpp"
#include "scar/copula/multivariate/student/distribution.hpp"
#include "scar/copula/multivariate/student/quantile.hpp"
#include "scar/detail/linalg.hpp"
#include "scar/detail/safety.hpp"
#include "scar/detail/scar_ou/grid.hpp"
#include "scar/detail/scar_ou/transition.hpp"
#include "scar/factor.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace scar {
using namespace evaluator_detail;

namespace {

bool valid_student_spec(const CopulaSpec& copula) {
    std::size_t matrix_size = 0;
    if (copula.family != CopulaFamily::Student || copula.dim < 2) {
        return false;
    }
    if (copula.correlation_kind == CorrelationKind::Factor) {
        return copula.factor_operator() != nullptr
            && copula.factor_operator()->dimension()
                == static_cast<std::size_t>(copula.dim)
            && std::isfinite(copula.factor_operator()->logdet());
    }
    return copula.correlation_kind == CorrelationKind::DenseCholesky
        && scar_internal::checked_size_mul(
            static_cast<std::size_t>(copula.dim),
            static_cast<std::size_t>(copula.dim),
            matrix_size)
        && copula.dense_inverse_cholesky().size() == matrix_size
        && std::isfinite(copula.dense_log_determinant());
}

double multivariate_student_log_pdf(
    std::size_t dimension,
    double df,
    double log_determinant,
    double quadratic_form) {

    const double dimension_value = static_cast<double>(dimension);
    return std::lgamma(0.5 * (df + dimension_value))
        - std::lgamma(0.5 * df)
        - 0.5 * dimension_value * std::log(df * 3.14159265358979323846)
        - 0.5 * log_determinant
        - 0.5 * (df + dimension_value)
            * std::log1p(quadratic_form / df);
}

double univariate_student_log_pdf(double value, double df) {
    return std::lgamma(0.5 * (df + 1.0))
        - std::lgamma(0.5 * df)
        - 0.5 * std::log(df * 3.14159265358979323846)
        - 0.5 * (df + 1.0) * std::log1p(value * value / df);
}

bool student_rosenblatt_impl(
    const CopulaSpec& copula,
    ObservationView u,
    const scar_internal::OuGrid& grid,
    const scar_internal::GridTransitionOperator& transition,
    std::vector<double>& out) {

    const Result<std::size_t> output_shape = rosenblatt_output_size(
        u, copula.model_descriptor().expected_dimension());
    if (!output_shape.is_ok()
        || output_shape.value == 0
        || u.size() < 2
        || u.data() == nullptr
        || !valid_student_spec(copula)) {
        return false;
    }
    const std::size_t output_size = output_shape.value;

    const std::size_t dimension = static_cast<std::size_t>(copula.dim);
    const std::size_t K = static_cast<std::size_t>(grid.K);
    std::size_t quantile_size = 0;
    if (!scar_internal::checked_size_mul(K, dimension, quantile_size)) {
        return false;
    }
    out.assign(output_size, 0.0);

    std::vector<double> df_grid(K, 0.0);
    for (std::size_t state = 0; state < K; ++state) {
        df_grid[state] = scar_internal::copula_transform(
            copula, grid.x_grid[state]);
        if (!std::isfinite(df_grid[state]) || df_grid[state] <= 2.0) {
            return false;
        }
    }

    const bool factor_correlation =
        copula.correlation_kind == CorrelationKind::Factor;
    std::vector<double> prefix_log_determinant(dimension, 0.0);
    std::vector<double> prefix_core_inverse;
    std::vector<double> factor_conditional_variance;
    const FactorCorrelationOperator* factor = nullptr;
    std::size_t factor_rank = 0;
    if (factor_correlation) {
        factor = copula.factor_operator().get();
        factor_rank = factor->rank();
        std::size_t prefix_inverse_size = 0;
        if (!scar_internal::checked_size_mul(
                dimension, factor_rank, prefix_inverse_size)
            || !scar_internal::checked_size_mul(
                prefix_inverse_size,
                factor_rank,
                prefix_inverse_size)) {
            return false;
        }
        prefix_core_inverse.assign(prefix_inverse_size, 0.0);
        factor_conditional_variance.assign(dimension, 0.0);
        std::vector<double> core(factor_rank * factor_rank, 0.0);
        std::vector<double> identity(factor_rank * factor_rank, 0.0);
        for (std::size_t diagonal = 0;
             diagonal < factor_rank;
             ++diagonal) {
            core[diagonal * factor_rank + diagonal] = 1.0;
            identity[diagonal * factor_rank + diagonal] = 1.0;
        }
        double diagonal_log_determinant = 0.0;
        const auto& loadings = factor->loadings();
        const auto& uniqueness = factor->uniqueness();
        for (std::size_t column = 1;
             column < dimension;
             ++column) {
            const std::size_t previous = column - 1;
            const double previous_uniqueness = uniqueness[previous];
            diagonal_log_determinant += std::log(previous_uniqueness);
            for (std::size_t left = 0; left < factor_rank; ++left) {
                for (std::size_t right = 0;
                     right < factor_rank;
                     ++right) {
                    core[left * factor_rank + right] +=
                        loadings[previous * factor_rank + left]
                        * loadings[previous * factor_rank + right]
                        / previous_uniqueness;
                }
            }
            std::vector<double> lower;
            double applied_jitter = 0.0;
            if (!scar_internal::linalg::cholesky_symmetric_with_jitter(
                    core.data(),
                    factor_rank,
                    lower,
                    &applied_jitter)
                || applied_jitter != 0.0) {
                return false;
            }
            std::vector<double> inverse;
            if (!scar_internal::linalg::solve_spd(
                    lower.data(),
                    factor_rank,
                    identity.data(),
                    factor_rank,
                    inverse)) {
                return false;
            }
            std::copy(
                inverse.begin(),
                inverse.end(),
                prefix_core_inverse.begin()
                    + column * factor_rank * factor_rank);
            double core_log_determinant = 0.0;
            for (std::size_t diagonal = 0;
                 diagonal < factor_rank;
                 ++diagonal) {
                core_log_determinant += 2.0 * std::log(
                    lower[diagonal * factor_rank + diagonal]);
            }
            prefix_log_determinant[column] =
                diagonal_log_determinant + core_log_determinant;

            const double* loading =
                loadings.data() + column * factor_rank;
            double loading_quadratic = 0.0;
            for (std::size_t left = 0; left < factor_rank; ++left) {
                for (std::size_t right = 0;
                     right < factor_rank;
                     ++right) {
                    loading_quadratic +=
                        loading[left]
                        * inverse[left * factor_rank + right]
                        * loading[right];
                }
            }
            factor_conditional_variance[column] =
                uniqueness[column] + loading_quadratic;
            if (!std::isfinite(factor_conditional_variance[column])
                || factor_conditional_variance[column] <= 0.0) {
                return false;
            }
        }
    } else {
        for (std::size_t prefix = 1; prefix < dimension; ++prefix) {
            const double diagonal =
                copula.dense_inverse_cholesky()[(prefix - 1) * dimension + prefix - 1];
            if (!std::isfinite(diagonal) || diagonal <= 0.0) {
                return false;
            }
            prefix_log_determinant[prefix] =
                prefix_log_determinant[prefix - 1]
                - 2.0 * std::log(diagonal);
        }
    }

    std::vector<double> quantiles(quantile_size, 0.0);
    std::vector<double> conditional_cdf(K, 0.0);
    std::vector<double> log_weights(K, 0.0);
    std::vector<double> source(K, 0.0);
    std::vector<double> factor_projection(K * factor_rank, 0.0);
    std::vector<double> factor_diagonal_quadratic(K, 0.0);
    std::vector<double> factor_solved(factor_rank, 0.0);
    bool valid = true;

    auto advance = [&](
        const std::vector<double>& phi,
        const std::vector<double>& emission,
        std::vector<double>& phi_next) {

        for (std::size_t state = 0; state < K; ++state) {
            const double value =
                phi[state] * emission[state] * grid.trap_w[state];
            if (!std::isfinite(value) || value < 0.0) {
                return false;
            }
            source[state] = value;
        }
        scar_internal::grid_predict_matvec(
            transition, grid, source, phi_next);
        return scar_internal::normalize_density_by_max(phi_next);
    };

    auto on_row = [&](
        std::int64_t t,
        const std::vector<double>& weights,
        const std::vector<double>&) {

        if (!valid) {
            return;
        }
        const std::size_t row =
            static_cast<std::size_t>(t) * dimension;
        for (std::size_t state = 0; state < K; ++state) {
            const std::size_t state_offset = state * dimension;
            for (std::size_t column = 0; column < dimension; ++column) {
                const double probability =
                    scar_internal::clip_pseudo_observation(
                        u.data()[row + column]);
                quantiles[state_offset + column] =
                    scar_internal::student_quantile_for_observation(
                        copula,
                        probability,
                        df_grid[state],
                        t,
                        static_cast<int>(column));
                if (!std::isfinite(
                        quantiles[state_offset + column])) {
                    valid = false;
                    return;
                }
            }
        }

        out[row] = scar_internal::clip_pseudo_observation(u.data()[row]);
        if (factor_correlation) {
            const auto& weighted_loadings = factor->weighted_loadings();
            const auto& uniqueness = factor->uniqueness();
            for (std::size_t state = 0; state < K; ++state) {
                const double first = quantiles[state * dimension];
                factor_diagonal_quadratic[state] =
                    first * first / uniqueness[0];
                for (std::size_t component = 0;
                     component < factor_rank;
                     ++component) {
                    factor_projection[state * factor_rank + component] =
                        first * weighted_loadings[component];
                }
            }
        }
        for (std::size_t column = 1; column < dimension; ++column) {
            double maximum_log_weight =
                -std::numeric_limits<double>::infinity();
            for (std::size_t state = 0; state < K; ++state) {
                const std::size_t state_offset = state * dimension;
                const double df = df_grid[state];

                double conditional_mean = 0.0;
                double quadratic_form = 0.0;
                double conditional_standard_deviation = 0.0;
                if (factor_correlation) {
                    const double* inverse =
                        prefix_core_inverse.data()
                        + column * factor_rank * factor_rank;
                    const double* projection =
                        factor_projection.data()
                        + state * factor_rank;
                    const double* loading =
                        factor->loadings().data()
                        + column * factor_rank;
                    for (std::size_t left = 0;
                         left < factor_rank;
                         ++left) {
                        double value = 0.0;
                        for (std::size_t right = 0;
                             right < factor_rank;
                             ++right) {
                            value +=
                                inverse[left * factor_rank + right]
                                * projection[right];
                        }
                        factor_solved[left] = value;
                        conditional_mean += loading[left] * value;
                        quadratic_form += projection[left] * value;
                    }
                    quadratic_form =
                        factor_diagonal_quadratic[state]
                        - quadratic_form;
                    quadratic_form = std::max(quadratic_form, 0.0);
                    conditional_standard_deviation = std::sqrt(
                        factor_conditional_variance[column]);
                } else {
                    const double diagonal =
                        copula.dense_inverse_cholesky()[column * dimension + column];
                    if (!std::isfinite(diagonal) || diagonal <= 0.0) {
                        valid = false;
                        return;
                    }
                    for (std::size_t prefix = 0;
                         prefix < column;
                         ++prefix) {
                        conditional_mean -=
                            copula.dense_inverse_cholesky()[column * dimension + prefix]
                            * quantiles[state_offset + prefix]
                            / diagonal;
                    }
                    for (std::size_t factor_row = 0;
                         factor_row < column;
                         ++factor_row) {
                        double whitened = 0.0;
                        for (std::size_t prefix = 0;
                             prefix <= factor_row;
                             ++prefix) {
                            whitened +=
                                copula.dense_inverse_cholesky()[
                                    factor_row * dimension + prefix]
                                * quantiles[state_offset + prefix];
                        }
                        quadratic_form += whitened * whitened;
                    }
                    conditional_standard_deviation = 1.0 / diagonal;
                }

                const double scale = std::max(
                    (df + quadratic_form)
                        / (df + static_cast<double>(column)),
                    1e-12);
                const double standardized =
                    (quantiles[state_offset + column] - conditional_mean)
                    / (
                        conditional_standard_deviation
                        * std::sqrt(scale));
                conditional_cdf[state] =
                    scar_internal::student_cdf_value(
                        standardized,
                        df + static_cast<double>(column));
                if (!std::isfinite(conditional_cdf[state])) {
                    valid = false;
                    return;
                }

                double prefix_log_density = 0.0;
                if (column > 1) {
                    prefix_log_density = multivariate_student_log_pdf(
                        column,
                        df,
                        prefix_log_determinant[column],
                        quadratic_form);
                    for (std::size_t prefix = 0;
                         prefix < column;
                         ++prefix) {
                        prefix_log_density -=
                            univariate_student_log_pdf(
                                quantiles[state_offset + prefix], df);
                    }
                }
                log_weights[state] = weights[state] > 0.0
                    ? std::log(weights[state]) + prefix_log_density
                    : -std::numeric_limits<double>::infinity();
                maximum_log_weight =
                    std::max(maximum_log_weight, log_weights[state]);
            }

            double mixture = 0.0;
            double total = 0.0;
            if (std::isfinite(maximum_log_weight)) {
                for (std::size_t state = 0; state < K; ++state) {
                    const double value =
                        std::exp(log_weights[state] - maximum_log_weight);
                    total += value;
                    mixture += value * conditional_cdf[state];
                }
            }
            if (std::isfinite(total) && total > 0.0) {
                mixture /= total;
            } else {
                mixture = 0.0;
                for (std::size_t state = 0; state < K; ++state) {
                    mixture += weights[state] * conditional_cdf[state];
                }
            }
            if (!std::isfinite(mixture)) {
                valid = false;
                return;
            }
            out[row + column] =
                scar_internal::clip_pseudo_observation(mixture);
            if (factor_correlation) {
                const auto& weighted_loadings =
                    factor->weighted_loadings();
                const auto& uniqueness = factor->uniqueness();
                const double* weighted =
                    weighted_loadings.data() + column * factor_rank;
                for (std::size_t state = 0; state < K; ++state) {
                    const double value =
                        quantiles[state * dimension + column];
                    factor_diagonal_quadratic[state] +=
                        value * value / uniqueness[column];
                    for (std::size_t component = 0;
                         component < factor_rank;
                         ++component) {
                        factor_projection[
                            state * factor_rank + component]
                            += value * weighted[component];
                    }
                }
            }
        }
    };

    const bool filtered = scar_internal::forward_filter_grid(
        copula,
        grid,
        u.data(),
        static_cast<std::int64_t>(u.size()),
        advance,
        on_row);
    if (!filtered || !valid) {
        out.assign(output_size, 0.0);
        return false;
    }
    return true;
}

std::vector<double> invalid_student_rosenblatt(
    ObservationView u,
    const CopulaSpec& copula,
    int error,
    int& status) {

    status = error;
    const Result<std::size_t> output_shape = rosenblatt_output_size(
        u, copula.model_descriptor().expected_dimension());
    return std::vector<double>(
        output_shape.is_ok() ? output_shape.value : 0, 0.0);
}

std::vector<double> student_rosenblatt_backend(
    const OuParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const OuNumericalConfig& config,
    OuBackend backend,
    int& status) {

    status = SCAR_OK;
    if (!valid_student_spec(copula)) {
        return invalid_student_rosenblatt(
            u, copula, SCAR_INVALID_TRANSFORM, status);
    }
    if (!valid_ou_params(params) || !finite_config_doubles(config)) {
        return invalid_student_rosenblatt(
            u, copula, SCAR_INVALID_PARAMETER, status);
    }
    const Result<std::size_t> output_shape = rosenblatt_output_size(
        u, copula.model_descriptor().expected_dimension());
    if (u.size() < 2
        || !output_shape.is_ok()
        || output_shape.value == 0) {
        return invalid_student_rosenblatt(
            u, copula, SCAR_INVALID_SIZE, status);
    }

    scar_internal::OuGrid grid;
    scar_internal::GridTransitionOperator transition;
    if (!valid_grid_config(config, backend)
        || !scar_internal::build_ou_grid(
            params.kappa,
            params.mu,
            params.nu,
            static_cast<std::int64_t>(u.size()),
            config.K,
            config.grid_range,
            config.adaptive,
            config.pts_per_sigma,
            config.max_K,
            grid)
        || !scar_internal::build_grid_transition_operator(
            grid,
            backend,
            config.grid_method,
            config.gh_order,
            transition)) {
        return invalid_student_rosenblatt(
            u, copula, SCAR_INVALID_SIZE, status);
    }

    std::vector<double> out;
    if (!student_rosenblatt_impl(copula, u, grid, transition, out)) {
        return invalid_student_rosenblatt(
            u, copula, SCAR_NUMERICAL_FAILURE, status);
    }
    return out;
}

}  // namespace

std::vector<double> ScarOuEvaluator::student_rosenblatt_matrix(
    const OuParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const OuNumericalConfig& config,
    int& status) const {

    return student_rosenblatt_backend(
        params, copula, u, config, OuBackend::Matrix, status);
}

std::vector<double> ScarOuEvaluator::student_rosenblatt_local_gh(
    const OuParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const OuNumericalConfig& config,
    int& status) const {

    return student_rosenblatt_backend(
        params, copula, u, config, OuBackend::LocalGh, status);
}

std::vector<double> ScarOuEvaluator::student_rosenblatt_auto(
    const OuParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const OuNumericalConfig& config,
    OuBackend& backend,
    int& status) const {

    status = SCAR_OK;
    if (!valid_student_spec(copula)) {
        return invalid_student_rosenblatt(
            u, copula, SCAR_INVALID_TRANSFORM, status);
    }
    if (!valid_ou_params(params) || !finite_config_doubles(config)) {
        return invalid_student_rosenblatt(
            u, copula, SCAR_INVALID_PARAMETER, status);
    }

    scar_internal::OuGrid grid;
    const Result<std::size_t> output_shape = rosenblatt_output_size(
        u, copula.model_descriptor().expected_dimension());
    if (u.size() < 2
        || !output_shape.is_ok()
        || output_shape.value == 0
        || !scar_internal::build_ou_grid(
            params.kappa,
            params.mu,
            params.nu,
            static_cast<std::int64_t>(u.size()),
            config.K,
            config.grid_range,
            config.adaptive,
            config.pts_per_sigma,
            config.max_K,
            grid)) {
        return invalid_student_rosenblatt(
            u, copula, SCAR_INVALID_SIZE, status);
    }

    const int selected = scar_internal::select_grid_transition_backend(
        grid, config.r_gh);
    backend = selected == 0 ? OuBackend::Matrix : OuBackend::LocalGh;
    if (backend == OuBackend::Matrix) {
        return student_rosenblatt_matrix(
            params, copula, u, config, status);
    }
    return student_rosenblatt_local_gh(
        params, copula, u, config, status);
}

}  // namespace scar
