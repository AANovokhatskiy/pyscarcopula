#include "scar/ou.hpp"

#include "evaluator_internal.hpp"
#include "scar/detail/copula.hpp"
#include "scar/detail/safety.hpp"
#include "scar/detail/scar_ou/grid.hpp"
#include "scar/detail/scar_ou/transition.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace scar {
using namespace evaluator_detail;

namespace {

std::size_t rosenblatt_output_size(
    ObservationView u,
    const CopulaSpec& copula) {

    std::size_t output_size = 0;
    if (copula.dim < 2
        || u.dim != copula.dim
        || !scar_internal::checked_size_mul(
            u.size(),
            static_cast<std::size_t>(copula.dim),
            output_size)) {
        return 0;
    }
    return output_size;
}

bool valid_dense_student_spec(const CopulaSpec& copula) {
    std::size_t matrix_size = 0;
    return copula.family == CopulaFamily::Student
        && copula.correlation_kind == CorrelationKind::DenseCholesky
        && scar_internal::checked_size_mul(
            static_cast<std::size_t>(copula.dim),
            static_cast<std::size_t>(copula.dim),
            matrix_size)
        && copula.l_inv.size() == matrix_size
        && std::isfinite(copula.log_det);
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

    const std::size_t output_size = rosenblatt_output_size(u, copula);
    if (output_size == 0
        || u.size() < 2
        || u.data() == nullptr
        || !valid_dense_student_spec(copula)) {
        return false;
    }

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

    std::vector<double> prefix_log_determinant(dimension, 0.0);
    for (std::size_t prefix = 1; prefix < dimension; ++prefix) {
        const double diagonal =
            copula.l_inv[(prefix - 1) * dimension + prefix - 1];
        if (!std::isfinite(diagonal) || diagonal <= 0.0) {
            return false;
        }
        prefix_log_determinant[prefix] =
            prefix_log_determinant[prefix - 1]
            - 2.0 * std::log(diagonal);
    }

    std::vector<double> quantiles(quantile_size, 0.0);
    std::vector<double> conditional_cdf(K, 0.0);
    std::vector<double> log_weights(K, 0.0);
    std::vector<double> source(K, 0.0);
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
        for (std::size_t column = 1; column < dimension; ++column) {
            double maximum_log_weight =
                -std::numeric_limits<double>::infinity();
            for (std::size_t state = 0; state < K; ++state) {
                const std::size_t state_offset = state * dimension;
                const double df = df_grid[state];

                double conditional_mean = 0.0;
                const double diagonal =
                    copula.l_inv[column * dimension + column];
                if (!std::isfinite(diagonal) || diagonal <= 0.0) {
                    valid = false;
                    return;
                }
                for (std::size_t prefix = 0;
                     prefix < column;
                     ++prefix) {
                    conditional_mean -=
                        copula.l_inv[column * dimension + prefix]
                        * quantiles[state_offset + prefix]
                        / diagonal;
                }

                double quadratic_form = 0.0;
                for (std::size_t factor_row = 0;
                     factor_row < column;
                     ++factor_row) {
                    double whitened = 0.0;
                    for (std::size_t prefix = 0;
                         prefix <= factor_row;
                         ++prefix) {
                        whitened +=
                            copula.l_inv[
                                factor_row * dimension + prefix]
                            * quantiles[state_offset + prefix];
                    }
                    quadratic_form += whitened * whitened;
                }

                const double scale = std::max(
                    (df + quadratic_form)
                        / (df + static_cast<double>(column)),
                    1e-12);
                const double standardized =
                    (quantiles[state_offset + column] - conditional_mean)
                    * diagonal / std::sqrt(scale);
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
    return std::vector<double>(
        rosenblatt_output_size(u, copula), 0.0);
}

std::vector<double> student_rosenblatt_backend(
    const OuParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const OuNumericalConfig& config,
    OuBackend backend,
    int& status) {

    status = SCAR_OK;
    if (!valid_dense_student_spec(copula)) {
        return invalid_student_rosenblatt(
            u, copula, SCAR_INVALID_TRANSFORM, status);
    }
    if (!valid_ou_params(params) || !finite_config_doubles(config)) {
        return invalid_student_rosenblatt(
            u, copula, SCAR_INVALID_PARAMETER, status);
    }
    if (u.size() < 2 || rosenblatt_output_size(u, copula) == 0) {
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
    if (!valid_dense_student_spec(copula)) {
        return invalid_student_rosenblatt(
            u, copula, SCAR_INVALID_TRANSFORM, status);
    }
    if (!valid_ou_params(params) || !finite_config_doubles(config)) {
        return invalid_student_rosenblatt(
            u, copula, SCAR_INVALID_PARAMETER, status);
    }

    scar_internal::OuGrid grid;
    if (u.size() < 2
        || rosenblatt_output_size(u, copula) == 0
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
