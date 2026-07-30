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

double normal_cdf(double value) {
    return 0.5 * std::erfc(-value / std::sqrt(2.0));
}

bool equicorr_gaussian_rosenblatt_impl(
    const CopulaSpec& copula,
    ObservationView u,
    const scar_internal::OuGrid& grid,
    const scar_internal::GridTransitionOperator& transition,
    std::vector<double>& out) {

    const std::size_t output_size = rosenblatt_output_size(u, copula);
    if (output_size == 0 || u.size() < 2 || u.data() == nullptr) {
        return false;
    }

    const std::size_t dimension = static_cast<std::size_t>(copula.dim);
    const std::size_t K = static_cast<std::size_t>(grid.K);
    out.assign(output_size, 0.0);

    std::vector<double> rho_grid(K, 0.0);
    for (std::size_t state = 0; state < K; ++state) {
        rho_grid[state] = scar_internal::equicorr_transform(
            copula, grid.x_grid[state]);
        if (!std::isfinite(rho_grid[state])) {
            return false;
        }
    }

    std::vector<double> quantiles(dimension, 0.0);
    std::vector<double> conditional_cdf(K, 0.0);
    std::vector<double> reweighted(K, 0.0);
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
        for (std::size_t column = 0; column < dimension; ++column) {
            quantiles[column] = scar_internal::normal_quantile_refined(
                scar_internal::clip_pseudo_observation(
                    u.data()[row + column]));
            if (!std::isfinite(quantiles[column])) {
                valid = false;
                return;
            }
        }

        out[row] = scar_internal::clip_pseudo_observation(u.data()[row]);
        double prefix_sum = quantiles[0];
        double prefix_sum_squares = quantiles[0] * quantiles[0];

        for (std::size_t column = 1; column < dimension; ++column) {
            const double prefix_dimension = static_cast<double>(column);
            double total = 0.0;
            for (std::size_t state = 0; state < K; ++state) {
                const double rho = rho_grid[state];
                const double denominator =
                    1.0 + (prefix_dimension - 1.0) * rho;
                const double one_minus_rho = 1.0 - rho;
                if (denominator <= 0.0 || one_minus_rho <= 0.0) {
                    valid = false;
                    return;
                }

                const double conditional_mean =
                    rho * prefix_sum / denominator;
                const double conditional_variance = std::max(
                    1.0
                        - prefix_dimension * rho * rho / denominator,
                    1e-10);
                const double standardized =
                    (quantiles[column] - conditional_mean)
                    / std::sqrt(conditional_variance);
                conditional_cdf[state] = normal_cdf(standardized);

                double prefix_density = 1.0;
                if (column > 1) {
                    const double log_determinant =
                        (prefix_dimension - 1.0)
                            * std::log(one_minus_rho)
                        + std::log(denominator);
                    const double square_sum =
                        prefix_sum * prefix_sum;
                    const double log_density =
                        -0.5 * log_determinant
                        -0.5 * (
                            (rho / one_minus_rho)
                                * prefix_sum_squares
                            - (rho
                               / (one_minus_rho * denominator))
                                * square_sum);
                    prefix_density = std::exp(log_density);
                }
                const double value = weights[state] * prefix_density;
                if (!std::isfinite(conditional_cdf[state])
                    || !std::isfinite(value)
                    || value < 0.0) {
                    valid = false;
                    return;
                }
                reweighted[state] = value;
                total += value;
            }

            double mixture = 0.0;
            const bool use_reweighted =
                std::isfinite(total) && total > 0.0;
            for (std::size_t state = 0; state < K; ++state) {
                const double state_weight = use_reweighted
                    ? reweighted[state] / total
                    : weights[state];
                mixture += state_weight * conditional_cdf[state];
            }
            if (!std::isfinite(mixture)) {
                valid = false;
                return;
            }
            out[row + column] =
                scar_internal::clip_pseudo_observation(mixture);
            prefix_sum += quantiles[column];
            prefix_sum_squares +=
                quantiles[column] * quantiles[column];
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

std::vector<double> invalid_gaussian_rosenblatt(
    ObservationView u,
    const CopulaSpec& copula,
    int error,
    int& status) {

    status = error;
    return std::vector<double>(
        rosenblatt_output_size(u, copula), 0.0);
}

}  // namespace

std::vector<double> ScarOuEvaluator::gaussian_rosenblatt_matrix(
    const OuParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const OuNumericalConfig& config,
    int& status) const {

    status = SCAR_OK;
    if (copula.family != CopulaFamily::EquicorrGaussian) {
        return invalid_gaussian_rosenblatt(
            u, copula, SCAR_INVALID_TRANSFORM, status);
    }
    if (!valid_ou_params(params) || !finite_config_doubles(config)) {
        return invalid_gaussian_rosenblatt(
            u, copula, SCAR_INVALID_PARAMETER, status);
    }
    if (u.size() < 2 || rosenblatt_output_size(u, copula) == 0) {
        return invalid_gaussian_rosenblatt(
            u, copula, SCAR_INVALID_SIZE, status);
    }

    scar_internal::OuGrid grid;
    scar_internal::GridTransitionOperator transition;
    if (!valid_grid_config(config, OuBackend::Matrix)
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
            OuBackend::Matrix,
            config.grid_method,
            config.gh_order,
            transition)) {
        return invalid_gaussian_rosenblatt(
            u, copula, SCAR_INVALID_SIZE, status);
    }

    std::vector<double> out;
    if (!equicorr_gaussian_rosenblatt_impl(
            copula, u, grid, transition, out)) {
        return invalid_gaussian_rosenblatt(
            u, copula, SCAR_NUMERICAL_FAILURE, status);
    }
    return out;
}

std::vector<double> ScarOuEvaluator::gaussian_rosenblatt_local_gh(
    const OuParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const OuNumericalConfig& config,
    int& status) const {

    status = SCAR_OK;
    if (copula.family != CopulaFamily::EquicorrGaussian) {
        return invalid_gaussian_rosenblatt(
            u, copula, SCAR_INVALID_TRANSFORM, status);
    }
    if (!valid_ou_params(params) || !finite_config_doubles(config)) {
        return invalid_gaussian_rosenblatt(
            u, copula, SCAR_INVALID_PARAMETER, status);
    }
    if (u.size() < 2 || rosenblatt_output_size(u, copula) == 0) {
        return invalid_gaussian_rosenblatt(
            u, copula, SCAR_INVALID_SIZE, status);
    }

    scar_internal::OuGrid grid;
    scar_internal::GridTransitionOperator transition;
    if (!valid_grid_config(config, OuBackend::LocalGh)
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
            OuBackend::LocalGh,
            config.grid_method,
            config.gh_order,
            transition)) {
        return invalid_gaussian_rosenblatt(
            u, copula, SCAR_INVALID_SIZE, status);
    }

    std::vector<double> out;
    if (!equicorr_gaussian_rosenblatt_impl(
            copula, u, grid, transition, out)) {
        return invalid_gaussian_rosenblatt(
            u, copula, SCAR_NUMERICAL_FAILURE, status);
    }
    return out;
}

std::vector<double> ScarOuEvaluator::gaussian_rosenblatt_auto(
    const OuParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const OuNumericalConfig& config,
    OuBackend& backend,
    int& status) const {

    status = SCAR_OK;
    if (copula.family != CopulaFamily::EquicorrGaussian) {
        return invalid_gaussian_rosenblatt(
            u, copula, SCAR_INVALID_TRANSFORM, status);
    }
    if (!valid_ou_params(params) || !finite_config_doubles(config)) {
        return invalid_gaussian_rosenblatt(
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
        return invalid_gaussian_rosenblatt(
            u, copula, SCAR_INVALID_SIZE, status);
    }

    const int selected = scar_internal::select_grid_transition_backend(
        grid, config.r_gh);
    backend = selected == 0 ? OuBackend::Matrix : OuBackend::LocalGh;
    if (backend == OuBackend::Matrix) {
        return gaussian_rosenblatt_matrix(
            params, copula, u, config, status);
    }
    return gaussian_rosenblatt_local_gh(
        params, copula, u, config, status);
}

}  // namespace scar
