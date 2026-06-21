#include "scar/ou.hpp"

#include "evaluator_internal.hpp"
#include "scar/detail/copula.hpp"
#include "scar/detail/safety.hpp"
#include "scar/detail/scar_ou/grid.hpp"
#include "scar/detail/scar_ou/quadrature.hpp"
#include "scar/detail/scar_ou/transition.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace scar {
using namespace evaluator_detail;

LogLikResult ScarOuEvaluator::loglik_spectral(
    const OuParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const OuNumericalConfig& raw_config) const {

    const OuNumericalConfig config = with_default_quad_order(raw_config);
    const std::int64_t n_obs = static_cast<std::int64_t>(u.size());
    if (!supported_ou_copula(copula)) {
        return invalid_loglik(SCAR_INVALID_TRANSFORM, OuBackend::Spectral);
    }
    if (!valid_ou_params(params) || !finite_config_doubles(config)) {
        return invalid_loglik(SCAR_INVALID_PARAMETER, OuBackend::Spectral);
    }
    std::size_t spectral_elements = 0;
    if (n_obs <= 0
        || !scar_internal::valid_spectral_dimensions(
            config.spectral_quad_order,
            config.spectral_basis_order,
            spectral_elements)) {
        return invalid_loglik(SCAR_INVALID_SIZE, OuBackend::Spectral);
    }

    const double sigma = params.nu / std::sqrt(2.0 * params.kappa);
    if (!std::isfinite(sigma) || sigma <= 0.0) {
        return invalid_loglik(SCAR_NUMERICAL_FAILURE, OuBackend::Spectral);
    }

    std::vector<double> z;
    std::vector<double> weights;
    std::vector<double> basis;
    std::vector<double> weighted_basis;
    if (!scar_internal::standard_normal_hermite_rule_with_weighted_basis(
            config.spectral_quad_order,
            config.spectral_basis_order,
            z,
            weights,
            basis,
            weighted_basis)) {
        return invalid_loglik(SCAR_NUMERICAL_FAILURE, OuBackend::Spectral);
    }

    const double* observation_values = observation_data(copula, u);
    const double dt = n_obs > 1 ? 1.0 / static_cast<double>(n_obs - 1) : 1.0;
    const double rho = std::exp(-params.kappa * dt);

    std::vector<double> powers(
        static_cast<std::size_t>(config.spectral_basis_order), 1.0);
    for (int n = 1; n < config.spectral_basis_order; ++n) {
        powers[static_cast<std::size_t>(n)] =
            powers[static_cast<std::size_t>(n - 1)] * rho;
    }

    std::vector<double> x_grid(static_cast<std::size_t>(config.spectral_quad_order));
    for (int q = 0; q < config.spectral_quad_order; ++q) {
        x_grid[static_cast<std::size_t>(q)] =
            params.mu + sigma * z[static_cast<std::size_t>(q)];
    }
    std::vector<double> r_grid;
    std::vector<double> dpsi_grid;
    scar_internal::copula_prepare_grid_transform(
        copula, x_grid, r_grid, dpsi_grid);

    std::vector<double> coeff(
        static_cast<std::size_t>(config.spectral_basis_order), 0.0);
    std::vector<double> projected(
        static_cast<std::size_t>(config.spectral_basis_order), 0.0);
    std::vector<double> fi_row(
        static_cast<std::size_t>(config.spectral_quad_order), 0.0);
    coeff[0] = 1.0;
    double log_scale = 0.0;

    for (std::int64_t t = n_obs - 1; t >= 1; --t) {
        scar_internal::copula_pdf_row_precomputed_flat(
            copula,
            observation_values,
            t,
            r_grid,
            fi_row.data());

        scar_internal::project_multiply(
            coeff,
            fi_row,
            basis,
            weighted_basis,
            config.spectral_quad_order,
            config.spectral_basis_order,
            projected);

        double scale = 0.0;
        for (int n = 0; n < config.spectral_basis_order; ++n) {
            coeff[static_cast<std::size_t>(n)] =
                powers[static_cast<std::size_t>(n)]
                * projected[static_cast<std::size_t>(n)];
            scale = std::max(scale, std::abs(coeff[static_cast<std::size_t>(n)]));
        }
        if (!std::isfinite(scale) || scale <= 0.0) {
            return invalid_loglik(SCAR_NUMERICAL_FAILURE, OuBackend::Spectral);
        }
        for (double& value : coeff) {
            value /= scale;
        }
        log_scale += std::log(scale);
    }

    scar_internal::copula_pdf_row_precomputed_flat(
        copula,
        observation_values,
        0,
        r_grid,
        fi_row.data());
    scar_internal::project_multiply(
        coeff,
        fi_row,
        basis,
        weighted_basis,
        config.spectral_quad_order,
        config.spectral_basis_order,
        projected);

    const double likelihood_scaled = projected[0];
    if (!std::isfinite(likelihood_scaled) || likelihood_scaled <= 0.0) {
        return invalid_loglik(SCAR_NUMERICAL_FAILURE, OuBackend::Spectral);
    }
    return {
        std::log(likelihood_scaled) + log_scale,
        OuBackend::Spectral,
        SCAR_OK,
        -1,
        {},
        SCAR_FALLBACK_NONE,
    };
}

LogLikResult ScarOuEvaluator::loglik_local_gh(
    const OuParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const OuNumericalConfig& config) const {

    const std::int64_t n_obs = static_cast<std::int64_t>(u.size());
    if (!supported_ou_copula(copula)) {
        return invalid_loglik(SCAR_INVALID_TRANSFORM, OuBackend::LocalGh);
    }
    if (!valid_ou_params(params) || !finite_config_doubles(config)) {
        return invalid_loglik(SCAR_INVALID_PARAMETER, OuBackend::LocalGh);
    }
    if (n_obs < 2 || config.K < 2 || config.grid_range <= 0.0
        || config.pts_per_sigma <= 0 || config.gh_order <= 0) {
        return invalid_loglik(SCAR_INVALID_SIZE, OuBackend::LocalGh);
    }
    if (!valid_grid_config(config, OuBackend::LocalGh)
        || static_cast<std::size_t>(config.gh_order)
            > scar_internal::kMaxSpectralOrder) {
        return invalid_loglik(SCAR_INVALID_SIZE, OuBackend::LocalGh);
    }
    if (config.max_K > 0 && config.max_K < 2) {
        return invalid_loglik(SCAR_INVALID_SIZE, OuBackend::LocalGh);
    }
    if (adaptive_grid_exceeds_limit(params, n_obs, config)) {
        return invalid_loglik(SCAR_INVALID_SIZE, OuBackend::LocalGh);
    }
    scar_internal::OuGrid grid;
    if (!scar_internal::build_ou_grid(
            params.kappa,
            params.mu,
            params.nu,
            n_obs,
            config.K,
            config.grid_range,
            config.adaptive,
            config.pts_per_sigma,
            config.max_K,
            grid)) {
        return invalid_loglik(SCAR_NUMERICAL_FAILURE, OuBackend::LocalGh);
    }

    std::vector<double> gh_nodes;
    std::vector<double> gh_weights;
    if (!scar_internal::physicists_hermite_normal_rule(
            config.gh_order, gh_nodes, gh_weights)) {
        return invalid_loglik(SCAR_NUMERICAL_FAILURE, OuBackend::LocalGh);
    }

    const double* observation_values = observation_data(copula, u);
    std::vector<double> r_grid;
    std::vector<double> dpsi_grid;
    scar_internal::copula_prepare_grid_transform(
        copula, grid.x_grid, r_grid, dpsi_grid);
    std::vector<double> msg(static_cast<std::size_t>(grid.K), 1.0);
    std::vector<double> v(static_cast<std::size_t>(grid.K), 0.0);
    std::vector<double> next_msg(static_cast<std::size_t>(grid.K), 0.0);
    std::vector<double> fi_row(static_cast<std::size_t>(grid.K), 0.0);
    double log_scale = 0.0;
    for (std::int64_t t = n_obs - 1; t >= 1; --t) {
        scar_internal::copula_pdf_row_precomputed_flat(
            copula,
            observation_values,
            t,
            r_grid,
            fi_row.data());
        for (int j = 0; j < grid.K; ++j) {
            const std::size_t idx = static_cast<std::size_t>(j);
            v[idx] = fi_row[idx] * msg[idx];
        }

        scar_internal::local_gh_matvec(
            grid.z,
            grid.rho,
            grid.sigma_cond,
            gh_nodes,
            gh_weights,
            v,
            next_msg);

        double scale = 0.0;
        for (double value : next_msg) {
            scale = std::max(scale, std::abs(value));
        }
        if (!std::isfinite(scale) || scale <= 0.0) {
            return invalid_loglik(SCAR_NUMERICAL_FAILURE, OuBackend::LocalGh);
        }
        for (int j = 0; j < grid.K; ++j) {
            msg[static_cast<std::size_t>(j)] =
                next_msg[static_cast<std::size_t>(j)] / scale;
        }
        log_scale += std::log(scale);
    }

    scar_internal::copula_pdf_row_precomputed_flat(
        copula,
        observation_values,
        0,
        r_grid,
        fi_row.data());
    double result = 0.0;
    for (int j = 0; j < grid.K; ++j) {
        const std::size_t idx = static_cast<std::size_t>(j);
        result += fi_row[idx] * grid.p0[idx] * msg[idx] * grid.trap_w[idx];
    }
    if (!std::isfinite(result) || result <= 0.0) {
        return invalid_loglik(SCAR_NUMERICAL_FAILURE, OuBackend::LocalGh);
    }
    return {
        std::log(result) + log_scale,
        OuBackend::LocalGh,
        SCAR_OK,
        -1,
        {},
        SCAR_FALLBACK_NONE,
    };
}

LogLikResult ScarOuEvaluator::loglik_matrix(
    const OuParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const OuNumericalConfig& config) const {

    const std::int64_t n_obs = static_cast<std::int64_t>(u.size());
    if (!supported_ou_copula(copula)) {
        return invalid_loglik(SCAR_INVALID_TRANSFORM, OuBackend::Matrix);
    }
    if (!valid_ou_params(params) || !finite_config_doubles(config)) {
        return invalid_loglik(SCAR_INVALID_PARAMETER, OuBackend::Matrix);
    }
    if (n_obs < 2 || config.K < 2 || config.grid_range <= 0.0
        || config.pts_per_sigma <= 0) {
        return invalid_loglik(SCAR_INVALID_SIZE, OuBackend::Matrix);
    }
    if (!valid_grid_config(config, OuBackend::Matrix)) {
        return invalid_loglik(SCAR_INVALID_SIZE, OuBackend::Matrix);
    }
    if (adaptive_grid_exceeds_limit(params, n_obs, config)) {
        return invalid_loglik(SCAR_INVALID_SIZE, OuBackend::Matrix);
    }

    scar_internal::OuGrid grid;
    if (!scar_internal::build_ou_grid(
            params.kappa,
            params.mu,
            params.nu,
            n_obs,
            config.K,
            config.grid_range,
            config.adaptive,
            config.pts_per_sigma,
            config.max_K,
            grid)) {
        return invalid_loglik(SCAR_NUMERICAL_FAILURE, OuBackend::Matrix);
    }

    std::vector<double> matrix;
    if (!scar_internal::build_dense_transition_matrix(grid, matrix)) {
        return invalid_loglik(SCAR_INVALID_SIZE, OuBackend::Matrix);
    }
    double value = -std::numeric_limits<double>::infinity();
    const double* observation_values = observation_data(copula, u);
    if (!scar_internal::matrix_backward_loglik(
            copula,
            grid,
            matrix,
            observation_values,
            n_obs,
            value)) {
        return invalid_loglik(SCAR_NUMERICAL_FAILURE, OuBackend::Matrix);
    }
    return {
        value,
        OuBackend::Matrix,
        SCAR_OK,
        -1,
        {},
        SCAR_FALLBACK_NONE,
    };
}

}  // namespace scar
