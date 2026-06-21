#include "scar/ou.hpp"

#include "evaluator_internal.hpp"
#include "scar/detail/safety.hpp"
#include "scar/detail/scar_ou/grid.hpp"
#include "scar/detail/scar_ou/quadrature.hpp"
#include "scar/detail/scar_ou/transition.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace scar {
using namespace evaluator_detail;

namespace {

bool normalize_by_max(std::vector<double>& values) {
    double scale = 0.0;
    for (double value : values) {
        if (!std::isfinite(value)) {
            return false;
        }
        scale = std::max(scale, value);
    }
    if (scale <= 0.0) {
        return false;
    }
    const double negative_tolerance = 1e-12 * scale;
    for (double& value : values) {
        if (value < -negative_tolerance) {
            return false;
        }
        value = std::max(value, 0.0) / scale;
        if (!std::isfinite(value)) {
            return false;
        }
    }
    return true;
}

bool normalize_state_prob(
    const scar_internal::OuGrid& grid,
    const std::vector<double>& phi,
    std::vector<double>& prob) {

    if (grid.K <= 0
        || phi.size() != static_cast<std::size_t>(grid.K)
        || grid.trap_w.size() != static_cast<std::size_t>(grid.K)) {
        prob.clear();
        return false;
    }
    double scale = 0.0;
    for (double value : phi) {
        if (!std::isfinite(value)) {
            prob.clear();
            return false;
        }
        scale = std::max(scale, value);
    }
    if (scale <= 0.0) {
        prob.clear();
        return false;
    }
    const double negative_tolerance = 1e-12 * scale;
    prob.assign(static_cast<std::size_t>(grid.K), 0.0);
    double total = 0.0;
    for (int j = 0; j < grid.K; ++j) {
        const std::size_t idx = static_cast<std::size_t>(j);
        if (!std::isfinite(grid.trap_w[idx]) || grid.trap_w[idx] <= 0.0
            || phi[idx] < -negative_tolerance) {
            prob.clear();
            return false;
        }
        prob[idx] = std::max(phi[idx], 0.0) * grid.trap_w[idx];
        total += prob[idx];
    }
    if (!std::isfinite(total) || total <= 0.0) {
        prob.clear();
        return false;
    }
    for (double& value : prob) {
        value /= total;
        if (!std::isfinite(value)) {
            prob.clear();
            return false;
        }
    }
    return true;
}

template <typename AdvanceDensity>
StateDistribution state_distribution_impl(
    const CopulaSpec& copula,
    const scar_internal::OuGrid& grid,
    const double* u,
    std::int64_t n_obs,
    bool horizon_next,
    OuBackend backend,
    AdvanceDensity advance_density) {

    std::vector<double> phi = grid.p0;
    std::vector<double> source(static_cast<std::size_t>(grid.K), 0.0);
    std::vector<double> next_phi(static_cast<std::size_t>(grid.K), 0.0);
    std::vector<double> fi_row(static_cast<std::size_t>(grid.K), 0.0);

    auto advance = [&]() -> bool {
        for (int j = 0; j < grid.K; ++j) {
            const std::size_t idx = static_cast<std::size_t>(j);
            source[idx] = phi[idx] * grid.trap_w[idx];
        }
        advance_density(source, next_phi);
        phi.swap(next_phi);
        return normalize_by_max(phi);
    };

    for (std::int64_t t = 0; t < n_obs; ++t) {
        scar_internal::copula_fi_row_on_grid(
            copula, u, t, grid.x_grid, fi_row);
        for (int j = 0; j < grid.K; ++j) {
            const std::size_t idx = static_cast<std::size_t>(j);
            phi[idx] *= fi_row[idx];
        }
        if (t < n_obs - 1) {
            if (!advance()) {
                return invalid_state_distribution(
                    SCAR_NUMERICAL_FAILURE, backend);
            }
        }
        if (!normalize_by_max(phi)) {
            return invalid_state_distribution(
                SCAR_NUMERICAL_FAILURE, backend);
        }
    }

    if (horizon_next && !advance()) {
        return invalid_state_distribution(
            SCAR_NUMERICAL_FAILURE, backend);
    }

    StateDistribution out;
    out.z_grid = grid.x_grid;
    if (!normalize_state_prob(grid, phi, out.prob)) {
        return invalid_state_distribution(
            SCAR_NUMERICAL_FAILURE, backend);
    }
    out.backend = backend;
    out.status = SCAR_OK;
    return out;
}


}  // namespace

StateDistribution ScarOuEvaluator::state_distribution_matrix(
    const OuParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const OuNumericalConfig& config,
    bool horizon_next) const {

    const std::int64_t n_obs = static_cast<std::int64_t>(u.size());
    if (!supported_ou_copula(copula)) {
        return invalid_state_distribution(SCAR_INVALID_TRANSFORM, OuBackend::Matrix);
    }
    if (!valid_ou_params(params) || !finite_config_doubles(config)) {
        return invalid_state_distribution(SCAR_INVALID_PARAMETER, OuBackend::Matrix);
    }
    if (n_obs < 2) {
        return invalid_state_distribution(SCAR_INVALID_SIZE, OuBackend::Matrix);
    }

    scar_internal::OuGrid grid;
    if (!valid_grid_config(config, OuBackend::Matrix)
        || !scar_internal::build_ou_grid(
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
        return invalid_state_distribution(SCAR_INVALID_SIZE, OuBackend::Matrix);
    }

    std::vector<double> matrix;
    if (!scar_internal::build_dense_transition_matrix(grid, matrix)) {
        return invalid_state_distribution(
            SCAR_INVALID_SIZE, OuBackend::Matrix);
    }
    const double* observation_values = observation_data(copula, u);
    auto advance = [&](const std::vector<double>& source,
                       std::vector<double>& next_phi) {
        scar_internal::dense_predict_matvec(matrix, grid, source, next_phi);
    };
    return state_distribution_impl(
        copula, grid, observation_values, n_obs, horizon_next,
        OuBackend::Matrix, advance);
}

StateDistribution ScarOuEvaluator::state_distribution_local_gh(
    const OuParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const OuNumericalConfig& config,
    bool horizon_next) const {

    const std::int64_t n_obs = static_cast<std::int64_t>(u.size());
    if (!supported_ou_copula(copula)) {
        return invalid_state_distribution(SCAR_INVALID_TRANSFORM, OuBackend::LocalGh);
    }
    if (!valid_ou_params(params) || !finite_config_doubles(config)) {
        return invalid_state_distribution(SCAR_INVALID_PARAMETER, OuBackend::LocalGh);
    }
    if (n_obs < 2) {
        return invalid_state_distribution(SCAR_INVALID_SIZE, OuBackend::LocalGh);
    }

    scar_internal::OuGrid grid;
    std::vector<double> gh_nodes;
    std::vector<double> gh_weights;
    if (!valid_grid_config(config, OuBackend::LocalGh)
        || config.gh_order <= 0
        || static_cast<std::size_t>(config.gh_order)
            > scar_internal::kMaxSpectralOrder
        || !scar_internal::build_ou_grid(
            params.kappa,
            params.mu,
            params.nu,
            n_obs,
            config.K,
            config.grid_range,
            config.adaptive,
            config.pts_per_sigma,
            config.max_K,
            grid)
        || !scar_internal::physicists_hermite_normal_rule(
            config.gh_order, gh_nodes, gh_weights)) {
        return invalid_state_distribution(SCAR_INVALID_SIZE, OuBackend::LocalGh);
    }

    const double* observation_values = observation_data(copula, u);
    auto advance = [&](const std::vector<double>& source,
                       std::vector<double>& next_phi) {
        scar_internal::local_gh_predict_matvec(
            grid.z,
            grid.trap_w,
            grid.rho,
            grid.sigma_cond,
            gh_nodes,
            gh_weights,
            source,
            next_phi);
    };
    return state_distribution_impl(
        copula, grid, observation_values, n_obs, horizon_next,
        OuBackend::LocalGh, advance);
}

}  // namespace scar
