#include "scar/ou.hpp"

#include "evaluator_internal.hpp"
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

namespace {

SmoothedStateDistribution invalid_smoothed_state_distribution(
    int status,
    OuBackend backend) {

    SmoothedStateDistribution out;
    out.backend = backend;
    out.status = status;
    return out;
}

SmoothedStateDistribution smoothed_state_distribution_impl(
    const PreparedDynamicEmission& emission,
    ObservationView u,
    const scar_internal::OuGrid& grid,
    const scar_internal::GridTransitionOperator& transition,
    OuBackend backend) {

    const std::int64_t n_obs = static_cast<std::int64_t>(u.size());
    const std::size_t K = static_cast<std::size_t>(grid.K);
    std::size_t state_value_count = 0;
    if (n_obs < 2
        || K < 2
        || !scar_internal::checked_shape_size(
            static_cast<std::size_t>(n_obs), K, state_value_count)) {
        return invalid_smoothed_state_distribution(
            SCAR_INVALID_SIZE, backend);
    }

    std::vector<double> emissions(state_value_count, 0.0);
    const double* observation_values = observation_data(emission, u);
    std::vector<double> row(K, 0.0);
    for (std::int64_t t = 0; t < n_obs; ++t) {
        emission.fill_density_row_on_state_grid(
            observation_values, t, grid.x_grid, row);
        std::copy(
            row.begin(),
            row.end(),
            emissions.begin()
                + static_cast<std::ptrdiff_t>(
                    static_cast<std::size_t>(t) * K));
    }

    scar_internal::SmoothedStateResult smoothed;
    if (!scar_internal::smooth_state_emissions(
            grid,
            transition,
            emissions.data(),
            n_obs,
            smoothed)) {
        return invalid_smoothed_state_distribution(
            SCAR_NUMERICAL_FAILURE, backend);
    }

    SmoothedStateDistribution out;
    out.z_grid = grid.x_grid;
    out.weights = std::move(smoothed.weights);
    out.n_obs = n_obs;
    out.K = grid.K;
    out.backend = backend;
    out.status = SCAR_OK;
    return out;
}

template <typename AdvanceDensity>
StateDistribution state_distribution_impl(
    const PreparedDynamicEmission& emission,
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
        return scar_internal::normalize_density_by_max(phi);
    };

    for (std::int64_t t = 0; t < n_obs; ++t) {
        emission.fill_density_row_on_state_grid(
            u, t, grid.x_grid, fi_row);
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
        if (!scar_internal::normalize_density_by_max(phi)) {
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
    if (!scar_internal::predictive_weights_from_phi(
            grid, phi, out.prob)) {
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
    std::unique_ptr<PreparedDynamicEmission> emission_owner;
    const PreparedDynamicEmission& emission =
        resolve_dynamic_emission(copula, emission_owner);
    if (!supported_ou_copula(emission)) {
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

    scar_internal::MatrixTransitionOperator transition;
    if (!scar_internal::build_matrix_transition_operator(
            grid, config.grid_method, transition)) {
        return invalid_state_distribution(
            SCAR_INVALID_SIZE, OuBackend::Matrix);
    }
    const double* observation_values = observation_data(emission, u);
    auto advance = [&](const std::vector<double>& source,
                       std::vector<double>& next_phi) {
        scar_internal::matrix_predict_matvec(
            transition, grid, source, next_phi);
    };
    return state_distribution_impl(
        emission, grid, observation_values, n_obs, horizon_next,
        OuBackend::Matrix, advance);
}

StateDistribution ScarOuEvaluator::state_distribution_local_gh(
    const OuParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const OuNumericalConfig& config,
    bool horizon_next) const {

    const std::int64_t n_obs = static_cast<std::int64_t>(u.size());
    std::unique_ptr<PreparedDynamicEmission> emission_owner;
    const PreparedDynamicEmission& emission =
        resolve_dynamic_emission(copula, emission_owner);
    if (!supported_ou_copula(emission)) {
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

    const double* observation_values = observation_data(emission, u);
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
        emission, grid, observation_values, n_obs, horizon_next,
        OuBackend::LocalGh, advance);
}

SmoothedStateDistribution
ScarOuEvaluator::smoothed_state_distribution_matrix(
    const OuParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const OuNumericalConfig& config) const {

    const std::int64_t n_obs = static_cast<std::int64_t>(u.size());
    std::unique_ptr<PreparedDynamicEmission> emission_owner;
    const PreparedDynamicEmission& emission =
        resolve_dynamic_emission(copula, emission_owner);
    if (!supported_ou_copula(emission)) {
        return invalid_smoothed_state_distribution(
            SCAR_INVALID_TRANSFORM, OuBackend::Matrix);
    }
    if (!valid_ou_params(params) || !finite_config_doubles(config)) {
        return invalid_smoothed_state_distribution(
            SCAR_INVALID_PARAMETER, OuBackend::Matrix);
    }
    if (n_obs < 2) {
        return invalid_smoothed_state_distribution(
            SCAR_INVALID_SIZE, OuBackend::Matrix);
    }

    scar_internal::OuGrid grid;
    scar_internal::GridTransitionOperator transition;
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
            grid)
        || !scar_internal::build_grid_transition_operator(
            grid,
            OuBackend::Matrix,
            config.grid_method,
            config.gh_order,
            transition)) {
        return invalid_smoothed_state_distribution(
            SCAR_INVALID_SIZE, OuBackend::Matrix);
    }
    return smoothed_state_distribution_impl(
        emission, u, grid, transition, OuBackend::Matrix);
}

SmoothedStateDistribution
ScarOuEvaluator::smoothed_state_distribution_local_gh(
    const OuParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const OuNumericalConfig& config) const {

    const std::int64_t n_obs = static_cast<std::int64_t>(u.size());
    std::unique_ptr<PreparedDynamicEmission> emission_owner;
    const PreparedDynamicEmission& emission =
        resolve_dynamic_emission(copula, emission_owner);
    if (!supported_ou_copula(emission)) {
        return invalid_smoothed_state_distribution(
            SCAR_INVALID_TRANSFORM, OuBackend::LocalGh);
    }
    if (!valid_ou_params(params) || !finite_config_doubles(config)) {
        return invalid_smoothed_state_distribution(
            SCAR_INVALID_PARAMETER, OuBackend::LocalGh);
    }
    if (n_obs < 2) {
        return invalid_smoothed_state_distribution(
            SCAR_INVALID_SIZE, OuBackend::LocalGh);
    }

    scar_internal::OuGrid grid;
    scar_internal::GridTransitionOperator transition;
    if (!valid_grid_config(config, OuBackend::LocalGh)
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
        || !scar_internal::build_grid_transition_operator(
            grid,
            OuBackend::LocalGh,
            config.grid_method,
            config.gh_order,
            transition)) {
        return invalid_smoothed_state_distribution(
            SCAR_INVALID_SIZE, OuBackend::LocalGh);
    }
    return smoothed_state_distribution_impl(
        emission, u, grid, transition, OuBackend::LocalGh);
}

}  // namespace scar
