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

std::vector<double> assemble_forward_rosenblatt(
    ObservationView u,
    const std::vector<double>& mixture_h,
    int& status) {

    std::vector<double> out(2 * u.size(), 0.0);
    if (status != SCAR_OK) {
        return out;
    }
    if (u.dim != 2
        || (!u.empty() && u.data() == nullptr)
        || mixture_h.size() != u.size()) {
        status = SCAR_INVALID_SIZE;
        return out;
    }

    for (std::size_t t = 0; t < u.size(); ++t) {
        const double first = u.data()[2 * t];
        const double second = mixture_h[t];
        if (!std::isfinite(first) || !std::isfinite(second)) {
            status = SCAR_NUMERICAL_FAILURE;
            return std::vector<double>(2 * u.size(), 0.0);
        }
        out[2 * t] = std::clamp(
            first, scar_internal::kRosenblattEps,
            1.0 - scar_internal::kRosenblattEps);
        out[2 * t + 1] = std::clamp(
            second, scar_internal::kRosenblattEps,
            1.0 - scar_internal::kRosenblattEps);
    }
    return out;
}

}  // namespace

std::vector<double> ScarOuEvaluator::predictive_mean_local_gh(
    const OuParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const OuNumericalConfig& config,
    int& status) const {

    status = SCAR_OK;
    std::vector<double> out(u.size(), 0.0);
    std::unique_ptr<PreparedDynamicEmission> emission_owner;
    const PreparedDynamicEmission& emission =
        resolve_dynamic_emission(copula, emission_owner);
    if (!supported_ou_copula(emission)) {
        status = SCAR_INVALID_TRANSFORM;
        return out;
    }
    if (!valid_ou_params(params) || !finite_config_doubles(config)) {
        status = SCAR_INVALID_PARAMETER;
        return out;
    }
    scar_internal::OuGrid grid;
    std::vector<double> gh_nodes;
    std::vector<double> gh_weights;
    if (!valid_grid_config(config, OuBackend::LocalGh)
        || config.gh_order <= 0
        || static_cast<std::size_t>(config.gh_order)
            > scar_internal::kMaxSpectralOrder
        || !scar_internal::build_ou_grid(
            params.kappa, params.mu, params.nu, static_cast<std::int64_t>(u.size()),
            config.K, config.grid_range, config.adaptive,
            config.pts_per_sigma, config.max_K, grid)
        || !scar_internal::physicists_hermite_normal_rule(
            config.gh_order, gh_nodes, gh_weights)) {
        status = SCAR_INVALID_SIZE;
        return out;
    }
    const double* observation_values = observation_data(emission, u);
    if (!scar_internal::local_forward_predictive_mean(
            emission,
            grid, gh_nodes, gh_weights, observation_values,
            static_cast<std::int64_t>(u.size()),
            out.data())) {
        status = SCAR_NUMERICAL_FAILURE;
    }
    return out;
}

std::vector<double> ScarOuEvaluator::predictive_mean_matrix(
    const OuParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const OuNumericalConfig& config,
    int& status) const {

    status = SCAR_OK;
    std::vector<double> out(u.size(), 0.0);
    std::unique_ptr<PreparedDynamicEmission> emission_owner;
    const PreparedDynamicEmission& emission =
        resolve_dynamic_emission(copula, emission_owner);
    if (!supported_ou_copula(emission)) {
        status = SCAR_INVALID_TRANSFORM;
        return out;
    }
    if (!valid_ou_params(params) || !finite_config_doubles(config)) {
        status = SCAR_INVALID_PARAMETER;
        return out;
    }
    scar_internal::OuGrid grid;
    if (!valid_grid_config(config, OuBackend::Matrix)
        || !scar_internal::build_ou_grid(
            params.kappa, params.mu, params.nu, static_cast<std::int64_t>(u.size()),
            config.K, config.grid_range, config.adaptive,
            config.pts_per_sigma, config.max_K, grid)) {
        status = SCAR_INVALID_SIZE;
        return out;
    }
    scar_internal::MatrixTransitionOperator transition;
    if (!scar_internal::build_matrix_transition_operator(
            grid, config.grid_method, transition)) {
        status = SCAR_INVALID_SIZE;
        return out;
    }
    const double* observation_values = observation_data(emission, u);
    if (!scar_internal::matrix_forward_predictive_mean(
            emission, grid, transition, observation_values,
            static_cast<std::int64_t>(u.size()),
            out.data())) {
        status = SCAR_NUMERICAL_FAILURE;
    }
    return out;
}

std::vector<double> ScarOuEvaluator::forward_rosenblatt_local_gh(
    const OuParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const OuNumericalConfig& config,
    int& status) const {

    status = SCAR_OK;
    std::vector<double> values(u.size(), 0.0);
    std::unique_ptr<PreparedDynamicEmission> emission_owner;
    const PreparedDynamicEmission& emission =
        resolve_dynamic_emission(copula, emission_owner);
    if (u.dim != 2) {
        status = SCAR_INVALID_SIZE;
        return std::vector<double>(2 * u.size(), 0.0);
    }
    if (!supported_ou_copula(emission)) {
        status = SCAR_INVALID_TRANSFORM;
        return std::vector<double>(2 * u.size(), 0.0);
    }
    if (!valid_ou_params(params) || !finite_config_doubles(config)) {
        status = SCAR_INVALID_PARAMETER;
        return std::vector<double>(2 * u.size(), 0.0);
    }
    scar_internal::OuGrid grid;
    std::vector<double> gh_nodes;
    std::vector<double> gh_weights;
    if (!valid_grid_config(config, OuBackend::LocalGh)
        || config.gh_order <= 0
        || static_cast<std::size_t>(config.gh_order)
            > scar_internal::kMaxSpectralOrder
        || !scar_internal::build_ou_grid(
            params.kappa, params.mu, params.nu,
            static_cast<std::int64_t>(u.size()), config.K,
            config.grid_range, config.adaptive, config.pts_per_sigma,
            config.max_K, grid)
        || !scar_internal::physicists_hermite_normal_rule(
            config.gh_order, gh_nodes, gh_weights)) {
        status = SCAR_INVALID_SIZE;
        return std::vector<double>(2 * u.size(), 0.0);
    }
    const double* observation_values = observation_data(emission, u);
    if (!scar_internal::local_forward_mixture_h(
            emission,
            grid,
            gh_nodes,
            gh_weights,
            observation_values,
            static_cast<std::int64_t>(u.size()),
            values.data())) {
        status = SCAR_NUMERICAL_FAILURE;
    }
    return assemble_forward_rosenblatt(u, values, status);
}

std::vector<double> ScarOuEvaluator::forward_rosenblatt_matrix(
    const OuParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const OuNumericalConfig& config,
    int& status) const {

    status = SCAR_OK;
    std::vector<double> values(u.size(), 0.0);
    std::unique_ptr<PreparedDynamicEmission> emission_owner;
    const PreparedDynamicEmission& emission =
        resolve_dynamic_emission(copula, emission_owner);
    if (u.dim != 2) {
        status = SCAR_INVALID_SIZE;
        return std::vector<double>(2 * u.size(), 0.0);
    }
    if (!supported_ou_copula(emission)) {
        status = SCAR_INVALID_TRANSFORM;
        return std::vector<double>(2 * u.size(), 0.0);
    }
    if (!valid_ou_params(params) || !finite_config_doubles(config)) {
        status = SCAR_INVALID_PARAMETER;
        return std::vector<double>(2 * u.size(), 0.0);
    }
    scar_internal::OuGrid grid;
    if (!valid_grid_config(config, OuBackend::Matrix)
        || !scar_internal::build_ou_grid(
            params.kappa, params.mu, params.nu,
            static_cast<std::int64_t>(u.size()), config.K,
            config.grid_range, config.adaptive, config.pts_per_sigma,
            config.max_K, grid)) {
        status = SCAR_INVALID_SIZE;
        return std::vector<double>(2 * u.size(), 0.0);
    }
    scar_internal::MatrixTransitionOperator transition;
    if (!scar_internal::build_matrix_transition_operator(
            grid, config.grid_method, transition)) {
        status = SCAR_INVALID_SIZE;
        return std::vector<double>(2 * u.size(), 0.0);
    }
    const double* observation_values = observation_data(emission, u);
    if (!scar_internal::matrix_forward_mixture_h(
            emission,
            grid,
            transition,
            observation_values,
            static_cast<std::int64_t>(u.size()),
            values.data())) {
        status = SCAR_NUMERICAL_FAILURE;
    }
    return assemble_forward_rosenblatt(u, values, status);
}

std::vector<double> ScarOuEvaluator::forward_rosenblatt_auto(
    const OuParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const OuNumericalConfig& config,
    OuBackend& backend,
    int& status) const {

    status = SCAR_OK;
    std::unique_ptr<PreparedDynamicEmission> emission_owner;
    const PreparedDynamicEmission& emission =
        resolve_dynamic_emission(copula, emission_owner);
    if (u.dim != 2) {
        status = SCAR_INVALID_SIZE;
        return std::vector<double>(2 * u.size(), 0.0);
    }
    scar_internal::OuGrid grid;
    if (!supported_ou_copula(emission)) {
        status = SCAR_INVALID_TRANSFORM;
        return std::vector<double>(2 * u.size(), 0.0);
    }
    if (!valid_ou_params(params) || !finite_config_doubles(config)) {
        status = SCAR_INVALID_PARAMETER;
        return std::vector<double>(2 * u.size(), 0.0);
    }
    if (!valid_grid_config(config, OuBackend::LocalGh)
        || config.gh_order <= 0
        || static_cast<std::size_t>(config.gh_order)
            > scar_internal::kMaxSpectralOrder
        || !scar_internal::build_ou_grid(
            params.kappa, params.mu, params.nu,
            static_cast<std::int64_t>(u.size()), config.K,
            config.grid_range, config.adaptive, config.pts_per_sigma,
            config.max_K, grid)) {
        status = SCAR_INVALID_SIZE;
        return std::vector<double>(2 * u.size(), 0.0);
    }
    const int selected = scar_internal::select_grid_transition_backend(
        grid, config.r_gh);
    backend = selected == 0 ? OuBackend::Matrix : OuBackend::LocalGh;
    if (backend == OuBackend::Matrix) {
        return forward_rosenblatt_matrix(
            params, copula, u, config, status);
    }
    return forward_rosenblatt_local_gh(
        params, copula, u, config, status);
}

std::vector<double> ScarOuEvaluator::mixture_h_local_gh(
    const OuParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const OuNumericalConfig& config,
    int& status) const {

    status = SCAR_OK;
    std::vector<double> out(u.size(), 0.0);
    std::unique_ptr<PreparedDynamicEmission> emission_owner;
    const PreparedDynamicEmission& emission =
        resolve_dynamic_emission(copula, emission_owner);
    if (!supported_ou_copula(emission)) {
        status = SCAR_INVALID_TRANSFORM;
        return out;
    }
    if (!valid_ou_params(params) || !finite_config_doubles(config)) {
        status = SCAR_INVALID_PARAMETER;
        return out;
    }
    scar_internal::OuGrid grid;
    std::vector<double> gh_nodes;
    std::vector<double> gh_weights;
    if (!valid_grid_config(config, OuBackend::LocalGh)
        || config.gh_order <= 0
        || static_cast<std::size_t>(config.gh_order)
            > scar_internal::kMaxSpectralOrder
        || !scar_internal::build_ou_grid(
            params.kappa, params.mu, params.nu, static_cast<std::int64_t>(u.size()),
            config.K, config.grid_range, config.adaptive,
            config.pts_per_sigma, config.max_K, grid)
        || !scar_internal::physicists_hermite_normal_rule(
            config.gh_order, gh_nodes, gh_weights)) {
        status = SCAR_INVALID_SIZE;
        return out;
    }
    const double* observation_values = observation_data(emission, u);
    if (!scar_internal::local_forward_mixture_h(
            emission,
            grid, gh_nodes, gh_weights, observation_values,
            static_cast<std::int64_t>(u.size()),
            out.data())) {
        status = SCAR_NUMERICAL_FAILURE;
    }
    return out;
}

std::vector<double> ScarOuEvaluator::mixture_h_matrix(
    const OuParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const OuNumericalConfig& config,
    int& status) const {

    status = SCAR_OK;
    std::vector<double> out(u.size(), 0.0);
    std::unique_ptr<PreparedDynamicEmission> emission_owner;
    const PreparedDynamicEmission& emission =
        resolve_dynamic_emission(copula, emission_owner);
    if (!supported_ou_copula(emission)) {
        status = SCAR_INVALID_TRANSFORM;
        return out;
    }
    if (!valid_ou_params(params) || !finite_config_doubles(config)) {
        status = SCAR_INVALID_PARAMETER;
        return out;
    }
    scar_internal::OuGrid grid;
    if (!scar_internal::build_ou_grid(
            params.kappa, params.mu, params.nu, static_cast<std::int64_t>(u.size()),
            config.K, config.grid_range, config.adaptive,
            config.pts_per_sigma, config.max_K, grid)) {
        status = SCAR_INVALID_SIZE;
        return out;
    }
    scar_internal::MatrixTransitionOperator transition;
    if (!scar_internal::build_matrix_transition_operator(
            grid, config.grid_method, transition)) {
        status = SCAR_INVALID_SIZE;
        return out;
    }
    const double* observation_values = observation_data(emission, u);
    if (!scar_internal::matrix_forward_mixture_h(
            emission, grid, transition, observation_values,
            static_cast<std::int64_t>(u.size()),
            out.data())) {
        status = SCAR_NUMERICAL_FAILURE;
    }
    return out;
}

std::vector<double> ScarOuEvaluator::mixture_h_pair_local_gh(
    const OuParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const OuNumericalConfig& config,
    int& status) const {

    status = SCAR_OK;
    std::vector<double> out(2 * u.size(), 0.0);
    std::unique_ptr<PreparedDynamicEmission> emission_owner;
    const PreparedDynamicEmission& emission =
        resolve_dynamic_emission(copula, emission_owner);
    if (!supported_ou_copula(emission)) {
        status = SCAR_INVALID_TRANSFORM;
        return out;
    }
    if (!valid_ou_params(params) || !finite_config_doubles(config)) {
        status = SCAR_INVALID_PARAMETER;
        return out;
    }
    scar_internal::OuGrid grid;
    std::vector<double> gh_nodes;
    std::vector<double> gh_weights;
    if (!valid_grid_config(config, OuBackend::LocalGh)
        || config.gh_order <= 0
        || static_cast<std::size_t>(config.gh_order)
            > scar_internal::kMaxSpectralOrder
        || !scar_internal::build_ou_grid(
            params.kappa, params.mu, params.nu,
            static_cast<std::int64_t>(u.size()), config.K,
            config.grid_range, config.adaptive, config.pts_per_sigma,
            config.max_K, grid)
        || !scar_internal::physicists_hermite_normal_rule(
            config.gh_order, gh_nodes, gh_weights)) {
        status = SCAR_INVALID_SIZE;
        return out;
    }
    const double* observation_values = observation_data(emission, u);
    if (!scar_internal::local_forward_mixture_h(
            emission, grid, gh_nodes, gh_weights, observation_values,
            static_cast<std::int64_t>(u.size()), out.data(),
            out.data() + u.size())) {
        status = SCAR_NUMERICAL_FAILURE;
    }
    return out;
}

std::vector<double> ScarOuEvaluator::mixture_h_pair_matrix(
    const OuParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const OuNumericalConfig& config,
    int& status) const {

    status = SCAR_OK;
    std::vector<double> out(2 * u.size(), 0.0);
    std::unique_ptr<PreparedDynamicEmission> emission_owner;
    const PreparedDynamicEmission& emission =
        resolve_dynamic_emission(copula, emission_owner);
    if (!supported_ou_copula(emission)) {
        status = SCAR_INVALID_TRANSFORM;
        return out;
    }
    if (!valid_ou_params(params) || !finite_config_doubles(config)) {
        status = SCAR_INVALID_PARAMETER;
        return out;
    }
    scar_internal::OuGrid grid;
    if (!scar_internal::build_ou_grid(
            params.kappa, params.mu, params.nu,
            static_cast<std::int64_t>(u.size()), config.K,
            config.grid_range, config.adaptive, config.pts_per_sigma,
            config.max_K, grid)) {
        status = SCAR_INVALID_SIZE;
        return out;
    }
    scar_internal::MatrixTransitionOperator transition;
    if (!scar_internal::build_matrix_transition_operator(
            grid, config.grid_method, transition)) {
        status = SCAR_INVALID_SIZE;
        return out;
    }
    const double* observation_values = observation_data(emission, u);
    if (!scar_internal::matrix_forward_mixture_h(
            emission, grid, transition, observation_values,
            static_cast<std::int64_t>(u.size()), out.data(),
            out.data() + u.size())) {
        status = SCAR_NUMERICAL_FAILURE;
    }
    return out;
}

}  // namespace scar
