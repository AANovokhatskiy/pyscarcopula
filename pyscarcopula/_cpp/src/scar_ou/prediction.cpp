#include "scar/ou.hpp"

#include "evaluator_internal.hpp"
#include "scar/detail/safety.hpp"
#include "scar/detail/scar_ou/grid.hpp"
#include "scar/detail/scar_ou/quadrature.hpp"
#include "scar/detail/scar_ou/transition.hpp"

#include <vector>

namespace scar {
using namespace evaluator_detail;

std::vector<double> ScarOuEvaluator::predictive_mean_local_gh(
    const OuParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const OuNumericalConfig& config,
    int& status) const {

    status = SCAR_OK;
    std::vector<double> out(u.size(), 0.0);
    if (!supported_ou_copula(copula)) {
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
    const double* observation_values = observation_data(copula, u);
    if (!scar_internal::local_forward_predictive_mean(
            copula,
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
    if (!supported_ou_copula(copula)) {
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
    std::vector<double> matrix;
    if (!scar_internal::build_dense_transition_matrix(grid, matrix)) {
        status = SCAR_INVALID_SIZE;
        return out;
    }
    const double* observation_values = observation_data(copula, u);
    if (!scar_internal::matrix_forward_predictive_mean(
            copula, grid, matrix, observation_values,
            static_cast<std::int64_t>(u.size()),
            out.data())) {
        status = SCAR_NUMERICAL_FAILURE;
    }
    return out;
}

std::vector<double> ScarOuEvaluator::mixture_h_local_gh(
    const OuParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const OuNumericalConfig& config,
    int& status) const {

    status = SCAR_OK;
    std::vector<double> out(u.size(), 0.0);
    if (!supported_ou_copula(copula)) {
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
    const double* observation_values = observation_data(copula, u);
    if (!scar_internal::local_forward_mixture_h(
            copula,
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
    if (!supported_ou_copula(copula)) {
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
    std::vector<double> matrix;
    if (!scar_internal::build_dense_transition_matrix(grid, matrix)) {
        status = SCAR_INVALID_SIZE;
        return out;
    }
    const double* observation_values = observation_data(copula, u);
    if (!scar_internal::matrix_forward_mixture_h(
            copula, grid, matrix, observation_values,
            static_cast<std::int64_t>(u.size()),
            out.data())) {
        status = SCAR_NUMERICAL_FAILURE;
    }
    return out;
}

}  // namespace scar
