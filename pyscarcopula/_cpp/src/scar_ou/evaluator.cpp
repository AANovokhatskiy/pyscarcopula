#include "scar/ou.hpp"

#include "evaluator_internal.hpp"
#include "scar/detail/scar_ou/grid.hpp"
#include "scar/detail/scar_ou/transition.hpp"

#include <vector>

namespace scar {
using namespace evaluator_detail;

GradLogLikResult ScarOuEvaluator::neg_loglik_with_grad_and_corr_auto(
    const OuParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const OuNumericalConfig& config) const {

    if (!valid_ou_params(params) || !finite_config_doubles(config)) {
        return invalid_grad(SCAR_INVALID_PARAMETER, OuBackend::Spectral);
    }
    if (u.empty() || config.auto_small_kdt <= 0.0) {
        return invalid_grad(SCAR_INVALID_SIZE, OuBackend::Spectral);
    }
    const double kdt = u.size() <= 1
        ? params.kappa
        : params.kappa / static_cast<double>(u.size() - 1);
    if (kdt < config.auto_small_kdt) {
        return neg_loglik_with_grad_and_corr_local_gh(
            params, copula, u, config);
    }
    GradLogLikResult result =
        neg_loglik_with_grad_and_corr_spectral(params, copula, u, config);
    if (auto_grad_accepted(result)) {
        return result;
    }
    if (result.status != SCAR_OK
        && !recoverable_numerical_status(result.status)) {
        set_auto_fallback(result, {OuBackend::Spectral});
        return result;
    }
    result = neg_loglik_with_grad_and_corr_matrix(
        params, copula, u, config);
    const int matrix_reason = auto_grad_accepted(result)
        ? matrix_grid_fallback_reason(params, u, config)
        : SCAR_FALLBACK_FAILED;
    if (auto_grad_accepted(result)
        && matrix_reason == SCAR_FALLBACK_NONE) {
        set_auto_fallback(result, {OuBackend::Spectral});
        return result;
    }
    if (result.status != SCAR_OK
        && !recoverable_numerical_status(result.status)) {
        set_auto_fallback(result, {OuBackend::Spectral, OuBackend::Matrix},
                          matrix_reason);
        return result;
    }
    result = neg_loglik_with_grad_and_corr_local_gh(
        params, copula, u, config);
    set_auto_fallback(result, {OuBackend::Spectral, OuBackend::Matrix},
                      matrix_reason);
    return result;
}

GradLogLikResult ScarOuEvaluator::neg_loglik_with_grad_auto(
    const OuParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const OuNumericalConfig& config) const {

    if (!valid_ou_params(params) || !finite_config_doubles(config)) {
        return invalid_grad(SCAR_INVALID_PARAMETER, OuBackend::Spectral);
    }
    if (u.empty() || config.auto_small_kdt <= 0.0) {
        return invalid_grad(SCAR_INVALID_SIZE, OuBackend::Spectral);
    }
    const double kdt = u.size() <= 1
        ? params.kappa
        : params.kappa / static_cast<double>(u.size() - 1);
    if (kdt < config.auto_small_kdt) {
        return neg_loglik_with_grad_local_gh(params, copula, u, config);
    }
    GradLogLikResult result =
        neg_loglik_with_grad_spectral(params, copula, u, config);
    if (auto_grad_accepted(result)) {
        return result;
    }
    if (result.status != SCAR_OK
        && !recoverable_numerical_status(result.status)) {
        set_auto_fallback(result, {OuBackend::Spectral});
        return result;
    }
    result = neg_loglik_with_grad_matrix(params, copula, u, config);
    const int matrix_reason = auto_grad_accepted(result)
        ? matrix_grid_fallback_reason(params, u, config)
        : SCAR_FALLBACK_FAILED;
    if (auto_grad_accepted(result)
        && matrix_reason == SCAR_FALLBACK_NONE) {
        set_auto_fallback(result, {OuBackend::Spectral});
        return result;
    }
    if (result.status != SCAR_OK
        && !recoverable_numerical_status(result.status)) {
        set_auto_fallback(result, {OuBackend::Spectral, OuBackend::Matrix},
                          matrix_reason);
        return result;
    }
    result = neg_loglik_with_grad_local_gh(params, copula, u, config);
    set_auto_fallback(result, {OuBackend::Spectral, OuBackend::Matrix},
                      matrix_reason);
    return result;
}

LogLikResult ScarOuEvaluator::loglik_auto(
    const OuParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const OuNumericalConfig& config) const {

    if (!valid_ou_params(params) || !finite_config_doubles(config)) {
        return invalid_loglik(SCAR_INVALID_PARAMETER, OuBackend::Spectral);
    }
    if (u.empty() || config.auto_small_kdt <= 0.0) {
        return invalid_loglik(SCAR_INVALID_SIZE, OuBackend::Spectral);
    }
    const double kdt = u.size() <= 1
        ? params.kappa
        : params.kappa / static_cast<double>(u.size() - 1);
    if (kdt < config.auto_small_kdt) {
        return loglik_local_gh(params, copula, u, config);
    }

    LogLikResult result = loglik_spectral(params, copula, u, config);
    if (auto_loglik_accepted(result)) {
        return result;
    }
    if (result.status != SCAR_OK
        && !recoverable_numerical_status(result.status)) {
        set_auto_fallback(result, {OuBackend::Spectral});
        return result;
    }
    result = loglik_matrix(params, copula, u, config);
    const int matrix_reason = auto_loglik_accepted(result)
        ? matrix_grid_fallback_reason(params, u, config)
        : SCAR_FALLBACK_FAILED;
    if (auto_loglik_accepted(result)
        && matrix_reason == SCAR_FALLBACK_NONE) {
        set_auto_fallback(result, {OuBackend::Spectral});
        return result;
    }
    if (result.status != SCAR_OK
        && !recoverable_numerical_status(result.status)) {
        set_auto_fallback(result, {OuBackend::Spectral, OuBackend::Matrix},
                          matrix_reason);
        return result;
    }
    result = loglik_local_gh(params, copula, u, config);
    set_auto_fallback(result, {OuBackend::Spectral, OuBackend::Matrix},
                      matrix_reason);
    return result;
}

std::vector<double> ScarOuEvaluator::predictive_mean_auto(
    const OuParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const OuNumericalConfig& config,
    OuBackend& backend,
    int& status) const {

    scar_internal::OuGrid grid;
    if (!supported_ou_copula(copula)) {
        status = SCAR_INVALID_TRANSFORM;
        return std::vector<double>(u.size(), 0.0);
    }
    if (!valid_ou_params(params) || !finite_config_doubles(config)) {
        status = SCAR_INVALID_PARAMETER;
        return std::vector<double>(u.size(), 0.0);
    }
    if (!valid_grid_config(config, OuBackend::LocalGh)
        || config.gh_order <= 0
        || static_cast<std::size_t>(config.gh_order)
            > scar_internal::kMaxSpectralOrder
        || !scar_internal::build_ou_grid(
            params.kappa, params.mu, params.nu, static_cast<std::int64_t>(u.size()),
            config.K, config.grid_range, config.adaptive,
            config.pts_per_sigma, config.max_K, grid)) {
        status = SCAR_INVALID_SIZE;
        return std::vector<double>(u.size(), 0.0);
    }
    const int selected = scar_internal::select_grid_transition_backend(
        grid, config.r_gh);
    backend = selected == 0 ? OuBackend::Matrix : OuBackend::LocalGh;
    if (backend == OuBackend::Matrix) {
        return predictive_mean_matrix(params, copula, u, config, status);
    }
    return predictive_mean_local_gh(params, copula, u, config, status);
}

std::vector<double> ScarOuEvaluator::mixture_h_auto(
    const OuParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const OuNumericalConfig& config,
    OuBackend& backend,
    int& status) const {

    scar_internal::OuGrid grid;
    if (!supported_ou_copula(copula)) {
        status = SCAR_INVALID_TRANSFORM;
        return std::vector<double>(u.size(), 0.0);
    }
    if (!valid_ou_params(params) || !finite_config_doubles(config)) {
        status = SCAR_INVALID_PARAMETER;
        return std::vector<double>(u.size(), 0.0);
    }
    if (!scar_internal::build_ou_grid(
            params.kappa, params.mu, params.nu, static_cast<std::int64_t>(u.size()),
            config.K, config.grid_range, config.adaptive,
            config.pts_per_sigma, config.max_K, grid)) {
        status = SCAR_INVALID_SIZE;
        return std::vector<double>(u.size(), 0.0);
    }
    const int selected = scar_internal::select_grid_transition_backend(
        grid, config.r_gh);
    backend = selected == 0 ? OuBackend::Matrix : OuBackend::LocalGh;
    if (backend == OuBackend::Matrix) {
        return mixture_h_matrix(params, copula, u, config, status);
    }
    return mixture_h_local_gh(params, copula, u, config, status);
}

StateDistribution ScarOuEvaluator::state_distribution_auto(
    const OuParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const OuNumericalConfig& config,
    bool horizon_next) const {

    scar_internal::OuGrid grid;
    if (!supported_ou_copula(copula)) {
        return invalid_state_distribution(SCAR_INVALID_TRANSFORM, OuBackend::Matrix);
    }
    if (!valid_ou_params(params) || !finite_config_doubles(config)) {
        return invalid_state_distribution(SCAR_INVALID_PARAMETER, OuBackend::Matrix);
    }
    if (!scar_internal::build_ou_grid(
            params.kappa, params.mu, params.nu, static_cast<std::int64_t>(u.size()),
            config.K, config.grid_range, config.adaptive,
            config.pts_per_sigma, config.max_K, grid)) {
        return invalid_state_distribution(SCAR_INVALID_SIZE, OuBackend::Matrix);
    }
    const int selected = scar_internal::select_grid_transition_backend(
        grid, config.r_gh);
    if (selected == 0) {
        return state_distribution_matrix(params, copula, u, config, horizon_next);
    }
    return state_distribution_local_gh(params, copula, u, config, horizon_next);
}

}  // namespace scar
