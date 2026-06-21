#pragma once

#include "scar/ou.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace scar::evaluator_detail {

const double* observation_data(const CopulaSpec& copula, ObservationView u);
bool supported_ou_copula(const CopulaSpec& copula);
bool valid_ou_params(const OuParams& params);
bool finite_config_doubles(const OuNumericalConfig& config);
bool valid_grid_config(
    const OuNumericalConfig& config,
    OuBackend backend);
bool valid_observation_grid_size(std::size_t n_obs, int K);
bool adaptive_grid_exceeds_limit(
    const OuParams& params,
    std::int64_t n_obs,
    const OuNumericalConfig& config);
bool recoverable_numerical_status(int status);
bool auto_loglik_accepted(const LogLikResult& result);
bool auto_grad_accepted(const GradLogLikResult& result);
LogLikResult invalid_loglik(int status, OuBackend backend);
GradLogLikResult invalid_grad(int status, OuBackend backend);
StateDistribution invalid_state_distribution(int status, OuBackend backend);
OuNumericalConfig with_default_quad_order(OuNumericalConfig config);
int matrix_grid_fallback_reason(
    const OuParams& params,
    ObservationView u,
    const OuNumericalConfig& config);
void set_auto_fallback(
    LogLikResult& result,
    const std::vector<OuBackend>& chain,
    int matrix_reason = SCAR_FALLBACK_NONE);
void set_auto_fallback(
    GradLogLikResult& result,
    const std::vector<OuBackend>& chain,
    int matrix_reason = SCAR_FALLBACK_NONE);

}  // namespace scar::evaluator_detail
