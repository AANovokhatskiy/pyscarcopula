#include "scar/ou.hpp"

#include "evaluator_internal.hpp"
#include "scar/detail/copula.hpp"
#include "scar/detail/safety.hpp"
#include "scar/detail/scar_ou/grid.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace scar::evaluator_detail {

const double* observation_data(const CopulaSpec& copula, ObservationView u) {
    const int expected_dim =
        (copula.family == CopulaFamily::Student
         || copula.family == CopulaFamily::EquicorrGaussian)
        ? copula.dim
        : 2;
    if (u.dim != expected_dim) {
        throw std::invalid_argument("u dimension does not match CopulaSpec::dim");
    }
    if (!u.empty() && u.data() == nullptr) {
        throw std::invalid_argument("u data pointer must not be null");
    }
    return u.data();
}

bool supported_ou_copula(const CopulaSpec& copula) {
    return scar_internal::copula_is_supported_for_ou(copula);
}

bool valid_ou_params(const OuParams& params) {
    return std::isfinite(params.kappa)
        && std::isfinite(params.mu)
        && std::isfinite(params.nu)
        && params.kappa > 0.0
        && params.nu > 0.0;
}

bool finite_config_doubles(const OuNumericalConfig& config) {
    return std::isfinite(config.grid_range)
        && std::isfinite(config.r_gh)
        && std::isfinite(config.auto_small_kdt);
}

bool valid_grid_config(
    const OuNumericalConfig& config,
    OuBackend backend) {

    std::size_t K = 0;
    if (!scar_internal::checked_positive_int_size(
            config.K, scar_internal::kMaxGridSize, K)) {
        return false;
    }
    if (backend == OuBackend::Matrix
        && K > scar_internal::kMaxDenseGridSize) {
        return false;
    }
    if (config.max_K > 0) {
        std::size_t max_K = 0;
        if (!scar_internal::checked_positive_int_size(
                config.max_K, scar_internal::kMaxGridSize, max_K)) {
            return false;
        }
        if (backend == OuBackend::Matrix
            && max_K > scar_internal::kMaxDenseGridSize) {
            return false;
        }
    }
    return true;
}

bool valid_observation_grid_size(std::size_t n_obs, int K) {
    std::size_t K_size = 0;
    std::size_t elements = 0;
    return scar_internal::checked_positive_int_size(
               K, scar_internal::kMaxGridSize, K_size)
        && scar_internal::checked_size_mul(n_obs, K_size, elements);
}

bool adaptive_grid_exceeds_limit(
    const OuParams& params,
    std::int64_t n_obs,
    const OuNumericalConfig& config) {

    if (!config.adaptive || config.max_K > 0 || n_obs < 2) {
        return false;
    }
    const double dt = 1.0 / static_cast<double>(n_obs - 1);
    const double conditional_variance =
        -std::expm1(-2.0 * params.kappa * dt);
    if (!std::isfinite(conditional_variance)
        || conditional_variance <= 0.0) {
        return false;
    }
    const double K_min_value = std::ceil(
        2.0 * config.grid_range
        * static_cast<double>(config.pts_per_sigma)
        / std::sqrt(conditional_variance)) + 1.0;
    return std::isfinite(K_min_value)
        && (K_min_value > static_cast<double>(scar_internal::kMaxGridSize)
            || K_min_value > static_cast<double>(INT_MAX));
}

bool recoverable_numerical_status(int status) {
    return status == SCAR_NUMERICAL_FAILURE;
}

bool auto_loglik_accepted(const LogLikResult& result) {
    return result.status == SCAR_OK
        && std::isfinite(result.log_likelihood)
        && result.log_likelihood > -1e9;
}

bool auto_grad_accepted(const GradLogLikResult& result) {
    return result.status == SCAR_OK
        && std::isfinite(result.neg_log_likelihood)
        && result.neg_log_likelihood < 1e9;
}

LogLikResult invalid_loglik(int status, OuBackend backend) {
    return {
        -std::numeric_limits<double>::infinity(),
        backend,
        status,
        -1,
        {},
        SCAR_FALLBACK_NONE,
    };
}

GradLogLikResult invalid_grad(int status, OuBackend backend) {
    return {
        1e10,
        std::vector<double>{0.0, 0.0, 0.0},
        backend,
        status,
        -1,
        {},
        SCAR_FALLBACK_NONE,
        {},
    };
}

StateDistribution invalid_state_distribution(int status, OuBackend backend) {
    return {{}, {}, backend, status};
}

OuNumericalConfig with_default_quad_order(OuNumericalConfig config) {
    if (config.spectral_quad_order <= 0) {
        if (config.spectral_basis_order <= 0
            || static_cast<std::size_t>(config.spectral_basis_order)
                > scar_internal::kMaxSpectralOrder) {
            return config;
        }
        config.spectral_quad_order =
            std::max(2 * config.spectral_basis_order + 16, 48);
    }
    return config;
}

int matrix_grid_fallback_reason(
    const OuParams& params,
    ObservationView u,
    const OuNumericalConfig& config) {

    scar_internal::OuGrid grid;
    if (!scar_internal::build_ou_grid(
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
        return SCAR_FALLBACK_FAILED;
    }
    return grid.adaptive_was_capped
        ? SCAR_FALLBACK_CAPPED
        : SCAR_FALLBACK_NONE;
}

void set_auto_fallback(
    LogLikResult& result,
    const std::vector<OuBackend>& chain,
    int matrix_reason) {

    result.fallback_chain = chain;
    result.fallback_from = chain.empty()
        ? -1
        : static_cast<int>(chain.back());
    result.matrix_fallback_reason = matrix_reason;
}

void set_auto_fallback(
    GradLogLikResult& result,
    const std::vector<OuBackend>& chain,
    int matrix_reason) {

    result.fallback_chain = chain;
    result.fallback_from = chain.empty()
        ? -1
        : static_cast<int>(chain.back());
    result.matrix_fallback_reason = matrix_reason;
}


}  // namespace scar::evaluator_detail
