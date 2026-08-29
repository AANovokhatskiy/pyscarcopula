#include "scar/ou.hpp"

#include "evaluator_internal.hpp"
#include "scar/core/threading.hpp"
#include "scar/detail/safety.hpp"
#include "scar/detail/scar_ou/grid.hpp"
#include "scar/scar_ou/quadrature.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace scar::evaluator_detail {

const double* observation_data(
    const PreparedDynamicEmission& emission,
    ObservationView u) {

    const int expected_dim = emission.expected_dimension();
    if (u.dim != expected_dim) {
        throw std::invalid_argument("u dimension does not match CopulaSpec::dim");
    }
    const bool prepared_equicorr =
        emission.kind() == DynamicEmissionKind::Equicorrelation
        && emission.has_cached_observations(u.size());
    if (!u.empty() && u.data() == nullptr && !prepared_equicorr) {
        throw std::invalid_argument("u data pointer must not be null");
    }
    return u.data();
}

Result<std::size_t> rosenblatt_output_size(
    ObservationView u,
    int expected_dimension) noexcept {

    Result<std::size_t> result;
    std::size_t output_size = 0;
    if (expected_dimension < 2 || u.dim != expected_dimension) {
        result.status = Status::InvalidSize;
        result.failure.coordinate = u.dim;
        return result;
    }
    if (!scar_internal::checked_shape_size(
            u.size(),
            static_cast<std::size_t>(expected_dimension),
            output_size)) {
        result.status = Status::InvalidSize;
        return result;
    }
    return success(output_size);
}

bool supported_ou_copula(const PreparedDynamicEmission& emission) {
    return emission.is_supported_for_ou();
}

bool valid_ou_params(const OuParams& params) {
    return std::isfinite(params.kappa)
        && std::isfinite(params.mu)
        && std::isfinite(params.nu)
        && params.kappa > 0.0
        && params.nu > 0.0;
}

bool finite_config_doubles(const OuNumericalConfig& config) {
    const bool valid_grid_method =
        config.grid_method == OuGridMethod::Auto
        || config.grid_method == OuGridMethod::Dense
        || config.grid_method == OuGridMethod::Sparse;
    return std::isfinite(config.grid_range)
        && std::isfinite(config.r_gh)
        && std::isfinite(config.auto_small_kdt)
        && scar_internal::valid_thread_count(config.n_threads)
        && valid_grid_method;
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
        && config.grid_method == OuGridMethod::Dense
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
            && config.grid_method == OuGridMethod::Dense
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

bool recoverable_numerical_status(Status status) {
    return status == Status::NumericalFailure;
}

bool auto_loglik_accepted(const LogLikResult& result) {
    return result.is_ok()
        && std::isfinite(result.log_likelihood)
        && result.log_likelihood > -1e9;
}

bool auto_grad_accepted(const GradLogLikResult& result) {
    return result.is_ok()
        && std::isfinite(result.neg_log_likelihood)
        && result.neg_log_likelihood < 1e9;
}

LogLikResult invalid_loglik(int status, OuBackend backend) {
    LogLikResult out;
    out.log_likelihood = -std::numeric_limits<double>::infinity();
    out.backend = backend;
    out.status = status_from_int(status);
    return out;
}

GradLogLikResult invalid_grad(int status, OuBackend backend) {
    GradLogLikResult out;
    out.neg_log_likelihood = 1e10;
    out.neg_gradient = {0.0, 0.0, 0.0};
    out.backend = backend;
    out.status = status_from_int(status);
    return out;
}

StateDistribution invalid_state_distribution(int status, OuBackend backend) {
    StateDistribution out;
    out.backend = backend;
    out.status = status_from_int(status);
    return out;
}

OuNumericalConfig with_default_quad_order(OuNumericalConfig config) {
    if (config.spectral_quad_order <= 0) {
        if (config.spectral_basis_order <= 0
            || static_cast<std::size_t>(config.spectral_basis_order)
                > scar_internal::kMaxSpectralOrder) {
            return config;
        }
        config.spectral_quad_order =
            ou_default_quad_order(config.spectral_basis_order).value;
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
    result.failure.fallback_from = chain.empty()
        ? -1
        : static_cast<int>(chain.back());
    result.matrix_fallback_reason = matrix_reason;
}

void set_auto_fallback(
    GradLogLikResult& result,
    const std::vector<OuBackend>& chain,
    int matrix_reason) {

    result.fallback_chain = chain;
    result.failure.fallback_from = chain.empty()
        ? -1
        : static_cast<int>(chain.back());
    result.matrix_fallback_reason = matrix_reason;
}


}  // namespace scar::evaluator_detail
