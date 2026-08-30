#include "scar/scar_ou/policy.hpp"

#include "scar/detail/safety.hpp"
#include "scar/scar_ou/quadrature.hpp"

#include <cmath>
#include <cstddef>

namespace scar {

Result<double> ou_kappa_dt(double kappa, int n_obs) noexcept {
    if (n_obs < 2) {
        return {0.0, Status::InvalidParameter, {}};
    }
    return success(kappa / static_cast<double>(n_obs - 1));
}

Result<OuBackend> ou_auto_backend(
    double kappa, int n_obs, double small_kdt) noexcept {

    const auto kappa_dt = ou_kappa_dt(kappa, n_obs);
    if (!kappa_dt.is_ok()
        || !std::isfinite(kappa_dt.value)
        || kappa_dt.value <= 0.0
        || !std::isfinite(small_kdt)
        || small_kdt <= 0.0) {
        return {OuBackend::Spectral, Status::InvalidParameter, {}};
    }
    return success(
        kappa_dt.value < small_kdt
            ? OuBackend::LocalGh
            : OuBackend::Spectral);
}

Result<int> ou_adaptive_spectral_basis_order(
    double kappa, int n_obs) noexcept {

    const auto kappa_dt = ou_kappa_dt(kappa, n_obs);
    if (!kappa_dt.is_ok()) {
        return {0, kappa_dt.status, kappa_dt.failure};
    }
    if (!std::isfinite(kappa_dt.value) || kappa_dt.value <= 0.0) {
        return {0, Status::InvalidParameter, {}};
    }
    if (kappa_dt.value < 0.015) {
        return success(128);
    }
    if (kappa_dt.value < 0.025) {
        return success(96);
    }
    if (kappa_dt.value < 0.06) {
        return success(64);
    }
    return success(32);
}

Result<int> ou_resolve_quad_order(
    int basis_order,
    int explicit_quad_order,
    bool has_explicit_quad_order) noexcept {

    if (basis_order <= 0
        || static_cast<std::size_t>(basis_order)
            > scar_internal::kMaxSpectralOrder) {
        return {0, Status::InvalidParameter, {}};
    }
    const auto resolved = has_explicit_quad_order
        ? success(explicit_quad_order)
        : ou_default_quad_order(basis_order);
    if (!resolved.is_ok()
        || resolved.value < basis_order
        || resolved.value <= 0
        || static_cast<std::size_t>(resolved.value)
            > scar_internal::kMaxSpectralOrder) {
        return {0, Status::InvalidParameter, {}};
    }
    return resolved;
}

}  // namespace scar
