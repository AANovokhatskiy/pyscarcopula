#pragma once

#include "scar/core/result.hpp"
#include "scar/scar_ou/types.hpp"

namespace scar {

Result<double> ou_kappa_dt(double kappa, int n_obs) noexcept;
Result<OuBackend> ou_auto_backend(
    double kappa, int n_obs, double small_kdt) noexcept;
Result<int> ou_adaptive_spectral_basis_order(
    double kappa, int n_obs) noexcept;
Result<int> ou_resolve_quad_order(
    int basis_order,
    int explicit_quad_order,
    bool has_explicit_quad_order) noexcept;

}  // namespace scar
