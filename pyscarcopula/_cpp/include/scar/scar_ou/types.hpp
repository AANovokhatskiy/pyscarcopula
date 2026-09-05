#pragma once

#include <cstdint>

namespace scar {

/// Numerical representation used for SCAR-OU state propagation.
enum class OuBackend : int {
    Spectral = 0,
    LocalGh = 1,
    Matrix = 2,
};

/// Storage used by the Gaussian grid transition in the matrix backend.
enum class OuGridMethod : int {
    Auto = 0,
    Dense = 1,
    Sparse = 2,
};

/// Ornstein-Uhlenbeck parameters `(kappa, mu, nu)`.
struct OuParams {
    double kappa = 1.0;
    double mu = 0.0;
    double nu = 1.0;
};

/// Grid, quadrature, and automatic-dispatch settings for SCAR-OU kernels.
struct OuNumericalConfig {
    int K = 300;
    double grid_range = 5.0;
    bool adaptive = true;
    int pts_per_sigma = 4;
    int max_K = 1000;
    double r_gh = 3.0;
    int gh_order = 5;
    double auto_small_kdt = 1e-2;
    int spectral_basis_order = 32;
    int spectral_quad_order = 0;
    int n_threads = 1;
    OuGridMethod grid_method = OuGridMethod::Auto;
    // Active density, state-derivative and forward-history blocks only.
    // PPF tables, transitions and checkpoint vectors are budgeted separately.
    std::uint64_t corr_gradient_block_bytes = 64U * 1024U * 1024U;
};

}  // namespace scar
