#pragma once

#include <cstdint>
#include <limits>

namespace scar {

inline constexpr int kMaxJacobiOrder = 2048;
inline constexpr std::uint64_t kDefaultJacobiMemoryBudgetBytes =
    1024ULL * 1024ULL * 1024ULL;
inline constexpr double kJacobiRawClip = 50.0;
inline constexpr double kJacobiDriftDenominatorFloor = 1e-300;

enum class JacobiBoundaryPolicy : int {
    Reflect = 0,
    Clip = 1,
};

enum class JacobiTransitionMethod : int {
    Auto = 0,
    SpectralMatrix = 1,
    Local = 2,
    LocalFixed = 3,
    SpectralCoeff = 4,
};

enum class JacobiTransitionStorage : int {
    Dense = 0,
    Sparse = 1,
};

enum class JacobiStationarityCorrection : int {
    None = 0,
    MetropolisHastings = 1,
    IpFp = 2,
};

/// Physical Jacobi diffusion parameters `(kappa, m, xi)`.
struct JacobiParams {
    double kappa = 1.0;
    double m = 0.5;
    double xi = 0.25;
};

/// Physical optimizer limits plus the open-unit margin for `m`.
struct JacobiParameterBounds {
    double kappa_lower = 1e-3;
    double kappa_upper = 100.0;
    double xi_lower = 1e-3;
    double xi_upper = 5.0;
    double tau_eps = 1e-6;
};

/// Domain-level Jacobi settings shared by later transition/evaluator stages.
struct JacobiNumericalConfig {
    int quad_order = 48;
    int basis_order = 16;
    int gh_order = 5;
    std::int64_t n_obs = 0;
    bool matrix = true;
    bool gradient = false;
    std::uint64_t memory_budget_bytes =
        kDefaultJacobiMemoryBudgetBytes;
    double tau_eps = 1e-6;
    double theta_cap = std::numeric_limits<double>::quiet_NaN();
    double stationary_shape_max = 500.0;
    double lamperti_eps = 1e-10;
    JacobiBoundaryPolicy boundary = JacobiBoundaryPolicy::Reflect;
};

/// Complete native transition-construction policy.  Transition construction
/// requires at least two observations and always uses the fixed product grid
/// step `1 / (numerical.n_obs - 1)` on `[0, 1]`.
struct JacobiTransitionConfig {
    JacobiNumericalConfig numerical{};
    JacobiTransitionMethod method = JacobiTransitionMethod::Auto;
    JacobiTransitionStorage storage = JacobiTransitionStorage::Dense;
    JacobiStationarityCorrection correction =
        JacobiStationarityCorrection::None;
    double negative_mass_tolerance = 1e-5;
    bool clip_negative = false;
    bool derivatives = false;
    double ipfp_tolerance = 1e-15;
    int ipfp_max_iterations = 10000;
};

struct JacobiAdaptiveThresholds {
    double max_full_horizon_tv = 0.02;
    double max_relative_variance_error = 0.10;
    double max_conditional_mean_rmse = 1e-3;
    double max_lag_one_correlation_error = 1e-2;
};

}  // namespace scar
