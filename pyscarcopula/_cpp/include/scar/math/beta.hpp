#pragma once

namespace scar::math {

/// Regularized incomplete beta I_x(a,b). Returns NaN outside its domain.
double regularized_beta(
    double x,
    double a,
    double b,
    double tolerance = 3e-14,
    double log_normalization = 0.0,
    bool has_log_normalization = false) noexcept;

/// Inverse regularized incomplete beta for a raw uniform probability.
double beta_quantile(double probability, double a, double b) noexcept;

}  // namespace scar::math
