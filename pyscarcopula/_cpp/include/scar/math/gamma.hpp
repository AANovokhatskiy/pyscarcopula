#pragma once

namespace scar::math {

/// Regularized lower incomplete gamma P(a,x). Returns NaN outside its domain.
double regularized_gamma_p(double a, double x) noexcept;

/// Inverse chi-square CDF for a raw uniform probability.
double chi_square_quantile(double probability, double df) noexcept;

}  // namespace scar::math
