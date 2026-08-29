#pragma once

namespace scar::math {

/// Thread-safe log(|Gamma(x)|). Avoids glibc's process-global signgam state.
double log_gamma(double value) noexcept;

/// Regularized lower incomplete gamma P(a,x). Returns NaN outside its domain.
double regularized_gamma_p(double a, double x) noexcept;

/// Inverse chi-square CDF for a raw uniform probability.
double chi_square_quantile(double probability, double df) noexcept;

}  // namespace scar::math
