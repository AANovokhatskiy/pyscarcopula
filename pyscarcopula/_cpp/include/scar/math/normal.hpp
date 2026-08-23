#pragma once

namespace scar::math {

double normal_cdf(double value) noexcept;
double normal_quantile(double probability);
double normal_quantile_refined(double probability);

}  // namespace scar::math
