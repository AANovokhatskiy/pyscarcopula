#pragma once

namespace scar::math {

double normal_cdf(double value) noexcept;
double normal_quantile(double probability);
double normal_quantile_refined(double probability);

/// Correlate two independent standard normal draws and map them to uniforms.
void bivariate_normal_draws_to_uniforms(
    double first_draw,
    double second_draw,
    double rho,
    double& first,
    double& second) noexcept;

}  // namespace scar::math
