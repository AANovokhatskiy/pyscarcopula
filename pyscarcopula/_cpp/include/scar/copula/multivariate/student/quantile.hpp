#pragma once

#include "scar/copula/spec.hpp"
#include "scar/copula/multivariate/student/ppf_cache.hpp"

#include <cstdint>

namespace scar_internal {

bool use_large_df_quantile(const scar::CopulaSpec& spec, double df);
bool use_large_df_quantile(
    const scar::copula::multivariate::student::PpfCache& cache,
    double df);
void student_quantile_for_emission(
    const scar::CopulaSpec& spec,
    double p,
    double df,
    double& value,
    double* derivative);
void student_quantile_for_emission(
    const scar::copula::multivariate::student::PpfCache& cache,
    double p,
    double df,
    double& value,
    double* derivative);
double student_quantile_value(double p, double df);
double student_quantile_for_observation(
    const scar::CopulaSpec& spec,
    double p,
    double df,
    std::int64_t row_index,
    int column);
void student_quantile_value_and_derivative(
    double p,
    double df,
    double& value,
    double& derivative);
void student_quantile_large_df_value_and_derivative(
    double p,
    double df,
    double& value,
    double& derivative);

}  // namespace scar_internal
