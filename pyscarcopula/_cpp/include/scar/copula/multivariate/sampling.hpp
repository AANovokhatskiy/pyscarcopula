#pragma once

#include "scar/copula/multivariate/conditional_result.hpp"
#include "scar/core/result.hpp"
#include "scar/core/span.hpp"

#include <cstdint>
#include <vector>

namespace scar {

class FactorCorrelationOperator;

/// Required raw common-factor draw count for an equicorrelation path.
Result<std::int64_t> equicorr_gaussian_common_draw_count(
    DoubleView rho,
    int dimension,
    std::int64_t n_rows);

Status validate_equicorrelation_path(
    DoubleView rho,
    int dimension,
    std::int64_t n_rows) noexcept;

ConditionalSampleResult multivariate_gaussian_sample_dense(
    DoubleView correlation,
    int dimension,
    DoubleView normal_draws,
    std::int64_t n_rows,
    int n_threads = 1);

ConditionalSampleResult multivariate_gaussian_sample_equicorrelation(
    DoubleView rho,
    int dimension,
    DoubleView normal_draws,
    DoubleView common_draws,
    std::int64_t n_rows,
    int n_threads = 1);

ConditionalSampleResult multivariate_student_sample_dense(
    DoubleView correlation,
    int dimension,
    DoubleView df,
    DoubleView normal_draws,
    DoubleView chi_square_draws,
    std::int64_t n_rows,
    int n_threads = 1);

ConditionalSampleResult multivariate_student_sample_dense_from_uniforms(
    DoubleView correlation,
    int dimension,
    DoubleView df,
    DoubleView normal_draws,
    DoubleView chi_square_uniforms,
    std::int64_t n_rows,
    int n_threads = 1);

ConditionalSampleResult multivariate_gaussian_sample_factor(
    const FactorCorrelationOperator& correlation,
    DoubleView factor_draws,
    DoubleView residual_draws,
    std::int64_t n_rows,
    int n_threads = 1);

ConditionalSampleResult multivariate_student_sample_factor(
    const FactorCorrelationOperator& correlation,
    DoubleView df,
    DoubleView factor_draws,
    DoubleView residual_draws,
    DoubleView chi_square_draws,
    std::int64_t n_rows,
    int n_threads = 1);

ConditionalSampleResult multivariate_student_sample_factor_from_uniforms(
    const FactorCorrelationOperator& correlation,
    DoubleView df,
    DoubleView factor_draws,
    DoubleView residual_draws,
    DoubleView chi_square_uniforms,
    std::int64_t n_rows,
    int n_threads = 1);

ConditionalSampleResult multivariate_gaussian_conditional_from_uniforms(
    DoubleView correlations,
    std::int64_t correlation_rows,
    int dimension,
    const std::vector<int>& given_indices,
    DoubleView given_uniforms,
    DoubleView normal_draws,
    std::int64_t n_rows,
    int n_threads = 1);

ConditionalSampleResult
multivariate_gaussian_conditional_equicorrelation_from_uniforms(
    DoubleView rho,
    int dimension,
    const std::vector<int>& given_indices,
    DoubleView given_uniforms,
    DoubleView normal_draws,
    std::int64_t n_rows,
    int n_threads = 1);

ConditionalSampleResult multivariate_student_conditional_from_uniforms(
    DoubleView correlations,
    std::int64_t correlation_rows,
    int dimension,
    const std::vector<int>& given_indices,
    DoubleView given_uniforms,
    DoubleView df,
    DoubleView normal_draws,
    DoubleView chi_square_draws,
    std::int64_t n_rows,
    int n_threads = 1);

ConditionalSampleResult
multivariate_student_conditional_from_normal_uniforms(
    DoubleView correlations,
    std::int64_t correlation_rows,
    int dimension,
    const std::vector<int>& given_indices,
    DoubleView given_uniforms,
    DoubleView df,
    DoubleView normal_draws,
    DoubleView chi_square_uniforms,
    std::int64_t n_rows,
    int n_threads = 1);

ConditionalSampleResult multivariate_gaussian_conditional_factor(
    const FactorCorrelationOperator& correlation,
    const std::vector<int>& given_indices,
    DoubleView given_uniforms,
    DoubleView factor_draws,
    DoubleView residual_draws,
    std::int64_t n_rows,
    int n_threads = 1);

ConditionalSampleResult multivariate_student_conditional_factor(
    const FactorCorrelationOperator& correlation,
    const std::vector<int>& given_indices,
    DoubleView given_uniforms,
    DoubleView df,
    DoubleView factor_draws,
    DoubleView residual_draws,
    DoubleView chi_square_draws,
    std::int64_t n_rows,
    int n_threads = 1);

ConditionalSampleResult
multivariate_student_conditional_factor_from_normal_uniforms(
    const FactorCorrelationOperator& correlation,
    const std::vector<int>& given_indices,
    DoubleView given_uniforms,
    DoubleView df,
    DoubleView factor_draws,
    DoubleView residual_draws,
    DoubleView chi_square_uniforms,
    std::int64_t n_rows,
    int n_threads = 1);

}  // namespace scar
