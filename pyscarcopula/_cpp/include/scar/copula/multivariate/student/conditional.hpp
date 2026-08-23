#pragma once

#include "scar/copula/multivariate/conditional_result.hpp"
#include "scar/core/span.hpp"

#include <cstdint>
#include <vector>

namespace scar {

ConditionalSampleResult multivariate_student_conditional(
    const std::vector<double>& correlations,
    std::int64_t correlation_rows,
    int dimension,
    const std::vector<int>& given_indices,
    const std::vector<double>& given_latent,
    const std::vector<double>& df,
    const std::vector<double>& normal_draws,
    const std::vector<double>& chi_square_draws,
    std::int64_t n_rows,
    int n_threads = 1);

ConditionalSampleResult multivariate_student_conditional(
    DoubleView correlations,
    std::int64_t correlation_rows,
    int dimension,
    const std::vector<int>& given_indices,
    DoubleView given_latent,
    DoubleView df,
    DoubleView normal_draws,
    DoubleView chi_square_draws,
    std::int64_t n_rows,
    int n_threads = 1);

}  // namespace scar
