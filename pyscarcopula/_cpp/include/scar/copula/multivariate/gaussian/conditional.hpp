#pragma once

#include "scar/copula/multivariate/conditional_result.hpp"
#include "scar/core/span.hpp"

#include <cstdint>
#include <vector>

namespace scar {

ConditionalSampleResult multivariate_gaussian_conditional(
    const std::vector<double>& correlations,
    std::int64_t correlation_rows,
    int dimension,
    const std::vector<int>& given_indices,
    const std::vector<double>& given_latent,
    const std::vector<double>& normal_draws,
    std::int64_t n_rows,
    int n_threads = 1);

ConditionalSampleResult multivariate_gaussian_conditional(
    DoubleView correlations,
    std::int64_t correlation_rows,
    int dimension,
    const std::vector<int>& given_indices,
    DoubleView given_latent,
    DoubleView normal_draws,
    std::int64_t n_rows,
    int n_threads = 1);

}  // namespace scar
