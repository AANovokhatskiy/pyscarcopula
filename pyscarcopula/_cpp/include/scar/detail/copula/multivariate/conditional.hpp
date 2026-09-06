#pragma once

#include "scar/copula/multivariate/conditional_result.hpp"
#include "scar/core/span.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace scar_internal {

struct ConditionalScale {
    double covariance = 1.0;
    double radial = 1.0;
};

using ConditionalScaleFunction = int (*)(
    const double* given,
    const double* solved_given,
    std::size_t n_given,
    std::size_t row,
    const void* context,
    ConditionalScale& scale);

/// Model policy applied around the shared conditional correlation algebra.
struct ConditionalPolicy {
    bool auxiliary_sizes_valid = true;
    bool require_prepared_factorization = true;
    bool allow_jittered_prepared_factor = true;
    ConditionalScaleFunction scale = nullptr;
    const void* context = nullptr;
};

scar::ConditionalSampleResult conditional_latent(
    scar::DoubleView correlations,
    std::int64_t correlation_rows,
    int dimension,
    const std::vector<int>& given_indices,
    scar::DoubleView given_latent,
    scar::DoubleView normal_draws,
    std::int64_t n_rows,
    int n_threads,
    const ConditionalPolicy& policy);

}  // namespace scar_internal
