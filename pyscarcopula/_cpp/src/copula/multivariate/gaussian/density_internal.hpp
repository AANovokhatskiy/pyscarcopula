#pragma once

#include "scar/copula/multivariate/correlation/dense.hpp"

#include <cstddef>
#include <cstdint>

namespace scar::copula::multivariate::gaussian {

bool accumulate_log_pdf(
    const correlation::DenseCorrelation& correlation,
    int dimension,
    const double* scores,
    std::size_t rows,
    double& log_likelihood,
    std::int64_t& failure_index);

}  // namespace scar::copula::multivariate::gaussian
