#pragma once

#include "scar/copula/multivariate/correlation/dense.hpp"
#include "scar/copula/spec.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace scar::copula::multivariate::gaussian {

double log_pdf(
    const correlation::DenseCorrelation& correlation,
    int dimension,
    const double* scores);
bool accumulate_log_pdf(
    const correlation::DenseCorrelation& correlation,
    int dimension,
    const double* scores,
    std::size_t rows,
    double& log_likelihood,
    std::int64_t& failure_index);
double log_pdf(
    const FactorCorrelationOperator& correlation,
    const double* scores,
    std::vector<double>& factor_projection,
    std::vector<double>& factor_solved);

double log_pdf(
    const CopulaSpec& spec,
    const double* scores,
    std::vector<double>& factor_projection,
    std::vector<double>& factor_solved);

}  // namespace scar::copula::multivariate::gaussian
