#pragma once

#include "scar/copula/pair/kernel.hpp"

namespace scar::copula::pair {

const PairKernelFunctions& gaussian_kernel() noexcept;

double gaussian_h_from_quantiles(double z_u, double z_v, double rho);
void gaussian_h_pair_from_quantiles(
    double z_first,
    double z_second,
    double rho,
    double& first_next,
    double& second_next);
void gaussian_fill_grid_row_from_stats(
    double sum_squares,
    double cross_product,
    const std::vector<double>& parameter_grid,
    const std::vector<double>& derivative_grid,
    double* pdf_row,
    double* gradient_row);

}  // namespace scar::copula::pair
