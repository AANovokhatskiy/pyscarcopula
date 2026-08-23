#include "scar/copula/pair/independent.hpp"

#include <algorithm>
#include <limits>

namespace scar::copula::pair {
namespace {

double undefined_mapping(double) {
    return std::numeric_limits<double>::quiet_NaN();
}

double log_pdf(double, double, double) {
    return 0.0;
}

double zero(double, double, double) {
    return 0.0;
}

void pdf_and_gradient(
    double,
    double,
    double,
    double,
    double,
    double& density,
    double& gradient) {

    density = 1.0;
    gradient = 0.0;
}

double h(double first, double, double) {
    return first;
}

void fill_density_grid_row(
    double,
    double,
    const std::vector<double>& parameters,
    double* densities) {

    std::fill(densities, densities + parameters.size(), 1.0);
}

void fill_density_gradient_grid_row(
    double,
    double,
    const std::vector<double>& parameters,
    const std::vector<double>&,
    double* densities,
    double* gradients) {

    std::fill(densities, densities + parameters.size(), 1.0);
    std::fill(gradients, gradients + parameters.size(), 0.0);
}

const PairKernelFunctions kFunctions = {
    undefined_mapping,
    undefined_mapping,
    log_pdf,
    zero,
    pdf_and_gradient,
    h,
    h,
    fill_density_grid_row,
    fill_density_gradient_grid_row,
};

}  // namespace

const PairKernelFunctions& independent_kernel() noexcept {
    return kFunctions;
}

}  // namespace scar::copula::pair
