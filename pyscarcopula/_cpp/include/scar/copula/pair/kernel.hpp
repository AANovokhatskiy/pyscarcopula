#pragma once

#include <vector>

namespace scar::copula::pair {

using PairScalarKernel = double (*)(double);
using PairDensityKernel = double (*)(double, double, double);
using PairDensityGradientKernel = void (*)(
    double,
    double,
    double,
    double,
    double,
    double&,
    double&);
using PairConditionalKernel = double (*)(double, double, double);
using PairDensityGridKernel = void (*)(
    double,
    double,
    const std::vector<double>&,
    double*);
using PairDensityGradientGridKernel = void (*)(
    double,
    double,
    const std::vector<double>&,
    const std::vector<double>&,
    double*,
    double*);

/// Function table implemented by one pair-copula package.
///
/// Common coordinate and parameter wrappers deliberately do not appear in
/// family implementations. PreparedPairKernel owns those policies.
struct PairKernelFunctions {
    PairScalarKernel tau_to_parameter = nullptr;
    PairScalarKernel parameter_to_tau = nullptr;
    PairDensityKernel log_pdf = nullptr;
    PairDensityKernel dlog_pdf_dparameter = nullptr;
    PairDensityGradientKernel pdf_and_gradient = nullptr;
    PairConditionalKernel h = nullptr;
    PairConditionalKernel inverse_h = nullptr;
    PairDensityGridKernel fill_density_grid_row = nullptr;
    PairDensityGradientGridKernel fill_density_gradient_grid_row = nullptr;
};

}  // namespace scar::copula::pair
