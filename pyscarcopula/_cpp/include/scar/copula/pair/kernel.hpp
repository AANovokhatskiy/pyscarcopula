#pragma once

#include "scar/copula/conditional_options.hpp"

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
using PairConfiguredInverseKernel = double (*)(
    double, double, double, const HInverseOptions&);
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
    PairConfiguredInverseKernel inverse_h_with_options = nullptr;
    // Tail-preserving forms avoid rounding 1-q to one during rotation.
    // h_reflected(u,v) = 1-h(1-u,v);
    // inverse_h_reflected(q,v) = 1-inverse_h(1-q,v).
    PairConditionalKernel h_reflected = nullptr;
    PairConditionalKernel inverse_h_reflected = nullptr;
    PairConfiguredInverseKernel inverse_h_reflected_with_options = nullptr;
};

}  // namespace scar::copula::pair
