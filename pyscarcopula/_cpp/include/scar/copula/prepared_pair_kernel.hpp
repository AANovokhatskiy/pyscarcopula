#pragma once

#include "scar/copula.hpp"
#include "scar/copula/pair/kernel.hpp"

#include <vector>

namespace scar {

/// Pair-copula function table selected once from a compatibility CopulaSpec.
///
/// The object is cheap to copy and contains no owning model state. Its row and
/// scalar methods invoke the selected family directly, without family checks
/// in the observation or parameter-grid loops.
class PreparedPairKernel {
public:
    explicit PreparedPairKernel(const CopulaSpec& spec) noexcept;

    bool is_registered() const noexcept;
    bool is_supported() const noexcept;
    CopulaFamily family() const noexcept;

    double transform(double value) const;
    double inverse_transform(double parameter) const;
    double dtransform(double value) const;
    double tau_to_parameter(double tau) const;
    double parameter_to_tau(double parameter) const;

    double log_pdf_unrotated(
        double first,
        double second,
        double parameter) const;
    double dlog_pdf_dparameter_unrotated(
        double first,
        double second,
        double parameter) const;
    double log_pdf(
        double first,
        double second,
        double parameter) const;
    double pdf_unrotated(double first, double second, double value) const;
    double pdf(double first, double second, double value) const;
    void pdf_and_gradient_unrotated(
        double first,
        double second,
        double value,
        double& density,
        double& gradient) const;
    void pdf_and_gradient(
        double first,
        double second,
        double value,
        double& density,
        double& gradient) const;
    double h(double first, double second, double parameter) const;
    double inverse_h(double quantile, double given, double parameter) const;

    void prepare_parameter_grid(
        const std::vector<double>& values,
        std::vector<double>& parameters,
        std::vector<double>& derivatives) const;
    void fill_grid_row(
        double first,
        double second,
        const std::vector<double>& parameters,
        double* densities) const;
    void fill_grid_row_with_gradient(
        double first,
        double second,
        const std::vector<double>& parameters,
        const std::vector<double>& derivatives,
        double* densities,
        double* gradients) const;

private:
    const copula::pair::PairKernelFunctions* functions_ = nullptr;
    CopulaFamily family_ = CopulaFamily::Independent;
    Rotation rotation_ = Rotation::R0;
    Transform transform_ = Transform::Softplus;
    double offset_ = 0.0;
    bool supported_ = false;
};

bool is_pair_copula_family(CopulaFamily family) noexcept;
CopulaSpec default_pair_copula_spec(CopulaFamily family) noexcept;

}  // namespace scar
