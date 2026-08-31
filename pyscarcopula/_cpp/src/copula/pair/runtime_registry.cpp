#include "scar/copula/prepared_pair_kernel.hpp"

#include "scar/copula/pair/gaussian.hpp"
#include "scar/math/normal.hpp"
#include "scar/numerical_constants.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace scar {

namespace copula::pair {

#define SCAR_PAIR_FAMILY(                                                \
    enum_name, package_name, enum_value, transform_policy, rotation_policy, \
    default_transform, default_offset)                                  \
    const PairKernelFunctions& package_name##_kernel() noexcept;
#include "scar/copula/pair/families.def"
#undef SCAR_PAIR_FAMILY

}  // namespace copula::pair

namespace {

enum class PairTransformPolicy {
    Any,
    Archimedean,
    GaussianTanh,
};

enum class PairRotationPolicy {
    Any,
    R0Only,
};

struct PairFamilyRegistration {
    const copula::pair::PairKernelFunctions* functions = nullptr;
    PairTransformPolicy transform_policy = PairTransformPolicy::Any;
    PairRotationPolicy rotation_policy = PairRotationPolicy::Any;
    Transform default_transform = Transform::Softplus;
    double default_offset = 0.0;
};

const PairFamilyRegistration* registered_kernel(
    CopulaFamily family) noexcept {

    // This is the only runtime registration point for pair families.
    switch (family) {
#define SCAR_PAIR_FAMILY(                                                \
    enum_name, package_name, enum_value, transform_policy, rotation_policy, \
    default_transform, default_offset)                                  \
    case CopulaFamily::enum_name: {                                     \
        static const PairFamilyRegistration registration = {            \
            &copula::pair::package_name##_kernel(),                     \
            PairTransformPolicy::transform_policy,                      \
            PairRotationPolicy::rotation_policy,                        \
            Transform::default_transform,                               \
            default_offset,                                              \
        };                                                               \
        return &registration;                                            \
    }
#include "scar/copula/pair/families.def"
#undef SCAR_PAIR_FAMILY
    default:
        return nullptr;
    }
}

bool supports_archimedean_transform(Transform transform) noexcept {
    return transform == Transform::Softplus
        || transform == Transform::XTanh
        || transform == Transform::Exponential
        || transform == Transform::Logistic;
}

bool supports_spec(
    const PairFamilyRegistration& registration,
    const CopulaSpec& spec) noexcept {

    if (!copula::is_valid_rotation(static_cast<int>(spec.rotation))
        || !std::isfinite(spec.offset)
        || spec.offset < 0.0) {
        return false;
    }
    if (registration.rotation_policy == PairRotationPolicy::R0Only
        && spec.rotation != Rotation::R0) {
        return false;
    }
    switch (registration.transform_policy) {
    case PairTransformPolicy::Any:
        return true;
    case PairTransformPolicy::Archimedean:
        return supports_archimedean_transform(spec.transform);
    case PairTransformPolicy::GaussianTanh:
        return spec.transform == Transform::GaussianTanh;
    }
    return false;
}

double quiet_nan() {
    return std::numeric_limits<double>::quiet_NaN();
}

}  // namespace

bool is_pair_copula_family(CopulaFamily family) noexcept {
    return registered_kernel(family) != nullptr;
}

PreparedPairKernel::PreparedPairKernel(const CopulaSpec& spec) noexcept
    : family_(spec.family),
      rotation_(spec.rotation),
      transform_(spec.transform),
      offset_(spec.offset) {

    const PairFamilyRegistration* registration =
        registered_kernel(spec.family);
    if (registration != nullptr) {
        functions_ = registration->functions;
        supported_ = supports_spec(*registration, spec);
    }
}

bool PreparedPairKernel::is_registered() const noexcept {
    return functions_ != nullptr;
}

bool PreparedPairKernel::is_supported() const noexcept {
    return supported_;
}

CopulaFamily PreparedPairKernel::family() const noexcept {
    return family_;
}

CopulaSpec default_pair_copula_spec(CopulaFamily family) noexcept {
    CopulaSpec spec;
    spec.family = family;
    spec.rotation = Rotation::R0;
    const PairFamilyRegistration* registration = registered_kernel(family);
    if (registration != nullptr) {
        spec.transform = registration->default_transform;
        spec.offset = registration->default_offset;
    }
    return spec;
}

double PreparedPairKernel::transform(double value) const {
    if (!is_supported()) {
        return quiet_nan();
    }
    if (family_ == CopulaFamily::Independent) {
        return 0.0;
    }
    return copula::transform_parameter(transform_, value, offset_);
}

double PreparedPairKernel::inverse_transform(double parameter) const {
    if (!is_supported()) {
        return quiet_nan();
    }
    if (family_ == CopulaFamily::Independent) {
        return 0.0;
    }
    return copula::inverse_transform_parameter(
        transform_, parameter, offset_, false);
}

double PreparedPairKernel::dtransform(double value) const {
    if (!is_supported()) {
        return quiet_nan();
    }
    if (family_ == CopulaFamily::Independent) {
        return 0.0;
    }
    return copula::d_transform_parameter(transform_, value);
}

double PreparedPairKernel::tau_to_parameter(double tau) const {
    return is_supported() && functions_->tau_to_parameter != nullptr
        ? functions_->tau_to_parameter(tau)
        : quiet_nan();
}

double PreparedPairKernel::parameter_to_tau(double parameter) const {
    return is_supported() && functions_->parameter_to_tau != nullptr
        ? functions_->parameter_to_tau(parameter)
        : quiet_nan();
}

double PreparedPairKernel::log_pdf_unrotated(
    double first,
    double second,
    double parameter) const {

    return is_supported() && functions_->log_pdf != nullptr
        ? functions_->log_pdf(first, second, parameter)
        : -std::numeric_limits<double>::infinity();
}

double PreparedPairKernel::dlog_pdf_dparameter_unrotated(
    double first,
    double second,
    double parameter) const {

    return is_supported() && functions_->dlog_pdf_dparameter != nullptr
        ? functions_->dlog_pdf_dparameter(first, second, parameter)
        : quiet_nan();
}

double PreparedPairKernel::dlog_pdf_dparameter(
    double first,
    double second,
    double parameter) const {

    if (!is_supported()) {
        return quiet_nan();
    }
    double rotated_first = 0.0;
    double rotated_second = 0.0;
    copula::apply_rotation(
        first,
        second,
        static_cast<int>(rotation_),
        rotated_first,
        rotated_second);
    return functions_->dlog_pdf_dparameter(
        rotated_first, rotated_second, parameter);
}

double PreparedPairKernel::log_pdf(
    double first,
    double second,
    double parameter) const {

    if (!is_supported()) {
        return -std::numeric_limits<double>::infinity();
    }
    double rotated_first = 0.0;
    double rotated_second = 0.0;
    copula::apply_rotation(
        first,
        second,
        static_cast<int>(rotation_),
        rotated_first,
        rotated_second);
    return functions_->log_pdf(rotated_first, rotated_second, parameter);
}

double PreparedPairKernel::pdf(
    double first,
    double second,
    double value) const {

    return std::exp(log_pdf(first, second, transform(value)));
}

double PreparedPairKernel::pdf_unrotated(
    double first,
    double second,
    double value) const {

    return std::exp(log_pdf_unrotated(
        first, second, transform(value)));
}

void PreparedPairKernel::pdf_and_gradient_unrotated(
    double first,
    double second,
    double value,
    double& density,
    double& gradient) const {

    if (!is_supported() || functions_->pdf_and_gradient == nullptr) {
        density = quiet_nan();
        gradient = quiet_nan();
        return;
    }
    functions_->pdf_and_gradient(
        first,
        second,
        value,
        transform(value),
        dtransform(value),
        density,
        gradient);
}

void PreparedPairKernel::pdf_and_gradient(
    double first,
    double second,
    double value,
    double& density,
    double& gradient) const {

    double rotated_first = 0.0;
    double rotated_second = 0.0;
    copula::apply_rotation(
        first,
        second,
        static_cast<int>(rotation_),
        rotated_first,
        rotated_second);
    pdf_and_gradient_unrotated(
        rotated_first,
        rotated_second,
        value,
        density,
        gradient);
}

double PreparedPairKernel::h(
    double first,
    double second,
    double parameter) const {

    return is_supported() && functions_->h != nullptr
        ? copula::evaluate_rotated_conditional(
            first,
            second,
            parameter,
            static_cast<int>(rotation_),
            functions_->h)
        : quiet_nan();
}

void PreparedPairKernel::h_pair(
    double first,
    double second,
    double parameter,
    double& first_next,
    double& second_next) const {

    if (is_unrotated_gaussian()) {
        const double first_quantile = math::normal_quantile(
            std::clamp(
                first,
                numerical::kPseudoObservationEps,
                1.0 - numerical::kPseudoObservationEps));
        const double second_quantile = math::normal_quantile(
            std::clamp(
                second,
                numerical::kPseudoObservationEps,
                1.0 - numerical::kPseudoObservationEps));
        copula::pair::gaussian_h_pair_from_quantiles(
            first_quantile,
            second_quantile,
            parameter,
            first_next,
            second_next);
        return;
    }
    first_next = h(first, second, parameter);
    PreparedPairKernel transposed = *this;
    if (transposed.rotation_ == Rotation::R90) {
        transposed.rotation_ = Rotation::R270;
    } else if (transposed.rotation_ == Rotation::R270) {
        transposed.rotation_ = Rotation::R90;
    }
    second_next = transposed.h(second, first, parameter);
}

double PreparedPairKernel::inverse_h(
    double quantile,
    double given,
    double parameter,
    const HInverseOptions* options) const {

    if (is_supported() && options != nullptr
        && functions_->inverse_h_with_options != nullptr) {
        return copula::evaluate_rotated_conditional_with(
            quantile, given, parameter, static_cast<int>(rotation_),
            [this, options](double q, double v, double r) {
                return functions_->inverse_h_with_options(q, v, r, *options);
            });
    }
    return is_supported() && functions_->inverse_h != nullptr
        ? copula::evaluate_rotated_conditional(
            quantile,
            given,
            parameter,
            static_cast<int>(rotation_),
            functions_->inverse_h)
        : quiet_nan();
}

bool PreparedPairKernel::is_unrotated_gaussian() const noexcept {
    return is_supported()
        && family_ == CopulaFamily::Gaussian
        && rotation_ == Rotation::R0;
}

double PreparedPairKernel::prepare_conditional_value(double value) const {
    if (!is_unrotated_gaussian()) {
        return quiet_nan();
    }
    return math::normal_quantile(
        std::clamp(
            value,
            numerical::kPseudoObservationEps,
            1.0 - numerical::kPseudoObservationEps));
}

double PreparedPairKernel::h_from_prepared_values(
    double first,
    double second,
    double parameter) const {
    return is_unrotated_gaussian()
        ? copula::pair::gaussian_h_from_quantiles(
            first, second, parameter)
        : quiet_nan();
}

void PreparedPairKernel::h_pair_from_prepared_values(
    double first,
    double second,
    double parameter,
    double& first_next,
    double& second_next) const {

    if (!is_unrotated_gaussian()) {
        first_next = quiet_nan();
        second_next = quiet_nan();
        return;
    }
    copula::pair::gaussian_h_pair_from_quantiles(
        first,
        second,
        parameter,
        first_next,
        second_next);
}

void PreparedPairKernel::prepare_parameter_grid(
    const std::vector<double>& values,
    std::vector<double>& parameters,
    std::vector<double>& derivatives) const {

    parameters.resize(values.size());
    derivatives.resize(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        parameters[index] = transform(values[index]);
        derivatives[index] = dtransform(values[index]);
    }
}

void PreparedPairKernel::fill_grid_row(
    double first,
    double second,
    const std::vector<double>& parameters,
    double* densities) const {

    if (!is_supported()
        || functions_->fill_density_grid_row == nullptr) {
        std::fill(
            densities,
            densities + parameters.size(),
            quiet_nan());
        return;
    }
    double rotated_first = 0.0;
    double rotated_second = 0.0;
    copula::apply_rotation(
        first,
        second,
        static_cast<int>(rotation_),
        rotated_first,
        rotated_second);
    functions_->fill_density_grid_row(
        rotated_first,
        rotated_second,
        parameters,
        densities);
}

void PreparedPairKernel::fill_grid_row_with_gradient(
    double first,
    double second,
    const std::vector<double>& parameters,
    const std::vector<double>& derivatives,
    double* densities,
    double* gradients) const {

    if (!is_supported()
        || functions_->fill_density_gradient_grid_row == nullptr) {
        std::fill(
            densities,
            densities + parameters.size(),
            quiet_nan());
        if (gradients != nullptr) {
            std::fill(
                gradients,
                gradients + parameters.size(),
                quiet_nan());
        }
        return;
    }
    double rotated_first = 0.0;
    double rotated_second = 0.0;
    copula::apply_rotation(
        first,
        second,
        static_cast<int>(rotation_),
        rotated_first,
        rotated_second);
    functions_->fill_density_gradient_grid_row(
        rotated_first,
        rotated_second,
        parameters,
        derivatives,
        densities,
        gradients);
}

}  // namespace scar
