#include "scar/scar_ou/parameterization.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace scar {
namespace {

bool finite_vector(const std::vector<double>& values) noexcept {
    return std::all_of(
        values.begin(), values.end(),
        [](double value) { return std::isfinite(value); });
}

bool valid_scale(
    const std::vector<double>& values,
    const std::vector<double>& scale) noexcept {
    return values.size() == scale.size()
        && finite_vector(values)
        && std::all_of(
            scale.begin(), scale.end(),
            [](double value) {
                return std::isfinite(value) && value > 0.0;
            });
}

OuParameterVectorResult invalid() noexcept {
    return {{}, Status::InvalidParameter, {}};
}

}  // namespace

OuParameterVectorResult ou_to_log_stationary(
    const std::vector<double>& physical) noexcept {

    if (physical.size() < 3 || !finite_vector(physical)
        || physical[0] <= 0.0 || physical[2] <= 0.0) {
        return invalid();
    }
    const double stationary_scale =
        physical[2] / std::sqrt(2.0 * physical[0]);
    if (!std::isfinite(stationary_scale) || stationary_scale <= 0.0) {
        return invalid();
    }
    std::vector<double> values = physical;
    values[0] = std::log(physical[0]);
    values[2] = std::log(stationary_scale);
    return success(std::move(values));
}

OuParameterVectorResult ou_from_log_stationary(
    const std::vector<double>& optimizer) noexcept {

    if (optimizer.size() < 3 || !finite_vector(optimizer)) {
        return invalid();
    }
    const double kappa = std::exp(optimizer[0]);
    const double stationary_scale = std::exp(optimizer[2]);
    const double nu = stationary_scale * std::sqrt(2.0 * kappa);
    if (!std::isfinite(kappa) || !std::isfinite(nu)
        || kappa <= 0.0 || nu <= 0.0) {
        return invalid();
    }
    std::vector<double> values = optimizer;
    values[0] = kappa;
    values[2] = nu;
    return success(std::move(values));
}

OuParameterVectorResult ou_gradient_to_log_stationary(
    const std::vector<double>& physical,
    const std::vector<double>& gradient) noexcept {

    if (physical.size() < 3 || physical.size() != gradient.size()
        || !finite_vector(physical) || !finite_vector(gradient)
        || physical[0] <= 0.0 || physical[2] <= 0.0) {
        return invalid();
    }
    std::vector<double> values = gradient;
    values[0] = gradient[0] * physical[0]
        + 0.5 * gradient[2] * physical[2];
    values[2] = gradient[2] * physical[2];
    return finite_vector(values) ? success(std::move(values)) : invalid();
}

OuParameterVectorResult ou_gradient_from_log_stationary(
    const std::vector<double>& physical,
    const std::vector<double>& gradient) noexcept {

    if (physical.size() < 3 || physical.size() != gradient.size()
        || !finite_vector(physical) || !finite_vector(gradient)
        || physical[0] <= 0.0 || physical[2] <= 0.0) {
        return invalid();
    }
    std::vector<double> values = gradient;
    values[0] = (gradient[0] - 0.5 * gradient[2]) / physical[0];
    values[2] = gradient[2] / physical[2];
    return finite_vector(values) ? success(std::move(values)) : invalid();
}

OuParameterVectorResult ou_project_optimizer_block(
    const std::vector<double>& values,
    const std::vector<double>& lower,
    const std::vector<double>& upper) noexcept {

    if (values.size() < 3 || lower.size() != 3 || upper.size() != 3
        || !finite_vector(values)) {
        return invalid();
    }
    std::vector<double> projected = values;
    for (std::size_t index = 0; index < 3; ++index) {
        if (std::isnan(lower[index]) || std::isnan(upper[index])
            || lower[index] > upper[index]) {
            return invalid();
        }
        projected[index] = std::clamp(
            projected[index], lower[index], upper[index]);
    }
    return success(std::move(projected));
}

OuParameterVectorResult optimizer_scaled_to_physical(
    const std::vector<double>& values,
    const std::vector<double>& scale) noexcept {

    if (!valid_scale(values, scale)) {
        return invalid();
    }
    std::vector<double> result(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        result[index] = values[index] * scale[index];
    }
    return finite_vector(result) ? success(std::move(result)) : invalid();
}

OuParameterVectorResult physical_to_optimizer_scaled(
    const std::vector<double>& values,
    const std::vector<double>& scale) noexcept {

    if (!valid_scale(values, scale)) {
        return invalid();
    }
    std::vector<double> result(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        result[index] = values[index] / scale[index];
    }
    return finite_vector(result) ? success(std::move(result)) : invalid();
}

OuParameterVectorResult gradient_to_optimizer_scaled(
    const std::vector<double>& gradient,
    const std::vector<double>& scale) noexcept {
    return optimizer_scaled_to_physical(gradient, scale);
}

OuParameterVectorResult gradient_from_optimizer_scaled(
    const std::vector<double>& gradient,
    const std::vector<double>& scale) noexcept {
    return physical_to_optimizer_scaled(gradient, scale);
}

}  // namespace scar
