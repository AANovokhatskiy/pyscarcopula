#include "scar/model_policy.hpp"

#include "scar/copula/prepared_pair_kernel.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace scar {
namespace {

constexpr double kDefaultOptimizerFailureValue = 1e10;

ParameterBoundsResult bounds(
    std::vector<double> lower, std::vector<double> upper) {
    return success(ParameterBounds{std::move(lower), std::move(upper)});
}

double infinity() {
    return std::numeric_limits<double>::infinity();
}

bool finite_vector(const std::vector<double>& values) {
    return std::all_of(
        values.begin(), values.end(),
        [](double value) { return std::isfinite(value); });
}

}  // namespace

ParameterBoundsResult model_public_parameter_bounds(const CopulaSpec& spec) {
    switch (spec.family) {
    case CopulaFamily::Independent:
        return bounds({}, {});
    case CopulaFamily::Clayton:
    case CopulaFamily::Frank:
        return bounds(
            {0.0001},
            {spec.transform == Transform::Logistic ? 20.0001 : infinity()});
    case CopulaFamily::Gumbel:
    case CopulaFamily::Joe:
        return bounds(
            {1.0001},
            {spec.transform == Transform::Logistic ? 21.0001 : infinity()});
    case CopulaFamily::Gaussian:
        return bounds({-0.9999}, {0.9999});
    case CopulaFamily::EquicorrGaussian:
    case CopulaFamily::Student:
        return bounds({-10.0}, {10.0});
    default:
        return {{}, Status::InvalidFamily, {}};
    }
}

ParameterBoundsResult ou_parameter_bounds() {
    return bounds({0.001, -infinity(), 0.001},
                  {infinity(), infinity(), infinity()});
}

ParameterBoundsResult ou_scaled_optimizer_bounds(
    const std::vector<double>& scale) {
    if (scale.size() != 3
        || !std::all_of(
            scale.begin(), scale.end(),
            [](double value) {
                return std::isfinite(value) && value > 0.0;
            })) {
        return {{}, Status::InvalidParameter, {}};
    }
    auto result = ou_parameter_bounds();
    for (std::size_t index = 0; index < 3; ++index) {
        result.value.lower[index] /= scale[index];
        result.value.upper[index] /= scale[index];
    }
    return result;
}

ParameterBoundsResult ou_log_stationary_parameter_bounds() {
    return bounds(
        {std::log(0.001), -infinity(), -infinity()},
        {infinity(), infinity(), infinity()});
}

ParameterBoundsResult ou_log_stationary_parameter_bounds_for_scale(
    double lower, bool has_lower, double upper, bool has_upper) {
    if ((has_lower && (!std::isfinite(lower) || lower <= 0.0))
        || (has_upper && (!std::isfinite(upper) || upper <= 0.0))
        || (has_lower && has_upper && lower >= upper)) {
        return {{}, Status::InvalidParameter, {}};
    }
    auto result = ou_log_stationary_parameter_bounds();
    if (has_lower) {
        result.value.lower[2] = std::log(lower);
    }
    if (has_upper) {
        result.value.upper[2] = std::log(upper);
    }
    return result;
}

ParameterBoundsResult ou_log_stationary_result_bounds() {
    return bounds(
        {0.001, -infinity(), 0.0},
        {infinity(), infinity(), infinity()});
}

ParameterBoundsResult jacobi_parameter_bounds() {
    return bounds({0.001, 1e-6, 0.001},
                  {infinity(), 1.0 - 1e-6, infinity()});
}

ParameterBoundsResult gas_parameter_bounds(
    double gamma_bound, double beta_bound) {
    if (!std::isfinite(gamma_bound) || !std::isfinite(beta_bound)) {
        return {{}, Status::InvalidParameter, {}};
    }
    return bounds({-infinity(), -gamma_bound, -beta_bound},
                  {infinity(), gamma_bound, beta_bound});
}

ParameterBoundsResult default_gas_parameter_bounds() {
    return gas_parameter_bounds(10.0, 0.999);
}

Result<ParameterBounds> normalize_positive_bounds(
    double lower, bool has_lower, double upper, bool has_upper) {
    const double resolved_lower = has_lower ? lower : 1e-300;
    const double resolved_upper = has_upper ? upper : infinity();
    if (!std::isfinite(resolved_lower) || resolved_lower <= 0.0
        || (has_upper && (!std::isfinite(resolved_upper)
            || resolved_upper <= 0.0))
        || resolved_lower >= resolved_upper) {
        return {{}, Status::InvalidParameter, {}};
    }
    return bounds({resolved_lower}, {resolved_upper});
}

Result<ParameterBounds> stationary_scale_bounds() {
    return bounds({0.001}, {10000.0});
}

FitParameterPolicyResult student_fit_parameter_policy(
    int dimension, bool stochastic) {
    if (dimension < 2) {
        return {{}, Status::InvalidParameter, {}};
    }
    FitParameterPolicy policy;
    policy.initial = stochastic
        ? 5.0 : std::max(static_cast<double>(dimension), 5.0);
    policy.lower = stochastic ? 2.0 + 1e-6 : 2.001;
    policy.has_lower = true;
    policy.upper = stochastic ? 0.0 : 10000.0;
    policy.has_upper = !stochastic;
    return success(policy);
}

FitParameterPolicyResult equicorr_fit_parameter_policy() {
    return success(FitParameterPolicy{0.5, -8.0, 8.0, true, true});
}

Result<double> default_pair_mle_parameter(const CopulaSpec& spec) {
    const PreparedPairKernel kernel(spec);
    if (!kernel.is_supported() || spec.family == CopulaFamily::Independent) {
        return {0.0, Status::InvalidParameter, {}};
    }
    const double value = kernel.transform(1.5);
    return std::isfinite(value)
        ? success(value)
        : Result<double>{0.0, Status::NumericalFailure, {}};
}

Result<std::vector<double>> gas_default_initial_point(double mu) {
    if (!std::isfinite(mu)) {
        return {{}, Status::InvalidParameter, {}};
    }
    return success(std::vector<double>{mu * 0.05, 0.05, 0.95});
}

Result<std::vector<double>> optimizer_unit_scale(
    const std::vector<double>& parameters) {

    if (!finite_vector(parameters)) {
        return {{}, Status::InvalidParameter, {}};
    }
    std::vector<double> scale;
    scale.reserve(parameters.size());
    for (double value : parameters) {
        scale.push_back(std::max(std::abs(value), 1.0));
    }
    return success(std::move(scale));
}

Result<std::vector<double>> project_optimizer_point(
    const std::vector<double>& parameters,
    const std::vector<double>& lower,
    const std::vector<double>& upper) {

    if (parameters.size() != lower.size()
        || parameters.size() != upper.size()
        || !finite_vector(parameters)) {
        return {{}, Status::InvalidParameter, {}};
    }
    std::vector<double> projected;
    projected.reserve(parameters.size());
    for (std::size_t index = 0; index < parameters.size(); ++index) {
        if (std::isnan(lower[index]) || std::isnan(upper[index])
            || lower[index] > upper[index]) {
            return {{}, Status::InvalidParameter, {}};
        }
        projected.push_back(std::min(
            std::max(parameters[index], lower[index]), upper[index]));
    }
    return success(std::move(projected));
}

Result<double> optimizer_failure_objective() {
    return success(kDefaultOptimizerFailureValue);
}

Result<double> optimizer_failure_objective(double fail_value) {
    if (!std::isfinite(fail_value) || fail_value <= 0.0) {
        return {0.0, Status::InvalidParameter, {}};
    }
    return success(fail_value);
}

OptimizerFailureEvaluationResult optimizer_failure_evaluation(
    const std::vector<double>& parameters,
    const std::vector<double>& initial_parameters,
    double fail_value,
    bool directional_gradient) {

    if (parameters.size() != initial_parameters.size()
        || !finite_vector(parameters)
        || !finite_vector(initial_parameters)
        || !std::isfinite(fail_value)
        || fail_value <= 0.0) {
        return {{}, Status::InvalidParameter, {}};
    }

    OptimizerFailureEvaluation output;
    output.objective = fail_value;
    output.gradient.assign(parameters.size(), 0.0);
    if (!directional_gradient || parameters.empty()) {
        return success(std::move(output));
    }

    double squared_norm = 0.0;
    for (std::size_t index = 0; index < parameters.size(); ++index) {
        output.gradient[index] = parameters[index] - initial_parameters[index];
        squared_norm += output.gradient[index] * output.gradient[index];
    }
    const double norm = std::sqrt(squared_norm);
    if (!std::isfinite(norm) || norm == 0.0) {
        std::fill(output.gradient.begin(), output.gradient.end(), 1.0);
    } else {
        for (double& value : output.gradient) {
            value /= norm;
        }
    }
    const double scale = std::sqrt(fail_value);
    for (double& value : output.gradient) {
        value *= scale;
    }
    return success(std::move(output));
}

OptimizerFailureEvaluationResult optimizer_failure_evaluation_for_size(
    std::size_t gradient_size,
    double fail_value) {

    if (!std::isfinite(fail_value) || fail_value <= 0.0) {
        return {{}, Status::InvalidParameter, {}};
    }
    OptimizerFailureEvaluation output;
    output.objective = fail_value;
    output.gradient.assign(gradient_size, 0.0);
    return success(std::move(output));
}

}  // namespace scar
