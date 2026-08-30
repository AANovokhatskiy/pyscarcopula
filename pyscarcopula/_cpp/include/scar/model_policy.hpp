#pragma once

#include "scar/copula/spec.hpp"
#include "scar/core/result.hpp"

#include <cstddef>
#include <vector>

namespace scar {

struct ParameterBounds {
    std::vector<double> lower;
    std::vector<double> upper;
};

struct FitParameterPolicy {
    double initial = 0.0;
    double lower = 0.0;
    double upper = 0.0;
    bool has_lower = false;
    bool has_upper = false;
};

struct OptimizerFailureEvaluation {
    double objective = 0.0;
    std::vector<double> gradient;
};

using ParameterBoundsResult = Result<ParameterBounds>;
using FitParameterPolicyResult = Result<FitParameterPolicy>;
using OptimizerFailureEvaluationResult = Result<OptimizerFailureEvaluation>;

ParameterBoundsResult model_public_parameter_bounds(const CopulaSpec& spec);
ParameterBoundsResult ou_parameter_bounds();
ParameterBoundsResult ou_scaled_optimizer_bounds(
    const std::vector<double>& scale);
ParameterBoundsResult ou_log_stationary_parameter_bounds();
ParameterBoundsResult ou_log_stationary_parameter_bounds_for_scale(
    double lower, bool has_lower, double upper, bool has_upper);
ParameterBoundsResult ou_log_stationary_result_bounds();
ParameterBoundsResult jacobi_parameter_bounds();
ParameterBoundsResult gas_parameter_bounds(double gamma_bound, double beta_bound);
ParameterBoundsResult default_gas_parameter_bounds();
Result<ParameterBounds> normalize_positive_bounds(
    double lower, bool has_lower, double upper, bool has_upper);
Result<ParameterBounds> stationary_scale_bounds();
FitParameterPolicyResult student_fit_parameter_policy(
    int dimension, bool stochastic);
FitParameterPolicyResult equicorr_fit_parameter_policy();
Result<double> default_pair_mle_parameter(const CopulaSpec& spec);
// Family-scale Kendall tau: signed for Gaussian, positive for other pairs.
// Exact limits require a finite model bound; unbounded itau limits fail.
Result<double> pair_mle_initial_parameter(const CopulaSpec& spec, double tau);
Result<std::vector<double>> gas_default_initial_point(double mu);
Result<std::vector<double>> optimizer_unit_scale(
    const std::vector<double>& parameters);
Result<std::vector<double>> project_optimizer_point(
    const std::vector<double>& parameters,
    const std::vector<double>& lower,
    const std::vector<double>& upper);
Result<double> optimizer_failure_objective();
Result<double> optimizer_failure_objective(double fail_value);
OptimizerFailureEvaluationResult optimizer_failure_evaluation(
    const std::vector<double>& parameters,
    const std::vector<double>& initial_parameters,
    double fail_value,
    bool directional_gradient);
OptimizerFailureEvaluationResult optimizer_failure_evaluation_for_size(
    std::size_t gradient_size,
    double fail_value);

}  // namespace scar
