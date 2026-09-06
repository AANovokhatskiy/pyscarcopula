#include "scar/gas.hpp"

#include "scar/copula/prepared_dynamic_emission.hpp"
#include "scar/copula/prepared_pair_kernel.hpp"
#include "scar/detail/copula/common.hpp"
#include "scar/detail/safety.hpp"
#include "scar/math/normal.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <utility>

namespace scar {
namespace {

struct RowEvaluation {
    double r = 0.0;
    double log_likelihood = 0.0;
    double score = 0.0;
    int status = SCAR_OK;
};

bool valid_params(const GasParams& params) {
    return std::isfinite(params.omega)
        && std::isfinite(params.gamma)
        && std::isfinite(params.beta);
}

bool valid_config(const GasConfig& config) {
    return (config.scaling == GasScaling::Unit
            || config.scaling == GasScaling::Fisher)
        && std::isfinite(config.score_eps)
        && config.score_eps > 0.0
        && std::isfinite(config.g_clip)
        && config.g_clip > 0.0
        && std::isfinite(config.score_clip)
        && config.score_clip > 0.0
        && std::isfinite(config.fisher_floor)
        && config.fisher_floor > 0.0
        && std::isfinite(config.stationary_beta_tol)
        && config.stationary_beta_tol >= 0.0
        && config.stationary_beta_tol < 1.0
        && std::isfinite(config.optimizer_gradient_eps)
        && config.optimizer_gradient_eps > 0.0;
}

int validate_copula(const PreparedDynamicEmission& emission) {
    const CopulaSpec& copula = emission.compatibility_spec();
    if (copula.family == CopulaFamily::Student) {
        return emission.is_supported()
            ? SCAR_OK
            : SCAR_INVALID_FAMILY;
    }
    if (copula.family == CopulaFamily::EquicorrGaussian) {
        return copula.rotation == Rotation::R0
                && copula.transform == Transform::GaussianTanh
                && copula.dim >= 2
            ? SCAR_OK
            : SCAR_INVALID_FAMILY;
    }
    if (copula.dim != 2) {
        return SCAR_INVALID_FAMILY;
    }
    if (!scar::copula::is_valid_rotation(
            static_cast<int>(copula.rotation))) {
        return SCAR_INVALID_ROTATION;
    }
    if (copula.family == CopulaFamily::Gaussian
        && copula.transform != Transform::GaussianTanh) {
        return SCAR_INVALID_TRANSFORM;
    }
    if (copula.family != CopulaFamily::Independent
        && copula.family != CopulaFamily::Gaussian
        && copula.transform != Transform::Softplus
        && copula.transform != Transform::XTanh
        && copula.transform != Transform::Exponential
        && copula.transform != Transform::Logistic) {
        return SCAR_INVALID_TRANSFORM;
    }
    if (!emission.is_supported()) {
        return SCAR_INVALID_FAMILY;
    }
    return SCAR_OK;
}

int validate_inputs(
    const GasParams& params,
    const PreparedDynamicEmission& emission,
    ObservationView u,
    const GasConfig& config) {

    if (!valid_params(params) || !valid_config(config)) {
        return SCAR_INVALID_PARAMETER;
    }
    const int copula_status = validate_copula(emission);
    if (copula_status != SCAR_OK) {
        return copula_status;
    }
    return static_cast<int>(emission.validate_observations(u));
}

double initial_g(
    const GasParams& params,
    const GasConfig& config) {

    if (std::abs(params.beta) < 1.0 - config.stationary_beta_tol) {
        return params.omega / (1.0 - params.beta);
    }
    return params.omega;
}

double gas_transform(const PreparedDynamicEmission& emission, double g) {
    return emission.transform_state(g);
}

double gas_dtransform(const PreparedDynamicEmission& emission, double g) {
    return emission.dtransform_state(g);
}

double log_pdf_at_g(
    const PreparedDynamicEmission& emission,
    const double* row,
    std::int64_t row_index,
    double g,
    PreparedDynamicEmissionWorkspace& workspace) {

    return emission.log_pdf_at_state(row, row_index, g, workspace);
}

RowEvaluation evaluate_row(
    const PreparedDynamicEmission& emission,
    const double* row,
    std::int64_t row_index,
    double g,
    const GasConfig& config,
    bool need_score,
    PreparedDynamicEmissionWorkspace& workspace) {

    RowEvaluation out;
    out.r = gas_transform(emission, g);
    if (!std::isfinite(out.r)) {
        out.status = SCAR_NUMERICAL_FAILURE;
        return out;
    }

    const bool analytic_derivative = need_score;
    const DynamicEmissionRowResult emission_row =
        emission.evaluate_parameter(
            row,
            row_index,
            out.r,
            analytic_derivative,
            workspace);
    out.log_likelihood = emission_row.log_pdf;
    if (!emission_row.is_ok()) {
        out.status = static_cast<int>(emission_row.status);
        return out;
    }
    if (!need_score) {
        return out;
    }

    const double analytic_score = emission_row.dlog_dparameter
        * gas_dtransform(emission, g);
    if (config.scaling == GasScaling::Unit) {
        out.score = analytic_score;
    } else {
        const double ll_plus = log_pdf_at_g(
            emission, row, row_index, g + config.score_eps, workspace);
        const double ll_minus = log_pdf_at_g(
            emission, row, row_index, g - config.score_eps, workspace);
        if (!std::isfinite(ll_plus) || !std::isfinite(ll_minus)) {
            out.status = SCAR_NUMERICAL_FAILURE;
            return out;
        }
        const double curvature_denominator =
            config.score_eps * config.score_eps;
        const double second_derivative =
            (ll_plus - 2.0 * out.log_likelihood + ll_minus)
            / curvature_denominator;
        const double fisher =
            std::max(-second_derivative, config.fisher_floor);
        out.score = analytic_score / fisher;
    }

    if (!std::isfinite(out.score)) {
        out.status = SCAR_NUMERICAL_FAILURE;
        return out;
    }
    out.score = std::clamp(
        out.score, -config.score_clip, config.score_clip);
    return out;
}

double next_g(
    const GasParams& params,
    const GasConfig& config,
    double g,
    double score) {

    return std::clamp(
        params.omega + params.beta * g + params.gamma * score,
        -config.g_clip,
        config.g_clip);
}

template <typename Result>
void set_failure(Result& result, int status, std::int64_t index) {
    result.status = status_from_int(status);
    result.failure.index = index;
}

GasLogLikResult run_log_likelihood(
    const GasParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const GasConfig& config) {

    GasLogLikResult out;
    PreparedDynamicEmission emission =
        PreparedDynamicEmission::borrow(copula);
    const int status = validate_inputs(params, emission, u, config);
    if (status != SCAR_OK) {
        set_failure(out, status, -1);
        return out;
    }

    double g = initial_g(params, config);
    if (!std::isfinite(g)) {
        set_failure(out, SCAR_NUMERICAL_FAILURE, -1);
        return out;
    }
    PreparedDynamicEmissionWorkspace workspace =
        emission.make_workspace(true);
    for (std::size_t t = 0; t < u.n_obs; ++t) {
        const double* row = u.values == nullptr
            ? nullptr
            : u.values + static_cast<std::size_t>(u.dim) * t;
        const bool need_score = t + 1 < u.n_obs;
        const RowEvaluation evaluation = evaluate_row(
            emission,
            row,
            static_cast<std::int64_t>(t),
            g,
            config,
            need_score,
            workspace);
        if (evaluation.status != SCAR_OK) {
            set_failure(
                out, evaluation.status, static_cast<std::int64_t>(t));
            return out;
        }
        out.log_likelihood += evaluation.log_likelihood;
        if (!std::isfinite(out.log_likelihood)) {
            set_failure(
                out, SCAR_NUMERICAL_FAILURE,
                static_cast<std::int64_t>(t));
            return out;
        }
        if (need_score) {
            g = next_g(params, config, g, evaluation.score);
            if (!std::isfinite(g)) {
                set_failure(
                    out, SCAR_NUMERICAL_FAILURE,
                    static_cast<std::int64_t>(t));
                return out;
            }
        }
    }
    return out;
}

GasLogLikResult run_shrinkage_log_likelihood(
    const GasParams& params,
    const CopulaSpec& template_spec,
    std::optional<CopulaSpec>& workspace,
    DoubleView base_correlation,
    double raw_shrinkage,
    ObservationView u,
    const GasConfig& config) {

    GasLogLikResult out;
    if (!workspace) {
        auto spec = prepare_shrinkage_dynamic_spec(
            template_spec, base_correlation, raw_shrinkage);
        if (!spec.is_ok()) {
            out.status = spec.status;
            out.failure = spec.failure;
            return out;
        }
        workspace.emplace(std::move(spec.value));
    } else {
        const auto updated = update_shrinkage_dynamic_spec(
            *workspace, base_correlation, raw_shrinkage);
        if (!updated.is_ok()) {
            out.status = updated.status;
            out.failure = updated.failure;
            return out;
        }
    }
    return run_log_likelihood(params, *workspace, u, config);
}

double optimizer_gradient_step(
    const GasConfig& config,
    double value,
    double lower,
    double upper) noexcept {

    // Match SciPy's two-point provider: absolute eps points forward even
    // for negative coordinates; an explicit relative step follows sign(x).
    const double sign = value >= 0.0 ? 1.0 : -1.0;
    double step = config.optimizer_gradient_relative
        ? config.optimizer_gradient_eps * sign * std::abs(value)
        : config.optimizer_gradient_eps;
    if ((value + step) - value == 0.0) {
        step = std::sqrt(std::numeric_limits<double>::epsilon())
            * sign * std::max(1.0, std::abs(value));
    }
    const double lower_distance = value - lower;
    const double upper_distance = upper - value;
    const double shifted = value + step;
    if (shifted < lower || shifted > upper) {
        if (std::abs(step) <= std::max(lower_distance, upper_distance)) {
            step = -step;
        } else {
            step = upper_distance >= lower_distance
                ? upper_distance : -lower_distance;
        }
    }
    return step;
}

template <std::size_t Size, typename Evaluate>
GasObjectiveGradientResult optimizer_value_gradient(
    const std::array<double, Size>& values,
    const GasConfig& config,
    Evaluate evaluate) {

    GasObjectiveGradientResult out;
    const bool bounded = !config.optimizer_lower_bounds.empty()
        || !config.optimizer_upper_bounds.empty();
    if (bounded && (config.optimizer_lower_bounds.size() != Size
                    || config.optimizer_upper_bounds.size() != Size)) {
        out.status = Status::InvalidParameter;
        return out;
    }
    for (std::size_t coordinate = 0; coordinate < Size; ++coordinate) {
        if (bounded) {
            const double lower = config.optimizer_lower_bounds[coordinate];
            const double upper = config.optimizer_upper_bounds[coordinate];
            if (std::isnan(lower) || std::isnan(upper) || lower >= upper
                || values[coordinate] < lower || values[coordinate] > upper) {
                out.status = Status::InvalidParameter;
                out.failure.coordinate = static_cast<int>(coordinate);
                return out;
            }
        }
    }
    const GasLogLikResult base = evaluate(values);
    ++out.objective_evaluations;
    if (!base.is_ok()) {
        out.status = base.status;
        out.failure = base.failure;
        return out;
    }
    out.objective = -base.log_likelihood;
    out.gradient.assign(Size, 0.0);
    for (std::size_t coordinate = 0; coordinate < Size; ++coordinate) {
        const double lower = bounded ? config.optimizer_lower_bounds[coordinate]
            : -std::numeric_limits<double>::infinity();
        const double upper = bounded ? config.optimizer_upper_bounds[coordinate]
            : std::numeric_limits<double>::infinity();
        auto shifted = values;
        shifted[coordinate] += optimizer_gradient_step(
            config, values[coordinate], lower, upper);
        // Floating-point addition determines the actual representable step.
        const double delta = shifted[coordinate] - values[coordinate];
        if (!std::isfinite(delta) || delta == 0.0) {
            out.status = Status::NumericalFailure;
            out.failure.coordinate = static_cast<int>(coordinate);
            out.gradient.clear();
            return out;
        }
        const GasLogLikResult moved = evaluate(shifted);
        ++out.objective_evaluations;
        if (!moved.is_ok()) {
            out.status = moved.status;
            out.failure = moved.failure;
            out.failure.coordinate = static_cast<int>(coordinate);
            out.gradient.clear();
            return out;
        }
        out.gradient[coordinate] = -(
            moved.log_likelihood - base.log_likelihood) / delta;
        if (!std::isfinite(out.gradient[coordinate])) {
            out.status = Status::NumericalFailure;
            out.failure.coordinate = static_cast<int>(coordinate);
            out.gradient.clear();
            return out;
        }
    }
    return out;
}

}  // namespace

GasStateResult GasEvaluator::initial_state(
    const GasParams& params,
    const CopulaSpec& copula,
    const GasConfig& config) const {

    PreparedDynamicEmission emission =
        PreparedDynamicEmission::borrow(copula);
    return initial_state_prepared(params, emission, config);
}

GasStateResult GasEvaluator::initial_state_prepared(
    const GasParams& params,
    const PreparedDynamicEmission& emission,
    const GasConfig& config) const {

    GasStateResult out;
    if (!valid_params(params) || !valid_config(config)) {
        out.status = Status::InvalidParameter;
        return out;
    }
    out.status = status_from_int(validate_copula(emission));
    if (!out.is_ok()) {
        return out;
    }
    out.g = initial_g(params, config);
    out.parameter = gas_transform(emission, out.g);
    if (!std::isfinite(out.g) || !std::isfinite(out.parameter)) {
        out.status = Status::NumericalFailure;
    }
    return out;
}

GasFilterResult GasEvaluator::filter(
    const GasParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const GasConfig& config) const {

    GasFilterResult out;
    PreparedDynamicEmission emission =
        PreparedDynamicEmission::borrow(copula);
    const int status = validate_inputs(params, emission, u, config);
    if (status != SCAR_OK) {
        set_failure(out, status, -1);
        return out;
    }

    out.g_path.resize(u.n_obs);
    out.r_path.resize(u.n_obs);
    out.score_path.resize(u.n_obs - 1);

    double g = initial_g(params, config);
    if (!std::isfinite(g)) {
        set_failure(out, SCAR_NUMERICAL_FAILURE, -1);
        return out;
    }
    PreparedDynamicEmissionWorkspace workspace =
        emission.make_workspace(true);
    for (std::size_t t = 0; t < u.n_obs; ++t) {
        out.g_path[t] = g;
        const double* row = u.values == nullptr
            ? nullptr
            : u.values + static_cast<std::size_t>(u.dim) * t;
        const bool need_score = t + 1 < u.n_obs;
        const RowEvaluation evaluation = evaluate_row(
            emission,
            row,
            static_cast<std::int64_t>(t),
            g,
            config,
            need_score,
            workspace);
        out.r_path[t] = evaluation.r;
        if (evaluation.status != SCAR_OK) {
            set_failure(
                out, evaluation.status, static_cast<std::int64_t>(t));
            return out;
        }
        out.log_likelihood += evaluation.log_likelihood;
        if (!std::isfinite(out.log_likelihood)) {
            set_failure(
                out, SCAR_NUMERICAL_FAILURE,
                static_cast<std::int64_t>(t));
            return out;
        }
        if (need_score) {
            out.score_path[t] = evaluation.score;
            g = next_g(params, config, g, evaluation.score);
            if (!std::isfinite(g)) {
                set_failure(
                    out, SCAR_NUMERICAL_FAILURE,
                    static_cast<std::int64_t>(t));
                return out;
            }
        }
    }
    return out;
}

GasLogLikResult GasEvaluator::log_likelihood(
    const GasParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const GasConfig& config) const {

    return run_log_likelihood(params, copula, u, config);
}

GasLogLikResult GasEvaluator::negative_log_likelihood(
    const GasParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const GasConfig& config) const {

    GasLogLikResult out = run_log_likelihood(
        params, copula, u, config);
    if (out.is_ok()) {
        out.log_likelihood = -out.log_likelihood;
    }
    return out;
}

GasObjectiveGradientResult
GasEvaluator::negative_log_likelihood_and_gradient(
    const GasParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const GasConfig& config) const {

    return optimizer_value_gradient(
        std::array<double, 3>{params.omega, params.gamma, params.beta},
        config,
        [&](const std::array<double, 3>& values) {
            const GasParams point{values[0], values[1], values[2]};
            return run_log_likelihood(point, copula, u, config);
        });
}

GasObjectiveGradientResult
GasEvaluator::negative_log_likelihood_and_gradient_shrinkage(
    const GasParams& params,
    const CopulaSpec& copula,
    DoubleView base_correlation,
    double raw_shrinkage,
    ObservationView u,
    const GasConfig& config) const {

    // Own one copy of the immutable PPF table for all finite-difference
    // evaluations in this call. Only its small correlation payload changes.
    // Lazy preparation preserves optimizer validation and failure ordering.
    std::optional<CopulaSpec> workspace;
    return optimizer_value_gradient(
        std::array<double, 4>{
            params.omega, params.gamma, params.beta, raw_shrinkage},
        config,
        [&](const std::array<double, 4>& values) {
            const GasParams point{values[0], values[1], values[2]};
            return run_shrinkage_log_likelihood(
                point, copula, workspace, base_correlation, values[3], u, config);
        });
}


GasUpdateResult GasEvaluator::update_one(
    const GasParams& params,
    const CopulaSpec& copula,
    double g,
    double u1,
    double u2,
    const GasConfig& config) const {

    const double values[2] = {u1, u2};
    return update_observation(
        params, copula, g, {values, 1, 2}, config);
}

GasUpdateResult GasEvaluator::update_one_prepared(
    const GasParams& params,
    const PreparedDynamicEmission& emission,
    PreparedDynamicEmissionWorkspace& workspace,
    double g,
    double u1,
    double u2,
    const GasConfig& config) const {

    const double values[2] = {u1, u2};
    return update_observation_prepared(
        params, emission, workspace, g, {values, 1, 2}, config);
}

GasUpdateResult GasEvaluator::update_observation(
    const GasParams& params,
    const CopulaSpec& copula,
    double g,
    ObservationView observation,
    const GasConfig& config) const {

    PreparedDynamicEmission emission =
        PreparedDynamicEmission::borrow(copula);
    PreparedDynamicEmissionWorkspace workspace =
        emission.make_workspace(true);
    return update_observation_prepared(
        params, emission, workspace, g, observation, config);
}

GasUpdateResult GasEvaluator::update_observation_prepared(
    const GasParams& params,
    const PreparedDynamicEmission& emission,
    PreparedDynamicEmissionWorkspace& workspace,
    double g,
    ObservationView observation,
    const GasConfig& config) const {

    GasUpdateResult out;
    if (!valid_params(params) || !valid_config(config)
        || !std::isfinite(g)) {
        out.status = Status::InvalidParameter;
        return out;
    }
    out.status = status_from_int(validate_copula(emission));
    if (!out.is_ok()) {
        return out;
    }
    if (observation.values == nullptr
        || observation.n_obs != 1
        || observation.dim != emission.expected_dimension()) {
        out.status = Status::InvalidSize;
        return out;
    }
    out.status = emission.validate_observations(observation);
    if (!out.is_ok()) {
        return out;
    }

    const RowEvaluation evaluation = evaluate_row(
        emission, observation.values, 0, g, config, true, workspace);
    out.status = status_from_int(evaluation.status);
    out.r = evaluation.r;
    out.log_likelihood = evaluation.log_likelihood;
    out.score = evaluation.score;
    if (!out.is_ok()) {
        return out;
    }
    out.g_next = next_g(params, config, g, evaluation.score);
    if (!std::isfinite(out.g_next)) {
        out.status = Status::NumericalFailure;
        return out;
    }
    out.r_next = gas_transform(emission, out.g_next);
    if (!std::isfinite(out.r_next)) {
        out.status = Status::NumericalFailure;
    }
    return out;
}

GasPredictResult GasEvaluator::predict_parameter(
    const GasParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const GasConfig& config,
    bool horizon_next) const {

    GasPredictResult out;
    PreparedDynamicEmission emission =
        PreparedDynamicEmission::borrow(copula);
    const int status = validate_inputs(params, emission, u, config);
    if (status != SCAR_OK) {
        set_failure(out, status, -1);
        return out;
    }

    double g = initial_g(params, config);
    if (!std::isfinite(g)) {
        set_failure(out, SCAR_NUMERICAL_FAILURE, -1);
        return out;
    }
    PreparedDynamicEmissionWorkspace workspace =
        emission.make_workspace(true);
    for (std::size_t t = 0; t < u.n_obs; ++t) {
        const double* row = u.values == nullptr
            ? nullptr
            : u.values + static_cast<std::size_t>(u.dim) * t;
        const bool final_row = t + 1 == u.n_obs;
        const bool need_score = !final_row || horizon_next;
        const RowEvaluation evaluation = evaluate_row(
            emission,
            row,
            static_cast<std::int64_t>(t),
            g,
            config,
            need_score,
            workspace);
        if (evaluation.status != SCAR_OK) {
            set_failure(
                out, evaluation.status, static_cast<std::int64_t>(t));
            return out;
        }
        if (final_row) {
            out.parameter = horizon_next
                ? gas_transform(
                    emission,
                    next_g(params, config, g, evaluation.score))
                : evaluation.r;
            if (!std::isfinite(out.parameter)) {
                set_failure(
                    out,
                    SCAR_NUMERICAL_FAILURE,
                    static_cast<std::int64_t>(t));
            }
            return out;
        }
        g = next_g(params, config, g, evaluation.score);
        if (!std::isfinite(g)) {
            set_failure(
                out, SCAR_NUMERICAL_FAILURE,
                static_cast<std::int64_t>(t));
            return out;
        }
    }
    return out;
}

GasSampleResult GasEvaluator::sample_bivariate(
    const GasParams& params,
    const CopulaSpec& copula,
    ObservationView draws,
    const GasConfig& config) const {

    GasSampleResult out;
    out.n_rows = static_cast<std::int64_t>(draws.n_obs);
    PreparedDynamicEmission emission =
        PreparedDynamicEmission::borrow(copula);
    if (!valid_params(params) || !valid_config(config)) {
        set_failure(out, SCAR_INVALID_PARAMETER, -1);
        return out;
    }
    const int copula_status = validate_copula(emission);
    if (copula_status != SCAR_OK) {
        set_failure(out, copula_status, -1);
        return out;
    }
    if (emission.expected_dimension() != 2
        || copula.family == CopulaFamily::Student
        || copula.family == CopulaFamily::EquicorrGaussian
        || copula.family == CopulaFamily::MultivariateGaussian) {
        set_failure(out, SCAR_INVALID_FAMILY, -1);
        return out;
    }
    if (draws.values == nullptr || draws.n_obs == 0 || draws.dim != 2) {
        set_failure(out, SCAR_INVALID_SIZE, -1);
        return out;
    }

    const CopulaSpec transposed_copula =
        scar_internal::transposed_copula_spec(copula);
    const PreparedPairKernel sampling_kernel(transposed_copula);
    if (!sampling_kernel.is_supported()) {
        set_failure(out, SCAR_INVALID_FAMILY, -1);
        return out;
    }

    for (std::size_t t = 0; t < draws.n_obs; ++t) {
        const double* draw = draws.values + 2 * t;
        if (!std::isfinite(draw[0]) || !std::isfinite(draw[1])
            || (copula.family != CopulaFamily::Gaussian
                && (!(draw[0] > 0.0 && draw[0] < 1.0)
                    || !(draw[1] > 0.0 && draw[1] < 1.0)))) {
            set_failure(
                out, SCAR_INVALID_PARAMETER,
                static_cast<std::int64_t>(t));
            return out;
        }
    }
    double g = initial_g(params, config);
    if (!std::isfinite(g)) {
        set_failure(out, SCAR_NUMERICAL_FAILURE, -1);
        return out;
    }
    out.values.resize(2 * draws.n_obs);
    PreparedDynamicEmissionWorkspace workspace =
        emission.make_workspace(true);
    for (std::size_t t = 0; t < draws.n_obs; ++t) {
        const double* draw = draws.values + 2 * t;
        const double parameter = gas_transform(emission, g);
        double sample[2] = {0.0, 0.0};
        if (copula.family == CopulaFamily::Gaussian) {
            const double residual_scale = std::sqrt(
                std::max(0.0, 1.0 - parameter * parameter));
            sample[0] = math::normal_cdf(draw[0]);
            sample[1] = math::normal_cdf(
                parameter * draw[0] + residual_scale * draw[1]);
        } else {
            sample[0] = draw[0];
            sample[1] = sampling_kernel.inverse_h(
                draw[1], draw[0], parameter);
        }
        if (!std::isfinite(sample[0]) || !std::isfinite(sample[1])
            || !(sample[0] > 0.0 && sample[0] < 1.0)
            || !(sample[1] > 0.0 && sample[1] < 1.0)) {
            set_failure(
                out, SCAR_NUMERICAL_FAILURE,
                static_cast<std::int64_t>(t));
            return out;
        }
        out.values[2 * t] = sample[0];
        out.values[2 * t + 1] = sample[1];

        if (t + 1 < draws.n_obs) {
            const RowEvaluation evaluation = evaluate_row(
                emission,
                sample,
                static_cast<std::int64_t>(t),
                g,
                config,
                true,
                workspace);
            if (evaluation.status != SCAR_OK) {
                set_failure(
                    out, evaluation.status,
                    static_cast<std::int64_t>(t));
                return out;
            }
            g = next_g(params, config, g, evaluation.score);
            if (!std::isfinite(g)) {
                set_failure(
                    out, SCAR_NUMERICAL_FAILURE,
                    static_cast<std::int64_t>(t));
                return out;
            }
        }
    }
    return out;
}

GasPathResult GasEvaluator::h_path(
    const GasParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const GasConfig& config) const {

    GasPathResult out;
    PreparedDynamicEmission emission =
        PreparedDynamicEmission::borrow(copula);
    if (emission.expected_dimension() != 2
        || copula.family == CopulaFamily::Student
        || copula.family == CopulaFamily::EquicorrGaussian) {
        set_failure(out, SCAR_INVALID_FAMILY, -1);
        return out;
    }
    const GasFilterResult filtered = filter(
        params, copula, u, config);
    out.status = filtered.status;
    out.failure = filtered.failure;
    if (!out.is_ok()) {
        return out;
    }

    out.values.resize(u.n_obs);
    const CopulaSpec transposed_copula =
        scar_internal::transposed_copula_spec(copula);
    PreparedDynamicEmission transposed_emission =
        PreparedDynamicEmission::borrow(transposed_copula);
    for (std::size_t t = 0; t < u.n_obs; ++t) {
        const double* row = u.values + 2 * t;
        const double value = transposed_emission.h(
            row[1], row[0], filtered.r_path[t]);
        if (!std::isfinite(value)) {
            set_failure(
                out, SCAR_NUMERICAL_FAILURE,
                static_cast<std::int64_t>(t));
            return out;
        }
        out.values[t] = std::clamp(
            value,
            scar_internal::kHEps,
            1.0 - scar_internal::kHEps);
    }
    return out;
}

GasOuInitializationResult GasEvaluator::ou_initial_point(
    double static_mu,
    const CopulaSpec& copula,
    ObservationView u,
    const GasConfig& config) const {
    GasOuInitializationResult out;
    if (!std::isfinite(static_mu) || u.n_obs < 2) {
        out.status = Status::InvalidParameter;
        return out;
    }
    constexpr double betas[] = {0.90, 0.95, 0.98, 0.99};
    constexpr double gammas[] = {0.01, 0.05, 0.1, 0.3, 0.5};
    double best = -1e10;
    for (const double beta : betas) {
        for (const double gamma : gammas) {
            const GasParams candidate{
                static_mu * (1.0 - beta), gamma, beta};
            const GasLogLikResult evaluated = log_likelihood(
                candidate, copula, u, config);
            if (evaluated.is_ok() && evaluated.log_likelihood > best) {
                best = evaluated.log_likelihood;
                out.selected_omega = candidate.omega;
                out.selected_gamma = candidate.gamma;
                out.selected_beta = candidate.beta;
                out.grid_candidate_found = true;
            }
        }
    }
    if (!out.grid_candidate_found) {
        out.mu = static_mu;
        return out;
    }
    const GasFilterResult filtered = filter(
        GasParams{out.selected_omega, out.selected_gamma, out.selected_beta},
        copula, u, config);
    if (!filtered.is_ok()) {
        out.status = filtered.status;
        out.failure = filtered.failure;
        return out;
    }
    out.best_log_likelihood = best;
    long double sum = 0.0L;
    for (const double value : filtered.g_path) sum += value;
    out.mu = static_cast<double>(
        sum / static_cast<long double>(filtered.g_path.size()));
    long double variance_sum = 0.0L;
    for (const double value : filtered.g_path) {
        const long double centered =
            static_cast<long double>(value) - out.mu;
        variance_sum += centered * centered;
    }
    const double variance = static_cast<double>(
        variance_sum / static_cast<long double>(filtered.g_path.size()));
    if (variance < 1e-10) {
        return out;
    }
    long double covariance_sum = 0.0L;
    for (std::size_t index = 0;
         index + 1 < filtered.g_path.size(); ++index) {
        covariance_sum +=
            (static_cast<long double>(filtered.g_path[index]) - out.mu)
            * (static_cast<long double>(filtered.g_path[index + 1]) - out.mu);
    }
    const double covariance = static_cast<double>(
        covariance_sum
        / static_cast<long double>(filtered.g_path.size() - 1));
    const double autocorrelation = std::clamp(
        covariance / variance, 0.01, 0.999);
    const double dt =
        1.0 / static_cast<double>(u.n_obs - 1);
    out.kappa = std::clamp(
        -std::log(autocorrelation) / dt, 0.01, 100.0);
    out.nu = std::clamp(
        std::sqrt(2.0 * out.kappa * variance), 0.01, 50.0);
    return out;
}

}  // namespace scar
