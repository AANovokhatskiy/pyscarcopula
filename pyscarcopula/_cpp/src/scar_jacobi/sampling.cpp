#include "scar/jacobi.hpp"

#include "scar/copula/prepared_pair_kernel.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <vector>

namespace scar {
namespace {

template <typename ResultType>
ResultType failure(Status status, int operation = -1) {
    ResultType result;
    result.status = status;
    result.failure.operation = operation;
    return result;
}

bool valid_draw(double value) noexcept {
    return std::isfinite(value) && value >= 0.0 && value < 1.0;
}

std::size_t select_index(
    const double* probabilities,
    std::size_t count,
    double total,
    double draw) noexcept {

    double cumulative = 0.0;
    for (std::size_t index = 0; index < count; ++index) {
        cumulative += probabilities[index];
        // Divide the CDF, not the draw: multiplying a draw by a subnormal
        // total can round across an atom boundary and bias the sample.
        if (draw < cumulative / total) {
            return index;
        }
    }
    return count - 1;
}

JacobiTrajectoryResult sample_stationary_only(
    const JacobiParams& params,
    const JacobiTransitionConfig& config,
    double uniform) {

    JacobiTrajectoryResult result;
    result.value.transition.method_requested = config.method;
    result.value.transition.method_used = config.method;
    result.value.transition.storage = config.storage;
    result.value.transition.correction = config.correction;
    result.value.transition.memory_budget_bytes =
        config.numerical.memory_budget_bytes;
    if (!valid_draw(uniform)) {
        result.status = Status::InvalidParameter;
        result.failure.index = 0;
        return result;
    }

    std::vector<double> tau;
    std::vector<double> weights;
    if (config.method == JacobiTransitionMethod::LocalFixed) {
        const JacobiFixedRuleResult rule = build_fixed_jacobi_rule(
            params,
            config.numerical.quad_order,
            config.numerical.memory_budget_bytes);
        if (!rule.is_ok()) {
            result.status = rule.status;
            result.failure = rule.failure;
            return result;
        }
        tau = rule.value.tau;
        weights = rule.value.weights;
    } else {
        const JacobiShapeResult shape = jacobi_stationary_shape(params);
        if (!shape.is_ok()) {
            result.status = shape.status;
            result.failure = shape.failure;
            return result;
        }
        const JacobiBasisResult rule = build_jacobi_rule(
            shape.value.alpha,
            shape.value.beta,
            config.numerical.quad_order,
            1,
            config.numerical.memory_budget_bytes);
        if (!rule.is_ok()) {
            result.status = rule.status;
            result.failure = rule.failure;
            return result;
        }
        tau = rule.value.tau;
        weights = rule.value.weights;
    }
    const JacobiScalarResult mass = validate_jacobi_state_distribution(
        tau, weights);
    if (!mass.is_ok()) {
        result.status = Status::NumericalFailure;
        return result;
    }
    const std::size_t index = select_index(
        weights.data(), weights.size(), mass.value, uniform);
    result.value.tau = {tau[index]};
    result.value.draws_used = 1;
    return result;
}

JacobiTrajectoryResult sample_dense(
    const JacobiDenseTransition& transition,
    const std::vector<double>& uniforms) {

    JacobiTrajectoryResult result;
    result.value.transition = transition.diagnostics;
    const std::size_t order = transition.order > 0
        ? static_cast<std::size_t>(transition.order) : 0;
    if (order == 0
        || transition.tau.size() != order
        || transition.weights.size() != order
        || transition.probabilities.size() != order * order) {
        result.status = Status::InvalidSize;
        return result;
    }
    const JacobiScalarResult mass = validate_jacobi_state_distribution(
        transition.tau, transition.weights);
    if (!mass.is_ok()) {
        result.status = Status::NumericalFailure;
        return result;
    }
    for (std::size_t draw = 0; draw < uniforms.size(); ++draw) {
        if (!valid_draw(uniforms[draw])) {
            result.status = Status::InvalidParameter;
            result.failure.index = static_cast<std::int64_t>(draw);
            return result;
        }
    }

    result.value.tau.resize(uniforms.size());
    std::size_t state = select_index(
        transition.weights.data(), order, mass.value, uniforms[0]);
    result.value.tau[0] = transition.tau[state];
    for (std::size_t observation = 1;
         observation < uniforms.size(); ++observation) {
        const double* row = transition.probabilities.data() + state * order;
        double row_total = 0.0;
        for (std::size_t column = 0; column < order; ++column) {
            if (!std::isfinite(row[column]) || row[column] < 0.0) {
                result.status = Status::NumericalFailure;
                result.failure.row = static_cast<std::int64_t>(state);
                result.value.tau.clear();
                return result;
            }
            row_total += row[column];
        }
        if (!std::isfinite(row_total) || row_total <= 0.0) {
            result.status = Status::NumericalFailure;
            result.failure.row = static_cast<std::int64_t>(state);
            result.value.tau.clear();
            return result;
        }
        state = select_index(row, order, row_total, uniforms[observation]);
        result.value.tau[observation] = transition.tau[state];
    }
    result.value.draws_used = static_cast<std::int64_t>(uniforms.size());
    return result;
}

JacobiTrajectoryResult sample_sparse(
    const JacobiSparseTransition& transition,
    const std::vector<double>& uniforms) {

    JacobiTrajectoryResult result;
    result.value.transition = transition.diagnostics;
    const std::size_t order = transition.order > 0
        ? static_cast<std::size_t>(transition.order) : 0;
    const std::size_t width = transition.max_width > 0
        ? static_cast<std::size_t>(transition.max_width) : 0;
    if (order == 0 || width == 0
        || transition.tau.size() != order
        || transition.weights.size() != order
        || transition.counts.size() != order
        || transition.indices.size() != order * width
        || transition.probabilities.size() != order * width) {
        result.status = Status::InvalidSize;
        return result;
    }
    const JacobiScalarResult mass = validate_jacobi_state_distribution(
        transition.tau, transition.weights);
    if (!mass.is_ok()) {
        result.status = Status::NumericalFailure;
        return result;
    }
    for (std::size_t draw = 0; draw < uniforms.size(); ++draw) {
        if (!valid_draw(uniforms[draw])) {
            result.status = Status::InvalidParameter;
            result.failure.index = static_cast<std::int64_t>(draw);
            return result;
        }
    }

    result.value.tau.resize(uniforms.size());
    std::size_t state = select_index(
        transition.weights.data(), order, mass.value, uniforms[0]);
    result.value.tau[0] = transition.tau[state];
    for (std::size_t observation = 1;
         observation < uniforms.size(); ++observation) {
        const std::int64_t raw_count = transition.counts[state];
        if (raw_count <= 0
            || raw_count > static_cast<std::int64_t>(width)) {
            result.status = Status::NumericalFailure;
            result.failure.row = static_cast<std::int64_t>(state);
            result.value.tau.clear();
            return result;
        }
        const std::size_t count = static_cast<std::size_t>(raw_count);
        const std::size_t offset = state * width;
        double row_total = 0.0;
        for (std::size_t slot = 0; slot < count; ++slot) {
            const double probability = transition.probabilities[offset + slot];
            const std::int64_t index = transition.indices[offset + slot];
            if (!std::isfinite(probability) || probability < 0.0
                || index < 0 || index >= static_cast<std::int64_t>(order)) {
                result.status = Status::NumericalFailure;
                result.failure.row = static_cast<std::int64_t>(state);
                result.value.tau.clear();
                return result;
            }
            row_total += probability;
        }
        if (!std::isfinite(row_total) || row_total <= 0.0) {
            result.status = Status::NumericalFailure;
            result.failure.row = static_cast<std::int64_t>(state);
            result.value.tau.clear();
            return result;
        }
        const std::size_t slot = select_index(
            transition.probabilities.data() + offset,
            count,
            row_total,
            uniforms[observation]);
        state = static_cast<std::size_t>(transition.indices[offset + slot]);
        result.value.tau[observation] = transition.tau[state];
    }
    result.value.draws_used = static_cast<std::int64_t>(uniforms.size());
    return result;
}

}  // namespace

JacobiTrajectoryResult sample_prepared_jacobi_sparse_trajectory(
    const std::vector<double>& tau,
    const std::vector<double>& weights,
    const JacobiSparseTransition& transition,
    const std::vector<double>& uniforms) {

    if (uniforms.empty()) {
        return JacobiTrajectoryResult{};
    }
    try {
        JacobiSparseTransition prepared = transition;
        prepared.tau = tau;
        prepared.weights = weights;
        return sample_sparse(prepared, uniforms);
    } catch (const std::bad_alloc&) {
        return failure<JacobiTrajectoryResult>(Status::InvalidSize);
    }
}

JacobiTrajectoryResult sample_jacobi_grid_trajectory(
    const JacobiParams& params,
    const JacobiTransitionConfig& requested_config,
    const std::vector<double>& uniforms) {

    if (uniforms.empty()) {
        JacobiTrajectoryResult result;
        result.value.transition.method_requested = requested_config.method;
        result.value.transition.method_used = requested_config.method;
        result.value.transition.storage = requested_config.storage;
        result.value.transition.correction = requested_config.correction;
        result.value.transition.memory_budget_bytes =
            requested_config.numerical.memory_budget_bytes;
        return result;
    }
    if (requested_config.numerical.n_obs
        != static_cast<std::int64_t>(uniforms.size())) {
        return failure<JacobiTrajectoryResult>(Status::InvalidSize);
    }
    if (!ok(validate_jacobi_params(
            params, requested_config.numerical.stationary_shape_max))) {
        return failure<JacobiTrajectoryResult>(Status::InvalidParameter);
    }

    try {
        JacobiTransitionConfig config = requested_config;
        const JacobiTransitionMethod model_method = config.method;
        if (config.method == JacobiTransitionMethod::SpectralCoeff) {
            config.method = JacobiTransitionMethod::Auto;
        }
        if (uniforms.size() == 1) {
            JacobiTrajectoryResult result = sample_stationary_only(
                params, config, uniforms[0]);
            result.value.transition.method_requested = model_method;
            return result;
        }

        if (config.storage == JacobiTransitionStorage::Sparse) {
            const JacobiSparseTransitionResult transition =
                build_jacobi_sparse_transition(params, config);
            if (!transition.is_ok()) {
                JacobiTrajectoryResult result = failure<JacobiTrajectoryResult>(
                    transition.status, transition.failure.operation);
                result.failure = transition.failure;
                result.value.transition = transition.value.diagnostics;
                return result;
            }
            JacobiTrajectoryResult result = sample_sparse(
                transition.value, uniforms);
            result.value.transition.method_requested = model_method;
            return result;
        }

        const JacobiDenseTransitionResult transition =
            build_jacobi_dense_transition(params, config);
        if (!transition.is_ok()) {
            JacobiTrajectoryResult result = failure<JacobiTrajectoryResult>(
                transition.status, transition.failure.operation);
            result.failure = transition.failure;
            result.value.transition = transition.value.diagnostics;
            return result;
        }
        JacobiTrajectoryResult result = sample_dense(
            transition.value, uniforms);
        result.value.transition.method_requested = model_method;
        return result;
    } catch (const std::bad_alloc&) {
        return failure<JacobiTrajectoryResult>(Status::InvalidSize);
    }
}

JacobiTrajectoryResult sample_jacobi_lamperti_chunk(
    const JacobiParams& params,
    const JacobiLampertiSamplingConfig& config,
    double initial_lamperti_value,
    const std::vector<double>& normal_draws) {

    if (!ok(validate_jacobi_params(
            params, std::numeric_limits<double>::infinity()))
        || config.n_obs < 2
        || config.substeps <= 0
        || !std::isfinite(config.interior_eps)
        || !(config.interior_eps > 0.0 && config.interior_eps < 0.5)
        || (config.boundary != JacobiBoundaryPolicy::Reflect
            && config.boundary != JacobiBoundaryPolicy::Clip)
        || !std::isfinite(initial_lamperti_value)
        || normal_draws.size() % static_cast<std::size_t>(config.substeps) != 0) {
        return failure<JacobiTrajectoryResult>(Status::InvalidParameter);
    }
    const double upper = std::acos(-1.0) / params.xi;
    if (initial_lamperti_value < 0.0
        || initial_lamperti_value > upper) {
        return failure<JacobiTrajectoryResult>(Status::InvalidParameter);
    }
    const std::size_t intervals =
        normal_draws.size() / static_cast<std::size_t>(config.substeps);
    if (intervals > static_cast<std::size_t>(config.n_obs - 1)) {
        return failure<JacobiTrajectoryResult>(Status::InvalidSize);
    }

    try {
        JacobiTrajectoryResult result;
        result.value.tau.resize(intervals);
        const double h = 1.0 / (
            static_cast<double>(config.n_obs - 1)
            * static_cast<double>(config.substeps));
        const double sqrt_h = std::sqrt(h);
        double y = initial_lamperti_value;
        std::int64_t interventions = 0;
        std::size_t draw = 0;
        for (std::size_t row = 0; row < intervals; ++row) {
            for (int substep = 0; substep < config.substeps; ++substep) {
                const double innovation = normal_draws[draw];
                if (!std::isfinite(innovation)) {
                    result.status = Status::InvalidParameter;
                    result.failure.index = static_cast<std::int64_t>(draw);
                    result.value.tau.clear();
                    return result;
                }
                const double sine = std::sin(0.5 * params.xi * y);
                const double tau = sine * sine;
                const JacobiScalarResult drift = jacobi_lamperti_drift(
                    params, tau, config.interior_eps);
                if (!drift.is_ok()) {
                    result.status = drift.status;
                    result.failure.index = static_cast<std::int64_t>(draw);
                    result.value.tau.clear();
                    return result;
                }
                const double candidate =
                    y + drift.value * h + sqrt_h * innovation;
                if (!std::isfinite(candidate)) {
                    result.status = Status::NumericalFailure;
                    result.failure.index = static_cast<std::int64_t>(draw);
                    result.value.tau.clear();
                    return result;
                }
                const JacobiBoundaryResult bounded = apply_jacobi_boundary(
                    candidate, upper, config.boundary);
                if (!bounded.is_ok()) {
                    result.status = bounded.status;
                    result.failure.index = static_cast<std::int64_t>(draw);
                    result.value.tau.clear();
                    return result;
                }
                interventions += bounded.value.intervened ? 1 : 0;
                y = bounded.value.value;
                ++draw;
            }
            const double sine = std::sin(0.5 * params.xi * y);
            result.value.tau[row] = sine * sine;
        }
        result.value.final_lamperti_value = y;
        result.value.normal_draws_used = static_cast<std::int64_t>(draw);
        result.value.euler_steps = static_cast<std::int64_t>(draw);
        result.value.boundary_interventions = interventions;
        return result;
    } catch (const std::bad_alloc&) {
        return failure<JacobiTrajectoryResult>(Status::InvalidSize);
    }
}

JacobiStateSampleResult sample_jacobi_state_distribution(
    const CopulaSpec& copula,
    const std::vector<double>& tau,
    const std::vector<double>& probability,
    const std::vector<double>& selection_draws,
    const std::vector<double>& jitter_draws,
    JacobiStateSamplingMode mode,
    double theta_cap) {

    const PreparedPairKernel kernel(copula);
    if (!kernel.is_supported()
        || tau.empty()
        || tau.size() != probability.size()
        || (mode != JacobiStateSamplingMode::Grid
            && mode != JacobiStateSamplingMode::Histogram)
        || (!std::isnan(theta_cap)
            && (!std::isfinite(theta_cap) || theta_cap <= 0.0))) {
        return failure<JacobiStateSampleResult>(Status::InvalidParameter);
    }
    const bool needs_jitter =
        mode == JacobiStateSamplingMode::Histogram && tau.size() > 1;
    if ((!needs_jitter && !jitter_draws.empty())
        || (needs_jitter && jitter_draws.size() != selection_draws.size())) {
        return failure<JacobiStateSampleResult>(Status::InvalidSize);
    }
    const JacobiScalarResult mass = validate_jacobi_state_distribution(
        tau, probability);
    if (!mass.is_ok()) {
        JacobiStateSampleResult result = failure<JacobiStateSampleResult>(
            mass.status);
        result.failure = mass.failure;
        return result;
    }
    for (std::size_t index = 0; index < selection_draws.size(); ++index) {
        if (!valid_draw(selection_draws[index])
            || (needs_jitter && !valid_draw(jitter_draws[index]))) {
            JacobiStateSampleResult result = failure<JacobiStateSampleResult>(
                Status::InvalidParameter);
            result.failure.index = static_cast<std::int64_t>(index);
            return result;
        }
    }

    try {
        JacobiStateSampleResult result;
        result.value.tau.resize(selection_draws.size());
        result.value.parameters.resize(selection_draws.size());
        for (std::size_t draw = 0; draw < selection_draws.size(); ++draw) {
            const std::size_t index = select_index(
                probability.data(),
                probability.size(),
                mass.value,
                selection_draws[draw]);
            double sampled_tau = tau[index];
            if (needs_jitter) {
                const double left = index == 0
                    ? tau[0] : 0.5 * (tau[index - 1] + tau[index]);
                const double right = index + 1 == tau.size()
                    ? tau.back() : 0.5 * (tau[index] + tau[index + 1]);
                sampled_tau = left + (right - left) * jitter_draws[draw];
            }
            double parameter = kernel.tau_to_parameter(sampled_tau);
            if (!std::isnan(theta_cap)) {
                parameter = std::min(parameter, theta_cap);
            }
            if (!std::isfinite(parameter)) {
                result.status = Status::NumericalFailure;
                result.failure.index = static_cast<std::int64_t>(draw);
                result.value.tau.clear();
                result.value.parameters.clear();
                return result;
            }
            result.value.tau[draw] = sampled_tau;
            result.value.parameters[draw] = parameter;
        }
        result.value.selection_draws_used =
            static_cast<std::int64_t>(selection_draws.size());
        result.value.jitter_draws_used =
            static_cast<std::int64_t>(jitter_draws.size());
        return result;
    } catch (const std::bad_alloc&) {
        return failure<JacobiStateSampleResult>(Status::InvalidSize);
    }
}

JacobiHistogramCellsResult jacobi_state_histogram_cells(
    const std::vector<double>& tau,
    const std::vector<std::int64_t>& indices) {

    if (tau.empty()) {
        return failure<JacobiHistogramCellsResult>(Status::InvalidSize);
    }
    for (std::size_t index = 0; index < tau.size(); ++index) {
        if (!std::isfinite(tau[index])
            || (index > 0 && !(tau[index] > tau[index - 1]))) {
            JacobiHistogramCellsResult result =
                failure<JacobiHistogramCellsResult>(Status::InvalidParameter);
            result.failure.index = static_cast<std::int64_t>(index);
            return result;
        }
    }
    try {
        JacobiHistogramCellsResult result;
        result.value.left.resize(indices.size());
        result.value.right.resize(indices.size());
        for (std::size_t draw = 0; draw < indices.size(); ++draw) {
            const std::int64_t raw_index = indices[draw];
            if (raw_index < 0
                || raw_index >= static_cast<std::int64_t>(tau.size())) {
                result.status = Status::InvalidParameter;
                result.failure.index = static_cast<std::int64_t>(draw);
                result.value.left.clear();
                result.value.right.clear();
                return result;
            }
            const std::size_t index = static_cast<std::size_t>(raw_index);
            result.value.left[draw] = index == 0
                ? tau[0] : 0.5 * (tau[index - 1] + tau[index]);
            result.value.right[draw] = index + 1 == tau.size()
                ? tau.back() : 0.5 * (tau[index] + tau[index + 1]);
        }
        return result;
    } catch (const std::bad_alloc&) {
        return failure<JacobiHistogramCellsResult>(Status::InvalidSize);
    }
}

}  // namespace scar
