#include "scar/scar_ou/sampling.hpp"

#include "evaluator_internal.hpp"
#include "scar/copula/prepared_dynamic_emission.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace scar {
namespace {

struct TrajectoryParameters {
    double rho = 0.0;
    double sigma_cond = 0.0;
    double sigma_stationary = 0.0;
};

Result<TrajectoryParameters> prepare_parameters(const OuParams& params, std::size_t count) {
    Result<TrajectoryParameters> result;
    if (!evaluator_detail::valid_ou_params(params)) {
        result.status = Status::InvalidParameter;
        return result;
    }
    const double dt = count > 1 ? 1.0 / static_cast<double>(count - 1) : 1.0;
    auto& out = result.value;
    out.rho = std::exp(-params.kappa * dt);
    out.sigma_stationary = params.nu / std::sqrt(2.0 * params.kappa);
    out.sigma_cond = std::sqrt(params.nu * params.nu / (2.0 * params.kappa)
        * (1.0 - out.rho * out.rho));
    if (!std::isfinite(out.rho) || !std::isfinite(out.sigma_stationary)
        || !std::isfinite(out.sigma_cond)) {
        result.status = Status::InvalidParameter;
    }
    return result;
}

}  // namespace

Status validate_ou_trajectory_parameters(const OuParams& params, std::size_t count) {
    return prepare_parameters(params, count).status;
}

ScarOuVectorResult ou_trajectory_from_innovations(
    double x0, double mu, double rho, double sigma_cond, DoubleView innovations) {

    ScarOuVectorResult result;
    if ((!innovations.empty() && innovations.data() == nullptr)
        || innovations.size() >= result.values.max_size()) {
        result.status = Status::InvalidSize;
        return result;
    }
    if (!std::isfinite(x0) || !std::isfinite(mu) || !std::isfinite(rho)
        || !std::isfinite(sigma_cond) || sigma_cond < 0.0) {
        result.status = Status::InvalidParameter;
        return result;
    }
    result.values.resize(innovations.size() + 1);
    result.values[0] = x0;
    for (std::size_t t = 1; t < result.values.size(); ++t) {
        if (!std::isfinite(innovations[t - 1])) {
            result.status = Status::InvalidParameter;
            result.failure.index = static_cast<std::int64_t>(t - 1);
            result.values.clear();
            return result;
        }
        // Keep the frozen scalar recurrence's evaluation order.
        result.values[t] = mu + rho * (result.values[t - 1] - mu)
            + sigma_cond * innovations[t - 1];
        if (!std::isfinite(result.values[t])) {
            result.status = Status::NumericalFailure;
            result.failure.index = static_cast<std::int64_t>(t);
            result.values.clear();
            return result;
        }
    }
    return result;
}

ScarOuVectorResult sample_ou_trajectory(
    const OuParams& params, DoubleView standard_normals) {

    ScarOuVectorResult result;
    if (standard_normals.empty()) {
        return result;
    }
    if (standard_normals.data() == nullptr) {
        result.status = Status::InvalidSize;
        return result;
    }
    const auto prepared = prepare_parameters(params, standard_normals.size());
    if (!prepared.is_ok()) {
        result.status = prepared.status;
        return result;
    }
    const double x0 = params.mu
        + prepared.value.sigma_stationary * standard_normals[0];
    return ou_trajectory_from_innovations(
        x0, params.mu, prepared.value.rho, prepared.value.sigma_cond,
        DoubleView{standard_normals.data() + 1, standard_normals.size() - 1});
}

ScarOuVectorResult sample_ou_stationary(
    const OuParams& params, DoubleView standard_normals) {

    ScarOuVectorResult result;
    const auto prepared = prepare_parameters(params, standard_normals.size());
    if (!prepared.is_ok()) {
        result.status = prepared.status;
        return result;
    }
    result.values.resize(standard_normals.size());
    for (std::size_t index = 0; index < standard_normals.size(); ++index) {
        const double draw = standard_normals[index];
        if (!std::isfinite(draw)) {
            result.status = Status::InvalidParameter;
            result.failure.index = static_cast<std::int64_t>(index);
            result.values.clear();
            return result;
        }
        result.values[index] = params.mu
            + prepared.value.sigma_stationary * draw;
        if (!std::isfinite(result.values[index])) {
            result.status = Status::NumericalFailure;
            result.failure.index = static_cast<std::int64_t>(index);
            result.values.clear();
            return result;
        }
    }
    return result;
}

OuStateSampleResult sample_ou_state_distribution(
    DoubleView z_grid,
    DoubleView probability,
    DoubleView selection_uniforms,
    DoubleView jitter_uniforms,
    OuStateSamplingMode mode) {

    OuStateSampleResult result;
    if (z_grid.empty() || z_grid.size() != probability.size()
        || z_grid.data() == nullptr || probability.data() == nullptr
        || (!selection_uniforms.empty() && selection_uniforms.data() == nullptr)
        || (!jitter_uniforms.empty() && jitter_uniforms.data() == nullptr)) {
        result.status = Status::InvalidSize;
        return result;
    }
    const bool histogram = mode == OuStateSamplingMode::Histogram;
    const std::size_t required_jitter =
        histogram && z_grid.size() > 1 ? selection_uniforms.size() : 0;
    if ((mode != OuStateSamplingMode::Grid && !histogram)
        || jitter_uniforms.size() != required_jitter) {
        result.status = Status::InvalidSize;
        return result;
    }
    std::vector<double> cumulative(probability.size(), 0.0);
    double total = 0.0;
    for (std::size_t index = 0; index < z_grid.size(); ++index) {
        if (!std::isfinite(z_grid[index])
            || (index > 0 && !(z_grid[index] > z_grid[index - 1]))
            || !std::isfinite(probability[index])
            || probability[index] < 0.0) {
            result.status = Status::InvalidParameter;
            result.failure.index = static_cast<std::int64_t>(index);
            return result;
        }
        total += probability[index];
        cumulative[index] = total;
    }
    if (!std::isfinite(total) || !(total > 0.0)) {
        result.status = Status::InvalidParameter;
        return result;
    }

    result.value.values.resize(selection_uniforms.size());
    for (std::size_t row = 0; row < selection_uniforms.size(); ++row) {
        const double selection = selection_uniforms[row];
        if (!std::isfinite(selection) || selection < 0.0 || selection >= 1.0) {
            result.status = Status::InvalidParameter;
            result.failure.index = static_cast<std::int64_t>(row);
            result.value.values.clear();
            return result;
        }
        const double target = selection * total;
        const auto found = std::upper_bound(
            cumulative.begin(), cumulative.end(), target);
        const std::size_t index = std::min(
            static_cast<std::size_t>(found - cumulative.begin()),
            z_grid.size() - 1);
        double value = z_grid[index];
        if (required_jitter > 0) {
            const double jitter = jitter_uniforms[row];
            if (!std::isfinite(jitter) || jitter < 0.0 || jitter >= 1.0) {
                result.status = Status::InvalidParameter;
                result.failure.index = static_cast<std::int64_t>(row);
                result.value.values.clear();
                return result;
            }
            const double left = index == 0
                ? z_grid[0]
                : 0.5 * (z_grid[index - 1] + z_grid[index]);
            const double right = index + 1 == z_grid.size()
                ? z_grid[index]
                : 0.5 * (z_grid[index] + z_grid[index + 1]);
            value = left + jitter * (right - left);
        }
        result.value.values[row] = value;
    }
    result.value.selection_draws_used = static_cast<std::int64_t>(
        selection_uniforms.size());
    result.value.jitter_draws_used = static_cast<std::int64_t>(
        jitter_uniforms.size());
    return result;
}

StateDistribution condition_ou_state_distribution(
    const CopulaSpec& copula,
    DoubleView z_grid,
    DoubleView probability,
    ObservationView observation) {

    StateDistribution result;
    if (z_grid.empty() || z_grid.size() != probability.size()
        || z_grid.data() == nullptr || probability.data() == nullptr
        || observation.size() != 1) {
        result.status = Status::InvalidSize;
        return result;
    }
    PreparedDynamicEmission emission(copula);
    const Status observation_status = emission.validate_observations(observation);
    if (!emission.is_supported_for_ou() || !ok(observation_status)) {
        result.status = emission.is_supported_for_ou()
            ? observation_status : Status::InvalidFamily;
        return result;
    }
    result.z_grid.assign(z_grid.data(), z_grid.data() + z_grid.size());
    result.prob.assign(probability.data(), probability.data() + probability.size());
    auto workspace = emission.make_workspace(false);
    std::vector<double> log_density(z_grid.size(), 0.0);
    double maximum = -std::numeric_limits<double>::infinity();
    double prior_total = 0.0;
    for (std::size_t index = 0; index < z_grid.size(); ++index) {
        if (!std::isfinite(z_grid[index]) || !std::isfinite(probability[index])
            || probability[index] < 0.0) {
            result.status = Status::InvalidParameter;
            result.failure.index = static_cast<std::int64_t>(index);
            result.prob.clear();
            return result;
        }
        prior_total += probability[index];
        log_density[index] = emission.log_pdf_at_state(
            observation.data(), 0, z_grid[index], workspace);
        if (probability[index] > 0.0 && std::isfinite(log_density[index])) {
            maximum = std::max(maximum, log_density[index]);
        }
    }
    if (!std::isfinite(prior_total) || !(prior_total > 0.0)) {
        result.status = Status::InvalidParameter;
        result.prob.clear();
        return result;
    }
    if (!std::isfinite(maximum)) {
        result.status = Status::NumericalFailure;
        result.prob.clear();
        return result;
    }
    double total = 0.0;
    for (std::size_t index = 0; index < result.prob.size(); ++index) {
        result.prob[index] = std::isfinite(log_density[index])
            ? probability[index] * std::exp(log_density[index] - maximum)
            : 0.0;
        total += result.prob[index];
    }
    if (!std::isfinite(total) || !(total > 0.0)) {
        result.status = Status::NumericalFailure;
        result.prob.clear();
        return result;
    }
    for (double& weight : result.prob) weight /= total;
    return result;
}

}  // namespace scar
