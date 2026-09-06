#include "scar/scar_ou/initialization.hpp"

#include "scar/copula/multivariate/correlation/parameterization.hpp"

#include <algorithm>
#include <cmath>

namespace scar {
namespace {

bool finite_config(const OuInitializationConfig& config) {
    return std::isfinite(config.rho_target)
        && std::isfinite(config.sigma_fraction)
        && std::isfinite(config.weak_tau)
        && std::isfinite(config.strong_tau)
        && std::isfinite(config.weak_log_likelihood_per_observation)
        && std::isfinite(config.strong_log_likelihood_per_observation)
        && std::isfinite(config.weak_stationary_scale)
        && std::isfinite(config.maximum_stationary_scale)
        && config.rho_target > 0.0 && config.rho_target < 1.0
        && config.sigma_fraction > 0.0
        && config.strong_tau > config.weak_tau
        && config.strong_log_likelihood_per_observation
            > config.weak_log_likelihood_per_observation
        && config.weak_stationary_scale > 0.0
        && config.maximum_stationary_scale >= config.weak_stationary_scale;
}

OuInitializationResult invalid() {
    return {{}, Status::InvalidParameter, {}};
}

}  // namespace

Result<double> ou_initial_kappa(
    std::size_t observation_count,
    double rho_target,
    double kappa_min,
    double kappa_max) {
    if (!std::isfinite(rho_target) || rho_target <= 0.0 || rho_target >= 1.0
        || !std::isfinite(kappa_min) || !std::isfinite(kappa_max)
        || kappa_min <= 0.0 || kappa_min >= kappa_max) {
        return {0.0, Status::InvalidParameter, {}};
    }
    if (observation_count < 2) {
        return success(kappa_min);
    }
    const double dt = 1.0 / static_cast<double>(observation_count - 1);
    const double kappa = -std::log(rho_target) / dt;
    return success(std::clamp(kappa, kappa_min, kappa_max));
}

OuInitializationResult ou_default_initial_point(double mu) {
    if (!std::isfinite(mu)) return invalid();
    OuInitialization output;
    output.params = OuParams{1.0, mu, 1.0};
    return success(output);
}

OuInitializationResult ou_heuristic_initial_point(
    std::size_t observation_count,
    double mu,
    double rho_target,
    double sigma_fraction) {
    if (observation_count < 2 || !std::isfinite(mu)
        || !std::isfinite(sigma_fraction) || sigma_fraction <= 0.0) {
        return invalid();
    }
    const auto kappa = ou_initial_kappa(
        observation_count, rho_target, 0.01, 100.0);
    if (!kappa.is_ok()) return invalid();
    const double sigma = sigma_fraction * std::max(std::abs(mu), 1.0);
    const double nu = std::clamp(
        sigma * std::sqrt(2.0 * kappa.value), 0.01, 50.0);
    OuInitialization output;
    output.params = OuParams{kappa.value, mu, nu};
    output.rho_target = rho_target;
    output.stationary_scale = sigma;
    return success(output);
}

OuInitializationResult ou_stochastic_student_initial_point(
    std::size_t observation_count,
    double theta_mle,
    double mu,
    double static_log_likelihood,
    double rho_target,
    double nu) {
    if (!std::isfinite(theta_mle) || !std::isfinite(mu)
        || !std::isfinite(static_log_likelihood) || !std::isfinite(nu)) {
        return invalid();
    }
    const auto kappa = ou_initial_kappa(observation_count, rho_target);
    if (!kappa.is_ok()) return invalid();
    OuInitialization output;
    output.params = OuParams{
        kappa.value, mu, std::clamp(nu, 0.001, 50.0)};
    output.stationary_scale = output.params.nu / std::sqrt(2.0 * kappa.value);
    output.theta_mle = theta_mle;
    output.static_log_likelihood = static_log_likelihood;
    output.rho_target = rho_target;
    return success(output);
}

Result<std::vector<double>> ou_student_initial_stencil(double mu) {
    if (!std::isfinite(mu)) return {{}, Status::InvalidParameter, {}};
    const double step = 0.001 * std::max(1.0, std::abs(mu));
    if (!std::isfinite(mu - step) || !std::isfinite(mu + step)) {
        return {{}, Status::InvalidParameter, {}};
    }
    return success(std::vector<double>{mu - step, mu, mu + step, step});
}

OuInitializationResult ou_student_score_initial_point(
    ObservationView log_emissions,
    double theta_mle, double mu, double static_log_likelihood,
    double step, double rho_target, double maximum_stationary_scale) {
    if (log_emissions.n_obs < 2 || log_emissions.dim != 3
        || log_emissions.values == nullptr || !std::isfinite(theta_mle)
        || !std::isfinite(mu) || !std::isfinite(static_log_likelihood)
        || !std::isfinite(step) || step <= 0.0
        || !std::isfinite(maximum_stationary_scale)
        || maximum_stationary_scale < 0.01) return invalid();
    const auto kappa = ou_initial_kappa(log_emissions.n_obs, rho_target);
    if (!kappa.is_ok()) return invalid();
    const double rho = std::exp(-kappa.value
        / static_cast<double>(log_emissions.n_obs - 1));
    double previous_score = 0.0, quadratic = 0.0;
    double score_squares = 0.0, curvature = 0.0;
    double correlation_squares = 0.0, trace = 0.0;
    for (std::size_t t = 0; t < log_emissions.n_obs; ++t) {
        const double* row = log_emissions.values + 3 * t;
        if (!std::isfinite(row[0]) || !std::isfinite(row[1])
            || !std::isfinite(row[2])) return invalid();
        const double score = (row[2] - row[0]) / (2.0 * step);
        const double second = ((row[2] - row[1]) + (row[0] - row[1]))
            / (step * step);
        score_squares += score * score;
        curvature += second;
        quadratic += score * score + 2.0 * score * previous_score;
        previous_score = rho * (previous_score + score);
        // tr(C_rho^2), with no dense T-by-T allocation.
        trace += 1.0 + 2.0 * correlation_squares;
        correlation_squares = rho * rho * (1.0 + correlation_squares);
    }
    const double information = score_squares
        / static_cast<double>(log_emissions.n_obs);
    const double fisher = 0.5 * information * information * trace;
    const double variance_score = 0.5 * (quadratic + curvature);
    if (!std::isfinite(fisher) || !std::isfinite(variance_score)) return invalid();
    // The v=0 score is exact in the local Taylor model. Its Gaussian
    // information proxy sets a one-standard-error interior variance floor,
    // preventing a vanishing log-scale gradient without claiming evidence
    // for dynamics. The cap matches the existing pair initializer.
    double floor = maximum_stationary_scale;
    double scale = maximum_stationary_scale;
    if (fisher > 0.0) {
        floor = std::pow(fisher, -0.25);
        scale = std::sqrt(std::max(floor * floor, variance_score / fisher));
    }
    scale = std::clamp(scale, 0.01, maximum_stationary_scale);
    OuInitialization output;
    const double nu = scale * std::sqrt(2.0 * kappa.value);
    if (!std::isfinite(nu)) return invalid();
    output.params = {kappa.value, mu, nu};
    output.theta_mle = theta_mle;
    output.static_log_likelihood = static_log_likelihood;
    output.rho_target = rho_target;
    output.stationary_scale = scale;
    output.variance_score = variance_score;
    output.variance_information = fisher;
    output.stationary_scale_floor = std::clamp(
        floor, 0.01, maximum_stationary_scale);
    return success(output);
}

OuInitializationResult ou_strength_aware_initial_point(
    ObservationView observations,
    double theta_mle,
    double mu,
    double static_log_likelihood,
    const OuInitializationConfig& config) {
    if (observations.n_obs == 0 || observations.dim < 2
        || observations.values == nullptr || !std::isfinite(theta_mle)
        || !std::isfinite(mu) || !std::isfinite(static_log_likelihood)
        || !finite_config(config)) {
        return invalid();
    }
    double tau = kendall_tau_b(observations, 0, 1);
    if (!std::isfinite(tau)) tau = 0.0;
    tau = std::abs(tau);
    const double per_observation = static_log_likelihood
        / static_cast<double>(observations.n_obs);
    const double tau_strength = std::clamp(
        (tau - config.weak_tau) / (config.strong_tau - config.weak_tau),
        0.0, 1.0);
    const double likelihood_strength = std::clamp(
        (per_observation - config.weak_log_likelihood_per_observation)
        / (config.strong_log_likelihood_per_observation
            - config.weak_log_likelihood_per_observation),
        0.0, 1.0);
    const double strength = std::max(tau_strength, likelihood_strength);
    const double legacy_scale = std::clamp(
        config.sigma_fraction * std::max(std::abs(mu), 1.0),
        config.weak_stationary_scale, config.maximum_stationary_scale);
    const double stationary_scale = config.weak_stationary_scale * std::pow(
        legacy_scale / config.weak_stationary_scale, strength);
    const auto kappa = ou_initial_kappa(
        observations.n_obs, config.rho_target);
    if (!kappa.is_ok()) return invalid();
    const double nu = std::clamp(
        stationary_scale * std::sqrt(2.0 * kappa.value), 0.01, 50.0);
    OuInitialization output;
    output.params = OuParams{kappa.value, mu, nu};
    output.theta_mle = theta_mle;
    output.static_log_likelihood = static_log_likelihood;
    output.static_log_likelihood_per_observation = per_observation;
    output.absolute_kendall_tau = tau;
    output.strength = strength;
    output.stationary_scale = stationary_scale;
    output.legacy_stationary_scale = legacy_scale;
    output.rho_target = config.rho_target;
    output.regime = tau < config.weak_tau
            && per_observation < config.weak_log_likelihood_per_observation
        ? OuInitializationRegime::Weak
        : (strength > 0.75
            ? OuInitializationRegime::Strong
            : OuInitializationRegime::Medium);
    return success(output);
}

}  // namespace scar
