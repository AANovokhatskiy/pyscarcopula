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
    output.theta_mle = theta_mle;
    output.static_log_likelihood = static_log_likelihood;
    output.rho_target = rho_target;
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
