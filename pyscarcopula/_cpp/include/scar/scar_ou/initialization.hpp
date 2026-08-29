#pragma once

#include "scar/core/result.hpp"
#include "scar/observation.hpp"
#include "scar/scar_ou/types.hpp"

#include <cstddef>

namespace scar {

enum class OuInitializationRegime : int {
    NotApplicable = 0,
    Weak = 1,
    Medium = 2,
    Strong = 3,
};

struct OuInitializationConfig {
    double rho_target = 0.96;
    double sigma_fraction = 0.3;
    double weak_tau = 0.06;
    double strong_tau = 0.25;
    double weak_log_likelihood_per_observation = 0.003;
    double strong_log_likelihood_per_observation = 0.04;
    double weak_stationary_scale = 0.01;
    double maximum_stationary_scale = 2.0;
};

struct OuInitialization {
    OuParams params{};
    double theta_mle = 0.0;
    double static_log_likelihood = 0.0;
    double static_log_likelihood_per_observation = 0.0;
    double absolute_kendall_tau = 0.0;
    double strength = 0.0;
    double stationary_scale = 0.0;
    double legacy_stationary_scale = 0.0;
    double rho_target = 0.0;
    OuInitializationRegime regime = OuInitializationRegime::NotApplicable;
};

using OuInitializationResult = Result<OuInitialization>;

Result<double> ou_initial_kappa(
    std::size_t observation_count,
    double rho_target = 0.96,
    double kappa_min = 0.01,
    double kappa_max = 100.0);
OuInitializationResult ou_default_initial_point(double mu);
OuInitializationResult ou_heuristic_initial_point(
    std::size_t observation_count,
    double mu,
    double rho_target = 0.95,
    double sigma_fraction = 0.3);
OuInitializationResult ou_stochastic_student_initial_point(
    std::size_t observation_count,
    double theta_mle,
    double mu,
    double static_log_likelihood,
    double rho_target = 0.96,
    double nu = 0.1);
OuInitializationResult ou_strength_aware_initial_point(
    ObservationView observations,
    double theta_mle,
    double mu,
    double static_log_likelihood,
    const OuInitializationConfig& config = {});

}  // namespace scar
