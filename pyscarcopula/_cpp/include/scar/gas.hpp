#pragma once

#include "scar/copula/spec.hpp"
#include "scar/gas/result.hpp"
#include "scar/observation.hpp"

#include <vector>

namespace scar {

class PreparedDynamicEmission;
class PreparedDynamicEmissionWorkspace;

/// Scaling applied to the GAS score in the state recursion.
enum class GasScaling : int {
    Unit = 0,
    Fisher = 1,
};

/// Parameters of `g[t+1] = omega + beta*g[t] + gamma*score[t]`.
struct GasParams {
    double omega = 0.0;
    double gamma = 0.0;
    double beta = 0.0;
};

/// Numerical safeguards and score-scaling settings for GAS evaluation.
struct GasConfig {
    GasScaling scaling = GasScaling::Unit;
    double score_eps = 1e-4;
    double g_clip = 50.0;
    double score_clip = 100.0;
    double fisher_floor = 1e-6;
    double stationary_beta_tol = 1e-8;
    double optimizer_gradient_eps = 1e-5;
    bool optimizer_gradient_relative = false;
};

/// Native evaluator for bivariate score-driven copula dynamics.
class GasEvaluator {
public:
    GasStateResult initial_state(
        const GasParams& params,
        const CopulaSpec& copula,
        const GasConfig& config) const;

    GasStateResult initial_state_prepared(
        const GasParams& params,
        const PreparedDynamicEmission& emission,
        const GasConfig& config) const;

    GasFilterResult filter(
        const GasParams& params,
        const CopulaSpec& copula,
        ObservationView u,
        const GasConfig& config) const;

    GasLogLikResult log_likelihood(
        const GasParams& params,
        const CopulaSpec& copula,
        ObservationView u,
        const GasConfig& config) const;

    GasLogLikResult negative_log_likelihood(
        const GasParams& params,
        const CopulaSpec& copula,
        ObservationView u,
        const GasConfig& config) const;

    GasObjectiveGradientResult negative_log_likelihood_and_gradient(
        const GasParams& params,
        const CopulaSpec& copula,
        ObservationView u,
        const GasConfig& config) const;

    GasObjectiveGradientResult
    negative_log_likelihood_and_gradient_shrinkage(
        const GasParams& params,
        const CopulaSpec& copula,
        DoubleView base_correlation,
        double raw_shrinkage,
        ObservationView u,
        const GasConfig& config) const;

    GasUpdateResult update_one(
        const GasParams& params,
        const CopulaSpec& copula,
        double g,
        double u1,
        double u2,
        const GasConfig& config) const;

    GasUpdateResult update_one_prepared(
        const GasParams& params,
        const PreparedDynamicEmission& emission,
        PreparedDynamicEmissionWorkspace& workspace,
        double g,
        double u1,
        double u2,
        const GasConfig& config) const;

    GasUpdateResult update_observation(
        const GasParams& params,
        const CopulaSpec& copula,
        double g,
        ObservationView observation,
        const GasConfig& config) const;

    GasUpdateResult update_observation_prepared(
        const GasParams& params,
        const PreparedDynamicEmission& emission,
        PreparedDynamicEmissionWorkspace& workspace,
        double g,
        ObservationView observation,
        const GasConfig& config) const;

    GasPredictResult predict_parameter(
        const GasParams& params,
        const CopulaSpec& copula,
        ObservationView u,
        const GasConfig& config,
        bool horizon_next) const;

    GasPathResult h_path(
        const GasParams& params,
        const CopulaSpec& copula,
        ObservationView u,
        const GasConfig& config) const;

    GasOuInitializationResult ou_initial_point(
        double static_mu,
        const CopulaSpec& copula,
        ObservationView u,
        const GasConfig& config) const;
};

}  // namespace scar
