#pragma once

#include "scar/core/span.hpp"
#include "scar/copula/spec.hpp"
#include "scar/observation.hpp"
#include "scar/scar_ou/result.hpp"
#include "scar/scar_ou/types.hpp"

namespace scar {

enum class OuStateSamplingMode : int {
    Grid = 0,
    Histogram = 1,
};

/// Validate model parameters and derived scales before the caller draws RNG.
Status validate_ou_trajectory_parameters(const OuParams& params, std::size_t count);

/// Exact OU recurrence, with the initial state and innovations supplied by caller.
ScarOuVectorResult ou_trajectory_from_innovations(
    double x0, double mu, double rho, double sigma_cond, DoubleView innovations);

/// Stationary OU path over [0,1]; the first standard normal initializes the state.
ScarOuVectorResult sample_ou_trajectory(
    const OuParams& params, DoubleView standard_normals);

/// One bounded block of a path whose time step is set by total_count.
/// initialize consumes the first draw for the stationary state; otherwise
/// every draw advances previous_state by one step.
ScarOuVectorResult sample_ou_trajectory_block(
    const OuParams& params, std::size_t total_count,
    double previous_state, bool initialize, DoubleView standard_normals);

/// Stationary OU states from caller-owned raw standard normals.
ScarOuVectorResult sample_ou_stationary(
    const OuParams& params, DoubleView standard_normals);

/// Discrete or midpoint-cell state sampling from caller-owned raw uniforms.
OuStateSampleResult sample_ou_state_distribution(
    DoubleView z_grid,
    DoubleView probability,
    DoubleView selection_uniforms,
    DoubleView jitter_uniforms,
    OuStateSamplingMode mode);

/// Bayes-reweight one OU grid state using a single native model emission.
StateDistribution condition_ou_state_distribution(
    const CopulaSpec& copula,
    DoubleView z_grid,
    DoubleView probability,
    ObservationView observation);

}  // namespace scar
