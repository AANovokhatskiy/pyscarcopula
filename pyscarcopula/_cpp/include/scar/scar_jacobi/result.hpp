#pragma once

#include "scar/core/result.hpp"
#include "scar/scar_jacobi/types.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace scar {

struct JacobiRawBounds {
    std::array<double, 3> lower{};
    std::array<double, 3> upper{};
};

struct JacobiStationaryShape {
    double alpha = 0.0;
    double beta = 0.0;
    std::array<double, 3> dalpha{};
    std::array<double, 3> dbeta{};
};

struct JacobiMemoryEstimate {
    std::size_t elements = 0;
    std::uint64_t bytes = 0;
    std::uint64_t budget_bytes = 0;
    bool within_budget = false;
};

struct JacobiBoundaryValue {
    double value = 0.0;
    bool intervened = false;
};

struct JacobiQuadratureRule {
    std::vector<double> nodes;
    std::vector<double> weights;
};

/// Gauss-Jacobi nodes on tau in (0, 1), probability weights, and a row-major
/// orthonormal basis plus its derivative with respect to tau.
struct JacobiBasisRule {
    std::vector<double> tau;
    std::vector<double> weights;
    std::vector<double> basis;
    std::vector<double> basis_derivative;
    int quad_order = 0;
    int basis_order = 0;
};

/// Parameter-independent tau grid, normalized stationary masses, and their
/// row-major derivatives with respect to `(kappa, m, xi)`.
struct JacobiFixedRule {
    std::vector<double> tau;
    std::vector<double> weights;
    std::vector<double> weight_derivatives;
    int quad_order = 0;
};

struct JacobiTransitionDiagnostics {
    double dt = 0.0;
    double alpha = 0.0;
    double beta = 0.0;
    double raw_min_entry = 0.0;
    double raw_negative_mass = 0.0;
    double min_entry = 0.0;
    double max_row_sum_error_before_normalization = 0.0;
    double max_row_sum_error = 0.0;
    double stationary_error = 0.0;
    double probability_cleanup_negative_mass = 0.0;
    double probability_min_entry_before_cleanup = 0.0;
    bool clipped_negative = false;
    bool probability_cleanup_applied = false;
    int gh_order = 0;
    JacobiTransitionMethod method_requested = JacobiTransitionMethod::Auto;
    JacobiTransitionMethod method_used = JacobiTransitionMethod::Auto;
    JacobiTransitionStorage storage = JacobiTransitionStorage::Dense;
    JacobiStationarityCorrection correction =
        JacobiStationarityCorrection::None;
    Status spectral_status = Status::Ok;
    std::size_t nnz = 0;
    int max_width = 0;
    std::uint64_t retained_bytes = 0;
    std::uint64_t dense_bytes = 0;
    std::uint64_t estimated_workspace_bytes = 0;
    std::uint64_t memory_budget_bytes = 0;
    double mean_accepted_off_diagonal_mass = 0.0;
    double mean_proposed_off_diagonal_mass = 0.0;
    double acceptance_mass_ratio = 1.0;
    double min_row_acceptance_ratio = 1.0;
    double mean_stay_probability = 0.0;
    double max_stay_probability = 0.0;
    double reverse_missing_edge_fraction = 0.0;
    double detailed_balance_error = 0.0;
    int ipfp_iterations = 0;
    double ipfp_stationary_residual = 0.0;
    double ipfp_kl_divergence = 0.0;
    double ipfp_max_probability_change = 0.0;
};

struct JacobiDenseTransition {
    std::vector<double> tau;
    std::vector<double> weights;
    std::vector<double> probabilities;
    std::vector<double> derivatives;
    std::vector<double> spectral_powers;
    int order = 0;
    JacobiTransitionDiagnostics diagnostics{};
};

/// Spectral-coefficient transition setup.  The basis is row-major with
/// `(quad_order, basis_order)` shape; propagation is diagonal in coefficient
/// space through `spectral_powers`.
struct JacobiCoefficientTransition {
    std::vector<double> tau;
    std::vector<double> weights;
    std::vector<double> basis;
    std::vector<double> spectral_powers;
    int quad_order = 0;
    int basis_order = 0;
    JacobiTransitionDiagnostics diagnostics{};
};

/// Fixed-width row storage.  Inactive indices are -1 and all arrays are
/// row-major; derivatives use `(parameter, row, slot)` ordering.
struct JacobiSparseTransition {
    std::vector<double> tau;
    std::vector<double> weights;
    std::vector<std::int64_t> indices;
    std::vector<double> probabilities;
    std::vector<std::int64_t> counts;
    std::vector<double> derivatives;
    int order = 0;
    int max_width = 0;
    JacobiTransitionDiagnostics diagnostics{};
};

struct JacobiHorizonDiagnostics {
    std::int64_t steps = 0;
    double one_step_stationary_tv = 0.0;
    double full_horizon_stationary_tv = 0.0;
    double target_mean = 0.0;
    double propagated_mean = 0.0;
    double target_variance = 0.0;
    double propagated_variance = 0.0;
    double relative_variance_error = 0.0;
    double conditional_mean_rmse = 0.0;
    double conditional_mean_max_error = 0.0;
    double lag_one_correlation = 0.0;
    double target_lag_one_correlation = 0.0;
    double lag_one_correlation_error = 0.0;
};

struct JacobiAdaptiveCandidate {
    int quad_order = 0;
    bool passed = false;
    bool memory_limited = false;
    Status status = Status::Ok;
    std::uint64_t retained_bytes = 0;
    JacobiHorizonDiagnostics diagnostics{};
};

struct JacobiAdaptiveSelection {
    JacobiSparseTransition transition{};
    std::vector<double> tau;
    std::vector<double> weights;
    int selected_quad_order = 0;
    bool passed = false;
    bool exhausted = false;
    std::vector<JacobiAdaptiveCandidate> candidates;
};

struct JacobiFilterDiagnostics {
    JacobiTransitionDiagnostics transition{};
    std::int64_t n_obs = 0;
    int order = 0;
    double log_likelihood = 0.0;
    double minimum_scale = 0.0;
    double maximum_scale = 0.0;
    double max_predictive_mass_error = 0.0;
    double max_filtered_mass_error = 0.0;
    double max_smoothed_mass_error = 0.0;
    std::uint64_t preparation_generation = 0;
};

/// Complete forward/backward evaluator payload.  Observation-grid arrays are
/// row-major with shape `(n_obs, order)`.
struct JacobiFilterState {
    std::vector<double> tau;
    std::vector<double> theta;
    std::vector<double> emissions;
    std::vector<double> predicted;
    std::vector<double> filtered;
    std::vector<double> smoothed;
    std::vector<double> scales;
    std::vector<double> current_probability;
    std::vector<double> next_probability;
    std::int64_t n_obs = 0;
    int order = 0;
    JacobiFilterDiagnostics diagnostics{};
};

struct JacobiObjectiveValue {
    double log_likelihood = 0.0;
    double objective = 0.0;
    JacobiFilterDiagnostics diagnostics{};
};

struct JacobiObjectiveGradient {
    double objective = 0.0;
    std::array<double, 3> gradient{};
    JacobiFilterDiagnostics diagnostics{};
};

struct JacobiEvaluatorVector {
    std::vector<double> values;
    JacobiFilterDiagnostics diagnostics{};
};

struct JacobiEvaluatorPair {
    std::vector<double> first;
    std::vector<double> second;
    JacobiFilterDiagnostics diagnostics{};
};

struct JacobiStateDistribution {
    std::vector<double> tau;
    std::vector<double> probability;
    JacobiStateHorizon horizon = JacobiStateHorizon::Current;
    JacobiFilterDiagnostics diagnostics{};
};

/// Fixed-draw trajectory payload shared by dense/sparse TM-grid and
/// Lamperti--Euler sampling.  Grid sampling populates transition diagnostics;
/// Lamperti sampling populates the boundary/final-state fields.
struct JacobiTrajectory {
    std::vector<double> tau;
    std::int64_t draws_used = 0;
    std::int64_t normal_draws_used = 0;
    std::int64_t euler_steps = 0;
    std::int64_t boundary_interventions = 0;
    double final_lamperti_value = 0.0;
    JacobiTransitionDiagnostics transition{};
};

/// Samples from a native Jacobi state distribution after the native pair
/// kernel maps sampled tau values to copula parameters.
struct JacobiStateSample {
    std::vector<double> tau;
    std::vector<double> parameters;
    std::int64_t selection_draws_used = 0;
    std::int64_t jitter_draws_used = 0;
};

struct JacobiHistogramCells {
    std::vector<double> left;
    std::vector<double> right;
};

using JacobiParamsResult = Result<JacobiParams>;
using JacobiRawParamsResult = Result<std::array<double, 3>>;
using JacobiRawBoundsResult = Result<JacobiRawBounds>;
using JacobiShapeResult = Result<JacobiStationaryShape>;
using JacobiScalarResult = Result<double>;
using JacobiIntResult = Result<int>;
using JacobiVectorResult = Result<std::vector<double>>;
using JacobiBoundaryResult = Result<JacobiBoundaryValue>;
using JacobiMemoryResult = Result<JacobiMemoryEstimate>;
using JacobiQuadratureResult = Result<JacobiQuadratureRule>;
using JacobiBasisResult = Result<JacobiBasisRule>;
using JacobiFixedRuleResult = Result<JacobiFixedRule>;
using JacobiDenseTransitionResult = Result<JacobiDenseTransition>;
using JacobiCoefficientTransitionResult =
    Result<JacobiCoefficientTransition>;
using JacobiSparseTransitionResult = Result<JacobiSparseTransition>;
using JacobiHorizonResult = Result<JacobiHorizonDiagnostics>;
using JacobiAdaptiveSelectionResult = Result<JacobiAdaptiveSelection>;
using JacobiFilterResult = Result<JacobiFilterState>;
using JacobiObjectiveResult = Result<JacobiObjectiveValue>;
using JacobiGradientResult = Result<JacobiObjectiveGradient>;
using JacobiEvaluatorVectorResult = Result<JacobiEvaluatorVector>;
using JacobiEvaluatorPairResult = Result<JacobiEvaluatorPair>;
using JacobiStateDistributionResult = Result<JacobiStateDistribution>;
using JacobiTrajectoryResult = Result<JacobiTrajectory>;
using JacobiStateSampleResult = Result<JacobiStateSample>;
using JacobiHistogramCellsResult = Result<JacobiHistogramCells>;

}  // namespace scar
