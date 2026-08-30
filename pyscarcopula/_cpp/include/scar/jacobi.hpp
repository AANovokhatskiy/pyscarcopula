#pragma once

#include "scar/scar_jacobi/result.hpp"
#include "scar/scar_jacobi/types.hpp"
#include "scar/copula/spec.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace scar {

Status validate_jacobi_params(
    const JacobiParams& params,
    double stationary_shape_max = 500.0) noexcept;

Status validate_jacobi_bounds(
    const JacobiParameterBounds& bounds) noexcept;

Status validate_jacobi_config(
    const JacobiNumericalConfig& config) noexcept;

/// Validate ordered tau atoms and nonnegative masses; return their finite,
/// positive total. Callers may supply unnormalized probability measures.
JacobiScalarResult validate_jacobi_state_distribution(
    const std::vector<double>& tau,
    const std::vector<double>& probability) noexcept;

JacobiParamsResult jacobi_raw_to_physical(
    const std::array<double, 3>& raw) noexcept;

JacobiRawParamsResult jacobi_physical_to_raw(
    const JacobiParams& params,
    double tau_eps) noexcept;

JacobiRawParamsResult jacobi_gradient_to_raw(
    const JacobiParams& params,
    const std::array<double, 3>& physical_gradient) noexcept;

JacobiRawBoundsResult jacobi_raw_bounds(
    const JacobiParameterBounds& bounds) noexcept;

JacobiParamsResult jacobi_initial_point(
    double tau, bool has_tau, double tau_eps) noexcept;

JacobiShapeResult jacobi_stationary_shape(
    const JacobiParams& params) noexcept;

/// Transform caller-owned raw uniforms into the stationary Jacobi beta law.
JacobiVectorResult sample_jacobi_stationary(
    const JacobiParams& params,
    const std::vector<double>& uniforms);

JacobiScalarResult jacobi_resolve_dt(
    std::int64_t n_obs) noexcept;

JacobiMemoryResult estimate_jacobi_workspace(
    const JacobiNumericalConfig& config) noexcept;

JacobiMemoryResult estimate_jacobi_sampling_workspace(
    std::int64_t n,
    const JacobiNumericalConfig& config) noexcept;

JacobiScalarResult jacobi_lamperti(
    double tau,
    double xi) noexcept;

JacobiVectorResult jacobi_lamperti_values(
    const std::vector<double>& tau,
    double xi);

JacobiScalarResult jacobi_inverse_lamperti(
    double value,
    double xi) noexcept;

JacobiVectorResult jacobi_inverse_lamperti_values(
    const std::vector<double>& values,
    double xi);

JacobiScalarResult jacobi_lamperti_drift(
    const JacobiParams& params,
    double tau,
    double interior_eps = 0.0) noexcept;

JacobiVectorResult jacobi_lamperti_drift_values(
    const JacobiParams& params,
    const std::vector<double>& tau,
    double interior_eps = 0.0);

JacobiBoundaryResult apply_jacobi_boundary(
    double value,
    double upper,
    JacobiBoundaryPolicy policy) noexcept;

JacobiScalarResult jacobi_log_beta(
    double alpha,
    double beta) noexcept;

JacobiScalarResult jacobi_digamma(double value) noexcept;

JacobiScalarResult jacobi_trigamma(double value) noexcept;

JacobiQuadratureResult gauss_hermite_probability_rule(
    int order,
    std::uint64_t memory_budget_bytes = kDefaultJacobiMemoryBudgetBytes);

JacobiQuadratureResult gauss_jacobi_probability_rule(
    double alpha,
    double beta,
    int order,
    std::uint64_t memory_budget_bytes = kDefaultJacobiMemoryBudgetBytes);

JacobiBasisResult build_jacobi_rule(
    double alpha,
    double beta,
    int quad_order,
    int basis_order,
    std::uint64_t memory_budget_bytes =
        kDefaultJacobiMemoryBudgetBytes);

JacobiFixedRuleResult build_fixed_jacobi_rule(
    const JacobiParams& params,
    int quad_order,
    std::uint64_t memory_budget_bytes =
        kDefaultJacobiMemoryBudgetBytes);

JacobiFixedRuleResult build_fixed_jacobi_shape_rule(
    double alpha,
    double beta,
    int quad_order,
    std::uint64_t memory_budget_bytes =
        kDefaultJacobiMemoryBudgetBytes);

JacobiVectorResult evaluate_jacobi_polynomials(
    double x,
    double alpha,
    double beta,
    int order,
    bool derivative = false);

Status validate_jacobi_transition_config(
    const JacobiTransitionConfig& config) noexcept;

JacobiMemoryResult estimate_jacobi_sparse_workspace(
    const JacobiTransitionConfig& config) noexcept;

JacobiMemoryResult estimate_jacobi_sparse_storage(
    const JacobiTransitionConfig& config) noexcept;

JacobiIntResult default_jacobi_quad_order(int basis_order) noexcept;

JacobiIntResult resolve_jacobi_basis_order(
    int requested_basis_order,
    int quad_order) noexcept;

JacobiIntResult jacobi_horizon_steps(std::int64_t n_obs) noexcept;

JacobiVectorResult jacobi_transition_powers(
    const JacobiParams& params,
    std::int64_t n_obs,
    int basis_order);

JacobiCoefficientTransitionResult build_jacobi_coefficient_transition(
    const JacobiParams& params,
    const JacobiTransitionConfig& config);

JacobiVectorResult apply_jacobi_coefficient_transition(
    const JacobiCoefficientTransition& transition,
    const std::vector<double>& coefficients);

JacobiDenseTransitionResult build_jacobi_spectral_transition(
    const JacobiParams& params,
    const JacobiTransitionConfig& config);

JacobiDenseTransitionResult build_jacobi_local_transition(
    const JacobiParams& params,
    const JacobiTransitionConfig& config);

JacobiDenseTransitionResult build_jacobi_fixed_transition(
    const JacobiParams& params,
    const JacobiTransitionConfig& config);

JacobiDenseTransitionResult build_jacobi_dense_transition(
    const JacobiParams& params,
    const JacobiTransitionConfig& config);

JacobiSparseTransitionResult build_jacobi_sparse_transition(
    const JacobiParams& params,
    const JacobiTransitionConfig& config);

enum class JacobiSparseValidationCode : int {
    Ok = 0,
    InvalidShape,
    InvalidCount,
    InvalidProbability,
    IndexOutOfRange,
    IndicesNotIncreasing,
    RowSum,
};

JacobiSparseValidationCode validate_jacobi_sparse_transition(
    const JacobiSparseTransition& transition) noexcept;

JacobiVectorResult jacobi_sparse_left_multiply(
    const JacobiSparseTransition& transition,
    const std::vector<double>& values);

/// Materialize a validated sparse transition as a row-major dense matrix.
/// This diagnostic conversion remains inside the native numerical boundary.
JacobiVectorResult jacobi_sparse_to_dense(
    const JacobiSparseTransition& transition);

JacobiHorizonResult jacobi_sparse_full_horizon_diagnostics(
    const JacobiParams& params,
    const std::vector<double>& tau,
    const std::vector<double>& weights,
    const JacobiSparseTransition& transition,
    std::int64_t steps);

JacobiAdaptiveSelectionResult select_sparse_jacobi_order(
    const JacobiParams& params,
    const JacobiTransitionConfig& config,
    const std::vector<int>& quad_orders,
    const JacobiAdaptiveThresholds& thresholds,
    bool require_pass);

/// Build the requested dense or sparse TM-grid transition and sample one
/// path from caller-owned uniform draws.  One draw is consumed per returned
/// state, including the stationary initial state.
JacobiTrajectoryResult sample_jacobi_grid_trajectory(
    const JacobiParams& params,
    const JacobiTransitionConfig& config,
    const std::vector<double>& uniforms);

JacobiTrajectoryResult sample_prepared_jacobi_sparse_trajectory(
    const std::vector<double>& tau,
    const std::vector<double>& weights,
    const JacobiSparseTransition& transition,
    const std::vector<double>& uniforms);

/// Advance a Lamperti--Euler path across complete observation intervals.
/// `normal_draws` is row-major `(intervals, substeps)` and every successful
/// call reports the exact number consumed plus the final Lamperti state for
/// deterministic chunk continuation.
JacobiTrajectoryResult sample_jacobi_lamperti_chunk(
    const JacobiParams& params,
    const JacobiLampertiSamplingConfig& config,
    double initial_lamperti_value,
    const std::vector<double>& normal_draws);

/// Select tau values from an arbitrary native filtered/conditioned state and
/// map them to parameters with the prepared pair-copula kernel.  Histogram
/// mode consumes one additional jitter draw per sample when the grid has
/// more than one atom.
JacobiStateSampleResult sample_jacobi_state_distribution(
    const CopulaSpec& copula,
    const std::vector<double>& tau,
    const std::vector<double>& probability,
    const std::vector<double>& selection_draws,
    const std::vector<double>& jitter_draws,
    JacobiStateSamplingMode mode,
    double theta_cap);

JacobiHistogramCellsResult jacobi_state_histogram_cells(
    const std::vector<double>& tau,
    const std::vector<std::int64_t>& indices);

/// Reusable SCAR-TM-Jacobi evaluator for one immutable copula/observation
/// cell.  The implementation owns observation caches, transformed grids,
/// emissions, transitions, filter/smoother states, and reusable workspaces.
class PreparedScarJacobiEvaluator {
public:
    PreparedScarJacobiEvaluator(
        CopulaSpec copula,
        std::vector<double> observations,
        std::int64_t n_obs,
        int dim,
        JacobiEvaluatorConfig config);
    ~PreparedScarJacobiEvaluator();

    PreparedScarJacobiEvaluator(const PreparedScarJacobiEvaluator&) = delete;
    PreparedScarJacobiEvaluator& operator=(
        const PreparedScarJacobiEvaluator&) = delete;
    PreparedScarJacobiEvaluator(PreparedScarJacobiEvaluator&&) noexcept;
    PreparedScarJacobiEvaluator& operator=(
        PreparedScarJacobiEvaluator&&) noexcept;

    JacobiFilterResult filter(const JacobiParams& params) const;
    JacobiObjectiveResult loglik(const JacobiParams& params) const;
    JacobiGradientResult neg_loglik_with_grad(
        const JacobiParams& params) const;
    JacobiEvaluatorVectorResult predictive_mean(
        const JacobiParams& params) const;
    JacobiEvaluatorVectorResult mixture_h(
        const JacobiParams& params) const;
    JacobiEvaluatorPairResult mixture_h_pair(
        const JacobiParams& params) const;
    JacobiEvaluatorPairResult rosenblatt(
        const JacobiParams& params) const;
    JacobiEvaluatorPairResult gaussian_rosenblatt(
        const JacobiParams& params) const;
    JacobiStateDistributionResult state_distribution(
        const JacobiParams& params,
        JacobiStateHorizon horizon) const;
    JacobiStateDistributionResult condition_state(
        const std::vector<double>& tau,
        const std::vector<double>& probability,
        const std::array<double, 2>& observation,
        JacobiStateHorizon horizon = JacobiStateHorizon::Current) const;

    std::uint64_t preparation_count() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace scar
