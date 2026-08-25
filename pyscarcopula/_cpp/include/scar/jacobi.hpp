#pragma once

#include "scar/scar_jacobi/result.hpp"
#include "scar/scar_jacobi/types.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace scar {

Status validate_jacobi_params(
    const JacobiParams& params,
    double stationary_shape_max = 500.0) noexcept;

Status validate_jacobi_bounds(
    const JacobiParameterBounds& bounds) noexcept;

Status validate_jacobi_config(
    const JacobiNumericalConfig& config) noexcept;

JacobiParamsResult jacobi_raw_to_physical(
    const std::array<double, 3>& raw) noexcept;

JacobiRawParamsResult jacobi_physical_to_raw(
    const JacobiParams& params,
    double tau_eps) noexcept;

JacobiRawBoundsResult jacobi_raw_bounds(
    const JacobiParameterBounds& bounds) noexcept;

JacobiShapeResult jacobi_stationary_shape(
    const JacobiParams& params) noexcept;

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

JacobiVectorResult jacobi_sparse_left_multiply(
    const JacobiSparseTransition& transition,
    const std::vector<double>& values);

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

}  // namespace scar
