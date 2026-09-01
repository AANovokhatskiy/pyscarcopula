"""Sparse local-transition operators for the Jacobi latent process."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyscarcopula._native import jacobi as jacobi_native
from pyscarcopula.numerical._arrays import (
    as_float64_array,
    validate_float64_allocation,
    validate_integer,
    validate_positive_int,
)
from pyscarcopula.numerical._transition_methods import (
    normalize_jacobi_matrix_transition_method,
)
from pyscarcopula.numerical.jacobi_tm import (
    DEFAULT_JACOBI_MEMORY_BUDGET_BYTES,
    _validate_jacobi_order,
    _validate_transition_count,
)


@dataclass(frozen=True)
class SparseJacobiTransition:
    """Compact row-wise representation of a local Jacobi transition."""

    indices: np.ndarray
    probabilities: np.ndarray
    counts: np.ndarray

    def __post_init__(self):
        indices = np.asarray(self.indices, dtype=np.intp)
        probabilities = as_float64_array(
            self.probabilities, name="probabilities")
        counts = np.asarray(self.counts, dtype=np.intp)
        if indices.ndim != 2 or probabilities.shape != indices.shape:
            raise ValueError(
                "indices and probabilities must have the same 2D shape")
        if counts.shape != (indices.shape[0],):
            raise ValueError("counts must have shape (n_rows,)")
        jacobi_native.validate_sparse_transition(
            indices, probabilities, counts)
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "probabilities", probabilities)
        object.__setattr__(self, "counts", counts)

    @property
    def shape(self):
        n_rows = int(self.indices.shape[0])
        return n_rows, n_rows

    @property
    def max_width(self):
        return int(self.indices.shape[1])

    @property
    def nnz(self):
        return int(np.sum(self.counts))

    @property
    def retained_bytes(self):
        return int(
            self.indices.nbytes
            + self.probabilities.nbytes
            + self.counts.nbytes)

    def left_multiply(self, vector):
        """Return ``vector @ P`` without dense materialization."""
        vector = as_float64_array(vector, name="vector")
        if vector.shape != (self.shape[0],):
            raise ValueError(
                f"vector must have shape ({self.shape[0]},)")
        return jacobi_native.sparse_left_multiply(
            self.indices, self.probabilities, self.counts, vector)

    def to_dense(self, *, memory_budget_bytes=None):
        """Materialize a guarded dense matrix for diagnostics."""
        n_rows = self.shape[0]
        validate_float64_allocation(
            (n_rows, n_rows),
            name="dense Jacobi transition diagnostic",
            memory_budget_bytes=memory_budget_bytes,
        )
        return jacobi_native.sparse_to_dense(
            self.indices, self.probabilities, self.counts)


def _validate_sparse_workspace(
        *,
        quad_order,
        gh_order,
        correction="none",
        memory_budget_bytes=None):
    """Compatibility facade for the native sparse-storage preflight."""
    if memory_budget_bytes is None:
        memory_budget_bytes = DEFAULT_JACOBI_MEMORY_BUDGET_BYTES
    return jacobi_native.estimate_sparse_storage(
        quad_order=quad_order,
        gh_order=gh_order,
        correction=correction,
        memory_budget_bytes=memory_budget_bytes,
    )


def jacobi_sparse_local_transition(
        kappa,
        m,
        xi,
        *,
        n_obs,
        quad_order=128,
        basis_order=1,
        gh_order=5,
        correction="none",
        memory_budget_bytes=None,
        return_diagnostics=False):
    """Build a direct sparse local-GH Jacobi transition."""
    quad_order = _validate_jacobi_order(quad_order, "quad_order")
    basis_order = _validate_jacobi_order(basis_order, "basis_order")
    gh_order = _validate_jacobi_order(gh_order, "gh_order")
    if correction not in {"none", "mh", "ipfp"}:
        raise ValueError("correction must be 'none', 'mh', or 'ipfp'")
    n_obs = _validate_transition_count(n_obs)
    budget = (
        DEFAULT_JACOBI_MEMORY_BUDGET_BYTES
        if memory_budget_bytes is None else memory_budget_bytes)
    (
        tau,
        weights,
        indices,
        probabilities,
        counts,
        _,
        native,
    ) = jacobi_native.sparse_transition(
        kappa,
        m,
        xi,
        n_obs=n_obs,
        quad_order=quad_order,
        basis_order=basis_order,
        gh_order=gh_order,
        method="local",
        correction=correction,
        memory_budget_bytes=budget,
    )
    transition = SparseJacobiTransition(
        indices=indices,
        probabilities=probabilities,
        counts=counts,
    )
    diagnostics = {
        "dt": native["dt"],
        "alpha": native["alpha"],
        "beta": native["beta"],
        "gh_order": native["gh_order"],
        "transition_method": "local_sparse",
        "correction": correction,
        "nnz": int(native["nnz"]),
        "max_width": int(native["max_width"]),
        "retained_bytes": int(native["retained_bytes"]),
        "dense_bytes": int(native["dense_bytes"]),
        "max_row_sum_error": native["max_row_sum_error"],
        "stationary_error": native["stationary_error"],
    }
    if correction == "mh":
        diagnostics.update({
            "mean_accepted_off_diagonal_mass": native[
                "mean_accepted_off_diagonal_mass"],
            "mean_proposed_off_diagonal_mass": native[
                "mean_proposed_off_diagonal_mass"],
            "acceptance_mass_ratio": native["acceptance_mass_ratio"],
            "min_row_acceptance_ratio": native[
                "min_row_acceptance_ratio"],
            "mean_stay_probability": native["mean_stay_probability"],
            "max_stay_probability": native["max_stay_probability"],
            "reverse_missing_edge_fraction": native[
                "reverse_missing_edge_fraction"],
            "detailed_balance_error": native["detailed_balance_error"],
        })
    elif correction == "ipfp":
        diagnostics.update({
            "ipfp_iterations": int(native["ipfp_iterations"]),
            "ipfp_stationary_residual": native[
                "ipfp_stationary_residual"],
            "ipfp_kl_divergence": native["ipfp_kl_divergence"],
            "ipfp_max_probability_change": native[
                "ipfp_max_probability_change"],
            "mean_stay_probability": native["mean_stay_probability"],
            "max_stay_probability": native["max_stay_probability"],
        })
    if return_diagnostics:
        return tau, weights, transition, diagnostics
    return tau, weights, transition


def jacobi_sparse_fixed_grid_transition(
        kappa,
        m,
        xi,
        *,
        n_obs,
        quad_order=128,
        gh_order=5,
        return_grad=False,
        memory_budget_bytes=None,
        return_diagnostics=False):
    """Build a sparse fixed-grid local-GH transition and derivatives."""
    quad_order = _validate_jacobi_order(quad_order, "quad_order")
    gh_order = _validate_jacobi_order(gh_order, "gh_order")
    n_obs = _validate_transition_count(n_obs)
    budget = (
        DEFAULT_JACOBI_MEMORY_BUDGET_BYTES
        if memory_budget_bytes is None else memory_budget_bytes)
    (
        tau,
        weights,
        indices,
        probabilities,
        counts,
        dprobabilities,
        native,
    ) = jacobi_native.sparse_transition(
        kappa,
        m,
        xi,
        n_obs=n_obs,
        quad_order=quad_order,
        basis_order=1,
        gh_order=gh_order,
        method="local_fixed",
        return_grad=return_grad,
        memory_budget_bytes=budget,
    )
    transition = SparseJacobiTransition(
        indices=indices,
        probabilities=probabilities,
        counts=counts,
    )
    diagnostics = {
        "dt": native["dt"],
        "alpha": native["alpha"],
        "beta": native["beta"],
        "gh_order": native["gh_order"],
        "transition_method": "local_fixed_sparse",
        "correction": "none",
        "nnz": int(native["nnz"]),
        "max_width": int(native["max_width"]),
        "retained_bytes": int(native["retained_bytes"]),
        "dense_bytes": int(native["dense_bytes"]),
        "stationary_error": native["stationary_error"],
    }
    if return_grad and return_diagnostics:
        return (
            tau,
            weights,
            transition,
            dprobabilities,
            diagnostics,
        )
    if return_grad:
        return tau, weights, transition, dprobabilities
    if return_diagnostics:
        return tau, weights, transition, diagnostics
    return tau, weights, transition


def sparse_jacobi_full_horizon_diagnostics(
        tau, weights, transition, *, steps, kappa, m):
    """Return deterministic stationarity and first-moment diagnostics."""
    steps = validate_integer(steps, "steps")
    return jacobi_native.sparse_full_horizon_diagnostics(
        kappa,
        m,
        1.0,
        tau,
        weights,
        transition.indices,
        transition.probabilities,
        transition.counts,
        steps,
    )


def select_sparse_jacobi_order(
        kappa,
        m,
        xi,
        *,
        n_obs,
        quad_orders=(48, 80, 128, 192, 384, 768),
        max_quad_order=None,
        basis_order=32,
        gh_order=5,
        max_full_horizon_tv=0.02,
        max_relative_variance_error=0.10,
        max_conditional_mean_rmse=1e-3,
        max_lag_one_correlation_error=1e-2,
        memory_budget_bytes=None,
        require_pass=False):
    """Select the first sparse local grid satisfying full-horizon gates.

    This is an explicit prototype: it never changes an already fitted
    model's order and performs no random sampling.
    """
    n_obs = validate_positive_int(n_obs, "n_obs")
    if n_obs < 2:
        raise ValueError("n_obs must be at least 2 for adaptive selection")
    steps = max(n_obs - 1, 0)
    basis_order = _validate_jacobi_order(basis_order, "basis_order")
    gh_order = _validate_jacobi_order(gh_order, "gh_order")
    if isinstance(quad_orders, (str, bytes)):
        raise TypeError("quad_orders must be an iterable of integers")
    orders = tuple(
        _validate_jacobi_order(order, "quad_orders")
        for order in quad_orders)
    if not orders:
        raise ValueError("quad_orders must not be empty")
    if any(right <= left for left, right in zip(orders, orders[1:])):
        raise ValueError("quad_orders must be strictly increasing")
    if max_quad_order is not None:
        max_quad_order = _validate_jacobi_order(
            max_quad_order, "max_quad_order")
        orders = tuple(order for order in orders if order <= max_quad_order)
        if not orders:
            raise ValueError("max_quad_order excludes every candidate")

    thresholds = {
        "full_horizon_stationary_tv": float(max_full_horizon_tv),
        "relative_variance_error": float(max_relative_variance_error),
        "conditional_mean_rmse": float(max_conditional_mean_rmse),
        "absolute_lag_one_correlation_error": float(
            max_lag_one_correlation_error),
    }
    if any(
            not np.isfinite(value) or value < 0.0
            for value in thresholds.values()):
        raise ValueError(
            "adaptive Jacobi thresholds must be finite and non-negative")

    budget = (
        DEFAULT_JACOBI_MEMORY_BUDGET_BYTES
        if memory_budget_bytes is None else memory_budget_bytes)
    (
        tau,
        weights,
        indices,
        probabilities,
        counts,
        _,
        report,
    ) = jacobi_native.select_sparse_order(
        kappa,
        m,
        xi,
        n_obs=n_obs,
        quad_orders=orders,
        basis_order=basis_order,
        gh_order=gh_order,
        max_full_horizon_tv=max_full_horizon_tv,
        max_relative_variance_error=max_relative_variance_error,
        max_conditional_mean_rmse=max_conditional_mean_rmse,
        max_lag_one_correlation_error=max_lag_one_correlation_error,
        memory_budget_bytes=budget,
        require_pass=require_pass,
    )
    transition = SparseJacobiTransition(indices, probabilities, counts)
    report["thresholds"] = thresholds
    return tau, weights, transition, report


def compare_sparse_jacobi_corrections(
        kappa,
        m,
        xi,
        *,
        n_obs,
        quad_orders=(48, 80, 128),
        corrections=("none", "mh", "ipfp"),
        basis_order=32,
        gh_order=5,
        memory_budget_bytes=None):
    """Compare correction candidates and higher orders deterministically."""
    n_obs = validate_positive_int(n_obs, "n_obs")
    if n_obs < 2:
        raise ValueError("n_obs must be at least 2")
    orders = tuple(
        _validate_jacobi_order(order, "quad_orders")
        for order in quad_orders)
    if not orders:
        raise ValueError("quad_orders must not be empty")
    records = []
    for order in orders:
        for correction in corrections:
            try:
                tau, weights, transition, construction = (
                    jacobi_sparse_local_transition(
                        kappa,
                        m,
                        xi,
                        n_obs=n_obs,
                        quad_order=order,
                        basis_order=jacobi_native.resolve_basis_order(
                            basis_order, order),
                        gh_order=gh_order,
                        correction=correction,
                        memory_budget_bytes=memory_budget_bytes,
                        return_diagnostics=True,
                    )
                )
                horizon = sparse_jacobi_full_horizon_diagnostics(
                    tau,
                    weights,
                    transition,
                    steps=jacobi_native.horizon_steps(n_obs),
                    kappa=kappa,
                    m=m,
                )
                records.append({
                    "quad_order": order,
                    "correction": correction,
                    "status": "ok",
                    **construction,
                    **horizon,
                })
            except (FloatingPointError, MemoryError) as exc:
                records.append({
                    "quad_order": order,
                    "correction": correction,
                    "status": "unsupported",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })
    return records


def jacobi_sparse_matrix_loglik(
        kappa,
        m,
        xi,
        u,
        copula,
        *,
        basis_order=32,
        quad_order=None,
        theta_cap=None,
        transition_method="local",
        gh_order=5,
        correction="none",
        memory_budget_bytes=None):
    """Evaluate a local-GH Jacobi likelihood with sparse filtering."""
    evaluator = jacobi_native.PreparedScarJacobiEvaluator(
        u,
        copula,
        basis_order=basis_order,
        quad_order=quad_order,
        theta_cap=theta_cap,
        transition_method=transition_method,
        storage="sparse",
        correction=correction,
        gh_order=gh_order,
        memory_budget_bytes=(
            DEFAULT_JACOBI_MEMORY_BUDGET_BYTES
            if memory_budget_bytes is None else memory_budget_bytes),
    )
    return evaluator.loglik(kappa, m, xi)


def jacobi_sparse_matrix_neg_loglik_with_grad(
        kappa,
        m,
        xi,
        u,
        copula,
        *,
        basis_order=32,
        quad_order=None,
        theta_cap=None,
        transition_method="local_fixed",
        gh_order=5,
        correction="none",
        memory_budget_bytes=None):
    """Return fixed-grid sparse negative likelihood and exact gradient."""
    if transition_method != "local_fixed" or correction != "none":
        raise ValueError(
            "sparse analytical gradient requires uncorrected "
            "transition_method='local_fixed'")
    basis_order = _validate_jacobi_order(basis_order, "basis_order")
    if quad_order is None:
        from pyscarcopula.numerical.jacobi_tm import default_quad_order

        quad_order = default_quad_order(basis_order)
    quad_order = _validate_jacobi_order(quad_order, "quad_order")
    gh_order = _validate_jacobi_order(gh_order, "gh_order")
    evaluator = jacobi_native.PreparedScarJacobiEvaluator(
        u,
        copula,
        basis_order=basis_order,
        quad_order=quad_order,
        theta_cap=theta_cap,
        transition_method=transition_method,
        storage="sparse",
        correction=correction,
        gh_order=gh_order,
        memory_budget_bytes=(
            DEFAULT_JACOBI_MEMORY_BUDGET_BYTES
            if memory_budget_bytes is None else memory_budget_bytes),
    )
    return evaluator.neg_loglik_with_grad(kappa, m, xi)


def jacobi_sparse_matrix_forward_predictive_mean(
        kappa,
        m,
        xi,
        u,
        copula,
        *,
        basis_order=32,
        quad_order=None,
        theta_cap=None,
        transition_method="local",
        gh_order=5,
        correction="none",
        memory_budget_bytes=None):
    """Return sparse local-GH predictive copula-parameter means."""
    evaluator = jacobi_native.PreparedScarJacobiEvaluator(
        u, copula, basis_order=basis_order, quad_order=quad_order,
        theta_cap=theta_cap, transition_method=transition_method,
        storage="sparse", correction=correction, gh_order=gh_order,
        memory_budget_bytes=(
            DEFAULT_JACOBI_MEMORY_BUDGET_BYTES
            if memory_budget_bytes is None else memory_budget_bytes))
    return evaluator.predictive_mean(kappa, m, xi)


def jacobi_sparse_matrix_forward_mixture_h(
        kappa,
        m,
        xi,
        u,
        copula,
        *,
        basis_order=32,
        quad_order=None,
        theta_cap=None,
        transition_method="local",
        gh_order=5,
        correction="none",
        memory_budget_bytes=None):
    """Return sparse local-GH predictive h-function mixtures."""
    evaluator = jacobi_native.PreparedScarJacobiEvaluator(
        u, copula, basis_order=basis_order, quad_order=quad_order,
        theta_cap=theta_cap, transition_method=transition_method,
        storage="sparse", correction=correction, gh_order=gh_order,
        memory_budget_bytes=(
            DEFAULT_JACOBI_MEMORY_BUDGET_BYTES
            if memory_budget_bytes is None else memory_budget_bytes))
    return evaluator.mixture_h(kappa, m, xi)


def jacobi_sparse_matrix_forward_mixture_h_pair(
        kappa,
        m,
        xi,
        u,
        copula,
        *,
        basis_order=32,
        quad_order=None,
        theta_cap=None,
        transition_method="local",
        gh_order=5,
        correction="none",
        memory_budget_bytes=None):
    """Return both sparse local-GH predictive h-function directions."""
    evaluator = jacobi_native.PreparedScarJacobiEvaluator(
        u, copula, basis_order=basis_order, quad_order=quad_order,
        theta_cap=theta_cap, transition_method=transition_method,
        storage="sparse", correction=correction, gh_order=gh_order,
        memory_budget_bytes=(
            DEFAULT_JACOBI_MEMORY_BUDGET_BYTES
            if memory_budget_bytes is None else memory_budget_bytes))
    return evaluator.mixture_h_pair(kappa, m, xi)


def jacobi_sparse_matrix_state_distribution(
        kappa,
        m,
        xi,
        u,
        copula,
        *,
        basis_order=32,
        quad_order=None,
        theta_cap=None,
        transition_method="local",
        gh_order=5,
        correction="none",
        memory_budget_bytes=None,
        horizon="current"):
    """Return a sparse local-GH current or next state distribution."""
    horizon = str(horizon).lower()
    if horizon not in {"current", "next"}:
        raise ValueError("horizon must be 'current' or 'next'")
    evaluator = jacobi_native.PreparedScarJacobiEvaluator(
        u, copula, basis_order=basis_order, quad_order=quad_order,
        theta_cap=theta_cap, transition_method=transition_method,
        storage="sparse", correction=correction, gh_order=gh_order,
        memory_budget_bytes=(
            DEFAULT_JACOBI_MEMORY_BUDGET_BYTES
            if memory_budget_bytes is None else memory_budget_bytes))
    return evaluator.state_distribution(kappa, m, xi, horizon=horizon)


def sample_sparse_jacobi_trajectory(
        tau, weights, transition, n, *, rng=None,
        memory_budget_bytes=None):
    """Sample one path from a prepared sparse Jacobi transition."""
    n = validate_integer(n, "n")
    tau_values = as_float64_array(tau, name="tau")
    weight_values = as_float64_array(weights, name="weights")
    if n == 0:
        validate_float64_allocation(
            (0,),
            name="sparse Jacobi trajectory",
            memory_budget_bytes=memory_budget_bytes,
        )
        return np.empty(0, dtype=np.float64)
    retained_bytes = int(transition.retained_bytes)
    grid_bytes = int(tau_values.nbytes + weight_values.nbytes)
    path_bytes = validate_float64_allocation(
        (n,), name="sparse Jacobi trajectory")
    # Conservatively allow for contiguous adapter copies and C++ vector
    # copies of every prepared input while the caller uniforms, native path,
    # and returned NumPy path are simultaneously live.
    required_bytes = (
        2 * retained_bytes + 2 * grid_bytes + 4 * path_bytes)
    validate_float64_allocation(
        (required_bytes // np.dtype(np.float64).itemsize,),
        name="prepared sparse Jacobi fixed-draw boundary",
        memory_budget_bytes=memory_budget_bytes,
    )
    if rng is None:
        rng = np.random.default_rng()
    uniforms = np.asarray(rng.random(n), dtype=np.float64)
    path, _ = jacobi_native.sample_prepared_sparse_trajectory_fixed_draws(
        tau_values,
        weight_values,
        transition.indices,
        transition.probabilities,
        transition.counts,
        uniforms,
    )
    return path


__all__ = [
    "SparseJacobiTransition",
    "compare_sparse_jacobi_corrections",
    "jacobi_sparse_fixed_grid_transition",
    "jacobi_sparse_matrix_forward_mixture_h",
    "jacobi_sparse_matrix_forward_mixture_h_pair",
    "jacobi_sparse_matrix_forward_predictive_mean",
    "jacobi_sparse_matrix_loglik",
    "jacobi_sparse_matrix_neg_loglik_with_grad",
    "jacobi_sparse_matrix_state_distribution",
    "jacobi_sparse_local_transition",
    "sample_sparse_jacobi_trajectory",
    "select_sparse_jacobi_order",
    "sparse_jacobi_full_horizon_diagnostics",
]
