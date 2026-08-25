"""Sparse local-transition operators for the Jacobi latent process."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from pyscarcopula._native import jacobi as jacobi_native
from pyscarcopula._utils import clip_h_function_values
from pyscarcopula.numerical._arrays import (
    validate_float64_allocation,
    validate_positive_int,
)
from pyscarcopula.numerical._transition_methods import (
    normalize_jacobi_matrix_transition_method,
)
from pyscarcopula.numerical.jacobi_tm import (
    DEFAULT_JACOBI_MEMORY_BUDGET_BYTES,
    _emission_grid,
    _fixed_tau_weight_derivatives,
    _h_grid_on_theta,
    _h_pair_grids_on_theta,
    _validate_jacobi_order,
    _validate_nonnegative_int,
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
        probabilities = np.asarray(self.probabilities, dtype=np.float64)
        counts = np.asarray(self.counts, dtype=np.intp)
        if indices.ndim != 2 or probabilities.shape != indices.shape:
            raise ValueError(
                "indices and probabilities must have the same 2D shape")
        if counts.shape != (indices.shape[0],):
            raise ValueError("counts must have shape (n_rows,)")
        if np.any(counts < 1) or np.any(counts > indices.shape[1]):
            raise ValueError("counts are outside the sparse row width")
        if np.any(~np.isfinite(probabilities)) or np.any(
                probabilities < 0.0):
            raise ValueError(
                "probabilities must be finite and non-negative")
        n_rows = indices.shape[0]
        for row in range(n_rows):
            count = int(counts[row])
            active_indices = indices[row, :count]
            active_probabilities = probabilities[row, :count]
            if np.any(active_indices < 0) or np.any(
                    active_indices >= n_rows):
                raise ValueError("active sparse indices are out of range")
            if np.any(np.diff(active_indices) <= 0):
                raise ValueError(
                    "active sparse indices must be strictly increasing")
            if not np.isclose(
                    np.sum(active_probabilities), 1.0,
                    rtol=0.0,
                    atol=1e-14):
                raise ValueError("sparse transition rows must sum to one")
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
        vector = np.asarray(vector, dtype=np.float64)
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
        return _sparse_to_dense(
            self.indices, self.probabilities, self.counts)


@njit(cache=True, nogil=True)
def _sparse_filter_loglik_kernel(
        weights, indices, probabilities, counts, fi_grid):
    predicted = weights.copy()
    posterior = np.empty_like(predicted)
    log_likelihood = 0.0
    for observation in range(fi_grid.shape[0]):
        scale = 0.0
        for node in range(predicted.size):
            value = predicted[node] * fi_grid[observation, node]
            posterior[node] = value
            scale += value
        if not np.isfinite(scale) or scale <= 0.0:
            return -np.inf
        inverse_scale = 1.0 / scale
        for node in range(predicted.size):
            posterior[node] *= inverse_scale
        log_likelihood += np.log(scale)
        if observation < fi_grid.shape[0] - 1:
            predicted.fill(0.0)
            for row in range(posterior.size):
                source_mass = posterior[row]
                for slot in range(counts[row]):
                    predicted[indices[row, slot]] += (
                        source_mass * probabilities[row, slot])
            total = 0.0
            for node in range(predicted.size):
                total += predicted[node]
            if not np.isfinite(total) or total <= 0.0:
                return -np.inf
            inverse_total = 1.0 / total
            for node in range(predicted.size):
                predicted[node] *= inverse_total
    return log_likelihood


@njit(cache=True, nogil=True)
def _sparse_neg_loglik_grad_kernel(
        weights,
        dweights,
        indices,
        probabilities,
        dprobabilities,
        counts,
        fi_grid):
    predicted = weights.copy()
    dpredicted = dweights.copy()
    posterior = np.empty_like(predicted)
    dposterior = np.empty_like(dpredicted)
    next_predicted = np.empty_like(predicted)
    next_dpredicted = np.empty_like(dpredicted)
    log_likelihood = 0.0
    gradient = np.zeros(3, dtype=np.float64)

    for observation in range(fi_grid.shape[0]):
        scale = 0.0
        dscale = np.zeros(3, dtype=np.float64)
        for node in range(predicted.size):
            emission = fi_grid[observation, node]
            posterior[node] = predicted[node] * emission
            scale += posterior[node]
            for parameter in range(3):
                dposterior[parameter, node] = (
                    dpredicted[parameter, node] * emission)
                dscale[parameter] += dposterior[parameter, node]
        if not np.isfinite(scale) or scale <= 0.0:
            return 1e10, np.zeros(3, dtype=np.float64)
        for node in range(predicted.size):
            raw_probability = posterior[node]
            posterior[node] = raw_probability / scale
            for parameter in range(3):
                dposterior[parameter, node] = (
                    dposterior[parameter, node] * scale
                    - raw_probability * dscale[parameter]
                ) / (scale * scale)
        log_likelihood += np.log(scale)
        gradient += dscale / scale

        if observation < fi_grid.shape[0] - 1:
            next_predicted.fill(0.0)
            next_dpredicted.fill(0.0)
            for row in range(posterior.size):
                for slot in range(counts[row]):
                    target = indices[row, slot]
                    probability = probabilities[row, slot]
                    next_predicted[target] += (
                        posterior[row] * probability)
                    for parameter in range(3):
                        next_dpredicted[parameter, target] += (
                            dposterior[parameter, row] * probability
                            + posterior[row]
                            * dprobabilities[parameter, row, slot]
                        )
            total = 0.0
            dtotal = np.zeros(3, dtype=np.float64)
            for node in range(next_predicted.size):
                total += next_predicted[node]
                for parameter in range(3):
                    dtotal[parameter] += next_dpredicted[parameter, node]
            if not np.isfinite(total) or total <= 0.0:
                return 1e10, np.zeros(3, dtype=np.float64)
            for node in range(next_predicted.size):
                raw_probability = next_predicted[node]
                predicted[node] = raw_probability / total
                for parameter in range(3):
                    dpredicted[parameter, node] = (
                        next_dpredicted[parameter, node] * total
                        - raw_probability * dtotal[parameter]
                    ) / (total * total)
    if (
            not np.isfinite(log_likelihood)
            or np.any(~np.isfinite(gradient))):
        return 1e10, np.zeros(3, dtype=np.float64)
    return -log_likelihood, -gradient


@njit(cache=True, nogil=True)
def _sparse_to_dense(indices, probabilities, counts):
    n_rows = indices.shape[0]
    dense = np.zeros((n_rows, n_rows), dtype=np.float64)
    for row in range(n_rows):
        for slot in range(counts[row]):
            dense[row, indices[row, slot]] = probabilities[row, slot]
    return dense


@njit(cache=True, nogil=True)
def _sample_sparse_path_kernel(
        tau, indices, probabilities, counts, uniforms, initial_index):
    path = np.empty(uniforms.size, dtype=np.float64)
    index = initial_index
    path[0] = tau[index]
    for observation in range(1, uniforms.size):
        draw = uniforms[observation]
        cumulative = 0.0
        selected = counts[index] - 1
        for slot in range(counts[index]):
            cumulative += probabilities[index, slot]
            if draw < cumulative:
                selected = slot
                break
        index = indices[index, selected]
        path[observation] = tau[index]
    return path


def _validate_sparse_filter_workspace(
        *,
        n_obs,
        quad_order,
        gh_order,
        gradient=False,
        correction="none",
        memory_budget_bytes=None):
    if memory_budget_bytes is None:
        memory_budget_bytes = DEFAULT_JACOBI_MEMORY_BUDGET_BYTES
    width = 2 * gh_order + 1
    # Transition indices/probabilities/counts, filter vectors, and the
    # emission plus two directional h grids used by the largest operation.
    transition_arrays = {
        "none": 2,
        "mh": 4,
        "ipfp": 5,
    }[correction]
    elements = (
        transition_arrays * quad_order * width
        + 7 * quad_order
        + 3 * n_obs * quad_order)
    if gradient:
        elements += (
            3 * quad_order * width
            + 8 * quad_order)
    return validate_float64_allocation(
        (elements,),
        name="sparse Jacobi filtering workspace",
        memory_budget_bytes=memory_budget_bytes,
    )


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
    steps = _validate_nonnegative_int(steps, "steps")
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
                        basis_order=min(basis_order, order),
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
                    steps=n_obs - 1,
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


def _iter_sparse_filter(weights, transition, fi_grid):
    predicted = np.asarray(weights, dtype=np.float64).copy()
    for observation in range(fi_grid.shape[0]):
        weighted = predicted * fi_grid[observation]
        scale = float(np.sum(weighted))
        if not np.isfinite(scale) or scale <= 0.0:
            raise FloatingPointError("sparse Jacobi filter update failed")
        posterior = weighted / scale
        posterior_total = float(np.sum(posterior))
        if not np.isfinite(posterior_total) or posterior_total <= 0.0:
            raise FloatingPointError("sparse Jacobi posterior is invalid")
        posterior /= posterior_total
        yield observation, predicted, posterior, scale
        if observation < fi_grid.shape[0] - 1:
            predicted = transition.left_multiply(posterior)
            total = float(np.sum(predicted))
            if not np.isfinite(total) or total <= 0.0:
                raise FloatingPointError("sparse Jacobi prediction failed")
            predicted /= total


def _sparse_filter_setup(
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
    u = np.asarray(u, dtype=np.float64)
    if u.ndim != 2 or u.shape[1] != 2 or len(u) < 1:
        return None
    basis_order = _validate_jacobi_order(basis_order, "basis_order")
    if quad_order is None:
        from pyscarcopula.numerical.jacobi_tm import default_quad_order

        quad_order = default_quad_order(basis_order)
    quad_order = _validate_jacobi_order(quad_order, "quad_order")
    gh_order = _validate_jacobi_order(gh_order, "gh_order")
    transition_method = normalize_jacobi_matrix_transition_method(
        transition_method)
    if transition_method not in {"local", "local_fixed"}:
        raise ValueError(
            "sparse filtering requires transition_method='local' "
            "or 'local_fixed'")
    if transition_method == "local_fixed" and correction != "none":
        raise ValueError(
            "stationarity correction is not supported for sparse "
            "local_fixed")
    _validate_sparse_filter_workspace(
        n_obs=len(u),
        quad_order=quad_order,
        gh_order=gh_order,
        correction=correction,
        memory_budget_bytes=memory_budget_bytes,
    )
    if transition_method == "local_fixed":
        tau, weights, transition = (
            jacobi_sparse_fixed_grid_transition(
                kappa,
                m,
                xi,
                n_obs=len(u),
                quad_order=quad_order,
                gh_order=gh_order,
                memory_budget_bytes=memory_budget_bytes,
            )
        )
    else:
        tau, weights, transition = jacobi_sparse_local_transition(
            kappa,
            m,
            xi,
            n_obs=len(u),
            basis_order=basis_order,
            quad_order=quad_order,
            gh_order=gh_order,
            correction=correction,
            memory_budget_bytes=memory_budget_bytes,
        )
    fi_grid, theta = _emission_grid(
        u, copula, tau, theta_cap=theta_cap)
    return u, tau, weights, transition, fi_grid, theta


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
    try:
        setup = _sparse_filter_setup(
            kappa,
            m,
            xi,
            u,
            copula,
            basis_order=basis_order,
            quad_order=quad_order,
            theta_cap=theta_cap,
            transition_method=transition_method,
            gh_order=gh_order,
            correction=correction,
            memory_budget_bytes=memory_budget_bytes,
        )
    except MemoryError:
        raise
    except Exception:
        return -np.inf
    if setup is None:
        return -np.inf
    _, _, weights, transition, fi_grid, _ = setup
    return float(_sparse_filter_loglik_kernel(
        weights,
        transition.indices,
        transition.probabilities,
        transition.counts,
        fi_grid,
    ))


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
    fail = 1e10, np.zeros(3, dtype=np.float64)
    if transition_method != "local_fixed" or correction != "none":
        raise ValueError(
            "sparse analytical gradient requires uncorrected "
            "transition_method='local_fixed'")
    u = np.asarray(u, dtype=np.float64)
    if u.ndim != 2 or u.shape[1] != 2 or len(u) < 1:
        return fail
    basis_order = _validate_jacobi_order(basis_order, "basis_order")
    if quad_order is None:
        from pyscarcopula.numerical.jacobi_tm import default_quad_order

        quad_order = default_quad_order(basis_order)
    quad_order = _validate_jacobi_order(quad_order, "quad_order")
    gh_order = _validate_jacobi_order(gh_order, "gh_order")
    _validate_sparse_filter_workspace(
        n_obs=len(u),
        quad_order=quad_order,
        gh_order=gh_order,
        gradient=True,
        correction="none",
        memory_budget_bytes=memory_budget_bytes,
    )
    try:
        tau, weights, transition, dprobabilities = (
            jacobi_sparse_fixed_grid_transition(
                kappa,
                m,
                xi,
                n_obs=len(u),
                quad_order=quad_order,
                gh_order=gh_order,
                return_grad=True,
                memory_budget_bytes=memory_budget_bytes,
            )
        )
        dweights = _fixed_tau_weight_derivatives(
            kappa, m, xi, tau, weights)
        fi_grid, _ = _emission_grid(
            u, copula, tau, theta_cap=theta_cap)
    except MemoryError:
        raise
    except Exception:
        return fail
    value, gradient = _sparse_neg_loglik_grad_kernel(
        weights,
        dweights,
        transition.indices,
        transition.probabilities,
        dprobabilities,
        transition.counts,
        fi_grid,
    )
    return float(value), np.asarray(gradient, dtype=np.float64)


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
    setup = _sparse_filter_setup(
        kappa,
        m,
        xi,
        u,
        copula,
        basis_order=basis_order,
        quad_order=quad_order,
        theta_cap=theta_cap,
        transition_method=transition_method,
        gh_order=gh_order,
        correction=correction,
        memory_budget_bytes=memory_budget_bytes,
    )
    if setup is None:
        raise ValueError("invalid Jacobi parameters or observations")
    _, _, weights, transition, fi_grid, theta = setup
    output = np.empty(len(fi_grid), dtype=np.float64)
    for observation, predicted, _, _ in _iter_sparse_filter(
            weights, transition, fi_grid):
        output[observation] = np.sum(predicted * theta)
    return output


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
    setup = _sparse_filter_setup(
        kappa,
        m,
        xi,
        u,
        copula,
        basis_order=basis_order,
        quad_order=quad_order,
        theta_cap=theta_cap,
        transition_method=transition_method,
        gh_order=gh_order,
        correction=correction,
        memory_budget_bytes=memory_budget_bytes,
    )
    if setup is None:
        raise ValueError("invalid Jacobi parameters or observations")
    u, _, weights, transition, fi_grid, theta = setup
    h_grid = _h_grid_on_theta(copula, u, theta)
    output = np.empty(len(fi_grid), dtype=np.float64)
    for observation, predicted, _, _ in _iter_sparse_filter(
            weights, transition, fi_grid):
        output[observation] = np.sum(
            predicted * h_grid[observation])
    return clip_h_function_values(output)


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
    setup = _sparse_filter_setup(
        kappa,
        m,
        xi,
        u,
        copula,
        basis_order=basis_order,
        quad_order=quad_order,
        theta_cap=theta_cap,
        transition_method=transition_method,
        gh_order=gh_order,
        correction=correction,
        memory_budget_bytes=memory_budget_bytes,
    )
    if setup is None:
        raise ValueError("invalid Jacobi parameters or observations")
    u, _, weights, transition, fi_grid, theta = setup
    first_grid, second_grid = _h_pair_grids_on_theta(copula, u, theta)
    first = np.empty(len(fi_grid), dtype=np.float64)
    second = np.empty(len(fi_grid), dtype=np.float64)
    for observation, predicted, _, _ in _iter_sparse_filter(
            weights, transition, fi_grid):
        first[observation] = np.sum(
            predicted * first_grid[observation])
        second[observation] = np.sum(
            predicted * second_grid[observation])
    return (
        clip_h_function_values(first),
        clip_h_function_values(second),
    )


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
    setup = _sparse_filter_setup(
        kappa,
        m,
        xi,
        u,
        copula,
        basis_order=basis_order,
        quad_order=quad_order,
        theta_cap=theta_cap,
        transition_method=transition_method,
        gh_order=gh_order,
        correction=correction,
        memory_budget_bytes=memory_budget_bytes,
    )
    if setup is None:
        raise ValueError("invalid Jacobi parameters or observations")
    _, tau, weights, transition, fi_grid, _ = setup
    posterior = None
    for _, _, posterior, _ in _iter_sparse_filter(
            weights, transition, fi_grid):
        pass
    probability = posterior
    if horizon == "next":
        probability = transition.left_multiply(probability)
        probability /= np.sum(probability)
    return tau.copy(), probability


def sample_sparse_jacobi_trajectory(
        tau, weights, transition, n, *, rng=None,
        memory_budget_bytes=None):
    """Sample one path from a prepared sparse Jacobi transition."""
    n = _validate_nonnegative_int(n, "n")
    validate_float64_allocation(
        (n,),
        name="sparse Jacobi trajectory",
        memory_budget_bytes=memory_budget_bytes,
    )
    if n == 0:
        return np.empty(0, dtype=np.float64)
    if rng is None:
        rng = np.random.default_rng()
    uniforms = np.asarray(rng.random(n), dtype=np.float64)
    stationary_cdf = np.cumsum(weights)
    stationary_cdf[-1] = 1.0
    initial_index = int(np.searchsorted(
        stationary_cdf, uniforms[0], side="right"))
    return _sample_sparse_path_kernel(
        np.asarray(tau, dtype=np.float64),
        transition.indices,
        transition.probabilities,
        transition.counts,
        uniforms,
        initial_index,
    )


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
