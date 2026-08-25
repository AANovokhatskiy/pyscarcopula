"""Jacobi spectral likelihood for copulas driven by Kendall's tau."""

from __future__ import annotations

import numpy as np

from pyscarcopula._native import jacobi as jacobi_native
from pyscarcopula.numerical._arrays import (
    validate_float64_allocation,
    validate_positive_int,
)
from pyscarcopula.numerical._transition_methods import (
    normalize_jacobi_stationarity_correction,
    normalize_jacobi_matrix_transition_method,
    normalize_jacobi_strategy_transition_method,
    normalize_jacobi_transition_storage,
)


_validate_positive_int = validate_positive_int
MAX_JACOBI_ORDER = 2048
DEFAULT_JACOBI_MEMORY_BUDGET_BYTES = 1024 ** 3


def _validate_jacobi_order(value, name):
    value = _validate_positive_int(value, name)
    if value > MAX_JACOBI_ORDER:
        raise ValueError(
            f"{name} must be <= {MAX_JACOBI_ORDER}; larger Jacobi grids "
            "are disabled to prevent unsafe quadratic allocations")
    return value


def _validate_nonnegative_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)):
        raise TypeError(f"{name} must be a non-negative integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _raise_jacobi_memory_error(exc):
    raise MemoryError(
        f"{exc}; reduce quad_order, basis_order, or the observation "
        "count, or increase memory_budget_bytes") from exc


def _validate_jacobi_workspace(
        *, quad_order, basis_order=1, n_obs=0, gradient=False,
        matrix=True, gh_order=1, memory_budget_bytes=None):
    """Preflight a conservative upper bound for simultaneous float64 arrays."""
    quad_order = _validate_jacobi_order(quad_order, "quad_order")
    basis_order = _validate_jacobi_order(basis_order, "basis_order")
    gh_order = _validate_jacobi_order(gh_order, "gh_order")
    if basis_order > quad_order:
        raise ValueError("quad_order must be >= basis_order")
    if isinstance(n_obs, (bool, np.bool_)) or not isinstance(
            n_obs, (int, np.integer)):
        raise TypeError("n_obs must be a non-negative integer")
    n_obs = int(n_obs)
    if n_obs < 0:
        raise ValueError("n_obs must be non-negative")

    k = quad_order
    b = basis_order
    if memory_budget_bytes is None:
        memory_budget_bytes = DEFAULT_JACOBI_MEMORY_BUDGET_BYTES

    try:
        return jacobi_native.estimate_workspace(
            quad_order=k,
            basis_order=b,
            n_obs=n_obs,
            gradient=gradient,
            matrix=matrix,
            gh_order=gh_order,
            memory_budget_bytes=memory_budget_bytes,
        )
    except MemoryError as exc:
        _raise_jacobi_memory_error(exc)


def _validate_jacobi_sampling_workspace(
        *, n, quad_order, basis_order, gh_order=1,
        memory_budget_bytes=None):
    """Preflight transition construction and fixed-draw boundary copies."""
    n = _validate_nonnegative_int(n, "n")
    quad_order = _validate_jacobi_order(quad_order, "quad_order")
    basis_order = _validate_jacobi_order(basis_order, "basis_order")
    gh_order = _validate_jacobi_order(gh_order, "gh_order")
    if basis_order > quad_order:
        raise ValueError("quad_order must be >= basis_order")
    if memory_budget_bytes is None:
        memory_budget_bytes = DEFAULT_JACOBI_MEMORY_BUDGET_BYTES
    k = quad_order
    b = basis_order
    try:
        native_peak_bytes = jacobi_native.estimate_sampling_workspace(
            n=n,
            quad_order=k,
            basis_order=b,
            gh_order=gh_order,
            memory_budget_bytes=memory_budget_bytes,
        )
        # The native estimate already includes its trajectory result.  Across
        # the Python/pybind boundary the caller-owned uniform array, the C++
        # input copy, and the returned NumPy copy are simultaneously live.
        boundary_bytes = validate_float64_allocation(
            (3, n), name="Jacobi fixed-draw boundary buffers")
        required_bytes = native_peak_bytes + boundary_bytes
        validate_float64_allocation(
            (required_bytes // np.dtype(np.float64).itemsize,),
            name="Jacobi sampling workspace",
            memory_budget_bytes=memory_budget_bytes,
        )
        return required_bytes
    except MemoryError as exc:
        _raise_jacobi_memory_error(exc)


def _jacobi_stationary_shape(kappa, m, xi):
    return jacobi_native.stationary_shape(kappa, m, xi)


def default_quad_order(basis_order: int) -> int:
    """Conservative quadrature order for Jacobi projected multiplication."""
    basis_order = _validate_jacobi_order(basis_order, "basis_order")
    quad_order = jacobi_native.default_quad_order(basis_order)
    return _validate_jacobi_order(quad_order, "quad_order")


def _fixed_tau_rule(alpha, beta, quad_order):
    """Return a parameter-independent tau grid and beta stationary masses."""
    quad_order = _validate_jacobi_order(quad_order, "quad_order")
    if quad_order < 2:
        raise ValueError("quad_order must be >= 2")
    tau, weights, _ = jacobi_native.fixed_shape_rule(
        alpha,
        beta,
        quad_order,
        DEFAULT_JACOBI_MEMORY_BUDGET_BYTES,
    )
    return tau, weights


def jacobi_rule(
        alpha, beta, quad_order, basis_order, memory_budget_bytes=None):
    """Return tau nodes, probability weights, and orthonormal Jacobi basis.

    Parameters
    ----------
    alpha, beta : float
        Stationary beta-shape parameters for tau on ``(0, 1)``.
    quad_order : int
        Number of Gauss-Jacobi quadrature nodes.
    basis_order : int
        Number of basis functions, including the constant mode.

    Returns
    -------
    tuple
        ``(tau, weights, basis)`` where ``weights`` sum to one and
        ``basis.T @ diag(weights) @ basis`` is numerically the identity.
    """
    alpha = float(alpha)
    beta = float(beta)
    if not np.isfinite(alpha) or not np.isfinite(beta):
        raise ValueError("alpha and beta must be finite")
    if alpha <= 0.0 or beta <= 0.0:
        raise ValueError("alpha and beta must be positive")

    quad_order = _validate_jacobi_order(quad_order, "quad_order")
    basis_order = _validate_jacobi_order(basis_order, "basis_order")
    if quad_order < basis_order:
        raise ValueError("quad_order must be >= basis_order")
    _validate_jacobi_workspace(
        quad_order=quad_order,
        basis_order=basis_order,
        matrix=False,
        memory_budget_bytes=memory_budget_bytes,
    )

    budget = (
        DEFAULT_JACOBI_MEMORY_BUDGET_BYTES
        if memory_budget_bytes is None
        else memory_budget_bytes)
    return jacobi_native.jacobi_rule(
        alpha, beta, quad_order, basis_order, budget)


def _jacobi_powers(kappa, xi, n_obs, basis_order):
    return _jacobi_transition_powers(kappa, xi, n_obs, basis_order)


def _validate_transition_count(n_obs):
    n_obs = _validate_positive_int(n_obs, "n_obs")
    if n_obs < 2:
        raise ValueError("n_obs must be at least 2")
    return n_obs


def _jacobi_transition_powers(kappa, xi, n_obs, basis_order):
    n_obs = _validate_transition_count(n_obs)
    basis_order = _validate_jacobi_order(basis_order, "basis_order")
    return jacobi_native.transition_powers(kappa, xi, n_obs, basis_order)


def jacobi_spectral_transition_matrix(
        kappa,
        m,
        xi,
        *,
        n_obs,
        basis_order=32,
        quad_order=None,
        clip_negative=False,
        memory_budget_bytes=None,
        return_diagnostics=False):
    """Build a node-space transition matrix from the Jacobi spectral density.

    For ``T = n_obs`` observations on ``[0, 1]``, the transition step is
    always ``delta_t = 1 / (T - 1)``. The truncated transition density is

    ``p_delta_t(y | x) = pi(y) * sum_n exp(-lambda_n * delta_t) q_n(x) q_n(y)``.

    With Gauss-Jacobi nodes and probability weights for ``pi``, the returned
    row-stochastic mass matrix is

    ``T[i, j] = w[j] * sum_n exp(-lambda_n * delta_t) q_n(tau[i]) q_n(tau[j])``.

    Negative entries are possible when the spectral series is truncated,
    especially for small ``delta_t``. By default they are left untouched and
    reported in diagnostics.  Set ``clip_negative=True`` only for exploratory
    node-space filtering with explicit renormalization.
    """
    basis_order = _validate_jacobi_order(basis_order, "basis_order")
    if quad_order is None:
        quad_order = default_quad_order(basis_order)
    quad_order = _validate_jacobi_order(quad_order, "quad_order")
    n_obs = _validate_transition_count(n_obs)
    budget = (
        DEFAULT_JACOBI_MEMORY_BUDGET_BYTES
        if memory_budget_bytes is None else memory_budget_bytes)
    tau, weights, transition, _, _, native = jacobi_native.dense_transition(
        kappa,
        m,
        xi,
        n_obs=n_obs,
        quad_order=quad_order,
        basis_order=basis_order,
        gh_order=1,
        method="spectral_matrix",
        raw_backend="spectral",
        clip_negative=clip_negative,
        memory_budget_bytes=budget,
    )
    diagnostics = {
        "dt": native["dt"],
        "alpha": native["alpha"],
        "beta": native["beta"],
        "raw_min_entry": native["raw_min_entry"],
        "raw_negative_mass": native["raw_negative_mass"],
        "max_row_sum_error_before_normalization": native[
            "max_row_sum_error_before_normalization"],
        "stationary_error": native["stationary_error"],
        "clipped_negative": native["clipped_negative"],
    }

    if return_diagnostics:
        return tau, weights, transition, diagnostics
    return tau, weights, transition


def jacobi_fixed_grid_transition_matrix(
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
    """Build local-GH Jacobi transition on a fixed tau grid.

    The tau nodes are independent of model parameters.  This backend is used
    for analytical-gradient optimization, because all parameter sensitivities
    of the discrete likelihood are explicit: beta initial masses, Lamperti
    drift, inverse Lamperti map, and linear interpolation weights.
    """
    n_obs = _validate_transition_count(n_obs)
    quad_order = _validate_jacobi_order(quad_order, "quad_order")
    gh_order = _validate_jacobi_order(gh_order, "gh_order")
    budget = (
        DEFAULT_JACOBI_MEMORY_BUDGET_BYTES
        if memory_budget_bytes is None else memory_budget_bytes)
    tau, weights, transition, dtransition, _, native = (
        jacobi_native.dense_transition(
            kappa,
            m,
            xi,
            n_obs=n_obs,
            quad_order=quad_order,
            basis_order=1,
            gh_order=gh_order,
            method="local_fixed",
            raw_backend="local_fixed",
            return_grad=return_grad,
            memory_budget_bytes=budget,
        )
    )
    diagnostics = {
        "dt": native["dt"],
        "alpha": native["alpha"],
        "beta": native["beta"],
        "gh_order": native["gh_order"],
        "min_entry": native["min_entry"],
        "max_row_sum_error": native["max_row_sum_error"],
        "stationary_error": native["stationary_error"],
        "transition_method": "local_fixed",
    }
    if return_grad and return_diagnostics:
        return tau, weights, transition, dtransition, diagnostics
    if return_grad:
        return tau, weights, transition, dtransition
    if return_diagnostics:
        return tau, weights, transition, diagnostics
    return tau, weights, transition


def jacobi_local_transition_matrix(
        kappa,
        m,
        xi,
        *,
        n_obs,
        quad_order=128,
        basis_order=1,
        gh_order=5,
        memory_budget_bytes=None,
        return_diagnostics=False):
    """Build a local transition matrix for Jacobi diffusion.

    The transition is approximated in the Lamperti coordinate

    ``y = 2 / xi * asin(sqrt(tau))``,

    where the diffusion coefficient is one.  A one-step local Gaussian
    approximation is then mapped back to ``tau`` and linearly interpolated on
    the Jacobi quadrature grid.  Rows are source nodes and columns are target
    nodes; rows are nonnegative and normalized.
    """
    quad_order = _validate_jacobi_order(quad_order, "quad_order")
    # ``basis_order`` is accepted so callers can use the same constructor
    # signature as the spectral matrix path; only the grid order matters here.
    basis_order = _validate_jacobi_order(basis_order, "basis_order")
    gh_order = _validate_jacobi_order(gh_order, "gh_order")
    n_obs = _validate_transition_count(n_obs)
    budget = (
        DEFAULT_JACOBI_MEMORY_BUDGET_BYTES
        if memory_budget_bytes is None else memory_budget_bytes)
    tau, weights, transition, _, _, native = jacobi_native.dense_transition(
        kappa,
        m,
        xi,
        n_obs=n_obs,
        quad_order=quad_order,
        basis_order=basis_order,
        gh_order=gh_order,
        method="local",
        raw_backend="local",
        memory_budget_bytes=budget,
    )
    diagnostics = {
        "dt": native["dt"],
        "alpha": native["alpha"],
        "beta": native["beta"],
        "gh_order": native["gh_order"],
        "min_entry": native["min_entry"],
        "max_row_sum_error": native["max_row_sum_error"],
        "stationary_error": native["stationary_error"],
    }
    if return_diagnostics:
        return tau, weights, transition, diagnostics
    return tau, weights, transition


def jacobi_transition_matrix(
        kappa,
        m,
        xi,
        *,
        n_obs,
        basis_order=32,
        quad_order=None,
        transition_method="auto",
        clip_negative=False,
        negative_mass_tol=1e-5,
        gh_order=5,
        memory_budget_bytes=None,
        return_diagnostics=False):
    """Build a Jacobi transition matrix using the requested backend.

    ``spectral_matrix`` uses the truncated exact spectral density on the
    Jacobi quadrature grid.  ``local`` uses a local Gaussian step in the
    Lamperti coordinate.  ``auto`` tries the spectral matrix first and falls
    back to ``local`` when truncation creates material negative mass.
    """
    method_requested = normalize_jacobi_matrix_transition_method(
        transition_method)
    if _jacobi_stationary_shape(kappa, m, xi) is None:
        raise ValueError("invalid Jacobi parameters")
    n_obs = _validate_transition_count(n_obs)
    if quad_order is None:
        quad_order = default_quad_order(basis_order)
    basis_order = _validate_jacobi_order(basis_order, "basis_order")
    quad_order = _validate_jacobi_order(quad_order, "quad_order")
    gh_order = _validate_jacobi_order(gh_order, "gh_order")
    budget = (
        DEFAULT_JACOBI_MEMORY_BUDGET_BYTES
        if memory_budget_bytes is None else memory_budget_bytes)
    tau, weights, transition, _, _, native = jacobi_native.dense_transition(
        kappa,
        m,
        xi,
        n_obs=n_obs,
        quad_order=quad_order,
        basis_order=basis_order,
        gh_order=gh_order,
        method=method_requested,
        clip_negative=clip_negative,
        negative_mass_tol=negative_mass_tol,
        memory_budget_bytes=budget,
    )
    method_used = native["transition_method"]
    if method_used == "spectral_matrix":
        diagnostics = {
            "dt": native["dt"],
            "alpha": native["alpha"],
            "beta": native["beta"],
            "raw_min_entry": native["raw_min_entry"],
            "raw_negative_mass": native["raw_negative_mass"],
            "max_row_sum_error_before_normalization": native[
                "max_row_sum_error_before_normalization"],
            "stationary_error": native["stationary_error"],
            "clipped_negative": native["clipped_negative"],
            "transition_method_requested": method_requested,
            "transition_method": method_used,
            "probability_cleanup_applied": native[
                "probability_cleanup_applied"],
            "probability_cleanup_negative_mass": native[
                "probability_cleanup_negative_mass"],
            "probability_min_entry_before_cleanup": native[
                "probability_min_entry_before_cleanup"],
        }
    else:
        diagnostics = {
            "dt": native["dt"],
            "alpha": native["alpha"],
            "beta": native["beta"],
            "gh_order": native["gh_order"],
            "min_entry": native["min_entry"],
            "max_row_sum_error": native["max_row_sum_error"],
            "stationary_error": native["stationary_error"],
            "transition_method_requested": method_requested,
            "transition_method": method_used,
        }
        if method_used == "local_fixed":
            diagnostics["transition_method"] = "local_fixed"
        if int(native["spectral_status"]) != 0:
            diagnostics["spectral_error"] = (
                "FloatingPointError: C++ Jacobi spectral transition failed")
    if return_diagnostics:
        return tau, weights, transition, diagnostics
    return tau, weights, transition


def _legacy_grid_sampling_diagnostics(
        native, *, sampling_method, requested_method, storage, correction, n):
    """Preserve the frozen public diagnostics while C++ owns execution."""
    method_used = native["transition_method"]
    if n == 1:
        return {
            "transition_method_requested": requested_method,
            "sampling_transition_method_requested": sampling_method,
            "transition_method": "stationary_only",
            "n": 1,
        }
    if storage == "sparse":
        diagnostics = {
            "dt": native["dt"],
            "alpha": native["alpha"],
            "beta": native["beta"],
            "gh_order": native["gh_order"],
            "transition_method": sampling_method,
            "correction": correction,
            "nnz": int(native["nnz"]),
            "max_width": int(native["max_width"]),
            "retained_bytes": int(native["retained_bytes"]),
            "dense_bytes": int(native["dense_bytes"]),
            "stationary_error": native["stationary_error"],
        }
        if sampling_method == "local":
            diagnostics["max_row_sum_error"] = native[
                "max_row_sum_error"]
        if correction == "mh":
            for name in (
                    "mean_accepted_off_diagonal_mass",
                    "mean_proposed_off_diagonal_mass",
                    "acceptance_mass_ratio",
                    "min_row_acceptance_ratio",
                    "mean_stay_probability",
                    "max_stay_probability",
                    "reverse_missing_edge_fraction",
                    "detailed_balance_error"):
                diagnostics[name] = native[name]
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
        diagnostics.update({
            "transition_method_requested": requested_method,
            "sampling_transition_method_requested": sampling_method,
            "model_transition_method_requested": requested_method,
            "transition_storage": "sparse",
            "n": n,
        })
        return diagnostics
    if method_used == "spectral_matrix":
        diagnostics = {
            "dt": native["dt"],
            "alpha": native["alpha"],
            "beta": native["beta"],
            "raw_min_entry": native["raw_min_entry"],
            "raw_negative_mass": native["raw_negative_mass"],
            "max_row_sum_error_before_normalization": native[
                "max_row_sum_error_before_normalization"],
            "stationary_error": native["stationary_error"],
            "clipped_negative": native["clipped_negative"],
            "transition_method_requested": sampling_method,
            "transition_method": method_used,
            "probability_cleanup_applied": native[
                "probability_cleanup_applied"],
            "probability_cleanup_negative_mass": native[
                "probability_cleanup_negative_mass"],
            "probability_min_entry_before_cleanup": native[
                "probability_min_entry_before_cleanup"],
        }
    else:
        diagnostics = {
            "dt": native["dt"],
            "alpha": native["alpha"],
            "beta": native["beta"],
            "gh_order": native["gh_order"],
            "min_entry": native["min_entry"],
            "max_row_sum_error": native["max_row_sum_error"],
            "stationary_error": native["stationary_error"],
            "transition_method_requested": sampling_method,
            "transition_method": method_used,
        }
        if int(native["spectral_status"]) != 0:
            diagnostics["spectral_error"] = (
                "FloatingPointError: C++ Jacobi spectral transition failed")
    diagnostics["sampling_transition_method_requested"] = sampling_method
    diagnostics["model_transition_method_requested"] = requested_method
    diagnostics["n"] = n
    return diagnostics


def sample_jacobi_grid_trajectory(
        kappa,
        m,
        xi,
        n,
        *,
        rng=None,
        basis_order=32,
        quad_order=None,
        transition_method="auto",
        clip_negative=False,
        negative_mass_tol=1e-5,
        gh_order=5,
        transition_storage="dense",
        stationarity_correction="none",
        memory_budget_bytes=None,
        return_diagnostics=False):
    """Sample the discrete Jacobi Markov model through fixed-draw C++."""
    n = _validate_nonnegative_int(n, "n")
    stationarity_correction = normalize_jacobi_stationarity_correction(
        stationarity_correction)
    transition_storage = normalize_jacobi_transition_storage(
        transition_storage)
    if n == 0:
        empty = np.empty(0, dtype=np.float64)
        if return_diagnostics:
            return empty, {
                "transition_method_requested": str(transition_method),
                "transition_method": "not_built",
                "n": 0,
            }
        return empty

    if _jacobi_stationary_shape(kappa, m, xi) is None:
        raise ValueError("invalid Jacobi parameters")
    basis_order = _validate_jacobi_order(basis_order, "basis_order")
    if quad_order is None:
        quad_order = default_quad_order(basis_order)
    quad_order = _validate_jacobi_order(quad_order, "quad_order")
    gh_order = _validate_jacobi_order(gh_order, "gh_order")

    requested_method = normalize_jacobi_strategy_transition_method(
        transition_method)
    sampling_method = (
        "auto" if requested_method == "spectral_coeff"
        else requested_method)
    use_sparse_sampling = (
        n > 1
        and (
            sampling_method == "local"
            or (
                sampling_method == "local_fixed"
                and transition_storage == "sparse"
            )
        )
    )
    sampling_storage = "sparse" if use_sparse_sampling else "dense"
    if stationarity_correction != "none" and sampling_method != "local":
        raise ValueError(
            "stationarity_correction currently requires "
            "transition_method='local'")
    budget = (
        DEFAULT_JACOBI_MEMORY_BUDGET_BYTES
        if memory_budget_bytes is None else memory_budget_bytes)
    # Workspace and output preflights remain before the first RNG draw.
    if sampling_storage == "sparse" and n > 1:
        sparse_workspace_bytes = jacobi_native.estimate_sparse_workspace(
            quad_order=quad_order,
            gh_order=gh_order,
            correction=stationarity_correction,
            memory_budget_bytes=budget,
        )
        retained = jacobi_native.estimate_sparse_storage(
            quad_order=quad_order,
            gh_order=gh_order,
            correction=stationarity_correction,
            memory_budget_bytes=budget,
        )
        native_peak_bytes = max(
            sparse_workspace_bytes,
            retained + (2 * quad_order + n) * 8,
        )
        boundary_bytes = validate_float64_allocation(
            (3, n), name="sparse Jacobi fixed-draw boundary buffers")
        required_bytes = native_peak_bytes + boundary_bytes
        validate_float64_allocation(
            (required_bytes // np.dtype(np.float64).itemsize,),
            name="sparse Jacobi sampling workspace",
            memory_budget_bytes=budget,
        )
    else:
        _validate_jacobi_sampling_workspace(
            n=n,
            quad_order=quad_order,
            basis_order=basis_order,
            gh_order=gh_order,
            memory_budget_bytes=budget,
        )
    if rng is None:
        rng = np.random.default_rng()
    uniforms = np.asarray(rng.random(n), dtype=np.float64)
    path, diagnostics = jacobi_native.sample_grid_trajectory_fixed_draws(
        kappa,
        m,
        xi,
        uniforms,
        basis_order=basis_order,
        quad_order=quad_order,
        method=sampling_method,
        storage=sampling_storage,
        correction=stationarity_correction,
        clip_negative=clip_negative,
        negative_mass_tol=negative_mass_tol,
        gh_order=gh_order,
        memory_budget_bytes=budget,
    )
    if return_diagnostics:
        diagnostics = _legacy_grid_sampling_diagnostics(
            diagnostics,
            sampling_method=sampling_method,
            requested_method=requested_method,
            storage=sampling_storage,
            correction=stationarity_correction,
            n=n,
        )
        return path, diagnostics
    return path


def jacobi_matrix_loglik(
        kappa,
        m,
        xi,
        u,
        copula,
        basis_order=32,
        quad_order=None,
        theta_cap=None,
        transition_method="auto",
        clip_negative=False,
        negative_mass_tol=1e-5,
        gh_order=5,
        memory_budget_bytes=None):
    """Evaluate log-likelihood by filtering probability masses on tau nodes."""
    resolved_quad_order = (
        default_quad_order(basis_order)
        if quad_order is None else quad_order)
    _validate_jacobi_workspace(
        quad_order=resolved_quad_order,
        basis_order=basis_order,
        n_obs=len(u),
        matrix=True,
        gh_order=gh_order,
        memory_budget_bytes=memory_budget_bytes,
    )
    try:
        evaluator = jacobi_native.PreparedScarJacobiEvaluator(
            u,
            copula,
            basis_order=basis_order,
            quad_order=resolved_quad_order,
            theta_cap=theta_cap,
            transition_method=transition_method,
            storage="dense",
            clip_negative=clip_negative,
            negative_mass_tol=negative_mass_tol,
            gh_order=gh_order,
            memory_budget_bytes=(
                DEFAULT_JACOBI_MEMORY_BUDGET_BYTES
                if memory_budget_bytes is None else memory_budget_bytes),
        )
        return evaluator.loglik(kappa, m, xi)
    except MemoryError:
        raise
    except Exception:
        return -np.inf


def jacobi_matrix_neg_loglik(*args, **kwargs):
    """Minus matrix-filter log-likelihood wrapper for optimizers."""
    value = jacobi_matrix_loglik(*args, **kwargs)
    if not np.isfinite(value):
        return 1e10
    return -value


def jacobi_matrix_neg_loglik_with_grad(
        kappa,
        m,
        xi,
        u,
        copula,
        basis_order=32,
        quad_order=None,
        theta_cap=None,
        transition_method="local_fixed",
        clip_negative=False,
        negative_mass_tol=1e-5,
        gh_order=5,
        fd_rel_step=1e-5,
        memory_budget_bytes=None):
    """Evaluate Jacobi matrix negative log-likelihood and gradient.

    Returns ``(neg_log_likelihood, neg_gradient)`` with derivatives with
    respect to physical parameters ``(kappa, m, xi)``.

    ``local_fixed`` uses explicit transition and stationary-weight
    derivatives.  Moving-grid backends such as ``spectral_matrix`` use a
    semi-analytical gradient: finite differences for setup-level arrays
    ``weights``, ``transition`` and ``fi_grid``, followed by analytical
    differentiation of the filtering recursion.
    """
    fail = 1e10, np.zeros(3, dtype=np.float64)
    method = normalize_jacobi_matrix_transition_method(transition_method)

    shapes = _jacobi_stationary_shape(kappa, m, xi)
    if shapes is None:
        return fail
    alpha, beta = shapes

    u = np.asarray(u, dtype=np.float64)
    if u.ndim != 2 or u.shape[1] != 2 or len(u) < 1:
        return fail

    basis_order = _validate_jacobi_order(basis_order, "basis_order")
    if quad_order is None:
        quad_order = default_quad_order(basis_order)
    quad_order = _validate_jacobi_order(quad_order, "quad_order")
    _validate_jacobi_workspace(
        quad_order=quad_order,
        basis_order=basis_order,
        n_obs=len(u),
        matrix=True,
        gradient=True,
        gh_order=gh_order,
        memory_budget_bytes=memory_budget_bytes,
    )

    try:
        evaluator = jacobi_native.PreparedScarJacobiEvaluator(
            u,
            copula,
            basis_order=basis_order,
            quad_order=quad_order,
            theta_cap=theta_cap,
            transition_method=method,
            storage="dense",
            clip_negative=clip_negative,
            negative_mass_tol=negative_mass_tol,
            gh_order=gh_order,
            memory_budget_bytes=(
                DEFAULT_JACOBI_MEMORY_BUDGET_BYTES
                if memory_budget_bytes is None else memory_budget_bytes),
            fd_rel_step=fd_rel_step,
        )
        return evaluator.neg_loglik_with_grad(kappa, m, xi)
    except MemoryError:
        raise
    except Exception:
        return fail


def jacobi_matrix_forward_predictive_mean(
        kappa,
        m,
        xi,
        u,
        copula,
        basis_order=32,
        quad_order=None,
        theta_cap=None,
        transition_method="auto",
        clip_negative=False,
        negative_mass_tol=1e-5,
        gh_order=5,
        memory_budget_bytes=None):
    """Return node-space E[theta(tau_k) | u_{1:k-1}]."""
    evaluator = jacobi_native.PreparedScarJacobiEvaluator(
        u, copula, basis_order=basis_order, quad_order=quad_order,
        theta_cap=theta_cap, transition_method=transition_method,
        storage="dense", clip_negative=clip_negative,
        negative_mass_tol=negative_mass_tol, gh_order=gh_order,
        memory_budget_bytes=(
            DEFAULT_JACOBI_MEMORY_BUDGET_BYTES
            if memory_budget_bytes is None else memory_budget_bytes))
    return evaluator.predictive_mean(kappa, m, xi)


def jacobi_matrix_forward_mixture_h(
        kappa,
        m,
        xi,
        u,
        copula,
        basis_order=32,
        quad_order=None,
        theta_cap=None,
        transition_method="auto",
        clip_negative=False,
        negative_mass_tol=1e-5,
        gh_order=5,
        memory_budget_bytes=None):
    """Return node-space E[h(u2 | u1; theta(tau_k)) | u_{1:k-1}]."""
    evaluator = jacobi_native.PreparedScarJacobiEvaluator(
        u, copula, basis_order=basis_order, quad_order=quad_order,
        theta_cap=theta_cap, transition_method=transition_method,
        storage="dense", clip_negative=clip_negative,
        negative_mass_tol=negative_mass_tol, gh_order=gh_order,
        memory_budget_bytes=(
            DEFAULT_JACOBI_MEMORY_BUDGET_BYTES
            if memory_budget_bytes is None else memory_budget_bytes))
    return evaluator.mixture_h(kappa, m, xi)


def jacobi_matrix_forward_mixture_h_pair(
        kappa,
        m,
        xi,
        u,
        copula,
        basis_order=32,
        quad_order=None,
        theta_cap=None,
        transition_method="auto",
        clip_negative=False,
        negative_mass_tol=1e-5,
        gh_order=5,
        memory_budget_bytes=None):
    """Return both h-directions from one Jacobi matrix filter pass."""
    evaluator = jacobi_native.PreparedScarJacobiEvaluator(
        u, copula, basis_order=basis_order, quad_order=quad_order,
        theta_cap=theta_cap, transition_method=transition_method,
        storage="dense", clip_negative=clip_negative,
        negative_mass_tol=negative_mass_tol, gh_order=gh_order,
        memory_budget_bytes=(
            DEFAULT_JACOBI_MEMORY_BUDGET_BYTES
            if memory_budget_bytes is None else memory_budget_bytes))
    return evaluator.mixture_h_pair(kappa, m, xi)


def jacobi_matrix_state_distribution(
        kappa,
        m,
        xi,
        u,
        copula,
        basis_order=32,
        quad_order=None,
        theta_cap=None,
        transition_method="auto",
        clip_negative=False,
        negative_mass_tol=1e-5,
        gh_order=5,
        memory_budget_bytes=None,
        horizon="current"):
    """Return a node-space tau distribution at current or next horizon."""
    horizon = str(horizon).lower()
    if horizon not in ("current", "next"):
        raise ValueError("horizon must be 'current' or 'next'")

    evaluator = jacobi_native.PreparedScarJacobiEvaluator(
        u, copula, basis_order=basis_order, quad_order=quad_order,
        theta_cap=theta_cap, transition_method=transition_method,
        storage="dense", clip_negative=clip_negative,
        negative_mass_tol=negative_mass_tol, gh_order=gh_order,
        memory_budget_bytes=(
            DEFAULT_JACOBI_MEMORY_BUDGET_BYTES
            if memory_budget_bytes is None else memory_budget_bytes))
    return evaluator.state_distribution(kappa, m, xi, horizon=horizon)


def jacobi_loglik(
        kappa,
        m,
        xi,
        u,
        copula,
        basis_order=32,
        quad_order=None,
        theta_cap=None,
        memory_budget_bytes=None):
    """Evaluate the Jacobi-diffusion copula log-likelihood.

    The latent state is Kendall's tau on ``(0, 1)``.  Observation emissions are
    evaluated at ``copula.tau_to_param(tau)``.
    """
    resolved_quad_order = (
        default_quad_order(basis_order)
        if quad_order is None else quad_order)
    _validate_jacobi_workspace(
        quad_order=resolved_quad_order,
        basis_order=basis_order,
        n_obs=len(u),
        matrix=False,
        memory_budget_bytes=memory_budget_bytes,
    )
    try:
        evaluator = jacobi_native.PreparedScarJacobiEvaluator(
            u,
            copula,
            basis_order=basis_order,
            quad_order=resolved_quad_order,
            theta_cap=theta_cap,
            transition_method="spectral_coeff",
            storage="dense",
            gh_order=1,
            memory_budget_bytes=(
                DEFAULT_JACOBI_MEMORY_BUDGET_BYTES
                if memory_budget_bytes is None else memory_budget_bytes),
        )
        return evaluator.loglik(kappa, m, xi)
    except MemoryError:
        raise
    except Exception:
        return -np.inf


def jacobi_neg_loglik(*args, **kwargs):
    """Minus log-likelihood wrapper for optimizers."""
    value = jacobi_loglik(*args, **kwargs)
    if not np.isfinite(value):
        return 1e10
    return -value


def jacobi_forward_predictive_mean(
        kappa,
        m,
        xi,
        u,
        copula,
        basis_order=32,
        quad_order=None,
        theta_cap=None,
        memory_budget_bytes=None):
    """Return E[theta(tau_k) | u_{1:k-1}] for each observation."""
    evaluator = jacobi_native.PreparedScarJacobiEvaluator(
        u, copula, basis_order=basis_order, quad_order=quad_order,
        theta_cap=theta_cap, transition_method="spectral_coeff",
        storage="dense", gh_order=1,
        memory_budget_bytes=(
            DEFAULT_JACOBI_MEMORY_BUDGET_BYTES
            if memory_budget_bytes is None else memory_budget_bytes))
    return evaluator.predictive_mean(kappa, m, xi)


def jacobi_forward_mixture_h(
        kappa,
        m,
        xi,
        u,
        copula,
        basis_order=32,
        quad_order=None,
        theta_cap=None,
        memory_budget_bytes=None):
    """Return E[h(u2 | u1; theta(tau_k)) | u_{1:k-1}]."""
    evaluator = jacobi_native.PreparedScarJacobiEvaluator(
        u, copula, basis_order=basis_order, quad_order=quad_order,
        theta_cap=theta_cap, transition_method="spectral_coeff",
        storage="dense", gh_order=1,
        memory_budget_bytes=(
            DEFAULT_JACOBI_MEMORY_BUDGET_BYTES
            if memory_budget_bytes is None else memory_budget_bytes))
    return evaluator.mixture_h(kappa, m, xi)


def jacobi_forward_mixture_h_pair(
        kappa,
        m,
        xi,
        u,
        copula,
        basis_order=32,
        quad_order=None,
        theta_cap=None,
        memory_budget_bytes=None):
    """Return both h-directions from one Jacobi coefficient filter pass."""
    evaluator = jacobi_native.PreparedScarJacobiEvaluator(
        u, copula, basis_order=basis_order, quad_order=quad_order,
        theta_cap=theta_cap, transition_method="spectral_coeff",
        storage="dense", gh_order=1,
        memory_budget_bytes=(
            DEFAULT_JACOBI_MEMORY_BUDGET_BYTES
            if memory_budget_bytes is None else memory_budget_bytes))
    return evaluator.mixture_h_pair(kappa, m, xi)


def jacobi_state_distribution(
        kappa,
        m,
        xi,
        u,
        copula,
        basis_order=32,
        quad_order=None,
        theta_cap=None,
        memory_budget_bytes=None,
        horizon="current"):
    """Return a discrete tau distribution at the current or next horizon."""
    horizon = str(horizon).lower()
    if horizon not in ("current", "next"):
        raise ValueError("horizon must be 'current' or 'next'")

    evaluator = jacobi_native.PreparedScarJacobiEvaluator(
        u, copula, basis_order=basis_order, quad_order=quad_order,
        theta_cap=theta_cap, transition_method="spectral_coeff",
        storage="dense", gh_order=1,
        memory_budget_bytes=(
            DEFAULT_JACOBI_MEMORY_BUDGET_BYTES
            if memory_budget_bytes is None else memory_budget_bytes))
    return evaluator.state_distribution(kappa, m, xi, horizon=horizon)
