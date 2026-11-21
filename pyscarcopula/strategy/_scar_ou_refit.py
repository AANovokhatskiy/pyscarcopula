"""Warm-start refinement of SCAR-OU spectral and matrix likelihoods."""
from copy import copy
from dataclasses import asdict, replace
from math import ceil

import numpy as np

from pyscarcopula._native import model_policy, scar_ou
from pyscarcopula._native.errors import NativeError
from pyscarcopula.strategy._scar_ou_resolution import PROFILES, _probe
from pyscarcopula.numerical._scar_ou_config import (
    CPP_MAX_SPECTRAL_ORDER, cpp_matrix_grid_limit,
)


def _matrix_grid(strategy, first_size):
    radius = strategy.refine_matrix_grid_range
    radius = max(8.0, float(strategy.grid_range)) if radius is None else float(radius)
    limit = cpp_matrix_grid_limit(strategy.grid_method)
    cap = strategy.refine_matrix_max_K
    if cap is None and strategy.max_K is not None:
        cap = min(limit, max(10_000, strategy.max_K, strategy.refine_matrix_K or 0))
    size = strategy.refine_matrix_K
    points = strategy.pts_per_sigma
    if size is None:
        if strategy.adaptive:
            points = min(2_147_483_647, 2 * int(points))
        intervals = (first_size - 1) * (2.0 * radius / float(strategy.grid_range))
        if strategy.transition_method == "auto":
            # Retain a request above the cap so auto can fall back to local.
            size = 1 + ceil(min(2_147_483_646, intervals))
        else:
            limit = min(limit, cap or limit)
            size = 1 + ceil(min(limit - 1, intervals))
    return int(size), radius, None if cap is None else int(cap), int(points)


def _profiles(method, m, q):
    if method == "matrix":
        return [(m, q, 0, m, q, 0)]
    profiles = [profile for profile in PROFILES if profile[0] > m and profile[1] >= q]
    if profiles:
        return profiles
    # Custom orders above the panel table can still use a larger Hermite rule.
    limit = CPP_MAX_SPECTRAL_ORDER
    if m >= limit - 1:
        return []
    # Reserve a basis mode for an independently finer control rule.
    order = min(limit - 1, m + max(1, m // 4))
    quad = min(limit, max(q, 2 * order))
    control_order = min(limit, order + max(1, order // 4))
    return [(order, quad, 0, control_order, min(limit, 2 * quad), 0)]


def _projected_score(strategy, copula, start, end, gradient, n):
    from pyscarcopula.strategy.scar_tm import (
        _log_stationary_optimizer_bounds, _ou_scaled_optimizer_bounds,
        _uses_log_stationary_scale,
    )
    if _uses_log_stationary_scale(copula, strategy.log_stationary_scale_optimization):
        x = scar_ou.to_log_stationary(end)
        grad = scar_ou.gradient_to_log_stationary(end, gradient)
        lo, hi, _ = _log_stationary_optimizer_bounds(
            copula, strategy.log_stationary_scale_optimization, strategy.stationary_scale_bounds, n)
    else:
        scale = model_policy.optimizer_unit_scale(start)
        x = scar_ou.physical_to_scaled(end, scale)
        grad = scar_ou.gradient_to_scaled(gradient, scale)
        lo, hi = _ou_scaled_optimizer_bounds(scale[:3], n)
    projected = grad / n
    projected[:3] = x[:3] - np.clip(x[:3]-projected[:3], lo, hi)
    return float(np.max(np.abs(projected)))


def fit_with_refinement(ordinary, copula, u, initial, run, *, optimizer_gtol):
    base = run(ordinary, initial)
    report = dict(status="skipped", tolerance=ordinary.refine_loglik_tol, attempts=[])
    base.diagnostics["refine_fit"] = report
    if not base.success:
        report["reason"] = "first_fit_failed"
        return base

    params = base.params.values
    joint = base.parameter_count > 3
    start = np.r_[params, copula._pack_corr_params()] if joint else params
    n = len(u)
    restart_gtol = min(optimizer_gtol, n * ordinary.refine_score_tol)
    m0 = ordinary._spectral_basis_order_for(params[0], n)
    report.update(status="rejected", first_backend=base.diagnostics["last_backend"],
                  first_log_likelihood=base.log_likelihood,
                  first_basis_order=m0, score_tolerance=ordinary.refine_score_tol, reason="profile_limit")
    evaluations = base.nfev
    method = ordinary.transition_method
    first_size = base.diagnostics.get("last_K_effective")
    matrix_grid = None
    if method in ("auto", "matrix"):
        matrix_grid = _matrix_grid(ordinary, first_size or min(ordinary.K, ordinary.max_K or ordinary.K))
        report.update(matrix_K=matrix_grid[0], matrix_grid_range=matrix_grid[1],
                      matrix_max_K=matrix_grid[2], matrix_pts_per_sigma=matrix_grid[3],
                      first_K_effective=first_size,
                      first_grid_step=(2 * float(ordinary.grid_range) / (first_size - 1) if first_size else None))
    q0 = model_policy.ou_resolve_quad_order(m0, ordinary.spectral_quad_order)
    for order, q, degree, control_order, control_q, control_degree in _profiles(method, m0, q0):
        stage = copy(ordinary)
        if method == "matrix":
            stage.K, stage.grid_range, stage.max_K, stage.pts_per_sigma = matrix_grid
        else:
            stage._matrix_grid = matrix_grid
            stage.spectral_basis_order = order
            stage.spectral_quad_order = q
            stage._panel_degree = degree
        attempt = dict(basis_order=order, quad_order=q)
        report["attempts"].append(attempt)
        accepted = False
        try:
            cfg = stage._auto_config(kappa=params[0], n_obs=n)
            control = replace(cfg, basis_order=control_order, panel_degree=control_degree,
                              quad_order=control_q)

            start_nll, _ = _probe(u, copula, start, control)
            candidate = run(stage, start, restart_gtol=restart_gtol)
            evaluations += candidate.nfev
            if not candidate.success:
                attempt["reason"] = "second_fit_failed"
                continue
            final = (np.r_[candidate.params.values, copula._pack_corr_params()]
                     if joint else candidate.params.values)
            nll, gradient = _probe(u, copula, final, control)
            end = -nll
            stationarity = _projected_score(stage, copula, start, final, gradient, n)
            error = abs(end-candidate.log_likelihood)/n
            gain = (end+start_nll)/n
            attempt.update(control_basis_order=control_order, control_quad_order=control.quad_order,
                           error_per_obs=error, control_gain_per_obs=gain, projected_score_per_obs=stationarity)
            if not stationarity <= ordinary.refine_score_tol:
                attempt["reason"] = "control_score_exceeds_tolerance"
                continue
            if not (np.isfinite(gain) and gain >= 0 and error <= ordinary.refine_loglik_tol):
                attempt["reason"] = "control_disagreement_or_loss"
                continue
            report.update(status="accepted", reason="control_passed", basis_order=order,
                          quad_order=stage.spectral_quad_order, panel_degree=degree)
            report["likelihood_config"] = asdict(cfg)
            backend = candidate.diagnostics.get("last_backend")
            size = candidate.diagnostics.get("last_K_effective")
            radius = matrix_grid[1] if backend == "matrix" else float(ordinary.grid_range)
            report.update(final_backend=backend, final_K_effective=size,
                          final_grid_step=(2 * radius / (size - 1) if size else None))
            candidate.diagnostics["refine_fit"] = report
            accepted = True
            return replace(candidate, nfev=evaluations)
        except (NativeError, FloatingPointError) as exc:
            attempt["reason"] = str(exc)
        finally:
            if joint and not accepted:
                copula._set_corr_from_params(start[3:])
    if report["attempts"]:
        report["reason"] = "profiles_exhausted"
    return replace(base, nfev=evaluations)
