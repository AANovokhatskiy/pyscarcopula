"""Final-point comparisons of the native Hermite likelihood and score."""

from dataclasses import replace

import numpy as np

from pyscarcopula._native import model_policy, scar_ou
from pyscarcopula._native.errors import NativeError
from pyscarcopula.copula.multivariate.corr_param import (
    _corr_gradient_to_raw_params,
    _shrinkage_raw_corr_direction,
)
from pyscarcopula.numerical._scar_ou_config import CPP_MAX_SPECTRAL_ORDER


# Candidate M, Q, panel degree and their independently finer control profile.
PROFILES = (
    (32, 320, 32, 64, 672, 48),
    (64, 448, 32, 96, 896, 56),
    (96, 640, 40, 128, 1008, 56),
    (128, 720, 40, 160, 1020, 51),
    (160, 880, 44, 192, 1020, 51),
)


def _panel_check_orders(config, mode):
    for m, q, degree, high_m, high_q, high_degree in PROFILES:
        if (m, q, degree) == (config.basis_order, config.quad_order, config.panel_degree):
            nodes_per_degree = q // degree
            high_nodes_per_degree = high_q // high_degree
            middle = (degree + high_degree) // 2
            base = (m, q, degree)
            fine = (m, nodes_per_degree * high_degree, high_degree)
            control = (high_m, high_q, high_degree)
            if mode == "screen":
                return (base, fine, control)
            return (base, (m, nodes_per_degree * middle, middle), fine,
                    (high_m, high_nodes_per_degree * middle, middle), control)
    raise ValueError("unknown spectral refit profile")


def resolution_tolerances(values):
    if values is None:
        return None
    if not isinstance(values, (tuple, list)) or len(values) != 2:
        raise ValueError("spectral_check_tolerances requires (loglik_atol, score_atol)")
    for tol in values:
        if (isinstance(tol, (bool, np.bool_))
                or not isinstance(tol, (float, int, np.floating, np.integer))
                or not np.isfinite(float(tol)) or tol <= 0):
            raise ValueError("spectral_check_tolerances must be positive and finite")
    return tuple(map(float, values))


def _probe(u, copula, params, config):
    objective = scar_ou.prepare_objective(u, copula, config)
    ou = params[:3]
    if len(params) == 3:
        value, gradient, _ = objective.neg_loglik_with_grad_info(*ou)
        return value, gradient
    raw = params[3:]
    if copula.corr_mode == "shrinkage":
        direction = _shrinkage_raw_corr_direction(raw, copula._corr_base)
        value, gradient, corr, _ = (
            objective.neg_loglik_with_grad_and_corr_directional_info(*ou, direction))
    else:
        value, gradient, corr, _ = objective.neg_loglik_with_grad_and_corr_info(*ou)
        corr = _corr_gradient_to_raw_params(
            copula.corr_mode, raw, copula.R, corr, copula._corr_base)
    return value, np.concatenate((gradient, corr))


def check_final_resolution(
        u, copula, params, config_for, fit, diagnostics, tolerances, mode):
    """Reuse the validated base value/gradient. Stop on the first failed refinement."""
    loglik_tol, score_tol = tolerances
    probes = []
    check = dict(mode=mode, loglik_atol=loglik_tol, score_atol=score_tol,
                 certificate=False, probes=probes)
    diagnostics["spectral_resolution_check"] = check
    if not fit.success:
        check["reason"] = "fit_unsuccessful"
        return
    if diagnostics.get("last_backend") != "spectral":
        check["reason"] = "final_backend_not_spectral"
        return

    diagnostics["final_integration_status"] = "unresolved"
    config = config_for(kappa=params[0], n_obs=len(u))
    m = config.basis_order
    q = model_policy.ou_resolve_quad_order(m, config.quad_order)
    if config.panel_degree:
        orders = _panel_check_orders(config, mode)
    elif mode == "screen":
        next_q = q + (q + 3) // 4
        next_m = m + (m + 3) // 4
        # If the original rule cannot integrate the larger basis, compare
        # the two bases at next_q, already evaluated by the first probe.
        orders = ((m, q, 0), (m, next_q, 0), (next_m, q if q >= next_m else next_q, 0))
    else:
        middle = m + (m + 1) // 2
        high_basis = max(2 * m, middle + 1)
        high_quad = 4 * q
        orders = ((m, q, 0), (m, 2 * q, 0), (m, high_quad, 0),
                  (middle, high_quad, 0), (high_basis, high_quad, 0))
    for i, (basis, quad, degree) in enumerate(orders):
        if quad > CPP_MAX_SPECTRAL_ORDER:
            check.update(reason="order_limit", pending_orders=[basis, quad])
            return
        try:
            if i == 0:
                value, gradient = fit.fun, fit.jac
            else:
                cfg = replace(config, transition_method="spectral",
                              basis_order=basis, quad_order=quad, panel_degree=degree)
                value, gradient = _probe(u, copula, params, cfg)
            score = scar_ou.gradient_to_log_stationary(params, gradient)
            vector = np.r_[-value, -score]
            if not np.all(np.isfinite(vector)):
                raise FloatingPointError("nonfinite log-likelihood or score")
        except (FloatingPointError, NativeError, ValueError, MemoryError) as exc:
            check.update(reason="probe_failure", failed_orders=[basis, quad],
                         error=str(exc))
            return
        probes.append(dict(basis_order=basis, quad_order=quad,
                           log_likelihood=float(vector[0]), score=vector[1:].tolist()))
        if i == 0:
            low = high = vector
            continue
        low = np.minimum(low, vector)
        high = np.maximum(high, vector)
        spread = high - low
        check.update(loglik_spread=float(spread[0]), score_spread=spread[1:].tolist())
        if spread[0] > loglik_tol or np.any(spread[1:] > score_tol):
            diagnostics["final_integration_converged"] = False
            check["reason"] = "tolerance_exceeded"
            return
    if mode == "screen":
        diagnostics["final_integration_status"] = "screen_passed"
    else:
        diagnostics["final_integration_status"] = "numerical_agreement"
        diagnostics["final_integration_converged"] = True
