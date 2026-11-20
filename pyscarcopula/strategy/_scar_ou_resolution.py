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
    if mode == "screen":
        next_q = q + (q + 3) // 4
        next_m = m + (m + 3) // 4
        # If the original rule cannot integrate the larger basis, compare
        # the two bases at next_q, already evaluated by the first probe.
        orders = ((m, q), (m, next_q), (next_m, q if q >= next_m else next_q))
    else:
        middle = m + (m + 1) // 2
        high_basis = max(2 * m, middle + 1)
        high_quad = 4 * q
        orders = ((m, q), (m, 2 * q), (m, high_quad),
                  (middle, high_quad), (high_basis, high_quad))
    for i, (basis, quad) in enumerate(orders):
        if quad > CPP_MAX_SPECTRAL_ORDER:
            check.update(reason="order_limit", pending_orders=[basis, quad])
            return
        try:
            if i == 0:
                value, gradient = fit.fun, fit.jac
            else:
                cfg = replace(config, transition_method="spectral",
                              basis_order=basis, quad_order=quad)
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
