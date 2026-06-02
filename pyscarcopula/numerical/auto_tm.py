"""Automatic likelihood evaluator for SCAR-OU models.

This module dispatches between automatic numerical backends:

* Hermite spectral evaluator for ordinary transitions,
* local transition for very narrow kernels;
* matrix transition as the first spectral numerical fallback;
* local transition if the matrix fallback fails or its adaptive grid is capped.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from pyscarcopula.numerical._arrays import as_float64_array
from pyscarcopula.numerical.hermite_tm import (
    default_quad_order,
    hermite_loglik,
    hermite_loglik_with_grad,
)
from pyscarcopula.numerical.tm_gradient import tm_loglik_with_grad
from pyscarcopula.numerical.tm_functions import tm_loglik
from pyscarcopula.numerical.tm_grid import TMGrid
from pyscarcopula.numerical._transition_methods import (
    normalize_ou_transition_method,
)


@dataclass(frozen=True)
class AutoTMConfig:
    """Configuration for the automatic evaluator.

    ``transition_method`` is one of ``"auto"``, ``"matrix"``, ``"local"``, or
    ``"spectral"``.  In ``"auto"`` mode, ``small_kdt`` selects the narrow-kernel
    local path; all other regimes try the Hermite spectral path.  Spectral
    numerical failure falls back to matrix; matrix numerical failure or an
    adaptive-grid cap falls back to local.
    """

    transition_method: str = "auto"
    small_kdt: float = 1e-2
    basis_order: int = 32
    quad_order: int | None = None
    K: int = 300
    grid_range: float = 5.0
    grid_method: str = "auto"
    adaptive: bool = True
    pts_per_sigma: int = 4
    max_K: int | None = 1000
    gh_order: int = 5
    r_gh: float = 3.0


def _kappa_dt(kappa: float, n_obs: int) -> float:
    if n_obs <= 1:
        return float(kappa)
    return float(kappa) / float(n_obs - 1)


def _spectral_loglik_failed(value: float) -> bool:
    return (not np.isfinite(value)) or float(value) <= -1e9


def _spectral_neg_loglik_failed(value: float) -> bool:
    return (not np.isfinite(value)) or float(value) >= 1e9


def _grid_neg_loglik_failed(value: float) -> bool:
    return (not np.isfinite(value)) or float(value) >= 1e9


def _matrix_diagnostics(kappa, mu, nu, n_obs, cfg: AutoTMConfig) -> dict:
    grid = TMGrid(
        kappa, mu, nu, n_obs,
        K=cfg.K,
        grid_range=cfg.grid_range,
        grid_method=cfg.grid_method,
        adaptive=cfg.adaptive,
        pts_per_sigma=cfg.pts_per_sigma,
        transition_method="matrix",
        max_K=cfg.max_K,
        r_gh=cfg.r_gh,
        gh_order=cfg.gh_order,
    )
    return grid.diagnostics()


def _grid_loglik(kappa, mu, nu, u, copula, cfg: AutoTMConfig,
                 transition_method: str) -> float:
    neg_ll = tm_loglik(
        kappa, mu, nu, u, copula,
        K=cfg.K,
        grid_range=cfg.grid_range,
        grid_method=cfg.grid_method,
        adaptive=cfg.adaptive,
        pts_per_sigma=cfg.pts_per_sigma,
        transition_method=transition_method,
        max_K=cfg.max_K,
        r_gh=cfg.r_gh,
        gh_order=cfg.gh_order,
    )
    return -float(neg_ll)


def _grid_loglik_info(kappa, mu, nu, u, copula, cfg: AutoTMConfig,
                      transition_method: str) -> tuple[float, dict]:
    ll = _grid_loglik(kappa, mu, nu, u, copula, cfg, transition_method)
    grid = TMGrid(
        kappa, mu, nu, len(u),
        K=cfg.K,
        grid_range=cfg.grid_range,
        grid_method=cfg.grid_method,
        adaptive=cfg.adaptive,
        pts_per_sigma=cfg.pts_per_sigma,
        transition_method=transition_method,
        max_K=cfg.max_K,
        r_gh=cfg.r_gh,
        gh_order=cfg.gh_order,
    )
    return ll, grid.diagnostics()


def select_auto_backend(kappa: float, n_obs: int,
                        config: AutoTMConfig | None = None) -> str:
    """Return ``"spectral"`` or ``"local"`` for this regime."""
    cfg = config or AutoTMConfig()
    method = normalize_ou_transition_method(cfg.transition_method)
    if method != "auto":
        return method

    kdt = _kappa_dt(kappa, n_obs)
    if kdt < cfg.small_kdt:
        return "local"
    return "spectral"


def auto_loglik_with_info(kappa, mu, nu, u, copula,
                          config: AutoTMConfig | None = None):
    """Evaluate log-likelihood and return backend diagnostics.

    Returns
    -------
    tuple
        ``(log_likelihood, info_dict)``.  ``log_likelihood`` is ``-inf`` on
        numerical failure, matching the Hermite prototype convention.
    """
    cfg = config or AutoTMConfig()
    u = as_float64_array(u)
    n_obs = len(u)
    kdt = _kappa_dt(kappa, n_obs)
    backend = select_auto_backend(kappa, n_obs, cfg)
    requested_method = normalize_ou_transition_method(cfg.transition_method)

    info = asdict(cfg)
    info.update({
        "backend": backend,
        "transition_method": requested_method,
        "kappa_dt": float(kdt),
        "n_obs": int(n_obs),
    })

    if backend == "spectral":
        quad_order = cfg.quad_order
        if quad_order is None:
            quad_order = default_quad_order(cfg.basis_order)
        ll = hermite_loglik(
            kappa, mu, nu, u, copula,
            basis_order=cfg.basis_order,
            quad_order=quad_order,
        )
        info["quad_order_used"] = int(quad_order)
        if (not _spectral_loglik_failed(ll)) or info["transition_method"] != "auto":
            return ll, info
        info["fallback_from"] = "spectral"
        info["fallback_chain"] = ["spectral"]
        backend = "matrix"
        info["backend"] = backend

    transition_method = "local" if backend == "local" else "matrix"
    try:
        ll, diag = _grid_loglik_info(
            kappa, mu, nu, u, copula, cfg, transition_method)
        info.update(diag)
    except Exception:
        ll = -1e10

    if (requested_method == "auto"
            and backend == "matrix"
            and (
                _grid_neg_loglik_failed(-ll)
                or bool(info.get("adaptive_was_capped", False))
            )):
        reason = "capped" if info.get("adaptive_was_capped", False) else "failed"
        info["matrix_fallback_reason"] = reason
        info.setdefault("fallback_chain", []).append("matrix")
        info["fallback_from"] = "matrix"
        backend = "local"
        info["backend"] = backend
        try:
            ll, diag = _grid_loglik_info(
                kappa, mu, nu, u, copula, cfg, "local")
            info.update(diag)
        except Exception:
            ll = -1e10

    return ll, info


def auto_loglik(kappa, mu, nu, u, copula,
                config: AutoTMConfig | None = None):
    """Evaluate only the automatic-dispatch log-likelihood."""
    ll, _ = auto_loglik_with_info(kappa, mu, nu, u, copula, config)
    return ll


def auto_neg_loglik(*args, **kwargs):
    """Minus log-likelihood wrapper for optimizers."""
    value, _ = auto_neg_loglik_info(*args, **kwargs)
    return value


def auto_neg_loglik_info(kappa, mu, nu, u, copula,
                         config: AutoTMConfig | None = None):
    """Minus log-likelihood wrapper plus backend diagnostics."""
    value, info = auto_loglik_with_info(kappa, mu, nu, u, copula, config)
    if not np.isfinite(value):
        return 1e10, info
    return -value, info


def auto_neg_loglik_with_grad(kappa, mu, nu, u, copula,
                              config: AutoTMConfig | None = None):
    """Automatic-dispatch negative log-likelihood and gradient.

    The backend selection itself is treated as fixed within one evaluation.
    This matches the existing TM analytical-gradient convention: it does not
    differentiate through adaptive grid-size or method-switch decisions.
    """
    value, grad, _ = auto_neg_loglik_with_grad_info(
        kappa, mu, nu, u, copula, config)
    return value, grad


def auto_neg_loglik_with_grad_info(kappa, mu, nu, u, copula,
                                   config: AutoTMConfig | None = None):
    """Automatic-dispatch objective, gradient, and backend diagnostics."""
    cfg = config or AutoTMConfig()
    u = as_float64_array(u)
    n_obs = len(u)
    kdt = _kappa_dt(kappa, n_obs)
    method = normalize_ou_transition_method(cfg.transition_method)
    backend = select_auto_backend(kappa, n_obs, cfg)
    info = asdict(cfg)
    info.update({
        "backend": backend,
        "transition_method": method,
        "kappa_dt": float(kdt),
        "n_obs": int(n_obs),
    })

    if backend == "spectral":
        quad_order = cfg.quad_order
        if quad_order is None:
            quad_order = default_quad_order(cfg.basis_order)
        out = hermite_loglik_with_grad(
            kappa, mu, nu, u, copula,
            basis_order=cfg.basis_order,
            quad_order=quad_order,
        )
        info["quad_order_used"] = int(quad_order)
        if not _spectral_neg_loglik_failed(out[0]):
            return out[0], out[1], info
        if method != "auto":
            return out[0], out[1], info
        info["fallback_from"] = "spectral"
        info["fallback_chain"] = ["spectral"]
        backend = "matrix"
        info["backend"] = backend

    transition_method = "local" if backend == "local" else "matrix"
    out = tm_loglik_with_grad(
        kappa, mu, nu, u, copula,
        K=cfg.K,
        grid_range=cfg.grid_range,
        grid_method=cfg.grid_method,
        adaptive=cfg.adaptive,
        pts_per_sigma=cfg.pts_per_sigma,
        transition_method=transition_method,
        max_K=cfg.max_K,
        r_gh=cfg.r_gh,
        gh_order=cfg.gh_order,
    )
    if method != "auto" or backend != "matrix":
        return out[0], out[1], info

    matrix_capped = False
    try:
        diag = _matrix_diagnostics(kappa, mu, nu, n_obs, cfg)
        info.update(diag)
        matrix_capped = bool(diag.get("adaptive_was_capped", False))
    except Exception:
        matrix_capped = True
    if not matrix_capped and not _grid_neg_loglik_failed(out[0]):
        return out[0], out[1], info

    reason = "capped" if matrix_capped else "failed"
    info["matrix_fallback_reason"] = reason
    info.setdefault("fallback_chain", []).append("matrix")
    info["fallback_from"] = "matrix"
    info["backend"] = "local"

    out = tm_loglik_with_grad(
        kappa, mu, nu, u, copula,
        K=cfg.K,
        grid_range=cfg.grid_range,
        grid_method=cfg.grid_method,
        adaptive=cfg.adaptive,
        pts_per_sigma=cfg.pts_per_sigma,
        transition_method="local",
        max_K=cfg.max_K,
        r_gh=cfg.r_gh,
        gh_order=cfg.gh_order,
    )
    return out[0], out[1], info
