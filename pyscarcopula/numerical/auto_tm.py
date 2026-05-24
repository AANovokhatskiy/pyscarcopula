"""Automatic likelihood evaluator for SCAR-OU models.

This module dispatches between two automatic numerical backends:

* Hermite spectral evaluator for ordinary transitions,
* local transition for very narrow kernels or spectral numerical fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

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
    local path; all other regimes try the Hermite spectral path and fall back
    to the local path on numerical failure.  ``large_kdt`` is retained for
    result compatibility with older fits and no longer gates the spectral path.
    """

    transition_method: str = "auto"
    small_kdt: float = 1e-3
    large_kdt: float = 5e-2
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
    u = np.asarray(u, dtype=np.float64)
    n_obs = len(u)
    kdt = _kappa_dt(kappa, n_obs)
    backend = select_auto_backend(kappa, n_obs, cfg)

    info = asdict(cfg)
    info.update({
        "backend": backend,
        "transition_method": normalize_ou_transition_method(
            cfg.transition_method),
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
        if np.isfinite(ll) or info["transition_method"] != "auto":
            return ll, info
        info["fallback_from"] = "spectral"
        backend = "local"
        info["backend"] = backend

    transition_method = "local" if backend == "local" else "matrix"
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
    ll = -float(neg_ll)

    try:
        grid = TMGrid(
            kappa, mu, nu, n_obs,
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
        info.update(grid.diagnostics())
    except Exception:
        pass

    return ll, info


def auto_loglik(kappa, mu, nu, u, copula,
                config: AutoTMConfig | None = None):
    """Evaluate only the automatic-dispatch log-likelihood."""
    ll, _ = auto_loglik_with_info(kappa, mu, nu, u, copula, config)
    return ll


def auto_neg_loglik(*args, **kwargs):
    """Minus log-likelihood wrapper for optimizers."""
    value = auto_loglik(*args, **kwargs)
    if not np.isfinite(value):
        return 1e10
    return -value


def auto_neg_loglik_with_grad(kappa, mu, nu, u, copula,
                              config: AutoTMConfig | None = None):
    """Automatic-dispatch negative log-likelihood and gradient.

    The backend selection itself is treated as fixed within one evaluation.
    This matches the existing TM analytical-gradient convention: it does not
    differentiate through adaptive grid-size or method-switch decisions.
    """
    cfg = config or AutoTMConfig()
    u = np.asarray(u, dtype=np.float64)
    n_obs = len(u)
    method = normalize_ou_transition_method(cfg.transition_method)
    backend = select_auto_backend(kappa, n_obs, cfg)

    if backend == "spectral":
        quad_order = cfg.quad_order
        if quad_order is None:
            quad_order = default_quad_order(cfg.basis_order)
        out = hermite_loglik_with_grad(
            kappa, mu, nu, u, copula,
            basis_order=cfg.basis_order,
            quad_order=quad_order,
        )
        if np.isfinite(out[0]) and float(out[0]) < 1e9:
            return out
        if method != "auto":
            return out
        backend = "local"

    transition_method = "local" if backend == "local" else "matrix"
    return tm_loglik_with_grad(
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
