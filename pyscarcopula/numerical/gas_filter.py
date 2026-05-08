"""
pyscarcopula.numerical.gas_filter — core GAS filter computation.

Single source of truth for the GAS (Generalized Autoregressive Score) filter.

Implements the observation-driven model of Creal, Koopman and Lucas (2013):

    g_t = omega + beta * g_{t-1} + gamma * s_{t-1}

where s_t = S_t * nabla_t is the scaled score.

Scaling options:
    'unit'   : S_t = 1  (identity — simple and robust)
    'fisher' : S_t = I_t^{-1}  (inverse Fisher information — optimal but less stable)

Used by:
    - strategy/gas.py (GASStrategy) for fitting and evaluation
    - latent/initial_point.py for GAS-based initial point estimation
    - stattests.py for GAS Rosenblatt transform
    - vine/_edge.py for GAS h-functions in vine edges
"""

import numpy as np
from numba import njit

from pyscarcopula.copula.clayton import (
    ClaytonCopula,
    _clayton_dlogc_dr,
    _clayton_dtransform,
    _clayton_log_pdf,
    _clayton_softplus_dtransform,
    _clayton_softplus_transform,
    _clayton_transform,
)
from pyscarcopula.copula.elliptical import (
    BivariateGaussianCopula,
    _gauss_dlog_pdf_drho,
    _gauss_log_pdf_numba,
)
from pyscarcopula.copula.frank import (
    FrankCopula,
    _frank_dlogc_dr,
    _frank_dtransform,
    _frank_log_pdf,
    _frank_softplus_dtransform,
    _frank_softplus_transform,
    _frank_transform,
)
from pyscarcopula.copula.gumbel import (
    GumbelCopula,
    _gumbel_dlogc_dr,
    _gumbel_dtransform,
    _gumbel_log_pdf,
    _gumbel_softplus_dtransform,
    _gumbel_softplus_transform,
    _gumbel_transform,
)
from pyscarcopula.copula.joe import (
    JoeCopula,
    _joe_dlogc_dr,
    _joe_dtransform,
    _joe_log_pdf,
    _joe_softplus_dtransform,
    _joe_softplus_transform,
    _joe_transform,
)

G_CLIP = 50.0
S_CLIP = 100.0

_GAS_FAMILY_GAUSSIAN = 1
_GAS_FAMILY_CLAYTON = 2
_GAS_FAMILY_FRANK = 3
_GAS_FAMILY_GUMBEL = 4
_GAS_FAMILY_JOE = 5


@njit(cache=True)
def _gas_apply_rotation(u1, u2, rotation):
    if rotation == 90:
        return 1.0 - u1, u2
    if rotation == 180:
        return 1.0 - u1, 1.0 - u2
    if rotation == 270:
        return u1, 1.0 - u2
    return u1, u2


@njit(cache=True)
def _gas_transform(g_t, family, use_softplus, x_arr):
    x_arr[0] = g_t
    if family == _GAS_FAMILY_GAUSSIAN:
        return 0.9999 * np.tanh(g_t / 4.0)
    if family == _GAS_FAMILY_CLAYTON:
        if use_softplus:
            return _clayton_softplus_transform(x_arr)[0]
        return _clayton_transform(x_arr)[0]
    if family == _GAS_FAMILY_FRANK:
        if use_softplus:
            return _frank_softplus_transform(x_arr)[0]
        return _frank_transform(x_arr)[0]
    if family == _GAS_FAMILY_GUMBEL:
        if use_softplus:
            return _gumbel_softplus_transform(x_arr)[0]
        return _gumbel_transform(x_arr)[0]
    if family == _GAS_FAMILY_JOE:
        if use_softplus:
            return _joe_softplus_transform(x_arr)[0]
        return _joe_transform(x_arr)[0]
    raise ValueError("Unsupported GAS copula family")


@njit(cache=True)
def _gas_dtransform(g_t, family, use_softplus, x_arr):
    x_arr[0] = g_t
    if family == _GAS_FAMILY_GAUSSIAN:
        th = np.tanh(g_t / 4.0)
        return 0.9999 / 4.0 * (1.0 - th * th)
    if family == _GAS_FAMILY_CLAYTON:
        if use_softplus:
            return _clayton_softplus_dtransform(x_arr)[0]
        return _clayton_dtransform(x_arr)[0]
    if family == _GAS_FAMILY_FRANK:
        if use_softplus:
            return _frank_softplus_dtransform(x_arr)[0]
        return _frank_dtransform(x_arr)[0]
    if family == _GAS_FAMILY_GUMBEL:
        if use_softplus:
            return _gumbel_softplus_dtransform(x_arr)[0]
        return _gumbel_dtransform(x_arr)[0]
    if family == _GAS_FAMILY_JOE:
        if use_softplus:
            return _joe_softplus_dtransform(x_arr)[0]
        return _joe_dtransform(x_arr)[0]
    raise ValueError("Unsupported GAS copula family")


@njit(cache=True)
def _gas_unit_log_pdf(family, u1_arr, u2_arr, r_arr):
    if family == _GAS_FAMILY_GAUSSIAN:
        return _gauss_log_pdf_numba(u1_arr, u2_arr, r_arr)[0]
    if family == _GAS_FAMILY_CLAYTON:
        return _clayton_log_pdf(u1_arr, u2_arr, r_arr)[0]
    if family == _GAS_FAMILY_FRANK:
        return _frank_log_pdf(u1_arr, u2_arr, r_arr)[0]
    if family == _GAS_FAMILY_GUMBEL:
        return _gumbel_log_pdf(u1_arr, u2_arr, r_arr)[0]
    if family == _GAS_FAMILY_JOE:
        return _joe_log_pdf(u1_arr, u2_arr, r_arr)[0]
    raise ValueError("Unsupported GAS copula family")


@njit(cache=True)
def _gas_unit_dlog_dr(family, u1_arr, u2_arr, r_arr):
    if family == _GAS_FAMILY_GAUSSIAN:
        return _gauss_dlog_pdf_drho(u1_arr, u2_arr, r_arr)[0]
    if family == _GAS_FAMILY_CLAYTON:
        return _clayton_dlogc_dr(u1_arr, u2_arr, r_arr)[0]
    if family == _GAS_FAMILY_FRANK:
        return _frank_dlogc_dr(u1_arr, u2_arr, r_arr)[0]
    if family == _GAS_FAMILY_GUMBEL:
        return _gumbel_dlogc_dr(u1_arr, u2_arr, r_arr)[0]
    if family == _GAS_FAMILY_JOE:
        return _joe_dlogc_dr(u1_arr, u2_arr, r_arr)[0]
    raise ValueError("Unsupported GAS copula family")


@njit(cache=True)
def _gas_filter_unit_numba(omega, gamma, beta, u, family, rotation,
                           use_softplus):
    T = len(u)
    g_path = np.empty(T)
    r_path = np.empty(T)
    total_logL = 0.0

    if abs(beta) < 1.0 - 1e-8:
        g_t = omega / (1.0 - beta)
    else:
        g_t = omega

    x_arr = np.empty(1, dtype=np.float64)
    u1_arr = np.empty(1, dtype=np.float64)
    u2_arr = np.empty(1, dtype=np.float64)
    r_arr = np.empty(1, dtype=np.float64)

    for t in range(T):
        g_path[t] = g_t
        r_t = _gas_transform(g_t, family, use_softplus, x_arr)
        r_path[t] = r_t

        u1, u2 = _gas_apply_rotation(u[t, 0], u[t, 1], rotation)
        u1_arr[0] = u1
        u2_arr[0] = u2
        r_arr[0] = r_t
        ll_t = _gas_unit_log_pdf(family, u1_arr, u2_arr, r_arr)
        if not np.isfinite(ll_t):
            return g_path, r_path, -1e10

        total_logL += ll_t

        if t < T - 1:
            dlog_dr = _gas_unit_dlog_dr(family, u1_arr, u2_arr, r_arr)
            dpsi_dg = _gas_dtransform(g_t, family, use_softplus, x_arr)
            s_t = dlog_dr * dpsi_dg
            if not np.isfinite(s_t):
                return g_path, r_path, -1e10

            if s_t > S_CLIP:
                s_t = S_CLIP
            elif s_t < -S_CLIP:
                s_t = -S_CLIP

            g_t = omega + beta * g_t + gamma * s_t
            if g_t > G_CLIP:
                g_t = G_CLIP
            elif g_t < -G_CLIP:
                g_t = -G_CLIP

    return g_path, r_path, total_logL


def _gas_unit_kernel_args(copula, scaling):
    if scaling != 'unit':
        return None
    if isinstance(copula, BivariateGaussianCopula):
        return _GAS_FAMILY_GAUSSIAN, 0, True
    if isinstance(copula, ClaytonCopula):
        return (
            _GAS_FAMILY_CLAYTON,
            int(copula.rotate),
            getattr(copula, '_transform_type', 'softplus') == 'softplus',
        )
    if isinstance(copula, FrankCopula):
        return (
            _GAS_FAMILY_FRANK,
            0,
            getattr(copula, '_transform_type', 'softplus') == 'softplus',
        )
    if isinstance(copula, GumbelCopula):
        return (
            _GAS_FAMILY_GUMBEL,
            int(copula.rotate),
            getattr(copula, '_transform_type', 'softplus') == 'softplus',
        )
    if isinstance(copula, JoeCopula):
        return (
            _GAS_FAMILY_JOE,
            int(copula.rotate),
            getattr(copula, '_transform_type', 'softplus') == 'softplus',
        )
    return None


def _gas_score(u1, u2, g_t, r_t, ll_t, copula, scaling, score_eps):
    """Score d log c(u_t; Psi(g_t)) / d g_t."""
    has_analytical = (
        hasattr(copula, 'dlog_pdf_dr')
        and hasattr(copula, 'dtransform')
    )
    if scaling != 'fisher' and has_analytical:
        dlog_dr = float(copula.dlog_pdf_dr(u1, u2, np.array([r_t]))[0])
        dpsi_dg = float(copula.dtransform(np.array([g_t]))[0])
        return dlog_dr * dpsi_dg

    g_plus = g_t + score_eps
    g_minus = g_t - score_eps
    r_plus = float(copula.transform(np.array([g_plus]))[0])
    r_minus = float(copula.transform(np.array([g_minus]))[0])

    ll_plus = float(copula.log_pdf(u1, u2, np.array([r_plus]))[0])
    ll_minus = float(copula.log_pdf(u1, u2, np.array([r_minus]))[0])

    nabla_t = (ll_plus - ll_minus) / (2.0 * score_eps)
    if scaling != 'fisher':
        return nabla_t

    d2 = (ll_plus - 2.0 * ll_t + ll_minus) / (score_eps ** 2)
    fisher = max(-d2, 1e-6)
    return nabla_t / fisher


# ══════════════════════════════════════════════════════════════════
# Core GAS filter
# ══════════════════════════════════════════════════════════════════

def gas_filter(omega, gamma, beta, u, copula, scaling='unit',
               score_eps=1e-4):
    """Run GAS filter, return full path and log-likelihood.

    At each step t:
      1. r_t = Psi(g_t) — copula parameter
      2. log c(u_t; r_t) — contributes to log L
      3. nabla_t = d log c / d g_t — numerical score (central diff in g-space)
      4. (Optional) Fisher scaling: I_t^{-1}
      5. g_{t+1} = omega + beta*g_t + gamma*s_t

    Parameters
    ----------
    omega, gamma, beta : float
        GAS recursion parameters.
    u : (T, 2) pseudo-observations
    copula : CopulaProtocol
    scaling : 'unit' or 'fisher'
        Score scaling. 'unit' uses S_t=1, 'fisher' uses inverse Fisher info.
    score_eps : float
        Step size for central difference score approximation.

    Returns
    -------
    g_path : (T,) — untransformed parameter path
    r_path : (T,) — copula parameter path Psi(g_t)
    total_logL : float
    """
    kernel_args = _gas_unit_kernel_args(copula, scaling)
    if kernel_args is not None:
        family, rotation, use_softplus = kernel_args
        return _gas_filter_unit_numba(
            float(omega), float(gamma), float(beta),
            np.asarray(u, dtype=np.float64),
            family, rotation, use_softplus)

    T = len(u)
    g_path = np.empty(T)
    r_path = np.empty(T)
    total_logL = 0.0

    # Initial value: unconditional mean g_bar = omega / (1 - beta)
    if abs(beta) < 1.0 - 1e-8:
        g_bar = omega / (1.0 - beta)
    else:
        g_bar = omega

    g_t = g_bar

    for t in range(T):
        g_path[t] = g_t
        r_t = float(copula.transform(np.array([g_t]))[0])
        r_path[t] = r_t

        # Log-likelihood contribution
        u1 = u[t:t+1, 0]
        u2 = u[t:t+1, 1]
        ll_t = float(copula.log_pdf(u1, u2, np.array([r_t]))[0])

        if not np.isfinite(ll_t):
            return g_path, r_path, -1e10  # signal failure

        total_logL += ll_t

        # Score for next step
        if t < T - 1:
            s_t = _gas_score(
                u1, u2, g_t, r_t, ll_t, copula, scaling, score_eps)
            if not np.isfinite(s_t):
                return g_path, r_path, -1e10

            s_t = np.clip(s_t, -S_CLIP, S_CLIP)
            g_t = omega + beta * g_t + gamma * s_t
            g_t = np.clip(g_t, -G_CLIP, G_CLIP)

    return g_path, r_path, total_logL


def gas_predict_param(omega, gamma, beta, u, copula, scaling='unit',
                      score_eps=1e-4, horizon='next'):
    """Return Psi(g) for GAS predictive sampling.

    ``current`` returns the last in-sample parameter Psi(g_{T-1}).
    ``next`` applies the final observation score and returns Psi(g_T).
    """
    g_path, r_path, _ = gas_filter(
        omega, gamma, beta, u, copula, scaling, score_eps)
    if len(g_path) == 0:
        raise ValueError("GAS predict requires at least one observation")

    if horizon in (0, '0'):
        horizon = 'current'
    elif horizon in (1, '1'):
        horizon = 'next'
    else:
        horizon = str(horizon).lower()

    if horizon == 'current':
        return float(r_path[-1])
    if horizon != 'next':
        raise ValueError("horizon must be 'current' or 'next'")

    g_t = float(g_path[-1])
    u1 = u[-1:, 0]
    u2 = u[-1:, 1]

    r_t = float(copula.transform(np.array([g_t]))[0])
    ll_t = float(copula.log_pdf(u1, u2, np.array([r_t]))[0])
    s_t = _gas_score(u1, u2, g_t, r_t, ll_t, copula, scaling, score_eps)

    s_t = np.clip(s_t, -S_CLIP, S_CLIP)
    g_next = omega + beta * g_t + gamma * s_t
    g_next = np.clip(g_next, -G_CLIP, G_CLIP)
    return float(copula.transform(np.array([g_next]))[0])


def gas_negloglik(omega, gamma, beta, u, copula, scaling='unit',
                  score_eps=1e-4):
    """Minus log-likelihood for optimizer.

    Returns
    -------
    float : -logL (for minimization). Returns 1e10 on failure.
    """
    try:
        _, _, ll = gas_filter(omega, gamma, beta, u, copula,
                              scaling, score_eps)
        if np.isnan(ll) or np.isinf(ll):
            return 1e10
        return -ll
    except Exception:
        return 1e10


# ══════════════════════════════════════════════════════════════════
# GAS Rosenblatt transform (for GoF tests)
# ══════════════════════════════════════════════════════════════════

def gas_rosenblatt(omega, gamma, beta, u, copula, scaling='unit',
                   score_eps=1e-4):
    """Rosenblatt transform for the bivariate GAS copula model.

    Because GAS is observation-driven, r_t is deterministic (no
    latent uncertainty to marginalize over). So the Rosenblatt is:
        e_1 = u_1,    e_2 = h(u_2, u_1; r_t).

    Returns
    -------
    e : ndarray (T, 2)
    """
    _, r_path, _ = gas_filter(omega, gamma, beta, u, copula,
                              scaling, score_eps)
    T = len(u)
    e = np.empty((T, 2))
    e[:, 0] = u[:, 0]
    e[:, 1] = copula.h(u[:, 1], u[:, 0], r_path)
    return np.clip(e, 1e-6, 1.0 - 1e-6)


# ══════════════════════════════════════════════════════════════════
# GAS h-function (for vine pseudo-observations)
# ══════════════════════════════════════════════════════════════════

def gas_mixture_h(omega, gamma, beta, u, copula, scaling='unit',
                  score_eps=1e-4):
    """h(u2 | u1; Psi(g_t)) along GAS-filtered path.

    This is the GAS counterpart of SCAR's mixture h-function.
    Unlike SCAR (which integrates over the predictive distribution),
    GAS uses the point estimate Psi(g_t).

    Returns
    -------
    h_vals : ndarray (T,)
    """
    _, r_path, _ = gas_filter(omega, gamma, beta, u, copula,
                              scaling, score_eps)
    return np.clip(
        copula.h(u[:, 1], u[:, 0], r_path),
        1e-6, 1.0 - 1e-6)
