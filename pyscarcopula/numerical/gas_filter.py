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
    - strategy/initial_point.py for GAS-based initial point estimation
    - stattests.py for GAS Rosenblatt transform
    - vine/_edge.py for GAS h-functions in vine edges
"""

import numpy as np
from numba import njit

from pyscarcopula.copula.base import (
    _softplus_dtransform,
    _softplus_transform,
    _xtanh_dtransform,
    _xtanh_transform,
)
from pyscarcopula.copula.clayton import (
    ClaytonCopula,
    _clayton_dlogc_dr,
    _clayton_log_pdf,
)
from pyscarcopula.copula.elliptical import (
    BivariateGaussianCopula,
    _gauss_dlog_pdf_drho,
    _gauss_log_pdf_numba,
)
from pyscarcopula.copula.frank import (
    FrankCopula,
    _frank_dlogc_dr,
    _frank_log_pdf,
)
from pyscarcopula.copula.gumbel import (
    GumbelCopula,
    _gumbel_dlogc_dr,
    _gumbel_log_pdf,
)
from pyscarcopula.copula.joe import (
    JoeCopula,
    _joe_dlogc_dr,
    _joe_log_pdf,
)

G_CLIP = 50.0
S_CLIP = 100.0

_GAS_FAMILY_GAUSSIAN = 1
_GAS_FAMILY_CLAYTON = 2
_GAS_FAMILY_FRANK = 3
_GAS_FAMILY_GUMBEL = 4
_GAS_FAMILY_JOE = 5

_GAS_TRANSFORM_SOFTPLUS = 1
_GAS_TRANSFORM_XTANH = 2


def _gas_initial_g(omega, beta):
    if abs(beta) < 1.0 - 1e-8:
        return omega / (1.0 - beta)
    return omega


def _gas_update_from_score(omega, gamma, beta, g_t, s_t):
    s_t = np.clip(s_t, -S_CLIP, S_CLIP)
    g_next = omega + beta * g_t + gamma * s_t
    return float(np.clip(g_next, -G_CLIP, G_CLIP))


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
def _gas_archimedean_offset(family):
    if family == _GAS_FAMILY_CLAYTON or family == _GAS_FAMILY_FRANK:
        return 0.0001
    if family == _GAS_FAMILY_GUMBEL or family == _GAS_FAMILY_JOE:
        return 1.0001
    raise ValueError("Unsupported GAS copula family")


@njit(cache=True)
def _gas_transform(g_t, family, transform_type, x_arr):
    x_arr[0] = g_t
    if family == _GAS_FAMILY_GAUSSIAN:
        if transform_type == _GAS_TRANSFORM_SOFTPLUS:
            return 0.9999 * np.tanh(g_t / 4.0)
        if transform_type == _GAS_TRANSFORM_XTANH:
            return 0.9999 * np.tanh(g_t / 4.0)
        raise ValueError("Unsupported GAS transform_type")
    if (family == _GAS_FAMILY_CLAYTON or family == _GAS_FAMILY_FRANK
            or family == _GAS_FAMILY_GUMBEL or family == _GAS_FAMILY_JOE):
        offset = _gas_archimedean_offset(family)
        if transform_type == _GAS_TRANSFORM_SOFTPLUS:
            return _softplus_transform(x_arr, offset)[0]
        if transform_type == _GAS_TRANSFORM_XTANH:
            return _xtanh_transform(x_arr, offset)[0]
        raise ValueError("Unsupported GAS transform_type")
    raise ValueError("Unsupported GAS copula family")


@njit(cache=True)
def _gas_dtransform(g_t, family, transform_type, x_arr):
    x_arr[0] = g_t
    if family == _GAS_FAMILY_GAUSSIAN:
        if transform_type == _GAS_TRANSFORM_SOFTPLUS:
            th = np.tanh(g_t / 4.0)
            return 0.9999 / 4.0 * (1.0 - th * th)
        if transform_type == _GAS_TRANSFORM_XTANH:
            th = np.tanh(g_t / 4.0)
            return 0.9999 / 4.0 * (1.0 - th * th)
        raise ValueError("Unsupported GAS transform_type")
    if (family == _GAS_FAMILY_CLAYTON or family == _GAS_FAMILY_FRANK
            or family == _GAS_FAMILY_GUMBEL or family == _GAS_FAMILY_JOE):
        if transform_type == _GAS_TRANSFORM_SOFTPLUS:
            return _softplus_dtransform(x_arr)[0]
        if transform_type == _GAS_TRANSFORM_XTANH:
            return _xtanh_dtransform(x_arr)[0]
        raise ValueError("Unsupported GAS transform_type")
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
                           transform_type):
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
        r_t = _gas_transform(g_t, family, transform_type, x_arr)
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
            dpsi_dg = _gas_dtransform(g_t, family, transform_type, x_arr)
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
        transform_type = _gas_transform_type_code(copula)
        return _GAS_FAMILY_GAUSSIAN, 0, transform_type
    if isinstance(copula, ClaytonCopula):
        transform_type = _gas_transform_type_code(copula)
        return (
            _GAS_FAMILY_CLAYTON,
            int(copula.rotate),
            transform_type,
        )
    if isinstance(copula, FrankCopula):
        transform_type = _gas_transform_type_code(copula)
        return (
            _GAS_FAMILY_FRANK,
            0,
            transform_type,
        )
    if isinstance(copula, GumbelCopula):
        transform_type = _gas_transform_type_code(copula)
        return (
            _GAS_FAMILY_GUMBEL,
            int(copula.rotate),
            transform_type,
        )
    if isinstance(copula, JoeCopula):
        transform_type = _gas_transform_type_code(copula)
        return (
            _GAS_FAMILY_JOE,
            int(copula.rotate),
            transform_type,
        )
    return None


def _gas_transform_type_code(copula):
    transform_type = getattr(copula, '_transform_type', 'softplus')
    if transform_type == 'softplus':
        return _GAS_TRANSFORM_SOFTPLUS
    if transform_type == 'xtanh':
        return _GAS_TRANSFORM_XTANH
    raise ValueError(
        "transform_type must be 'xtanh' or 'softplus', "
        f"got '{transform_type}'"
    )


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


def _gas_score_finite_difference(u1, u2, g_t, ll_t, copula, scaling,
                                 score_eps):
    g_plus = g_t + score_eps
    g_minus = g_t - score_eps
    r_plus = float(copula.transform(np.array([g_plus]))[0])
    r_minus = float(copula.transform(np.array([g_minus]))[0])

    ll_plus = float(np.sum(
        copula.log_pdf(u1, u2, np.full(len(u1), r_plus))))
    ll_minus = float(np.sum(
        copula.log_pdf(u1, u2, np.full(len(u1), r_minus))))

    nabla = (ll_plus - ll_minus) / (2.0 * score_eps)
    if scaling != 'fisher':
        return nabla

    d2 = (ll_plus - 2.0 * ll_t + ll_minus) / (score_eps ** 2)
    fisher = max(-d2, 1e-6)
    return nabla / fisher


# ══════════════════════════════════════════════════════════════════
# Core GAS filter
# ══════════════════════════════════════════════════════════════════

def _supports_multivariate_log_pdf(copula, u):
    if np.ndim(u) != 2 or not hasattr(copula, 'log_pdf_rows'):
        return False
    expected_d = getattr(copula, 'd', None)
    return expected_d is None or u.shape[1] == expected_d


def _validate_multivariate_log_pdf_input(copula, u):
    if np.ndim(u) != 2 or not hasattr(copula, 'log_pdf_rows'):
        return False
    expected_d = getattr(copula, 'd', None)
    if expected_d is not None and u.shape[1] != expected_d:
        raise ValueError(
            f"u must have shape (T, {expected_d}) for {copula.name}, "
            f"got {u.shape}")
    return True


def _multivariate_log_pdf_row(copula, u_row, r_t, t_index):
    method = copula.log_pdf_rows
    code = getattr(method, '__code__', None)
    if code is None or 't_index' in code.co_varnames:
        vals = copula.log_pdf_rows(
            u_row, np.array([r_t], dtype=np.float64), t_index=t_index)
    else:
        vals = copula.log_pdf_rows(
            u_row, np.array([r_t], dtype=np.float64))
    return float(np.asarray(vals, dtype=np.float64)[0])


def _transform_scalar(copula, g_t):
    method = getattr(copula, 'transform_scalar', None)
    if method is not None:
        return float(method(g_t))
    return float(copula.transform(np.array([g_t], dtype=np.float64))[0])


def _dtransform_scalar(copula, g_t):
    method = getattr(copula, 'dtransform_scalar', None)
    if method is not None:
        return float(method(g_t))
    return float(copula.dtransform(np.array([g_t], dtype=np.float64))[0])


def _gas_score_multivariate(u_row, t_index, g_t, ll_t, copula, scaling,
                            score_eps):
    """Score d log c(u_t; Psi(g_t)) / d g_t for d-dimensional copulas."""
    if scaling != 'fisher' and hasattr(copula, 'dlog_pdf_dr_rows'):
        r_t = _transform_scalar(copula, g_t)
        dlog_dr = float(copula.dlog_pdf_dr_rows(
            u_row, np.array([r_t], dtype=np.float64),
            t_index=t_index)[0])
        dpsi_dg = _dtransform_scalar(copula, g_t)
        return dlog_dr * dpsi_dg

    g_plus = g_t + score_eps
    g_minus = g_t - score_eps
    r_plus = _transform_scalar(copula, g_plus)
    r_minus = _transform_scalar(copula, g_minus)

    ll_plus = _multivariate_log_pdf_row(copula, u_row, r_plus, t_index)
    ll_minus = _multivariate_log_pdf_row(copula, u_row, r_minus, t_index)

    nabla_t = (ll_plus - ll_minus) / (2.0 * score_eps)
    if scaling != 'fisher':
        return nabla_t

    d2 = (ll_plus - 2.0 * ll_t + ll_minus) / (score_eps ** 2)
    fisher = max(-d2, 1e-6)
    return nabla_t / fisher


def _gas_next_g_from_observation(
        omega, gamma, beta, g_t, u_row, copula, scaling='unit',
        score_eps=1e-4, t_index=None, r_t=None, ll_t=None,
        analytical=True):
    """Apply one GAS score update from a single observed row."""
    u_row = np.asarray(u_row, dtype=np.float64)
    if u_row.ndim != 2 or len(u_row) == 0:
        raise ValueError("u_row must contain at least one observation")

    if _supports_multivariate_log_pdf(copula, u_row):
        t_index = 0 if t_index is None else int(t_index)
        if r_t is None:
            r_t = _transform_scalar(copula, g_t)
        if ll_t is None:
            ll_t = _multivariate_log_pdf_row(copula, u_row, r_t, t_index)
        s_t = _gas_score_multivariate(
            u_row, t_index, g_t, ll_t, copula, scaling, score_eps)
    else:
        if r_t is None:
            r_t = float(copula.transform(np.array([g_t]))[0])
        u1 = u_row[:, 0]
        u2 = u_row[:, 1]
        if ll_t is None:
            ll_t = float(copula.log_pdf(u1, u2, np.array([r_t]))[0])
        if analytical:
            s_t = _gas_score(
                u1, u2, g_t, r_t, ll_t, copula, scaling, score_eps)
        else:
            s_t = _gas_score_finite_difference(
                u1, u2, g_t, ll_t, copula, scaling, score_eps)

    if not np.isfinite(ll_t) or not np.isfinite(s_t):
        raise FloatingPointError("invalid GAS log-likelihood or score")
    g_next = _gas_update_from_score(omega, gamma, beta, g_t, s_t)
    return g_next, float(r_t), float(ll_t), float(np.clip(s_t, -S_CLIP, S_CLIP))


def _gas_filter_multivariate(omega, gamma, beta, u, copula, scaling='unit',
                             score_eps=1e-4):
    """Run GAS filter for scalar-latent multivariate copulas."""
    T = len(u)
    g_path = np.empty(T)
    r_path = np.empty(T)
    total_logL = 0.0

    g_t = _gas_initial_g(omega, beta)

    combined_rows = None
    if scaling != 'fisher':
        combined_rows = getattr(copula, 'log_pdf_and_dlog_dr_rows', None)

    for t in range(T):
        g_path[t] = g_t
        r_t = _transform_scalar(copula, g_t)
        r_path[t] = r_t

        u_row = u[t:t + 1]
        s_t = None
        if combined_rows is not None and t < T - 1:
            ll_vals, dlog_vals = combined_rows(
                u_row, np.array([r_t], dtype=np.float64), t_index=t)
            ll_t = float(np.asarray(ll_vals, dtype=np.float64)[0])
            dlog_dr = float(np.asarray(dlog_vals, dtype=np.float64)[0])
            s_t = dlog_dr * _dtransform_scalar(copula, g_t)
        else:
            ll_t = _multivariate_log_pdf_row(copula, u_row, r_t, t)
        if not np.isfinite(ll_t):
            return g_path, r_path, -1e10

        total_logL += ll_t

        if t < T - 1:
            if s_t is None:
                try:
                    g_t, _, _, _ = _gas_next_g_from_observation(
                        omega, gamma, beta, g_t, u_row, copula,
                        scaling, score_eps, t_index=t, r_t=r_t, ll_t=ll_t)
                except FloatingPointError:
                    return g_path, r_path, -1e10
            else:
                if not np.isfinite(s_t):
                    return g_path, r_path, -1e10
                g_t = _gas_update_from_score(omega, gamma, beta, g_t, s_t)

    return g_path, r_path, total_logL


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
    u = np.asarray(u, dtype=np.float64)
    if _validate_multivariate_log_pdf_input(copula, u):
        return _gas_filter_multivariate(
            float(omega), float(gamma), float(beta),
            u, copula, scaling, score_eps)

    kernel_args = _gas_unit_kernel_args(copula, scaling)
    if kernel_args is not None:
        family, rotation, transform_type = kernel_args
        return _gas_filter_unit_numba(
            float(omega), float(gamma), float(beta),
            u,
            family, rotation, transform_type)

    T = len(u)
    g_path = np.empty(T)
    r_path = np.empty(T)
    total_logL = 0.0

    # Initial value: unconditional mean g_bar = omega / (1 - beta)
    g_t = _gas_initial_g(omega, beta)

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
            try:
                g_t, _, _, _ = _gas_next_g_from_observation(
                    omega, gamma, beta, g_t, u[t:t + 1], copula,
                    scaling, score_eps, r_t=r_t, ll_t=ll_t)
            except FloatingPointError:
                return g_path, r_path, -1e10

    return g_path, r_path, total_logL


def gas_predict_param(omega, gamma, beta, u, copula, scaling='unit',
                      score_eps=1e-4, horizon='next'):
    """Return Psi(g) for GAS predictive sampling.

    ``current`` returns the last in-sample parameter Psi(g_{T-1}).
    ``next`` applies the final observation score and returns Psi(g_T).
    """
    u = np.asarray(u, dtype=np.float64)
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
    g_next, _, _, _ = _gas_next_g_from_observation(
        omega, gamma, beta, g_t, u[-1:], copula, scaling, score_eps,
        t_index=len(u) - 1)
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

def _gas_h_path(omega, gamma, beta, u, copula, scaling='unit',
                score_eps=1e-4):
    _, r_path, _ = gas_filter(omega, gamma, beta, u, copula,
                              scaling, score_eps)
    return np.clip(
        copula.h(u[:, 1], u[:, 0], r_path),
        1e-6, 1.0 - 1e-6)


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
    T = len(u)
    e = np.empty((T, 2))
    e[:, 0] = u[:, 0]
    e[:, 1] = _gas_h_path(omega, gamma, beta, u, copula, scaling, score_eps)
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
    return _gas_h_path(omega, gamma, beta, u, copula, scaling, score_eps)
