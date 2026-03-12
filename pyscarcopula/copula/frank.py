import numpy as np
from numba import njit

from pyscarcopula.copula.base import BivariateCopula, _broadcast


@njit(cache=True)
def _log1mexp(x):
    """log(1 - exp(-x)) for x > 0, stable."""
    if x > 0.693:
        return np.log1p(-np.exp(-x))
    elif x > 0:
        return np.log(-np.expm1(-x))
    else:
        return -np.inf


@njit(cache=True)
def _frank_log_pdf(u1, u2, r):
    n = len(u1)
    out = np.empty(n)
    for i in range(n):
        v1 = u1[i]
        v2 = u2[i]
        ri = r[i] if r.shape[0] > 1 else r[0]

        a = ri * v1
        b = ri * v2
        s = ri

        log_num = np.log(ri) + _log1mexp(s) - a - b
        log_t1 = -a + _log1mexp(b)
        log_t2 = -b + _log1mexp(s - b)

        log_max = max(log_t1, log_t2)
        log_min = min(log_t1, log_t2)
        log_absD = log_max + np.log1p(np.exp(log_min - log_max))

        out[i] = log_num - 2.0 * log_absD
    return out


@njit(cache=True)
def _frank_pdf(u1, u2, r):
    return np.exp(_frank_log_pdf(u1, u2, r))


@njit(cache=True)
def _frank_h(u, v, r):
    """
    h-function for Frank copula:  h(u, v; r) = C(u | v; r) = dC(v, u; r)/dv.

    Numerically stable formulation in log-space.

    The direct formula is:
        h = exp(-rv) * (1 - exp(-ru)) / D
    where D = exp(-ru)*(1 - exp(-rv)) + exp(-rv)*(1 - exp(-r(1-v))).

    We compute log(h) = -rv + log(1-exp(-ru)) - log(D) using log1mexp
    and logsumexp to avoid catastrophic cancellation for large r.
    """
    n = len(u)
    out = np.empty(n)
    eps = 1e-10
    for i in range(n):
        _u = min(max(u[i], eps), 1.0 - eps)
        _v = min(max(v[i], eps), 1.0 - eps)
        ri = r[i] if r.shape[0] > 1 else r[0]

        if abs(ri) < 1e-8:
            out[i] = _u
            continue

        ru = ri * _u
        rv = ri * _v
        r_1mv = ri * (1.0 - _v)

        # log(numerator) = -rv + log(1 - exp(-ru))
        log_numer = -rv + _log1mexp(ru)

        # Denominator D = A + B where:
        #   A = exp(-ru) * (1 - exp(-rv))     -> log_A = -ru + log(1 - exp(-rv))
        #   B = exp(-rv) * (1 - exp(-r(1-v))) -> log_B = -rv + log(1 - exp(-r(1-v)))
        log_A = -ru + _log1mexp(rv)
        log_B = -rv + _log1mexp(r_1mv)

        # log(D) = logsumexp(log_A, log_B)
        log_max_d = max(log_A, log_B)
        log_D = log_max_d + np.log1p(np.exp(min(log_A, log_B) - log_max_d))

        log_h = log_numer - log_D

        if log_h < -700.0:
            out[i] = eps
        elif log_h > -eps:
            out[i] = 1.0 - eps
        else:
            out[i] = np.exp(log_h)

    return out


@njit(cache=True)
def _frank_h_inv(u, v, r):
    """
    Inverse h-function for Frank copula: find t such that h(t, v; r) = u.

    Stable formulation: from h = u, solving for t gives
        arg = (Q + exp(-r)) / (Q + 1)
        t   = -log(arg) / r
    where Q = (1-u)/u * exp(-r*v).
    """
    n = len(u)
    out = np.empty(n)
    eps = 1e-10
    for i in range(n):
        _u = min(max(u[i], eps), 1.0 - eps)
        _v = min(max(v[i], eps), 1.0 - eps)
        ri = r[i] if r.shape[0] > 1 else r[0]

        if abs(ri) < 1e-8:
            out[i] = _u
            continue

        x2 = np.exp(-ri * _v)
        x3 = np.exp(-ri)

        Q = (1.0 - _u) / _u * x2

        # arg = (Q + x3) / (Q + 1)
        # For large r: x3 ≈ 0, so arg ≈ Q / (Q + 1)
        # For Q very small: arg ≈ x3, t ≈ -log(x3)/r = 1
        # For Q very large: arg ≈ 1, t ≈ 0
        # Both limits make sense: u->0 => t->0, u->1 => t->1

        numer = Q + x3
        denom = Q + 1.0

        if denom < eps:
            out[i] = eps
            continue

        arg = numer / denom

        if arg <= 0:
            out[i] = eps
        elif arg >= 1.0 - eps:
            # log(arg) ≈ 0 but we need precision
            # Use log1p: log(arg) = log(1 - (1-arg)) = log(1 - (1-x3-Q)/(Q+1))
            # 1 - arg = (1 - x3) / (Q + 1)
            one_m_arg = (1.0 - x3) / denom
            if one_m_arg < eps:
                out[i] = 1.0 - eps
            else:
                t = -np.log1p(-one_m_arg) / ri
                out[i] = min(max(t, eps), 1.0 - eps)
        else:
            t = -np.log(arg) / ri
            out[i] = min(max(t, eps), 1.0 - eps)

    return out


@njit(cache=True)
def _frank_bivariate_sample(n, r):
    """Direct conditional inversion sampling for Frank (faster than Marshall-Olkin)."""
    out = np.empty((n, 2))
    for k in range(n):
        ri = r[k] if r.shape[0] > 1 else r[0]
        u0 = np.random.uniform(0, 1)
        v = np.random.uniform(0, 1)
        t = np.exp(-ri * u0)
        p = np.exp(-ri)
        f1 = v * (1.0 - p)
        f2 = t + v * (1.0 - t)
        if abs(f1 - f2) < 1e-9:
            u1 = u0
        else:
            u1 = -np.log(1.0 - f1 / f2) / ri
        out[k, 0] = u0
        out[k, 1] = u1
    return out


@njit(cache=True)
def _frank_transform(x):
    return x * np.tanh(x) + 0.0001


class FrankCopula(BivariateCopula):
    """Frank copula. No rotation support (symmetric)."""

    def __init__(self, rotate: int = 0):
        if rotate != 0:
            raise ValueError("Rotation not supported for Frank copula")
        super().__init__(0)
        self._name = "Frank copula"
        self._bounds = [(0.0001, np.inf)]

    @staticmethod
    def transform(x):
        return _frank_transform(x)

    @staticmethod
    def inv_transform(r):
        return r

    def pdf_unrotated(self, u1, u2, r):
        return _frank_pdf(*_broadcast(u1, u2, r))

    def log_pdf_unrotated(self, u1, u2, r):
        return _frank_log_pdf(*_broadcast(u1, u2, r))

    def sample(self, n, r, rng=None):
        _r = np.atleast_1d(np.asarray(r, dtype=np.float64))
        if _r.size == 1:
            _r = np.full(n, _r[0])
        return _frank_bivariate_sample(n, _r)

    def h_unrotated(self, u, v, r):
        return _frank_h(*_broadcast(u, v, r))

    def h_inverse_unrotated(self, u, v, r):
        return _frank_h_inv(*_broadcast(u, v, r))
