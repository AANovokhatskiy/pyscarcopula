import numpy as np
from numba import njit

from pyscarcopula.copula.base import BivariateCopula, _broadcast


@njit(cache=True)
def _log1mexp(x):
    if x > 0.693:
        return np.log1p(-np.exp(-x))
    else:
        return np.log(-np.expm1(-x))


@njit(cache=True)
def _joe_log_pdf(u1, u2, r):
    n = len(u1)
    out = np.empty(n)
    for i in range(n):
        eps = 1e-300
        v1 = min(max(u1[i], eps), 1.0 - eps)
        v2 = min(max(u2[i], eps), 1.0 - eps)
        ri = r[i] if r.shape[0] > 1 else r[0]

        q1 = 1.0 - v1
        q2 = 1.0 - v2
        q1 = min(max(q1, eps), 1.0 - eps)
        q2 = min(max(q2, eps), 1.0 - eps)

        log_q1 = np.log(q1)
        log_q2 = np.log(q2)

        log_t1 = ri * log_q1 + _log1mexp(-ri * log_q2)
        log_t2 = ri * log_q2

        log_max = max(log_t1, log_t2)
        log_min = min(log_t1, log_t2)
        log_B = log_max + np.log1p(np.exp(log_min - log_max))

        B = np.exp(log_B)
        if B > ri - 1.0:
            log_rp = log_B + np.log1p((ri - 1.0) / B)
        else:
            log_rp = np.log(ri - 1.0) + np.log1p(B / (ri - 1.0))

        out[i] = ((ri - 1.0) * (log_q1 + log_q2)
                  + log_rp
                  - (2.0 - 1.0 / ri) * log_B)
    return out


@njit(cache=True)
def _joe_pdf(u1, u2, r):
    return np.exp(_joe_log_pdf(u1, u2, r))


@njit(cache=True)
def _joe_h(u0, u1, r):
    n = len(u0)
    out = np.empty(n)
    eps = 1e-6
    for i in range(n):
        v0 = min(max(u0[i], eps), 1.0 - eps)
        v1 = min(max(u1[i], eps), 1.0 - eps)
        ri = r[i] if r.shape[0] > 1 else r[0]

        if ri < 1.0 + 1e-8:
            out[i] = v0  # independence limit
        else:
            q0 = (1.0 - v0) ** ri
            q1 = (1.0 - v1) ** ri
            B = q0 + q1 - q0 * q1
            if B < 1e-300:
                out[i] = v0
            else:
                val = q1 * (1.0 - q0) * B ** (1.0 / ri - 2.0) / (1.0 - v1)
                # At r=1 limit: val = (1-v1)*(v0) / (1-v1) = v0
                # General: h = dC/dv1 with sign
                # The formula: -(x3 * (x1 - x3*x2)^(1/r-1) * x2 / (1-v1))
                # where x1 = q0, x2 = q1, x3 = q0 - 1
                x3 = q0 - 1.0
                inner = q0 - x3 * q1
                if inner < 1e-300:
                    out[i] = v0
                else:
                    val = -(x3 * inner ** (1.0 / ri - 1.0) * q1 / (1.0 - v1))
                    out[i] = min(max(val, eps), 1.0 - eps)
    return out


@njit(cache=True)
def _joe_V(n, r_val):
    """Sample V for Joe copula (Sibuya distribution)."""
    out = np.empty(n)
    p0 = 1.0 / r_val
    for k in range(n):
        u = np.random.uniform(0, 1)
        i = 1
        p = p0
        F = p
        while u > F:
            mult = (-1.0) * (p0 - float(i + 1) + 1.0) / float(i + 1)
            p = mult * p
            F = F + p
            i += 1
        out[k] = float(i)
    return out


@njit(cache=True)
def _joe_transform(x):
    return x * np.tanh(x) + 1.0001


class JoeCopula(BivariateCopula):

    def __init__(self, rotate: int = 0):
        super().__init__(rotate)
        self._name = "Joe copula"
        self._bounds = [(1.0001, np.inf)]

    @staticmethod
    def transform(x):
        return _joe_transform(x)

    @staticmethod
    def inv_transform(r):
        return r - 1.0

    def pdf_unrotated(self, u1, u2, r):
        return _joe_pdf(*_broadcast(u1, u2, r))

    def log_pdf_unrotated(self, u1, u2, r):
        return _joe_log_pdf(*_broadcast(u1, u2, r))

    @staticmethod
    def psi(t, r):
        return 1.0 - (1.0 - np.exp(-t)) ** (1.0 / r)

    def V(self, n, r):
        _r = np.atleast_1d(np.asarray(r, dtype=np.float64))
        return _joe_V(n, _r[0])

    def h_unrotated(self, u, v, r):
        return _joe_h(*_broadcast(u, v, r))

    # def h_inverse_unrotated(self, u, v, r):
    #     raise NotImplementedError("Joe h_inverse requires numerical inversion")
