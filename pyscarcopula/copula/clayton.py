import numpy as np
from numba import njit

from pyscarcopula.copula.base import BivariateCopula, _broadcast

@njit(cache=True)
def _clayton_log_pdf(u1, u2, r):
    n = len(u1)
    out = np.empty(n)
    for i in range(n):
        eps = 1e-300
        v1 = min(max(u1[i], eps), 1.0 - eps)
        v2 = min(max(u2[i], eps), 1.0 - eps)
        ri = r[i] if r.shape[0] > 1 else r[0]

        log_v1 = np.log(v1)
        log_v2 = np.log(v2)

        a = -ri * log_v1
        b = -ri * log_v2
        log_max = max(a, b)
        log_min = min(a, b)
        correction = np.exp(log_min - log_max) - np.exp(-log_max)
        log_S = log_max + np.log1p(correction)

        out[i] = (np.log1p(ri)
                  + (-ri - 1.0) * log_v1
                  + (-ri - 1.0) * log_v2
                  + (-2.0 - 1.0 / ri) * log_S)
    return out


@njit(cache=True)
def _clayton_pdf(u1, u2, r):
    return np.exp(_clayton_log_pdf(u1, u2, r))


@njit(cache=True)
def _clayton_h(u0, u1, r):
    n = len(u0)
    out = np.empty(n)
    eps = 1e-6
    for i in range(n):
        v0 = min(max(u0[i], eps), 1.0 - eps)
        v1 = min(max(u1[i], eps), 1.0 - eps)
        ri = r[i] if r.shape[0] > 1 else r[0]

        if ri < 1e-8:
            out[i] = v0  # independence limit
        else:
            S = v0 ** (-ri) + v1 ** (-ri) - 1.0
            if S < 1e-300:
                out[i] = v0
            else:
                val = v1 ** (-ri - 1.0) * S ** (-1.0 - 1.0 / ri)
                out[i] = min(max(val, eps), 1.0 - eps)
    return out


@njit(cache=True)
def _clayton_h_inv(u0, u1, r):
    n = len(u0)
    out = np.empty(n)
    eps = 1e-6
    for i in range(n):
        v0 = min(max(u0[i], eps), 1.0 - eps)
        v1 = min(max(u1[i], eps), 1.0 - eps)
        ri = r[i] if r.shape[0] > 1 else r[0]

        if ri < 1e-8:
            out[i] = v0  # independence limit
        else:
            a = v0 * v1 ** (ri + 1.0)
            if a < 1e-300:
                out[i] = eps
            else:
                base = a ** (-ri / (1.0 + ri)) + 1.0 - v1 ** (-ri)
                if base < 1e-300:
                    out[i] = eps
                else:
                    val = base ** (-1.0 / ri)
                    out[i] = min(max(val, eps), 1.0 - eps)
    return out


@njit(cache=True)
def _clayton_transform(x):
    return x * np.tanh(x) + 0.0001


class ClaytonCopula(BivariateCopula):

    def __init__(self, rotate: int = 0):
        super().__init__(rotate)
        self._name = "Clayton copula"
        self._bounds = [(0.0001, np.inf)]

    @staticmethod
    def transform(x):
        return _clayton_transform(x)

    @staticmethod
    def inv_transform(r):
        return r

    def pdf_unrotated(self, u1, u2, r):
        u1a, u2a, ra = _broadcast(u1, u2, r)
        return _clayton_pdf(u1a, u2a, ra)

    def log_pdf_unrotated(self, u1, u2, r):
        u1a, u2a, ra = _broadcast(u1, u2, r)
        return _clayton_log_pdf(u1a, u2a, ra)

    @staticmethod
    def psi(t, r):
        return (1.0 + t * r) ** (-1.0 / r)

    def V(self, n, r):
        _r = np.atleast_1d(np.asarray(r, dtype=np.float64))
        return np.random.gamma(1.0 / _r[0], scale=1.0, size=n)

    def h_unrotated(self, u, v, r):
        return _clayton_h(*_broadcast(u, v, r))

    def h_inverse_unrotated(self, u, v, r):
        return _clayton_h_inv(*_broadcast(u, v, r))
