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


@njit(cache=True)
def _clayton_dtransform(x):
    """d/dx [x·tanh(x)] = tanh(x) + x·(1 - tanh²(x))"""
    n = len(x)
    out = np.empty(n)
    for i in range(n):
        th = np.tanh(x[i])
        out[i] = th + x[i] * (1.0 - th * th)
    return out


@njit(cache=True)
def _clayton_dlogc_dr(u1, u2, r):
    """Analytical d(log c)/dr for Clayton copula."""
    n = len(u1)
    out = np.empty(n)
    for i in range(n):
        eps = 1e-300
        v1 = min(max(u1[i], eps), 1.0 - eps)
        v2 = min(max(u2[i], eps), 1.0 - eps)
        ri = r[i] if r.shape[0] > 1 else r[0]

        log_v1 = np.log(v1)
        log_v2 = np.log(v2)

        t1 = v1 ** (-ri)
        t2 = v2 ** (-ri)
        S = t1 + t2 - 1.0
        if S < eps:
            S = eps

        dS = -log_v1 * t1 - log_v2 * t2

        out[i] = (1.0 / (1.0 + ri)
                  - log_v1 - log_v2
                  + np.log(S) / (ri * ri)
                  + (-2.0 - 1.0 / ri) * dS / S)
    return out


@njit(cache=True)
def _clayton_pdf_and_grad_batch(u_all, r_grid, dpsi, rotation):
    """Fused batch: fi and dfi_dx for all T observations at once."""
    T = u_all.shape[0]
    K = len(r_grid)
    fi = np.empty((T, K))
    dfi = np.empty((T, K))
    eps = 1e-300

    for t in range(T):
        u1_raw = u_all[t, 0]
        u2_raw = u_all[t, 1]
        if rotation == 90:
            u1_raw = 1.0 - u1_raw
        elif rotation == 180:
            u1_raw = 1.0 - u1_raw
            u2_raw = 1.0 - u2_raw
        elif rotation == 270:
            u2_raw = 1.0 - u2_raw

        v1 = min(max(u1_raw, eps), 1.0 - eps)
        v2 = min(max(u2_raw, eps), 1.0 - eps)
        log_v1 = np.log(v1)
        log_v2 = np.log(v2)

        for j in range(K):
            ri = r_grid[j]

            a = -ri * log_v1
            b = -ri * log_v2
            log_max = max(a, b)
            log_min = min(a, b)
            correction = np.exp(log_min - log_max) - np.exp(-log_max)
            log_S = log_max + np.log1p(correction)

            log_c = (np.log1p(ri)
                     + (-ri - 1.0) * log_v1
                     + (-ri - 1.0) * log_v2
                     + (-2.0 - 1.0 / ri) * log_S)
            c_val = np.exp(log_c)
            fi[t, j] = c_val

            # d(log c)/dr
            t1 = v1 ** (-ri)
            t2 = v2 ** (-ri)
            S = t1 + t2 - 1.0
            if S < eps:
                S = eps
            dS = -log_v1 * t1 - log_v2 * t2

            dlogc = (1.0 / (1.0 + ri)
                     - log_v1 - log_v2
                     + np.log(S) / (ri * ri)
                     + (-2.0 - 1.0 / ri) * dS / S)
            dfi[t, j] = c_val * dlogc * dpsi[j]

    return fi, dfi


class ClaytonCopula(BivariateCopula):

    def __init__(self, rotate: int = 0):
        super().__init__(rotate)
        self._name = "Clayton copula"
        self._bounds = [(0.0001, np.inf)]

    @staticmethod
    def transform(x):
        return _clayton_transform(x)

    @staticmethod
    def dtransform(x):
        return _clayton_dtransform(np.atleast_1d(np.asarray(x, dtype=np.float64)))

    @staticmethod
    def inv_transform(r):
        return r

    def pdf_unrotated(self, u1, u2, r):
        u1a, u2a, ra = _broadcast(u1, u2, r)
        return _clayton_pdf(u1a, u2a, ra)

    def log_pdf_unrotated(self, u1, u2, r):
        u1a, u2a, ra = _broadcast(u1, u2, r)
        return _clayton_log_pdf(u1a, u2a, ra)

    def dlog_pdf_dr_unrotated(self, u1, u2, r):
        u1a, u2a, ra = _broadcast(u1, u2, r)
        return _clayton_dlogc_dr(u1a, u2a, ra)

    @staticmethod
    def psi(t, r):
        return (1.0 + t * r) ** (-1.0 / r)

    def V(self, n, r):
        _r = np.atleast_1d(np.asarray(r, dtype=np.float64))
        return np.random.gamma(1.0 / _r, scale=_r, size=n)

    def h_unrotated(self, u, v, r):
        return _clayton_h(*_broadcast(u, v, r))

    def h_inverse_unrotated(self, u, v, r):
        return _clayton_h_inv(*_broadcast(u, v, r))

    def pdf_and_grad_on_grid_batch(self, u, x_grid):
        x = np.asarray(x_grid, dtype=np.float64)
        r_grid = _clayton_transform(x)
        dpsi = _clayton_dtransform(x)
        return _clayton_pdf_and_grad_batch(
            np.asarray(u, dtype=np.float64), r_grid, dpsi, self._rotate)

    def copula_grid_batch(self, u, x_grid):
        x = np.asarray(x_grid, dtype=np.float64)
        r_grid = _clayton_transform(x)
        dpsi = _clayton_dtransform(x)
        fi, _ = _clayton_pdf_and_grad_batch(
            np.asarray(u, dtype=np.float64), r_grid, dpsi, self._rotate)
        return fi
