import numpy as np
from numba import njit

from pyscarcopula.copula.base import (
    BivariateCopula,
    _broadcast,
    _inv_xtanh_transform,
    _softplus_dtransform,
    _softplus_inv_transform,
    _softplus_transform,
    _xtanh_dtransform,
    _xtanh_transform,
)

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
    log_eps = np.log(eps)
    log_one_minus_eps = np.log(1.0 - eps)
    for i in range(n):
        v0 = min(max(u0[i], eps), 1.0 - eps)
        v1 = min(max(u1[i], eps), 1.0 - eps)
        ri = r[i] if r.shape[0] > 1 else r[0]

        if ri < 1e-8:
            out[i] = v0  # independence limit
        else:
            log_v0 = np.log(v0)
            log_v1 = np.log(v1)
            a = -ri * log_v0
            b = -ri * log_v1
            log_max = max(a, b)
            log_min = min(a, b)
            correction = np.exp(log_min - log_max) - np.exp(-log_max)
            if correction <= -1.0:
                out[i] = v0
            else:
                log_S = log_max + np.log1p(correction)
                log_h = (
                    (-ri - 1.0) * log_v1
                    + (-1.0 - 1.0 / ri) * log_S
                )
                if log_h <= log_eps:
                    out[i] = eps
                elif log_h >= log_one_minus_eps:
                    out[i] = 1.0 - eps
                else:
                    out[i] = np.exp(log_h)
    return out


@njit(cache=True)
def _clayton_h_pair(u0, u1, r):
    n = len(u0)
    out01 = np.empty(n)
    out10 = np.empty(n)
    eps = 1e-6
    log_eps = np.log(eps)
    log_one_minus_eps = np.log(1.0 - eps)
    for i in range(n):
        v0 = min(max(u0[i], eps), 1.0 - eps)
        v1 = min(max(u1[i], eps), 1.0 - eps)
        ri = r[i] if r.shape[0] > 1 else r[0]

        if ri < 1e-8:
            out01[i] = v0
            out10[i] = v1
        else:
            log_v0 = np.log(v0)
            log_v1 = np.log(v1)
            a = -ri * log_v0
            b = -ri * log_v1
            log_max = max(a, b)
            log_min = min(a, b)
            correction = np.exp(log_min - log_max) - np.exp(-log_max)
            if correction <= -1.0:
                out01[i] = v0
                out10[i] = v1
            else:
                log_S = log_max + np.log1p(correction)
                log_common = (-1.0 - 1.0 / ri) * log_S

                log_h01 = (-ri - 1.0) * log_v1 + log_common
                if log_h01 <= log_eps:
                    out01[i] = eps
                elif log_h01 >= log_one_minus_eps:
                    out01[i] = 1.0 - eps
                else:
                    out01[i] = np.exp(log_h01)

                log_h10 = (-ri - 1.0) * log_v0 + log_common
                if log_h10 <= log_eps:
                    out10[i] = eps
                elif log_h10 >= log_one_minus_eps:
                    out10[i] = 1.0 - eps
                else:
                    out10[i] = np.exp(log_h10)
    return out01, out10


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

        # log-space computation of S
        a = -ri * log_v1
        b = -ri * log_v2
        log_max = max(a, b)
        log_min = min(a, b)
        correction = np.exp(log_min - log_max) - np.exp(-log_max)
        log_S = log_max + np.log1p(correction)

        # log-space computation of dS/S
        log_abs_logv1 = np.log(-log_v1)
        log_abs_logv2 = np.log(-log_v2)
        p = log_abs_logv1 + a
        q = log_abs_logv2 + b
        pq_max = max(p, q)
        pq_min = min(p, q)
        log_dS = pq_max + np.log1p(np.exp(pq_min - pq_max))
        dS_over_S = np.exp(log_dS - log_S)

        out[i] = (1.0 / (1.0 + ri)
                  - log_v1 - log_v2
                  + log_S / (ri * ri)
                  + (-2.0 - 1.0 / ri) * dS_over_S)
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
        log_abs_logv1 = np.log(-log_v1)
        log_abs_logv2 = np.log(-log_v2)

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

            # d(log c)/dr — log-scale (reuses a, b, log_S from PDF block)
            p = log_abs_logv1 + a
            q = log_abs_logv2 + b
            pq_max = max(p, q)
            pq_min = min(p, q)
            log_dS = pq_max + np.log1p(np.exp(pq_min - pq_max))
            dS_over_S = np.exp(log_dS - log_S)

            dlogc = (1.0 / (1.0 + ri)
                     - log_v1 - log_v2
                     + log_S / (ri * ri)
                     + (-2.0 - 1.0 / ri) * dS_over_S)
            dfi[t, j] = c_val * dlogc * dpsi[j]

    return fi, dfi


class ClaytonCopula(BivariateCopula):

    def __init__(self, rotate: int = 0, transform_type: str = 'softplus'):
        super().__init__(rotate)
        self._name = "Clayton copula"
        if transform_type not in ('xtanh', 'softplus'):
            raise ValueError(f"transform_type must be 'xtanh' or 'softplus', got '{transform_type}'")
        self._transform_type = transform_type
        self._bounds = [(0.0001, np.inf)]

    def transform(self, x):
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        if self._transform_type == 'softplus':
            return _softplus_transform(x, 0.0001)
        if self._transform_type == 'xtanh':
            return _xtanh_transform(x, 0.0001)
        raise ValueError(f"Unsupported transform_type: {self._transform_type}")

    def dtransform(self, x):
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        if self._transform_type == 'softplus':
            return _softplus_dtransform(x)
        if self._transform_type == 'xtanh':
            return _xtanh_dtransform(x)
        raise ValueError(f"Unsupported transform_type: {self._transform_type}")

    def inv_transform(self, r):
        r = np.atleast_1d(np.asarray(r, dtype=np.float64))
        if self._transform_type == 'softplus':
            return _softplus_inv_transform(r, 0.0001)
        if self._transform_type == 'xtanh':
            return _inv_xtanh_transform(r, 0.0001)
        raise ValueError(f"Unsupported transform_type: {self._transform_type}")

    def tau_to_param(self, tau):
        tau = np.atleast_1d(np.asarray(tau, dtype=np.float64))
        if np.any((tau <= 0.0) | (tau >= 1.0)):
            raise ValueError("Clayton Kendall tau must be in (0, 1)")
        return 2.0 * tau / (1.0 - tau)

    def param_to_tau(self, r):
        r = np.atleast_1d(np.asarray(r, dtype=np.float64))
        if np.any(r <= 0.0):
            raise ValueError("Clayton parameter must be positive")
        return r / (r + 2.0)

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

    def V(self, n, r, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        _r = np.atleast_1d(np.asarray(r, dtype=np.float64))
        if _r.size == 1:
            _r = np.full(n, _r[0])
        return rng.gamma(1.0 / _r, scale=_r)

    def h_unrotated(self, u, v, r):
        return _clayton_h(*_broadcast(u, v, r))

    def h_pair(self, u, v, r):
        ua, va, ra = _broadcast(u, v, r)
        if self._rotate == 0:
            return _clayton_h_pair(ua, va, ra)
        return self.h(ua, va, ra), self.h(va, ua, ra)

    def h_inverse_unrotated(self, u, v, r):
        return _clayton_h_inv(*_broadcast(u, v, r))

    def pdf_and_grad_on_grid_batch(self, u, x_grid):
        x = np.asarray(x_grid, dtype=np.float64)
        r_grid = self.transform(x)
        dpsi = self.dtransform(x)
        return _clayton_pdf_and_grad_batch(
            np.asarray(u, dtype=np.float64), r_grid, dpsi, self._rotate)

    def copula_grid_batch(self, u, x_grid):
        x = np.asarray(x_grid, dtype=np.float64)
        r_grid = self.transform(x)
        dpsi = self.dtransform(x)
        fi, _ = _clayton_pdf_and_grad_batch(
            np.asarray(u, dtype=np.float64), r_grid, dpsi, self._rotate)
        return fi
