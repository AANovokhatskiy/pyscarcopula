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
def _joe_h_inverse_newton(u, v, r):
    """Bracketed Newton-Raphson inversion of Joe h-function.

    Finds t such that h(t, v, r) = u using Newton's method
    with numerical derivative and bisection fallback.
    """
    n = len(u)
    out = np.empty(n)
    eps = 1e-10

    for i in range(n):
        ui = min(max(u[i], eps), 1.0 - eps)
        vi = min(max(v[i], eps), 1.0 - eps)
        ri = r[i] if r.shape[0] > 1 else r[0]

        if ri < 1.0 + 1e-8:
            out[i] = ui
            continue

        q1 = (1.0 - vi) ** ri
        lo = eps
        hi = 1.0 - eps
        t = ui

        for _ in range(40):
            t = min(max(t, lo), hi)
            q0 = (1.0 - t) ** ri
            x3 = q0 - 1.0
            inner = q0 - x3 * q1
            if inner < 1e-300:
                t = 0.5 * (lo + hi)
                continue

            h_val = -(x3 * inner ** (1.0 / ri - 1.0) * q1 / (1.0 - vi))

            err = h_val - ui
            if abs(err) < 1e-10:
                break

            # Update bracket
            if err > 0:
                hi = t
            else:
                lo = t

            # Numerical derivative via forward FD
            dt_fd = max(t * 1e-7, 1e-12)
            t_p = min(t + dt_fd, 1.0 - eps)
            q0_p = (1.0 - t_p) ** ri
            x3_p = q0_p - 1.0
            inner_p = q0_p - x3_p * q1
            if inner_p < 1e-300:
                t = 0.5 * (lo + hi)
                continue
            h_p = -(x3_p * inner_p ** (1.0 / ri - 1.0) * q1 / (1.0 - vi))
            dh_dt = (h_p - h_val) / (t_p - t)

            if abs(dh_dt) < 1e-300:
                t = 0.5 * (lo + hi)
            else:
                t_new = t - err / dh_dt
                if t_new > lo and t_new < hi:
                    t = t_new
                else:
                    t = 0.5 * (lo + hi)

        out[i] = min(max(t, eps), 1.0 - eps)
    return out


@njit(cache=True)
def _joe_V(n, r):
    """Sample V for Joe copula (Sibuya distribution)."""
    out = np.empty(n)
    
    for k in range(n):
        u = np.random.uniform(0, 1)
        i = 1
        p0 = 1.0 / r[k]
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


@njit(cache=True)
def _joe_dtransform(x):
    n = len(x)
    out = np.empty(n)
    for i in range(n):
        th = np.tanh(x[i])
        out[i] = th + x[i] * (1.0 - th * th)
    return out



@njit(cache=True)
def _joe_softplus_transform(x):
    n = len(x)
    out = np.empty(n)
    for i in range(n):
        if x[i] > 20.0:
            out[i] = x[i] + 1.0001
        elif x[i] < -20.0:
            out[i] = 1.0001
        else:
            out[i] = np.log1p(np.exp(x[i])) + 1.0001
    return out


@njit(cache=True)
def _joe_softplus_dtransform(x):
    n = len(x)
    out = np.empty(n)
    for i in range(n):
        if x[i] > 20.0:
            out[i] = 1.0
        elif x[i] < -20.0:
            out[i] = np.exp(x[i])
        else:
            out[i] = 1.0 / (1.0 + np.exp(-x[i]))
    return out


@njit(cache=True)
def _joe_softplus_inv_transform(r):
    n = len(r)
    out = np.empty(n)
    for i in range(n):
        y = r[i] - 1.0001
        if y > 20.0:
            out[i] = y
        elif y < 1e-10:
            out[i] = -20.0
        else:
            out[i] = np.log(np.exp(y) - 1.0)
    return out


@njit(cache=True)
def _joe_dlogc_dr(u1, u2, r):
    """Analytical d(log c)/dr for Joe copula."""
    n = len(u1)
    out = np.empty(n)
    for i in range(n):
        eps = 1e-300
        v1 = min(max(u1[i], eps), 1.0 - eps)
        v2 = min(max(u2[i], eps), 1.0 - eps)
        ri = r[i] if r.shape[0] > 1 else r[0]

        q1 = max(1.0 - v1, eps)
        q2 = max(1.0 - v2, eps)
        log_q1 = np.log(q1)
        log_q2 = np.log(q2)

        q1r = q1 ** ri
        q2r = q2 ** ri
        B = q1r + q2r - q1r * q2r
        if B < eps:
            B = eps

        dB = q1r * log_q1 * (1.0 - q2r) + q2r * log_q2 * (1.0 - q1r)

        term1 = log_q1 + log_q2
        term2 = (1.0 + dB) / (ri - 1.0 + B)
        term3 = -np.log(B) / (ri * ri) - (2.0 - 1.0 / ri) * dB / B

        out[i] = term1 + term2 + term3
    return out


@njit(cache=True)
def _joe_pdf_and_grad_batch(u_all, r_grid, dpsi, rotation):
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
        q1 = max(1.0 - v1, eps)
        q2 = max(1.0 - v2, eps)
        log_q1 = np.log(q1)
        log_q2 = np.log(q2)

        for j in range(K):
            ri = r_grid[j]

            q1r = q1 ** ri
            q2r = q2 ** ri

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

            log_c = ((ri - 1.0) * (log_q1 + log_q2)
                     + log_rp
                     - (2.0 - 1.0 / ri) * log_B)
            c_val = np.exp(log_c)
            fi[t, j] = c_val

            # d(log c)/dr
            B2 = q1r + q2r - q1r * q2r
            if B2 < eps:
                B2 = eps
            dB = q1r * log_q1 * (1.0 - q2r) + q2r * log_q2 * (1.0 - q1r)

            dlogc = (log_q1 + log_q2
                     + (1.0 + dB) / (ri - 1.0 + B2)
                     - np.log(B2) / (ri * ri) - (2.0 - 1.0 / ri) * dB / B2)
            dfi[t, j] = c_val * dlogc * dpsi[j]

    return fi, dfi


class JoeCopula(BivariateCopula):

    def __init__(self, rotate: int = 0, transform_type: str = 'xtanh'):
        super().__init__(rotate)
        self._name = "Joe copula"
        if transform_type not in ('xtanh', 'softplus'):
            raise ValueError(f"transform_type must be 'xtanh' or 'softplus', got '{transform_type}'")
        self._transform_type = transform_type
        self._bounds = [(1.0001, np.inf)]

    def transform(self, x):
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        if self._transform_type == 'softplus':
            return _joe_softplus_transform(x)
        return _joe_transform(x)

    def dtransform(self, x):
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        if self._transform_type == 'softplus':
            return _joe_softplus_dtransform(x)
        return _joe_dtransform(x)

    @staticmethod
    def inv_transform(r):
        return r - 1.0

    def pdf_unrotated(self, u1, u2, r):
        return _joe_pdf(*_broadcast(u1, u2, r))

    def log_pdf_unrotated(self, u1, u2, r):
        return _joe_log_pdf(*_broadcast(u1, u2, r))

    def dlog_pdf_dr_unrotated(self, u1, u2, r):
        return _joe_dlogc_dr(*_broadcast(u1, u2, r))

    @staticmethod
    def psi(t, r):
        return 1.0 - (1.0 - np.exp(-t)) ** (1.0 / r)

    def V(self, n, r):
        _r_input = np.asarray(r, dtype=np.float64)

        if _r_input.ndim == 0:
            _r = np.full(n, _r_input.item())
        else:
            _r = _r_input
        
        return _joe_V(n, _r)

    def h_unrotated(self, u, v, r):
        return _joe_h(*_broadcast(u, v, r))

    def h_inverse_unrotated(self, u, v, r):
        return _joe_h_inverse_newton(*_broadcast(u, v, r))

    def pdf_and_grad_on_grid_batch(self, u, x_grid):
        x = np.asarray(x_grid, dtype=np.float64)
        r_grid = self.transform(x)
        dpsi = self.dtransform(x)
        return _joe_pdf_and_grad_batch(
            np.asarray(u, dtype=np.float64), r_grid, dpsi, self._rotate)

    def copula_grid_batch(self, u, x_grid):
        x = np.asarray(x_grid, dtype=np.float64)
        r_grid = self.transform(x)
        dpsi = self.dtransform(x)
        fi, _ = _joe_pdf_and_grad_batch(
            np.asarray(u, dtype=np.float64), r_grid, dpsi, self._rotate)
        return fi