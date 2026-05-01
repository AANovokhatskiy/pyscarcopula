import numpy as np
from numba import njit

from pyscarcopula.copula.base import BivariateCopula, _broadcast


# ══════════════════════════════════════════════════════════════════
# Numba kernels
# ══════════════════════════════════════════════════════════════════

@njit(cache=True)
def _gumbel_log_pdf(u1, u2, r):
    """Log-density of bivariate Gumbel copula (unrotated). Vectorized over arrays."""
    n = len(u1)
    out = np.empty(n)
    for i in range(n):
        eps = 1e-300
        v1 = min(max(u1[i], eps), 1.0 - eps)
        v2 = min(max(u2[i], eps), 1.0 - eps)

        log_v1 = np.log(v1)
        log_v2 = np.log(v2)
        log_p1 = np.log(-log_v1)
        log_p2 = np.log(-log_v2)

        log_max = max(log_p1, log_p2)
        log_min = min(log_p1, log_p2)

        ri = r[i] if r.shape[0] > 1 else r[0]

        delta = ri * (log_min - log_max)
        S = ri * log_max + np.log1p(np.exp(delta))
        log_A = S / ri
        A = np.exp(log_A)

        out[i] = ((ri - 1.0) * (log_p1 + log_p2)
                  + (1.0 / ri - 2.0) * S
                  + np.log(ri - 1.0 + A)
                  - A
                  - log_v1 - log_v2)
    return out


@njit(cache=True)
def _gumbel_pdf(u1, u2, r):
    return np.exp(_gumbel_log_pdf(u1, u2, r))


@njit(cache=True)
def _gumbel_h(u0, u1, r):
    """h-function: C(u0 | u1; r) for unrotated Gumbel."""
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
            log_v1 = np.log(v1)
            t1 = (-log_v1) ** ri
            t2 = (-np.log(v0)) ** ri
            S = t1 + t2
            if S < 1e-300:
                out[i] = v0
            else:
                A = S ** (1.0 / ri)
                val = t1 / S * A * np.exp(-A) / (-log_v1 * v1)
                out[i] = min(max(val, eps), 1.0 - eps)
    return out


@njit(cache=True)
def _gumbel_h_pair(u0, u1, r):
    """Both Gumbel h-functions for one pair, sharing common terms."""
    n = len(u0)
    out01 = np.empty(n)
    out10 = np.empty(n)
    eps = 1e-6
    for i in range(n):
        v0 = min(max(u0[i], eps), 1.0 - eps)
        v1 = min(max(u1[i], eps), 1.0 - eps)
        ri = r[i] if r.shape[0] > 1 else r[0]

        if ri < 1.0 + 1e-8:
            out01[i] = v0
            out10[i] = v1
        else:
            y0 = -np.log(v0)
            y1 = -np.log(v1)
            t0 = y0 ** ri
            t1 = y1 ** ri
            S = t0 + t1
            if S < 1e-300:
                out01[i] = v0
                out10[i] = v1
            else:
                A = S ** (1.0 / ri)
                common = A * np.exp(-A) / S
                val01 = t1 * common / (y1 * v1)
                val10 = t0 * common / (y0 * v0)
                out01[i] = min(max(val01, eps), 1.0 - eps)
                out10[i] = min(max(val10, eps), 1.0 - eps)
    return out01, out10


@njit(cache=True)
def _generate_levy_stable(alpha, beta, loc = 0, scale = 1, size = 1):
    '''
    Weron, R. (1996). On the Chambers-Mallows-Stuck method for simulating skewed stable random variables
    Borak et. al. (2008), Stable Distributions
    '''

    V = np.random.uniform(-np.pi/2, np.pi/2, size = size)
    u = np.random.uniform(0, 1, size = size)
    W = -np.log(1 - u)

    indicator0 = (alpha != 1)
    indicator1 = np.invert(indicator0)

    B = np.arctan(beta * np.tan(np.pi/2 * alpha)) / alpha
    S = (1 + beta**2 * np.tan(np.pi/2 * alpha)**2)**(1 / (2 * alpha))

    X0 = S * np.sin(alpha * (V + B)) / np.cos(V)**(1/alpha) * (np.cos(V - alpha * (V + B)) / W)**((1 - alpha) / alpha)
    X1 = 2 / np.pi * ((np.pi/2 + beta * V) * np.tan(V) - beta * np.log(np.pi/2 * W * np.cos(V) / (np.pi / 2 + beta * V)))

    X = X0 * indicator0 + X1 * indicator1

    Y0 = scale * X + loc
    Y1 = scale * X + 2 / np.pi * beta * scale * np.log(scale) + loc

    Y = Y0 * indicator0 + Y1 * indicator1
        
    return Y


@njit(cache=True)
def _generate_levy_stable_from_uniforms(alpha, beta, loc, scale, V, u):
    W = -np.log(1 - u)

    indicator0 = (alpha != 1)
    indicator1 = np.invert(indicator0)

    B = np.arctan(beta * np.tan(np.pi/2 * alpha)) / alpha
    S = (1 + beta**2 * np.tan(np.pi/2 * alpha)**2)**(1 / (2 * alpha))

    X0 = S * np.sin(alpha * (V + B)) / np.cos(V)**(1/alpha) * (np.cos(V - alpha * (V + B)) / W)**((1 - alpha) / alpha)
    X1 = 2 / np.pi * ((np.pi/2 + beta * V) * np.tan(V) - beta * np.log(np.pi/2 * W * np.cos(V) / (np.pi / 2 + beta * V)))

    X = X0 * indicator0 + X1 * indicator1

    Y0 = scale * X + loc
    Y1 = scale * X + 2 / np.pi * beta * scale * np.log(scale) + loc

    Y = Y0 * indicator0 + Y1 * indicator1
        
    return Y

@njit(cache=True)
def _gumbel_transform(x):
    return x * np.tanh(x) + 1.0001


@njit(cache=True)
def _gumbel_dtransform(x):
    n = len(x)
    out = np.empty(n)
    for i in range(n):
        th = np.tanh(x[i])
        out[i] = th + x[i] * (1.0 - th * th)
    return out


@njit(cache=True)
def _gumbel_softplus_transform(x):
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
def _gumbel_softplus_dtransform(x):
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
def _gumbel_softplus_inv_transform(r):
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
def _gumbel_dlogc_dr(u1, u2, r):
    """Analytical d(log c)/dr for Gumbel copula."""
    n = len(u1)
    out = np.empty(n)
    for i in range(n):
        eps = 1e-300
        v1 = min(max(u1[i], eps), 1.0 - eps)
        v2 = min(max(u2[i], eps), 1.0 - eps)
        ri = r[i] if r.shape[0] > 1 else r[0]

        log_v1 = np.log(v1)
        log_v2 = np.log(v2)
        log_p1 = np.log(max(-log_v1, eps))
        log_p2 = np.log(max(-log_v2, eps))

        log_max = max(log_p1, log_p2)
        log_min = min(log_p1, log_p2)
        delta = ri * (log_min - log_max)

        S_log = ri * log_max + np.log1p(np.exp(delta))

        ed = np.exp(delta)
        sig = ed / (1.0 + ed)
        dS_dr = log_max + (log_min - log_max) * sig

        log_A = S_log / ri
        A = np.exp(log_A)
        dlogA_dr = (dS_dr * ri - S_log) / (ri * ri)
        dA_dr = A * dlogA_dr

        term1 = log_p1 + log_p2
        term2 = -S_log / (ri * ri) + (1.0 / ri - 2.0) * dS_dr
        term3 = (1.0 + dA_dr) / (ri - 1.0 + A)
        term4 = -dA_dr

        out[i] = term1 + term2 + term3 + term4
    return out


@njit(cache=True)
def _gumbel_pdf_and_grad_batch(u_all, r_grid, dpsi, rotation):
    """Fused batch: fi and dfi_dx for all T observations at once."""
    T = u_all.shape[0]
    K = len(r_grid)
    fi = np.empty((T, K))
    dfi = np.empty((T, K))

    for t in range(T):
        eps = 1e-300
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
        log_p1 = np.log(max(-log_v1, eps))
        log_p2 = np.log(max(-log_v2, eps))
        log_max = max(log_p1, log_p2)
        log_min = min(log_p1, log_p2)

        for j in range(K):
            ri = r_grid[j]
            delta = ri * (log_min - log_max)
            exp_delta = np.exp(delta)
            S = ri * log_max + np.log1p(exp_delta)
            log_A = S / ri
            A = np.exp(log_A)

            log_c = ((ri - 1.0) * (log_p1 + log_p2)
                     + (1.0 / ri - 2.0) * S
                     + np.log(ri - 1.0 + A)
                     - A - log_v1 - log_v2)
            c_val = np.exp(log_c)
            fi[t, j] = c_val

            sig = exp_delta / (1.0 + exp_delta)
            dS_dr = log_max + (log_min - log_max) * sig
            dlogA_dr = (dS_dr * ri - S) / (ri * ri)
            dA_dr = A * dlogA_dr

            dlogc = (log_p1 + log_p2
                     - S / (ri * ri) + (1.0 / ri - 2.0) * dS_dr
                     + (1.0 + dA_dr) / (ri - 1.0 + A)
                     - dA_dr)
            dfi[t, j] = c_val * dlogc * dpsi[j]

    return fi, dfi


@njit(cache=True)
def _gumbel_inv_transform(r):
    return r - 1.0


# ══════════════════════════════════════════════════════════════════
# Class
# ══════════════════════════════════════════════════════════════════

class GumbelCopula(BivariateCopula):

    def __init__(self, rotate: int = 0, transform_type: str = 'xtanh'):
        """
        Parameters
        ----------
        rotate : int — 0, 90, 180, 270
        transform_type : str — 'xtanh' (default) or 'softplus'
            'xtanh': Psi(x) = x*tanh(x) + 1.0001 (symmetric)
            'softplus': Psi(x) = log(1+exp(x)) + 1.0001 (asymmetric, floor at 1)
        """
        super().__init__(rotate)
        self._name = "Gumbel copula"
        self._bounds = [(1.0001, np.inf)]
        if transform_type not in ('xtanh', 'softplus'):
            raise ValueError(f"transform_type must be 'xtanh' or 'softplus', got '{transform_type}'")
        self._transform_type = transform_type

    def transform(self, x):
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        if self._transform_type == 'softplus':
            return _gumbel_softplus_transform(x)
        return _gumbel_transform(x)

    def dtransform(self, x):
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        if self._transform_type == 'softplus':
            return _gumbel_softplus_dtransform(x)
        return _gumbel_dtransform(x)

    def inv_transform(self, r):
        r = np.atleast_1d(np.asarray(r, dtype=np.float64))
        if self._transform_type == 'softplus':
            return _gumbel_softplus_inv_transform(r)
        return _gumbel_inv_transform(r)

    def pdf_unrotated(self, u1, u2, r):
        u1a, u2a, ra = _broadcast(u1, u2, r)
        return _gumbel_pdf(u1a, u2a, ra)

    def log_pdf_unrotated(self, u1, u2, r):
        u1a, u2a, ra = _broadcast(u1, u2, r)
        return _gumbel_log_pdf(u1a, u2a, ra)

    def dlog_pdf_dr_unrotated(self, u1, u2, r):
        u1a, u2a, ra = _broadcast(u1, u2, r)
        return _gumbel_dlogc_dr(u1a, u2a, ra)

    @staticmethod
    def psi(t, r):
        return np.exp(-t ** (1.0 / r))

    def V(self, n, r, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        _r = np.atleast_1d(np.asarray(r, dtype=np.float64))
        if _r.size == 1:
            _r = np.full(n, _r[0])
        uniforms_v = rng.uniform(-np.pi / 2.0, np.pi / 2.0, size=n)
        uniforms_u = rng.uniform(0.0, 1.0, size=n)
        alpha = 1.0 / _r
        scale = np.cos(np.pi / (2.0 * _r)) ** _r
        res = _generate_levy_stable_from_uniforms(
            alpha, 1.0, 0.0, scale, uniforms_v, uniforms_u)
        return np.maximum(res, 1e-300)
    
    def h_unrotated(self, u, v, r):
        ua, va, ra = _broadcast(u, v, r)
        return _gumbel_h(ua, va, ra)

    def h_pair(self, u, v, r):
        ua, va, ra = _broadcast(u, v, r)
        if self._rotate == 0:
            return _gumbel_h_pair(ua, va, ra)
        return self.h(ua, va, ra), self.h(va, ua, ra)

    def pdf_and_grad_on_grid_batch(self, u, x_grid):
        x = np.asarray(x_grid, dtype=np.float64)
        r_grid = self.transform(x)
        dpsi = self.dtransform(x)
        return _gumbel_pdf_and_grad_batch(
            np.asarray(u, dtype=np.float64), r_grid, dpsi, self._rotate)

    def copula_grid_batch(self, u, x_grid):
        x = np.asarray(x_grid, dtype=np.float64)
        r_grid = self.transform(x)
        dpsi = self.dtransform(x)
        fi, _ = _gumbel_pdf_and_grad_batch(
            np.asarray(u, dtype=np.float64), r_grid, dpsi, self._rotate)
        return fi

    # def h_inverse_unrotated(self, u, v, r):
    #     raise NotImplementedError("Gumbel h_inverse requires numerical inversion")

    def h_inverse_unrotated(self, u, v, r):
        ua, va, ra = _broadcast(u, v, r)
        return _gumbel_h_inverse_newton(ua, va, ra)


@njit(cache=True)
def _gumbel_h_inverse_newton(u, v, r):
    """Invert the Gumbel h-function.

    The direct inverse has no elementary closed form, but after setting
    ``A = ((-log(t))**r + (-log(v))**r)**(1/r)`` the equation is scalar and
    monotone:

        log(u) = (r - 1) * log(-log(v)) - log(v) + (1 - r) * log(A) - A

    Solving for ``A`` avoids finite-difference derivatives in the old
    bracketed Newton loop.
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

        y = -np.log(vi)
        target = np.log(ui) - ((ri - 1.0) * np.log(y) - np.log(vi))

        lo = y
        hi = max(y - np.log(ui) + ri, y + 1.0)

        # Ensure the upper bracket has log_h(A) <= log(ui).
        for _ in range(24):
            f_hi = (1.0 - ri) * np.log(hi) - hi - target
            if f_hi <= 0.0:
                break
            hi *= 2.0

        A = min(max(y - np.log(ui), lo), hi)
        for _ in range(32):
            f = (1.0 - ri) * np.log(A) - A - target
            if abs(f) < 1e-12:
                break
            if f > 0.0:
                lo = A
            else:
                hi = A

            fp = (1.0 - ri) / A - 1.0
            A_new = A - f / fp
            if A_new > lo and A_new < hi:
                A = A_new
            else:
                A = 0.5 * (lo + hi)

        z_pow = A ** ri - y ** ri
        if z_pow <= 0.0:
            out[i] = 1.0 - eps
        else:
            out[i] = min(max(np.exp(-(z_pow ** (1.0 / ri))), eps), 1.0 - eps)
    return out
