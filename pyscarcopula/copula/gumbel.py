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

    def V(self, n, r):
        res = _generate_levy_stable(alpha = 1/r, beta = 1, loc = 0,
                                    scale = np.cos(np.pi / (2 * r))**r, size = n)
        return np.maximum(res, 1e-300)
    
    def h_unrotated(self, u, v, r):
        ua, va, ra = _broadcast(u, v, r)
        return _gumbel_h(ua, va, ra)

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
    """Newton-Raphson inversion of Gumbel h-function.

    Finds t such that h(t, v, r) = u using Newton's method
    with numerical derivative. Converges in ~5-10 iterations
    (vs 60 for bisection).
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

        log_vi = np.log(vi)
        t1_base = (-log_vi) ** ri

        # Bracketed Newton: maintain [lo, hi] and use Newton when safe
        lo = eps
        hi = 1.0 - eps
        t = ui  # initial guess

        for _ in range(40):
            t = min(max(t, lo), hi)

            # Evaluate h(t, v; r)
            log_t = np.log(t)
            t2 = (-log_t) ** ri
            S = t1_base + t2
            if S < 1e-300:
                t = 0.5 * (lo + hi)
                continue
            A = S ** (1.0 / ri)
            h_val = t1_base / S * A * np.exp(-A) / (-log_vi * vi)

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
            log_tp = np.log(t_p)
            t2_p = (-log_tp) ** ri
            S_p = t1_base + t2_p
            if S_p < 1e-300:
                t = 0.5 * (lo + hi)
                continue
            A_p = S_p ** (1.0 / ri)
            h_p = t1_base / S_p * A_p * np.exp(-A_p) / (-log_vi * vi)
            dh_dt = (h_p - h_val) / (t_p - t)

            if abs(dh_dt) < 1e-300:
                t = 0.5 * (lo + hi)
            else:
                t_new = t - err / dh_dt
                # Accept Newton step only if it stays in bracket
                if t_new > lo and t_new < hi:
                    t = t_new
                else:
                    t = 0.5 * (lo + hi)

        out[i] = min(max(t, eps), 1.0 - eps)
    return out