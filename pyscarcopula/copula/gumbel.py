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
    return x * x + 1.00001


@njit(cache=True)
def _gumbel_inv_transform(r):
    return np.sqrt(max(r - 1.00001, 0.0))


# ══════════════════════════════════════════════════════════════════
# Class
# ══════════════════════════════════════════════════════════════════

class GumbelCopula(BivariateCopula):

    def __init__(self, rotate: int = 0):
        super().__init__(rotate)
        self._name = "Gumbel copula"
        self._bounds = [(1.0001, np.inf)]

    @staticmethod
    def transform(x):
        return _gumbel_transform(x)

    @staticmethod
    def inv_transform(r):
        return _gumbel_inv_transform(r)

    def pdf_unrotated(self, u1, u2, r):
        u1a, u2a, ra = _broadcast(u1, u2, r)
        return _gumbel_pdf(u1a, u2a, ra)

    def log_pdf_unrotated(self, u1, u2, r):
        u1a, u2a, ra = _broadcast(u1, u2, r)
        return _gumbel_log_pdf(u1a, u2a, ra)

    @staticmethod
    def psi(t, r):
        return np.exp(-t ** (1.0 / r))

    def V(self, n, r):
        # _r = np.atleast_1d(np.asarray(r, dtype=np.float64))
        # if _r.size == 1:
        #     return _generate_levy_stable(1.0 / _r[0], n)
        # # vector r — use first element (all equal in Marshall-Olkin with scalar param)
        # return _generate_levy_stable(1.0 / _r[0], n)
        res = _generate_levy_stable(alpha = 1/r, beta = 1, loc = 0, 
                                    scale = np.cos(np.pi / (2 * r))**r, size = n)
        return res
    
    def h_unrotated(self, u, v, r):
        ua, va, ra = _broadcast(u, v, r)
        return _gumbel_h(ua, va, ra)

    # def h_inverse_unrotated(self, u, v, r):
    #     raise NotImplementedError("Gumbel h_inverse requires numerical inversion")
