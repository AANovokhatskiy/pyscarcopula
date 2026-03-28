"""
Base class for bivariate Archimedean copulas.
dim=2 only.

Subclass contract — must override:
    pdf_unrotated(u1, u2, r)    — vectorized numpy
    log_pdf_unrotated(u1, u2, r)
    transform(x), inv_transform(r)
    psi(t, r), V(n, r)

Optional overrides:
    h_unrotated(u, v, r), h_inverse_unrotated(u, v, r)
    sample(n, r, rng)
"""

import numpy as np

from pyscarcopula._utils import pobs


# ── broadcast helper ──────────────────────────────────────────────
def _broadcast(u1, u2, r):
    """Ensure all inputs are 1D float64 arrays of the same length."""
    u1a = np.atleast_1d(np.asarray(u1, dtype=np.float64)).ravel()
    u2a = np.atleast_1d(np.asarray(u2, dtype=np.float64)).ravel()
    ra = np.atleast_1d(np.asarray(r, dtype=np.float64)).ravel()
    n = max(len(u1a), len(u2a), len(ra))
    if len(u1a) == 1 and n > 1:
        u1a = np.full(n, u1a[0])
    if len(u2a) == 1 and n > 1:
        u2a = np.full(n, u2a[0])
    if len(ra) == 1 and n > 1:
        ra = np.full(n, ra[0])
    return u1a, u2a, ra


class BivariateCopula:
    """
    Base class for bivariate copulas (dim=2).

    Provides a unified interface for fitting, evaluation, sampling, and
    diagnostics. Subclasses must override the core density methods;
    optional overrides enable analytical gradients and batch evaluation.

    Subclass contract — must override:
        pdf_unrotated, log_pdf_unrotated, transform, inv_transform, psi, V

    Recommended overrides (enable analytical gradient in TM):
        dtransform(x)                    — d Psi / dx
        dlog_pdf_dr_unrotated(u1, u2, r) — d(log c) / dr

    Optional overrides (fused numba batch kernels for speed):
        pdf_and_grad_on_grid_batch(u, x_grid)
        copula_grid_batch(u, x_grid)

    Estimation methods (via .fit()):
        'mle'        — constant parameter (1 param)
        'scar-tm-ou' — transfer matrix (3 params: theta, mu, nu)
        'gas'        — GAS score-driven (3 params: omega, alpha, beta)
        'scar-p-ou'  — MC p-sampler, 'scar-m-ou' — MC m-sampler with EIS

    Parameters
    ----------
    rotate : int
        Copula rotation: 0, 90, 180, or 270 degrees.
    """

    def __init__(self, rotate: int = 0):
        if rotate not in (0, 90, 180, 270):
            raise ValueError(f"rotate must be 0/90/180/270, got {rotate}")
        self._rotate = rotate
        self._name = "BivariateCopula"
        self._bounds = [(-np.inf, np.inf)]
        self.fit_result = None

    @property
    def name(self):
        return self._name

    @property
    def rotate(self):
        return self._rotate

    @property
    def bounds(self):
        return self._bounds

    @staticmethod
    def list_of_methods():
        from pyscarcopula.strategy._base import list_methods
        return list_methods()

    # ── transform ──────────────────────────────────────────────────
    @staticmethod
    def transform(x):
        """R -> copula parameter domain."""
        return x

    @staticmethod
    def inv_transform(r):
        """Copula parameter domain -> R."""
        return r

    @staticmethod
    def dtransform(x):
        """d Psi(x) / dx.  Override per copula."""
        return np.ones_like(np.asarray(x, dtype=np.float64))

    # ── PDF / log-PDF ─────────────────────────────────────────────
    def pdf_unrotated(self, u1, u2, r):
        raise NotImplementedError

    def log_pdf_unrotated(self, u1, u2, r):
        return np.log(np.maximum(self.pdf_unrotated(u1, u2, r), 1e-300))

    def pdf(self, u1, u2, r):
        v1, v2 = self._apply_rotation(u1, u2)
        return self.pdf_unrotated(v1, v2, r)

    def log_pdf(self, u1, u2, r):
        v1, v2 = self._apply_rotation(u1, u2)
        return self.log_pdf_unrotated(v1, v2, r)

    def dlog_pdf_dr_unrotated(self, u1, u2, r):
        """d(log c)/dr — analytical derivative w.r.t. copula parameter.

        Default: central finite differences. Override for analytical version.
        """
        u1a = np.atleast_1d(np.asarray(u1, dtype=np.float64)).ravel()
        u2a = np.atleast_1d(np.asarray(u2, dtype=np.float64)).ravel()
        ra = np.atleast_1d(np.asarray(r, dtype=np.float64)).ravel()
        eps = 1e-6
        lp = self.log_pdf_unrotated(u1a, u2a, ra + eps)
        lm = self.log_pdf_unrotated(u1a, u2a, ra - eps)
        return (lp - lm) / (2.0 * eps)

    def dlog_pdf_dr(self, u1, u2, r):
        """d(log c)/dr with rotation applied."""
        v1, v2 = self._apply_rotation(u1, u2)
        return self.dlog_pdf_dr_unrotated(v1, v2, r)

    def _apply_rotation(self, u1, u2):
        rot = self._rotate
        if rot == 0:
            return u1, u2
        elif rot == 90:
            return 1.0 - u1, u2
        elif rot == 180:
            return 1.0 - u1, 1.0 - u2
        else:
            return u1, 1.0 - u2

    # ── sampling ──────────────────────────────────────────────────
    def psi(self, t, r):
        """Inverse generator (Laplace-Stieltjes)."""
        return np.exp(-t)

    def V(self, n, r):
        """Sample from F = LS^{-1}(psi). Override per copula."""
        return np.ones(n)

    def sample(self, n, r, rng=None):
        """
        Marshall-Olkin sampling.
        r: scalar or array (n,).
        Returns (n, 2).
        """
        if rng is None:
            rng = np.random.default_rng()

        _r = np.atleast_1d(np.asarray(r, dtype=np.float64))
        if _r.size == 1:
            _r = np.full(n, _r[0])

        x = rng.uniform(0, 1, size=(n, 2))
        V_data = np.clip(self.V(n, _r), 1e-50, None)

        u = np.empty((n, 2))
        u[:, 0] = self.psi(-np.log(x[:, 0]) / V_data, _r)
        u[:, 1] = self.psi(-np.log(x[:, 1]) / V_data, _r)

        rot = self._rotate
        if rot == 90:
            u[:, 0] = 1.0 - u[:, 0]
        elif rot == 180:
            u[:, 0] = 1.0 - u[:, 0]
            u[:, 1] = 1.0 - u[:, 1]
        elif rot == 270:
            u[:, 1] = 1.0 - u[:, 1]

        return u

    # ── h-functions ───────────────────────────────────────────────
    def h_unrotated(self, u, v, r):
        raise NotImplementedError

    def h_inverse_unrotated(self, u, v, r):
        """Default: numerical bisection inversion of h."""
        return self._h_inverse_bisection(u, v, r)

    def _h_inverse_bisection(self, u, v, r, tol=1e-10, maxiter=60):
        """
        Numerical inversion: find t such that h(t, v, r) = u.
        Uses bisection on [eps, 1-eps].
        """
        u = np.atleast_1d(np.asarray(u, dtype=np.float64))
        v = np.atleast_1d(np.asarray(v, dtype=np.float64))
        r = np.atleast_1d(np.asarray(r, dtype=np.float64))
        n = max(len(u), len(v), len(r))
        if len(u) == 1 and n > 1: u = np.full(n, u[0])
        if len(v) == 1 and n > 1: v = np.full(n, v[0])
        if len(r) == 1 and n > 1: r = np.full(n, r[0])

        eps = 1e-10
        lo = np.full(n, eps)
        hi = np.full(n, 1.0 - eps)

        for _ in range(maxiter):
            mid = 0.5 * (lo + hi)
            h_mid = self.h_unrotated(mid, v, r)
            mask = h_mid < u
            lo = np.where(mask, mid, lo)
            hi = np.where(mask, hi, mid)
            if np.max(hi - lo) < tol:
                break

        return 0.5 * (lo + hi)

    def h(self, u, v, r):
        rot = self._rotate
        if rot == 0:
            return self.h_unrotated(u, v, r)
        elif rot == 90:
            return 1.0 - self.h_unrotated(1.0 - u, v, r)
        elif rot == 180:
            return 1.0 - self.h_unrotated(1.0 - u, 1.0 - v, r)
        else:
            return self.h_unrotated(u, 1.0 - v, r)

    def h_inverse(self, u, v, r):
        rot = self._rotate
        if rot == 0:
            return self.h_inverse_unrotated(u, v, r)
        elif rot == 90:
            return 1.0 - self.h_inverse_unrotated(1.0 - u, v, r)
        elif rot == 180:
            return 1.0 - self.h_inverse_unrotated(1.0 - u, 1.0 - v, r)
        else:
            return self.h_inverse_unrotated(u, 1.0 - v, r)
        
    # ── log-likelihood ────────────────────────────────────────────
    def log_likelihood(self, u, r):
        """u: (T, 2), r: scalar or (T,)."""
        return np.sum(self.log_pdf(u[:, 0], u[:, 1], r))

    # ── evaluate pdf on a grid of latent states (for transfer matrix) ──
    def pdf_on_grid(self, u_row, z_grid):
        """
        c(u_row; Psi(z_j)) for each z_j in z_grid.
        u_row: (2,), z_grid: (K,). Returns (K,).
        """
        u1 = np.full(len(z_grid), u_row[0])
        u2 = np.full(len(z_grid), u_row[1])
        return self.pdf(u1, u2, self.transform(z_grid))

    def pdf_and_grad_on_grid(self, u_row, z_grid):
        """
        Compute fi(z) and dfi/dz on the grid analytically.

        Uses chain rule: dfi/dz = fi * d(log c)/dr * Psi'(z).

        u_row: (2,), z_grid: (K,).
        Returns (fi, dfi_dz) each of shape (K,).
        """
        z = np.asarray(z_grid, dtype=np.float64)
        r = self.transform(z)
        u1 = np.full(len(z), u_row[0])
        u2 = np.full(len(z), u_row[1])

        v1, v2 = self._apply_rotation(u1, u2)
        fi = self.pdf_unrotated(v1, v2, r)
        dlogc = self.dlog_pdf_dr_unrotated(v1, v2, r)
        dpsi = self.dtransform(z)

        dfi_dz = fi * dlogc * dpsi
        return fi, dfi_dz

    def pdf_and_grad_on_grid_batch(self, u, x_grid):
        """
        Batch version: compute fi and dfi_dx for all T observations.

        u : (T, 2), x_grid : (K,).
        Returns (fi, dfi_dx) each of shape (T, K).

        Default: Python loop over pdf_and_grad_on_grid.
        Override in subclasses with a fused numba kernel for speed.
        """
        n = len(u)
        K = len(x_grid)
        fi = np.empty((n, K))
        dfi = np.empty((n, K))
        for i in range(n):
            fi[i], dfi[i] = self.pdf_and_grad_on_grid(u[i], x_grid)
        return fi, dfi

    def copula_grid_batch(self, u, x_grid):
        """
        Batch version of pdf_on_grid (value only, no gradient).

        u : (T, 2), x_grid : (K,).
        Returns fi of shape (T, K).

        Default: Python loop. Override for speed.
        """
        n = len(u)
        K = len(x_grid)
        fi = np.empty((n, K))
        for i in range(n):
            fi[i] = self.pdf_on_grid(u[i], x_grid)
        return fi

    # ══════════════════════════════════════════════════════════════
    # Fit — delegates to api.fit() / strategy
    # ══════════════════════════════════════════════════════════════

    def fit(self, data, method='scar-tm-ou', to_pobs=False, **kwargs):
        """Fit the copula. Delegates to pyscarcopula.api.fit().

        Convenience method: equivalent to
            from pyscarcopula.api import fit
            result = fit(copula, data, method=method, ...)

        Returns
        -------
        FitResult (immutable dataclass)
        """
        from pyscarcopula.api import fit as _api_fit

        u = np.asarray(data, dtype=np.float64)
        if to_pobs:
            u = pobs(u)

        result = _api_fit(self, u, method=method, **kwargs)
        self.fit_result = result
        return result