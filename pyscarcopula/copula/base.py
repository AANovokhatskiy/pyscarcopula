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
from scipy.optimize import minimize
from typing import Literal

from pyscarcopula.utils import pobs

METHODS = ('MLE', 'SCAR-P-OU', 'SCAR-M-OU', 'SCAR-TM-OU')


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

    def __init__(self, rotate: int = 0):
        if rotate not in (0, 90, 180, 270):
            raise ValueError(f"rotate must be 0/90/180/270, got {rotate}")
        self._rotate = rotate
        self._name = "BivariateCopula"
        self._bounds = [(-np.inf, np.inf)]
        self.fit_result = None
        self._latent = None  # OULatentProcess, created on demand

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
        return list(METHODS)

    # ── transform ──────────────────────────────────────────────────
    @staticmethod
    def transform(x):
        """R -> copula parameter domain."""
        return x

    @staticmethod
    def inv_transform(r):
        """Copula parameter domain -> R."""
        return r

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
        V_data = np.clip(self.V(n, _r), 1e-20, 1e+20)

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

    # ══════════════════════════════════════════════════════════════
    # Negative log-likelihood evaluation
    # ══════════════════════════════════════════════════════════════

    def mlog_likelihood(self, alpha, u, 
                        method: Literal['mle', 'scar-p-ou', 'scar-m-ou', 
                                        'scar-tm-ou'],
                        n_tr=500, M_iterations=3, seed=None,
                        dwt=None, stationary=True, K=300,
                        grid_range=5.0):
        """
        Compute minus log-likelihood for given parameters.

        Parameters
        ----------
        alpha : array
            MLE: scalar or (1,) — copula parameter (untransformed).
            SCAR: (3,) — [theta, mu, nu].
        u : array (T, 2) — pseudo-observations
        method : str — 'mle', 'scar-p-ou', 'scar-m-ou', 'scar-tm-ou'
        n_tr : int — MC trajectories (SCAR-P/M)
        M_iterations : int — EIS iterations (SCAR-M)
        seed : int or None
        dwt : (T, n_tr) or None
        stationary : bool
        K : int — grid size (SCAR-TM)
        grid_range : float

        Returns
        -------
        float : minus log-likelihood
        """
        from pyscarcopula.latent.ou_process import (
            _p_sampler_loglik, _m_sampler_loglik,
            _eis_find_auxiliary, _tm_loglik, OULatentProcess
        )

        u = np.asarray(u, dtype=np.float64)
        alpha = np.asarray(alpha, dtype=np.float64)
        method_up = method.upper()

        if method_up not in METHODS:
            raise ValueError(
                f"Unknown method '{method}'. Available: {list(METHODS)}")

        if method_up == 'MLE':
            r = self.transform(alpha[0]) if alpha.ndim > 0 else self.transform(alpha)
            return -self.log_likelihood(u, r)

        theta, mu, nu = alpha[0], alpha[1], alpha[2]
        T_len = len(u)

        if method_up == 'SCAR-TM-OU':
            return _tm_loglik(theta, mu, nu, u, self, K, grid_range)

        # MC methods need dwt
        if dwt is None:
            _seed = seed if seed is not None else np.random.randint(1, 1000000)
            dwt = OULatentProcess.calculate_dwt(T_len, n_tr, _seed)

        if method_up == 'SCAR-P-OU':
            return _p_sampler_loglik(theta, mu, nu, u, dwt,
                                    self, stationary)

        elif method_up == 'SCAR-M-OU':
            a1t, a2t = _eis_find_auxiliary(alpha, u, M_iterations,
                                           dwt, self, stationary)
            return _m_sampler_loglik(theta, mu, nu, u, dwt,
                                    a1t, a2t, self, stationary)

    # ══════════════════════════════════════════════════════════════
    # Unified fit — single entry point for all methods
    # ══════════════════════════════════════════════════════════════

    def fit(self, data, method: Literal['mle', 'scar-p-ou', 'scar-m-ou', 
                                        'scar-tm-ou'] = 'scar-tm-ou', 
            to_pobs=False, **kwargs):
        """
        Fit the copula to data.

        Parameters
        ----------
        data : array (T, 2)
            Log-returns or pseudo-observations.
        method : str
            'mle', 'scar-p-ou', 'scar-m-ou', 'scar-tm-ou'
        to_pobs : bool
            If True, transform data to pseudo-observations first.
        **kwargs
            Forwarded to OULatentProcess.fit() for SCAR methods:
            alpha0, tol, n_tr, M_iterations, seed, dwt,
            stationary, K, grid_range, verbose

        Returns
        -------
        scipy.optimize.OptimizeResult with extra fields:
            .method, .name, .log_likelihood
            .copula_param (MLE) or .alpha (SCAR)
        """
        u = np.asarray(data, dtype=np.float64)
        if to_pobs:
            u = pobs(u)

        method_up = method.upper()
        if method_up not in METHODS:
            raise ValueError(
                f"Unknown method '{method}'. Available: {list(METHODS)}")

        if method_up == 'MLE':
            return self._fit_mle(u)
        else:
            return self._fit_scar(u, method_up, **kwargs)

    def _fit_mle(self, u):
        """Fit constant copula parameter via MLE."""
        x0 = np.array([self.transform(1.5)])
        result = minimize(
            lambda x: -self.log_likelihood(u, x[0]),
            x0,
            method='L-BFGS-B',
            bounds=self.bounds,
            options={'gtol': 1e-4, 'eps': 1e-5},
        )
        result.copula_param = result.x[0]
        result.log_likelihood = -result.fun
        result.method = 'MLE'
        result.name = self.name
        self.fit_result = result
        self._latent = None
        return result

    def _fit_scar(self, u, method, **kwargs):
        """Delegate SCAR fitting to OULatentProcess."""
        from pyscarcopula.latent.ou_process import OULatentProcess

        ou = OULatentProcess(self)
        result = ou.fit(u, method=method, **kwargs)
        self.fit_result = result
        self._latent = ou
        return result

    # ══════════════════════════════════════════════════════════════
    # Forwarding methods (require prior SCAR fit)
    # ══════════════════════════════════════════════════════════════

    def smoothed_params(self, u, **kwargs):
        """
        Smoothed copula parameter E[Psi(x_k) | u_{1:k-1}] for all k.
        Requires prior SCAR fit (uses transfer matrix forward pass).
        """
        self._require_latent('smoothed_params')
        return self._latent.smoothed_params(u, **kwargs)

    def xT_distribution(self, u, **kwargs):
        """
        Distribution of x_T on grid for VaR/CVaR scenarios.
        Returns (z_grid, prob_weights).
        Requires prior SCAR fit.
        """
        self._require_latent('xT_distribution')
        return self._latent.xT_distribution(u, **kwargs)

    def predict(self, n):
        """
        Generate pseudo-observations for prediction.
        Uses fitted parameters (MLE: constant param, SCAR: sample from x_T).
        """
        if self.fit_result is None:
            raise ValueError("Fit the model first")

        if self.fit_result.method == 'MLE':
            r = self.fit_result.copula_param
        else:
            self._require_latent('predict')
            alpha = self.fit_result.alpha
            x_T = self._latent.final_state_sample(
                alpha, n, method=self.fit_result.method)
            r = self.transform(x_T)

        return self.sample(n, r)

    def _require_latent(self, method_name):
        if self._latent is None:
            raise ValueError(
                f"{method_name}() requires a prior SCAR fit. "
                f"Call copula.fit(data, method='scar-tm-ou') first."
            )
