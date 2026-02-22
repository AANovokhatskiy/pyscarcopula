"""
Elliptical copulas: Gaussian and Student-t.

BivariateGaussianCopula — inherits BivariateCopula, works with vine.
    Parameter: rho in (-1, 1), no rotation.

GaussianCopula — d-dimensional, correlation matrix parametrization.
StudentCopula — d-dimensional, correlation + df.

Usage:
    from pyscarcopula.copula.elliptical import (
        BivariateGaussianCopula, GaussianCopula, StudentCopula)

    # Bivariate (works in vine)
    cop = BivariateGaussianCopula()
    cop.fit(u, method='mle')

    # Multivariate
    gcop = GaussianCopula()
    gcop.fit(data, to_pobs=True)
    samples = gcop.sample(10000)
"""

import numpy as np
from scipy.stats import norm, t as t_dist, multivariate_normal, multivariate_t, kendalltau
from scipy.optimize import minimize
from numba import njit

from pyscarcopula.copula.base import BivariateCopula
from pyscarcopula.utils import pobs


def _gauss_log_pdf_scipy(u1, u2, rho):
    """Bivariate Gaussian copula log-density (scipy-based, vectorized)."""
    eps = 1e-10
    v1 = np.clip(u1, eps, 1.0 - eps)
    v2 = np.clip(u2, eps, 1.0 - eps)
    x1 = norm.ppf(v1)
    x2 = norm.ppf(v2)
    r2 = rho ** 2
    return (-0.5 * np.log(1.0 - r2)
            - 0.5 * (r2 * (x1 ** 2 + x2 ** 2) - 2.0 * rho * x1 * x2) / (1.0 - r2))


def _gauss_pdf_scipy(u1, u2, rho):
    return np.exp(_gauss_log_pdf_scipy(u1, u2, rho))


def _gauss_h(u, v, rho):
    """h(u|v; rho) = Phi((Phi^{-1}(u) - rho*Phi^{-1}(v)) / sqrt(1-rho^2))"""
    eps = 1e-10
    u_c = np.clip(u, eps, 1.0 - eps)
    v_c = np.clip(v, eps, 1.0 - eps)
    return norm.cdf((norm.ppf(u_c) - rho * norm.ppf(v_c)) / np.sqrt(1.0 - rho ** 2))


def _gauss_h_inv(u, v, rho):
    """h^{-1}(u|v; rho) = Phi(Phi^{-1}(u)*sqrt(1-rho^2) + rho*Phi^{-1}(v))"""
    eps = 1e-10
    u_c = np.clip(u, eps, 1.0 - eps)
    v_c = np.clip(v, eps, 1.0 - eps)
    return norm.cdf(norm.ppf(u_c) * np.sqrt(1.0 - rho ** 2) + rho * norm.ppf(v_c))


def _broadcast(u1, u2, r):
    u1a = np.atleast_1d(np.asarray(u1, dtype=np.float64)).ravel()
    u2a = np.atleast_1d(np.asarray(u2, dtype=np.float64)).ravel()
    ra = np.atleast_1d(np.asarray(r, dtype=np.float64)).ravel()
    n = max(len(u1a), len(u2a), len(ra))
    if len(u1a) == 1 and n > 1: u1a = np.full(n, u1a[0])
    if len(u2a) == 1 and n > 1: u2a = np.full(n, u2a[0])
    if len(ra) == 1 and n > 1: ra = np.full(n, ra[0])
    return u1a, u2a, ra


# ══════════════════════════════════════════════════════════════════
# BivariateGaussianCopula — fits into BivariateCopula hierarchy
# ══════════════════════════════════════════════════════════════════

class BivariateGaussianCopula(BivariateCopula):
    """
    Bivariate Gaussian copula.
    Parameter: rho in (-1, 1). No rotation.
    Transform: rho = 0.9999 * tanh(x/4), so x is unconstrained.
    """

    def __init__(self, rotate: int = 0):
        if rotate != 0:
            raise ValueError("Rotation not supported for Gaussian copula")
        super().__init__(0)
        self._name = "Gaussian copula"
        self._bounds = [(-0.9999, 0.9999)]

    @property
    def rotatable(self):
        return False

    @staticmethod
    def transform(x):
        """x -> rho: maps R to (-0.9999, 0.9999)."""
        return 0.9999 * np.tanh(np.asarray(x) / 4.0)

    @staticmethod
    def inv_transform(rho):
        """rho -> x."""
        return 4.0 * np.arctanh(np.asarray(rho) / 0.9999)

    def pdf_unrotated(self, u1, u2, r):
        u1a, u2a, ra = _broadcast(u1, u2, r)
        return _gauss_pdf_scipy(u1a, u2a, ra)

    def log_pdf_unrotated(self, u1, u2, r):
        u1a, u2a, ra = _broadcast(u1, u2, r)
        return _gauss_log_pdf_scipy(u1a, u2a, ra)

    def h_unrotated(self, u, v, r):
        ua, va, ra = _broadcast(u, v, r)
        return _gauss_h(ua, va, ra)

    def h_inverse_unrotated(self, u, v, r):
        ua, va, ra = _broadcast(u, v, r)
        return _gauss_h_inv(ua, va, ra)

    def sample(self, n, r, rng=None):
        """Sample from bivariate Gaussian copula."""
        _r = np.atleast_1d(np.asarray(r, dtype=np.float64))
        rho = _r[0] if _r.size == 1 else _r

        if rng is None:
            rng = np.random.default_rng()

        if np.isscalar(rho) or (isinstance(rho, np.ndarray) and rho.size == 1):
            rho_val = float(rho)
            z = rng.standard_normal((n, 2))
            x1 = z[:, 0]
            x2 = rho_val * z[:, 0] + np.sqrt(1.0 - rho_val ** 2) * z[:, 1]
            u = np.column_stack((norm.cdf(x1), norm.cdf(x2)))
        else:
            # Vector rho — sample each with its own correlation
            rho_arr = np.asarray(rho).ravel()
            z = rng.standard_normal((n, 2))
            x1 = z[:, 0]
            x2 = rho_arr * z[:, 0] + np.sqrt(1.0 - rho_arr ** 2) * z[:, 1]
            u = np.column_stack((norm.cdf(x1), norm.cdf(x2)))

        rot = self._rotate
        if rot == 90:
            u[:, 0] = 1.0 - u[:, 0]
        elif rot == 180:
            u[:, 0] = 1.0 - u[:, 0]
            u[:, 1] = 1.0 - u[:, 1]
        elif rot == 270:
            u[:, 1] = 1.0 - u[:, 1]
        return u


# ══════════════════════════════════════════════════════════════════
# GaussianCopula — d-dimensional
# ══════════════════════════════════════════════════════════════════

class GaussianCopula:
    """
    d-dimensional Gaussian copula.
    Parameter: correlation matrix R (d x d).

    log c(u; R) = sum_t [ log f_d(x_t; R) - sum_j log f_1(x_{t,j}) ]
    where x = Phi^{-1}(u).
    """

    def __init__(self):
        self.corr = None
        self.fit_result = None
        self._name = "Gaussian copula"

    @property
    def name(self):
        return self._name

    def fit(self, data, to_pobs=False, **kwargs):
        """
        Fit via MLE: R = corr(Phi^{-1}(u)).
        """
        u = np.asarray(data, dtype=np.float64)
        if to_pobs:
            u = pobs(u)

        eps = 1e-10
        u_c = np.clip(u, eps, 1.0 - eps)
        x = norm.ppf(u_c)
        self.corr = np.corrcoef(x.T)
        self.fit_result = {'corr': self.corr, 'method': 'MLE'}

        nll = self._nll(u)
        self.fit_result['log_likelihood'] = -nll
        return self.corr

    def log_likelihood(self, u):
        """Compute log-likelihood."""
        return -self._nll(u)

    def _nll(self, u):
        """Negative log-likelihood."""
        eps = 1e-10
        u_c = np.clip(u, eps, 1.0 - eps)
        x = norm.ppf(u_c)
        d = x.shape[1]

        ll_joint = multivariate_normal.logpdf(x, mean=np.zeros(d), cov=self.corr)
        ll_marginals = np.sum(norm.logpdf(x), axis=1)
        return -np.sum(ll_joint - ll_marginals)

    def sample(self, n, rng=None):
        """Sample from fitted Gaussian copula."""
        if self.corr is None:
            raise ValueError("Fit first")
        if rng is None:
            rng = np.random.default_rng()

        d = self.corr.shape[0]
        x = rng.multivariate_normal(np.zeros(d), self.corr, size=n)
        return norm.cdf(x)

    def predict(self, n):
        """Alias for sample (no latent dynamics)."""
        return self.sample(n)


# ══════════════════════════════════════════════════════════════════
# StudentCopula — d-dimensional
# ══════════════════════════════════════════════════════════════════

def _kendall_tau_matrix(u):
    """Estimate correlation matrix via Kendall's tau: R_ij = sin(pi/2 * tau_ij)."""
    d = u.shape[1]
    R = np.eye(d)
    for i in range(d):
        for j in range(i + 1, d):
            tau, _ = kendalltau(u[:, i], u[:, j])
            R[i, j] = np.sin(np.pi / 2.0 * tau)
            R[j, i] = R[i, j]
    return R


def _ensure_positive_definite(R):
    """Project to nearest PD matrix if needed."""
    eigvals = np.linalg.eigvalsh(R)
    if np.min(eigvals) > 0:
        return R
    # Nearest PD via eigenvalue clipping
    vals, vecs = np.linalg.eigh(R)
    vals = np.maximum(vals, 1e-6)
    R_pd = vecs @ np.diag(vals) @ vecs.T
    # Re-normalize to correlation matrix
    d = np.sqrt(np.diag(R_pd))
    R_pd = R_pd / np.outer(d, d)
    np.fill_diagonal(R_pd, 1.0)
    return R_pd


class StudentCopula:
    """
    d-dimensional Student-t copula.
    Parameters: shape matrix R (d x d correlation) and df (degrees of freedom).

    log c(u; R, df) = sum_t [ log f_d(x_t; R, df) - sum_j log f_1(x_{t,j}; df) ]
    where x = t_{df}^{-1}(u).
    """

    def __init__(self):
        self.shape = None
        self.df = None
        self.fit_result = None
        self._name = "Student-t copula"

    @property
    def name(self):
        return self._name

    def fit(self, data, to_pobs=False, **kwargs):
        """
        Fit via profile MLE:
        1. Optimize df via 1D minimization.
        2. For each df, estimate R via Kendall's tau.
        """
        u = np.asarray(data, dtype=np.float64)
        if to_pobs:
            u = pobs(u)

        def nll_profile(df):
            # df = np.exp(log_df[0])
            return self._nll_for_df(u, df)

        # Initial guess
        d = u.shape[1]
        x0 = np.array([np.log(float(d))])

        result = minimize(nll_profile, x0,
                          method='L-BFGS-B',
                          bounds=[(0.0001, np.inf)],  # df in [~0.1, ~400]
                          options={'gtol': 1e-2, 'eps': 1e-4})

        self.df = np.exp(result.x[0])
        x = t_dist.ppf(np.clip(u, 1e-10, 1.0 - 1e-10), df=self.df)
        R = _kendall_tau_matrix(x)
        self.shape = _ensure_positive_definite(R)

        nll = self._nll(u)
        self.fit_result = {
            'shape': self.shape,
            'df': self.df,
            'log_likelihood': -nll,
            'method': 'MLE'
        }
        return self.shape, self.df

    def _nll_for_df(self, u, df):
        """NLL for a given df (profile likelihood)."""
        eps = 1e-10
        u_c = np.clip(u, eps, 1.0 - eps)
        x = t_dist.ppf(u_c, df=df)
        R = _kendall_tau_matrix(x)
        R = _ensure_positive_definite(R)
        return self._nll_with_params(u, R, df)

    def _nll_with_params(self, u, R, df):
        eps = 1e-10
        u_c = np.clip(u, eps, 1.0 - eps)
        d = u_c.shape[1]
        x = t_dist.ppf(u_c, df=df)

        try:
            ll_joint = multivariate_t.logpdf(x, loc=np.zeros(d),
                                              shape=R, df=df)
            ll_marginals = np.sum(t_dist.logpdf(x, df=df), axis=1)
            return -np.sum(ll_joint - ll_marginals)
        except Exception:
            return 1e10

    def _nll(self, u):
        return self._nll_with_params(u, self.shape, self.df)

    def log_likelihood(self, u):
        return -self._nll(u)

    def sample(self, n, rng=None):
        """Sample from fitted Student-t copula."""
        if self.shape is None:
            raise ValueError("Fit first")

        d = self.shape.shape[0]
        x = multivariate_t.rvs(loc=np.zeros(d), shape=self.shape,
                                df=self.df, size=n)
        return t_dist.cdf(x, df=self.df)

    def predict(self, n):
        return self.sample(n)
