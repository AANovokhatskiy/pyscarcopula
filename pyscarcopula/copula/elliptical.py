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
from math import erf, sqrt
from scipy.stats import norm, t as t_dist, multivariate_normal, multivariate_t, kendalltau
from scipy.optimize import minimize
from numba import njit

from pyscarcopula.copula.base import BivariateCopula
from pyscarcopula._utils import pobs


from pyscarcopula.copula.experimental.equicorr import _ndtri


@njit(cache=True)
def _gauss_log_pdf_numba(u1, u2, rho):
    """Bivariate Gaussian copula log-density (numba-compiled)."""
    n = len(u1)
    out = np.empty(n)
    for i in range(n):
        v1 = min(max(u1[i], 1e-10), 1.0 - 1e-10)
        v2 = min(max(u2[i], 1e-10), 1.0 - 1e-10)
        x1 = _ndtri(v1)
        x2 = _ndtri(v2)
        r = rho[i]
        r2 = r * r
        out[i] = (-0.5 * np.log(1.0 - r2)
                  - 0.5 * (r2 * (x1 * x1 + x2 * x2) - 2.0 * r * x1 * x2)
                  / (1.0 - r2))
    return out


@njit(cache=True)
def _gauss_dlog_pdf_drho(u1, u2, rho):
    """d(log c)/d(rho) for bivariate Gaussian copula."""
    n = len(u1)
    out = np.empty(n)
    for i in range(n):
        v1 = min(max(u1[i], 1e-10), 1.0 - 1e-10)
        v2 = min(max(u2[i], 1e-10), 1.0 - 1e-10)
        x1 = _ndtri(v1)
        x2 = _ndtri(v2)
        r = rho[i]
        r2 = r * r
        omr2 = 1.0 - r2
        s1 = x1 * x1 + x2 * x2
        s12 = x1 * x2
        dlog_det = r / omr2
        num = (2.0 * r * s1 - 2.0 * s12) * omr2 + 2.0 * r * (r2 * s1 - 2.0 * r * s12)
        dquad = num / (omr2 * omr2)
        out[i] = dlog_det - 0.5 * dquad
    return out


@njit(cache=True)
def _gauss_precompute_x(u1, u2):
    """Precompute Phi^{-1}(u) once for repeated evaluation."""
    n = len(u1)
    x1 = np.empty(n)
    x2 = np.empty(n)
    for i in range(n):
        x1[i] = _ndtri(min(max(u1[i], 1e-10), 1.0 - 1e-10))
        x2[i] = _ndtri(min(max(u2[i], 1e-10), 1.0 - 1e-10))
    return x1, x2


@njit(cache=True)
def _gauss_negloglik_and_grad_from_x(x1, x2, rho_scalar):
    """Fused -logL and d(-logL)/drho from precomputed x1, x2.

    rho_scalar: single float (MLE case).
    Returns (negloglik, grad) as scalars.
    """
    n = len(x1)
    r = rho_scalar
    r2 = r * r
    omr2 = 1.0 - r2

    sum_ll = 0.0
    sum_grad = 0.0
    log_det = -0.5 * np.log(omr2)
    dlog_det = r / omr2

    for i in range(n):
        s1 = x1[i] * x1[i] + x2[i] * x2[i]
        s12 = x1[i] * x2[i]

        quad = (r2 * s1 - 2.0 * r * s12) / omr2
        sum_ll += log_det - 0.5 * quad

        num = (2.0 * r * s1 - 2.0 * s12) * omr2 + 2.0 * r * (r2 * s1 - 2.0 * r * s12)
        dquad = num / (omr2 * omr2)
        sum_grad += dlog_det - 0.5 * dquad

    return -sum_ll, -sum_grad


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


@njit(cache=True)
def _norm_cdf_numba(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


@njit(cache=True)
def _gauss_h_numba(u, v, rho):
    n = len(u)
    out = np.empty(n)
    for i in range(n):
        u_c = min(max(u[i], 1e-10), 1.0 - 1e-10)
        v_c = min(max(v[i], 1e-10), 1.0 - 1e-10)
        r = rho[i]
        z = (_ndtri(u_c) - r * _ndtri(v_c)) / np.sqrt(1.0 - r * r)
        out[i] = _norm_cdf_numba(z)
    return out


@njit(cache=True)
def _gauss_h_inv_numba(u, v, rho):
    n = len(u)
    out = np.empty(n)
    for i in range(n):
        u_c = min(max(u[i], 1e-10), 1.0 - 1e-10)
        v_c = min(max(v[i], 1e-10), 1.0 - 1e-10)
        r = rho[i]
        z = _ndtri(u_c) * np.sqrt(1.0 - r * r) + r * _ndtri(v_c)
        out[i] = _norm_cdf_numba(z)
    return out


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

    Transform types:
        'xtanh' (default): rho = 0.9999 * tanh(x/4) — slow saturation
        'softplus': rho = 0.9999 * tanh(x) — faster saturation, sharper transitions
    """

    def __init__(self, rotate: int = 0, transform_type: str = 'xtanh'):
        if rotate != 0:
            raise ValueError("Rotation not supported for Gaussian copula")
        super().__init__(0)
        self._name = "Gaussian copula"
        self._bounds = [(-0.9999, 0.9999)]  # bounds in rho-space (copula parameter)
        if transform_type not in ('xtanh', 'softplus'):
            raise ValueError(f"transform_type must be 'xtanh' or 'softplus', got '{transform_type}'")
        self._transform_type = transform_type

    @property
    def rotatable(self):
        return False

    def transform(self, x):
        """x -> rho: maps R to (-0.9999, 0.9999)."""
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        if self._transform_type == 'softplus':
            return 0.9999 * np.tanh(x)
        return 0.9999 * np.tanh(x / 4.0)

    def inv_transform(self, rho):
        """rho -> x."""
        rho = np.atleast_1d(np.asarray(rho, dtype=np.float64))
        if self._transform_type == 'softplus':
            return np.arctanh(np.clip(rho / 0.9999, -0.9999, 0.9999))
        return 4.0 * np.arctanh(np.clip(rho / 0.9999, -0.9999, 0.9999))

    def dtransform(self, x):
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        if self._transform_type == 'softplus':
            return 0.9999 * (1.0 - np.tanh(x) ** 2)
        return 0.9999 / 4.0 * (1.0 - np.tanh(x / 4.0) ** 2)

    def pdf_unrotated(self, u1, u2, r):
        u1a, u2a, ra = _broadcast(u1, u2, r)
        return np.exp(_gauss_log_pdf_numba(u1a, u2a, ra))

    def log_pdf_unrotated(self, u1, u2, r):
        u1a, u2a, ra = _broadcast(u1, u2, r)
        return _gauss_log_pdf_numba(u1a, u2a, ra)

    def dlog_pdf_dr_unrotated(self, u1, u2, r):
        u1a, u2a, ra = _broadcast(u1, u2, r)
        return _gauss_dlog_pdf_drho(u1a, u2a, ra)

    def mle_objective_fused(self, u):
        """Return fused (neg_loglik, neg_grad) callable with precomputed Phi^{-1}.

        Works directly in rho-space (copula parameter).
        Avoids recomputing norm.ppf on every L-BFGS-B iteration.
        Returns a function: rho_arr -> (float, ndarray).
        """
        x1, x2 = _gauss_precompute_x(u[:, 0], u[:, 1])

        def objective_and_grad(rho_arr):
            rho = rho_arr[0]
            nll, ngrad = _gauss_negloglik_and_grad_from_x(x1, x2, rho)
            return float(nll), np.array([float(ngrad)])

        return objective_and_grad

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

    def predict(self, n, rng=None):
        """Alias for sample (no latent dynamics)."""
        return self.sample(n, rng=rng)


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
        if rng is None:
            rng = np.random.default_rng()

        d = self.shape.shape[0]
        x = multivariate_t.rvs(loc=np.zeros(d), shape=self.shape,
                                df=self.df, size=n, random_state=rng)
        return t_dist.cdf(x, df=self.df)

    def predict(self, n, rng=None):
        return self.sample(n, rng=rng)
