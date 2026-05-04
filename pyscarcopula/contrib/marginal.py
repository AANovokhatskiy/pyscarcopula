"""
Marginal distribution models for portfolio risk estimation.

Usage:
    model = MarginalModel.create('johnsonsu')
    params = model.fit_rolling(data, window_len=250)
    r_samples = model.rvs(params[t], N=10000)
    r_from_u  = model.ppf(u_samples, params[t])
"""

import numpy as np
from numba import njit
from dataclasses import dataclass
from scipy.optimize import minimize
from scipy.stats import norm, johnsonsu, genlogistic, laplace_asymmetric, genhyperbolic, levy_stable
from joblib import Parallel, delayed
from typing import Literal

from pyscarcopula._utils import pobs

# ══════════════════════════════════════════════════════════════════
# Base class
# ══════════════════════════════════════════════════════════════════

class MarginalModel:
    """Base class for marginal distribution models."""

    name = 'base'
    n_params = 0  # number of params per asset

    def fit_single(self, data_slice):
        """
        Fit params for each asset in data_slice.
        data_slice: (window_len, dim).
        Returns: (dim, n_params).
        """
        raise NotImplementedError

    def fit_rolling(self, data, window_len, n_jobs=-1):
        """
        Rolling window fit. data: (T, dim).
        Returns: (T, dim, n_params).
        """
        T, dim = data.shape
        iters = T - window_len + 1
        res = np.zeros((T, dim, self.n_params))

        fit_results = Parallel(n_jobs=n_jobs)(
            delayed(self.fit_single)(data[i:i + window_len])
            for i in range(iters)
        )

        for i in range(iters):
            idx = i + window_len - 1
            res[idx] = np.array(fit_results[i])

        return res

    def ppf(self, u, params):
        """
        Inverse CDF. u: (N, dim), params: (dim, n_params).
        Returns: (N, dim).
        """
        raise NotImplementedError

    def cdf(self, x, params):
        """CDF. x: (N, dim), params: (dim, n_params). Returns: (N, dim)."""
        raise NotImplementedError

    def rvs(self, params, N):
        """
        Random samples. params: (dim, n_params).
        Returns: (N, dim).
        """
        raise NotImplementedError

    @staticmethod
    def create(name: Literal['normal', 'johnsonsu',
                             'logistic', 'laplace',
                             'hyperbolic', 'stable',
                             'arma-garch', 'armagarch',
                             'ar1-garch']):
        """Factory method."""
        registry = {
            'normal': NormalMarginal,
            'johnsonsu': JohnsonSUMarginal,
            'logistic': GenLogisticMarginal,
            'laplace': LaplaceMarginal,
            'hyperbolic': HyperbolicMarginal,
            'stable': StableMarginal,
            'arma-garch': ARMAGARCHMarginal,
            'armagarch': ARMAGARCHMarginal,
            'ar1-garch': ARMAGARCHMarginal,
        }
        name = name.lower()
        if name not in registry:
            raise ValueError(f"Unknown marginal '{name}'. Available: {list(registry.keys())}")
        return registry[name]()


# ══════════════════════════════════════════════════════════════════
# Scipy-based marginals (common pattern)
# ══════════════════════════════════════════════════════════════════

class ScipyMarginal(MarginalModel):
    """Marginal backed by a scipy.stats distribution."""

    _dist = None  # override in subclass

    def fit_single(self, data_slice):
        dim = data_slice.shape[1]
        result = np.zeros((dim, self.n_params))
        for k in range(dim):
            result[k] = self._dist.fit(data_slice[:, k], method='MLE')
        return result

    def ppf(self, u, params):
        dim = u.shape[1]
        res = np.empty_like(u)
        for k in range(dim):
            res[:, k] = self._dist.ppf(u[:, k], *params[k])
        return res

    def cdf(self, x, params):
        dim = x.shape[1]
        res = np.empty_like(x)
        for k in range(dim):
            res[:, k] = self._dist.cdf(x[:, k], *params[k])
        return res

    def rvs(self, params, N):
        dim = len(params)
        res = np.empty((N, dim))
        for k in range(dim):
            res[:, k] = self._dist.rvs(*params[k], size=N)
        return res


class JohnsonSUMarginal(ScipyMarginal):
    name = 'johnsonsu'
    n_params = 4
    _dist = johnsonsu


class GenLogisticMarginal(ScipyMarginal):
    name = 'logistic'
    n_params = 3
    _dist = genlogistic


class LaplaceMarginal(ScipyMarginal):
    name = 'laplace'
    n_params = 3
    _dist = laplace_asymmetric


# ══════════════════════════════════════════════════════════════════
# Normal — numba-optimized fit
# ══════════════════════════════════════════════════════════════════

@njit(cache=True)
def _normal_fit_rolling(data, window_len):
    T, dim = data.shape
    res = np.zeros((T, dim, 2))
    for i in range(T - window_len + 1):
        idx = i + window_len - 1
        for j in range(dim):
            sl = data[i:i + window_len, j]
            res[idx, j, 0] = np.mean(sl)
            res[idx, j, 1] = np.std(sl)
    return res


class NormalMarginal(MarginalModel):
    name = 'normal'
    n_params = 2

    def fit_single(self, data_slice):
        dim = data_slice.shape[1]
        result = np.zeros((dim, 2))
        for k in range(dim):
            result[k, 0] = np.mean(data_slice[:, k])
            result[k, 1] = np.std(data_slice[:, k])
        return result

    def fit_rolling(self, data, window_len, n_jobs=-1):
        """Override: numba is faster than joblib for normal."""
        return _normal_fit_rolling(data, window_len)

    def ppf(self, u, params):
        dim = u.shape[1]
        res = np.empty_like(u)
        for k in range(dim):
            res[:, k] = norm.ppf(u[:, k], params[k, 0], params[k, 1])
        return res

    def cdf(self, x, params):
        dim = x.shape[1]
        res = np.empty_like(x)
        for k in range(dim):
            res[:, k] = norm.cdf(x[:, k], params[k, 0], params[k, 1])
        return res

    def rvs(self, params, N):
        dim = len(params)
        res = np.empty((N, dim))
        for k in range(dim):
            res[:, k] = np.random.normal(params[k, 0], params[k, 1], size=N)
        return res


# ══════════════════════════════════════════════════════════════════
# Hyperbolic — scipy + inverse CDF for ppf
# ══════════════════════════════════════════════════════════════════

# =============================================================================
# AR(1)-GARCH(1,1) conditional marginal filter
# =============================================================================

@dataclass(frozen=True)
class ARMAGARCHFit:
    """Result for one univariate AR(1)-GARCH(1,1) marginal fit."""

    params: np.ndarray
    residuals: np.ndarray
    sigma2: np.ndarray
    standardized_residuals: np.ndarray
    log_likelihood: float
    success: bool
    nfev: int
    message: str

    @property
    def const(self) -> float:
        return float(self.params[0])

    @property
    def ar1(self) -> float:
        return float(self.params[1])

    @property
    def omega(self) -> float:
        return float(self.params[2])

    @property
    def alpha(self) -> float:
        return float(self.params[3])

    @property
    def beta(self) -> float:
        return float(self.params[4])


def _ar_residuals(x, const, ar1):
    eps = np.empty_like(x, dtype=np.float64)
    if x.size == 0:
        return eps
    eps[0] = x[0] - np.mean(x)
    if x.size > 1:
        eps[1:] = x[1:] - const - ar1 * x[:-1]
    return eps


def _fit_ar_mean(x, ar_order):
    if ar_order == 0 or x.size < 3:
        mean = float(np.mean(x))
        return mean, 0.0, x - mean
    if ar_order != 1:
        raise NotImplementedError("ARMAGARCHMarginal currently supports ar_order 0 or 1")

    y = x[1:]
    lag = x[:-1]
    design = np.column_stack([np.ones_like(lag), lag])
    try:
        const, ar1 = np.linalg.lstsq(design, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        const, ar1 = float(np.mean(x)), 0.0
    ar1 = float(np.clip(ar1, -0.99, 0.99))
    const = float(const)
    return const, ar1, _ar_residuals(x, const, ar1)


def _garch11_filter(eps, omega, alpha, beta):
    sigma2 = np.empty_like(eps, dtype=np.float64)
    unconditional = omega / max(1.0 - alpha - beta, 1e-6)
    sample_var = float(np.var(eps, ddof=1)) if eps.size > 1 else float(eps[0] ** 2)
    sigma2[0] = max(unconditional, sample_var, 1e-12)
    for t in range(1, eps.size):
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
    return np.maximum(sigma2, 1e-12)


def _fit_garch11(eps, maxiter=300):
    eps = np.asarray(eps, dtype=np.float64)
    var_eps = float(np.var(eps, ddof=1)) if eps.size > 1 else 1e-8
    var_eps = max(var_eps, 1e-12)
    alpha0 = 0.05
    beta0 = 0.90
    omega0 = max(var_eps * (1.0 - alpha0 - beta0), 1e-12)

    def neg_ll(params):
        omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or (alpha + beta) >= 0.999:
            return 1e12
        sigma2 = _garch11_filter(eps, omega, alpha, beta)
        return 0.5 * np.sum(np.log(2.0 * np.pi) + np.log(sigma2) + eps ** 2 / sigma2)

    x0 = np.array([omega0, alpha0, beta0], dtype=np.float64)
    bounds = [(1e-12, 10.0 * var_eps), (1e-8, 0.5), (1e-8, 0.999)]
    res = minimize(
        neg_ll,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": int(maxiter), "gtol": 1e-5},
    )
    omega, alpha, beta = res.x
    if alpha + beta >= 0.999:
        scale = 0.998 / (alpha + beta)
        alpha *= scale
        beta *= scale
    sigma2 = _garch11_filter(eps, omega, alpha, beta)
    ll = -float(neg_ll(np.array([omega, alpha, beta], dtype=np.float64)))
    return (float(omega), float(alpha), float(beta)), sigma2, ll, res


def _fit_arma_garch_series(x, ar_order=1, ma_order=0, maxiter=300):
    if ma_order != 0:
        raise NotImplementedError(
            "ARMAGARCHMarginal currently implements AR(0/1)-GARCH(1,1); "
            "MA terms are not included."
        )
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < 10:
        raise ValueError("ARMA-GARCH marginal fit requires at least 10 observations")
    if not np.all(np.isfinite(x)):
        raise ValueError("ARMA-GARCH marginal fit requires finite observations")

    const, ar1, eps = _fit_ar_mean(x, ar_order)
    (omega, alpha, beta), sigma2, ll, res = _fit_garch11(eps, maxiter=maxiter)
    z = eps / np.sqrt(sigma2)
    params = np.array([const, ar1, omega, alpha, beta], dtype=np.float64)
    return ARMAGARCHFit(
        params=params,
        residuals=eps,
        sigma2=sigma2,
        standardized_residuals=z,
        log_likelihood=ll,
        success=bool(res.success),
        nfev=int(getattr(res, "nfev", 0)),
        message=str(getattr(res, "message", "")),
    )


class ARMAGARCHMarginal(MarginalModel):
    """
    AR(1)-GARCH(1,1) marginal filter for copula pseudo-observations.

    This class is intended for robustness checks where raw returns are first
    filtered for linear autocorrelation and conditional heteroskedasticity, and
    the standardized residuals are then transformed to pseudo-observations.
    The implementation is dependency-free and uses the same Gaussian
    GARCH(1,1) likelihood as the experimental DCC module.
    """

    name = "arma-garch"
    n_params = 5

    def __init__(self, ar_order=1, ma_order=0, maxiter=300):
        if ar_order not in (0, 1):
            raise NotImplementedError("ARMAGARCHMarginal supports ar_order 0 or 1")
        if ma_order != 0:
            raise NotImplementedError("ARMAGARCHMarginal currently supports ma_order=0")
        self.ar_order = int(ar_order)
        self.ma_order = int(ma_order)
        self.maxiter = int(maxiter)

    def fit_series(self, x):
        """Fit one series and return an ARMAGARCHFit object."""
        return _fit_arma_garch_series(
            x,
            ar_order=self.ar_order,
            ma_order=self.ma_order,
            maxiter=self.maxiter,
        )

    def fit_single(self, data_slice):
        dim = data_slice.shape[1]
        result = np.zeros((dim, self.n_params), dtype=np.float64)
        for k in range(dim):
            result[k] = self.fit_series(data_slice[:, k]).params
        return result

    def standardize(self, data):
        """Fit each column and return standardized residuals plus diagnostics."""
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError("data must be a 2D array with shape (T, dim)")
        z = np.empty_like(data, dtype=np.float64)
        fits = []
        for k in range(data.shape[1]):
            fit = self.fit_series(data[:, k])
            z[:, k] = fit.standardized_residuals
            fits.append(fit)
        return z, fits

    def pseudo_observations(self, data, transform="rank"):
        """
        Convert returns to pseudo-observations after AR-GARCH filtering.

        Parameters
        ----------
        transform : {'rank', 'normal'}
            ``rank`` applies the empirical rank transform to standardized
            residuals. ``normal`` applies the standard-normal CDF.
        """
        z, fits = self.standardize(data)
        transform = transform.lower()
        if transform == "rank":
            return pobs(z), fits
        if transform == "normal":
            return np.clip(norm.cdf(z), 1e-10, 1.0 - 1e-10), fits
        raise ValueError("transform must be 'rank' or 'normal'")

    def ppf(self, u, params):
        raise NotImplementedError(
            "ARMAGARCHMarginal is a conditional filter; use standardize() "
            "or pseudo_observations() for copula workflows."
        )

    def cdf(self, x, params):
        raise NotImplementedError(
            "ARMAGARCHMarginal is a conditional filter; use standardize() "
            "or pseudo_observations() for copula workflows."
        )

    def rvs(self, params, N):
        raise NotImplementedError(
            "ARMAGARCHMarginal simulation requires a conditioning history and "
            "is not implemented in the static MarginalModel interface."
        )


def arma_garch_standardize(data, ar_order=1, ma_order=0, maxiter=300):
    """Return AR(0/1)-GARCH(1,1) standardized residuals and fits."""
    model = ARMAGARCHMarginal(
        ar_order=ar_order,
        ma_order=ma_order,
        maxiter=maxiter,
    )
    return model.standardize(data)


def arma_garch_pobs(data, ar_order=1, ma_order=0, maxiter=300, transform="rank"):
    """Return filtered pseudo-observations and per-margin fits."""
    model = ARMAGARCHMarginal(
        ar_order=ar_order,
        ma_order=ma_order,
        maxiter=maxiter,
    )
    return model.pseudo_observations(data, transform=transform)


class HyperbolicMarginal(ScipyMarginal):
    name = 'hyperbolic'
    n_params = 5
    _dist = genhyperbolic

    def ppf_from_samples(self, u, x_samples):
        """
        Inverse CDF via quantile of pre-generated samples.
        Useful when analytical ppf is slow.
        u: (N, dim), x_samples: (M, dim).
        """
        dim = u.shape[1]
        res = np.empty_like(u)
        for k in range(dim):
            res[:, k] = np.quantile(x_samples[:, k], u[:, k],
                                     method='median_unbiased')
        return res


# ══════════════════════════════════════════════════════════════════
# Stable — ScipyMarginal + batched rvs + ppf_from_samples
# ══════════════════════════════════════════════════════════════════

class StableMarginal(ScipyMarginal):
    name = 'stable'
    n_params = 4
    _dist = levy_stable

    def ppf_from_samples(self, u, x_samples):
        """Inverse CDF via quantile of pre-generated samples."""
        dim = u.shape[1]
        res = np.empty_like(u)
        for k in range(dim):
            res[:, k] = np.quantile(x_samples[:, k], u[:, k],
                                     method='median_unbiased')
        return res

    def rvs(self, params, N, batch_size=100000, n_jobs=-1):
        """Override: batched parallel rvs for large N."""
        if N >= 100 * batch_size:
            n_batches = (N + batch_size - 1) // batch_size
            results = Parallel(n_jobs=n_jobs)(
                delayed(_stable_rvs_batch)(params, batch_size)
                for _ in range(n_batches)
            )
            return np.vstack(results)[:N]
        return _stable_rvs_batch(params, N)


def _stable_rvs_batch(params, N):
    """Top-level function for joblib pickling."""
    dim = len(params)
    res = np.empty((N, dim))
    for k in range(dim):
        res[:, k] = levy_stable.rvs(*params[k], size=N)
    return res
