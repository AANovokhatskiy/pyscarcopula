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
from scipy.stats import norm, johnsonsu, genlogistic, laplace_asymmetric, genhyperbolic, levy_stable
from joblib import Parallel, delayed
from typing import Literal

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
                             'hyperbolic', 'stable']):
        """Factory method."""
        registry = {
            'normal': NormalMarginal,
            'johnsonsu': JohnsonSUMarginal,
            'logistic': GenLogisticMarginal,
            'laplace': LaplaceMarginal,
            'hyperbolic': HyperbolicMarginal,
            'stable': StableMarginal,
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