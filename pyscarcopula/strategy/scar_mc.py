"""
pyscarcopula.strategy.scar_mc — Monte Carlo SCAR strategies.

SCAR-P-OU: p-sampler (importance sampling without EIS).
SCAR-M-OU: m-sampler with Efficient Importance Sampling (EIS).

Both use the same underlying OU path generation and MC estimators
from numerical/mc_samplers.py and numerical/ou_kernels.py.

Parameters preserved from original:
  - n_tr (number of MC trajectories)
  - M_iterations (EIS iterations, m-sampler only)
  - stationary (initial distribution)
  - seed / dwt (Wiener process control)
"""

import numpy as np
from scipy.optimize import minimize, Bounds

from pyscarcopula._types import (
    LatentResult, NumericalConfig, DEFAULT_CONFIG,
    ou_params,
)
from pyscarcopula.strategy._base import register_strategy
from pyscarcopula.numerical.mc_samplers import (
    p_sampler_loglik, m_sampler_loglik, eis_find_auxiliary,
)
from pyscarcopula.numerical.ou_kernels import calculate_dwt


class _SCARMCBase:
    """Base class for MC-based SCAR strategies."""

    def __init__(self, config: NumericalConfig | None = None,
                 n_tr: int | None = None,
                 M_iterations: int = 3,
                 stationary: bool = True,
                 **kwargs):
        self.config = config or DEFAULT_CONFIG
        self.n_tr = n_tr if n_tr is not None else self.config.default_n_tr
        self.M_iterations = M_iterations
        self.stationary = stationary

    def _get_dwt(self, T, seed=None, dwt=None):
        """Get or generate Wiener increments."""
        if dwt is not None:
            return dwt
        _seed = seed if seed is not None else np.random.randint(1, 1000000)
        return calculate_dwt(T, self.n_tr, _seed)

    def _get_alpha0(self, copula, u):
        """Default initial point from MLE."""
        from pyscarcopula.strategy.mle import MLEStrategy
        mle = MLEStrategy(config=self.config)
        mle_result = mle.fit(copula, u)
        mu0 = float(np.atleast_1d(
            copula.inv_transform(np.atleast_1d(mle_result.copula_param))
        )[0])
        return np.array([1.0, mu0, 1.0])


@register_strategy('SCAR-P-OU')
class SCARPStrategy(_SCARMCBase):
    """P-sampler: MC likelihood without importance sampling."""

    def fit(self, copula, u: np.ndarray,
            alpha0: np.ndarray | None = None,
            tol: float | None = None,
            seed: int | None = None,
            dwt: np.ndarray | None = None,
            verbose: bool = False,
            **kwargs) -> LatentResult:

        tol = tol or self.config.default_tol_scar
        u = np.asarray(u, dtype=np.float64)
        T = len(u)
        dwt_data = self._get_dwt(T, seed, dwt)

        if alpha0 is None:
            alpha0 = self._get_alpha0(copula, u)

        bounds = Bounds([0.001, -np.inf, 0.001], [np.inf, np.inf, np.inf])

        def objective(alpha):
            if np.isnan(np.sum(alpha)):
                return 1e10
            try:
                return p_sampler_loglik(
                    alpha[0], alpha[1], alpha[2],
                    u, dwt_data, copula, self.stationary)
            except Exception:
                return 1e10

        if verbose:
            print(f"Fitting SCAR-P-OU, alpha0={alpha0}, n_tr={self.n_tr}")

        result = minimize(
            objective, alpha0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'gtol': tol, 'eps': 1e-4,
                     'maxfun': self.config.default_maxfun},
        )

        alpha = result.x
        params = ou_params(theta=alpha[0], mu=alpha[1], nu=alpha[2])

        return LatentResult(
            log_likelihood=-result.fun,
            method='SCAR-P-OU',
            copula_name=copula.name,
            success=result.success,
            nfev=result.nfev,
            params=params,
            n_tr=self.n_tr,
        )

    def log_likelihood(self, copula, u, result):
        raise NotImplementedError("MC log-likelihood is stochastic; use fit()")

    def smoothed_params(self, copula, u, result):
        raise NotImplementedError(
            "P-sampler does not provide smoothed params. "
            "Use SCAR-TM-OU for deterministic smoothing.")

    def rosenblatt_e2(self, copula, u, result):
        raise NotImplementedError("Use SCAR-TM-OU for GoF tests.")

    def mixture_h(self, copula, u, result):
        raise NotImplementedError("Use SCAR-TM-OU for vine h-functions.")

    def objective(self, copula, u: np.ndarray,
                  alpha: np.ndarray, **kwargs) -> float:
        """Minus log-likelihood via p-sampler (stochastic)."""
        seed = kwargs.get('seed')
        dwt = kwargs.get('dwt')
        dwt_data = self._get_dwt(len(u), seed, dwt)
        try:
            return p_sampler_loglik(
                alpha[0], alpha[1], alpha[2],
                u, dwt_data, copula, self.stationary)
        except Exception:
            return 1e10


@register_strategy('SCAR-M-OU')
class SCARMStrategy(_SCARMCBase):
    """M-sampler: MC likelihood with Efficient Importance Sampling."""

    def fit(self, copula, u: np.ndarray,
            alpha0: np.ndarray | None = None,
            tol: float | None = None,
            seed: int | None = None,
            dwt: np.ndarray | None = None,
            verbose: bool = False,
            **kwargs) -> LatentResult:

        tol = tol or self.config.default_tol_scar
        u = np.asarray(u, dtype=np.float64)
        T = len(u)
        dwt_data = self._get_dwt(T, seed, dwt)

        if alpha0 is None:
            alpha0 = self._get_alpha0(copula, u)

        bounds = Bounds([0.001, -np.inf, 0.001], [np.inf, np.inf, np.inf])

        def objective(alpha):
            if np.isnan(np.sum(alpha)):
                return 1e10
            try:
                a1t, a2t = eis_find_auxiliary(
                    alpha, u, self.M_iterations,
                    dwt_data, copula, self.stationary)
                return m_sampler_loglik(
                    alpha[0], alpha[1], alpha[2],
                    u, dwt_data, a1t, a2t, copula, self.stationary)
            except Exception:
                return 1e10

        if verbose:
            print(f"Fitting SCAR-M-OU, alpha0={alpha0}, n_tr={self.n_tr}, "
                  f"M_iterations={self.M_iterations}")

        result = minimize(
            objective, alpha0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'gtol': tol, 'eps': 1e-4,
                     'maxfun': self.config.default_maxfun},
        )

        alpha = result.x
        params = ou_params(theta=alpha[0], mu=alpha[1], nu=alpha[2])

        return LatentResult(
            log_likelihood=-result.fun,
            method='SCAR-M-OU',
            copula_name=copula.name,
            success=result.success,
            nfev=result.nfev,
            params=params,
            n_tr=self.n_tr,
            M_iterations=self.M_iterations,
        )

    def log_likelihood(self, copula, u, result):
        raise NotImplementedError("MC log-likelihood is stochastic; use fit()")

    def smoothed_params(self, copula, u, result):
        raise NotImplementedError(
            "M-sampler does not provide smoothed params. "
            "Use SCAR-TM-OU for deterministic smoothing.")

    def rosenblatt_e2(self, copula, u, result):
        raise NotImplementedError("Use SCAR-TM-OU for GoF tests.")

    def mixture_h(self, copula, u, result):
        raise NotImplementedError("Use SCAR-TM-OU for vine h-functions.")

    def objective(self, copula, u: np.ndarray,
                  alpha: np.ndarray, **kwargs) -> float:
        """Minus log-likelihood via m-sampler with EIS (stochastic)."""
        seed = kwargs.get('seed')
        dwt = kwargs.get('dwt')
        dwt_data = self._get_dwt(len(u), seed, dwt)
        try:
            a1t, a2t = eis_find_auxiliary(
                alpha, u, self.M_iterations,
                dwt_data, copula, self.stationary)
            return m_sampler_loglik(
                alpha[0], alpha[1], alpha[2],
                u, dwt_data, a1t, a2t, copula, self.stationary)
        except Exception:
            return 1e10
