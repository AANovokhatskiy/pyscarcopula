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
from pyscarcopula.strategy._base import copula_dimension, register_strategy
from pyscarcopula.strategy.predict_helpers import (
    predict_from_strategy,
    sample_predictive,
)
from pyscarcopula.strategy.initial_point import (
    _explicit_initialization_diagnostics,
    _fallback_initialization_diagnostics,
    _initialization_attempt,
    _initialization_diagnostics,
    smart_initial_point,
)
from pyscarcopula.numerical.mc_samplers import (
    p_sampler_loglik, m_sampler_loglik, eis_find_auxiliary,
)
from pyscarcopula.numerical.ou_kernels import calculate_dwt


class _SCARMCBase:
    """Base class for MC-based SCAR strategies."""

    def __init__(self, config: NumericalConfig | None = None,
                 n_tr: int | None = None,
                 M_iterations: int | None = None,
                 stationary: bool = True,
                 smart_init: bool = True,
                 **kwargs):
        self.config = config or DEFAULT_CONFIG
        self.n_tr = n_tr if n_tr is not None else self.config.default_n_tr
        self.M_iterations = (
            M_iterations if M_iterations is not None
            else self.config.default_M_iterations
        )
        if self.n_tr <= 0:
            raise ValueError("n_tr must be positive")
        if self.M_iterations < 0:
            raise ValueError("M_iterations must be non-negative")
        self.stationary = stationary
        self.smart_init = smart_init

    def _get_dwt(self, T, seed=None, dwt=None):
        """Get or generate Wiener increments."""
        if dwt is not None:
            return dwt
        _seed = seed if seed is not None else np.random.randint(1, 1000000)
        return calculate_dwt(T, self.n_tr, _seed)

    def _get_alpha0(self, copula, u, verbose):
        alpha0 = None
        initialization = None
        if self.smart_init:
            try:
                alpha0, init_info = smart_initial_point(
                    u, copula, verbose=verbose)
                initialization = init_info['initialization']
                if verbose:
                    print(f"Smart init: {init_info.get('chosen_method')}, "
                            f"alpha0={alpha0}")
            except Exception as exc:
                initialization = _initialization_diagnostics(
                    'automatic',
                    'failed',
                    np.array([1.0, 0.0, 1.0]),
                    [_initialization_attempt(
                        'smart_initial_point',
                        success=False,
                        error=exc)],
                )
                initialization['success'] = False
                if verbose:
                    print(
                        "Smart init failed "
                        f"({type(exc).__name__}: {exc}); "
                        "trying mle_default")
                alpha0 = None
        if alpha0 is None:
            """Default initial point from MLE."""
            from pyscarcopula.strategy.mle import MLEStrategy
            mle = MLEStrategy(config=self.config)
            mle_result = mle.fit(copula, u)
            mu0 = float(np.atleast_1d(
                copula.inv_transform(np.atleast_1d(mle_result.copula_param))
            )[0])
            alpha0 = np.array([1.0, mu0, 1.0])
            if initialization is None:
                initialization = _initialization_diagnostics(
                    'mle_default',
                    'mle_default',
                    alpha0,
                    [_initialization_attempt(
                        'mle_default', success=True)],
                )
            else:
                initialization = _fallback_initialization_diagnostics(
                    initialization, 'mle_default', alpha0)
        return alpha0, initialization

    def predictive_params(self, copula, u, result, n, rng=None, **kwargs):
        """Stationary OU predictive parameters for MC SCAR variants."""
        if rng is None:
            rng = np.random.default_rng()
        p = result.params
        sigma = p.nu / np.sqrt(2.0 * p.kappa)
        x_t = rng.normal(p.mu, sigma, n)
        return copula.transform(x_t)

    def sample(self, copula, u, result, n, rng=None, **kwargs):
        """Simulate n observations with an OU-driven copula parameter."""
        if rng is None:
            rng = np.random.default_rng()
        r = self.model_sample_params(copula, result, n, rng=rng)
        d = copula_dimension(copula, u)
        return sample_predictive(copula, n, r, rng=rng, d=d)

    def model_sample_params(self, copula, result, n, rng=None, **kwargs):
        """OU trajectory parameters for unconditional model reproduction."""
        if rng is None:
            rng = np.random.default_rng()
        p = result.params
        kappa, mu, nu = p.kappa, p.mu, p.nu
        dt = 1.0 / (n - 1) if n > 1 else 1.0
        rho_ou = np.exp(-kappa * dt)
        sigma_cond = np.sqrt(nu ** 2 / (2.0 * kappa) * (1.0 - rho_ou ** 2))
        x = np.empty(n)
        x[0] = rng.normal(mu, nu / np.sqrt(2.0 * kappa))
        for t in range(1, n):
            x[t] = (
                mu + rho_ou * (x[t - 1] - mu)
                + sigma_cond * rng.standard_normal()
            )
        return copula.transform(x)

    def model_sample_state(self, copula, result, **kwargs):
        return None

    def predict(self, copula, u, result, n, rng=None, **kwargs):
        """Predict by sampling from the stationary OU distribution."""
        return predict_from_strategy(
            self, copula, u, result, n, rng=rng, **kwargs)


@register_strategy('SCAR-P-OU')
class SCARPStrategy(_SCARMCBase):
    """P-sampler: MC likelihood without importance sampling."""

    def fit(self, copula, u: np.ndarray,
            alpha0: np.ndarray | None = None,
            gtol: float | None = None,
            ftol: float | None = None,
            maxfun: int | None = None,
            maxiter: int | None = None,
            maxls: int | None = None,
            eps: float | None = None,
            maxcor: int | None = None,
            finite_diff_rel_step: float | None = None,
            seed: int | None = None,
            dwt: np.ndarray | None = None,
            verbose: bool = False,
            **kwargs) -> LatentResult:

        if 'tol' in kwargs:
            raise TypeError("tol is not supported; use gtol")
        optimizer_options = self.config.scar_optimizer.options(
            gtol=gtol,
            ftol=ftol,
            maxfun=maxfun,
            maxiter=maxiter,
            maxls=maxls,
            eps=eps,
            maxcor=maxcor,
            finite_diff_rel_step=finite_diff_rel_step,
        )
        u = np.asarray(u, dtype=np.float64)
        T = len(u)
        dwt_data = self._get_dwt(T, seed, dwt)

        if alpha0 is None:
            alpha0, initialization = self._get_alpha0(
                copula, u, verbose)
        else:
            initialization = _explicit_initialization_diagnostics(alpha0)

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
            options=optimizer_options,
        )

        alpha = result.x
        params = ou_params(kappa=alpha[0], mu=alpha[1], nu=alpha[2])

        return LatentResult(
            log_likelihood=-result.fun,
            method='SCAR-P-OU',
            copula_name=copula.name,
            success=result.success,
            nfev=result.nfev,
            params=params,
            n_tr=self.n_tr,
            diagnostics={"initialization": initialization},
        )

    def log_likelihood(self, copula, u, result):
        raise NotImplementedError("MC log-likelihood is stochastic; use fit()")

    def predictive_mean(self, copula, u, result):
        raise NotImplementedError(
            "P-sampler does not provide predictive means. "
            "Use SCAR-TM-OU for deterministic predictive means.")

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
            gtol: float | None = None,
            ftol: float | None = None,
            maxfun: int | None = None,
            maxiter: int | None = None,
            maxls: int | None = None,
            eps: float | None = None,
            maxcor: int | None = None,
            finite_diff_rel_step: float | None = None,
            seed: int | None = None,
            dwt: np.ndarray | None = None,
            verbose: bool = False,
            **kwargs) -> LatentResult:

        if 'tol' in kwargs:
            raise TypeError("tol is not supported; use gtol")
        optimizer_options = self.config.scar_optimizer.options(
            gtol=gtol,
            ftol=ftol,
            maxfun=maxfun,
            maxiter=maxiter,
            maxls=maxls,
            eps=eps,
            maxcor=maxcor,
            finite_diff_rel_step=finite_diff_rel_step,
        )
        u = np.asarray(u, dtype=np.float64)
        T = len(u)
        dwt_data = self._get_dwt(T, seed, dwt)

        if alpha0 is None:
            alpha0, initialization = self._get_alpha0(
                copula, u, verbose)
        else:
            initialization = _explicit_initialization_diagnostics(alpha0)

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
            options=optimizer_options,
        )

        alpha = result.x
        params = ou_params(kappa=alpha[0], mu=alpha[1], nu=alpha[2])

        return LatentResult(
            log_likelihood=-result.fun,
            method='SCAR-M-OU',
            copula_name=copula.name,
            success=result.success,
            nfev=result.nfev,
            params=params,
            n_tr=self.n_tr,
            M_iterations=self.M_iterations,
            diagnostics={"initialization": initialization},
        )

    def log_likelihood(self, copula, u, result):
        raise NotImplementedError("MC log-likelihood is stochastic; use fit()")

    def predictive_mean(self, copula, u, result):
        raise NotImplementedError(
            "M-sampler does not provide predictive means. "
            "Use SCAR-TM-OU for deterministic predictive means.")

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
