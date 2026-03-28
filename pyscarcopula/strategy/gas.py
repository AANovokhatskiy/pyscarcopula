"""
pyscarcopula.strategy.gas — GAS estimation strategy.

Observation-driven model of Creal, Koopman and Lucas (2013).
All numerical computation delegated to numerical/gas_filter.py.
"""

import numpy as np
from scipy.optimize import minimize, Bounds

from pyscarcopula._types import (
    GASResult, NumericalConfig, DEFAULT_CONFIG,
    gas_params,
)
from pyscarcopula.strategy._base import register_strategy
from pyscarcopula.numerical.gas_filter import (
    gas_filter, gas_negloglik, gas_rosenblatt, gas_mixture_h,
)


@register_strategy('GAS')
class GASStrategy:
    """GAS estimation strategy.

    Parameters
    ----------
    config : NumericalConfig
    scaling : 'unit' or 'fisher'
        Score scaling type. 'unit' is default (simple, robust).
        'fisher' uses inverse Fisher information (optimal but less stable).
    """

    def __init__(self, config: NumericalConfig | None = None,
                 scaling: str = 'unit', **kwargs):
        self.config = config or DEFAULT_CONFIG
        self.scaling = scaling

    def fit(self, copula, u: np.ndarray,
            alpha0: np.ndarray | None = None,
            tol: float | None = None,
            verbose: bool = False,
            **kwargs) -> GASResult:
        """Fit GAS model.

        Parameters
        ----------
        copula : CopulaProtocol
        u : (T, 2) pseudo-observations
        alpha0 : (3,) initial guess [omega, alpha, beta], or None
        tol : optimizer tolerance
        verbose : print progress
        **kwargs : ignored

        Returns
        -------
        GASResult
        """
        tol = tol or self.config.default_tol_gas
        score_eps = self.config.gas_score_eps

        # Default initial point
        if alpha0 is None:
            from pyscarcopula.strategy.mle import MLEStrategy
            mle = MLEStrategy(config=self.config)
            mle_result = mle.fit(copula, u)
            mu_mle = float(np.atleast_1d(
                copula.inv_transform(np.atleast_1d(mle_result.copula_param))
            )[0])
            alpha0 = np.array([
                mu_mle * 0.05,    # omega ≈ f_bar * (1 - beta)
                0.05,             # alpha: moderate sensitivity
                0.95,             # beta: high persistence
            ])

        if verbose:
            print(f"GAS fit: alpha0={alpha0}, scaling={self.scaling}")

        bounds = Bounds(
            [-np.inf, -5.0, -0.999],
            [np.inf, 5.0, 0.999],
        )

        result = minimize(
            lambda x: gas_negloglik(x[0], x[1], x[2], u, copula,
                                    self.scaling, score_eps),
            alpha0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'gtol': tol, 'eps': 1e-5, 'maxfun': 200},
        )

        params = gas_params(
            omega=result.x[0],
            alpha=result.x[1],
            beta=result.x[2],
        )

        return GASResult(
            log_likelihood=-result.fun,
            method='GAS',
            copula_name=copula.name,
            success=result.success,
            nfev=result.nfev,
            message=str(result.message),
            params=params,
            scaling=self.scaling,
        )

    def log_likelihood(self, copula, u: np.ndarray,
                       result: GASResult) -> float:
        """Evaluate GAS log-likelihood."""
        p = result.params
        _, _, ll = gas_filter(
            p.omega, p.alpha, p.beta, u, copula,
            result.scaling, self.config.gas_score_eps)
        return ll

    def smoothed_params(self, copula, u: np.ndarray,
                        result: GASResult) -> np.ndarray:
        """Deterministic Psi(f_t) path."""
        p = result.params
        _, r_path, _ = gas_filter(
            p.omega, p.alpha, p.beta, u, copula,
            result.scaling, self.config.gas_score_eps)
        return r_path

    def rosenblatt_e2(self, copula, u: np.ndarray,
                      result: GASResult) -> np.ndarray:
        """e2 = h(u2, u1; Psi(f_t))."""
        r_path = self.smoothed_params(copula, u, result)
        return copula.h(u[:, 1], u[:, 0], r_path)

    def mixture_h(self, copula, u: np.ndarray,
                  result: GASResult) -> np.ndarray:
        """h along GAS-filtered path (point estimate, not mixture)."""
        p = result.params
        return gas_mixture_h(
            p.omega, p.alpha, p.beta, u, copula,
            result.scaling, self.config.gas_score_eps)

    def objective(self, copula, u: np.ndarray,
                  alpha: np.ndarray, **kwargs) -> float:
        """Minus log-likelihood: -logL(omega, alpha, beta)."""
        return gas_negloglik(
            alpha[0], alpha[1], alpha[2], u, copula,
            self.scaling, self.config.gas_score_eps)
