"""
pyscarcopula.strategy.mle — Maximum Likelihood Estimation.

Constant copula parameter (1 param). No latent process.
This is the simplest strategy and serves as a reference implementation.
"""

import numpy as np
from scipy.optimize import minimize

from pyscarcopula._types import MLEResult, NumericalConfig, DEFAULT_CONFIG
from pyscarcopula.strategy._base import register_strategy


@register_strategy('MLE')
class MLEStrategy:
    """Estimate a constant copula parameter via MLE.

    Solves: max_r  sum_t log c(u_{1t}, u_{2t}; r)
    using L-BFGS-B with bounds from the copula.
    """

    def __init__(self, config: NumericalConfig | None = None, **kwargs):
        self.config = config or DEFAULT_CONFIG

    def fit(self, copula, u: np.ndarray, **kwargs) -> MLEResult:
        """Fit constant copula parameter.

        Parameters
        ----------
        copula : CopulaProtocol
        u : (T, 2) pseudo-observations
        **kwargs : ignored (for interface compatibility)

        Returns
        -------
        MLEResult
        """
        # Starting point: transform(1.5) is a reasonable default for
        # most Archimedean copulas (parameter ~1.5 in natural domain)
        x0_val = copula.transform(np.array([1.5]))[0]
        x0 = np.array([x0_val])

        def neg_loglik(x):
            return -np.sum(copula.log_pdf(u[:, 0], u[:, 1], x))

        result = minimize(
            neg_loglik, x0,
            method='L-BFGS-B',
            bounds=copula.bounds,
            options={
                'gtol': self.config.default_tol_mle,
                'eps': 1e-5,
            },
        )

        return MLEResult(
            log_likelihood=-result.fun,
            method='MLE',
            copula_name=copula.name,
            success=result.success,
            nfev=result.nfev,
            message=str(result.message),
            copula_param=result.x[0],
        )

    def log_likelihood(self, copula, u: np.ndarray,
                       result: MLEResult) -> float:
        """sum log c(u1, u2; r_mle)."""
        r = np.full(len(u), result.copula_param)
        return float(np.sum(copula.log_pdf(u[:, 0], u[:, 1], r)))

    def smoothed_params(self, copula, u: np.ndarray,
                        result: MLEResult) -> np.ndarray:
        """Constant parameter for all time steps."""
        return np.full(len(u), result.copula_param)

    def rosenblatt_e2(self, copula, u: np.ndarray,
                      result: MLEResult) -> np.ndarray:
        """e2 = h(u2, u1; r_mle)."""
        r = np.full(len(u), result.copula_param)
        return copula.h(u[:, 1], u[:, 0], r)

    def mixture_h(self, copula, u: np.ndarray,
                  result: MLEResult) -> np.ndarray:
        """h(u2, u1; r_mle) — same as rosenblatt_e2 for MLE."""
        return self.rosenblatt_e2(copula, u, result)

    def objective(self, copula, u: np.ndarray,
                  alpha: np.ndarray, **kwargs) -> float:
        """Minus log-likelihood: -sum log c(u1, u2; alpha[0])."""
        try:
            return -float(np.sum(copula.log_pdf(u[:, 0], u[:, 1], alpha)))
        except Exception:
            return 1e10
