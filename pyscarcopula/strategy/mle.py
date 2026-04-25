"""
pyscarcopula.strategy.mle — Maximum Likelihood Estimation.

Constant copula parameter (1 param). No latent process.
This is the simplest strategy and serves as a reference implementation.
"""

import numpy as np
from scipy.optimize import minimize

from pyscarcopula._types import (
    MLEResult,
    NumericalConfig,
    DEFAULT_CONFIG,
    PredictiveState,
)
from pyscarcopula.strategy._base import register_strategy
from pyscarcopula.strategy.predict_helpers import conditional_sample_bivariate


@register_strategy('MLE')
class MLEStrategy:
    """Estimate a constant copula parameter via MLE.

    Solves: max_r  sum_t log c(u_{1t}, u_{2t}; r)
    using L-BFGS-B with bounds from the copula.
    """

    def __init__(self, config: NumericalConfig | None = None, **kwargs):
        self.config = config or DEFAULT_CONFIG

    def fit(self, copula, u: np.ndarray,
            alpha0: np.ndarray | None = None, **kwargs) -> MLEResult:
        """Fit constant copula parameter.

        Parameters
        ----------
        copula : CopulaProtocol
        u : (T, 2) pseudo-observations
        alpha0 : (1,) initial point in x-space, or None
        **kwargs : ignored (for interface compatibility)

        Returns
        -------
        MLEResult
        """
        if alpha0 is not None:
            x0 = np.atleast_1d(np.asarray(alpha0, dtype=np.float64))[:1]
        else:
            x0_val = copula.transform(np.array([1.5]))[0]
            x0 = np.array([x0_val])

        # Pre-extract columns to avoid repeated slicing
        u1 = u[:, 0]
        u2 = u[:, 1]

        # Use fused kernel if copula provides one (avoids redundant
        # Phi^{-1} recomputation for Gaussian copula).
        # With jac=True, minimize calls fun once and gets both f and g.
        fused = getattr(copula, 'mle_objective_fused', None)
        if fused is not None:
            obj_and_grad = fused(u)
            result = minimize(
                obj_and_grad, x0,
                jac=True,
                method='L-BFGS-B',
                bounds=copula.bounds,
                options={'gtol': self.config.default_tol_mle},
            )
        else:
            def neg_loglik(x):
                return -np.sum(copula.log_pdf(u1, u2, x))

            def neg_loglik_grad(x):
                return -np.sum(copula.dlog_pdf_dr(u1, u2, x))

            result = minimize(
                neg_loglik, x0,
                jac=neg_loglik_grad,
                method='L-BFGS-B',
                bounds=copula.bounds,
                options={'gtol': self.config.default_tol_mle},
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

    def sample(self, copula, u, result, n, rng=None, **kwargs):
        """Sample n observations with constant r = theta_mle."""
        r = np.full(n, result.copula_param)
        return copula.sample(n, r, rng=rng)

    def predict(self, copula, u, result, n, rng=None, **kwargs):
        """Predict = sample for MLE (constant parameter)."""
        r = self.predictive_params(copula, u, result, n, rng=rng, **kwargs)
        return conditional_sample_bivariate(
            copula, n, r, given=kwargs.get('given'), rng=rng)

    def predictive_params(self, copula, u, result, n, rng=None, **kwargs):
        """Constant predictive parameter for MLE."""
        state = self.predictive_state(copula, u, result, **kwargs)
        return self.sample_params(copula, state, n, rng=rng, **kwargs)

    def predictive_state(self, copula, u, result, **kwargs):
        horizon = str(kwargs.get('horizon', 'next')).lower()
        return PredictiveState(
            method='MLE',
            horizon=horizon,
            kind='point',
            r=np.array([result.copula_param], dtype=np.float64),
        )

    def condition_state(self, copula, state, observation, result, **kwargs):
        return state

    def sample_params(self, copula, state, n, rng=None, **kwargs):
        return np.full(n, float(np.asarray(state.r)[0]), dtype=np.float64)
