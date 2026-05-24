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
from pyscarcopula.strategy.predict_helpers import sample_predictive


@register_strategy('MLE')
class MLEStrategy:
    """Estimate a constant copula parameter via MLE.

    Solves: max_r  sum_t log c(u_{1t}, u_{2t}; r)
    using L-BFGS-B with bounds from the copula.
    """

    def __init__(self, config: NumericalConfig | None = None, **kwargs):
        self.config = config or DEFAULT_CONFIG

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
            **kwargs) -> MLEResult:
        """Fit constant copula parameter.

        Parameters
        ----------
        copula : CopulaProtocol
        u : (T, 2) pseudo-observations
        alpha0 : (1,) initial point in x-space, or None
        gtol, ftol, maxfun, maxiter, maxls, eps, maxcor,
        finite_diff_rel_step : L-BFGS-B options
        **kwargs : ignored (for interface compatibility)

        Returns
        -------
        MLEResult
        """
        if 'tol' in kwargs:
            raise TypeError("tol is not supported; use gtol")
        optimizer_overrides = {
            'gtol': gtol,
            'ftol': ftol,
            'maxfun': maxfun,
            'maxiter': maxiter,
            'maxls': maxls,
            'eps': eps,
            'maxcor': maxcor,
            'finite_diff_rel_step': finite_diff_rel_step,
        }

        if (
                u.ndim == 2
                and (u.shape[1] != 2 or hasattr(copula, 'log_pdf_rows'))):
            fit_mle = getattr(copula, '_fit_mle', None)
            if fit_mle is not None:
                return fit_mle(
                    u, config=self.config, **optimizer_overrides)
            direct_fit = getattr(copula, 'fit', None)
            if direct_fit is not None:
                direct_fit(u, to_pobs=False)
                result = MLEResult(
                    log_likelihood=float(copula.log_likelihood(u)),
                    method='MLE',
                    copula_name=copula.name,
                    success=True,
                    nfev=0,
                    message='direct multivariate fit',
                    copula_param=np.nan,
                )
                copula.fit_result = result
                return result

        optimizer_options = self.config.mle_optimizer.options(
            **optimizer_overrides,
        )

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
                options=optimizer_options,
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
                options=optimizer_options,
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
        if (
                u.ndim == 2
                and (u.shape[1] != 2 or hasattr(copula, 'log_pdf_rows'))):
            try:
                return float(copula.log_likelihood(u, result.copula_param))
            except TypeError:
                return float(copula.log_likelihood(u))
        r = np.full(len(u), result.copula_param)
        return float(np.sum(copula.log_pdf(u[:, 0], u[:, 1], r)))

    def predictive_mean(self, copula, u: np.ndarray,
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
            if u.ndim == 2 and u.shape[1] != 2:
                try:
                    return -float(copula.log_likelihood(u, float(alpha[0])))
                except TypeError:
                    return -float(copula.log_likelihood(u))
            return -float(np.sum(copula.log_pdf(u[:, 0], u[:, 1], alpha)))
        except Exception:
            return 1e10

    def sample(self, copula, u, result, n, rng=None, **kwargs):
        """Sample n observations with constant r = theta_mle."""
        r = np.full(n, result.copula_param)
        d = u.shape[1] if u is not None and np.ndim(u) == 2 else 2
        return sample_predictive(copula, n, r, rng=rng, d=d)

    def predict(self, copula, u, result, n, rng=None, **kwargs):
        """Predict = sample for MLE (constant parameter)."""
        r = self.predictive_params(copula, u, result, n, rng=rng, **kwargs)
        d = u.shape[1] if u is not None and np.ndim(u) == 2 else 2
        return sample_predictive(
            copula, n, r, given=kwargs.get('given'), rng=rng, d=d)

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

    def model_sample_params(self, copula, result, n, rng=None, **kwargs):
        """Constant parameter path for model reproduction."""
        return np.full(n, result.copula_param, dtype=np.float64)

    def model_sample_state(self, copula, result, **kwargs):
        return None
