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
from pyscarcopula.strategy._base import (
    copula_dimension,
    is_multivariate_copula,
    register_strategy,
)
from pyscarcopula.strategy.predict_helpers import (
    predict_from_strategy,
    sample_predictive,
)
from pyscarcopula.numerical import static_likelihood


@register_strategy('MLE')
class MLEStrategy:
    """Estimate a constant copula parameter via MLE.

    Solves ``max_r sum_t log c(u_{1t}, u_{2t}; r)`` directly in the
    copula's natural parameter ``r``. Latent-state transforms are not part of
    the MLE objective.
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
        alpha0 : (1,) array_like or None
            Initial point in the copula's natural parameter space. When
            omitted, ``copula.transform([1.5])`` is used only to construct a
            common valid natural starting value across copula families; the
            optimizer still evaluates the likelihood directly at that value.
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

        if is_multivariate_copula(copula):
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

        evaluator = static_likelihood.prepare(copula, u)

        def objective_and_gradient(x):
            return evaluator.objective_and_gradient(
                float(x[0]), fail_value=self.config.fail_value)

        result = minimize(
            objective_and_gradient, x0,
            jac=True,
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
            diagnostics={
                'model_score': 'not_applicable',
                'optimizer_gradient': 'analytical',
                'gradient_kind': 'analytical',
                'setup_derivative': 'not_applicable',
                'filter_derivative': 'not_applicable',
                'parameter_gradient': 'analytical',
            },
        )

    def log_likelihood(self, copula, u: np.ndarray,
                       result: MLEResult) -> float:
        """sum log c(u1, u2; r_mle)."""
        if is_multivariate_copula(copula):
            try:
                return float(copula.log_likelihood(u, result.copula_param))
            except TypeError:
                return float(copula.log_likelihood(u))
        evaluator = static_likelihood.prepare(copula, u)
        return evaluator.log_likelihood(result.copula_param)

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
            if is_multivariate_copula(copula):
                try:
                    return -float(copula.log_likelihood(u, float(alpha[0])))
                except TypeError:
                    return -float(copula.log_likelihood(u))
            evaluator = static_likelihood.prepare(copula, u)
            value, _ = evaluator.objective_and_gradient(
                float(alpha[0]), fail_value=self.config.fail_value)
            return value
        except Exception:
            return float(self.config.fail_value)

    def sample(self, copula, u, result, n, rng=None, **kwargs):
        """Sample n observations with constant r = theta_mle."""
        r = np.full(n, result.copula_param)
        d = copula_dimension(copula, u)
        return sample_predictive(copula, n, r, rng=rng, d=d)

    def predict(self, copula, u, result, n, rng=None, **kwargs):
        """Predict = sample for MLE (constant parameter)."""
        return predict_from_strategy(
            self, copula, u, result, n, rng=rng, **kwargs)

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
