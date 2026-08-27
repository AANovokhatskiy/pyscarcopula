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
    has_dynamic_scalar_parameter,
    is_multivariate_copula,
    lbfgsb_options,
    lbfgsb_overrides,
    register_strategy,
    reject_legacy_tol,
)
from pyscarcopula.strategy.predict_helpers import (
    predict_from_strategy,
    sample_predictive,
)
from pyscarcopula._native import pair as pair_native
from pyscarcopula._native import static as static_likelihood
from pyscarcopula._native.registry import registry_entry_for
from pyscarcopula.numerical._arrays import as_float64_array


def _validate_scalar_mle_parameter(copula, value, *, name):
    """Return one finite natural-space parameter inside model bounds."""
    array = np.atleast_1d(as_float64_array(value, name=name))
    if array.ndim != 1 or array.size != 1:
        raise ValueError(f"{name} must contain exactly one value")
    parameter = float(array[0])
    if not np.isfinite(parameter):
        raise ValueError(f"{name} must contain one finite value")

    lower, upper = copula.bounds[0]
    if (
            (lower is not None and parameter < lower)
            or (upper is not None and parameter > upper)):
        raise ValueError(
            f"{name}={parameter} is outside copula bounds "
            f"[{lower}, {upper}]")
    return np.array([parameter], dtype=np.float64)


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
            _prepared_evaluator=None,
            **kwargs) -> MLEResult:
        """Fit constant copula parameter.

        Parameters
        ----------
        copula : exact registered built-in copula
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
        registry_entry_for(copula)
        reject_legacy_tol(kwargs)
        optimizer_overrides = lbfgsb_overrides(
            gtol=gtol,
            ftol=ftol,
            maxfun=maxfun,
            maxiter=maxiter,
            maxls=maxls,
            eps=eps,
            maxcor=maxcor,
            finite_diff_rel_step=finite_diff_rel_step,
        )

        if not has_dynamic_scalar_parameter(copula):
            direct_fit = getattr(copula, 'fit', None)
            if direct_fit is not None:
                result = direct_fit(
                    u, to_pobs=False, config=self.config)
                if getattr(result, 'method', '').upper() == 'MLE':
                    return result

        if is_multivariate_copula(copula):
            fit_mle = getattr(copula, '_fit_mle', None)
            if fit_mle is not None:
                return fit_mle(
                    u, config=self.config, **optimizer_overrides)
            direct_fit = getattr(copula, 'fit', None)
            if direct_fit is not None:
                result = direct_fit(
                    u, to_pobs=False, config=self.config)
                if getattr(result, 'method', '').upper() == 'MLE':
                    return result

        optimizer_options = lbfgsb_options(
            self.config.mle_optimizer,
            **optimizer_overrides,
        )

        if alpha0 is not None:
            x0 = _validate_scalar_mle_parameter(
                copula, alpha0, name="alpha0")
        else:
            x0_val = copula.transform(np.array([1.5]))[0]
            x0 = _validate_scalar_mle_parameter(
                copula, x0_val, name="default MLE initial point")

        evaluator = _prepared_evaluator
        if evaluator is None:
            evaluator = static_likelihood.prepare(
                copula, u, n_threads=self.config.n_threads)

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
        fitted_parameter = _validate_scalar_mle_parameter(
            copula, result.x, name="MLE optimizer result")
        objective_value = float(result.fun)
        if not np.isfinite(objective_value):
            raise RuntimeError(
                "MLE optimizer returned a non-finite objective value")

        return MLEResult(
            log_likelihood=-objective_value,
            method='MLE',
            copula_name=copula.name,
            success=result.success,
            nfev=result.nfev,
            message=str(result.message),
            copula_param=fitted_parameter[0],
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
        registry_entry_for(copula)
        if is_multivariate_copula(copula):
            try:
                return float(copula.log_likelihood(u, result.copula_param))
            except TypeError:
                return float(copula.log_likelihood(u))
        evaluator = static_likelihood.prepare(
            copula, u, n_threads=self.config.n_threads)
        return evaluator.log_likelihood(result.copula_param)

    def predictive_mean(self, copula, u: np.ndarray,
                        result: MLEResult) -> np.ndarray:
        """Constant parameter for all time steps."""
        registry_entry_for(copula)
        return np.full(len(u), result.copula_param)

    def rosenblatt_e2(self, copula, u: np.ndarray,
                      result: MLEResult) -> np.ndarray:
        """e2 = h(u2, u1; r_mle)."""
        registry_entry_for(copula)
        r = np.full(len(u), result.copula_param)
        return pair_native.h(copula, u[:, 1], u[:, 0], r)

    def mixture_h(self, copula, u: np.ndarray,
                  result: MLEResult) -> np.ndarray:
        """h(u2, u1; r_mle) — same as rosenblatt_e2 for MLE."""
        return self.rosenblatt_e2(copula, u, result)

    def mixture_h_pair(self, copula, u: np.ndarray,
                       result: MLEResult,
                       **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """Both conditional directions at the constant MLE parameter."""
        registry_entry_for(copula)
        r = np.full(len(u), result.copula_param)
        first_given_second, second_given_first = pair_native.h_pair(
            copula, u[:, 0], u[:, 1], r)
        return second_given_first, first_given_second

    def objective(self, copula, u: np.ndarray,
                  alpha: np.ndarray, **kwargs) -> float:
        """Minus log-likelihood: -sum log c(u1, u2; alpha[0])."""
        registry_entry_for(copula)
        try:
            if is_multivariate_copula(copula):
                try:
                    return -float(copula.log_likelihood(u, float(alpha[0])))
                except TypeError:
                    return -float(copula.log_likelihood(u))
            evaluator = static_likelihood.prepare(
                copula, u, n_threads=self.config.n_threads)
            value, _ = evaluator.objective_and_gradient(
                float(alpha[0]), fail_value=self.config.fail_value)
            return value
        except Exception:
            return float(self.config.fail_value)

    def sample(self, copula, u, result, n, rng=None, **kwargs):
        """Sample n observations with constant r = theta_mle."""
        r = np.full(n, result.copula_param)
        d = copula_dimension(copula, u)
        return sample_predictive(
            copula, n, r, given=kwargs.get('given'), rng=rng, d=d)

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
