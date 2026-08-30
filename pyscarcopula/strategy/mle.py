"""
pyscarcopula.strategy.mle — Maximum Likelihood Estimation.

Constant copula parameter (1 param). No latent process.
This is the simplest strategy and serves as a reference implementation.
"""

import numpy as np
from scipy.optimize import minimize

from pyscarcopula._types import (
    MLEResult,
    MultivariateMLEResult,
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
    reject_unknown_mle_kwargs,
)
from pyscarcopula.strategy.predict_helpers import (
    predictive_params_from_state,
    sample_predictive,
    strategy_predict,
)
from pyscarcopula._native import pair as pair_native
from pyscarcopula._native import static as static_likelihood
from pyscarcopula._native import model_policy
from pyscarcopula._native.registry import registry_entry_for
from pyscarcopula.numerical._arrays import (
    as_float64_array,
    validate_float64_allocation,
)


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

    _strict_keyword_contract = True

    def __init__(self, config: NumericalConfig | None = None, **kwargs):
        reject_unknown_mle_kwargs(kwargs)
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
        **kwargs : unsupported keyword arguments are rejected

        Returns
        -------
        MLEResult
        """
        registry_entry_for(copula)
        reject_unknown_mle_kwargs(kwargs)
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
        optimizer_overrides = {
            key: value for key, value in optimizer_overrides.items()
            if value is not None
        }

        if is_multivariate_copula(copula) and alpha0 is not None:
            raise TypeError("alpha0 is not supported by multivariate MLE")

        if not has_dynamic_scalar_parameter(copula):
            direct_fit = getattr(copula, 'fit', None)
            if direct_fit is not None:
                result = direct_fit(
                    u, to_pobs=False, config=self.config,
                    **optimizer_overrides)
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
                    u, to_pobs=False, config=self.config,
                    **optimizer_overrides)
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
            x0_val = model_policy.default_pair_mle_parameter(copula)
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

        # A finite failure penalty with zero gradient can look converged to
        # L-BFGS-B. Require a genuine evaluation at the returned point, even
        # when the optimizer reports success or the numerical failure recovers.
        validated_value, _ = evaluator.validated_objective_and_gradient(
            float(fitted_parameter[0]))
        if objective_value != validated_value:
            raise FloatingPointError(
                "MLE optimizer objective does not match the final native "
                "evaluation; a numerical failure penalty cannot be fitted")

        return MLEResult(
            log_likelihood=-validated_value,
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
        if isinstance(result, MultivariateMLEResult):
            from pyscarcopula.strategy.multivariate_mle import (
                log_likelihood_from_result,
            )
            return log_likelihood_from_result(
                copula, u, result, n_threads=self.config.n_threads)
        if is_multivariate_copula(copula):
            return float(copula.log_likelihood(
                u, result.copula_param, n_threads=self.config.n_threads))
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

    def sample(self, copula, u, result, n, rng=None, **kwargs):
        """Sample n observations with constant r = theta_mle."""
        d = copula_dimension(copula, u)
        validate_float64_allocation(
            (n, d), name="MLE sample output",
            memory_budget_bytes=kwargs.get("memory_budget_bytes"))
        if isinstance(result, MultivariateMLEResult):
            from pyscarcopula.strategy.multivariate_mle import (
                sampling_model_from_result,
            )
            copula = sampling_model_from_result(copula, result)
        r = np.full(n, result.copula_param)
        return sample_predictive(
            copula, n, r, given=kwargs.get('given'), rng=rng, d=d,
            n_threads=kwargs.get("n_threads", 1),
            memory_budget_bytes=kwargs.get("memory_budget_bytes"))

    def predict(self, copula, u, result, n, rng=None, **kwargs):
        """Predict from the supplied static result without refitting its model."""
        if isinstance(result, MultivariateMLEResult):
            return self.sample(copula, u, result, n, rng=rng, **kwargs)
        return strategy_predict(self, copula, u, result, n, rng=rng, **kwargs)

    predictive_params = predictive_params_from_state

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

    def model_sample_params_batches(
            self, copula, result, n, *, batch_rows, rng=None):
        """Materialize only one block of the constant parameter path."""
        for start in range(0, n, batch_rows):
            yield self.model_sample_params(
                copula, result, min(batch_rows, n - start), rng=rng)
