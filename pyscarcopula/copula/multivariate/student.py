"""Static multivariate Student-t copula."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import (
    multivariate_t,
    t as t_dist,
)

from pyscarcopula._utils import pobs
from pyscarcopula._types import (
    DEFAULT_CONFIG,
    MultivariateMLEResult,
    NumericalConfig,
)
from pyscarcopula.copula.base import CopulaCapabilities
from pyscarcopula.copula.multivariate.base import MultivariateCopula
from pyscarcopula.copula.multivariate.corr_param import (
    estimate_kendall_correlation,
)
from pyscarcopula.copula.multivariate.correlation_policy import (
    CorrelationEstimator,
    CorrelationMode,
    CorrelationPolicy,
    FactorEstimation,
    normalize_correlation_mode,
)
from pyscarcopula.strategy.multivariate_mle import (
    StaticMLEEvaluation,
    StaticMLEProblem,
    run_static_multivariate_mle,
)


_LBFGSB_FIT_KEYS = (
    "gtol",
    "ftol",
    "maxfun",
    "maxiter",
    "maxls",
    "eps",
    "maxcor",
    "finite_diff_rel_step",
)


def _validate_student_fit_data(u):
    if u.ndim != 2:
        raise ValueError("data must have shape (n_observations, dimension)")
    if u.shape[0] == 0:
        raise ValueError("data must contain at least one observation")
    if u.shape[1] < 2:
        raise ValueError("data must contain at least two variables")
    if not np.all(np.isfinite(u)):
        raise ValueError("data must contain only finite values")


class StudentCopula(MultivariateCopula):
    """d-dimensional Student-t copula with fitted shape and degrees of freedom.

    Static MLE optimizes ``df`` directly in natural degrees-of-freedom units;
    no latent-state transform is used.
    """

    _capabilities = CopulaCapabilities(
        supports_conditional_sampling=True,
    )

    def __init__(self, *, corr_mode: CorrelationMode = "fixed") -> None:
        corr_mode = normalize_correlation_mode(corr_mode)
        if corr_mode != "fixed":
            raise NotImplementedError(
                f"StudentCopula corr_mode={corr_mode!r} will be enabled "
                "by the shared static MLE fitter")
        super().__init__(name="Student-t copula")
        self._corr_mode = corr_mode
        self.shape = None
        self.df = None
        self.correlation_preprocessing = None

    @property
    def corr_mode(self) -> CorrelationMode:
        return self._corr_mode

    @property
    def corr_estimator_(self) -> CorrelationEstimator:
        return "kendall_plugin"

    @property
    def factor_estimation(self) -> FactorEstimation | None:
        return None

    @property
    def correlation_policy_(self) -> CorrelationPolicy:
        """Return the immutable policy represented by current model state."""
        if self.dimension is None or self.correlation_preprocessing is None:
            raise ValueError("correlation policy requires a fitted model")
        return CorrelationPolicy.create(
            mode=self._corr_mode,
            estimator=self.corr_estimator_,
            dimension=self.dimension,
            preprocessing=self.correlation_preprocessing,
        )

    def fit(
            self,
            data: ArrayLike,
            to_pobs: bool = False,
            method: str = 'mle',
            **kwargs: Any) -> MultivariateMLEResult:
        if str(method).upper() != 'MLE':
            raise ValueError(
                f"StudentCopula supports only method='mle', "
                f"got {method!r}")
        u = np.asarray(data, dtype=np.float64)
        _validate_student_fit_data(u)
        if to_pobs:
            u = pobs(u)
            _validate_student_fit_data(u)
        config = kwargs.pop("config", None)
        if "tol" in kwargs:
            raise TypeError("tol is not supported; use gtol")
        optimizer_kwargs = {
            key: kwargs.pop(key)
            for key in _LBFGSB_FIT_KEYS
            if key in kwargs
        }
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(
                f"unexpected MLE keyword argument(s): {unexpected}")
        return self._fit_mle(u, config=config, **optimizer_kwargs)

    def _fit_mle(
            self,
            u: np.ndarray,
            config: NumericalConfig | None = None,
            **optimizer_overrides: float | int | None,
            ) -> MultivariateMLEResult:
        """Fit static df through the shared, atomic MLE workflow."""
        from pyscarcopula.numerical import static_likelihood

        config = config or DEFAULT_CONFIG
        options = config.static_student_optimizer.options(
            **optimizer_overrides)
        d = u.shape[1]
        preprocessing = estimate_kendall_correlation(u)
        correlation = preprocessing.correlation.copy()
        policy = CorrelationPolicy.create(
            mode="fixed",
            estimator="kendall_plugin",
            dimension=d,
            preprocessing=preprocessing,
        )
        evaluator = static_likelihood.prepare_student(
            correlation, u, n_threads=config.n_threads)

        def evaluate(parameters: np.ndarray) -> StaticMLEEvaluation:
            value, gradient = evaluator.objective_and_gradient(
                float(parameters[0]), fail_value=config.fail_value)
            return StaticMLEEvaluation(
                objective=value,
                gradient=gradient,
                correlation=correlation,
                state={"df": float(parameters[0])},
            )

        outcome = run_static_multivariate_mle(
            StaticMLEProblem(
                family="student",
                initial_parameters=np.array(
                    [max(float(d), 5.0)], dtype=np.float64),
                bounds=((2.001, None),),
                evaluate=evaluate,
            ),
            optimizer_options=options,
            fail_value=config.fail_value,
        )
        df_hat = float(outcome.parameters[0])
        parameter_count = d * (d - 1) // 2 + 1
        fit_result = MultivariateMLEResult(
            log_likelihood=-float(outcome.final_objective),
            method="MLE",
            copula_name=self.name,
            success=outcome.accepted,
            nfev=outcome.nfev,
            message=outcome.message,
            copula_param=df_hat,
            parameter_count=parameter_count,
            n_observations=len(u),
            model_parameters={
                "df": df_hat,
                "corr_mode": self._corr_mode,
                "corr_estimator": self.corr_estimator_,
                "correlation_matrix": correlation.copy(),
            },
            correlation_matrix=correlation.copy(),
            diagnostics={
                "model_score": "not_applicable",
                "optimizer_gradient": "analytical",
                "gradient_kind": "analytical",
                "setup_derivative": "not_applicable",
                "filter_derivative": "not_applicable",
                "df_gradient": "analytical",
                "corr_matrix": correlation.copy(),
                "n_threads": config.n_threads,
                **policy.diagnostics(),
                **outcome.diagnostics(),
            },
        )
        if outcome.accepted:
            self._set_dimension(d, allow_change=True)
            self.correlation_preprocessing = preprocessing
            self.shape = correlation.copy()
            self.df = df_hat
            self.fit_result = fit_result
            self._last_u = u
        return fit_result

    def _nll_with_params(self, u, R, df):
        from pyscarcopula.numerical import static_likelihood
        from pyscarcopula.numerical._cpp_extension import CppError
        try:
            value, _ = static_likelihood.prepare_student(
                R, u).objective_and_gradient(df)
            return value
        except (FloatingPointError, OverflowError, ValueError,
                np.linalg.LinAlgError, CppError):
            return 1e10

    def _nll(self, u):
        return self._nll_with_params(u, self.shape, self.df)

    def log_pdf_rows(self, u, parameter=None, **kwargs):
        from pyscarcopula.numerical import static_likelihood
        df = self.df if parameter is None else float(parameter)
        return static_likelihood.prepare(self, u).log_pdf_rows(df)

    def log_likelihood(self, u):
        from pyscarcopula.numerical import static_likelihood
        return static_likelihood.prepare(self, u).log_likelihood(self.df)

    def _fitted_parameters(self):
        result = self.fit_result
        if isinstance(result, MultivariateMLEResult):
            return result.correlation_matrix, float(result.copula_param)
        return self.shape, self.df

    def sample(self, n, u=None, rng=None):
        correlation, df = self._fitted_parameters()
        if correlation is None or df is None:
            raise ValueError("Fit first")
        if rng is None:
            rng = np.random.default_rng()

        d = correlation.shape[0]
        x = multivariate_t.rvs(
            loc=np.zeros(d),
            shape=correlation,
            df=df,
            size=n,
            random_state=rng,
        )
        return t_dist.cdf(x, df=df)

    def sample_conditional(self, n, given, rng=None, *, n_threads=1):
        """Sample conditionally with ``given={var_index: u_value}``."""
        correlation, df = self._fitted_parameters()
        if correlation is None or df is None:
            raise ValueError("Fit first")
        from pyscarcopula.copula.multivariate.conditional import (
            sample_student_conditional,
        )
        return sample_student_conditional(
            n, correlation, df, given=given, rng=rng,
            n_threads=n_threads)

    def predict(self, n, u=None, rng=None, given=None, horizon='next',
                predictive_r_mode=None, predict_config=None):
        if predict_config is not None:
            from pyscarcopula.api import _resolve_predict_config
            config = _resolve_predict_config(
                predict_config, given, horizon, {
                    "predictive_r_mode": predictive_r_mode,
                })
            given = config.given
        if given:
            return self.sample_conditional(n, given=given, rng=rng)
        return self.sample(n, u=u, rng=rng)
