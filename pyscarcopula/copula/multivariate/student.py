"""Static multivariate Student-t copula."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from pyscarcopula._types import (
    DEFAULT_CONFIG,
    MultivariateMLEResult,
    NumericalConfig,
)
from pyscarcopula._native import model_policy
from pyscarcopula._native import validation as native_validation
from pyscarcopula._utils import pobs
from pyscarcopula.numerical._arrays import (
    as_float64_array,
    as_float64_scalar,
    validate_integer,
    validate_sampling_n_threads as _validated_n_threads,
)
from pyscarcopula.copula.multivariate.base import (
    MultivariateCopula,
    as_real_array,
    factor_copula_getstate,
    model_state_locked,
)
from pyscarcopula.copula.multivariate.corr_param import (
    estimate_kendall_correlation,
    preprocess_correlation_matrix,
    sigmoid,
    validate_corr_matrix,
)
from pyscarcopula.copula.multivariate.correlation_policy import (
    CorrelationEstimator,
    CorrelationMode,
    CorrelationPolicy,
    FactorEstimation,
    factor_parameter_count,
    normalize_correlation_mode,
    normalize_factor_estimation,
    restore_correlation_result_metadata,
    validate_joint_factor_rank,
)
from pyscarcopula.copula.multivariate.factor_correlation import (
    FactorCorrelation,
    PreparedFactorCorrelation,
    _validate_dense_materialization,
)
from pyscarcopula.copula.multivariate.factor_estimation import (
    FactorLoadingParameterization,
    estimate_factor_loadings,
)
from pyscarcopula.copula.multivariate.factor_student import FactorStudentEvaluator
from pyscarcopula.strategy.multivariate_mle import (
    StaticMLEEvaluation,
    StaticMLEProblem,
    make_student_static_mle_evaluator,
    run_static_multivariate_mle,
)


_LBFGSB_FIT_KEYS = (
    "gtol", "ftol", "maxfun", "maxiter", "maxls", "eps", "maxcor",
    "finite_diff_rel_step",
)


def _integer(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _validate_student_fit_data(u: np.ndarray) -> None:
    if u.ndim != 2:
        raise ValueError("data must have shape (n_observations, dimension)")
    if u.shape[0] == 0:
        raise ValueError("data must contain at least one observation")
    if u.shape[1] < 2:
        raise ValueError("data must contain at least two variables")
    native_validation.validate_fit_data(u, "Student")


class StudentCopula(MultivariateCopula):
    """Static Student-t copula with configurable correlation estimation."""

    def __init__(
            self,
            d: int | None = None,
            R: ArrayLike | None = None,
            *,
            corr_mode: CorrelationMode = "fixed",
            corr_base: ArrayLike | None = None,
            corr_shrinkage_init: float = 0.8,
            cholesky_d_max: int = 10,
            allow_large_cholesky: bool = False,
            factor_rank: int | None = None,
            factor_loadings: ArrayLike | None = None,
            factor_estimation: FactorEstimation = "two-stage",
            factor_tile_size: int = 16384,
            factor_uniqueness_min: float = 1e-8,
            factor_joint_max_params: int = 100000,
            factor_joint_penalty: float = 1e-6,
            factor_joint_condition_max: float = 1e12,
            factor_seed: int = 0,
            factor_oversampling: int = 8) -> None:
        mode = normalize_correlation_mode(corr_mode)
        estimation = normalize_factor_estimation(factor_estimation)
        if R is not None and d is None:
            matrix = np.asarray(R)
            if matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
                d = int(matrix.shape[0])
        if factor_loadings is not None and d is None:
            loadings_array = np.asarray(factor_loadings)
            if loadings_array.ndim == 2:
                d = int(loadings_array.shape[0])
        super().__init__(dimension=d, name="Student-t copula")

        if mode == "fixed" and corr_base is not None:
            raise ValueError("corr_base is only valid for estimated corr modes")
        if mode == "factor" and (R is not None or corr_base is not None):
            raise ValueError("R and corr_base are forbidden in factor mode")
        if mode != "factor" and (
                factor_rank is not None or factor_loadings is not None):
            raise ValueError(
                "factor_rank and factor_loadings require corr_mode='factor'")
        if mode != "factor" and estimation != "two-stage":
            raise ValueError(
                "factor_estimation is only configurable in factor mode")
        corr_shrinkage_init = as_float64_scalar(
            corr_shrinkage_init, name="corr_shrinkage_init")
        if not 0.0 < corr_shrinkage_init < 1.0:
            raise ValueError("corr_shrinkage_init must be in (0, 1)")
        cholesky_d_max = _integer(
            "cholesky_d_max", cholesky_d_max, minimum=2)
        if (
                mode == "cholesky" and d is not None
                and d > cholesky_d_max and not allow_large_cholesky):
            raise ValueError(
                f"corr_mode='cholesky' is limited to d <= "
                f"{cholesky_d_max} by default")

        self._corr_mode = mode
        self._factor_estimation = estimation
        self._corr_shrinkage_init = corr_shrinkage_init
        self._cholesky_d_max = cholesky_d_max
        self._allow_large_cholesky = bool(allow_large_cholesky)
        self._correlation: np.ndarray | None = None
        self._supplied_preprocessing = (
            None if R is None else preprocess_correlation_matrix(
                as_float64_array(R, name="R"), source="supplied"))
        self._supplied_correlation = (
            None if self._supplied_preprocessing is None
            else self._supplied_preprocessing.correlation.copy())
        self._base_preprocessing = (
            None if corr_base is None else preprocess_correlation_matrix(
                as_float64_array(corr_base, name="corr_base"),
                source="corr_base"))
        self._corr_base = (
            None if self._base_preprocessing is None
            else self._base_preprocessing.correlation.copy())
        self._constructor_R = (
            None if self._supplied_preprocessing is None
            else self._supplied_preprocessing.input_correlation.copy())
        self._constructor_corr_base = (
            None if self._base_preprocessing is None
            else self._base_preprocessing.input_correlation.copy())
        for matrix, name in (
                (self._supplied_correlation, "R"),
                (self._corr_base, "corr_base")):
            if matrix is not None and self.dimension is not None and matrix.shape != (
                    self.dimension, self.dimension):
                raise ValueError(
                    f"{name} must have shape ({self.dimension}, "
                    f"{self.dimension})")

        self._factor_rank = None
        self._factor_loadings: np.ndarray | None = None
        self._factor_correlation: FactorCorrelation | None = None
        self._factor_operator = None
        self._factor_initialization_diagnostics: dict[str, object] = {}
        self._constructor_factor_loadings: np.ndarray | None = None
        self._factor_tile_size = _integer(
            "factor_tile_size", factor_tile_size, minimum=1)
        self._factor_uniqueness_min = as_float64_scalar(
            factor_uniqueness_min, name="factor_uniqueness_min")
        self._factor_joint_max_params = _integer(
            "factor_joint_max_params", factor_joint_max_params, minimum=1)
        self._factor_joint_penalty = as_float64_scalar(
            factor_joint_penalty, name="factor_joint_penalty")
        self._factor_joint_condition_max = as_float64_scalar(
            factor_joint_condition_max, name="factor_joint_condition_max")
        self._factor_seed = _integer("factor_seed", factor_seed)
        self._factor_oversampling = _integer(
            "factor_oversampling", factor_oversampling)
        if not (0.0 < self._factor_uniqueness_min < 1.0):
            raise ValueError("factor_uniqueness_min must be in (0, 1)")
        if not np.isfinite(self._factor_joint_penalty) or self._factor_joint_penalty < 0:
            raise ValueError("factor_joint_penalty must be finite and non-negative")
        if not np.isfinite(self._factor_joint_condition_max) or self._factor_joint_condition_max <= 1:
            raise ValueError("factor_joint_condition_max must be finite and > 1")
        if mode == "factor":
            if self.dimension is None:
                raise ValueError("d is required in factor mode")
            self._factor_rank = _integer("factor_rank", factor_rank, minimum=1)
            if self._factor_rank >= self.dimension:
                raise ValueError("factor_rank must satisfy 1 <= k < d")
            if estimation == "joint":
                validate_joint_factor_rank(self.dimension, self._factor_rank)
            expected = factor_parameter_count(self.dimension, self._factor_rank)
            if estimation == "joint" and expected > self._factor_joint_max_params:
                raise ValueError(
                    "joint factor estimation exceeds factor_joint_max_params")
            if factor_loadings is not None:
                self._set_factor_loadings(
                    factor_loadings, diagnostics={"source": "supplied"})
                self._constructor_factor_loadings = (
                    self._factor_loadings.copy())

        self.df: float | None = None
        self.correlation_preprocessing = None
        self._corr_estimator: CorrelationEstimator | None = None
        self._corr_params_raw = np.empty(0, dtype=np.float64)
        self._corr_alpha: float | None = None

    @property
    def shape(self) -> np.ndarray | None:
        """Compatibility alias for the fitted correlation matrix."""
        return None if self._correlation is None else self._correlation.copy()

    @shape.setter
    @model_state_locked
    def shape(self, value: ArrayLike | None) -> None:
        if value is None:
            self._correlation = None
            return
        correlation = as_float64_array(value, name="shape").copy()
        if correlation.ndim != 2 or correlation.shape[0] != correlation.shape[1]:
            raise ValueError("shape must be a square correlation matrix")
        self._validate_dimension_value(correlation.shape[0])
        dimension = self.dimension
        if dimension is not None and correlation.shape != (dimension, dimension):
            raise ValueError(f"shape must have shape ({dimension}, {dimension})")
        validate_corr_matrix(correlation)
        self._correlation = correlation

    @property
    def correlation(self) -> np.ndarray | None:
        return self.shape

    @property
    def corr_mode(self) -> CorrelationMode:
        return self._corr_mode

    @property
    def corr_estimator_(self) -> CorrelationEstimator:
        if self._corr_estimator is not None:
            return self._corr_estimator
        if self._corr_mode == "factor":
            return "factor_joint" if self._factor_estimation == "joint" else "factor_two_stage"
        if self._corr_mode in {"shrinkage", "cholesky"}:
            return "joint_mle"
        return "supplied" if self._supplied_correlation is not None else "kendall_plugin"

    @property
    def factor_estimation(self) -> FactorEstimation | None:
        return self._factor_estimation if self._corr_mode == "factor" else None

    @property
    def factor_rank(self) -> int | None:
        return self._factor_rank

    @property
    def factor_loadings_(self) -> np.ndarray | None:
        return None if self._factor_loadings is None else self._factor_loadings.copy()

    @property
    def factor_uniqueness_(self) -> np.ndarray | None:
        return None if self._factor_correlation is None else self._factor_correlation.uniqueness.copy()

    @property
    def correlation_operator_(self) -> PreparedFactorCorrelation:
        if self._corr_mode != "factor":
            raise AttributeError("correlation_operator_ is only available in factor mode")
        if self._factor_operator is None:
            raise ValueError("factor correlation is not initialized; call fit()")
        return self._factor_operator

    __getstate__ = factor_copula_getstate

    def __setstate__(self, state: dict[str, object]) -> None:
        super().__setstate__(state)
        legacy_state = "_corr_mode" not in state
        self._corr_mode = normalize_correlation_mode(
            str(getattr(self, "_corr_mode", "fixed")),
            allow_dense_alias=True,
            warn_on_dense=False,
        )
        self._factor_estimation = normalize_factor_estimation(
            str(getattr(self, "_factor_estimation", "two-stage")))
        self._correlation = getattr(self, "_correlation", None)
        if self._correlation is None:
            legacy_shape = state.get("shape", state.get("_shape"))
            if legacy_shape is not None:
                self._correlation = np.asarray(
                    legacy_shape, dtype=np.float64).copy()
        self.__dict__.pop("shape", None)
        self.__dict__.pop("_shape", None)
        self._supplied_correlation = getattr(
            self, "_supplied_correlation", None)
        self._corr_base = getattr(self, "_corr_base", None)
        self._constructor_R = getattr(self, "_constructor_R", None)
        self._constructor_corr_base = getattr(
            self, "_constructor_corr_base", None)
        self._supplied_preprocessing = getattr(
            self, "_supplied_preprocessing", None)
        self._base_preprocessing = getattr(
            self, "_base_preprocessing", None)
        self._corr_shrinkage_init = float(getattr(
            self, "_corr_shrinkage_init", 0.8))
        self._corr_params_raw = np.asarray(getattr(
            self, "_corr_params_raw", np.empty(0)),
            dtype=np.float64).reshape(-1).copy()
        self._corr_alpha = getattr(self, "_corr_alpha", None)
        self._cholesky_d_max = int(getattr(
            self, "_cholesky_d_max", 10))
        self._allow_large_cholesky = bool(getattr(
            self, "_allow_large_cholesky", False))
        self._factor_rank = getattr(self, "_factor_rank", None)
        self._factor_tile_size = int(getattr(
            self, "_factor_tile_size", 16384))
        self._factor_uniqueness_min = float(getattr(
            self, "_factor_uniqueness_min", 1e-8))
        self._factor_joint_max_params = int(getattr(
            self, "_factor_joint_max_params", 100000))
        self._factor_joint_penalty = float(getattr(
            self, "_factor_joint_penalty", 1e-6))
        self._factor_joint_condition_max = float(getattr(
            self, "_factor_joint_condition_max", 1e12))
        self._factor_seed = int(getattr(self, "_factor_seed", 0))
        self._factor_oversampling = int(getattr(
            self, "_factor_oversampling", 8))
        self._factor_loadings = getattr(self, "_factor_loadings", None)
        self._constructor_factor_loadings = getattr(
            self, "_constructor_factor_loadings", None)
        self._factor_initialization_diagnostics = dict(getattr(
            self, "_factor_initialization_diagnostics", {}))
        self.correlation_preprocessing = getattr(
            self, "correlation_preprocessing", None)
        self.df = getattr(self, "df", None)
        self._corr_estimator = getattr(self, "_corr_estimator", None)
        if legacy_state and self._correlation is not None:
            self._corr_estimator = "kendall_plugin"
        self._factor_correlation = None
        self._factor_operator = None
        if (
                getattr(self, "_corr_mode", "fixed") == "factor"
                and getattr(self, "_factor_loadings", None) is not None):
            self._set_factor_loadings(
                self._factor_loadings,
                diagnostics=getattr(
                    self, "_factor_initialization_diagnostics", {}),
            )
        result = getattr(self, "fit_result", None)
        if result is not None and self.dimension is not None:
            restore_correlation_result_metadata(
                result,
                self.correlation_policy_,
                raw_parameters=self._corr_params_raw,
                alpha=self._corr_alpha,
            )
            result_diagnostics = getattr(result, "diagnostics", None)
            if legacy_state and isinstance(result_diagnostics, dict):
                result_diagnostics.setdefault(
                    "corr_policy_migration",
                    "legacy_fixed_kendall_plugin",
                )

    def to_correlation_matrix(
            self, *, max_dimension: int = 2048,
            memory_budget_bytes: int | None = None) -> np.ndarray:
        if self._corr_mode == "factor":
            return self.correlation_operator_.to_dense(
                max_dimension=max_dimension,
                memory_budget_bytes=memory_budget_bytes)
        if self._correlation is None:
            raise ValueError("Fit first")
        _validate_dense_materialization(
            self._correlation.shape[0], max_dimension=max_dimension,
            memory_budget_bytes=memory_budget_bytes)
        return self._correlation.copy()

    def _set_factor_loadings(
            self, loadings: ArrayLike, *,
            diagnostics: dict[str, object] | None = None) -> None:
        array = as_float64_array(loadings, name="factor_loadings")
        expected = (self.dimension, self._factor_rank)
        if array.shape != expected:
            raise ValueError(f"factor_loadings must have shape {expected}")
        factor = FactorCorrelation(
            array, uniqueness_min=self._factor_uniqueness_min,
            diagnostics={} if diagnostics is None else diagnostics)
        self._factor_loadings = factor.loadings.copy()
        self._factor_correlation = factor
        self._factor_operator = factor.prepare()
        self._factor_initialization_diagnostics = dict(factor.diagnostics)

    @property
    def correlation_policy_(self) -> CorrelationPolicy:
        if self.dimension is None:
            raise ValueError("correlation policy requires a known dimension")
        return CorrelationPolicy.create(
            mode=self._corr_mode,
            estimator=self.corr_estimator_,
            dimension=self.dimension,
            supplied_correlation=(
                self._correlation if self.corr_estimator_ == "supplied" else None),
            base_correlation=(
                self._corr_base if self._corr_mode in {"shrinkage", "cholesky"} else None),
            preprocessing=self.correlation_preprocessing,
            factor_rank=self._factor_rank,
            factor_estimation=self.factor_estimation,
            shrinkage_initial=self._corr_shrinkage_init,
            initialization_source=(
                self._factor_initialization_diagnostics.get("source")
                if self._corr_mode == "factor"
                else None),
        )

    @model_state_locked
    def fit(
            self,
            data: ArrayLike,
            to_pobs: bool = False,
            method: str = "mle",
            config: NumericalConfig | None = None,
            **kwargs: Any) -> MultivariateMLEResult:
        if str(method).upper() != "MLE":
            raise ValueError(
                f"StudentCopula supports only method='mle', got {method!r}")
        u = as_real_array(data)
        if to_pobs:
            if u.ndim != 2 or u.shape[0] == 0 or u.shape[1] < 2 or not np.all(np.isfinite(u)):
                raise ValueError("data must be a finite non-empty 2D array")
            u = pobs(u)
        _validate_student_fit_data(u)
        if self.dimension is not None and u.shape[1] != self.dimension:
            raise ValueError(
                f"data must have {self.dimension} columns, got {u.shape[1]}")
        if (
                self._corr_mode == "cholesky"
                and u.shape[1] > self._cholesky_d_max
                and not self._allow_large_cholesky):
            raise ValueError(
                f"corr_mode='cholesky' is limited to d <= "
                f"{self._cholesky_d_max} by default")
        if "tol" in kwargs:
            raise TypeError("tol is not supported; use gtol")
        optimizer_kwargs = {
            key: kwargs.pop(key) for key in _LBFGSB_FIT_KEYS if key in kwargs}
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"unexpected MLE keyword argument(s): {unexpected}")
        return self._fit_mle(
            u, config=config or DEFAULT_CONFIG,
            optimizer_overrides=optimizer_kwargs)

    def _initial_dense_correlation(self, u: np.ndarray):
        if self._constructor_corr_base is not None:
            return self._base_preprocessing.correlation.copy(), self._base_preprocessing
        if self._supplied_correlation is not None:
            return self._supplied_correlation.copy(), self._supplied_preprocessing
        preprocessing = estimate_kendall_correlation(u, eps=1e-8)
        return preprocessing.correlation.copy(), preprocessing

    def _fit_mle(
            self, u: np.ndarray, *, config: NumericalConfig,
            optimizer_overrides: dict[str, float | int | None],
            ) -> MultivariateMLEResult:
        options = config.static_student_optimizer.options(**optimizer_overrides)
        if self._corr_mode == "factor":
            return self._fit_factor(u, config, options)

        d = u.shape[1]
        df_initial, df_bounds = model_policy.student_fit_policy(
            d, stochastic=False)
        initial_correlation, preprocessing = self._initial_dense_correlation(u)
        estimator: CorrelationEstimator = (
            "joint_mle" if self._corr_mode in {"shrinkage", "cholesky"}
            else ("supplied" if self._supplied_correlation is not None else "kendall_plugin"))
        policy = CorrelationPolicy.create(
            mode=self._corr_mode,
            estimator=estimator,
            dimension=d,
            supplied_correlation=(initial_correlation if estimator == "supplied" else None),
            base_correlation=(initial_correlation if self._corr_mode in {"shrinkage", "cholesky"} else None),
            preprocessing=preprocessing,
            shrinkage_initial=self._corr_shrinkage_init,
        )
        corr0 = policy.initial_raw_parameters()
        n_corr = policy.optimized_n_params
        evaluate = make_student_static_mle_evaluator(
            initial_correlation,
            policy,
            u,
            n_threads=config.n_threads,
            fail_value=config.fail_value,
        )

        outcome = run_static_multivariate_mle(
            StaticMLEProblem(
                family="student",
                initial_parameters=np.concatenate((
                    np.array([df_initial]), corr0)),
                bounds=(df_bounds,)
                + ((None, None),) * n_corr,
                evaluate=evaluate,
            ),
            optimizer_options=options,
            fail_value=config.fail_value,
        )
        correlation = (
            initial_correlation.copy() if outcome.evaluation is None
            else np.asarray(outcome.evaluation.correlation).copy())
        raw = outcome.parameters[1:].copy()
        alpha = float(sigmoid(raw[0])) if self._corr_mode == "shrinkage" and raw.size else None
        return self._make_result_and_commit(
            u=u, config=config, outcome=outcome, policy=policy,
            correlation=correlation, preprocessing=preprocessing,
            raw=raw, alpha=alpha, factor=None,
            initialization=None)

    def _fit_factor(self, u, config, options):
        df_initial, df_bounds = model_policy.student_fit_policy(
            u.shape[1], stochastic=False)
        if self._constructor_factor_loadings is None:
            loadings, initialization = estimate_factor_loadings(
                u, self._factor_rank,
                uniqueness_min=self._factor_uniqueness_min,
                dimension_tile=self._factor_tile_size,
                seed=self._factor_seed,
                oversampling=self._factor_oversampling)
        else:
            loadings = self._constructor_factor_loadings.copy()
            initialization = {"source": "supplied"}
        if self._factor_estimation == "joint":
            return self._fit_joint_factor(
                u, config, options, loadings, initialization)

        factor = FactorCorrelation(
            loadings, uniqueness_min=self._factor_uniqueness_min,
            diagnostics=initialization)
        evaluator = FactorStudentEvaluator(factor.prepare(), u)
        policy = CorrelationPolicy.create(
            mode="factor", estimator="factor_two_stage", dimension=u.shape[1],
            factor_rank=self._factor_rank, factor_estimation="two-stage",
            initialization_source=str(
                initialization.get("source", "factor_loadings")))

        def evaluate(parameters):
            value, gradient = evaluator.objective_and_gradient(
                float(parameters[0]), n_threads=config.n_threads)
            return StaticMLEEvaluation(
                objective=value, gradient=gradient,
                state={"df": float(parameters[0])})

        outcome = run_static_multivariate_mle(
            StaticMLEProblem(
                family="student_factor",
                initial_parameters=np.array([df_initial]),
                bounds=(df_bounds,), evaluate=evaluate),
            optimizer_options=options, fail_value=config.fail_value)
        return self._make_result_and_commit(
            u=u, config=config, outcome=outcome, policy=policy,
            correlation=None, preprocessing=None,
            raw=np.empty(0), alpha=None, factor=factor,
            initialization=initialization)

    def _fit_joint_factor(self, u, config, options, loadings, initialization):
        df_initial, df_bounds = model_policy.student_fit_policy(
            u.shape[1], stochastic=False)
        parameterization, factor0 = FactorLoadingParameterization.from_loadings(
            loadings, uniqueness_min=self._factor_uniqueness_min)
        if parameterization.n_parameters > self._factor_joint_max_params:
            raise ValueError("joint factor estimation exceeds factor_joint_max_params")
        policy = CorrelationPolicy.create(
            mode="factor", estimator="factor_joint", dimension=u.shape[1],
            factor_rank=self._factor_rank, factor_estimation="joint",
            initialization_source=str(
                initialization.get("source", "factor_loadings")))
        factor_evaluator = FactorStudentEvaluator(
            FactorCorrelation(
                loadings,
                uniqueness_min=self._factor_uniqueness_min,
                diagnostics={"source": "joint_static_mle_initial"}),
            u,
        )

        def evaluate(parameters):
            native = (
                factor_evaluator
                .penalized_parameterized_objective_and_gradient(
                    float(parameters[0]),
                    parameters[1:],
                    parameterization,
                    penalty=self._factor_joint_penalty,
                    condition_max=self._factor_joint_condition_max,
                    n_threads=config.n_threads,
                ))
            return StaticMLEEvaluation(
                objective=native.objective, gradient=native.gradient,
                state={
                    "df": float(parameters[0]),
                    "loadings": native.loadings.copy(),
                    "log_likelihood": float(native.log_likelihood),
                    "anchor_rows": parameterization.anchors.copy(),
                })

        joint_options = dict(options)
        joint_options.setdefault("ftol", 1e-10)
        # The loading pullback can be steep near the uniqueness boundary;
        # a longer line search avoids spurious ABNORMAL terminations while
        # preserving an explicitly supplied, larger maxls value.
        joint_options["maxls"] = max(
            int(joint_options.get("maxls", 20)), 100)
        outcome = run_static_multivariate_mle(
            StaticMLEProblem(
                family="student_factor",
                initial_parameters=np.concatenate((
                    np.array([df_initial]), factor0)),
                bounds=(df_bounds,)
                + ((None, None),) * parameterization.n_parameters,
                evaluate=evaluate),
            optimizer_options=joint_options, fail_value=config.fail_value)
        final_loadings = (
            np.asarray(loadings).copy() if outcome.evaluation is None
            else np.asarray(outcome.evaluation.state["loadings"]).copy())
        factor = FactorCorrelation(
            final_loadings, uniqueness_min=self._factor_uniqueness_min,
            diagnostics={
                **initialization,
                "source": "joint_static_mle",
                "joint_anchor_rows": parameterization.anchors.tolist(),
            })
        return self._make_result_and_commit(
            u=u, config=config, outcome=outcome, policy=policy,
            correlation=None, preprocessing=None,
            raw=outcome.parameters[1:].copy(), alpha=None, factor=factor,
            initialization=initialization,
            actual_log_likelihood=(
                None if outcome.evaluation is None
                else float(outcome.evaluation.state["log_likelihood"])),
            extra_diagnostics={
                "joint_static": True,
                "joint_anchor_rows": parameterization.anchors.copy(),
                "joint_penalty": self._factor_joint_penalty,
                "joint_condition_max": self._factor_joint_condition_max,
            })

    def _make_result_and_commit(
            self, *, u, config, outcome, policy, correlation, preprocessing,
            raw, alpha, factor, initialization,
            actual_log_likelihood=None, extra_diagnostics=None):
        df_hat = float(outcome.parameters[0])
        joint_static = bool(policy.optimized_n_params)
        gradient_mode = (
            "analytical_joint_factor"
            if policy.mode == "factor" and joint_static
            else "analytical_joint"
            if joint_static
            else "analytical_df")
        diagnostics = {
            "n_threads": config.n_threads,
            "parameterization": "natural_df",
            "optimizer_gradient": "analytical",
            "df_gradient": "analytical",
            "correlation_gradient": (
                "analytical_factor"
                if policy.mode == "factor" and joint_static
                else "analytical" if joint_static else "not_applicable"),
            "gradient_mode": gradient_mode,
            "joint_static": joint_static,
            "corr_params_raw": np.asarray(raw).copy(),
            "corr_alpha": alpha,
            **policy.diagnostics(),
            **outcome.diagnostics(),
            **({} if extra_diagnostics is None else extra_diagnostics),
        }
        model_parameters: dict[str, object] = {
            "df": df_hat,
            "corr_mode": self._corr_mode,
            "corr_estimator": policy.estimator,
            "corr_alpha": alpha,
            "corr_params_raw": np.asarray(raw).copy(),
        }
        if factor is None:
            diagnostics["corr_matrix"] = correlation.copy()
            model_parameters["correlation_matrix"] = correlation.copy()
        else:
            diagnostics.update({
                "factor_rank": self._factor_rank,
                "factor_estimation": self._factor_estimation,
                "factor_initialized": True,
                "representation": "factor_woodbury",
                **dict(factor.prepare().diagnostics),
                **{
                    f"initialization_{key}": value
                    for key, value in (initialization or {}).items()},
            })
            model_parameters.update({
                "factor_rank": self._factor_rank,
                "factor_loadings": factor.loadings.copy(),
                "factor_uniqueness": factor.uniqueness.copy(),
                "factor_estimation": self._factor_estimation,
            })
        result = MultivariateMLEResult(
            log_likelihood=(
                -float(outcome.final_objective)
                if actual_log_likelihood is None else actual_log_likelihood),
            method="MLE", copula_name=self.name,
            success=outcome.accepted, nfev=outcome.nfev,
            message=outcome.message, copula_param=df_hat,
            parameter_count=1 + policy.effective_n_params,
            n_observations=len(u), model_parameters=model_parameters,
            correlation_matrix=(None if factor is not None else correlation.copy()),
            diagnostics=diagnostics,
        )
        if outcome.accepted:
            self._set_dimension(u.shape[1], allow_change=False)
            self.df = df_hat
            self._corr_estimator = policy.estimator
            self._corr_params_raw = np.asarray(raw).copy()
            self._corr_alpha = alpha
            self.correlation_preprocessing = preprocessing
            if factor is None:
                self._correlation = correlation.copy()
                if self._corr_mode in {"shrinkage", "cholesky"}:
                    self._corr_base = policy.initial_correlation
            else:
                self._correlation = None
                self._set_factor_loadings(
                    factor.loadings, diagnostics=factor.diagnostics)
            self.fit_result = result
            self._last_u = u.copy()
        return result

    def _fitted_parameters(self):
        result = self.fit_result
        if isinstance(result, MultivariateMLEResult):
            correlation = result.correlation_matrix
            if correlation is None:
                correlation = self.to_correlation_matrix()
            return correlation.copy(), as_float64_scalar(
                result.copula_param, name="df")
        if self.df is None or self._correlation is None:
            raise ValueError("Fit first")
        return self._correlation.copy(), as_float64_scalar(self.df, name="df")

    def log_pdf_rows(self, u, parameter=None, *, n_threads=1):
        df = self.df if parameter is None else parameter
        if df is None:
            raise ValueError("Fit first")
        df = as_float64_scalar(df, name="df")
        if self._corr_mode == "factor":
            return FactorStudentEvaluator(
                self.correlation_operator_, u).log_pdf_rows(
                    df, n_threads=n_threads)
        from pyscarcopula._native import static as static_likelihood
        return static_likelihood.prepare_student(
            self._correlation, u, n_threads=n_threads).log_pdf_rows(df)

    def log_likelihood(self, u, parameter=None, *, n_threads=1):
        """Evaluate the current correlation at fitted or explicitly supplied df."""
        df = self.df if parameter is None else parameter
        if df is None:
            raise ValueError("Fit first")
        df = as_float64_scalar(df, name="df")
        if self._corr_mode == "factor":
            return FactorStudentEvaluator(
                self.correlation_operator_, u).evaluate(
                    df, n_threads=n_threads).log_likelihood
        from pyscarcopula._native import static as static_likelihood
        return static_likelihood.prepare_student(
            self._correlation, u,
            n_threads=n_threads).log_likelihood(df)

    def _nll_with_params(self, u, R, df):
        from pyscarcopula._native import static as static_likelihood
        return static_likelihood.prepare_student(
            R, u).objective_and_gradient(df)[0]

    def _nll(self, u):
        if self._correlation is None or self.df is None:
            return -self.log_likelihood(u)
        return self._nll_with_params(u, self._correlation, self.df)

    @model_state_locked
    def sample(self, n, u=None, rng=None, *, n_threads=1):
        n = validate_integer(n, "n")
        n_threads = _validated_n_threads(n_threads)
        if rng is None:
            rng = np.random.default_rng()
        from pyscarcopula._native import multivariate as multivariate_native
        if self._corr_mode == "factor":
            if isinstance(self.fit_result, MultivariateMLEResult):
                from pyscarcopula.strategy.multivariate_mle import (
                    sampling_model_from_result,
                )
                snapshot = sampling_model_from_result(self, self.fit_result)
                return snapshot.sample(n, rng=rng, n_threads=n_threads)
            if self.df is None:
                raise ValueError("Fit first")
            operator = self.correlation_operator_
            factor_draws = rng.standard_normal((n, operator.rank))
            residual_draws = rng.standard_normal((n, operator.dimension))
            chi_square_uniforms = rng.uniform(0.0, 1.0, size=n)
            return multivariate_native.factor_student_sample_from_normal_uniforms(
                operator,
                self.df,
                factor_draws,
                residual_draws,
                chi_square_uniforms,
                n_threads=n_threads,
            )
        correlation, df = self._fitted_parameters()
        chi_square_uniforms = rng.uniform(0.0, 1.0, size=n)
        normal_draws = rng.standard_normal((n, correlation.shape[0]))
        return multivariate_native.student_sample_from_normal_uniforms(
            correlation, df, normal_draws, chi_square_uniforms,
            n_threads=n_threads)

    @model_state_locked
    def sample_conditional(self, n, given, rng=None, *, n_threads=1):
        """Draw samples conditional on fixed copula-uniform coordinates."""
        n = validate_integer(n, "n")
        n_threads = _validated_n_threads(n_threads)
        from pyscarcopula.copula.multivariate.conditional import (
            validate_multivariate_given,
        )
        normalized = validate_multivariate_given(given, self.dimension)
        if not normalized:
            return self.sample(n, rng=rng, n_threads=n_threads)
        if isinstance(self.fit_result, MultivariateMLEResult):
            from pyscarcopula.strategy.multivariate_mle import (
                sampling_model_from_result,
            )
            snapshot = sampling_model_from_result(self, self.fit_result)
            return snapshot.sample_conditional(
                n, normalized, rng=rng, n_threads=n_threads)
        if self.df is None:
            raise ValueError("Fit first")
        if self._corr_mode == "factor":
            from pyscarcopula.copula.multivariate.conditional import (
                sample_factor_student_conditional,
            )
            return sample_factor_student_conditional(
                n, self.correlation_operator_, self.df, normalized,
                rng=rng, n_threads=n_threads)
        from pyscarcopula.copula.multivariate.conditional import (
            sample_student_conditional,
        )
        return sample_student_conditional(
            n, self._correlation, self.df, given=normalized, rng=rng,
            n_threads=n_threads)

    def predict(self, n, u=None, rng=None, given=None, horizon="next",
                predictive_r_mode=None, predict_config=None, *, n_threads=1):
        """Draw predictive samples, optionally conditional on fixed uniforms."""
        n_threads = _validated_n_threads(n_threads)
        from pyscarcopula.api import (
            _resolve_predict_config, _validate_non_vine_predict_config,
        )
        config = _resolve_predict_config(
            predict_config, given, horizon,
            {"predictive_r_mode": predictive_r_mode})
        _validate_non_vine_predict_config(config)
        given = config.given
        if given is not None:
            return self.sample_conditional(
                n, given=given, rng=rng, n_threads=n_threads)
        return self.sample(n, u=u, rng=rng, n_threads=n_threads)
