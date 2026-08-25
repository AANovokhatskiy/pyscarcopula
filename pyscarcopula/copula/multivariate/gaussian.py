"""Static multivariate Gaussian copula."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike

from pyscarcopula._utils import pobs
from pyscarcopula._types import (
    DEFAULT_CONFIG,
    MultivariateMLEResult,
    NumericalConfig,
)
from pyscarcopula.copula.base import CopulaCapabilities
from pyscarcopula.copula.multivariate.base import (
    MultivariateCopula,
    model_state_locked,
)
from pyscarcopula.copula.multivariate.corr_param import (
    sigmoid,
    validate_corr_matrix,
)
from pyscarcopula.copula.multivariate.correlation_policy import (
    CorrelationEstimator,
    CorrelationMode,
    CorrelationPolicy,
    FactorEstimation,
    normalize_correlation_mode,
    normalize_factor_estimation,
    restore_correlation_result_metadata,
)
from pyscarcopula.numerical._arrays import (
    validate_integer,
    validate_sampling_memory_budget as _validated_budget,
    validate_sampling_n_threads as _validated_n_threads,
)


_LBFGSB_FIT_KEYS = (
    "gtol", "ftol", "maxfun", "maxiter", "maxls", "eps", "maxcor",
    "finite_diff_rel_step",
)
from pyscarcopula.copula.multivariate.factor_correlation import (
    FactorCorrelation,
)
from pyscarcopula.copula.multivariate.factor_estimation import (
    estimate_factor_loadings,
)
from pyscarcopula.strategy.multivariate_mle import (
    StaticMLEEvaluation,
    StaticMLEProblem,
    run_static_multivariate_mle,
)


def _validate_gaussian_fit_data(u):
    if u.ndim != 2:
        raise ValueError("data must have shape (n_observations, dimension)")
    if u.shape[0] == 0:
        raise ValueError("data must contain at least one observation")
    if u.shape[1] < 2:
        raise ValueError("data must contain at least two variables")
    if not np.all(np.isfinite(u)):
        raise ValueError("data must contain only finite values")
    if np.any((u < 0.0) | (u > 1.0)):
        raise ValueError(
            "MLE expects pseudo-observations in [0, 1]; use to_pobs=True")
    if np.any(np.ptp(u, axis=0) == 0.0):
        raise ValueError(
            "Gaussian copula correlation is not identifiable for constant "
            "data columns")
    if any(
            np.array_equal(u[:, left], u[:, right])
            for right in range(1, u.shape[1])
            for left in range(right)):
        raise ValueError(
            "Gaussian copula correlation is not identifiable for duplicate "
            "data columns")


def _as_real_array(data):
    raw = np.asarray(data)
    if np.iscomplexobj(raw):
        raise ValueError("data must be real-valued")
    if raw.dtype.kind in {"O", "S", "U", "V", "b"}:
        raise TypeError("data must have a real numeric dtype")
    return np.asarray(raw, dtype=np.float64)


def _validated_correlation(value, *, name, dimension=None):
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real-valued")
    correlation = np.array(raw, dtype=np.float64, copy=True)
    if correlation.ndim != 2 or correlation.shape[0] != correlation.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    if dimension is not None and correlation.shape != (dimension, dimension):
        raise ValueError(
            f"{name} must have shape ({dimension}, {dimension})")
    validate_corr_matrix(correlation)
    return correlation


def _gaussian_score_correlation(u):
    from pyscarcopula.numerical import multivariate_native
    return multivariate_native.gaussian_score_correlation(u)


class GaussianCopula(MultivariateCopula):
    """Static Gaussian copula with dense or compact factor correlation."""

    _capabilities = CopulaCapabilities(
        supports_conditional_sampling=True,
    )

    def __init__(
            self,
            d: int | None = None,
            R: ArrayLike | None = None,
            *,
            corr_mode: CorrelationMode | Literal["dense"] = "fixed",
            corr_base: ArrayLike | None = None,
            corr_shrinkage_init: float = 0.8,
            cholesky_d_max: int = 10,
            allow_large_cholesky: bool = False,
            factor_rank: int | None = None,
            factor_loadings: ArrayLike | None = None,
            factor_estimation: FactorEstimation = "two-stage",
            factor_tile_size: int = 16384,
            factor_uniqueness_min: float = 1e-8,
            factor_seed: int = 0,
            factor_oversampling: int = 8) -> None:
        corr_mode = normalize_correlation_mode(
            corr_mode, allow_dense_alias=True)
        factor_estimation = normalize_factor_estimation(factor_estimation)
        if R is not None and d is None:
            raw_R = np.asarray(R)
            if raw_R.ndim == 2 and raw_R.shape[0] == raw_R.shape[1]:
                d = int(raw_R.shape[0])
        if corr_base is not None and d is None:
            raw_base = np.asarray(corr_base)
            if raw_base.ndim == 2 and raw_base.shape[0] == raw_base.shape[1]:
                d = int(raw_base.shape[0])
        if corr_mode == "fixed" and corr_base is not None:
            raise ValueError("corr_base is only valid for estimated corr modes")
        if corr_mode == "factor" and (R is not None or corr_base is not None):
            raise ValueError("R and corr_base are forbidden in factor mode")
        if corr_mode != "factor" and factor_estimation != "two-stage":
            raise ValueError(
                "factor_estimation is only configurable in factor mode")
        if not 0.0 < float(corr_shrinkage_init) < 1.0:
            raise ValueError("corr_shrinkage_init must be in (0, 1)")
        cholesky_d_max = validate_integer(
            cholesky_d_max, "cholesky_d_max", minimum=1)
        if (
                corr_mode == "cholesky" and d is not None
                and d > cholesky_d_max and not allow_large_cholesky):
            raise ValueError(
                f"corr_mode='cholesky' is limited to d <= "
                f"{cholesky_d_max} by default")
        if corr_mode == "factor":
            if factor_estimation == "joint":
                raise NotImplementedError(
                    "Gaussian factor_estimation='joint' requires a native "
                    "factor-loading score and is not implemented")
            if (
                    isinstance(factor_rank, (bool, np.bool_))
                    or not isinstance(factor_rank, (int, np.integer))):
                raise TypeError(
                    "factor_rank must be an integer in factor mode")
            factor_rank = int(factor_rank)
            if d is None:
                if factor_loadings is None:
                    raise ValueError(
                        "d is required when corr_mode='factor'")
                d = np.asarray(factor_loadings).shape[0]
            if not 1 <= factor_rank < int(d):
                raise ValueError("factor_rank must satisfy 1 <= k < d")
        elif factor_rank is not None or factor_loadings is not None:
            raise ValueError(
                "factor_rank and factor_loadings require "
                "corr_mode='factor'")

        super().__init__(dimension=d, name="Gaussian copula")
        self.corr = None
        self._corr_mode = corr_mode
        self._corr_estimator = None
        self._supplied_correlation = (
            None if R is None else _validated_correlation(
                R, name="R", dimension=self.dimension))
        self._corr_base = (
            None if corr_base is None else _validated_correlation(
                corr_base, name="corr_base", dimension=self.dimension))
        self._constructor_R = (
            None if self._supplied_correlation is None
            else self._supplied_correlation.copy())
        self._constructor_corr_base = (
            None if self._corr_base is None else self._corr_base.copy())
        self._corr_shrinkage_init = float(corr_shrinkage_init)
        self._corr_params_raw = np.empty(0, dtype=np.float64)
        self._corr_alpha = None
        self._cholesky_d_max = cholesky_d_max
        self._allow_large_cholesky = bool(allow_large_cholesky)
        self._factor_estimation = factor_estimation
        self._factor_rank = factor_rank
        self._factor_loadings = None
        self._factor_correlation = None
        self._factor_operator = None
        self._factor_initialization_diagnostics = {}
        self._constructor_factor_loadings = None
        self._factor_tile_size = validate_integer(
            factor_tile_size, "factor_tile_size", minimum=1)
        self._factor_uniqueness_min = float(factor_uniqueness_min)
        if not (
                np.isfinite(self._factor_uniqueness_min)
                and 0.0 < self._factor_uniqueness_min < 1.0):
            raise ValueError(
                "factor_uniqueness_min must be finite and in (0, 1)")
        self._factor_seed = validate_integer(factor_seed, "factor_seed")
        self._factor_oversampling = validate_integer(
            factor_oversampling, "factor_oversampling")
        if factor_loadings is not None:
            self._set_factor_loadings(
                factor_loadings, diagnostics={"source": "supplied"})
            self._constructor_factor_loadings = (
                self._factor_loadings.copy())

    @property
    def corr_mode(self) -> CorrelationMode:
        return self._corr_mode

    @property
    def corr_estimator_(self) -> CorrelationEstimator:
        """Correlation estimation procedure used by the static model."""
        if self._corr_estimator is not None:
            return self._corr_estimator
        if self._corr_mode == "factor":
            return "factor_two_stage"
        if self._corr_mode in {"shrinkage", "cholesky"}:
            return "joint_mle"
        if self._supplied_correlation is not None:
            return "supplied"
        return "gaussian_score"

    @property
    def factor_estimation(self) -> FactorEstimation | None:
        if self._corr_mode == "factor":
            return self._factor_estimation
        return None

    @property
    def correlation_policy_(self) -> CorrelationPolicy:
        """Return the immutable policy represented by current model state."""
        if self.dimension is None:
            raise ValueError("correlation policy requires a known dimension")
        return CorrelationPolicy.create(
            mode=self._corr_mode,
            estimator=self.corr_estimator_,
            dimension=self.dimension,
            supplied_correlation=(
                self.corr if self.corr_estimator_ == "supplied" else None),
            base_correlation=(
                self.corr
                if self._corr_mode == "fixed"
                else self._corr_base),
            shrinkage_initial=self._corr_shrinkage_init,
            factor_rank=self._factor_rank,
            factor_estimation=self.factor_estimation,
            initialization_source=(
                self._factor_initialization_diagnostics.get("source")
                if self._corr_mode == "factor"
                else None),
        )

    @property
    def factor_rank(self):
        return self._factor_rank

    @property
    def factor_tile_size(self):
        return self._factor_tile_size

    @property
    def factor_loadings_(self):
        if self._factor_loadings is None:
            return None
        return self._factor_loadings.copy()

    @property
    def factor_uniqueness_(self):
        if self._factor_correlation is None:
            return None
        return self._factor_correlation.uniqueness.copy()

    @property
    def correlation_operator_(self):
        if self._corr_mode != "factor":
            raise AttributeError(
                "correlation_operator_ is only available in factor mode")
        if self._factor_operator is None:
            raise ValueError(
                "factor correlation is not initialized; call fit() or "
                "initialize_factor()")
        return self._factor_operator

    def to_correlation_matrix(
            self, *, max_dimension=2048, memory_budget_bytes=None):
        if self._corr_mode == "factor":
            return self.correlation_operator_.to_dense(
                max_dimension=max_dimension,
                memory_budget_bytes=memory_budget_bytes,
            )
        if self.corr is None:
            raise ValueError("Fit first")
        return self.corr.copy()

    def __getstate__(self):
        state = super().__getstate__()
        state["_factor_correlation"] = None
        state["_factor_operator"] = None
        return state

    def __setstate__(self, state):
        stored_mode = state.get("_corr_mode")
        super().__setstate__(state)
        self._corr_mode = normalize_correlation_mode(
            getattr(self, "_corr_mode", "fixed"),
            allow_dense_alias=True,
            warn_on_dense=False,
        )
        self._corr_estimator = getattr(self, "_corr_estimator", None)
        self._supplied_correlation = getattr(
            self, "_supplied_correlation", None)
        self._corr_base = getattr(self, "_corr_base", None)
        self._constructor_R = getattr(self, "_constructor_R", None)
        self._constructor_corr_base = getattr(
            self, "_constructor_corr_base", None)
        self._corr_shrinkage_init = getattr(
            self, "_corr_shrinkage_init", 0.8)
        self._corr_params_raw = getattr(
            self, "_corr_params_raw", np.empty(0, dtype=np.float64))
        self._corr_alpha = getattr(self, "_corr_alpha", None)
        self._cholesky_d_max = getattr(self, "_cholesky_d_max", 10)
        self._allow_large_cholesky = getattr(
            self, "_allow_large_cholesky", False)
        self._factor_estimation = getattr(
            self, "_factor_estimation", "two-stage")
        self._factor_rank = getattr(self, "_factor_rank", None)
        self._factor_loadings = getattr(self, "_factor_loadings", None)
        self._constructor_factor_loadings = getattr(
            self, "_constructor_factor_loadings", None)
        self._factor_initialization_diagnostics = dict(getattr(
            self, "_factor_initialization_diagnostics", {}))
        self._factor_tile_size = int(getattr(
            self, "_factor_tile_size", 16384))
        self._factor_uniqueness_min = float(getattr(
            self, "_factor_uniqueness_min", 1e-8))
        self._factor_seed = int(getattr(self, "_factor_seed", 0))
        self._factor_oversampling = int(getattr(
            self, "_factor_oversampling", 8))
        self._factor_correlation = None
        self._factor_operator = None
        loadings = self._factor_loadings
        if self._corr_mode == "factor" and loadings is not None:
            self._set_factor_loadings(
                loadings,
                diagnostics=getattr(
                    self,
                    "_factor_initialization_diagnostics",
                    {"source": "restored"},
                ),
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
            if isinstance(result_diagnostics, dict):
                if stored_mode == "dense":
                    result_diagnostics.setdefault(
                        "corr_mode_migrated_from", "dense")
                elif stored_mode is None:
                    result_diagnostics.setdefault(
                        "corr_policy_migration", "legacy_fixed_default")

    def _set_factor_loadings(self, loadings, *, diagnostics=None):
        if self._corr_mode != "factor":
            raise ValueError(
                "factor loadings require corr_mode='factor'")
        loadings = np.asarray(loadings, dtype=np.float64)
        expected = (self.dimension, self._factor_rank)
        if loadings.shape != expected:
            raise ValueError(
                f"factor_loadings must have shape {expected}, "
                f"got {loadings.shape}")
        factor = FactorCorrelation(
            loadings,
            uniqueness_min=self._factor_uniqueness_min,
            diagnostics={} if diagnostics is None else diagnostics,
        )
        self._factor_loadings = factor.loadings
        self._factor_correlation = factor
        self._factor_operator = factor.prepare()
        self._factor_initialization_diagnostics = dict(
            factor.diagnostics)

    @model_state_locked
    def initialize_factor(self, data, *, to_pobs=False):
        if self._corr_mode != "factor":
            raise ValueError(
                "initialize_factor requires corr_mode='factor'")
        u = np.asarray(data, dtype=np.float64)
        _validate_gaussian_fit_data(u)
        if u.shape[1] != self.dimension:
            raise ValueError(
                f"data must have {self.dimension} columns")
        if to_pobs:
            u = pobs(u)
        elif np.any((u < 0.0) | (u > 1.0)):
            raise ValueError(
                "factor initialization expects pseudo-observations "
                "in [0, 1]; use to_pobs=True")
        if self._factor_operator is not None:
            return self._factor_operator
        loadings, diagnostics = estimate_factor_loadings(
            u,
            self._factor_rank,
            uniqueness_min=self._factor_uniqueness_min,
            dimension_tile=self._factor_tile_size,
            seed=self._factor_seed,
            oversampling=self._factor_oversampling,
        )
        self._set_factor_loadings(loadings, diagnostics=diagnostics)
        return self._factor_operator

    def factor_diagnostics(self):
        if self._corr_mode != "factor":
            return {}
        identifiable = (
            self.dimension * self._factor_rank
            - self._factor_rank * (self._factor_rank - 1) // 2
        )
        diagnostics = {
            "corr_mode": "factor",
            "factor_rank": self._factor_rank,
            "factor_n_params": identifiable,
            "factor_tile_size": self._factor_tile_size,
            "factor_uniqueness_min": self._factor_uniqueness_min,
            "factor_initialized": self._factor_operator is not None,
            "representation": "factor_woodbury",
        }
        if self._factor_operator is not None:
            diagnostics.update(dict(self._factor_operator.diagnostics))
            diagnostics.update({
                f"initialization_{key}": value
                for key, value
                in self._factor_correlation.diagnostics.items()
            })
        return diagnostics

    @model_state_locked
    def fit(
            self,
            data: ArrayLike,
            to_pobs: bool = False,
            method: str = 'mle',
            config: NumericalConfig | None = None,
            **kwargs: Any) -> MultivariateMLEResult:
        """Fit the correlation matrix in Gaussian score space.

        Only ``method='mle'`` is supported: this is a static model without
        a dynamic scalar parameter.
        """
        if str(method).upper() != 'MLE':
            raise ValueError(
                f"GaussianCopula supports only method='mle', "
                f"got {method!r}")
        u = _as_real_array(data)
        if to_pobs:
            if (
                    u.ndim != 2 or u.shape[0] == 0 or u.shape[1] < 2
                    or not np.all(np.isfinite(u))):
                raise ValueError("data must be a finite non-empty 2D array")
            u = pobs(u)
        _validate_gaussian_fit_data(u)
        if self.dimension is not None and u.shape[1] != self.dimension:
            raise ValueError(
                f"data must have {self.dimension} columns")

        if (
                self._corr_mode == "cholesky"
                and u.shape[1] > self._cholesky_d_max
                and not self._allow_large_cholesky):
            raise ValueError(
                f"corr_mode='cholesky' is limited to d <= "
                f"{self._cholesky_d_max} by default")
        if "tol" in kwargs:
            raise TypeError("tol is not supported; use gtol")
        optimizer_overrides = {
            key: kwargs.pop(key) for key in _LBFGSB_FIT_KEYS if key in kwargs}
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"unexpected MLE keyword argument(s): {unexpected}")
        if optimizer_overrides and self._corr_mode not in {
                "shrinkage", "cholesky"}:
            names = ", ".join(sorted(optimizer_overrides))
            raise TypeError(
                f"optimizer option(s) {names} require corr_mode="
                "'shrinkage' or 'cholesky'")

        config = config or DEFAULT_CONFIG
        n_threads = _validated_n_threads(
            config.n_threads)
        if self._corr_mode == "factor":
            if self._factor_operator is None:
                loadings, initialization = estimate_factor_loadings(
                    u,
                    self._factor_rank,
                    uniqueness_min=self._factor_uniqueness_min,
                    dimension_tile=self._factor_tile_size,
                    seed=self._factor_seed,
                    oversampling=self._factor_oversampling,
                )
                candidate = GaussianCopula(
                    d=u.shape[1],
                    corr_mode="factor",
                    factor_rank=self._factor_rank,
                    factor_loadings=loadings,
                    factor_tile_size=self._factor_tile_size,
                    factor_uniqueness_min=self._factor_uniqueness_min,
                    factor_seed=self._factor_seed,
                    factor_oversampling=self._factor_oversampling,
                )
                candidate._factor_initialization_diagnostics = initialization
                candidate._factor_correlation = FactorCorrelation(
                    candidate._factor_loadings,
                    uniqueness_min=self._factor_uniqueness_min,
                    diagnostics=initialization,
                )
                candidate._factor_operator = (
                    candidate._factor_correlation.prepare())
            else:
                candidate = GaussianCopula(
                    d=u.shape[1],
                    corr_mode="factor",
                    factor_rank=self._factor_rank,
                    factor_loadings=self._factor_loadings,
                    factor_tile_size=self._factor_tile_size,
                    factor_uniqueness_min=self._factor_uniqueness_min,
                    factor_seed=self._factor_seed,
                    factor_oversampling=self._factor_oversampling,
                )
                candidate._factor_initialization_diagnostics = dict(
                    self._factor_initialization_diagnostics)

            policy = candidate.correlation_policy_

            def evaluate_factor(
                    parameters: np.ndarray) -> StaticMLEEvaluation:
                return StaticMLEEvaluation(
                    objective=-candidate.log_likelihood(
                        u, n_threads=n_threads),
                    gradient=np.empty(0, dtype=np.float64),
                    state={"factor_loadings": candidate.factor_loadings_},
                )

            outcome = run_static_multivariate_mle(
                StaticMLEProblem(
                    family="gaussian",
                    initial_parameters=np.empty(0, dtype=np.float64),
                    bounds=(),
                    evaluate=evaluate_factor,
                    require_not_worse=False,
                ),
                optimizer_options={},
                fail_value=float(
                    config.fail_value),
            )
            diagnostics = {
                "estimator": "factor_gaussian_score_correlation",
                "corr_matrix": None,
                "n_threads": n_threads,
                "optimizer_gradient": "not_applicable",
                "correlation_gradient": "not_applicable",
                "gradient_mode": "not_applicable",
                "joint_static": False,
                "corr_params_raw": np.empty(0, dtype=np.float64),
                "corr_alpha": None,
                **policy.diagnostics(),
                **candidate.factor_diagnostics(),
                **outcome.diagnostics(),
            }
            result = MultivariateMLEResult(
                log_likelihood=-outcome.final_objective,
                method="MLE",
                copula_name=self.name,
                success=outcome.accepted,
                nfev=outcome.nfev,
                message=outcome.message,
                copula_param=None,
                parameter_count=policy.effective_n_params,
                n_observations=len(u),
                model_parameters={
                    "corr_mode": "factor",
                    "corr_estimator": self.corr_estimator_,
                    "corr_params_raw": np.empty(0, dtype=np.float64),
                    "corr_alpha": None,
                    "factor_loadings": candidate.factor_loadings_,
                    "factor_uniqueness": candidate.factor_uniqueness_,
                    "factor_rank": self._factor_rank,
                    "factor_estimation": "two-stage",
                },
                correlation_matrix=None,
                diagnostics=diagnostics,
            )
            if outcome.accepted:
                self._set_dimension(u.shape[1], allow_change=False)
                self._set_factor_loadings(
                    candidate._factor_loadings,
                    diagnostics=(
                        candidate._factor_initialization_diagnostics),
                )
                self.corr = None
                self.fit_result = result
                self._last_u = u.copy()
            return result

        if self._corr_base is not None:
            initial_correlation = self._corr_base.copy()
        elif self._supplied_correlation is not None:
            initial_correlation = self._supplied_correlation.copy()
        else:
            initial_correlation = _gaussian_score_correlation(u)
        initialization_source = (
            "corr_base" if self._corr_base is not None
            else "supplied" if self._supplied_correlation is not None
            else "gaussian_score")
        estimator: CorrelationEstimator = (
            "joint_mle"
            if self._corr_mode in {"shrinkage", "cholesky"}
            else (
                "supplied"
                if self._supplied_correlation is not None
                else "gaussian_score"))
        policy = CorrelationPolicy.create(
            mode=self._corr_mode,
            estimator=estimator,
            dimension=u.shape[1],
            supplied_correlation=(
                initial_correlation if estimator == "supplied" else None),
            base_correlation=(
                initial_correlation
                if self._corr_mode in {"shrinkage", "cholesky"}
                or estimator == "gaussian_score"
                else None),
            shrinkage_initial=self._corr_shrinkage_init,
            initialization_source=initialization_source,
        )
        from pyscarcopula.numerical import static_likelihood
        evaluator = static_likelihood.prepare_gaussian(
            initial_correlation, u, n_threads=n_threads)

        if self._corr_mode in {"shrinkage", "cholesky"}:
            corr0 = policy.initial_raw_parameters()

            def evaluate_joint(
                    parameters: np.ndarray) -> StaticMLEEvaluation:
                correlation = policy.trial_correlation(parameters)
                value, correlation_gradient = (
                    evaluator.gaussian_objective_and_gradient(
                        correlation, fail_value=config.fail_value))
                return StaticMLEEvaluation(
                    objective=value,
                    gradient=policy.raw_gradient(
                        parameters, correlation, correlation_gradient),
                    correlation=correlation,
                )

            outcome = run_static_multivariate_mle(
                StaticMLEProblem(
                    family="gaussian",
                    initial_parameters=corr0,
                    bounds=((None, None),) * policy.optimized_n_params,
                    evaluate=evaluate_joint,
                ),
                optimizer_options=config.mle_optimizer.options(
                    **optimizer_overrides),
                fail_value=config.fail_value,
            )
            correlation = (
                initial_correlation.copy()
                if outcome.evaluation is None
                else np.asarray(outcome.evaluation.correlation).copy())
            corr_raw = outcome.parameters.copy()
            corr_alpha = (
                float(sigmoid(corr_raw[0]))
                if self._corr_mode == "shrinkage" and corr_raw.size
                else None)
        else:
            correlation = initial_correlation.copy()
            corr_raw = np.empty(0, dtype=np.float64)
            corr_alpha = None

            def evaluate_fixed(
                    parameters: np.ndarray) -> StaticMLEEvaluation:
                return StaticMLEEvaluation(
                    objective=-evaluator.log_likelihood(0.0),
                    gradient=np.empty(0, dtype=np.float64),
                    correlation=correlation,
                )

            outcome = run_static_multivariate_mle(
                StaticMLEProblem(
                    family="gaussian",
                    initial_parameters=np.empty(0, dtype=np.float64),
                    bounds=(),
                    evaluate=evaluate_fixed,
                    require_not_worse=False,
                ),
                optimizer_options={},
                fail_value=config.fail_value,
            )

        result = MultivariateMLEResult(
            log_likelihood=-outcome.final_objective,
            method="MLE",
            copula_name=self.name,
            success=outcome.accepted,
            nfev=outcome.nfev,
            message=outcome.message,
            copula_param=None,
            parameter_count=policy.effective_n_params,
            n_observations=len(u),
            model_parameters={
                "corr_mode": self._corr_mode,
                "corr_estimator": estimator,
                "corr_alpha": corr_alpha,
                "corr_params_raw": corr_raw.copy(),
                "correlation_matrix": correlation.copy(),
            },
            correlation_matrix=correlation.copy(),
            diagnostics={
                "estimator": (
                    "gaussian_score_correlation"
                    if estimator == "gaussian_score" else estimator),
                "corr_matrix": correlation.copy(),
                "n_threads": n_threads,
                "optimizer_gradient": (
                    "analytical"
                    if policy.optimized_n_params else "not_applicable"),
                "correlation_gradient": (
                    "analytical"
                    if policy.optimized_n_params else "not_applicable"),
                "gradient_mode": (
                    "analytical_joint"
                    if policy.optimized_n_params else "not_applicable"),
                "joint_static": bool(policy.optimized_n_params),
                "corr_params_raw": corr_raw.copy(),
                "corr_alpha": corr_alpha,
                **policy.diagnostics(),
                **outcome.diagnostics(),
            },
        )
        if outcome.accepted:
            self._set_dimension(u.shape[1], allow_change=False)
            self.corr = correlation.copy()
            self._corr_estimator = estimator
            self._corr_params_raw = corr_raw.copy()
            self._corr_alpha = corr_alpha
            if self._corr_mode in {"shrinkage", "cholesky"}:
                self._corr_base = initial_correlation.copy()
            self.fit_result = result
            self._last_u = u.copy()
        return result

    def log_likelihood(self, u, *, n_threads=1):
        from pyscarcopula.numerical import static_likelihood
        return static_likelihood.prepare(
            self, u, n_threads=n_threads).log_likelihood(0.0)

    def log_pdf_rows(
            self, u, parameter=None, *, n_threads=1, **kwargs):
        from pyscarcopula.numerical import static_likelihood
        return static_likelihood.prepare(
            self, u, n_threads=n_threads).log_pdf_rows(0.0)

    def _nll(self, u):
        return -self.log_likelihood(u)

    def _fitted_correlation(self):
        result = self.fit_result
        if (
                isinstance(result, MultivariateMLEResult)
                and result.correlation_matrix is not None):
            return result.correlation_matrix
        return self.corr

    def _factor_sampling_peak_bytes(self, rows, *, conditional=False):
        factor_width = 3 * self._factor_rank if conditional else (
            self._factor_rank)
        return int(rows) * (2 * self.dimension + factor_width + 4) * 8

    @model_state_locked
    def sample(
            self,
            n,
            u=None,
            rng=None,
            *,
            n_threads=1,
            memory_budget_bytes=None):
        n = validate_integer(n, "n")
        n_threads = _validated_n_threads(n_threads)
        if rng is None:
            rng = np.random.default_rng()
        from pyscarcopula.numerical import multivariate_native
        if self._corr_mode == "factor":
            operator = self.correlation_operator_
            _validated_budget(
                memory_budget_bytes,
                self._factor_sampling_peak_bytes(n),
                "use sample_batches(), reduce batch_rows, or increase "
                "memory_budget_bytes",
            )
            factor_draws = rng.standard_normal((n, operator.rank))
            residual_draws = rng.standard_normal((n, operator.dimension))
            return multivariate_native.factor_gaussian_sample_from_normals(
                operator,
                factor_draws,
                residual_draws,
                n_threads=n_threads,
            )

        correlation = self._fitted_correlation()
        if correlation is None:
            raise ValueError("Fit first")
        d = correlation.shape[0]
        normal_draws = rng.standard_normal((n, d))
        return multivariate_native.gaussian_sample_from_normals(
            correlation, normal_draws, n_threads=n_threads)

    @model_state_locked
    def sample_batches(
            self,
            n,
            u=None,
            rng=None,
            *,
            batch_rows=128,
            given=None,
            n_threads=1,
            memory_budget_bytes=None):
        n = validate_integer(n, "n")
        batch_rows = validate_integer(batch_rows, "batch_rows", minimum=1)
        n_threads = _validated_n_threads(n_threads)
        if rng is None:
            rng = np.random.default_rng()
        if given is not None:
            from pyscarcopula.copula.multivariate.conditional import (
                validate_multivariate_given,
            )
            given = validate_multivariate_given(given, self.dimension)
        if self._corr_mode == "factor":
            _validated_budget(
                memory_budget_bytes,
                self._factor_sampling_peak_bytes(
                    min(n, batch_rows), conditional=bool(given)),
                "reduce batch_rows or increase memory_budget_bytes",
            )

        def blocks():
            for start in range(0, n, batch_rows):
                count = min(batch_rows, n - start)
                if given:
                    yield self.sample_conditional(
                        count,
                        given=given,
                        rng=rng,
                        n_threads=n_threads,
                        memory_budget_bytes=memory_budget_bytes,
                    )
                else:
                    yield self.sample(
                        count,
                        rng=rng,
                        n_threads=n_threads,
                        memory_budget_bytes=memory_budget_bytes,
                    )

        return blocks()

    @model_state_locked
    def sample_conditional(
            self,
            n,
            given,
            rng=None,
            *,
            n_threads=1,
            memory_budget_bytes=None):
        """Sample conditionally with ``given={var_index: u_value}``."""
        n = validate_integer(n, "n")
        n_threads = _validated_n_threads(n_threads)
        if self.dimension is not None:
            _validated_budget(
                memory_budget_bytes,
                n * self.dimension * 8,
                "use sample_batches(), reduce batch_rows, or increase "
                "memory_budget_bytes",
            )
        from pyscarcopula.copula.multivariate.conditional import (
            validate_multivariate_given,
        )
        normalized = validate_multivariate_given(given, self.dimension)
        if not normalized:
            return self.sample(
                n,
                rng=rng,
                n_threads=n_threads,
                memory_budget_bytes=memory_budget_bytes,
            )
        if self._corr_mode == "factor":
            from pyscarcopula.copula.multivariate.conditional import (
                sample_factor_gaussian_conditional,
            )
            _validated_budget(
                memory_budget_bytes,
                self._factor_sampling_peak_bytes(
                    n, conditional=True),
                "use sample_batches(), reduce batch_rows, or increase "
                "memory_budget_bytes",
            )
            return sample_factor_gaussian_conditional(
                n,
                self.correlation_operator_,
                normalized,
                rng=rng,
                n_threads=n_threads,
            )

        correlation = self._fitted_correlation()
        if correlation is None:
            raise ValueError("Fit first")
        from pyscarcopula.copula.multivariate.conditional import (
            sample_gaussian_copula_conditional,
        )
        return sample_gaussian_copula_conditional(
            n, correlation, given=normalized, rng=rng,
            n_threads=n_threads)

    @model_state_locked
    def predict(
            self,
            n,
            u=None,
            rng=None,
            given=None,
            horizon='next',
            predictive_r_mode=None,
            predict_config=None,
            *,
            n_threads=1,
            memory_budget_bytes=None):
        """Draw predictive samples, optionally conditional on fixed uniforms."""
        if predict_config is not None:
            from pyscarcopula.api import _resolve_predict_config
            config = _resolve_predict_config(
                predict_config, given, horizon, {
                    "predictive_r_mode": predictive_r_mode,
                })
            given = config.given
        sampling_options = {}
        if n_threads != 1:
            sampling_options["n_threads"] = n_threads
        if memory_budget_bytes is not None:
            sampling_options["memory_budget_bytes"] = memory_budget_bytes
        if given is not None:
            return self.sample_conditional(
                n,
                given=given,
                rng=rng,
                **sampling_options,
            )
        return self.sample(
            n,
            u=u,
            rng=rng,
            **sampling_options,
        )

    @model_state_locked
    def predict_batches(
            self,
            n,
            u=None,
            rng=None,
            *,
            batch_rows=128,
            given=None,
            horizon="next",
            predictive_r_mode=None,
            predict_config=None,
            n_threads=1,
            memory_budget_bytes=None):
        if predict_config is not None:
            from pyscarcopula.api import _resolve_predict_config
            config = _resolve_predict_config(
                predict_config, given, horizon, {
                    "predictive_r_mode": predictive_r_mode,
                })
            given = config.given
        return self.sample_batches(
            n,
            u=u,
            rng=rng,
            batch_rows=batch_rows,
            given=given,
            n_threads=n_threads,
            memory_budget_bytes=memory_budget_bytes,
        )
