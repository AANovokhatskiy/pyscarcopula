"""Typed correlation policy shared by multivariate copula models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias, cast
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pyscarcopula.copula.multivariate.corr_param import (
    CorrelationPreprocessingResult,
    _corr_gradient_to_raw_params,
    _make_shrinkage_corr_from_validated,
    cholesky_corr_n_params,
    logit,
    pack_cholesky_corr,
    unpack_cholesky_corr,
    validate_corr_matrix,
)


CorrelationMode: TypeAlias = Literal[
    "fixed",
    "shrinkage",
    "cholesky",
    "factor",
]
FactorEstimation: TypeAlias = Literal["two-stage", "joint"]
CorrelationEstimator: TypeAlias = Literal[
    "supplied",
    "gaussian_score",
    "kendall_plugin",
    "joint_mle",
    "factor_two_stage",
    "factor_joint",
]
FloatArray: TypeAlias = NDArray[np.float64]

_CORRELATION_MODES = frozenset(
    ("fixed", "shrinkage", "cholesky", "factor"))
_FACTOR_ESTIMATIONS = frozenset(("two-stage", "joint"))
_CORRELATION_ESTIMATORS = frozenset((
    "supplied",
    "gaussian_score",
    "kendall_plugin",
    "joint_mle",
    "factor_two_stage",
    "factor_joint",
))


def normalize_correlation_mode(
    value: str,
    *,
    allow_dense_alias: bool = False,
    warn_on_dense: bool = True,
) -> CorrelationMode:
    """Return a canonical, lowercase correlation mode.

    ``dense`` is a temporary Gaussian-only compatibility alias for ``fixed``.
    """
    if not isinstance(value, str):
        raise TypeError("corr_mode must be a string")
    mode = value.lower()
    if mode == "dense" and allow_dense_alias:
        if warn_on_dense:
            warnings.warn(
                "corr_mode='dense' is deprecated; use corr_mode='fixed'",
                DeprecationWarning,
                stacklevel=2,
            )
        return "fixed"
    if mode not in _CORRELATION_MODES:
        allowed = "'fixed', 'shrinkage', 'cholesky', or 'factor'"
        raise ValueError(f"corr_mode must be {allowed}")
    return cast(CorrelationMode, mode)


def normalize_factor_estimation(value: str) -> FactorEstimation:
    """Return a canonical, lowercase factor estimation policy."""
    if not isinstance(value, str):
        raise TypeError("factor_estimation must be a string")
    estimation = value.lower()
    if estimation not in _FACTOR_ESTIMATIONS:
        raise ValueError("factor_estimation must be 'two-stage' or 'joint'")
    return cast(FactorEstimation, estimation)


def _readonly_float_array(value: ArrayLike, *, name: str) -> FloatArray:
    array = np.array(value, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class CorrelationPolicy:
    """Immutable strategy-level description of correlation estimation.

    The object owns read-only copies of all arrays.  It can construct trial
    dense correlations and map native lower-triangle scores back to the raw
    optimizer parameterization without mutating model state.
    """

    mode: CorrelationMode
    estimator: CorrelationEstimator
    dimension: int
    supplied_correlation: FloatArray | None = None
    base_correlation: FloatArray | None = None
    preprocessing: CorrelationPreprocessingResult | None = None
    raw_parameters: FloatArray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64))
    optimized_n_params: int = field(init=False)
    plugin_n_params: int = field(init=False)
    effective_n_params: int = field(init=False)
    factor_rank: int | None = None
    factor_estimation: FactorEstimation | None = None
    shrinkage_initial: float = 0.8

    def __post_init__(self) -> None:
        mode = normalize_correlation_mode(self.mode)
        if self.estimator not in _CORRELATION_ESTIMATORS:
            raise ValueError(f"invalid correlation estimator {self.estimator!r}")
        dimension = int(self.dimension)
        if dimension < 2:
            raise ValueError("dimension must be at least 2")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "dimension", dimension)

        for name in ("supplied_correlation", "base_correlation"):
            value = getattr(self, name)
            if value is None:
                continue
            correlation = _readonly_float_array(value, name=name)
            if correlation.shape != (dimension, dimension):
                raise ValueError(
                    f"{name} must have shape ({dimension}, {dimension})")
            validate_corr_matrix(correlation)
            object.__setattr__(self, name, correlation)

        if self.preprocessing is not None:
            preprocessing_correlation = _readonly_float_array(
                self.preprocessing.correlation,
                name="preprocessing.correlation",
            )
            preprocessing_input = _readonly_float_array(
                self.preprocessing.input_correlation,
                name="preprocessing.input_correlation",
            )
            expected_shape = (dimension, dimension)
            if (
                    preprocessing_correlation.shape != expected_shape
                    or preprocessing_input.shape != expected_shape):
                raise ValueError(
                    "preprocessing correlations must match dimension")
            validate_corr_matrix(preprocessing_correlation)
            object.__setattr__(
                self,
                "preprocessing",
                CorrelationPreprocessingResult(
                    correlation=preprocessing_correlation,
                    input_correlation=preprocessing_input,
                    source=self.preprocessing.source,
                    projection_applied=self.preprocessing.projection_applied,
                    min_eigenvalue_before=(
                        self.preprocessing.min_eigenvalue_before),
                    min_eigenvalue_after=(
                        self.preprocessing.min_eigenvalue_after),
                    nonfinite_kendall_pairs=(
                        self.preprocessing.nonfinite_kendall_pairs),
                ),
            )

        raw = _readonly_float_array(
            self.raw_parameters, name="raw_parameters").reshape(-1)
        raw.setflags(write=False)
        object.__setattr__(self, "raw_parameters", raw)

        if not 0.0 < float(self.shrinkage_initial) < 1.0:
            raise ValueError("shrinkage_initial must be in (0, 1)")
        object.__setattr__(
            self, "shrinkage_initial", float(self.shrinkage_initial))

        if mode == "factor":
            if self.factor_rank is None:
                raise ValueError("factor_rank is required in factor mode")
            rank = int(self.factor_rank)
            if not 1 <= rank < dimension:
                raise ValueError("factor_rank must satisfy 1 <= k < dimension")
            estimation = normalize_factor_estimation(
                self.factor_estimation or "two-stage")
            object.__setattr__(self, "factor_rank", rank)
            object.__setattr__(self, "factor_estimation", estimation)
        elif self.factor_rank is not None or self.factor_estimation is not None:
            raise ValueError("factor metadata requires corr_mode='factor'")

        compatible_estimators = {
            "fixed": {"supplied", "gaussian_score", "kendall_plugin"},
            "shrinkage": {"joint_mle"},
            "cholesky": {"joint_mle"},
            "factor": {"factor_two_stage", "factor_joint"},
        }
        if self.estimator not in compatible_estimators[mode]:
            raise ValueError(
                f"corr_estimator={self.estimator!r} is incompatible with "
                f"corr_mode={mode!r}")

        dense_n = cholesky_corr_n_params(dimension)
        factor_n = 0
        if mode == "factor":
            factor_n = (
                dimension * self.factor_rank
                - self.factor_rank * (self.factor_rank - 1) // 2)
        optimized_n = {
            "fixed": 0,
            "shrinkage": 1,
            "cholesky": dense_n,
            "factor": factor_n if self.factor_estimation == "joint" else 0,
        }[mode]
        plugin_n = 0
        if mode == "fixed" and self.estimator in {
                "gaussian_score", "kendall_plugin"}:
            plugin_n = dense_n
        elif mode == "factor" and self.estimator == "factor_two_stage":
            plugin_n = factor_n
        object.__setattr__(self, "optimized_n_params", optimized_n)
        object.__setattr__(self, "plugin_n_params", plugin_n)
        object.__setattr__(
            self, "effective_n_params", optimized_n + plugin_n)

        expected_raw = self.optimized_n_params
        if mode == "factor":
            # Factor raw parameters use a separate identifiable loading
            # parameterization and are not materialized by this dense policy.
            expected_raw = 0
        if raw.size not in (0, expected_raw):
            raise ValueError(
                f"expected zero or {expected_raw} raw parameters, "
                f"got {raw.size}")

    @classmethod
    def create(
        cls,
        *,
        mode: CorrelationMode,
        estimator: CorrelationEstimator,
        dimension: int,
        supplied_correlation: ArrayLike | None = None,
        base_correlation: ArrayLike | None = None,
        preprocessing: CorrelationPreprocessingResult | None = None,
        raw_parameters: ArrayLike | None = None,
        factor_rank: int | None = None,
        factor_estimation: FactorEstimation | None = None,
        shrinkage_initial: float = 0.8,
    ) -> "CorrelationPolicy":
        """Build a validated policy and derive its parameter counts."""
        canonical_mode = normalize_correlation_mode(mode)
        dimension = int(dimension)
        canonical_factor_estimation = factor_estimation
        if canonical_mode == "factor":
            if factor_rank is None:
                raise ValueError("factor_rank is required in factor mode")
            canonical_factor_estimation = normalize_factor_estimation(
                factor_estimation or "two-stage")

        return cls(
            mode=canonical_mode,
            estimator=estimator,
            dimension=dimension,
            supplied_correlation=supplied_correlation,
            base_correlation=base_correlation,
            preprocessing=preprocessing,
            raw_parameters=(
                np.empty(0, dtype=np.float64)
                if raw_parameters is None else raw_parameters),
            factor_rank=factor_rank,
            factor_estimation=canonical_factor_estimation,
            shrinkage_initial=shrinkage_initial,
        )

    @property
    def initial_correlation(self) -> FloatArray | None:
        """Return a copy selected by ``base -> supplied -> preprocessing``."""
        correlation = self.base_correlation
        if correlation is None:
            correlation = self.supplied_correlation
        if correlation is None and self.preprocessing is not None:
            correlation = self.preprocessing.correlation
        if correlation is None:
            return None
        return np.array(correlation, dtype=np.float64, copy=True)

    def initial_raw_parameters(self) -> FloatArray:
        """Return an owned initial raw optimizer vector."""
        if self.raw_parameters.size:
            return np.array(self.raw_parameters, copy=True)
        if self.mode in {"fixed", "factor"}:
            return np.empty(0, dtype=np.float64)
        if self.mode == "shrinkage":
            return np.array(
                [float(logit(self.shrinkage_initial))], dtype=np.float64)
        correlation = self.initial_correlation
        if correlation is None:
            raise ValueError("cholesky mode requires an initial correlation")
        return pack_cholesky_corr(correlation)

    def trial_correlation(self, raw_parameters: ArrayLike) -> FloatArray:
        """Build a dense trial correlation without model-state mutation."""
        raw = np.asarray(raw_parameters, dtype=np.float64).reshape(-1)
        if self.mode == "fixed":
            correlation = self.initial_correlation
            if correlation is None:
                raise ValueError("fixed mode requires an initialized correlation")
            if raw.size:
                raise ValueError("fixed mode has no raw correlation parameters")
            return correlation
        if self.mode == "shrinkage":
            base = self.initial_correlation
            if base is None or raw.size != 1:
                raise ValueError(
                    "shrinkage mode requires a base and one raw parameter")
            return _make_shrinkage_corr_from_validated(float(raw[0]), base)
        if self.mode == "cholesky":
            return unpack_cholesky_corr(raw, self.dimension)
        raise NotImplementedError(
            "factor trials use the compact factor parameterization")

    def raw_gradient(
        self,
        raw_parameters: ArrayLike,
        correlation: ArrayLike,
        correlation_gradient: ArrayLike,
    ) -> FloatArray:
        """Map a native lower-triangle correlation score to raw space."""
        if self.mode not in {"shrinkage", "cholesky"}:
            raise ValueError(
                "raw correlation gradients require an optimized dense mode")
        return _corr_gradient_to_raw_params(
            self.mode,
            np.asarray(raw_parameters, dtype=np.float64),
            np.asarray(correlation, dtype=np.float64),
            np.asarray(correlation_gradient, dtype=np.float64),
            self.initial_correlation if self.mode == "shrinkage" else None,
        )

    def diagnostics(self) -> dict[str, object]:
        """Return canonical correlation metadata for fit results."""
        diagnostics: dict[str, object] = {
            "corr_mode": self.mode,
            "corr_estimator": self.estimator,
            "corr_n_params": self.optimized_n_params,
            "corr_plugin_n_params": self.plugin_n_params,
            "corr_effective_n_params": self.effective_n_params,
        }
        if self.preprocessing is not None:
            diagnostics.update(self.preprocessing.diagnostics())
        if self.mode == "factor":
            diagnostics.update({
                "factor_rank": self.factor_rank,
                "factor_estimation": self.factor_estimation,
            })
        return diagnostics
