"""Student copula likelihood adapter for reusable factor correlations."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from pyscarcopula.copula.multivariate.factor_correlation import (
    FactorCorrelation,
    PreparedFactorCorrelation,
    _validated_budget,
    _validated_n_threads,
)
from pyscarcopula.numerical._arrays import validate_integer


def _raise_native_status(result: Mapping[str, Any], operation: str) -> None:
    """Translate a mechanical native result at the Python adapter boundary."""
    from pyscarcopula._native.errors import raise_for_status

    raise_for_status(
        result,
        operation,
        prefix="C++ factor Student",
        failure_fields={"failure_index": "index"},
    )


@dataclass(frozen=True)
class FactorStudentEvaluation:
    """Immutable row likelihood and degrees-of-freedom derivative result."""

    log_pdf: np.ndarray
    dlog_ddf: np.ndarray
    _log_likelihood: float
    _dlog_likelihood_ddf: float
    _negative_log_likelihood: float
    _dnegative_log_likelihood_ddf: float
    diagnostics: Mapping[str, Any]
    common_df: bool

    @property
    def log_likelihood(self) -> float:
        """Sum the row log densities."""
        return self._log_likelihood

    @property
    def dlog_likelihood_ddf(self) -> float:
        """Sum df derivatives when the evaluation used one common df."""
        if not self.common_df:
            raise ValueError(
                "one aggregate df derivative requires a common scalar df")
        return self._dlog_likelihood_ddf

    @property
    def negative_log_likelihood(self) -> float:
        """Return the native negative aggregate likelihood."""
        return self._negative_log_likelihood

    @property
    def dnegative_log_likelihood_ddf(self) -> float:
        """Return its native derivative for one common ``df``."""
        if not self.common_df:
            raise ValueError(
                "one aggregate df derivative requires a common scalar df")
        return self._dnegative_log_likelihood_ddf


@dataclass(frozen=True)
class FactorStudentJointEvaluation:
    """Aggregate likelihood and analytical df/loading gradients."""

    log_likelihood: float
    dlog_likelihood_ddf: float
    dlog_likelihood_dloadings: np.ndarray
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class FactorStudentParameterizedEvaluation:
    """Native penalized objective in identifiable factor coordinates."""

    objective: float
    gradient: np.ndarray
    loadings: np.ndarray
    log_likelihood: float
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class FactorStudentGridEvaluation:
    """Immutable tiled Student log-density grid result."""

    log_pdf: np.ndarray
    dlog_ddf: np.ndarray
    diagnostics: Mapping[str, Any]

    def pdf_and_gradient(self) -> tuple[np.ndarray, np.ndarray]:
        """Convert stable log values into density and ``d pdf / d df``."""
        from pyscarcopula._native import _extension as _cpp_extension

        native_result = dict(
            _cpp_extension.load()._factor_student_density_from_log_grid(
                self.log_pdf, self.dlog_ddf))
        _raise_native_status(native_result, "density-grid conversion")
        density = np.asarray(native_result["pdf"], dtype=np.float64)
        gradient = np.asarray(
            native_result["d_pdf_ddf"], dtype=np.float64)
        if density.shape != self.log_pdf.shape or gradient.shape != (
                self.log_pdf.shape):
            raise RuntimeError(
                "native factor Student density-grid conversion returned "
                "invalid output")
        density.setflags(write=False)
        gradient.setflags(write=False)
        return density, gradient


class FactorStudentEvaluator:
    """Static Student likelihood composed with a factor correlation.

    The evaluator owns an immutable observation copy and shares the immutable
    :class:`PreparedFactorCorrelation`. It contains no optimizer or copula
    model state and is safe for concurrent read-only evaluations.
    """

    def __init__(
            self,
            correlation: FactorCorrelation | PreparedFactorCorrelation,
            observations: Any) -> None:
        if isinstance(correlation, FactorCorrelation):
            correlation = correlation.prepare()
        if not isinstance(correlation, PreparedFactorCorrelation):
            raise TypeError(
                "correlation must be FactorCorrelation or "
                "PreparedFactorCorrelation")

        values = np.array(observations, dtype=np.float64, order="C")
        if (
                values.ndim != 2
                or values.shape[0] < 1
                or values.shape[1] != correlation.dimension):
            raise ValueError(
                "observations must have shape "
                f"(n, {correlation.dimension}), n >= 1")
        if not np.all(np.isfinite(values)):
            raise ValueError(
                "observations must contain only finite values")
        values.setflags(write=False)
        self._correlation = correlation
        self._observations = values

    @property
    def correlation(self) -> PreparedFactorCorrelation:
        """Prepared factor-correlation operator shared by this evaluator."""
        return self._correlation

    @property
    def observations(self) -> np.ndarray:
        """Immutable pseudo-observations owned by this evaluator."""
        return self._observations

    @property
    def n_observations(self) -> int:
        """Number of observation rows."""
        return int(self._observations.shape[0])

    @property
    def dimension(self) -> int:
        """Number of variables per observation."""
        return self._correlation.dimension

    @property
    def rank(self) -> int:
        """Rank of the factor correlation."""
        return self._correlation.rank

    def _df_values(self, df):
        values = np.asarray(df, dtype=np.float64)
        common = values.ndim == 0
        if common:
            values = values.reshape(1)
        elif values.ndim != 1 or values.shape[0] != self.n_observations:
            raise ValueError(
                "df must be a scalar or one-dimensional with one "
                "value per observation")
        if np.any(~np.isfinite(values)) or np.any(values <= 2.0):
            raise ValueError(
                "df values must be finite and greater than 2")
        return np.ascontiguousarray(values), common

    def evaluate(
            self, df: Any, *, n_threads: int = 1
    ) -> FactorStudentEvaluation:
        """Evaluate row log densities and analytical df derivatives."""
        from pyscarcopula._native import _extension as _cpp_extension

        n_threads = _validated_n_threads(n_threads)
        df_values, common = self._df_values(df)
        native_result = dict(
            _cpp_extension.load()
            ._factor_student_log_pdf_and_dlog_ddf(
                self._correlation._native,
                self._observations,
                df_values,
                n_threads,
            )
        )
        _raise_native_status(native_result, "row evaluation")
        native_result.pop("status")
        log_pdf = np.asarray(
            native_result.pop("log_pdf"), dtype=np.float64)
        dlog_ddf = np.asarray(
            native_result.pop("dlog_ddf"), dtype=np.float64)
        log_likelihood = float(native_result.pop("log_likelihood"))
        dlog_likelihood_ddf = float(
            native_result.pop("dlog_likelihood_ddf"))
        negative_log_likelihood = float(
            native_result.pop("negative_log_likelihood"))
        dnegative_log_likelihood_ddf = float(
            native_result.pop("dnegative_log_likelihood_ddf"))
        if (
                log_pdf.shape != (self.n_observations,)
                or dlog_ddf.shape != (self.n_observations,)
                or np.any(~np.isfinite(log_pdf))
                or np.any(~np.isfinite(dlog_ddf))
                or not np.isfinite(log_likelihood)
                or not np.isfinite(dlog_likelihood_ddf)
                or not np.isfinite(negative_log_likelihood)
                or not np.isfinite(dnegative_log_likelihood_ddf)):
            raise RuntimeError(
                "native factor Student evaluation returned invalid output")
        log_pdf.setflags(write=False)
        dlog_ddf.setflags(write=False)
        diagnostics = MappingProxyType({
            **native_result,
            "dimension": self.dimension,
            "rank": self.rank,
            "n_observations": self.n_observations,
            "common_df": common,
            "representation": "factor_student_woodbury",
        })
        return FactorStudentEvaluation(
            log_pdf=log_pdf,
            dlog_ddf=dlog_ddf,
            _log_likelihood=log_likelihood,
            _dlog_likelihood_ddf=dlog_likelihood_ddf,
            _negative_log_likelihood=negative_log_likelihood,
            _dnegative_log_likelihood_ddf=
                dnegative_log_likelihood_ddf,
            diagnostics=diagnostics,
            common_df=common,
        )

    def log_pdf_and_dlog_ddf_rows(
            self, df: Any, *, n_threads: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return row log densities and their analytical df derivatives."""
        result = self.evaluate(df, n_threads=n_threads)
        return result.log_pdf, result.dlog_ddf

    def log_pdf_rows(
            self, df: Any, *, n_threads: int = 1) -> np.ndarray:
        """Return row log densities."""
        return self.evaluate(df, n_threads=n_threads).log_pdf

    def dlog_pdf_ddf_rows(
            self, df: Any, *, n_threads: int = 1) -> np.ndarray:
        """Return analytical row derivatives with respect to df."""
        return self.evaluate(df, n_threads=n_threads).dlog_ddf

    def log_likelihood_and_gradient(
            self, df: float, *, n_threads: int = 1
    ) -> tuple[float, float]:
        """Return likelihood and derivative for one common scalar ``df``."""
        if np.asarray(df).ndim != 0:
            raise ValueError(
                "log_likelihood_and_gradient requires a scalar df")
        result = self.evaluate(df, n_threads=n_threads)
        return result.log_likelihood, result.dlog_likelihood_ddf

    def objective_and_gradient(
            self, df: float, *, n_threads: int = 1
    ) -> tuple[float, np.ndarray]:
        """Return negative likelihood and a one-element optimizer gradient."""
        result = self.evaluate(df, n_threads=n_threads)
        return (
            result.negative_log_likelihood,
            np.asarray(
                [result.dnegative_log_likelihood_ddf],
                dtype=np.float64),
        )

    def penalized_parameterized_objective_and_gradient(
            self, df, parameters, parameterization, *, penalty,
            condition_max, n_threads=1
    ) -> FactorStudentParameterizedEvaluation:
        """Evaluate the joint factor objective entirely in native code."""
        from pyscarcopula._native import _extension as _cpp_extension

        df_value = float(df)
        if not np.isfinite(df_value) or df_value <= 2.0:
            raise ValueError("df must be finite and greater than 2")
        values = np.ascontiguousarray(parameters, dtype=np.float64)
        if values.ndim != 1 or values.shape != (
                parameterization.n_parameters,):
            raise ValueError("factor parameters have unexpected shape")
        n_threads = _validated_n_threads(n_threads)
        native_result = dict(
            _cpp_extension.load()
            ._factor_student_penalized_parameterized_objective_gradient(
                self._observations,
                df_value,
                values,
                np.ascontiguousarray(
                    parameterization.free_rows, dtype=np.float64),
                np.ascontiguousarray(
                    parameterization.free_columns, dtype=np.float64),
                np.ascontiguousarray(
                    parameterization.diagonal_entries, dtype=np.float64),
                self.dimension,
                self.rank,
                float(parameterization.max_norm),
                float(parameterization.uniqueness_min),
                float(condition_max),
                float(penalty),
                n_threads,
            )
        )
        _raise_native_status(native_result, "parameterized objective")
        native_result.pop("status")
        objective = float(native_result.pop("objective"))
        log_likelihood = float(native_result.pop("log_likelihood"))
        gradient = np.asarray(
            native_result.pop("gradient"), dtype=np.float64)
        loadings = np.asarray(
            native_result.pop("loadings"), dtype=np.float64)
        if (
                not np.isfinite(objective)
                or not np.isfinite(log_likelihood)
                or gradient.shape != (values.size + 1,)
                or loadings.shape != (self.dimension, self.rank)
                or np.any(~np.isfinite(gradient))
                or np.any(~np.isfinite(loadings))):
            raise RuntimeError(
                "native factor Student parameterized objective returned "
                "invalid output")
        gradient.setflags(write=False)
        loadings.setflags(write=False)
        return FactorStudentParameterizedEvaluation(
            objective=objective,
            gradient=gradient,
            loadings=loadings,
            log_likelihood=log_likelihood,
            diagnostics=MappingProxyType(native_result),
        )

    def joint_likelihood_and_gradient(
            self, df: float, *, n_threads: int = 1
    ) -> FactorStudentJointEvaluation:
        """Return aggregate analytical gradients for scalar ``df`` and B."""
        from pyscarcopula._native import _extension as _cpp_extension

        if np.asarray(df).ndim != 0:
            raise ValueError(
                "joint_likelihood_and_gradient requires a scalar df")
        df_value = float(df)
        if not np.isfinite(df_value) or df_value <= 2.0:
            raise ValueError("df must be finite and greater than 2")
        n_threads = _validated_n_threads(n_threads)
        native_result = dict(
            _cpp_extension.load()
            ._factor_student_joint_likelihood_gradient(
                self._correlation._native,
                self._observations,
                df_value,
                n_threads,
            )
        )
        _raise_native_status(native_result, "joint evaluation")
        native_result.pop("status")
        log_likelihood = float(native_result.pop("log_likelihood"))
        df_gradient = float(
            native_result.pop("dlog_likelihood_ddf"))
        loading_gradient = np.asarray(
            native_result.pop("dlog_likelihood_dloadings"),
            dtype=np.float64,
        )
        if (
                not np.isfinite(log_likelihood)
                or not np.isfinite(df_gradient)
                or loading_gradient.shape != (
                    self.dimension, self.rank)
                or np.any(~np.isfinite(loading_gradient))):
            raise RuntimeError(
                "native factor Student joint evaluation returned "
                "invalid output")
        loading_gradient.setflags(write=False)
        diagnostics = MappingProxyType({
            **native_result,
            "dimension": self.dimension,
            "rank": self.rank,
            "n_observations": self.n_observations,
            "representation": "factor_student_joint_woodbury",
            "gradient_kind": "analytical",
        })
        return FactorStudentJointEvaluation(
            log_likelihood=log_likelihood,
            dlog_likelihood_ddf=df_gradient,
            dlog_likelihood_dloadings=loading_gradient,
            diagnostics=diagnostics,
        )

    def _grid_values(self, df_grid):
        values = np.asarray(df_grid, dtype=np.float64)
        if values.ndim != 1 or values.shape[0] < 1:
            raise ValueError("df_grid must be a non-empty 1D array")
        if np.any(~np.isfinite(values)) or np.any(values <= 2.0):
            raise ValueError(
                "df_grid values must be finite and greater than 2")
        return np.ascontiguousarray(values)

    def _grid_peak_bytes(
            self,
            rows,
            grid_size,
            dimension_tile,
            n_threads):
        cells = rows * grid_size
        width = 4 + 2 * self.rank
        dimension_tiles = (
            self.dimension + dimension_tile - 1) // dimension_tile
        worker_bytes = (2 * width + self.rank) * 8
        cell_parallel = (
            n_threads > 1 and cells >= 4 * n_threads)
        dimension_parallel = (
            n_threads > 1
            and not cell_parallel
            and dimension_tiles >= n_threads)
        active_workers = n_threads if cell_parallel else 1
        partial_bytes = (
            dimension_tiles * (width * 8 + 1)
            if dimension_parallel else 0
        )
        # Native result vectors coexist briefly with the two NumPy copies.
        output_peak_bytes = 4 * cells * 8
        return (
            output_peak_bytes
            + active_workers * worker_bytes
            + partial_bytes
        )

    def _evaluate_grid_block(
            self,
            observations,
            df_grid,
            *,
            dimension_tile,
            n_threads,
            memory_budget_bytes):
        from pyscarcopula._native import _extension as _cpp_extension

        required = self._grid_peak_bytes(
            len(observations),
            len(df_grid),
            dimension_tile,
            n_threads,
        )
        _validated_budget(
            memory_budget_bytes,
            required,
            "use evaluate_grid_batches(), reduce batch_rows, or "
            "increase memory_budget_bytes",
        )
        native_result = dict(
            _cpp_extension.load()
            ._factor_student_log_pdf_and_dlog_ddf_grid(
                self._correlation._native,
                observations,
                df_grid,
                dimension_tile,
                n_threads,
            )
        )
        _raise_native_status(native_result, "grid evaluation")
        native_result.pop("status")
        log_pdf = np.asarray(
            native_result.pop("log_pdf"), dtype=np.float64)
        dlog_ddf = np.asarray(
            native_result.pop("dlog_ddf"), dtype=np.float64)
        expected_shape = (len(observations), len(df_grid))
        if (
                log_pdf.shape != expected_shape
                or dlog_ddf.shape != expected_shape
                or np.any(~np.isfinite(log_pdf))
                or np.any(~np.isfinite(dlog_ddf))):
            raise RuntimeError(
                "native tiled factor Student grid returned invalid output")
        log_pdf.setflags(write=False)
        dlog_ddf.setflags(write=False)
        axis = {
            0: "sequential",
            1: "cells",
            2: "dimension_tiles",
        }.get(int(native_result["parallel_axis"]), "unknown")
        diagnostics = MappingProxyType({
            **native_result,
            "parallel_axis": axis,
            "dimension": self.dimension,
            "rank": self.rank,
            "peak_bytes_required": required,
            "representation": "factor_student_tiled",
        })
        return FactorStudentGridEvaluation(
            log_pdf=log_pdf,
            dlog_ddf=dlog_ddf,
            diagnostics=diagnostics,
        )

    def evaluate_grid(
            self,
            df_grid: Any,
            *,
            dimension_tile: int = 16384,
            n_threads: int = 1,
            memory_budget_bytes: int | None = None
    ) -> FactorStudentGridEvaluation:
        """Evaluate a tiled ``(observations, df_grid)`` log-density grid."""
        dimension_tile = validate_integer(
            dimension_tile, "dimension_tile", minimum=1)
        n_threads = _validated_n_threads(n_threads)
        grid = self._grid_values(df_grid)
        return self._evaluate_grid_block(
            self._observations,
            grid,
            dimension_tile=dimension_tile,
            n_threads=n_threads,
            memory_budget_bytes=memory_budget_bytes,
        )

    def log_pdf_and_dlog_ddf_grid(
            self,
            df_grid: Any,
            *,
            dimension_tile: int = 16384,
            n_threads: int = 1,
            memory_budget_bytes: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return tiled row/grid log densities and df derivatives."""
        result = self.evaluate_grid(
            df_grid,
            dimension_tile=dimension_tile,
            n_threads=n_threads,
            memory_budget_bytes=memory_budget_bytes,
        )
        return result.log_pdf, result.dlog_ddf

    def pdf_and_grad_on_grid(
            self,
            df_grid: Any,
            *,
            dimension_tile: int = 16384,
            n_threads: int = 1,
            memory_budget_bytes: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return tiled row/grid densities and df derivatives."""
        result = self.evaluate_grid(
            df_grid,
            dimension_tile=dimension_tile,
            n_threads=n_threads,
            memory_budget_bytes=memory_budget_bytes,
        )
        return result.pdf_and_gradient()

    def evaluate_grid_batches(
            self,
            df_grid: Any,
            *,
            batch_rows: int = 128,
            dimension_tile: int = 16384,
            n_threads: int = 1,
            memory_budget_bytes: int | None = None
    ) -> Iterator[FactorStudentGridEvaluation]:
        """Yield bounded row batches of the tiled Student grid."""
        batch_rows = validate_integer(batch_rows, "batch_rows", minimum=1)
        dimension_tile = validate_integer(
            dimension_tile, "dimension_tile", minimum=1)
        n_threads = _validated_n_threads(n_threads)
        grid = self._grid_values(df_grid)
        required = self._grid_peak_bytes(
            min(batch_rows, self.n_observations),
            len(grid),
            dimension_tile,
            n_threads,
        )
        _validated_budget(
            memory_budget_bytes,
            required,
            "reduce batch_rows or increase memory_budget_bytes",
        )
        for start in range(0, self.n_observations, batch_rows):
            stop = min(self.n_observations, start + batch_rows)
            yield self._evaluate_grid_block(
                self._observations[start:stop],
                grid,
                dimension_tile=dimension_tile,
                n_threads=n_threads,
                memory_budget_bytes=memory_budget_bytes,
            )

    def pdf_and_grad_on_grid_batches(
            self,
            df_grid: Any,
            *,
            batch_rows: int = 128,
            dimension_tile: int = 16384,
            n_threads: int = 1,
            memory_budget_bytes: int | None = None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield bounded density/gradient grid batches."""
        for result in self.evaluate_grid_batches(
                df_grid,
                batch_rows=batch_rows,
                dimension_tile=dimension_tile,
                n_threads=n_threads,
                memory_budget_bytes=memory_budget_bytes):
            yield result.pdf_and_gradient()
