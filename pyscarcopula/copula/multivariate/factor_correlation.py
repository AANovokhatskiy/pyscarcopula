"""Reusable low-rank correlation operators independent of copula families."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import InitVar, dataclass, field
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from pyscarcopula._native.threads import validate_n_threads
from pyscarcopula.numerical._arrays import validate_integer


FACTOR_CORRELATION_FORMAT_VERSION = 1


def _validated_n_threads(value):
    return validate_n_threads(value)


def _validated_budget(memory_budget_bytes, required, guidance):
    if memory_budget_bytes is None:
        return
    if (
            isinstance(memory_budget_bytes, (bool, np.bool_))
            or not isinstance(memory_budget_bytes, (int, np.integer))):
        raise TypeError("memory_budget_bytes must be an integer")
    if int(memory_budget_bytes) < required:
        raise MemoryError(f"operation requires {required} bytes; {guidance}")


def _metadata(factor: "FactorCorrelation") -> dict[str, Any]:
    return {
        "format_version": factor.format_version,
        "uniqueness_min": factor.uniqueness_min,
        "diagnostics": dict(factor.diagnostics),
    }


@dataclass(frozen=True)
class FactorCorrelation:
    """Immutable factor correlation ``R = D + B B.T``.

    ``D`` is derived from the row norms of ``loadings`` so every diagonal
    entry of ``R`` equals one. This value object does not depend on a copula
    family. Call :meth:`prepare` to construct the reusable native Woodbury
    operator.
    """

    loadings: np.ndarray
    uniqueness_min: float = 1e-8
    format_version: int = FACTOR_CORRELATION_FORMAT_VERSION
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    _copy_arrays: InitVar[bool] = True
    _uniqueness: np.ndarray = field(init=False, repr=False)

    def __post_init__(self, _copy_arrays: bool) -> None:
        if self.format_version != FACTOR_CORRELATION_FORMAT_VERSION:
            raise ValueError(
                f"unsupported factor-correlation format version "
                f"{self.format_version}")
        if not np.isfinite(self.uniqueness_min) or not (
                0.0 < self.uniqueness_min < 1.0):
            raise ValueError(
                "uniqueness_min must be finite and in (0, 1)")

        convert = np.array if _copy_arrays else np.asanyarray
        loadings = convert(self.loadings, dtype=np.float64)
        if (
                loadings.ndim != 2
                or loadings.shape[0] < 2
                or loadings.shape[1] < 1
                or loadings.shape[1] >= loadings.shape[0]):
            raise ValueError(
                "loadings must have shape (d, k), 1 <= k < d")
        if not loadings.flags.c_contiguous:
            loadings = np.ascontiguousarray(loadings)
        if not np.all(np.isfinite(loadings)):
            raise ValueError("loadings must contain only finite values")

        from pyscarcopula._native import multivariate as multivariate_native
        try:
            validated_loadings, uniqueness = (
                multivariate_native.factor_correlation_from_loadings(
                    loadings, self.uniqueness_min)
            )
        except ValueError as exc:
            raise ValueError(
                "each loading row must satisfy "
                "1 - squared_norm >= uniqueness_min") from exc
        if _copy_arrays:
            loadings = validated_loadings
        loadings.setflags(write=False)
        uniqueness.setflags(write=False)
        object.__setattr__(self, "loadings", loadings)
        object.__setattr__(self, "_uniqueness", uniqueness)
        object.__setattr__(
            self,
            "diagnostics",
            MappingProxyType(dict(self.diagnostics)),
        )

    @property
    def dimension(self) -> int:
        """Number of correlated variables."""
        return int(self.loadings.shape[0])

    @property
    def rank(self) -> int:
        """Number of latent factors."""
        return int(self.loadings.shape[1])

    @property
    def uniqueness(self) -> np.ndarray:
        """Read-only diagonal uniqueness vector ``diag(D)``."""
        return self._uniqueness

    @property
    def storage_bytes(self) -> int:
        """Bytes occupied by the compact loading and uniqueness arrays."""
        return int(self.loadings.nbytes + self.uniqueness.nbytes)

    @classmethod
    def from_unconstrained(
            cls,
            values: Any,
            *,
            uniqueness_min: float = 1e-8) -> "FactorCorrelation":
        """Map arbitrary finite rows into the valid factor-correlation set."""
        unconstrained = np.asarray(values, dtype=np.float64)
        if (
                unconstrained.ndim != 2
                or unconstrained.shape[0] < 2
                or unconstrained.shape[1] < 1
                or unconstrained.shape[1] >= unconstrained.shape[0]
                or not np.all(np.isfinite(unconstrained))):
            raise ValueError(
                "unconstrained values must have shape (d, k), "
                "1 <= k < d, and be finite")
        if not np.isfinite(uniqueness_min) or not (
                0.0 < float(uniqueness_min) < 1.0):
            raise ValueError(
                "uniqueness_min must be finite and in (0, 1)")
        from pyscarcopula._native import multivariate as multivariate_native
        loadings = multivariate_native.factor_correlation_from_unconstrained(
            unconstrained, uniqueness_min)
        return cls(
            loadings=loadings,
            uniqueness_min=uniqueness_min,
            diagnostics={"source": "unconstrained_row_transform"},
        )

    def prepare(self) -> "PreparedFactorCorrelation":
        """Build the immutable native Woodbury workspace."""
        return PreparedFactorCorrelation(self)

    def to_dense(
            self,
            *,
            max_dimension: int = 2048,
            memory_budget_bytes: int | None = None) -> np.ndarray:
        """Explicitly materialize ``R`` for small diagnostic problems."""
        max_dimension = validate_integer(
            max_dimension, "max_dimension", minimum=1)
        required = self.dimension * self.dimension * 8
        if self.dimension > max_dimension:
            raise MemoryError(
                f"dense correlation is disabled for dimension "
                f"{self.dimension}; increase max_dimension explicitly")
        _validated_budget(
            memory_budget_bytes,
            required,
            "increase memory_budget_bytes or use prepare()",
        )
        from pyscarcopula._native import multivariate as multivariate_native
        return multivariate_native.factor_correlation_to_dense(
            self.loadings, self.uniqueness)

    def save_npz(self, path: str | Path) -> Path:
        """Write the compact factor representation."""
        target = Path(path)
        if target.suffix.lower() != ".npz":
            target = Path(f"{target}.npz")
        np.savez_compressed(
            target,
            loadings=np.asarray(self.loadings),
            metadata=np.asarray(json.dumps(
                _metadata(self), sort_keys=True, separators=(",", ":"))),
        )
        return target

    @classmethod
    def load_npz(cls, path: str | Path) -> "FactorCorrelation":
        """Load a compact factor representation created by :meth:`save_npz`."""
        with np.load(Path(path), allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            return cls(loadings=archive["loadings"], **metadata)

    def save_mmap(self, directory: str | Path) -> Path:
        """Write an mmap-friendly directory without overwriting it."""
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=False)
        np.save(target / "loadings.npy", np.asarray(self.loadings))
        (target / "metadata.json").write_text(
            json.dumps(
                _metadata(self), sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        return target

    @classmethod
    def load_mmap(cls, directory: str | Path) -> "FactorCorrelation":
        """Open an mmap-backed factor representation without copying it."""
        source = Path(directory)
        metadata = json.loads(
            (source / "metadata.json").read_text(encoding="utf-8"))
        return cls(
            loadings=np.load(
                source / "loadings.npy",
                mmap_mode="r",
                allow_pickle=False,
            ),
            _copy_arrays=False,
            **metadata,
        )


@dataclass(frozen=True, init=False)
class PreparedFactorCorrelation:
    """Thread-safe native Woodbury workspace for a factor correlation."""

    factor: FactorCorrelation
    _native: Any = field(repr=False)
    diagnostics: Mapping[str, Any]

    def __init__(self, factor: FactorCorrelation) -> None:
        if not isinstance(factor, FactorCorrelation):
            raise TypeError("factor must be a FactorCorrelation")
        from pyscarcopula._native import _extension as _cpp_extension

        native = _cpp_extension.load()._FactorCorrelationOperator(
            factor.loadings,
            factor.uniqueness_min,
        )
        diagnostics = MappingProxyType({
            "dimension": factor.dimension,
            "rank": factor.rank,
            "uniqueness_min": factor.uniqueness_min,
            "min_uniqueness": float(np.min(factor.uniqueness)),
            "max_uniqueness": float(np.max(factor.uniqueness)),
            "condition_estimate_m": float(native.condition_estimate),
            "prepared_storage_bytes": int(
                2 * factor.loadings.nbytes
                + 2 * factor.uniqueness.nbytes
                + factor.rank * factor.rank * 8
            ),
            "representation": "factor_woodbury",
        })
        object.__setattr__(self, "factor", factor)
        object.__setattr__(self, "_native", native)
        object.__setattr__(self, "diagnostics", diagnostics)

    @property
    def dimension(self) -> int:
        """Number of correlated variables."""
        return self.factor.dimension

    @property
    def rank(self) -> int:
        """Number of latent factors."""
        return self.factor.rank

    @property
    def loadings(self) -> np.ndarray:
        """Read-only factor loading matrix."""
        return self.factor.loadings

    @property
    def uniqueness(self) -> np.ndarray:
        """Read-only diagonal uniqueness vector."""
        return self.factor.uniqueness

    @property
    def logdet(self) -> float:
        """Log determinant of the represented correlation matrix."""
        return float(self._native.logdet)

    def _rows(self, values):
        array = np.asarray(values, dtype=np.float64)
        squeeze = array.ndim == 1
        if squeeze:
            array = array[None, :]
        if array.ndim != 2 or array.shape[1] != self.dimension:
            raise ValueError(
                f"values must have shape ({self.dimension},) or "
                f"(n, {self.dimension})")
        if not np.all(np.isfinite(array)):
            raise ValueError("values must contain only finite values")
        return np.ascontiguousarray(array), squeeze

    def matvec(self, values: Any, *, n_threads: int = 1) -> np.ndarray:
        """Multiply one or more row vectors by the correlation matrix."""
        rows, squeeze = self._rows(values)
        output = np.asarray(
            self._native.matvec(
                rows, _validated_n_threads(n_threads)),
            dtype=np.float64,
        )
        return output[0] if squeeze else output

    def solve(self, values: Any, *, n_threads: int = 1) -> np.ndarray:
        """Solve against the correlation matrix for one or more row vectors."""
        rows, squeeze = self._rows(values)
        output = np.asarray(
            self._native.solve(
                rows, _validated_n_threads(n_threads)),
            dtype=np.float64,
        )
        return output[0] if squeeze else output

    def quadratic_forms(
            self, values: Any, *, n_threads: int = 1) -> np.ndarray:
        """Evaluate ``x.T @ R**-1 @ x`` for one or more row vectors."""
        rows, _ = self._rows(values)
        return np.asarray(
            self._native.quadratic_forms(
                rows, _validated_n_threads(n_threads)),
            dtype=np.float64,
        )

    def quadratic_form(
            self, value: Any, *, n_threads: int = 1) -> float:
        """Evaluate ``x.T @ R**-1 @ x`` for one vector."""
        array = np.asarray(value)
        if array.ndim != 1:
            raise ValueError("quadratic_form expects one 1D vector")
        return float(self.quadratic_forms(
            array, n_threads=n_threads)[0])

    def to_dense(
            self,
            *,
            max_dimension: int = 2048,
            memory_budget_bytes: int | None = None) -> np.ndarray:
        """Explicitly materialize the correlation matrix."""
        return self.factor.to_dense(
            max_dimension=max_dimension,
            memory_budget_bytes=memory_budget_bytes,
        )

    def sample_normal(
            self,
            n: int,
            *,
            rng: np.random.Generator | None = None,
            n_threads: int = 1,
            memory_budget_bytes: int | None = None) -> np.ndarray:
        """Draw factor-normal rows using bounded, deterministic native work."""
        n = validate_integer(n, "n")
        required = n * (self.dimension + self.rank) * 8
        _validated_budget(
            memory_budget_bytes,
            required,
            "use sample_normal_batches() or increase memory_budget_bytes",
        )
        if rng is None:
            rng = np.random.default_rng()
        factors = np.ascontiguousarray(
            rng.standard_normal((n, self.rank)), dtype=np.float64)
        residuals = np.ascontiguousarray(
            rng.standard_normal((n, self.dimension)), dtype=np.float64)
        self._native.sample_normal_inplace(
            factors,
            residuals,
            _validated_n_threads(n_threads),
        )
        return residuals

    def transform_normal_draws(
            self,
            factor_draws: Any,
            residual_draws: Any,
            *,
            n_threads: int = 1) -> np.ndarray:
        """Transform fixed independent draws into factor-normal rows.

        The returned array is a copy of ``residual_draws``. Keeping random
        number generation outside the native operator makes results exactly
        reproducible across thread counts and also supports conditional
        factor distributions whose factor draws have a non-identity small
        covariance.
        """
        factors = np.asarray(factor_draws, dtype=np.float64)
        residuals = np.asarray(residual_draws, dtype=np.float64)
        if (
                factors.ndim != 2
                or factors.shape[1] != self.rank):
            raise ValueError(
                f"factor_draws must have shape (n, {self.rank})")
        if (
                residuals.ndim != 2
                or residuals.shape
                != (factors.shape[0], self.dimension)):
            raise ValueError(
                "residual_draws must have shape "
                f"(n, {self.dimension}) and share the row count")
        if (
                not np.all(np.isfinite(factors))
                or not np.all(np.isfinite(residuals))):
            raise ValueError(
                "factor_draws and residual_draws must be finite")
        factors = np.ascontiguousarray(factors)
        output = np.array(residuals, dtype=np.float64, order="C", copy=True)
        self._native.sample_normal_inplace(
            factors,
            output,
            _validated_n_threads(n_threads),
        )
        return output

    def sample_normal_batches(
            self,
            n: int,
            *,
            batch_rows: int = 128,
            rng: np.random.Generator | None = None,
            n_threads: int = 1,
            memory_budget_bytes: int | None = None) -> Iterator[np.ndarray]:
        """Yield bounded batches of factor-normal rows."""
        n = validate_integer(n, "n")
        batch_rows = validate_integer(batch_rows, "batch_rows", minimum=1)
        _validated_n_threads(n_threads)
        required = min(n, batch_rows) * (
            self.dimension + self.rank) * 8
        _validated_budget(
            memory_budget_bytes,
            required,
            "reduce batch_rows or increase memory_budget_bytes",
        )
        if rng is None:
            rng = np.random.default_rng()
        for start in range(0, n, batch_rows):
            count = min(batch_rows, n - start)
            yield self.sample_normal(
                count,
                rng=rng,
                n_threads=n_threads,
                memory_budget_bytes=memory_budget_bytes,
            )
