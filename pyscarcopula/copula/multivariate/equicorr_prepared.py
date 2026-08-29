"""Compact sufficient statistics for high-dimensional equicorrelation data."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from pyscarcopula._constants import PSEUDO_OBS_EPS
from pyscarcopula._native import validation as native_validation


EQUICORR_PREPARED_FORMAT_VERSION = 1


def _metadata(
        prepared: "EquicorrPreparedData") -> dict[str, Any]:
    return {
        "format_version": prepared.format_version,
        "n_obs": prepared.n_obs,
        "dimension": prepared.dimension,
        "clipping_epsilon": prepared.clipping_epsilon,
        "diagnostics": dict(prepared.diagnostics),
    }


@dataclass(frozen=True)
class EquicorrPreparedData:
    """Immutable O(T) sufficient statistics for equicorrelation emissions.

    The original pseudo-observations are deliberately not retained. Use
    :meth:`save_npz` for a compact portable file or :meth:`save_mmap` when
    subsequent processes should open the two statistic vectors without
    copying them into memory.
    """

    sum_z: np.ndarray
    sum_z2: np.ndarray
    n_obs: int
    dimension: int
    format_version: int = EQUICORR_PREPARED_FORMAT_VERSION
    clipping_epsilon: float = PSEUDO_OBS_EPS
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    _copy_arrays: InitVar[bool] = True

    def __post_init__(self, _copy_arrays: bool) -> None:
        if self.format_version != EQUICORR_PREPARED_FORMAT_VERSION:
            raise ValueError(
                f"unsupported prepared-data format version "
                f"{self.format_version}")
        if isinstance(self.n_obs, bool) or int(self.n_obs) < 1:
            raise ValueError("n_obs must be a positive integer")
        if isinstance(self.dimension, bool) or int(self.dimension) < 2:
            raise ValueError("dimension must be an integer >= 2")
        convert = np.array if _copy_arrays else np.asanyarray
        sum_z = convert(self.sum_z, dtype=np.float64)
        sum_z2 = convert(self.sum_z2, dtype=np.float64)
        expected = (int(self.n_obs),)
        if sum_z.shape != expected or sum_z2.shape != expected:
            raise ValueError(
                f"sum_z and sum_z2 must have shape {expected}")
        native_validation.validate_equicorr_prepared(
            sum_z,
            sum_z2,
            self.dimension,
            self.clipping_epsilon,
        )

        sum_z.setflags(write=False)
        sum_z2.setflags(write=False)
        object.__setattr__(self, "sum_z", sum_z)
        object.__setattr__(self, "sum_z2", sum_z2)
        object.__setattr__(self, "n_obs", int(self.n_obs))
        object.__setattr__(self, "dimension", int(self.dimension))
        object.__setattr__(
            self, "diagnostics",
            MappingProxyType(dict(self.diagnostics)))

    def __len__(self) -> int:
        return self.n_obs

    def save_npz(self, path: str | Path) -> Path:
        """Write one portable, compressed file."""
        target = Path(path)
        if target.suffix.lower() != ".npz":
            target = Path(f"{target}.npz")
        metadata = json.dumps(
            _metadata(self), sort_keys=True, separators=(",", ":"))
        np.savez_compressed(
            target,
            sum_z=np.asarray(self.sum_z),
            sum_z2=np.asarray(self.sum_z2),
            metadata=np.asarray(metadata),
        )
        return target

    @classmethod
    def load_npz(cls, path: str | Path) -> "EquicorrPreparedData":
        """Load a file created by :meth:`save_npz`."""
        with np.load(Path(path), allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            return cls(
                sum_z=archive["sum_z"],
                sum_z2=archive["sum_z2"],
                **metadata,
            )

    def save_mmap(self, directory: str | Path) -> Path:
        """Write an mmap-friendly directory without overwriting one."""
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=False)
        np.save(target / "sum_z.npy", np.asarray(self.sum_z))
        np.save(target / "sum_z2.npy", np.asarray(self.sum_z2))
        (target / "metadata.json").write_text(
            json.dumps(
                _metadata(self), sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        return target

    @classmethod
    def load_mmap(cls, directory: str | Path) -> "EquicorrPreparedData":
        """Open mmap-friendly statistics read-only and without array copies."""
        source = Path(directory)
        metadata = json.loads(
            (source / "metadata.json").read_text(encoding="utf-8"))
        return cls(
            sum_z=np.load(
                source / "sum_z.npy", mmap_mode="r", allow_pickle=False),
            sum_z2=np.load(
                source / "sum_z2.npy", mmap_mode="r", allow_pickle=False),
            _copy_arrays=False,
            **metadata,
        )
