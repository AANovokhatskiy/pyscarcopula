"""Shared Student quantile cache for multivariate Student copulas."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import count
import weakref

import numpy as np
from pyscarcopula._native import multivariate as native_multivariate
from pyscarcopula.numerical._arrays import as_float64_array, validate_integer


_CACHE_VERSIONS = count(1)

# Upper bound for the precomputed PPF table (nodes × T × d × 8 bytes).
# Above the limit the values table is skipped. Native calls use exact
# Student quantiles; dynamic specs retain the nodes so they can use
# the controlled large-df asymptotic above the final node.
DEFAULT_MAX_TABLE_BYTES = 256 * 1024 ** 2


def _normalized_snapshot(u) -> np.ndarray:
    values = np.ascontiguousarray(as_float64_array(u, name="u"))
    snapshot = values.copy()
    snapshot.setflags(write=False)
    return snapshot


def _interpolate_ppf_table(nodes, table, df):
    """Hermite-interpolate a node-major PPF table along its first axis."""
    return native_multivariate.interpolate_student_ppf_table(nodes, table, df)


class StudentPPFTable:
    """Precomputed Student inverse-CDF table with smooth df interpolation.

    The table has shape ``(n_df_nodes, T, d)`` and costs
    ``n_nodes * T * d * 8`` bytes. When that estimate exceeds
    ``max_table_bytes`` (default ``DEFAULT_MAX_TABLE_BYTES``, 256 MiB) the
    table is not built (``table is None``) and the native evaluator uses
    the exact Student quantile. Nodes include a dense boundary layer
    at the model's ``2 + 1e-6`` df limit and extend to 1000, above which the
    native dynamic-emission kernel uses a controlled normal asymptotic.
    """

    def __init__(self, u, df_lo=None, df_hi=None,
                 n_boundary=None, n_lo=None, n_hi=None,
                 max_table_bytes=DEFAULT_MAX_TABLE_BYTES):
        self.u, self.nodes, self.table = native_multivariate.prepare_student_ppf_table(
            u, df_lo=df_lo, df_hi=df_hi, n_boundary=n_boundary,
            n_lo=n_lo, n_hi=n_hi, max_table_bytes=max_table_bytes)
        self.u.setflags(write=False)
        self.nodes.setflags(write=False)
        if self.table is not None:
            self.table.setflags(write=False)

    def __call__(self, df):
        return native_multivariate.evaluate_student_ppf_table(
            self.u, self.nodes, self.table, df)

    def rows(self, df, start, stop):
        """Evaluate a contiguous row block, using exact tails when required."""
        start = validate_integer(start, "start")
        stop = validate_integer(stop, "stop")
        if stop < start or stop > len(self.u):
            raise ValueError(
                f"PPF table block [{start}:{stop}] is outside length {len(self.u)}")
        return native_multivariate.evaluate_student_ppf_table(
            self.u, self.nodes, self.table, df, start, stop)


@dataclass(frozen=True)
class StudentPPFCache:
    """Transient Student quantile cache tied to an immutable data snapshot."""

    u_shape: tuple
    ppf_nodes: np.ndarray
    ppf_table: np.ndarray | None
    d: int
    source_ref: object
    _ppf: object
    u_snapshot: np.ndarray | None = None
    version: int = 0

    def matches(self, source, values) -> bool:
        del source  # Source identity is diagnostic only, never correctness.
        if self.u_snapshot is None:
            return False
        normalized = np.ascontiguousarray(
            as_float64_array(values, name="values"))
        return (
            self.u_shape == tuple(normalized.shape)
            and np.array_equal(self.u_snapshot, normalized)
        )

    def ppf(self, df):
        return self._ppf(df)

    def block(self, n_rows, t_index=0, max_rows=None, expected_d=None):
        n_rows = validate_integer(n_rows, "n_rows")
        start = validate_integer(t_index, "t_index")
        if max_rows is not None:
            max_rows = validate_integer(max_rows, "max_rows")
        if expected_d is not None:
            expected_d = validate_integer(expected_d, "expected_d", minimum=1)
        if self.d != self.u_shape[1]:
            raise ValueError("PPF cache dimension is inconsistent")
        if expected_d is not None and self.d != expected_d:
            raise ValueError(
                f"PPF cache has dimension {self.d}, "
                f"expected {expected_d}")
        stop = start + n_rows
        limit = self.u_shape[0] if max_rows is None else min(
            self.u_shape[0], max_rows)
        if stop > limit:
            raise ValueError(
                f"PPF cache block [{start}:{stop}] is outside length {limit}")
        return start, stop

    def ppf_rows(self, df, start=0, stop=None):
        start = validate_integer(start, "start")
        if stop is None:
            stop = self.u_shape[0]
        stop = validate_integer(stop, "stop")
        start, stop = self.block(stop - start, start)
        return self._ppf.rows(df, start, stop)


def prepare_student_ppf_cache(
        cached, source, u, d, table_factory=StudentPPFTable):
    """Reuse or build the single PPF cache for a pseudo-observation array."""
    d = validate_integer(d, "d", minimum=1)
    normalized = np.ascontiguousarray(as_float64_array(u, name="u"))
    if normalized.ndim != 2 or normalized.shape[1] != d:
        raise ValueError(
            f"u must have shape (T, {d}), got {normalized.shape}")
    if cached is not None and cached.matches(source, normalized):
        return cached

    snapshot = _normalized_snapshot(normalized)
    table = table_factory(snapshot)
    table.nodes.setflags(write=False)
    if table.table is not None:
        table.table.setflags(write=False)
    try:
        source_ref = weakref.ref(source)
    except TypeError:
        source_ref = lambda: None
    return StudentPPFCache(
        u_shape=tuple(snapshot.shape),
        ppf_nodes=table.nodes,
        ppf_table=table.table,
        d=d,
        source_ref=source_ref,
        _ppf=table,
        u_snapshot=snapshot,
        version=next(_CACHE_VERSIONS),
    )
