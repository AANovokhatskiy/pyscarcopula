"""Shared Student quantile cache for multivariate Student copulas."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import count
import weakref

import numpy as np
from scipy.special import stdtrit

from pyscarcopula._utils import clip_pseudo_observations


_CACHE_VERSIONS = count(1)


def _normalized_snapshot(u) -> np.ndarray:
    values = np.ascontiguousarray(np.asarray(u, dtype=np.float64))
    snapshot = values.copy()
    snapshot.setflags(write=False)
    return snapshot


def _interpolate_ppf_table(nodes, table, df):
    """Hermite-interpolate a node-major PPF table along its first axis."""
    if not np.isfinite(df):
        raise ValueError("df must be finite")
    if df < nodes[0] or df > nodes[-1]:
        raise ValueError(
            f"df={df} is outside PPF table range "
            f"[{nodes[0]}, {nodes[-1]}]")
    index = np.searchsorted(nodes, df, side="right") - 1
    index = max(0, min(index, len(nodes) - 2))
    lo = nodes[index]
    hi = nodes[index + 1]
    interval = hi - lo
    alpha = (df - lo) / interval
    value_lo = table[index]
    value_hi = table[index + 1]
    if df == nodes[0] or df == nodes[-1]:
        return (1.0 - alpha) * value_lo + alpha * value_hi

    lo_slope_index = index if index == 0 else index - 1
    hi_slope_index = (
        index + 1 if index + 1 == len(nodes) - 1 else index + 2)
    slope_lo = (
        (value_hi - table[lo_slope_index])
        / (nodes[index + 1] - nodes[lo_slope_index])
    )
    slope_hi = (
        (table[hi_slope_index] - value_lo)
        / (nodes[hi_slope_index] - nodes[index])
    )
    alpha2 = alpha * alpha
    alpha3 = alpha2 * alpha
    h00 = 2.0 * alpha3 - 3.0 * alpha2 + 1.0
    h10 = alpha3 - 2.0 * alpha2 + alpha
    h01 = -2.0 * alpha3 + 3.0 * alpha2
    h11 = alpha3 - alpha2
    return (
        h00 * value_lo
        + h10 * interval * slope_lo
        + h01 * value_hi
        + h11 * interval * slope_hi
    )


class StudentPPFTable:
    """Precomputed Student inverse-CDF table with smooth df interpolation."""

    def __init__(self, u, df_lo=2.005, df_hi=250.0, n_lo=120, n_hi=80):
        u_c = clip_pseudo_observations(u)
        self.u = np.ascontiguousarray(u_c, dtype=np.float64)
        nodes_lo = np.linspace(df_lo, 5.0, n_lo)
        nodes_hi = np.geomspace(5.0, df_hi, n_hi)
        self.nodes = np.unique(np.concatenate([nodes_lo, nodes_hi]))
        self.table = np.empty(
            (len(self.nodes),) + u_c.shape, dtype=np.float64)
        for index, df_value in enumerate(self.nodes):
            self.table[index] = stdtrit(df_value, u_c)

    def __call__(self, df):
        df = float(df)
        if not np.isfinite(df):
            raise ValueError("df must be finite")
        if df < self.nodes[0] or df > self.nodes[-1]:
            return stdtrit(df, self.u)
        return _interpolate_ppf_table(self.nodes, self.table, df)

    def rows(self, df, start, stop):
        """Evaluate a contiguous row block, using exact tails when required."""
        df = float(df)
        if not np.isfinite(df):
            raise ValueError("df must be finite")
        if df < self.nodes[0] or df > self.nodes[-1]:
            return stdtrit(df, self.u[start:stop])
        return _interpolate_ppf_table(
            self.nodes, self.table[:, start:stop], df)


@dataclass(frozen=True)
class StudentPPFCache:
    """Transient Student quantile cache tied to an immutable data snapshot."""

    u_shape: tuple
    ppf_nodes: np.ndarray
    ppf_table: np.ndarray
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
            np.asarray(values, dtype=np.float64))
        return (
            self.u_shape == tuple(normalized.shape)
            and np.array_equal(self.u_snapshot, normalized)
        )

    def ppf(self, df):
        return self._ppf(df)

    def block(self, n_rows, t_index=0, max_rows=None, expected_d=None):
        if self.d != self.u_shape[1]:
            raise ValueError("PPF cache dimension is inconsistent")
        if expected_d is not None and self.d != int(expected_d):
            raise ValueError(
                f"PPF cache has dimension {self.d}, "
                f"expected {int(expected_d)}")
        start = int(t_index)
        stop = start + int(n_rows)
        limit = self.u_shape[0] if max_rows is None else min(
            self.u_shape[0], int(max_rows))
        if start < 0 or stop > limit:
            raise ValueError(
                f"PPF cache block [{start}:{stop}] is outside length {limit}")
        return start, stop

    def ppf_rows(self, df, start=0, stop=None):
        if stop is None:
            stop = self.u_shape[0]
        start, stop = self.block(int(stop) - int(start), start)
        return self._ppf.rows(df, start, stop)


def prepare_student_ppf_cache(
        cached, source, u, d, table_factory=StudentPPFTable):
    """Reuse or build the single PPF cache for a pseudo-observation array."""
    normalized = np.ascontiguousarray(np.asarray(u, dtype=np.float64))
    if normalized.ndim != 2 or normalized.shape[1] != int(d):
        raise ValueError(
            f"u must have shape (T, {int(d)}), got {normalized.shape}")
    if cached is not None and cached.matches(source, normalized):
        return cached

    snapshot = _normalized_snapshot(normalized)
    table = table_factory(snapshot)
    try:
        source_ref = weakref.ref(source)
    except TypeError:
        source_ref = lambda: None
    return StudentPPFCache(
        u_shape=tuple(snapshot.shape),
        ppf_nodes=table.nodes,
        ppf_table=table.table,
        d=int(d),
        source_ref=source_ref,
        _ppf=table,
        u_snapshot=snapshot,
        version=next(_CACHE_VERSIONS),
    )
