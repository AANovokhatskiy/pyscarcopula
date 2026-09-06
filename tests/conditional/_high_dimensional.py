"""Deterministic ``d=50`` fixtures for conditional-sampling validation.

The helpers in this module configure known-parameter models.  Optimizer
recovery is deliberately outside the high-dimensional target: the tests isolate the
high-dimensional sampling kernels and compare them with independent
closed-form oracles from :mod:`tests.conditional._analytical_oracles`.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from pyscarcopula import (
    BivariateGaussianCopula,
    ClaytonCopula,
    FrankCopula,
    GaussianCopula,
    GumbelCopula,
    IndependentCopula,
    JoeCopula,
    StudentCopula,
)
from pyscarcopula.vine import RVineMatrix, cvine_structure, dvine_structure

from ._vine_fixtures import EdgeSpec, fitted_static_vine, peel_order


DIMENSION = 50
FREE_COUNTS = (1, 3, 5, 10)


def _validate_free_count(k_free: int) -> int:
    value = int(k_free)
    if value not in FREE_COUNTS:
        raise ValueError(f"k_free must be one of {FREE_COUNTS}")
    return value


def scattered_free_indices(k_free: int) -> np.ndarray:
    """Return deterministic labels spread across all 50 coordinates."""

    k_free = _validate_free_count(k_free)
    # ``linspace`` is unique for these four counts and includes both tails.
    return np.rint(np.linspace(1, DIMENSION - 2, k_free)).astype(np.int64)


def given_for_free_indices(free_indices) -> dict[int, float]:
    free = {int(index) for index in np.asarray(free_indices).ravel()}
    given_indices = [index for index in range(DIMENSION) if index not in free]
    levels = np.linspace(0.07, 0.93, len(given_indices), dtype=np.float64)
    return {
        index: float(level)
        for index, level in zip(given_indices, levels, strict=True)
    }


def scattered_given(k_free: int) -> dict[int, float]:
    return given_for_free_indices(scattered_free_indices(k_free))


@lru_cache(maxsize=None)
def dense_correlation(kind: str) -> np.ndarray:
    """Return an SPD AR(1), block, or moderately near-singular matrix."""

    indices = np.arange(DIMENSION)
    if kind == "ar1":
        matrix = 0.64 ** np.abs(indices[:, None] - indices[None, :])
    elif kind == "block":
        same_block = (indices[:, None] // 10) == (indices[None, :] // 10)
        matrix = np.full((DIMENSION, DIMENSION), 0.04, dtype=np.float64)
        matrix[same_block] = 0.56
        np.fill_diagonal(matrix, 1.0)
    elif kind == "near-singular-moderate":
        matrix = np.full((DIMENSION, DIMENSION), 0.92, dtype=np.float64)
        np.fill_diagonal(matrix, 1.0)
    else:
        raise ValueError(f"unknown dense correlation fixture {kind!r}")
    matrix = np.asarray(matrix, dtype=np.float64)
    np.linalg.cholesky(matrix)
    matrix.setflags(write=False)
    return matrix


@lru_cache(maxsize=None)
def factor_loadings(rank: int) -> np.ndarray:
    """Return deterministic compact loadings with a non-degenerate core."""

    rank = int(rank)
    if rank not in (1, 3, 8):
        raise ValueError("rank must be 1, 3, or 8")
    row = np.arange(1, DIMENSION + 1, dtype=np.float64)[:, None]
    column = np.arange(1, rank + 1, dtype=np.float64)[None, :]
    raw = np.sin(0.17 * row * column) + 0.55 * np.cos(
        0.11 * row * (column + 1.0)
    )
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    magnitude = 0.43 + 0.08 * np.sin(0.13 * row)
    loadings = magnitude * raw
    assert float(np.max(np.sum(loadings * loadings, axis=1))) < 0.30
    loadings.setflags(write=False)
    return loadings


def factor_correlation(rank: int) -> np.ndarray:
    loadings = factor_loadings(rank)
    correlation = loadings @ loadings.T
    np.fill_diagonal(correlation, 1.0)
    np.linalg.cholesky(correlation)
    return correlation


@lru_cache(maxsize=None)
def configured_gaussian(kind: str, rank: int | None = None) -> GaussianCopula:
    """Build a fitted Gaussian runtime without estimating its known matrix."""

    if kind == "factor":
        if rank is None:
            raise ValueError("factor Gaussian requires rank")
        model = GaussianCopula(
            DIMENSION,
            corr_mode="factor",
            factor_rank=rank,
            factor_loadings=factor_loadings(rank),
            factor_tile_size=13,
        )
    else:
        if rank is not None:
            raise ValueError("rank is only valid for factor Gaussian")
        model = GaussianCopula(DIMENSION, R=dense_correlation(kind))
    training = np.random.default_rng(20267001).uniform(
        0.04, 0.96, size=(12, DIMENSION)
    )
    result = model.fit(training)
    assert result.success, result.message
    return model


def configured_student(
    *,
    df: float,
    kind: str = "ar1",
    rank: int | None = None,
) -> StudentCopula:
    """Build a known-parameter static Student runtime.

    ``StudentCopula`` has no public fixed-df constructor.  Setting ``df`` is
    the smallest deterministic fixture setup and keeps the public
    ``sample_conditional`` entrypoint under test.  Correlation construction
    and factor preparation still use the production constructors.
    """

    if rank is None:
        correlation = dense_correlation(kind)
        model = StudentCopula(DIMENSION, R=correlation)
        model.shape = correlation
    else:
        model = StudentCopula(
            DIMENSION,
            corr_mode="factor",
            factor_rank=rank,
            factor_loadings=factor_loadings(rank),
            factor_tile_size=13,
        )
    model.df = float(df)
    return model


def _clusters_to_edge(left: frozenset[int], right: frozenset[int]):
    conditioning = left & right
    conditioned = left ^ right
    return frozenset(conditioned), frozenset(conditioning)


def _lexicographic_spanning_tree(nodes: list[frozenset[int]]):
    """Select a deterministic spanning tree of the proximity graph."""

    selected = {0}
    edges: list[tuple[int, int]] = []
    while len(selected) < len(nodes):
        candidates = []
        for left in sorted(selected):
            for right in range(len(nodes)):
                if right in selected:
                    continue
                if len(nodes[left] & nodes[right]) == len(nodes[left]) - 1:
                    candidates.append((tuple(sorted(nodes[right])), left, right))
        if not candidates:
            raise ValueError("proximity graph is disconnected")
        _label, left, right = min(candidates)
        edges.append((left, right))
        selected.add(right)
    return edges


@lru_cache(maxsize=1)
def regular_vine_structure() -> RVineMatrix:
    """Return a deterministic 50D regular vine that is neither C nor D."""

    # A path with several side branches is a tree, but neither a star nor a
    # path.  Higher trees are deterministic spanning trees of successive
    # proximity graphs.
    first_pairs = [(0, 1), (1, 2), (1, 3), (3, 4)]
    first_pairs.extend((value - 1, value) for value in range(5, DIMENSION))
    nodes = [frozenset((left, right)) for left, right in first_pairs]
    trees = [[
        (frozenset((left, right)), frozenset())
        for left, right in first_pairs
    ]]
    while len(nodes) > 1:
        links = _lexicographic_spanning_tree(nodes)
        trees.append([
            _clusters_to_edge(nodes[left], nodes[right])
            for left, right in links
        ])
        nodes = [nodes[left] | nodes[right] for left, right in links]
    return RVineMatrix.from_trees(DIMENSION, trees)


@lru_cache(maxsize=None)
def high_dimensional_structure(kind: str) -> RVineMatrix:
    order = list(range(DIMENSION))
    if kind == "c-vine":
        return cvine_structure(DIMENSION, order)
    if kind == "d-vine":
        return dvine_structure(DIMENSION, order)
    if kind == "r-vine":
        return regular_vine_structure()
    raise ValueError(f"unknown vine structure {kind!r}")


def _gaussian_edge_spec(tree: int, edge: int) -> EdgeSpec:
    magnitude = 0.08 + 0.012 * ((3 * tree + edge) % 9)
    sign = -1.0 if (tree + edge) % 5 == 1 else 1.0
    return EdgeSpec(BivariateGaussianCopula, 0, sign * magnitude)


@lru_cache(maxsize=None)
def high_dimensional_gaussian_vine(kind: str):
    return fitted_static_vine(
        high_dimensional_structure(kind), _gaussian_edge_spec
    )


@lru_cache(maxsize=1)
def high_dimensional_mixed_truncated_vine():
    cycle = (
        EdgeSpec(ClaytonCopula, 90, 0.85),
        EdgeSpec(GumbelCopula, 180, 1.25),
        EdgeSpec(FrankCopula, 0, 1.75),
        EdgeSpec(JoeCopula, 270, 1.20),
        EdgeSpec(BivariateGaussianCopula, 0, -0.18),
    )

    def spec(tree: int, edge: int) -> EdgeSpec:
        if tree >= 2:
            return EdgeSpec(IndependentCopula, 0, 0.0)
        return cycle[(2 * tree + edge) % len(cycle)]

    return fitted_static_vine(regular_vine_structure(), spec)


def suffix_given(vine, k_free: int) -> dict[int, float]:
    """Choose exact-suffix ``given`` while scattering public variable labels."""

    k_free = _validate_free_count(k_free)
    order = peel_order(vine)
    given_variables = order[k_free:]
    # Repeated h/h-inverse recursion in a 50D vine can magnify even moderate
    # non-zero latent means.  Tail and varied-level contracts are covered in
    # the low-dimensional suites; this gate isolates the full conditional
    # covariance without open-unit clipping.
    levels = np.full(len(given_variables), 0.5, dtype=np.float64)
    return {
        int(variable): float(level)
        for variable, level in zip(given_variables, levels, strict=True)
    }


__all__ = [
    "DIMENSION",
    "FREE_COUNTS",
    "configured_gaussian",
    "configured_student",
    "dense_correlation",
    "factor_correlation",
    "factor_loadings",
    "given_for_free_indices",
    "high_dimensional_gaussian_vine",
    "high_dimensional_mixed_truncated_vine",
    "high_dimensional_structure",
    "regular_vine_structure",
    "scattered_free_indices",
    "scattered_given",
    "suffix_given",
]
