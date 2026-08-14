"""Deterministic fitted-vine fixtures for exact-conditioning tests.

The fixtures avoid optimizer noise: every edge has an explicit static fit
result, while the public :class:`VineCopula` prediction runtime is exercised
unchanged.  Public fit-time contracts are tested separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from pyscarcopula import (
    BivariateGaussianCopula,
    ClaytonCopula,
    FrankCopula,
    GumbelCopula,
    IndependentCopula,
    JoeCopula,
    VineCopula,
)
from pyscarcopula._types import IndependentResult, MLEResult
from pyscarcopula.vine import RVineMatrix, cvine_structure, dvine_structure
from pyscarcopula.vine._pair_copula import PairCopula
from pyscarcopula.vine._rvine_matrix_builder import (
    build_rvine_matrix_with_edge_map,
)


@dataclass(frozen=True)
class EdgeSpec:
    family: type
    rotation: int
    parameter: float


FAMILY_PARAMETERS = {
    IndependentCopula: 0.0,
    BivariateGaussianCopula: 0.45,
    ClaytonCopula: 1.35,
    GumbelCopula: 1.65,
    FrankCopula: 3.25,
    JoeCopula: 1.70,
}


def nontrivial_rvine_structure() -> RVineMatrix:
    """Return a fixed five-dimensional regular vine that is not C or D."""

    trees = [
        [
            (frozenset({0, 1}), frozenset()),
            (frozenset({1, 2}), frozenset()),
            (frozenset({1, 3}), frozenset()),
            (frozenset({3, 4}), frozenset()),
        ],
        [
            (frozenset({0, 2}), frozenset({1})),
            (frozenset({0, 3}), frozenset({1})),
            (frozenset({1, 4}), frozenset({3})),
        ],
        [
            (frozenset({2, 3}), frozenset({0, 1})),
            (frozenset({0, 4}), frozenset({1, 3})),
        ],
        [
            (frozenset({2, 4}), frozenset({0, 1, 3})),
        ],
    ]
    return RVineMatrix.from_trees(5, trees)


def exact_structure_cases() -> dict[str, RVineMatrix]:
    order = [2, 0, 4, 1, 3]
    return {
        "c-vine": cvine_structure(5, order),
        "d-vine": dvine_structure(5, order),
        "r-vine": nontrivial_rvine_structure(),
    }


def _pair(spec: EdgeSpec) -> PairCopula:
    copula = spec.family(rotate=spec.rotation)
    if spec.family is IndependentCopula:
        result = IndependentResult(
            log_likelihood=0.0,
            method="INDEPENDENT",
            copula_name=copula.name,
            success=True,
        )
    else:
        result = MLEResult(
            log_likelihood=0.0,
            method="MLE",
            copula_name=copula.name,
            success=True,
            copula_param=float(spec.parameter),
        )
    return PairCopula(
        copula=copula,
        param=float(spec.parameter),
        log_likelihood=0.0,
        nfev=0,
        tau=0.0,
        fit_result=result,
        fit_diagnostics={
            "requested_method": "MLE",
            "actual_method": result.method,
            "dynamic_attempted": False,
        },
    )


def fitted_static_vine(
    structure: RVineMatrix,
    spec_for_edge: Callable[[int, int], EdgeSpec],
) -> VineCopula:
    """Build a fully initialized static runtime from canonical edge specs."""

    structure = RVineMatrix(structure.matrix)
    trees = structure.to_trees()
    d = structure.d
    matrix, edge_map = build_rvine_matrix_with_edge_map(
        d, trees, validate=True
    )
    vine = VineCopula(structure=structure)
    vine.d = d
    vine._structure = RVineMatrix(structure.matrix)
    vine._natural_order_matrix = matrix
    vine._trees = tuple(tuple(level) for level in trees)
    vine._edge_map = dict(edge_map)
    vine._orig_edge_key = {
        (tree, original): (tree, column)
        for (tree, column), original in edge_map.items()
    }
    vine.pair_copulas = {
        (tree, column): _pair(spec_for_edge(tree, original))
        for (tree, column), original in edge_map.items()
    }
    vine._T = 0
    vine._last_u = None
    vine._log_likelihood = 0.0
    vine.method = "MLE"
    vine._target_given_vars = ()
    vine._conditional_fit_supported = True
    vine._conditional_mode = "suffix"
    vine._fit_diagnostics = None
    return vine


def homogeneous_vine(
    structure: RVineMatrix,
    family: type,
    rotation: int = 0,
    parameter: float | None = None,
) -> VineCopula:
    if parameter is None:
        parameter = FAMILY_PARAMETERS[family]
    spec = EdgeSpec(family, rotation, float(parameter))
    return fitted_static_vine(structure, lambda _tree, _edge: spec)


def gaussian_vine(structure: RVineMatrix) -> VineCopula:
    def spec(tree: int, edge: int) -> EdgeSpec:
        # Non-constant partial correlations make edge-map mistakes observable.
        magnitude = 0.16 + 0.035 * ((2 * tree + edge) % 6)
        sign = -1.0 if (tree + 2 * edge) % 4 == 1 else 1.0
        return EdgeSpec(BivariateGaussianCopula, 0, sign * magnitude)

    return fitted_static_vine(structure, spec)


def truncated_gaussian_vine(structure: RVineMatrix) -> VineCopula:
    def spec(tree: int, edge: int) -> EdgeSpec:
        if tree > 0:
            return EdgeSpec(IndependentCopula, 0, 0.0)
        return EdgeSpec(
            BivariateGaussianCopula,
            0,
            0.30 + 0.04 * (edge % 3),
        )

    return fitted_static_vine(structure, spec)


def mixed_vine(structure: RVineMatrix) -> VineCopula:
    cycle = (
        EdgeSpec(ClaytonCopula, 90, 1.25),
        EdgeSpec(GumbelCopula, 180, 1.55),
        EdgeSpec(FrankCopula, 0, 3.10),
        EdgeSpec(JoeCopula, 270, 1.65),
        EdgeSpec(BivariateGaussianCopula, 0, -0.32),
        EdgeSpec(IndependentCopula, 0, 0.0),
    )
    return fitted_static_vine(
        structure,
        lambda tree, edge: cycle[(3 * tree + edge) % len(cycle)],
    )


def peel_order(vine: VineCopula) -> list[int]:
    matrix = vine.natural_order_matrix
    return [
        int(matrix[vine.d - 1 - column, column])
        for column in range(vine.d)
    ]


def exact_given(vine: VineCopula, path: str) -> dict[int, float]:
    order = peel_order(vine)
    if path == "direct":
        variables = order[-2:]
    elif path == "rebuilt":
        variables = order[:1]
    else:
        raise ValueError(f"unknown exact suffix path {path!r}")
    levels = (0.27, 0.73)
    return {
        variable: levels[index]
        for index, variable in enumerate(variables)
    }


__all__ = [
    "EdgeSpec",
    "FAMILY_PARAMETERS",
    "exact_given",
    "exact_structure_cases",
    "fitted_static_vine",
    "gaussian_vine",
    "homogeneous_vine",
    "mixed_vine",
    "nontrivial_rvine_structure",
    "peel_order",
    "truncated_gaussian_vine",
]
