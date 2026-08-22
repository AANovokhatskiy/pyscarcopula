"""Deterministic model builders shared by R-vine runtime tests."""

from __future__ import annotations

import numpy as np

from pyscarcopula import (
    BivariateGaussianCopula,
    ClaytonCopula,
    IndependentCopula,
    RVineCopula,
)
from pyscarcopula._types import (
    GASResult,
    IndependentResult,
    LatentResult,
    MLEResult,
    gas_params,
    ou_params,
)
from pyscarcopula.vine._pair_copula import PairCopula
from pyscarcopula.vine._rvine_matrix_builder import (
    build_rvine_matrix_with_edge_map,
)
from pyscarcopula.vine._structure import dvine_structure


def fitted_pair(copula, parameter: float) -> PairCopula:
    """Build a minimal immutable-by-convention fitted static edge."""
    if isinstance(copula, IndependentCopula):
        result = IndependentResult(
            log_likelihood=0.0,
            method="INDEPENDENT",
            copula_name=copula.name,
            success=True,
        )
        parameter = 0.0
    else:
        result = MLEResult(
            log_likelihood=0.0,
            method="MLE",
            copula_name=copula.name,
            success=True,
            copula_param=float(parameter),
        )
    return PairCopula(
        copula=copula,
        param=float(parameter),
        log_likelihood=0.0,
        nfev=0,
        tau=0.0,
        fit_result=result,
    )


def configured_mixed_family_vine() -> RVineCopula:
    """Return the d=3 mixed-family vine used by golden/replay checks."""
    trees = [
        [
            (frozenset({1, 2}), frozenset()),
            (frozenset({0, 1}), frozenset()),
        ],
        [(frozenset({0, 2}), frozenset({1}))],
    ]
    matrix, edge_map = build_rvine_matrix_with_edge_map(3, trees)
    vine = RVineCopula()
    vine.d = 3
    vine.matrix = matrix
    vine._trees = tuple(tuple(level) for level in trees)
    vine._edge_map = dict(edge_map)
    vine._orig_edge_key = {
        (tree, original): key
        for key, original in edge_map.items()
        for tree in (key[0],)
    }
    vine.pair_copulas = {
        (0, 0): fitted_pair(ClaytonCopula(rotate=90), 0.8),
        (0, 1): fitted_pair(BivariateGaussianCopula(), -0.35),
        (1, 0): fitted_pair(IndependentCopula(), 0.0),
    }
    vine._last_u = None
    vine._target_given_vars = ()
    vine._conditional_fit_supported = True
    vine._T = 0
    vine._log_likelihood = 0.0
    vine.method = "MLE"
    return vine


def configured_static_dvine(
        dimension: int, *, independent: bool = False, order=None):
    """Return a fitted-looking static D-vine without optimizer noise."""
    structure = dvine_structure(int(dimension), order=order)
    trees = structure.to_trees()
    matrix, edge_map = build_rvine_matrix_with_edge_map(dimension, trees)
    vine = RVineCopula(structure=structure, vine_type="dvine")
    vine.d = int(dimension)
    vine.matrix = matrix
    vine._structure = structure
    vine._trees = tuple(tuple(level) for level in trees)
    vine._edge_map = dict(edge_map)
    vine._orig_edge_key = {
        (tree, original): key
        for key, original in edge_map.items()
        for tree in (key[0],)
    }
    vine.pair_copulas = {}
    for tree, column in edge_map:
        if independent or tree >= 2:
            edge = fitted_pair(IndependentCopula(), 0.0)
        else:
            rho = (-1.0 if column % 2 else 1.0) * (
                0.35 / float(tree + 1))
            edge = fitted_pair(BivariateGaussianCopula(), rho)
        vine.pair_copulas[(tree, column)] = edge
    vine._last_u = None
    vine._target_given_vars = ()
    vine._conditional_fit_supported = True
    vine._T = 0
    vine._log_likelihood = 0.0
    vine.method = "MLE"
    return vine


def configured_mixed_gas_vine() -> RVineCopula:
    """Return a mixed static/GAS model exercising the existing native path."""
    vine = configured_mixed_family_vine()
    copula = BivariateGaussianCopula()
    result = GASResult(
        log_likelihood=0.0,
        method="GAS",
        copula_name=copula.name,
        success=True,
        params=gas_params(0.0, 0.6, 0.0),
        scaling="unit",
        r_last=0.0,
    )
    vine.pair_copulas[(0, 1)] = PairCopula(
        copula=copula,
        param=0.0,
        log_likelihood=0.0,
        nfev=0,
        tau=0.0,
        fit_result=result,
    )
    vine.method = "MIXED"
    return vine


def configured_mixed_scar_vine() -> RVineCopula:
    """Return a mixed static/SCAR model for backend capability tests."""
    vine = configured_mixed_family_vine()
    copula = BivariateGaussianCopula()
    result = LatentResult(
        log_likelihood=0.0,
        method="SCAR-TM-OU",
        copula_name=copula.name,
        success=True,
        params=ou_params(1.0, 0.0, 0.4),
        K=20,
        grid_range=3.0,
    )
    vine.pair_copulas[(0, 1)] = PairCopula(
        copula=copula,
        param=0.0,
        log_likelihood=0.0,
        nfev=0,
        tau=0.0,
        fit_result=result,
    )
    vine.method = "MIXED"
    return vine


def scalar_parameters(vine) -> dict[tuple[int, int], np.ndarray]:
    return {
        key: np.array([float(edge.param)], dtype=np.float64)
        for key, edge in vine.pair_copulas.items()
    }
