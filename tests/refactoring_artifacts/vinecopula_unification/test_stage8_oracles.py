"""Independent release-gate oracles for the VineCopula unification.

These tests are refactoring evidence rather than permanent public contracts.
They may be removed with the other ``vinecopula_unification`` artifacts after
the release gate has been accepted.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm, rankdata

from pyscarcopula import (
    BivariateGaussianCopula,
    GumbelCopula,
    VineCopula,
)
from pyscarcopula.vine import RVineMatrix


pytestmark = pytest.mark.validation


def _pobs(values):
    values = np.asarray(values)
    return np.apply_along_axis(rankdata, 0, values) / (len(values) + 1)


def _positive_dependence_data(rows=400, dimension=4, seed=20260728):
    rng = np.random.default_rng(seed)
    common = rng.standard_normal((rows, 1))
    noise = rng.standard_normal((rows, dimension))
    raw = 0.7 * common + np.sqrt(1.0 - 0.7**2) * noise
    return np.asarray(_pobs(raw), dtype=np.float64, order="F")


def _decode_complete_pyvine_structure(structure):
    """Decode a full structure without touching absent truncated rows."""
    if structure.trunc_lvl != structure.dim - 1:
        raise ValueError("oracle requires a complete pyvinecopulib structure")
    order = [int(value) - 1 for value in structure.order]
    return [
        [
            (
                frozenset((
                    int(structure.struct_array(tree, edge, False)) - 1,
                    order[edge],
                )),
                frozenset(
                    int(structure.struct_array(level, edge, False)) - 1
                    for level in range(tree)
                ),
            )
            for edge in range(structure.dim - 1 - tree)
        ]
        for tree in range(structure.dim - 1)
    ]


def _normalized_edges(trees):
    return [
        {
            (tuple(sorted(pair)), tuple(sorted(conditioning)))
            for pair, conditioning in level
        }
        for level in trees
    ]


@pytest.mark.parametrize(
    ("pv_family_name", "pysca_family", "tolerance"),
    [
        ("gaussian", BivariateGaussianCopula, 2e-5),
        ("gumbel", GumbelCopula, 2e-4),
    ],
)
def test_fixed_dvine_likelihood_matches_pyvinecopulib(
        pv_family_name, pysca_family, tolerance):
    pv = pytest.importorskip("pyvinecopulib")
    family = getattr(pv.BicopFamily, pv_family_name)
    u = _positive_dependence_data()
    d = u.shape[1]
    pv_structure = pv.DVineStructure([1, 3, 2, 4])
    trees = _decode_complete_pyvine_structure(pv_structure)
    structure = RVineMatrix.from_trees(d, trees)

    pv_pairs = [
        [pv.Bicop(family, 0) for _ in level]
        for level in trees
    ]
    pv_model = pv.Vinecop.from_structure(
        structure=pv_structure,
        pair_copulas=pv_pairs,
    )
    pv_model.fit(
        u,
        controls=pv.FitControlsBicop(preselect_families=False),
        num_threads=1,
    )

    specs = [
        [(pysca_family, 0) for _ in level]
        for level in trees
    ]
    pysca_model = VineCopula(structure=structure).fit(
        u,
        method="mle",
        copulas=specs,
    )

    assert _normalized_edges(pysca_model.trees) == _normalized_edges(trees)
    actual = pysca_model.log_likelihood(u) / len(u)
    expected = pv_model.loglik(u) / len(u)
    assert actual == pytest.approx(expected, abs=tolerance)


def test_dvine_likelihood_is_invariant_under_variable_permutation():
    u = _positive_dependence_data(rows=300)
    original_order = [0, 2, 1, 3]
    permutation = np.array([2, 0, 3, 1])
    inverse = np.argsort(permutation)
    permuted_u = u[:, permutation]
    permuted_order = [int(inverse[value]) for value in original_order]

    original = VineCopula.dvine(
        d=4,
        order=original_order,
        candidates=[BivariateGaussianCopula],
        allow_rotations=False,
    ).fit(u, method="mle")
    permuted = VineCopula.dvine(
        d=4,
        order=permuted_order,
        candidates=[BivariateGaussianCopula],
        allow_rotations=False,
    ).fit(permuted_u, method="mle")

    assert original.log_likelihood(u) == pytest.approx(
        permuted.log_likelihood(permuted_u),
        rel=2e-9,
        abs=2e-8,
    )


def test_fixed_gaussian_dvine_density_normalizes_by_monte_carlo():
    rng = np.random.default_rng(20260729)
    correlation = np.fromfunction(
        lambda i, j: 0.35 ** np.abs(i - j),
        (3, 3),
    )
    training = norm.cdf(rng.multivariate_normal(
        np.zeros(3),
        correlation,
        size=500,
    ))
    model = VineCopula.dvine(
        d=3,
        candidates=[BivariateGaussianCopula],
        allow_rotations=False,
    ).fit(training, method="mle")

    integration_points = rng.uniform(
        1e-6,
        1.0 - 1e-6,
        size=(100_000, 3),
    )
    parameters = {
        key: np.array([float(edge.param)])
        for key, edge in model.pair_copulas.items()
    }
    density = np.exp(model._log_pdf_rows_with_r(
        integration_points,
        parameters,
    ))

    assert np.all(np.isfinite(density))
    assert np.mean(density) == pytest.approx(1.0, abs=0.02)
