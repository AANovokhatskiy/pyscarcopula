"""Static MLE parity contracts for legacy and generic C-vine runtimes."""

import warnings

import numpy as np
import pytest
from scipy.stats import norm

from pyscarcopula import (
    BivariateGaussianCopula,
    CVineCopula,
    FrankCopula,
    GumbelCopula,
    IndependentCopula,
    JoeCopula,
    VineCopula,
)


def _data(rows=420, dimension=4, seed=20260726):
    correlation = np.fromfunction(
        lambda i, j: 0.62 ** np.abs(i - j),
        (dimension, dimension),
    )
    rng = np.random.default_rng(seed)
    return norm.cdf(rng.multivariate_normal(
        np.zeros(dimension), correlation, size=rows))


def _legacy_cvine_trees(d):
    return [
        [
            (
                frozenset((tree, variable)),
                frozenset(range(tree)),
            )
            for variable in range(tree + 1, d)
        ]
        for tree in range(d - 1)
    ]


def _edge_id(edge):
    conditioned, conditioning = edge
    return (
        tuple(sorted(conditioned)),
        tuple(sorted(conditioning)),
    )


_FIXED_FAMILIES = {
    ((0, 1), ()): BivariateGaussianCopula,
    ((0, 2), ()): FrankCopula,
    ((0, 3), ()): IndependentCopula,
    ((1, 2), (0,)): BivariateGaussianCopula,
    ((1, 3), (0,)): IndependentCopula,
    ((2, 3), (0, 1)): FrankCopula,
}


def _fixed_specs(trees, family_map):
    return [
        [
            (family_map[_edge_id(edge)], 0)
            for edge in level
        ]
        for level in trees
    ]


def _legacy_semantic_edges(vine):
    return {
        _edge_id(edge): fitted
        for tree, level in enumerate(_legacy_cvine_trees(vine.d))
        for edge, fitted in zip(level, vine.edges[tree])
    }


def _generic_semantic_edges(vine):
    return {
        _edge_id(edge): vine.pair_copulas[
            vine._orig_edge_key[(tree, original_index)]
        ]
        for tree, level in enumerate(vine.trees)
        for original_index, edge in enumerate(level)
    }


def test_legacy_cvine_remains_importable_without_runtime_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = CVineCopula()

    assert isinstance(model, CVineCopula)
    assert not any(
        issubclass(item.category, DeprecationWarning)
        for item in caught
    )


def test_static_mle_structure_likelihood_and_fixed_families_match():
    data = _data()
    legacy_trees = _legacy_cvine_trees(data.shape[1])
    generic = VineCopula.cvine(
        data.shape[1],
        candidates=list(set(_FIXED_FAMILIES.values())),
        allow_rotations=False,
    )
    generic_trees = generic.structure.to_trees()

    legacy = CVineCopula(
        candidates=list(set(_FIXED_FAMILIES.values())),
        allow_rotations=False,
    ).fit(
        data,
        method="mle",
        copulas=_fixed_specs(legacy_trees, _FIXED_FAMILIES),
    )
    generic.fit(
        data,
        method="mle",
        copulas=_fixed_specs(generic_trees, _FIXED_FAMILIES),
    )

    assert [
        {_edge_id(edge) for edge in level}
        for level in generic.trees
    ] == [
        {_edge_id(edge) for edge in level}
        for level in legacy_trees
    ]
    legacy_edges = _legacy_semantic_edges(legacy)
    generic_edges = _generic_semantic_edges(generic)
    assert set(legacy_edges) == set(generic_edges)
    for edge_id in legacy_edges:
        legacy_edge = legacy_edges[edge_id]
        generic_edge = generic_edges[edge_id]
        assert type(legacy_edge.copula) is type(generic_edge.copula)
        assert legacy_edge.copula.rotate == generic_edge.copula.rotate
        assert legacy_edge.param == pytest.approx(
            generic_edge.param, abs=1e-10)
        assert legacy_edge.log_likelihood == pytest.approx(
            generic_edge.log_likelihood, abs=1e-10)

    assert legacy.log_likelihood(data) == pytest.approx(
        generic.log_likelihood(data),
        rel=1e-12,
        abs=1e-10,
    )


def test_static_gaussian_cvine_sampling_moments_match():
    data = _data()
    d = data.shape[1]
    legacy_trees = _legacy_cvine_trees(d)
    generic = VineCopula.cvine(
        d,
        candidates=[BivariateGaussianCopula],
        allow_rotations=False,
    )
    gaussian_families = {
        _edge_id(edge): BivariateGaussianCopula
        for level in legacy_trees
        for edge in level
    }
    legacy = CVineCopula(
        candidates=[BivariateGaussianCopula],
        allow_rotations=False,
    ).fit(
        data,
        method="mle",
        copulas=_fixed_specs(legacy_trees, gaussian_families),
    )
    generic.fit(
        data,
        method="mle",
        copulas=_fixed_specs(
            generic.structure.to_trees(), gaussian_families),
    )

    legacy_sample = legacy.sample(
        12_000, rng=np.random.default_rng(31))
    generic_sample = generic.sample(
        12_000, rng=np.random.default_rng(32))

    np.testing.assert_allclose(
        legacy_sample.mean(axis=0),
        generic_sample.mean(axis=0),
        atol=0.02,
        rtol=0.0,
    )
    legacy_scores = norm.ppf(np.clip(
        legacy_sample, 1e-12, 1.0 - 1e-12))
    generic_scores = norm.ppf(np.clip(
        generic_sample, 1e-12, 1.0 - 1e-12))
    np.testing.assert_allclose(
        np.corrcoef(legacy_scores, rowvar=False),
        np.corrcoef(generic_scores, rowvar=False),
        atol=0.04,
        rtol=0.0,
    )


def test_dynamic_gas_rotated_edges_match_legacy_cvine_factorization():
    data = _data(rows=90, dimension=4)
    legacy_trees = _legacy_cvine_trees(data.shape[1])
    family_specs = {
        ((0, 1), ()): (GumbelCopula, 180),
        ((0, 2), ()): (JoeCopula, 270),
        ((0, 3), ()): (FrankCopula, 0),
        ((1, 2), (0,)): (GumbelCopula, 180),
        ((1, 3), (0,)): (FrankCopula, 0),
        ((2, 3), (0, 1)): (JoeCopula, 90),
    }

    def specs(trees):
        return [
            [family_specs[_edge_id(edge)] for edge in level]
            for level in trees
        ]

    legacy = CVineCopula(
        candidates=[GumbelCopula, JoeCopula, FrankCopula],
    ).fit(
        data,
        method="gas",
        copulas=specs(legacy_trees),
        ftol=1e-9,
    )
    generic = VineCopula.cvine(
        data.shape[1],
        candidates=[GumbelCopula, JoeCopula, FrankCopula],
    )
    generic.fit(
        data,
        method="gas",
        copulas=specs(generic.structure.to_trees()),
        ftol=1e-9,
    )

    legacy_edges = _legacy_semantic_edges(legacy)
    generic_edges = _generic_semantic_edges(generic)
    for edge_id in legacy_edges:
        old = legacy_edges[edge_id]
        new = generic_edges[edge_id]
        np.testing.assert_allclose(
            old.fit_result.params.values,
            new.fit_result.params.values,
            rtol=1e-9,
            atol=1e-10,
        )
        assert old.log_likelihood == pytest.approx(
            new.log_likelihood, rel=1e-10, abs=1e-9)

    assert legacy.log_likelihood(data) == pytest.approx(
        generic.log_likelihood(data),
        rel=1e-10,
        abs=1e-8,
    )
