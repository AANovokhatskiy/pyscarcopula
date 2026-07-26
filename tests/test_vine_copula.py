"""Permanent contracts for the generic ``VineCopula`` runtime."""

import numpy as np
import pytest

from pyscarcopula import (
    IndependentCopula,
    RVineCopula,
    VineCopula,
)
from pyscarcopula.vine import (
    RVineMatrix,
    cvine_structure,
    dvine_structure,
)


def _data(rows=80, dimension=5, seed=20260726):
    return np.random.default_rng(seed).uniform(
        0.01, 0.99, size=(rows, dimension))


def _mixed_structure():
    return RVineMatrix.from_trees(5, [
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
    ])


def test_rvine_copula_is_an_identity_alias():
    assert RVineCopula is VineCopula
    assert isinstance(VineCopula(), RVineCopula)
    assert isinstance(RVineCopula(), VineCopula)


def test_named_factories_configure_expected_structure_mode():
    c_vine = VineCopula.cvine(4, order=[2, 0, 3, 1])
    d_vine = VineCopula.dvine(4, order=[2, 0, 3, 1])
    r_vine = VineCopula.rvine()

    assert c_vine.structure == cvine_structure(4, [2, 0, 3, 1])
    assert d_vine.structure == dvine_structure(4, [2, 0, 3, 1])
    assert r_vine.structure is None
    assert c_vine.structure_source == d_vine.structure_source == "fixed"
    assert r_vine.structure_source == "auto"


@pytest.mark.parametrize(
    ("structure", "label"),
    (
        (cvine_structure(5), "C-vine"),
        (dvine_structure(5), "D-vine"),
        (_mixed_structure(), "regular vine"),
    ),
)
def test_fixed_c_d_and_arbitrary_structures_share_runtime(
        monkeypatch, structure, label):
    def unexpected_selection(*args, **kwargs):
        raise AssertionError("Dissmann selection must not run")

    monkeypatch.setattr(
        "pyscarcopula.vine.vine.select_rvine", unexpected_selection)
    vine = VineCopula(
        structure=structure,
        candidates=[IndependentCopula],
    ).fit(_data(dimension=5))

    assert vine.structure == structure
    assert vine.structure is not structure
    assert vine.structure_label == label
    assert vine.log_likelihood() == 0.0
    assert vine.sample(6, rng=np.random.default_rng(7)).shape == (6, 5)
    assert "structure_source='fixed'" in vine.summary(as_string=True)


def test_fixed_structure_dimension_must_match_data():
    with pytest.raises(ValueError, match="structure dimension 4"):
        VineCopula(structure=dvine_structure(4)).fit(
            _data(dimension=3))


@pytest.mark.parametrize("structure", [object(), np.eye(3, dtype=int)])
def test_constructor_rejects_non_rvine_matrix_structure(structure):
    with pytest.raises(TypeError, match="RVineMatrix or None"):
        VineCopula(structure=structure)


def test_structure_and_matrix_state_are_defensive_and_refit_clears_caches():
    source = dvine_structure(4)
    vine = VineCopula(
        structure=source,
        candidates=[IndependentCopula],
    )
    source_matrix = source.matrix
    source_matrix[3, 0] = 99
    assert vine.structure == dvine_structure(4)

    vine.fit(_data(dimension=4))
    returned_structure = vine.structure
    returned_matrix = returned_structure.matrix
    returned_matrix[3, 0] = 99
    compatibility_matrix = vine.matrix
    compatibility_matrix[0, 0] = 99
    assert vine.structure == dvine_structure(4)
    assert vine.matrix[0, 0] != 99

    vine._suffix_state_cache["stale"] = object()
    vine._predict_history_cache["stale"] = object()
    vine.fit(_data(dimension=4, seed=20260727))
    assert vine._suffix_state_cache == {}
    assert vine._predict_history_cache == {}

