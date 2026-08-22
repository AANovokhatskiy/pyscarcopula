"""Persistence contracts for ``VineCopula``."""

import json
import numpy as np
import pytest

from pyscarcopula import (
    BivariateGaussianCopula,
    ClaytonCopula,
    CVineCopula,
    FrankCopula,
    GumbelCopula,
    IndependentCopula,
    RVineCopula,
    VineCopula,
    load_model,
)
from pyscarcopula.io import _to_jsonable
from pyscarcopula.vine import (
    RVineMatrix,
    cvine_structure,
    dvine_structure,
)


def _data(rows=60, dimension=5, seed=20260726):
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


def _factories():
    return (
        ("rvine", lambda: VineCopula(candidates=[IndependentCopula])),
        (
            "cvine",
            lambda: VineCopula.cvine(
                5, candidates=[IndependentCopula]),
        ),
        (
            "dvine",
            lambda: VineCopula.dvine(
                5, candidates=[IndependentCopula]),
        ),
        (
            "rvine",
            lambda: VineCopula(
                structure=_mixed_structure(),
                vine_type="rvine",
                candidates=[IndependentCopula],
            ),
        ),
    )


def test_persistence_preserves_heterogeneous_canonical_edge_mapping(
        tmp_path):
    structure = cvine_structure(4)
    specs = [
        [
            (IndependentCopula, 0),
            (BivariateGaussianCopula, 0),
            (FrankCopula, 0),
        ],
        [
            (ClaytonCopula, 0),
            (ClaytonCopula, 90),
        ],
        [
            (GumbelCopula, 180),
        ],
    ]
    vine = VineCopula(
        structure=structure,
        allow_rotations=True,
    ).fit(_data(rows=100, dimension=4), copulas=specs)
    path = tmp_path / "heterogeneous.json"
    vine.save(path, include_data=False)

    loaded = VineCopula.load(path)

    assert loaded.trees == structure.to_trees()
    for tree, level in enumerate(specs):
        for edge, (family, rotation) in enumerate(level):
            key = loaded._matrix_key(tree, edge)
            pair = loaded.pair_copulas[key]
            assert type(pair.copula) is family
            assert pair.copula.rotate == rotation
            assert pair.param == vine.pair_copulas[key].param


@pytest.mark.parametrize(("expected_type", "factory"), _factories())
def test_generic_vine_roundtrip_preserves_mode_structure_and_runtime(
        tmp_path, expected_type, factory):
    data = _data()
    vine = factory().fit(data, method="mle")
    vine._suffix_state_cache["transient"] = object()
    vine._predict_history_cache["transient"] = object()
    vine._native_rvine_cache["transient"] = object()
    vine._native_rvine_generation = 7
    path = tmp_path / f"{expected_type}.json"

    vine.save(path, include_data=True)

    envelope = json.loads(path.read_text(encoding="utf-8"))
    serialized = path.read_text(encoding="utf-8")
    assert set(envelope) == {
        "class", "format", "include_data", "state",
    }
    assert envelope["class"] == "pyscarcopula.vine.vine.VineCopula"
    assert envelope["state"]["class"] == (
        "pyscarcopula.vine.vine.VineCopula")
    assert envelope["state"]["state"]["_structure_source"] == (
        vine.structure_source)
    assert "_suffix_state_cache" not in serialized
    assert "_predict_history_cache" not in serialized
    assert "_native_rvine_cache" not in serialized
    assert "_native_rvine_generation" not in serialized

    loaded = load_model(path, expected_type=RVineCopula)
    assert type(loaded) is VineCopula
    assert loaded.vine_type == expected_type
    assert loaded.structure_source == vine.structure_source
    assert loaded.structure == vine.structure
    np.testing.assert_array_equal(
        loaded.natural_order_matrix, vine.natural_order_matrix)
    assert loaded.trees == vine.trees
    assert loaded._edge_map == vine._edge_map
    assert loaded._orig_edge_key == vine._orig_edge_key
    assert loaded.log_likelihood() == vine.log_likelihood()
    assert loaded._suffix_state_cache == {}
    assert loaded._predict_history_cache == {}
    assert loaded._native_rvine_cache == {}
    assert loaded._native_rvine_generation == 0
    assert loaded._last_u.flags.writeable is False
    np.testing.assert_array_equal(
        loaded.sample(8, rng=np.random.default_rng(9)),
        vine.sample(8, rng=np.random.default_rng(9)),
    )

    second_path = tmp_path / f"{expected_type}-second.json"
    loaded.save(second_path, include_data=True)
    second = VineCopula.load(second_path)
    assert second.vine_type == loaded.vine_type
    assert second.structure == loaded.structure
    np.testing.assert_array_equal(
        second.natural_order_matrix, loaded.natural_order_matrix)


def test_vine_include_data_false_preserves_fit_but_drops_history(tmp_path):
    data = _data(dimension=4)
    vine = VineCopula.dvine(
        4, candidates=[IndependentCopula]).fit(data)
    path = tmp_path / "without-data.json"

    vine.save(path, include_data=False)
    loaded = VineCopula.load(path)

    assert loaded._last_u is None
    assert loaded.log_likelihood() == vine.log_likelihood()
    assert loaded.predict(
        5, u=data, rng=np.random.default_rng(3)).shape == (5, 4)


def test_cvine_roundtrip_preserves_runtime_type(tmp_path):
    data = _data(dimension=3)
    legacy = CVineCopula(
        candidates=[IndependentCopula]).fit(data, method="mle")
    path = tmp_path / "cvine.json"
    legacy.save(path, include_data=True)

    loaded = load_model(path)
    assert type(loaded) is CVineCopula
    assert not isinstance(loaded, VineCopula)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        (
            "_natural_order_matrix",
            lambda: _to_jsonable(
                np.flipud(cvine_structure(4).matrix)),
            "natural-order matrix does not match trees",
        ),
        (
            "_configured_structure",
            lambda: _to_jsonable(cvine_structure(4)),
            "fixed structure does not match fitted trees",
        ),
    ),
)
def test_rejects_mismatched_persisted_structure_state(
        tmp_path, field, replacement, message):
    vine = VineCopula.dvine(
        4, candidates=[IndependentCopula]).fit(
            _data(dimension=4))
    path = tmp_path / "corrupt.json"
    vine.save(path, include_data=False)
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["state"]["state"][field] = replacement()
    path.write_text(json.dumps(envelope), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_model(path)


def test_vine_expected_type_accepts_alias_and_rejects_legacy_cvine(tmp_path):
    vine = VineCopula.cvine(
        3, candidates=[IndependentCopula]).fit(
            _data(dimension=3))
    path = tmp_path / "vine.json"
    vine.save(path)

    assert isinstance(load_model(path, expected_type=RVineCopula), VineCopula)
    with pytest.raises(TypeError, match="Expected CVineCopula"):
        load_model(path, expected_type=CVineCopula)
