"""Regressions for the remaining stage-5 Vine parameter-routing defects."""

import numpy as np
import pytest

from pyscarcopula import (
    BivariateGaussianCopula,
    IndependentCopula,
    VineCopula,
    api,
)


def _data(dimension=2):
    return np.random.default_rng(20260901).uniform(
        0.05, 0.95, size=(24, dimension))


def _fit_entry(vine, data, entry, **kwargs):
    if entry == "object":
        return vine.fit(data, **kwargs)
    return api.fit(vine, data, **kwargs)


@pytest.mark.parametrize("entry", ["object", "api"])
@pytest.mark.parametrize(
    "method", ["MLE", "GAS", "SCAR-TM-OU", "SCAR-TM-JACOBI"])
@pytest.mark.parametrize("external", [object(), None], ids=["object", "none"])
@pytest.mark.parametrize("short_path", ["independent", "truncated"])
def test_public_vine_fit_rejects_external_initial_mle_result_early(
        entry, method, external, short_path):
    vine = VineCopula.cvine(2, candidates=[IndependentCopula])
    fit_kwargs = {}
    if short_path == "truncated":
        fit_kwargs["truncation_level"] = 0

    with pytest.raises(
            TypeError,
            match="initial_mle_result is an internal per-edge argument"):
        _fit_entry(
            vine,
            _data(),
            entry,
            method=method,
            initial_mle_result=external,
            **fit_kwargs,
        )

    assert vine.fit_result is None
    assert vine.pair_copulas is None
    assert not hasattr(vine, "_last_u")


@pytest.mark.parametrize("entry", ["object", "api"])
def test_public_vine_fit_rejects_external_initial_mle_result_on_active_edge(
        entry):
    vine = VineCopula.cvine(
        2,
        candidates=[BivariateGaussianCopula],
        allow_rotations=False,
    )

    with pytest.raises(TypeError, match="initial_mle_result"):
        _fit_entry(
            vine,
            _data(),
            entry,
            method="GAS",
            copulas=[[(BivariateGaussianCopula, 0)]],
            initial_mle_result=object(),
        )

    assert vine.fit_result is None
    assert vine.pair_copulas is None


def test_rejected_external_initial_mle_result_preserves_fitted_state():
    vine = VineCopula.cvine(
        2, candidates=[IndependentCopula]).fit(_data(), method="MLE")
    original_matrix = vine.matrix
    original_fit_result = vine.fit_result
    original_pairs = vine.pair_copulas
    original_data = vine._last_u
    vine._native_rvine_cache["sentinel"] = object()
    generation = vine._native_rvine_generation

    with pytest.raises(TypeError, match="initial_mle_result"):
        vine.fit(
            np.array([[np.nan, 0.5]]),
            method="GAS",
            initial_mle_result=object(),
        )

    np.testing.assert_array_equal(vine.matrix, original_matrix)
    assert vine.fit_result is original_fit_result
    assert vine.pair_copulas is original_pairs
    assert vine._last_u is original_data
    assert "sentinel" in vine._native_rvine_cache
    assert vine._native_rvine_generation == generation


@pytest.mark.parametrize("entry", ["object", "api"])
@pytest.mark.parametrize("method", ["GAS", "SCAR-TM-OU"])
@pytest.mark.parametrize("backend", ["python", None], ids=["value", "none"])
@pytest.mark.parametrize("short_path", ["independent", "truncated"])
def test_removed_backend_is_rejected_before_independent_short_path(
        entry, method, backend, short_path):
    vine = VineCopula.cvine(2, candidates=[IndependentCopula])
    fit_kwargs = {}
    if short_path == "truncated":
        fit_kwargs["truncation_level"] = 0

    with pytest.raises(TypeError, match="backend selection was removed"):
        _fit_entry(
            vine,
            _data(),
            entry,
            method=method,
            backend=backend,
            **fit_kwargs,
        )

    assert vine.fit_result is None
    assert vine.pair_copulas is None
    assert not hasattr(vine, "_last_u")


def test_custom_strategy_backend_is_not_preflight_instantiated(monkeypatch):
    from pyscarcopula.strategy import _base

    constructor_calls = []

    class CustomStrategy:
        def __init__(self, config, backend=None):
            constructor_calls.append((config, backend))

        def fit(self, copula, u):
            raise AssertionError("independent short path must skip custom fit")

    method = "TEST-BACKEND-CONTRACT"
    monkeypatch.setitem(_base._REGISTRY, method, CustomStrategy)
    _base._strategy_keyword_contract.cache_clear()
    try:
        VineCopula.cvine(
            2, candidates=[IndependentCopula],
        ).fit(
            _data(),
            method=method,
            truncation_level=0,
            backend="custom",
        )
    finally:
        _base._strategy_keyword_contract.cache_clear()

    assert constructor_calls == []


@pytest.mark.parametrize("include_data", [True, False])
@pytest.mark.parametrize(("factory", "dimension", "order"), [
    ("cvine", 3, [2, 0, 1]),
    ("cvine", 5, [4, 1, 3, 0, 2]),
    ("dvine", 5, [3, 0, 4, 1, 2]),
])
def test_fixed_nondefault_order_roundtrips(
        tmp_path, include_data, factory, dimension, order):
    configured = getattr(VineCopula, factory)(
        dimension,
        order=order,
        candidates=[IndependentCopula],
    )
    specs = [
        [(IndependentCopula, 0)] * (dimension - tree - 1)
        for tree in range(dimension - 1)
    ]
    vine = configured.fit(
        _data(dimension), method="MLE", copulas=specs)
    path = tmp_path / f"ordered-{factory}-{dimension}-{include_data}.json"

    vine.save(path, include_data=include_data)
    restored = VineCopula.load(path)

    assert restored._configured_structure == vine._configured_structure
    assert restored.structure == vine.structure
    assert restored.trees == vine.trees
    np.testing.assert_array_equal(restored.matrix, vine.matrix)
    assert (restored._last_u is not None) is include_data


def test_legacy_matrix_setter_validates_before_mutating_runtime_state():
    vine = VineCopula.rvine(candidates=[IndependentCopula])
    vine.d = 3
    valid_natural_order = np.array([
        [2, 2, 2],
        [0, 0, 0],
        [1, 0, 0],
    ], dtype=np.int64)

    vine.matrix = valid_natural_order
    valid_natural_order[0, 0] = 99
    np.testing.assert_array_equal(
        vine.matrix,
        np.array([[2, 2, 2], [0, 0, 0], [1, 0, 0]], dtype=np.int64),
    )

    original = vine.matrix
    vine._native_rvine_cache["sentinel"] = object()
    generation = vine._native_rvine_generation
    invalid_values = (
        np.array([0, 1, 2], dtype=np.int64),
        np.zeros((2, 3), dtype=np.int64),
        np.zeros((1, 1), dtype=np.int64),
        np.zeros((2, 2), dtype=np.float64),
        np.zeros((2, 2), dtype=np.bool_),
        np.zeros((3, 3), dtype=np.int64),
        np.array([[1, 1], [0, 0]], dtype=np.int64),
    )

    for invalid in invalid_values:
        with pytest.raises((TypeError, ValueError)):
            vine.matrix = invalid
        np.testing.assert_array_equal(vine.matrix, original)
        assert "sentinel" in vine._native_rvine_cache
        assert vine._native_rvine_generation == generation


def test_legacy_matrix_setter_cannot_desynchronize_fitted_state():
    fitted = VineCopula.cvine(
        3,
        candidates=[IndependentCopula],
    ).fit(_data(3), method="MLE")
    alternative = VineCopula.dvine(
        3,
        candidates=[IndependentCopula],
    ).fit(_data(3), method="MLE").matrix
    original = fitted.matrix
    fitted._native_rvine_cache["sentinel"] = object()
    generation = fitted._native_rvine_generation

    for invalid in (None, alternative):
        with pytest.raises(
                ValueError,
                match="existing (?:fitted matrix|runtime trees)"):
            fitted.matrix = invalid
        np.testing.assert_array_equal(fitted.matrix, original)
        assert "sentinel" in fitted._native_rvine_cache
        assert fitted._native_rvine_generation == generation


def test_legacy_matrix_setter_matches_partially_assembled_runtime_trees():
    from pyscarcopula.vine._rvine_matrix_builder import (
        build_rvine_matrix_with_edge_map,
    )
    from pyscarcopula.vine._structure import cvine_structure, dvine_structure

    dimension = 4
    trees = cvine_structure(dimension).to_trees()
    expected, edge_map = build_rvine_matrix_with_edge_map(dimension, trees)
    alternative, _ = build_rvine_matrix_with_edge_map(
        dimension, dvine_structure(dimension).to_trees())
    assert not np.array_equal(expected, alternative)

    vine = VineCopula.rvine(candidates=[IndependentCopula])
    vine.d = dimension
    vine._trees = tuple(tuple(level) for level in trees)
    vine._edge_map = dict(edge_map)
    vine.pair_copulas = {}
    vine._native_rvine_cache["sentinel"] = object()
    generation = vine._native_rvine_generation

    with pytest.raises(ValueError, match="does not match existing runtime trees"):
        vine.matrix = alternative

    assert vine.matrix is None
    assert "sentinel" in vine._native_rvine_cache
    assert vine._native_rvine_generation == generation

    vine._edge_map = {(0, 0): 999}
    with pytest.raises(ValueError, match="edge map does not match runtime trees"):
        vine.matrix = expected

    assert vine.matrix is None
    assert "sentinel" in vine._native_rvine_cache
    assert vine._native_rvine_generation == generation

    vine._edge_map = dict(edge_map)
    vine.matrix = expected
    np.testing.assert_array_equal(vine.matrix, expected)
    assert vine._native_rvine_cache == {}
    assert vine._native_rvine_generation == generation + 1
