"""Stage-1 common syntax and behavioral contracts."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest

from pyscarcopula.api import predict as api_predict
from pyscarcopula._types import PredictConfig

from ._adapters import draw_conditional
from ._runtime import (
    RUNTIME_IDS,
    RuntimeHarness,
    build_runtime,
    build_unfitted_runtime,
)


MULTIVARIATE_IDS = (
    "multivariate-gaussian",
    "multivariate-student",
    "multivariate-equicorr-gaussian",
    "multivariate-stochastic-student",
)


@pytest.fixture(scope="module", params=RUNTIME_IDS, ids=RUNTIME_IDS)
def runtime(request) -> RuntimeHarness:
    return build_runtime(request.param)


def _draw(
        runtime: RuntimeHarness,
        n: int,
        given,
        *,
        seed: int = 20260820,
        rng=None):
    if rng is None:
        rng = np.random.default_rng(seed)
    return draw_conditional(
        runtime.model,
        n,
        given,
        u_train=runtime.u_train,
        parameter_path=runtime.parameter_path,
        rng=rng,
    )


def _assert_sample_contract(
        samples: np.ndarray,
        runtime: RuntimeHarness,
        n: int,
        given: Mapping[int, float] | None) -> None:
    assert type(samples) is np.ndarray
    assert samples.shape == (n, runtime.dimension)
    assert samples.dtype == np.float64
    assert np.all(np.isfinite(samples))
    assert np.all((samples > 0.0) & (samples < 1.0))
    for index, value in (given or {}).items():
        np.testing.assert_array_equal(
            samples[:, index], np.full(n, value, dtype=np.float64)
        )
    assert not np.shares_memory(samples, runtime.u_train)


def _given_layout(runtime: RuntimeHarness, layout: str):
    d = runtime.dimension
    if layout == "none":
        return None
    if layout == "empty":
        return {}
    if layout == "first":
        return {0: 0.21}
    if layout == "last":
        return {d - 1: 0.79}
    if layout == "scattered":
        return {d - 1: 0.79, 0: 0.21}
    if layout == "all":
        return {
            index: (index + 1.0) / (d + 1.0)
            for index in range(d)
        }
    raise AssertionError(layout)


@pytest.mark.parametrize("n_draws", [1, 3, 257])
def test_common_shape_dtype_range_fixed_and_non_aliasing(
        runtime, n_draws):
    given = {0: 0.37}
    before = dict(given)
    samples = _draw(runtime, n_draws, given)
    _assert_sample_contract(samples, runtime, n_draws, given)
    assert given == before


@pytest.mark.parametrize(
    "layout", ["none", "empty", "first", "last", "scattered", "all"]
)
def test_given_layout_matrix(runtime, layout):
    given = _given_layout(runtime, layout)
    before = None if given is None else dict(given)
    samples = _draw(runtime, 5, given)
    _assert_sample_contract(samples, runtime, 5, given)
    if given is not None:
        assert given == before


def test_none_and_empty_given_share_unconditional_stream_contract(runtime):
    none = _draw(runtime, 7, None, seed=20260821)
    empty = _draw(runtime, 7, {}, seed=20260821)
    np.testing.assert_array_equal(none, empty)


@pytest.mark.parametrize(
    "value",
    [np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0)],
    ids=["nextafter-zero", "nextafter-one"],
)
def test_open_interval_tail_values_are_preserved_exactly(runtime, value):
    samples = _draw(runtime, 3, {0: value})
    _assert_sample_contract(samples, runtime, 3, {0: value})


def test_new_generators_reproduce_and_reused_generator_advances(runtime):
    given = {0: 0.37}
    first = _draw(runtime, 8, given, seed=20260822)
    repeated = _draw(runtime, 8, given, seed=20260822)
    np.testing.assert_array_equal(first, repeated)

    shared = np.random.default_rng(20260822)
    stream_first = _draw(runtime, 8, given, rng=shared)
    stream_second = _draw(runtime, 8, given, rng=shared)
    free = [i for i in range(runtime.dimension) if i not in given]
    assert free
    assert not np.array_equal(
        stream_first[:, free], stream_second[:, free]
    )


def test_given_insertion_order_does_not_change_result(runtime):
    d = runtime.dimension
    ascending = {0: 0.23, d - 1: 0.77}
    descending = {d - 1: 0.77, 0: 0.23}
    first = _draw(runtime, 6, ascending, seed=20260823)
    second = _draw(runtime, 6, descending, seed=20260823)
    np.testing.assert_array_equal(first, second)


def test_all_given_shortcut_does_not_advance_rng(runtime):
    given = _given_layout(runtime, "all")
    used = np.random.default_rng(20260827)
    samples = _draw(runtime, 4, given, rng=used)
    observed_tail = used.random(5)

    untouched = np.random.default_rng(20260827)
    expected_tail = untouched.random(5)
    _assert_sample_contract(samples, runtime, 4, given)
    np.testing.assert_array_equal(observed_tail, expected_tail)


@pytest.mark.parametrize("n_draws", [0, -1, True, 1.5, "3"])
def test_adapter_rejects_invalid_n_draws(runtime, n_draws):
    expected = TypeError if isinstance(n_draws, (bool, float, str)) else ValueError
    with pytest.raises(expected):
        _draw(runtime, n_draws, {0: 0.4})


@pytest.mark.parametrize("given", [[], [(0, 0.5)], np.array([0.5])])
def test_given_requires_mapping_contract(runtime, given):
    with pytest.raises(TypeError, match="given must be a dict"):
        _draw(runtime, 3, given)


@pytest.mark.parametrize("key", [True, 0.5, "0"])
def test_given_rejects_non_integer_keys(runtime, key):
    with pytest.raises(TypeError, match="keys must be integers"):
        _draw(runtime, 3, {key: 0.4})


@pytest.mark.parametrize("key_kind", ["negative", "past-end"])
def test_given_rejects_out_of_range_keys(runtime, key_kind):
    key = -1 if key_kind == "negative" else runtime.dimension
    with pytest.raises(ValueError, match="given key must be"):
        _draw(runtime, 3, {key: 0.4})


@pytest.mark.parametrize(
    "value", [0.0, 1.0, -0.1, 1.1, np.nan, np.inf, -np.inf]
)
def test_given_rejects_values_outside_open_unit_interval(runtime, value):
    with pytest.raises(ValueError, match="pseudo-observation space"):
        _draw(runtime, 3, {0: value})


@pytest.mark.parametrize(
    "value", [True, "0.5", None, [0.5], np.array([0.5])]
)
def test_given_rejects_non_numeric_scalar_values(runtime, value):
    with pytest.raises(TypeError, match="values must be numeric scalars"):
        _draw(runtime, 3, {0: value})


def test_direct_and_top_level_api_are_seed_identical(runtime):
    given = {0: 0.37}
    direct = _draw(runtime, 9, given, seed=20260824)
    public = api_predict(
        runtime.model,
        runtime.u_train,
        runtime.result,
        9,
        given=given,
        rng=np.random.default_rng(20260824),
    )
    np.testing.assert_array_equal(direct, public)


def test_direct_predict_explicit_and_predict_config_are_seed_identical(runtime):
    given = {0: 0.37}
    explicit = runtime.model.predict(
        9,
        u=runtime.u_train,
        given=given,
        rng=np.random.default_rng(20260902),
    )
    configured = runtime.model.predict(
        9,
        u=runtime.u_train,
        predict_config=PredictConfig(given=given),
        rng=np.random.default_rng(20260902),
    )
    _assert_sample_contract(explicit, runtime, 9, given)
    np.testing.assert_array_equal(configured, explicit)


@pytest.mark.parametrize("model_id", MULTIVARIATE_IDS)
@pytest.mark.parametrize("layout", ["none", "empty", "first", "all"])
def test_multivariate_predict_and_sample_conditional_are_seed_identical(
        model_id, layout):
    runtime = build_runtime(model_id)
    given = _given_layout(runtime, layout)
    conditional = _draw(runtime, 7, given, seed=20260831)
    predicted = runtime.model.predict(
        7,
        u=runtime.u_train,
        given=given,
        rng=np.random.default_rng(20260831),
    )
    _assert_sample_contract(predicted, runtime, 7, given)
    np.testing.assert_array_equal(predicted, conditional)


@pytest.mark.parametrize("model_id", MULTIVARIATE_IDS)
def test_multivariate_predict_does_not_bypass_falsy_invalid_given(model_id):
    runtime = build_runtime(model_id)
    with pytest.raises(TypeError, match="given must be a dict"):
        runtime.model.predict(
            3,
            u=runtime.u_train,
            given=[],
            rng=np.random.default_rng(20260901),
        )


def test_top_level_api_uses_same_strict_given_validation(runtime):
    with pytest.raises(TypeError, match="keys must be integers"):
        api_predict(
            runtime.model,
            runtime.u_train,
            runtime.result,
            3,
            given={True: 0.4},
            rng=np.random.default_rng(20260828),
        )


@pytest.mark.parametrize("layout", ["c", "fortran", "readonly"])
def test_top_level_api_accepts_history_layout_without_aliasing(runtime, layout):
    if layout == "c":
        history = np.array(runtime.u_train, order="C", copy=True)
    elif layout == "fortran":
        history = np.array(runtime.u_train, order="F", copy=True)
    else:
        history = np.array(runtime.u_train, order="C", copy=True)
        history.flags.writeable = False

    before = history.copy()
    samples = api_predict(
        runtime.model,
        history,
        runtime.result,
        4,
        given={0: 0.37},
        rng=np.random.default_rng(20260825),
    )
    _assert_sample_contract(samples, runtime, 4, {0: 0.37})
    np.testing.assert_array_equal(history, before)
    assert not np.shares_memory(samples, history)


def test_generic_vine_diagnostics_remain_visible_through_adapter():
    runtime = build_runtime("vine-generic")
    samples, diagnostics = draw_conditional(
        runtime.model,
        5,
        {2: 0.37},
        u_train=runtime.u_train,
        rng=np.random.default_rng(20260826),
        return_diagnostics=True,
    )
    _assert_sample_contract(samples, runtime, 5, {2: 0.37})
    assert diagnostics["conditional_method"] == "suffix"


@pytest.mark.parametrize("model_id", RUNTIME_IDS)
def test_direct_predict_rejects_unfitted_runtime(model_id):
    model = build_unfitted_runtime(model_id)
    expected = RuntimeError if model_id == "vine-generic" else ValueError
    with pytest.raises(expected, match=r"Fit first|fit\(\.\.\.\)|not fitted"):
        model.predict(
            3,
            u=np.full((5, 2 if model_id.startswith("bivariate") else 3), 0.5),
            given={0: 0.4},
            rng=np.random.default_rng(20260829),
        )


@pytest.mark.parametrize("model_id", RUNTIME_IDS)
def test_conditional_contract_survives_save_load(tmp_path, model_id):
    runtime = build_runtime(model_id)
    expected = _draw(runtime, 5, {0: 0.37}, seed=20260830)

    path = tmp_path / f"{model_id}.json"
    runtime.model.save(path, include_data=True)
    loaded = type(runtime.model).load(path)
    loaded_runtime = RuntimeHarness(
        model_id=model_id,
        model=loaded,
        u_train=loaded._last_u,
        result=loaded.fit_result,
        dimension=runtime.dimension,
        parameter_path=runtime.parameter_path,
    )
    observed = _draw(loaded_runtime, 5, {0: 0.37}, seed=20260830)
    _assert_sample_contract(observed, loaded_runtime, 5, {0: 0.37})
    np.testing.assert_array_equal(observed, expected)


@pytest.mark.parametrize(
    "model_id",
    [
        "bivariate-gaussian",
        "multivariate-gaussian",
        "vine-legacy-cvine",
    ],
)
def test_adapter_rejects_diagnostics_for_unsupported_direct_apis(model_id):
    runtime = build_runtime(model_id)
    with pytest.raises(TypeError, match="does not expose conditional diagnostics"):
        draw_conditional(
            runtime.model,
            3,
            {0: 0.4},
            u_train=runtime.u_train,
            parameter_path=runtime.parameter_path,
            return_diagnostics=True,
        )


def test_adapter_rejects_unknown_runtime():
    with pytest.raises(TypeError, match="unsupported conditional runtime"):
        draw_conditional(object(), 3, {0: 0.4})
