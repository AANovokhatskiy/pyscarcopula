"""Public cache boundary and native posterior invalidation regressions."""

import gc
import weakref

import numpy as np
import pytest

from pyscarcopula import (
    BivariateGaussianCopula,
    EquicorrGaussianCopula,
    StochasticStudentCopula,
    api,
)
from pyscarcopula._native import scar_ou
from pyscarcopula._types import LatentResult, NumericalConfig, ou_params
from pyscarcopula.numerical._scar_ou_config import AutoTMConfig
from pyscarcopula.strategy.scar_tm import (
    SCARTMStrategy,
    _PreparedScarOuFitCache,
    _PreparedScarOuPosteriorCache,
)


VARIANTS = ("equicorr", "fixed", "shrinkage", "cholesky", "factor")
U = np.random.default_rng(20260831).uniform(0.12, 0.88, (19, 4))
LOADINGS = np.array([[0.43], [0.22], [-0.28], [0.35]])
PARAMS = (1.4, 0.13, 0.65)
CONFIG = AutoTMConfig(
    K=32, grid_range=4.0, adaptive=False, max_K=64,
    transition_method="local", gh_order=7, n_threads=1,
)


def _case(variant):
    if variant == "bivariate":
        model = BivariateGaussianCopula()
    elif variant == "equicorr":
        model = EquicorrGaussianCopula(4)
    elif variant == "factor":
        model = StochasticStudentCopula(
            d=4, corr_mode="factor", factor_rank=1,
            factor_loadings=LOADINGS, factor_estimation="two-stage")
    else:
        model = StochasticStudentCopula(
            d=4, corr_mode=variant, R=np.eye(4) * 0.7 + 0.3)
        model._ensure_corr_initialized(U)
    result = LatentResult(
        log_likelihood=0.0, method="SCAR-TM-OU", copula_name=model.name,
        success=True, params=ou_params(*PARAMS), K=32, grid_range=4.0,
        adaptive=False, max_K=64, transition_method="local", gh_order=7,
    )
    return model, result


def _strategy():
    return SCARTMStrategy(
        config=NumericalConfig(n_threads=1), K=32, grid_range=4.0,
        adaptive=False, max_K=64, transition_method="local", gh_order=7,
    )


@pytest.mark.parametrize("variant", (*VARIANTS, "bivariate"))
@pytest.mark.parametrize("cache", [{}, None], ids=["mapping", "none"])
def test_api_predict_rejects_internal_posterior_cache(variant, cache):
    model, result = _case(variant)
    data = U[:, :2] if variant == "bivariate" else U
    with pytest.raises(TypeError, match="posterior_cache is an internal"):
        api.predict(model, data, result, 3, posterior_cache=cache)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("layout", ["contiguous", "strided", "readonly_view"])
def test_internal_cache_reuses_then_invalidates_mutated_history(
        monkeypatch, variant, layout):
    model, result = _case(variant)
    strategy = _strategy()
    backing = np.repeat(U, 2, axis=1) if layout == "strided" else U.copy()
    data = backing[:, ::2] if layout == "strided" else backing
    if layout == "readonly_view":
        data = backing.view()
        data.setflags(write=False)
    cache = {}
    prepared = []
    original = scar_ou.prepare_objective

    def prepare(*args, **kwargs):
        value = original(*args, **kwargs)
        prepared.append(value)
        return value

    monkeypatch.setattr(scar_ou, "prepare_objective", prepare)
    before = strategy.predictive_mean(
        model, data, result, posterior_cache=cache)
    repeated = strategy.predictive_mean(
        model, data, result, posterior_cache=cache)
    np.testing.assert_array_equal(repeated, before)
    assert len(prepared) == 1

    backing[:] = backing ** 2
    actual = strategy.predictive_mean(
        model, data, result, posterior_cache=cache)
    assert len(prepared) == 2
    assert prepared[0] is not prepared[1]
    expected = strategy.predictive_mean(model, data, result)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    assert not np.allclose(actual, before, rtol=0, atol=1e-8)
    workspace = next(iter(cache.values()))
    assert len(workspace.cache) == 1


@pytest.mark.parametrize("variant", ["fixed", "shrinkage", "cholesky", "factor"])
@pytest.mark.parametrize("force", [False, True])
def test_prepared_posterior_updates_correlation_without_stale_values(
        variant, force):
    model, _ = _case(variant)
    prepared = scar_ou.prepare_objective(U, model, CONFIG)
    original_native = prepared._native
    before = prepared.predictive_mean(*PARAMS)
    prepared.update_copula(model)
    assert prepared._native is original_native

    if variant == "factor":
        model._set_factor_loadings(LOADINGS * 0.65)
    else:
        model._set_R(np.eye(4) * 0.92 + 0.08, source="test")
    prepared.update_copula(model, force=force)
    if variant == "factor":
        assert prepared._native is not original_native
        assert model._L_inv is None and model._log_det is None
    else:
        assert prepared._native is original_native

    updated_native = prepared._native
    actual = prepared.predictive_mean(*PARAMS)
    expected = scar_ou.predictive_mean(*PARAMS, U, model, CONFIG)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    assert not np.allclose(actual, before, rtol=0, atol=1e-8)
    value, _ = prepared.neg_loglik_info(*PARAMS)
    expected_value, _ = scar_ou.neg_loglik_info(*PARAMS, U, model, CONFIG)
    np.testing.assert_allclose(value, expected_value, rtol=1e-13, atol=1e-13)
    prepared.update_copula(model)
    assert prepared._native is updated_native


def test_internal_factor_cache_updates_current_correlation():
    model, result = _case("factor")
    strategy = _strategy()
    workspace = _PreparedScarOuPosteriorCache()
    strategy.predictive_mean(model, U, result, posterior_cache=workspace)
    model._set_factor_loadings(LOADINGS * 0.65)
    actual = strategy.predictive_mean(
        model, U, result, posterior_cache=workspace)
    expected = strategy.predictive_mean(model, U, result)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    assert len(workspace.cache) == 1


def test_immutable_prepared_history_reuses_evaluator_without_hash(monkeypatch):
    model, result = _case("equicorr")
    data = model.prepare_sufficient_statistics(U, n_threads=1)
    strategy = _strategy()
    workspace = _PreparedScarOuPosteriorCache()
    expected = strategy.predictive_mean(model, data, result)

    def unexpected_hash(*args, **kwargs):
        pytest.fail("immutable sufficient statistics must not be hashed")

    monkeypatch.setattr("pyscarcopula.strategy.scar_tm.hashlib.blake2b", unexpected_hash)
    for _ in range(2):
        actual = strategy.predictive_mean(
            model, data, result, posterior_cache=workspace)
        np.testing.assert_array_equal(actual, expected)
    assert len(workspace.cache) == 1


def test_posterior_history_hash_is_not_in_optimizer_evaluations(monkeypatch):
    def unexpected_hash(*args, **kwargs):
        pytest.fail("posterior fingerprint entered optimizer hot path")

    monkeypatch.setattr(
        _PreparedScarOuPosteriorCache, "_history_signature", unexpected_hash)
    model, _ = _case("fixed")
    diagnostics = {}
    workspace = _PreparedScarOuFitCache(U, model, diagnostics)
    actual, _ = workspace.neg_loglik_info(*PARAMS, CONFIG)
    repeated, _ = workspace.neg_loglik_info(*PARAMS, CONFIG)
    assert actual == repeated
    assert diagnostics["prepared_native_evaluator_count"] == 1


def test_noncontiguous_history_hash_has_bounded_temporary_chunks(monkeypatch):
    import hashlib

    source = np.arange(300_000, dtype=np.float64).reshape(1000, 300)
    expected = _PreparedScarOuPosteriorCache._history_signature(source)
    noncontiguous = np.asfortranarray(source)
    original = hashlib.blake2b
    sizes = []

    class RecordingHash:
        def __init__(self, *args, **kwargs):
            self.inner = original(*args, **kwargs)

        def update(self, chunk):
            sizes.append(len(chunk))
            self.inner.update(chunk)

        def digest(self):
            return self.inner.digest()

    monkeypatch.setattr("pyscarcopula.strategy.scar_tm.hashlib.blake2b", RecordingHash)
    actual = _PreparedScarOuPosteriorCache._history_signature(noncontiguous)
    assert actual == expected
    assert len(sizes) > 1
    assert max(sizes) <= 8192 * 8
    assert sum(sizes) == source.nbytes


def test_internal_cache_keeps_identity_owners_alive_until_disabled():
    model, result = _case("fixed")
    data = U.copy()
    model_ref, data_ref = weakref.ref(model), weakref.ref(data)
    workspace = _PreparedScarOuPosteriorCache()
    _strategy().predictive_mean(model, data, result, posterior_cache=workspace)
    del model, data
    gc.collect()
    assert model_ref() is not None and data_ref() is not None
    workspace.disable()
    gc.collect()
    assert model_ref() is None and data_ref() is None


def test_history_signature_includes_shape_not_only_observation_bytes():
    data = U.copy()
    before = _PreparedScarOuPosteriorCache._history_signature(data)
    data.shape = (38, 2)
    after = _PreparedScarOuPosteriorCache._history_signature(data)
    assert before != after
    assert before[2] == after[2]


def test_history_signature_includes_dtype_not_only_observation_bytes():
    data = U.copy()
    before = _PreparedScarOuPosteriorCache._history_signature(data)
    data.dtype = np.int64
    after = _PreparedScarOuPosteriorCache._history_signature(data)
    assert before != after
    assert before[0] == after[0]
    assert before[2] == after[2]


@pytest.mark.parametrize("layout", ["contiguous", "strided", "readonly_mmap"])
def test_float32_history_cache_reuse_does_not_copy_to_float64(
        monkeypatch, tmp_path, layout):
    data = U.astype(np.float32)
    if layout == "strided":
        data = np.repeat(data, 2, axis=1)[:, ::2]
    elif layout == "readonly_mmap":
        path = tmp_path / "observations.npy"
        np.save(path, data)
        data = np.load(path, mmap_mode="r")
    model, result = _case("fixed")
    strategy = _strategy()
    workspace = _PreparedScarOuPosteriorCache()
    original = scar_ou.prepare_objective
    prepared = []

    def prepare(*args, **kwargs):
        value = original(*args, **kwargs)
        prepared.append(value)
        return value

    def unexpected_conversion(*args, **kwargs):
        pytest.fail("numeric history fingerprint must not normalize/copy input")

    monkeypatch.setattr(scar_ou, "prepare_objective", prepare)
    monkeypatch.setattr(
        "pyscarcopula.strategy.scar_tm.as_float64_array", unexpected_conversion)
    before = strategy.predictive_mean(
        model, data, result, posterior_cache=workspace)
    repeated = strategy.predictive_mean(
        model, data, result, posterior_cache=workspace)
    assert len(prepared) == 1
    assert data.dtype == np.float32
    np.testing.assert_array_equal(repeated, before)
    expected = strategy.predictive_mean(model, data, result)
    np.testing.assert_allclose(repeated, expected, rtol=1e-13, atol=1e-13)
