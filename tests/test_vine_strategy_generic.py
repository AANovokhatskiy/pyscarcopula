"""Regression tests for strategy-generic vine dispatch."""

from collections import Counter

import numpy as np
import pytest

from pyscarcopula import api
from pyscarcopula._native.errors import NativeUnsupported
from pyscarcopula._types import LatentProcessParams, LatentResult
from pyscarcopula._utils import pobs
from pyscarcopula.copula.elliptical import BivariateGaussianCopula
from pyscarcopula.strategy import _base as strategy_base
from pyscarcopula.vine._edge_adapter import _strategy_kwargs
from pyscarcopula.vine.rvine import RVineCopula


METHOD = 'TEST-GENERIC'


class GenericFakeStrategy:
    """Minimal non-MLE strategy used to assert registry-based dispatch."""

    calls = Counter()
    initial_mle_results = []

    def __init__(self, config=None, **kwargs):
        self.config = config

    def fit(self, copula, u, **kwargs):
        self.calls['fit'] += 1
        if 'initial_mle_result' in kwargs:
            self.initial_mle_results.append(kwargs['initial_mle_result'])
        return LatentResult(
            log_likelihood=12.5,
            method=METHOD,
            copula_name=copula.name,
            success=True,
            params=LatentProcessParams(
                process_type='test',
                names=('level', 'scale'),
                values=np.array([0.2, 0.1], dtype=np.float64),
            ),
        )

    def log_likelihood(self, copula, u, result):
        self.calls['log_likelihood'] += 1
        return 3.0 + 0.01 * len(u)

    def predictive_mean(self, copula, u, result):
        self.calls['predictive_mean'] += 1
        return np.full(len(u), 0.2, dtype=np.float64)

    def mixture_h(self, copula, u, result, **kwargs):
        self.calls['mixture_h'] += 1
        r = np.full(len(u), 0.2, dtype=np.float64)
        return copula.h(u[:, 1], u[:, 0], r)

    def model_sample_params(self, copula, result, n, rng=None, **kwargs):
        self.calls['model_sample_params'] += 1
        return np.full(n, 0.15, dtype=np.float64)

    def predictive_params(self, copula, u, result, n, rng=None, **kwargs):
        self.calls['predictive_params'] += 1
        return np.full(n, 0.25, dtype=np.float64)

    def model_sample_state(self, copula, result, **kwargs):
        self.calls['model_sample_state'] += 1
        return None

    def sample(self, copula, u, result, n, **kwargs):
        self.calls['sample'] += 1
        rng = kwargs.get('rng') or np.random.default_rng(123)
        return rng.uniform(1e-3, 1.0 - 1e-3, size=(n, 2))

    def predict(self, copula, u, result, n, **kwargs):
        self.calls['predict'] += 1
        rng = kwargs.get('rng') or np.random.default_rng(456)
        return rng.uniform(1e-3, 1.0 - 1e-3, size=(n, 2))


@pytest.fixture
def generic_strategy():
    old = strategy_base._REGISTRY.get(METHOD)
    GenericFakeStrategy.calls.clear()
    GenericFakeStrategy.initial_mle_results.clear()
    strategy_base._REGISTRY[METHOD] = GenericFakeStrategy
    try:
        yield GenericFakeStrategy.calls
    finally:
        if old is None:
            strategy_base._REGISTRY.pop(METHOD, None)
        else:
            strategy_base._REGISTRY[METHOD] = old
        GenericFakeStrategy.calls.clear()
        GenericFakeStrategy.initial_mle_results.clear()


def _data(n=120, d=3):
    rng = np.random.default_rng(2025)
    z = rng.standard_normal((n, d))
    z[:, 1] = 0.65 * z[:, 0] + 0.35 * z[:, 1]
    if d > 2:
        z[:, 2] = -0.45 * z[:, 0] + 0.55 * z[:, 2]
    return pobs(z)


def _fixed_gaussian_edges(d):
    return [
        [(BivariateGaussianCopula, 0) for _ in range(d - tree - 1)]
        for tree in range(d - 1)
    ]


def test_api_uses_registered_generic_strategy(generic_strategy):
    u = _data(d=2)
    copula = BivariateGaussianCopula()

    result = api.fit(copula, u, method=METHOD)
    assert result.method == METHOD

    assert api.log_likelihood(copula, u, result) == pytest.approx(4.2)
    assert api.sample(copula, u, result, 5).shape == (5, 2)
    assert api.predict(copula, u, result, 6).shape == (6, 2)

    assert generic_strategy['fit'] == 1
    assert generic_strategy['log_likelihood'] == 1
    assert generic_strategy['sample'] == 1
    assert generic_strategy['predict'] == 1


def test_api_fit_dispatches_to_vine_object_fit():
    u = _data()
    vine = RVineCopula(
        candidates=[BivariateGaussianCopula],
        allow_rotations=False,
    )

    result = api.fit(
        vine,
        u,
        method="mle",
        copulas=_fixed_gaussian_edges(3),
    )

    assert result is vine.fit_result
    assert result.success
    assert np.isfinite(result.log_likelihood)


@pytest.mark.parametrize("method", ["mle", "mLe", "MLE"])
def test_vine_fit_normalizes_method_once(method):
    vine = RVineCopula(
        candidates=[BivariateGaussianCopula],
        allow_rotations=False,
    ).fit(
        _data(),
        method=method,
        copulas=_fixed_gaussian_edges(3),
    )

    assert vine.method == "MLE"
    assert vine.fit_result.method == "MLE"
    edge_results = [
        edge.fit_result for edge in vine.pair_copulas.values()
    ]
    assert all(result.method == "MLE" for result in edge_results)


def test_vine_strategy_kwargs_preserve_explicit_none_override():
    result = LatentResult(
        log_likelihood=0.0,
        method='SCAR-TM-OU',
        copula_name='BivariateGaussian',
        success=True,
    )

    kwargs = _strategy_kwargs(
        result,
        transition_method='auto',
        max_K=None,
        r_gh=2.5,
        unrelated=None,
    )

    assert kwargs == {
        'transition_method': 'auto',
        'max_K': None,
        'r_gh': 2.5,
    }


def test_rvine_uses_registered_generic_strategy_for_edge_runtime(
        generic_strategy):
    u = _data()
    vine = RVineCopula(
        candidates=[BivariateGaussianCopula],
        allow_rotations=False,
    ).fit(u, method=METHOD, copulas=_fixed_gaussian_edges(3))

    assert all(
        edge.fit_result.method == METHOD
        for edge in vine.pair_copulas.values()
    )
    assert all(edge.param is None for edge in vine.pair_copulas.values())

    fit_mixture_h_calls = generic_strategy['mixture_h']
    samples = vine.sample(7, rng=np.random.default_rng(1))
    predicted = vine.predict(8, u=u, rng=np.random.default_rng(2))
    with pytest.raises(
            NativeUnsupported,
            match="does not support edge .*LatentResult"):
        vine.log_likelihood(u)

    assert samples.shape == (7, 3)
    assert predicted.shape == (8, 3)
    assert generic_strategy['fit'] == 3
    assert generic_strategy['mixture_h'] == fit_mixture_h_calls
    assert len(GenericFakeStrategy.initial_mle_results) == 3
    assert all(
        result.method == 'MLE'
        for result in GenericFakeStrategy.initial_mle_results
    )
    assert generic_strategy['mixture_h'] > 0
    assert generic_strategy['model_sample_params'] == 3
    assert generic_strategy['predictive_params'] == 3
    assert generic_strategy['log_likelihood'] == 0


def test_rvine_prefers_strategy_mixture_h_pair(
        generic_strategy, monkeypatch):
    def mixture_h_pair(self, copula, u, result, **kwargs):
        self.calls['mixture_h_pair'] += 1
        r = np.full(len(u), 0.2, dtype=np.float64)
        return copula.h_pair(u[:, 1], u[:, 0], r)

    monkeypatch.setattr(
        GenericFakeStrategy, 'mixture_h_pair', mixture_h_pair, raising=False)
    u = _data()
    vine = RVineCopula(
        candidates=[BivariateGaussianCopula],
        allow_rotations=False,
    ).fit(u, method=METHOD, copulas=_fixed_gaussian_edges(3))

    # Two first-tree edges propagate to the second tree during fit.
    assert generic_strategy['mixture_h_pair'] == 2
    assert generic_strategy['mixture_h'] == 0

    generic_strategy.clear()
    with pytest.raises(
            NativeUnsupported,
            match="does not support edge .*LatentResult"):
        vine.log_likelihood(u)
    assert generic_strategy['mixture_h_pair'] == 0
    assert generic_strategy['mixture_h'] == 0


def test_dynamic_vine_runs_one_mle_per_edge(
        generic_strategy, monkeypatch):
    from pyscarcopula.strategy.mle import MLEStrategy

    original_fit = MLEStrategy.fit
    calls = 0

    def counted_fit(self, copula, u, **kwargs):
        nonlocal calls
        calls += 1
        return original_fit(self, copula, u, **kwargs)

    monkeypatch.setattr(MLEStrategy, 'fit', counted_fit)
    RVineCopula(
        candidates=[BivariateGaussianCopula],
        allow_rotations=False,
    ).fit(_data(), method=METHOD, copulas=_fixed_gaussian_edges(3))

    assert calls == 3


def test_rvine_caches_suffix_state_by_given_variable_set(monkeypatch):
    from pyscarcopula.vine import rvine as rvine_module

    calls = []

    def fake_suffix_state(*args):
        given = args[-1]
        calls.append(frozenset(given))
        return ('state', frozenset(given))

    monkeypatch.setattr(
        rvine_module, 'suffix_sampling_state', fake_suffix_state)
    vine = RVineCopula()

    first = vine._suffix_sampling_state({2: 0.2, 0: 0.4})
    second = vine._suffix_sampling_state({0: 0.8, 2: 0.6})
    third = vine._suffix_sampling_state({1: 0.5})

    assert first is second
    assert third != first
    assert calls == [frozenset({0, 2}), frozenset({1})]
