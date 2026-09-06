"""Rolling risk keeps validation and numerical settings with their owners."""
from functools import wraps

import numpy as np
import pytest

from pyscarcopula import (
    BivariateGaussianCopula, EquicorrGaussianCopula, GaussianCopula,
    IndependentCopula, NumericalConfig, StudentCopula, VineCopula,
)
from pyscarcopula.contrib import risk_metrics as risk
from pyscarcopula.contrib.marginal import MarginalModel


def _returns():
    return np.random.default_rng(704).normal(0, .01, (18, 2))


def _run(model, **kwargs):
    return risk.risk_metrics(model, _returns(), window_len=17,
                             marginals_method='normal', gamma=.9, N_mc=17,
                             rng=1, optimize_portfolio=False, **kwargs)


@pytest.mark.parametrize('policy', [None, False, '', 'ignore', 'RAISE', [],
                                    np.array(['raise', 'continue'])])
def test_invalid_failure_policy_rejected_before_any_work(monkeypatch, policy):
    def forbidden(*args, **kwargs):
        pytest.fail('marginal fitting reached for invalid failure policy')
    monkeypatch.setattr(MarginalModel, 'create', forbidden)
    with pytest.raises(ValueError, match='failure_policy'):
        _run(IndependentCopula(), failure_policy=policy)


@pytest.mark.parametrize('jobs', [1, 2])
@pytest.mark.parametrize('policy', ['raise', 'continue'])
@pytest.mark.parametrize('value', [np.full((18, 2), .1j), [[.01, .2j]] * 18,
                                   np.full((18, 2), np.array(.1j), dtype=object)])
def test_complex_returns_rejected_before_marginals(monkeypatch, jobs, policy, value):
    def forbidden(*args, **kwargs):
        pytest.fail('marginal fitting reached for complex returns')
    monkeypatch.setattr(MarginalModel, 'create', forbidden)
    with pytest.raises(TypeError, match='real'):
        risk.risk_metrics(GaussianCopula(), value, window_len=17,
                          n_jobs=jobs, failure_policy=policy)


@pytest.mark.parametrize('factory', [IndependentCopula, BivariateGaussianCopula,
                                    EquicorrGaussianCopula, GaussianCopula,
                                    StudentCopula, VineCopula])
@pytest.mark.parametrize('kwargs', [{'routing_typo': 7}, {'horizon': 'next'},
                                   {'config': {'n_threads': 2}}])
@pytest.mark.parametrize('jobs', [1, 2])
def test_fit_keywords_rejected_before_marginals_or_workers(monkeypatch, factory, kwargs, jobs):
    def forbidden(*args, **options):
        pytest.fail('marginal fitting or worker startup reached before preflight')
    monkeypatch.setattr(MarginalModel, 'create', forbidden)
    with pytest.raises((TypeError, ValueError)):
        _run(factory(), n_jobs=jobs, **kwargs)


@pytest.mark.parametrize('factory', [GaussianCopula, StudentCopula])
@pytest.mark.parametrize('policy', ['raise', 'continue'])
def test_static_fit_receives_options_and_thread_override(monkeypatch, factory, policy):
    calls = []
    original = factory.fit
    @wraps(original)
    def recording(self, data, *args, **kwargs):
        calls.append(dict(kwargs))
        return original(self, data, *args, **kwargs)
    monkeypatch.setattr(factory, 'fit', recording)
    config = NumericalConfig(n_threads=2, fail_value=7654321.)
    result = _run(factory(corr_mode='shrinkage'), n_threads=3,
                  config=config, maxiter=100, failure_policy=policy)
    assert len(calls) == 2
    for kwargs in calls:
        assert kwargs['maxiter'] == 100
        assert kwargs['config'].n_threads == 3
        assert kwargs['config'].fail_value == config.fail_value
        assert 'failure_policy' not in kwargs
    assert config.n_threads == 2
    assert result[.9][17]['diagnostics']['failure_policy'] == policy


@pytest.mark.parametrize('factory', [GaussianCopula, StudentCopula])
def test_static_model_keeps_documented_mle_method_override(factory):
    expected = _run(factory(), method='mle')
    actual = _run(factory(), method='GAS')
    for key in ('var', 'cvar'):
        np.testing.assert_array_equal(actual[.9][17][key], expected[.9][17][key])


@pytest.mark.parametrize('policy', ['raise', 'continue'])
def test_preflight_rejects_optimizer_option_in_nonoptimizing_gaussian_mode(monkeypatch, policy):
    def forbidden(*args, **kwargs):
        pytest.fail('marginal fitting reached for wrong-owner maxiter')
    monkeypatch.setattr(MarginalModel, 'create', forbidden)
    with pytest.raises(TypeError, match='optimizer options'):
        _run(GaussianCopula(), maxiter=13, failure_policy=policy)
