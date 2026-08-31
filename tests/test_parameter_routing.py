"""Regression contracts for distinct fit, objective and post-fit consumers."""
from functools import wraps
from types import SimpleNamespace

import numpy as np
import pytest

from pyscarcopula import GumbelCopula, api
from pyscarcopula._types import (
    LBFGSBConfig, LatentResult, MLEResult, NumericalConfig, ou_params,
)
from pyscarcopula._native import scar_ou as native_ou
from pyscarcopula.strategy import _base, initial_point, scar_tm
from pyscarcopula.strategy.mle import MLEStrategy


@pytest.fixture
def control():
    copula = GumbelCopula(rotate=180)
    u = np.array([[.2, .3], [.4, .6], [.8, .7], [.6, .5]])
    result = LatentResult(
        log_likelihood=0., method='SCAR-TM-OU', copula_name=copula.name,
        success=True, params=ou_params(1.7, .45, .65), K=17,
        grid_range=4.3, grid_method='dense', adaptive=False,
        transition_method='matrix', max_K=53,
    )
    return copula, u, result


@pytest.mark.parametrize('entry', [
    'log_likelihood', 'predictive_mean', 'mixture_h', 'sample', 'predict',
])
@pytest.mark.parametrize('keyword,value', [
    ('definitely_unknown', 17), ('maxiter', 2), ('alpha0', [1., .2, .4]),
    ('gamma0', [.1, .2, .3]),
])
def test_postfit_rejects_unknown_and_wrong_phase_before_compute(
        control, monkeypatch, entry, keyword, value):
    copula, u, result = control
    original = getattr(scar_tm.SCARTMStrategy, entry)

    @wraps(original)
    def forbidden(*args, **kwargs):
        pytest.fail('invalid option reached the computational operation')

    monkeypatch.setattr(scar_tm.SCARTMStrategy, entry, forbidden)
    args = (copula, u, result, 5) if entry in {'sample', 'predict'} else (
        copula, u, result)
    with pytest.raises(TypeError, match=keyword):
        getattr(api, entry)(*args, **{keyword: value})


@pytest.mark.parametrize('entry,extra', [
    ('sample', {}), ('predict', {'given': {0: .2, 1: .3}}),
])
def test_postfit_shortcuts_do_not_hide_unknown_keywords(control, entry, extra):
    copula, u, result = control
    with pytest.raises(TypeError, match='definitely_unknown'):
        getattr(api, entry)(
            copula, u, result, 0, definitely_unknown=True, **extra)


@pytest.mark.parametrize('entry', ['sample', 'predict'])
def test_postfit_routes_constructor_and_operation_options_separately(
        control, monkeypatch, entry):
    copula, u, result = control
    rng = np.random.default_rng(73)
    config = NumericalConfig(default_K=71)
    seen = []
    original = getattr(scar_tm.SCARTMStrategy, entry)

    @wraps(original)
    def consumer(self, copula_arg, u_arg, result_arg, n, rng=None, **kwargs):
        assert self.K == 43 and self.config is config
        assert copula_arg is copula and result_arg is result
        assert 'K' not in kwargs and 'max_K' not in kwargs
        seen.append((rng, kwargs))
        return np.full((n, 2), .37)

    monkeypatch.setattr(scar_tm.SCARTMStrategy, entry, consumer)
    out = getattr(api, entry)(
        copula, u, result, 5, config=config, K=43, rng=rng,
        given={1: .37}, memory_budget_bytes=8192, n_threads=1)
    np.testing.assert_array_equal(out, np.full((5, 2), .37))
    assert seen[0][0] is rng
    assert seen[0][1]['given'] == {1: .37}
    assert seen[0][1]['memory_budget_bytes'] == 8192
    assert seen[0][1]['n_threads'] == 1


def test_postfit_mixture_cache_kept_separate_from_constructor(control, monkeypatch):
    copula, u, result = control
    cache = {}
    original = scar_tm.SCARTMStrategy.mixture_h

    @wraps(original)
    def consumer(self, copula_arg, u_arg, result_arg, **kwargs):
        assert self.K == 43
        assert kwargs['state_cache'] is cache
        assert kwargs['current_cache_key'] == 'current'
        assert kwargs['next_cache_key'] == 'next'
        assert set(kwargs) == {
            'state_cache', 'current_cache_key', 'next_cache_key'}
        return np.full(len(u_arg), .5)

    monkeypatch.setattr(scar_tm.SCARTMStrategy, 'mixture_h', consumer)
    api.mixture_h(copula, u, result, K=43, state_cache=cache,
                  current_cache_key='current', next_cache_key='next')


def test_saved_numerical_settings_and_explicit_none_precedence(control):
    _, _, result = control
    config = NumericalConfig(default_K=71, n_threads=3)
    saved = _base.get_strategy_for_result(result, config=config)
    explicit = _base.get_strategy_for_result(result, config=config, K=43)
    inherited = _base.get_strategy_for_result(result, config=config, K=None)
    assert [saved.K, explicit.K, inherited.K] == [17, 43, 71]
    assert saved.config.n_threads == 3


@pytest.mark.parametrize('key', ['maxiter', 'rng', 'definitely_unknown'])
def test_result_factory_accepts_only_constructor_overrides(control, key):
    with pytest.raises(TypeError, match=key):
        _base.get_strategy_for_result(control[2], **{key: 17})


@pytest.mark.parametrize('key,value', [
    ('alpha0', [1., .2, .3]), ('initial_mle_result', object()),
    ('maxiter', 2), ('verbose', True), ('definitely_unknown', 17),
])
def test_ou_objective_rejects_fit_and_unknown_keywords(control, monkeypatch, key, value):
    copula, u, result = control

    def forbidden(*args, **kwargs):
        pytest.fail('invalid objective option reached native likelihood')

    monkeypatch.setattr(native_ou, 'neg_loglik', forbidden)
    with pytest.raises(TypeError, match=key):
        copula.mlog_likelihood(result.params.values, u, method='SCAR-TM-OU',
                               **{key: value})
    with pytest.raises(TypeError, match=key):
        scar_tm.SCARTMStrategy().objective(
            copula, u, result.params.values, **{key: value})


def test_ou_objective_passes_physical_parameters_and_numerical_overrides(control, monkeypatch):
    copula, u, result = control
    config = NumericalConfig(default_K=71)

    def consumer(kappa, mu, nu, u_arg, copula_arg, cfg):
        assert (kappa, mu, nu) == (1.7, .45, .65)
        assert cfg.K == 43 and cfg.transition_method == 'matrix'
        return 73.125

    monkeypatch.setattr(native_ou, 'neg_loglik', consumer)
    assert copula.mlog_likelihood(
        result.params.values, u, method='SCAR-TM-OU', config=config,
        K=43, transition_method='matrix') == 73.125


@pytest.mark.parametrize('method,alpha', [
    ('MLE', [1.5]), ('GAS', [.1, .2, .3]),
    ('SCAR-TM-JACOBI', [1., .5, .2]),
])
def test_shared_objective_contract_rejects_wrong_phase(control, method, alpha):
    copula, u, _ = control
    with pytest.raises(TypeError, match='maxiter'):
        copula.mlog_likelihood(alpha, u, method=method, maxiter=2)
    with pytest.raises(TypeError, match='definitely_unknown'):
        _base.get_strategy(method).objective(copula, u, np.array(alpha),
                                             definitely_unknown=True)


def test_gas_objective_specific_option_reaches_consumer(control, monkeypatch):
    from pyscarcopula.strategy import gas

    def consumer(omega, gamma, beta, u, copula, scaling, score_eps, *, fail_value):
        assert scaling == 'unit' and score_eps == .017
        assert fail_value == 1e10
        return 73.125

    monkeypatch.setattr(gas, 'gas_negloglik', consumer)
    copula, u, _ = control
    assert copula.mlog_likelihood([.1, .2, .3], u, method='GAS',
        scaling='unit', score_eps=.017) == 73.125


@pytest.mark.parametrize('smart,fallback', [
    (False, None), (True, None), (True, 'heuristic'), (True, 'mle_default'),
])
def test_initialization_preserves_mle_optimizer_config_at_optimizer(
        control, monkeypatch, smart, fallback):
    from pyscarcopula.strategy import mle

    copula, u, _ = control
    config = NumericalConfig(mle_optimizer=LBFGSBConfig(gtol=.019, maxiter=3))
    options_seen = []
    real_minimize = mle.minimize

    def optimizer(fun, x0, **kwargs):
        options_seen.append(kwargs['options'].copy())
        return real_minimize(fun, x0, **kwargs)

    monkeypatch.setattr(mle, 'minimize', optimizer)
    if fallback:
        def fail_primary(*args, **kwargs):
            raise ValueError('force smart heuristic fallback')
        if fallback == 'heuristic':
            monkeypatch.setattr(
                initial_point, '_strength_aware_initial_point', fail_primary)
        else:
            monkeypatch.setattr(scar_tm, 'smart_initial_point', fail_primary)
    alpha, info = scar_tm._resolve_initial_point(
        copula, u, config, smart, False, None)
    assert np.all(np.isfinite(alpha))
    assert len(options_seen) == 1
    assert options_seen[0]['gtol'] == .019
    assert options_seen[0]['maxiter'] == 3
    assert info['mle_source'] == 'strategy_fit'
    if fallback:
        assert info['selected_method'] == fallback


def test_smart_initialization_reuses_supplied_mle_without_refit(control, monkeypatch):
    copula, u, _ = control
    initial = MLEResult(log_likelihood=1.2, method='MLE', copula_name=copula.name,
                        success=True, copula_param=2.1)

    def forbidden(*args, **kwargs):
        pytest.fail('supplied initial MLE result must not be fitted again')

    monkeypatch.setattr(MLEStrategy, 'fit', forbidden)
    _, info = scar_tm._resolve_initial_point(
        copula, u, NumericalConfig(), True, False, None, initial)
    assert info['mle_source'] == 'selection_result'


@pytest.mark.parametrize('error_type', [RuntimeError, ValueError])
def test_smart_mle_failure_uses_constant_fallback(control, monkeypatch, error_type):
    import json

    copula, u, _ = control
    config = NumericalConfig(mle_optimizer=LBFGSBConfig(gtol=.019, maxiter=3))
    calls = []

    def failed_mle(self, copula_arg, data):
        calls.append(self.config)
        raise error_type('static MLE unavailable')

    monkeypatch.setattr(MLEStrategy, 'fit', failed_mle)
    alpha, info = scar_tm._resolve_initial_point(
        copula, u, config, True, False, None)
    assert calls == [config]
    np.testing.assert_array_equal(alpha, native_ou.default_initial_point(0.)[0])
    assert info['selected_method'] == 'constant_default'
    assert info['success'] is True
    assert info['mle_source'] == 'strategy_fit'
    assert info['attempts'][0] == {
        'method': 'static_mle', 'success': False,
        'error_type': error_type.__name__, 'error_message': 'static MLE unavailable',
    }
    assert info['attempts'][-1] == {'method': 'constant_default', 'success': True}
    json.dumps(info)


def test_non_smart_mle_failure_still_raises(control, monkeypatch):
    def failed_mle(*args, **kwargs):
        raise RuntimeError('static MLE unavailable')

    monkeypatch.setattr(MLEStrategy, 'fit', failed_mle)
    with pytest.raises(RuntimeError, match='static MLE unavailable'):
        scar_tm._resolve_initial_point(
            control[0], control[1], NumericalConfig(), False, False, None)


@pytest.mark.parametrize('smart', [False, True])
def test_explicit_initial_point_does_not_attempt_failed_mle(control, monkeypatch, smart):
    def forbidden(*args, **kwargs):
        pytest.fail('explicit alpha0 must bypass static MLE')

    monkeypatch.setattr(MLEStrategy, 'fit', forbidden)
    expected = np.array([1.7, .45, .65])
    alpha, info = scar_tm._resolve_initial_point(
        control[0], control[1], NumericalConfig(), smart, False, expected)
    np.testing.assert_array_equal(alpha, expected)
    assert info['selected_method'] == 'user_provided'


def test_smart_mle_failure_allows_public_ou_fit_to_reach_optimizer(control, monkeypatch):
    class DynamicOptimizerReached(Exception):
        pass

    def failed_mle(*args, **kwargs):
        raise RuntimeError('static MLE unavailable')

    def stop_at_dynamic_optimizer(*args, **kwargs):
        raise DynamicOptimizerReached

    monkeypatch.setattr(MLEStrategy, 'fit', failed_mle)
    monkeypatch.setattr(scar_tm, 'minimize', stop_at_dynamic_optimizer)
    with pytest.raises(DynamicOptimizerReached):
        control[0].fit(control[1], method='SCAR-TM-OU', smart_init=True,
                       K=17, transition_method='matrix', adaptive=False)


def test_custom_variadic_strategy_keeps_its_extension_contract(control, monkeypatch):
    class Custom:
        def __init__(self, config=None, **kwargs):
            self.options = kwargs

        def fit(self, copula, u, **kwargs):
            raise NotImplementedError

        def log_likelihood(self, copula, u, result, **kwargs):
            assert self.options['custom_option'] == 73
            return 73.125

    method = 'TEST-PARAMETER-ROUTING-CUSTOM'
    monkeypatch.setitem(_base._REGISTRY, method, Custom)
    _base._strategy_keyword_contract.cache_clear()
    try:
        copula, u, _ = control
        assert api.log_likelihood(copula, u, SimpleNamespace(method=method),
                                   custom_option=73) == 73.125
    finally:
        _base._strategy_keyword_contract.cache_clear()
