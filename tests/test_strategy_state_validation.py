"""Validation of direct strategy state operations and native parameter inputs."""

import numpy as np
import pytest

from pyscarcopula import ClaytonCopula, NumericalConfig, PredictConfig
from pyscarcopula._native import gas, jacobi, scar_ou
from pyscarcopula._types import (
    GASResult, LatentResult, MLEResult, gas_params, jacobi_params, ou_params,
)
from pyscarcopula.strategy._base import get_strategy_for_result
from pyscarcopula.strategy.initial_point import resolve_ou_initial_point
from pyscarcopula.strategy.mle import MLEStrategy
from pyscarcopula.strategy.predict_helpers import (
    conditional_sample_bivariate, predictive_params_from_state,
)


METHODS = ('MLE', 'GAS', 'SCAR-TM-OU', 'SCAR-TM-JACOBI')
U = np.random.default_rng(731).uniform(.12, .88, (16, 2))


@pytest.mark.parametrize('keyword', ['state_typo', 'maxiter', 'predict_config'])
def test_mle_mixture_rejects_unknown_before_result_access(keyword):
    with pytest.raises(TypeError, match=keyword):
        MLEStrategy().mixture_h_pair(ClaytonCopula(), U, None, **{keyword: 13})


def test_mle_mixture_accepts_shared_vine_cache_context():
    strategy = MLEStrategy()
    model = ClaytonCopula()
    result = MLEResult(method='MLE', log_likelihood=0., copula_name='Clayton',
                       success=True, copula_param=1.1)
    expected = strategy.mixture_h_pair(model, U, result)
    actual = strategy.mixture_h_pair(
        model, U, result, state_cache={}, current_cache_key='current',
        next_cache_key='next', posterior_cache={})
    np.testing.assert_array_equal(actual, expected)


@pytest.fixture(params=METHODS)
def state_case(request):
    method = request.param
    common = dict(method=method, log_likelihood=0., copula_name='Clayton',
                  success=True)
    if method == 'MLE':
        result = MLEResult(copula_param=1.1, **common)
    elif method == 'GAS':
        result = GASResult(params=gas_params(.2, .035, .6), r_last=1.2, **common)
    elif method == 'SCAR-TM-OU':
        result = LatentResult(params=ou_params(2., .3, .35), K=17,
                             adaptive=False, transition_method='matrix', **common)
    else:
        result = LatentResult(params=jacobi_params(2., .4, .3),
                             spectral_basis_order=8, spectral_quad_order=24,
                             transition_method='local', **common)
    return get_strategy_for_result(result), ClaytonCopula(), result


@pytest.mark.parametrize('entry', [
    'predictive_params', 'predictive_state', 'condition_state', 'sample_params',
    'model_sample_params', 'model_sample_state',
])
@pytest.mark.parametrize('key,value', [
    ('state_typo', 13), ('maxiter', 13), ('predict_config', PredictConfig()),
])
def test_direct_state_keywords_rejected_before_state_access(
        state_case, entry, key, value):
    strategy, model, _ = state_case
    # Invalid state/result owners demonstrate that keyword validation runs first.
    args = {
        'predictive_params': (model, None, None, 0),
        'predictive_state': (model, None, None),
        'condition_state': (model, None, None, None),
        'sample_params': (model, None, 0),
        'model_sample_params': (model, None, 0),
        'model_sample_state': (model, None),
    }
    with pytest.raises(TypeError, match=key):
        getattr(strategy, entry)(*args[entry], **{key: value})


def test_prediction_context_remains_accepted(state_case):
    strategy, model, result = state_case
    options = dict(horizon='current', predictive_r_mode='grid', given={0: .4},
                   n_threads=1, memory_budget_bytes=4096, state_cache={},
                   cache_key='edge', posterior_cache={})
    actual = strategy.predictive_params(
        model, U, result, 5, rng=np.random.default_rng(19), **options)
    expected = strategy.predictive_params(
        model, U, result, 5, rng=np.random.default_rng(19),
        horizon='current', predictive_r_mode='grid')
    np.testing.assert_array_equal(actual, expected)


def test_custom_variadic_state_hooks_keep_extension_keywords():
    class CustomStrategy:
        def predictive_state(self, copula, u, result, **kwargs):
            assert kwargs == {'custom_option': 17}
            return 'custom-state'

        def sample_params(self, copula, state, n, rng=None, **kwargs):
            assert state == 'custom-state'
            return np.full(n, kwargs['custom_option'])

    actual = predictive_params_from_state(
        CustomStrategy(), None, None, None, 3, custom_option=17)
    np.testing.assert_array_equal(actual, [17, 17, 17])


@pytest.mark.parametrize('dtype', [np.complex128, object])
def test_objective_rejects_complex_before_evaluation(state_case, dtype):
    strategy, model, result = state_case
    parameters = ([result.copula_param] if result.method == 'MLE'
                  else result.params.values)
    alpha = np.asarray([complex(value, .2) for value in parameters], dtype=dtype)
    with pytest.raises((TypeError, ValueError), match='real|complex'):
        strategy.objective(model, U, alpha)


@pytest.mark.parametrize('history', [None, U])
def test_invalid_horizon_is_rejected_for_every_history(state_case, history):
    strategy, model, result = state_case
    with pytest.raises(ValueError, match='horizon'):
        strategy.predictive_state(model, history, result, horizon='invalid')


def test_stationary_mode_rejection_does_not_advance_rng(state_case):
    strategy, model, result = state_case
    if result.method not in {'SCAR-TM-OU', 'SCAR-TM-JACOBI'}:
        return
    state = strategy.predictive_state(model, None, result)
    rng = np.random.default_rng(29)
    initial = rng.bit_generator.state
    with pytest.raises(ValueError, match='predictive_r_mode'):
        strategy.sample_params(model, state, 5, rng=rng,
                               predictive_r_mode='invalid')
    assert rng.bit_generator.state == initial


def test_conditioning_rejects_complex_observations(state_case):
    strategy, model, result = state_case
    if result.method == 'MLE':
        return  # A static point state has no observation update.
    state = strategy.predictive_state(model, U, result)
    with pytest.raises((TypeError, ValueError), match='real|complex'):
        strategy.condition_state(model, state, U[:1].astype(complex) + .2j, result)


def test_explicit_initial_point_and_conditional_parameter_reject_complex():
    model = ClaytonCopula()
    with pytest.raises(TypeError, match='alpha0'):
        resolve_ou_initial_point(model, U, NumericalConfig(), True, False,
                                 np.array([2., .3, .4]) + .2j)
    rng = np.random.default_rng(37)
    initial = rng.bit_generator.state
    with pytest.raises(TypeError, match='r must'):
        conditional_sample_bivariate(model, 4, np.array([1.2 + .2j]),
                                     given={0: .4}, rng=rng)
    assert rng.bit_generator.state == initial


@pytest.mark.parametrize('method', ['GAS', 'SCAR-TM-OU', 'SCAR-TM-JACOBI'])
@pytest.mark.parametrize('index', range(3))
def test_native_scalar_parameter_facades_reject_complex(method, index):
    parameters = [.2, .035, .6] if method == 'GAS' else [2., .4, .3]
    parameters[index] = np.complex128(parameters[index] + .2j)
    with pytest.raises((TypeError, ValueError)):
        if method == 'GAS':
            gas.initial_state(*parameters, ClaytonCopula())
        elif method == 'SCAR-TM-OU':
            scar_ou.sample_stationary_fixed_draws(*parameters, np.zeros(2))
        else:
            jacobi.sample_stationary_fixed_draws(*parameters, np.full(2, .5))
