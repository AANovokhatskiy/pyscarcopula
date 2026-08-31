"""Strategy keyword routing, scaling overrides and numerical controls."""

from dataclasses import replace

import numpy as np
import pytest

from pyscarcopula import (
    BivariateGaussianCopula, ClaytonCopula, FrankCopula, GaussianCopula,
    GumbelCopula, IndependentCopula, JoeCopula, StochasticStudentCopula,
    StudentCopula, VineCopula, api,
)
from pyscarcopula._native import gas as native_gas, pair as native_pair
from pyscarcopula._types import (
    GASResult, IndependentResult, LatentResult, MLEResult, NumericalConfig,
    PredictConfig, gas_params, jacobi_params, ou_params,
)
from pyscarcopula.strategy import gas
from pyscarcopula.strategy._base import get_strategy_for_result
from pyscarcopula.strategy.gas import GASStrategy
from pyscarcopula.strategy.mle import MLEStrategy
from pyscarcopula.strategy.predict_helpers import strategy_predict


PAIR_FAMILIES = (
    GumbelCopula, ClaytonCopula, FrankCopula, JoeCopula,
    BivariateGaussianCopula,
)
MODES = ('MLE', 'GAS', 'SCAR-TM-OU', 'SCAR-TM-JACOBI')
CASES = [(family, mode) for family in PAIR_FAMILIES for mode in MODES]
CASES.append((IndependentCopula, 'MLE'))
U = np.random.default_rng(314).uniform(.12, .88, (16, 2))
GAS_PARAMETERS = gas_params(.13, .055, .61)
VINE_OPTIONS = (
    ('dynamic_conditioning', 'given_only'), ('return_diagnostics', True),
    ('mcmc_steps', 7), ('mcmc_burnin', 2),
)


def fitted_result(copula, method):
    common = dict(log_likelihood=0., copula_name=copula.name,
                  method=method, success=True)
    if method == 'MLE':
        if isinstance(copula, IndependentCopula):
            return IndependentResult(**common)
        parameter = .47 if isinstance(copula, BivariateGaussianCopula) else 2.1
        return MLEResult(copula_param=parameter, **common)
    if method == 'GAS':
        return GASResult(params=GAS_PARAMETERS, score_eps=.00037,
                         r_last=2.1, **common)
    if method == 'SCAR-TM-OU':
        return LatentResult(params=ou_params(1.7, .45, .65), K=37,
                            adaptive=False, transition_method='matrix', **common)
    return LatentResult(
        params=jacobi_params(1.8, .43, .8), spectral_basis_order=12,
        spectral_quad_order=32, transition_method='local_fixed', **common)


@pytest.fixture(params=CASES, ids=lambda case: f'{case[0].__name__}-{case[1]}')
def case(request):
    factory, method = request.param
    copula = factory()
    return copula, fitted_result(copula, method)


@pytest.mark.parametrize('key,value', VINE_OPTIONS)
@pytest.mark.parametrize('channel', ['kwargs', 'predict_config'])
def test_nonvine_rejects_vine_prediction_options(case, key, value, channel):
    copula, result = case
    kwargs = {key: value} if channel == 'kwargs' else {
        'predict_config': PredictConfig(**{key: value})}
    with pytest.raises(TypeError, match=key):
        api.predict(copula, U, result, 0, given={0: .2, 1: .3}, **kwargs)


def test_default_predict_config_and_strategy_aliases_remain_usable(case):
    copula, result = case
    strategy = get_strategy_for_result(result)
    expected = strategy.predict(copula, U, result, 7,
        rng=np.random.default_rng(617), horizon='current', given={0: .43},
        predictive_r_mode='grid')
    actual = api.predict(copula, U, result, 7,
        rng=np.random.default_rng(617), given={0: .43},
        predict_config=PredictConfig(horizon='current', predictive_r_mode='grid'))
    np.testing.assert_array_equal(actual, expected)


def test_vine_prediction_options_reach_vine(monkeypatch):
    seen = []
    sentinel = object()

    def predict(self, n, *, u, predict_config, **kwargs):
        assert n == 7 and u is U
        seen.append(predict_config)
        return sentinel

    monkeypatch.setattr(VineCopula, 'predict', predict)
    config = PredictConfig(**dict(VINE_OPTIONS))
    assert api.predict(VineCopula(), U, None, 7,
                       predict_config=config, mcmc_steps=3) is sentinel
    assert seen == [replace(config, mcmc_steps=3)]


@pytest.mark.parametrize('entry', ['sample', 'predict'])
@pytest.mark.parametrize('n,given', [(0, None), (3, None), (3, {0: .2, 1: .3})])
@pytest.mark.parametrize('keyword', ['routing_typo', 'maxiter', 'optimizer_options'])
def test_direct_strategy_rejects_unknown_before_shortcuts(case, entry, n, given, keyword):
    copula, result = case
    strategy = get_strategy_for_result(result)
    with pytest.raises(TypeError, match=keyword):
        getattr(strategy, entry)(copula, U, result, n,
                                 given=given, **{keyword: 17})


@pytest.mark.parametrize('factory', PAIR_FAMILIES)
@pytest.mark.parametrize('entry', [
    'log_likelihood', 'predictive_mean', 'mixture_h', 'sample', 'predict',
])
@pytest.mark.parametrize('saved_scaling,override', [('unit', 'fisher'), ('fisher', 'unit')])
def test_gas_explicit_scaling_overrides_result_without_mutation(
        factory, entry, saved_scaling, override):
    copula = factory()
    result = replace(fitted_result(copula, 'GAS'), scaling=saved_scaling)
    changed = replace(result, scaling=override)

    def call(strategy, fitted):
        args = (copula, U, fitted)
        kwargs = {}
        if entry in {'sample', 'predict'}:
            args += (7,)
            kwargs['rng'] = np.random.default_rng(617)
        return getattr(strategy, entry)(*args, **kwargs)

    expected = call(GASStrategy(), changed)
    baseline = call(GASStrategy(), result)
    assert not np.allclose(expected, baseline)
    np.testing.assert_allclose(call(GASStrategy(scaling=override), result), expected)
    args = (copula, U, result)
    kwargs = {'scaling': override}
    if entry in {'sample', 'predict'}:
        args += (7,)
        kwargs['rng'] = np.random.default_rng(617)
    np.testing.assert_allclose(getattr(api, entry)(*args, **kwargs), expected)
    assert result.scaling == saved_scaling
    np.testing.assert_array_equal(result.params.values, GAS_PARAMETERS.values)


def test_gas_scaling_override_requires_history_for_cached_prediction():
    copula = GumbelCopula()
    result = fitted_result(copula, 'GAS')
    assert GASStrategy().predictive_state(copula, None, result).r[0] == result.r_last
    with pytest.raises(ValueError, match='history'):
        GASStrategy(scaling='fisher').predict(copula, None, result, 3)


@pytest.mark.parametrize('factory', [GaussianCopula, StudentCopula])
def test_multivariate_mle_predict_keeps_prediction_keys_out_of_sample(factory):
    copula = factory()
    u = np.random.default_rng(628).uniform(.12, .88, (32, 3))
    result = copula.fit(u, method='MLE')
    assert result.success
    expected = MLEStrategy().sample(copula, u, result, 7,
                                    given={1: .4}, rng=np.random.default_rng(617))
    actual = api.predict(copula, u, result, 7, given={1: .4},
        horizon='current', predictive_r_mode='grid', rng=np.random.default_rng(617))
    np.testing.assert_array_equal(actual, expected)
    with pytest.raises(TypeError, match='mcmc_steps'):
        api.predict(copula, u, result, 7, mcmc_steps=3)


def test_custom_variadic_strategy_remains_extensible():
    class Custom:
        predict = strategy_predict

        def predictive_params(self, copula, u, result, n, rng=None, **kwargs):
            assert kwargs['custom_option'] == 73
            return np.full(n, 2.1)

    output = Custom().predict(GumbelCopula(), U, None, 3,
                              custom_option=73, rng=np.random.default_rng(617))
    assert output.shape == (3, 2)


@pytest.mark.parametrize('factory', [GumbelCopula, JoeCopula])
@pytest.mark.parametrize('rotation', [0, 90, 180, 270])
@pytest.mark.parametrize('coordinate', [0, 1])
@pytest.mark.parametrize('control', ['bisection_tol', 'bisection_maxiter'])
def test_native_inverse_controls_change_accuracy(factory, rotation, coordinate, control):
    copula = factory(rotate=rotation)
    quantiles = np.array([.031, .173, .421, .793, .941])
    given = .43
    tight = dict(bisection_tol=1e-13, bisection_maxiter=100)
    coarse = {**tight, control: .1 if control == 'bisection_tol' else 1}

    def sample(options):
        return native_pair.conditional_sample_from_uniforms(
            copula, quantiles, 2.3, given_coordinate=coordinate,
            given_value=given, **options)

    accurate = sample(tight)
    limited = sample(coarse)
    conditional = copula.h_pair(accurate[:, 0], accurate[:, 1], 2.3)[1 - coordinate]
    np.testing.assert_allclose(conditional, quantiles, atol=2e-12, rtol=0)
    assert np.max(np.abs(limited - accurate)) > 1e-5
    np.testing.assert_array_equal(accurate[:, coordinate], np.full(len(quantiles), given))
    # No-config low-level callers keep the established family defaults.
    legacy = dict(bisection_tol=1e-12, bisection_maxiter=32) if factory is GumbelCopula else {
        'bisection_tol': 1e-10, 'bisection_maxiter': 50}
    np.testing.assert_array_equal(sample({}), sample(legacy))


@pytest.mark.parametrize('factory', [GumbelCopula, JoeCopula])
@pytest.mark.parametrize('method', MODES)
@pytest.mark.parametrize('entry', ['sample', 'predict'])
def test_conditional_operations_use_numerical_inverse_config(factory, method, entry):
    copula = factory(rotate=180)
    result = fitted_result(copula, method)

    def sample(config):
        return getattr(api, entry)(copula, U, result, 7,
            config=config, given={0: .43}, rng=np.random.default_rng(617))

    accurate = sample(NumericalConfig(bisection_tol=1e-13, bisection_maxiter=100))
    limited = sample(NumericalConfig(bisection_tol=1e-13, bisection_maxiter=1))
    assert np.max(np.abs(limited - accurate)) > 1e-5
    np.testing.assert_array_equal(limited[:, 0], np.full(7, .43))


@pytest.mark.parametrize('key,value,error', [
    ('bisection_tol', 0., ValueError), ('bisection_tol', -1., ValueError),
    ('bisection_tol', np.nan, ValueError), ('bisection_tol', np.inf, ValueError),
    ('bisection_maxiter', 0, ValueError), ('bisection_maxiter', -1, ValueError),
    ('bisection_maxiter', 1.5, TypeError), ('bisection_maxiter', True, TypeError),
])
def test_invalid_inverse_controls_rejected_even_for_empty_draws(key, value, error):
    with pytest.raises(error, match=key):
        NumericalConfig(**{key: value})
    with pytest.raises(error, match=key):
        native_pair.conditional_sample_from_uniforms(GumbelCopula(), [], 2.1,
            given_coordinate=0, given_value=.43, **{key: value})


@pytest.mark.parametrize('joint', [False, True])
@pytest.mark.parametrize('error', [FloatingPointError, TypeError, ValueError])
def test_gas_failure_penalty_uses_config_only_for_numerical_errors(monkeypatch, joint, error):
    copula = StochasticStudentCopula(d=3, corr_mode='shrinkage') if joint else GumbelCopula()
    u = np.random.default_rng(417).uniform(.12, .88, (16, 3 if joint else 2))
    name = 'negative_log_likelihood_and_gradient' + ('_shrinkage' if joint else '')
    penalty = 7654321.

    class OptimizerStopped(Exception):
        pass

    def fail(*args, **kwargs):
        raise error('consumer failure')

    def minimize(fun, x0, **kwargs):
        if error is FloatingPointError:
            value, gradient = fun(np.asarray(x0))
            assert value == penalty and np.isfinite(gradient).all()
        else:
            with pytest.raises(error, match='consumer failure'):
                fun(np.asarray(x0))
        raise OptimizerStopped

    monkeypatch.setattr(native_gas, name, fail)
    monkeypatch.setattr(gas, 'minimize', minimize)
    with pytest.raises(OptimizerStopped):
        GASStrategy(config=NumericalConfig(fail_value=penalty)).fit(
            copula, u, gamma0=GAS_PARAMETERS.values)
