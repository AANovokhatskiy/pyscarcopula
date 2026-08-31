"""Sampling controls reject unsafe requests before generating latent paths."""

import numpy as np
import pytest

from pyscarcopula import (
    BivariateGaussianCopula, ClaytonCopula, FrankCopula, GumbelCopula,
    JoeCopula, api,
)
from pyscarcopula._types import (
    GASResult, LatentResult, MLEResult, gas_params, jacobi_params, ou_params,
)
from pyscarcopula.strategy._base import get_strategy_for_result
from pyscarcopula.strategy.predict_helpers import sample_predictive


FAMILIES = (ClaytonCopula, FrankCopula, GumbelCopula, JoeCopula,
            BivariateGaussianCopula)
METHODS = ('MLE', 'GAS', 'SCAR-TM-OU', 'SCAR-TM-JACOBI')
U = np.array([[.2, .4], [.7, .6], [.4, .8], [.6, .3]])


def result_for(copula, method):
    common = dict(method=method, copula_name=copula.name,
                  log_likelihood=0., success=True)
    if method == 'MLE':
        return MLEResult(copula_param=copula.transform(0.4).item(), **common)
    if method == 'GAS':
        return GASResult(params=gas_params(.07, .08, .65), **common)
    if method == 'SCAR-TM-OU':
        return LatentResult(params=ou_params(1.2, .3, .5), K=24, max_K=24,
                            adaptive=False, transition_method='matrix', **common)
    return LatentResult(params=jacobi_params(1.2, .35, .5),
                        spectral_basis_order=4, spectral_quad_order=16,
                        transition_method='local_fixed', **common)


@pytest.mark.parametrize('family', FAMILIES)
@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('entry', ['api', 'strategy'])
@pytest.mark.parametrize('given', [None, {0: .4}, {0: .4, 1: .6}])
def test_prediction_budget_precedes_parameter_generation(
        monkeypatch, family, method, entry, given):
    copula = family()
    result = result_for(copula, method)
    strategy = get_strategy_for_result(result)

    def forbidden(*args, **kwargs):
        pytest.fail('A rejected request must not generate predictive parameters')

    monkeypatch.setattr(type(strategy), 'predictive_params', forbidden)
    rng = np.random.default_rng(31)
    before = rng.bit_generator.state
    call = api.predict if entry == 'api' else strategy.predict
    with pytest.raises(MemoryError):
        call(copula, U, result, 20, given=given, rng=rng,
             memory_budget_bytes=20 * 2 * 8 - 1)
    assert rng.bit_generator.state == before


@pytest.mark.parametrize('method', METHODS)
def test_prediction_overflow_is_rejected_before_parameter_generation(monkeypatch, method):
    copula = GumbelCopula()
    result = result_for(copula, method)
    strategy = get_strategy_for_result(result)

    def forbidden(*args, **kwargs):
        pytest.fail('Overflow must be rejected before parameter generation')

    monkeypatch.setattr(type(strategy), 'predictive_params', forbidden)
    with pytest.raises(MemoryError, match='too large'):
        strategy.predict(copula, U, result, int(np.iinfo(np.intp).max))


@pytest.mark.parametrize('family', FAMILIES)
@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('entry', ['api', 'strategy'])
@pytest.mark.parametrize('operation', ['sample', 'predict'])
@pytest.mark.parametrize('threads', [0, -1, True, 1.5, '2'])
def test_invalid_threads_are_rejected_without_consuming_rng(
        family, method, entry, operation, threads):
    copula = family()
    result = result_for(copula, method)
    strategy = get_strategy_for_result(result)
    call = getattr(api if entry == 'api' else strategy, operation)
    rng = np.random.default_rng(32)
    before = rng.bit_generator.state
    with pytest.raises((TypeError, ValueError)):
        call(copula, U, result, 8, n_threads=threads, rng=rng)
    assert rng.bit_generator.state == before


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('given', [None, {0: .4}, {1: .6}, {0: .4, 1: .6}])
def test_sufficient_budget_preserves_seeded_prediction(method, given):
    copula = ClaytonCopula(rotate=270)
    result = result_for(copula, method)
    expected = api.predict(copula, U, result, 8, given=given,
                           rng=np.random.default_rng(33))
    actual = api.predict(copula, U, result, 8, given=given,
                         memory_budget_bytes=10_000_000, n_threads=2,
                         rng=np.random.default_rng(33))
    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == (8, 2)
    for coordinate, value in (given or {}).items():
        np.testing.assert_array_equal(actual[:, coordinate], value)


@pytest.mark.parametrize('given', [None, {0: .3}, {0: .3, 1: .7}])
def test_direct_predictive_sampler_checks_budget_before_rng(given):
    rng = np.random.default_rng(34)
    before = rng.bit_generator.state
    with pytest.raises(MemoryError):
        sample_predictive(ClaytonCopula(), 8, 1.5, rng=rng, given=given,
                          memory_budget_bytes=127)
    assert rng.bit_generator.state == before


@pytest.mark.parametrize('method', ['MLE', 'SCAR-TM-OU', 'SCAR-TM-JACOBI'])
@pytest.mark.parametrize('operation', ['sample', 'predict'])
def test_fully_conditioned_empty_sample_keeps_legacy_contract(method, operation):
    copula = ClaytonCopula()
    result = result_for(copula, method)
    actual = getattr(api, operation)(
        copula, U, result, 0, given={0: .4, 1: .6},
        rng=np.random.default_rng(35))
    assert actual.shape == (0, 2)
