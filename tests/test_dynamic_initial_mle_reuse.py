import numpy as np

from pyscarcopula import GumbelCopula
from pyscarcopula._types import DEFAULT_CONFIG, MLEResult
from pyscarcopula.strategy.gas import _automatic_gas_start
from pyscarcopula.strategy.initial_point import smart_initial_point
from pyscarcopula.strategy.scar_jacobi import SCARJacobiStrategy


def _mle_result(theta=2.0, log_likelihood=7.5):
    return MLEResult(
        log_likelihood=log_likelihood,
        method='MLE',
        copula_name='Gumbel',
        success=True,
        copula_param=theta,
    )


def _fail_if_mle_runs(monkeypatch):
    from pyscarcopula.strategy.mle import MLEStrategy

    def fail(*args, **kwargs):
        raise AssertionError('unexpected duplicate MLE fit')

    monkeypatch.setattr(MLEStrategy, 'fit', fail)


def test_scar_ou_smart_start_reuses_initial_mle_result(monkeypatch):
    _fail_if_mle_runs(monkeypatch)
    copula = GumbelCopula()
    u = np.array([[0.2, 0.3], [0.4, 0.6], [0.8, 0.7]])
    mle_result = _mle_result()

    alpha0, info = smart_initial_point(
        u, copula, initial_mle_result=mle_result)

    expected_mu = float(copula.inv_transform([2.0])[0])
    assert alpha0[1] == expected_mu
    assert info['static_loglik'] == mle_result.log_likelihood


def test_gas_start_reuses_initial_mle_result(monkeypatch):
    _fail_if_mle_runs(monkeypatch)
    copula = GumbelCopula()
    mle_result = _mle_result()

    gamma0 = _automatic_gas_start(
        copula, np.empty((0, 2)), DEFAULT_CONFIG, mle_result)

    expected_mu = float(copula.inv_transform([2.0])[0])
    np.testing.assert_allclose(gamma0, [0.05 * expected_mu, 0.05, 0.95])


def test_scar_jacobi_start_reuses_initial_mle_result(monkeypatch):
    _fail_if_mle_runs(monkeypatch)
    copula = GumbelCopula()
    strategy = SCARJacobiStrategy()
    mle_result = _mle_result()

    alpha0, diagnostics = strategy._initial_point(
        copula, np.empty((0, 2)), mle_result)

    expected_tau = float(copula.param_to_tau(np.array([2.0]))[0])
    assert alpha0[1] == expected_tau
    assert diagnostics['selected_method'] == 'static_mle_tau'
    assert diagnostics['mle_source'] == 'selection_result'
