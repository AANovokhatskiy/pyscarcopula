"""Regression contracts for bounded Jacobi optimizer recovery."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from pyscarcopula import ClaytonCopula, GumbelCopula, VineCopula
from pyscarcopula._types import DEFAULT_CONFIG, jacobi_params
from pyscarcopula.api import fit, log_likelihood
from pyscarcopula.strategy import scar_jacobi
from pyscarcopula.vine import _vine_fit


def data():
    return np.column_stack((np.linspace(.05, .95, 30), np.clip(np.linspace(.06, .94, 30) + .06*np.sin(np.arange(30)), .01, .99)))


@pytest.mark.parametrize('budget', ['default', 'explicit', 'config'])
def test_evaluation_limit_retry_preserves_budget_and_total_work(monkeypatch, budget):
    calls = []

    def optimizer(fun, x, **kwargs):
        calls.append(np.array(x).copy())
        point = np.array(x) + (.01 if len(calls) == 1 else 0.)
        return OptimizeResult(x=point, fun=fun(point), nfev=304 if len(calls)==1 else 4,
                              success=len(calls)>1,
                              message=('STOP: TOTAL NO. of f AND g EVALUATIONS EXCEEDS LIMIT'
                                       if len(calls)==1 else 'CONVERGENCE'))

    monkeypatch.setattr(scar_jacobi, 'minimize', optimizer)
    options = {'maxfun': 12} if budget == 'explicit' else {}
    config = (replace(DEFAULT_CONFIG, scar_optimizer=replace(DEFAULT_CONFIG.scar_optimizer,
                                                            maxfun=12))
              if budget == 'config' else DEFAULT_CONFIG)
    r = scar_jacobi.SCARJacobiStrategy(config=config).fit(
        GumbelCopula(), data(), alpha0=np.array([1., .3, .5]), **options)
    assert len(calls) == (2 if budget == 'default' else 1)
    assert r.success == (budget == 'default')
    assert r.nfev == (308 if budget == 'default' else 304)
    assert len(r.diagnostics['optimizer_attempts']) == len(calls)
    if len(calls)==2:
        np.testing.assert_allclose(calls[0] + .01, calls[1])


@pytest.mark.parametrize('message', ['STOP: TOTAL NO. of ITERATIONS REACHED LIMIT', 'ABNORMAL'])
def test_other_optimizer_failures_are_not_retried(monkeypatch, message):
    calls=[]
    def optimizer(fun,x,**kwargs):
        calls.append(1)
        return OptimizeResult(x=x,fun=fun(x),nfev=4,success=False,message=message)
    monkeypatch.setattr(scar_jacobi,'minimize',optimizer)
    r=scar_jacobi.SCARJacobiStrategy().fit(GumbelCopula(),data(),alpha0=np.array([1.,.3,.5]))
    assert len(calls)==1
    assert not r.success


def test_boundary_candidate_keeps_dynamic_model_and_does_not_promote_failure(monkeypatch):
    def optimizer(fun,x,**kwargs):
        return OptimizeResult(x=x,fun=fun(x),nfev=4,success=False,message='ABNORMAL')
    monkeypatch.setattr(scar_jacobi,'minimize',optimizer)
    c=ClaytonCopula(rotate=90)
    r=fit(c,data(),method='scar-tm-jacobi',to_pobs=False,alpha0=[.7,3e-5,.5])
    assert not r.success
    assert r.method=='SCAR-TM-JACOBI'
    assert r.diagnostics['boundary_candidate']['selected']
    assert r.diagnostics['selected_point_source']=='near_independence_boundary'
    assert r.params.m==pytest.approx(r.tau_eps)
    assert r.log_likelihood==pytest.approx(log_likelihood(c,data(),r))
    assert r.nfev==11
    assert r.diagnostics['boundary_candidate']['stationarity_validated']


def test_fixed_vine_retains_discarded_dynamic_point(monkeypatch):
    from types import SimpleNamespace
    attempted = {'optimizer_retry_count': 1, 'optimizer_attempts': [{'nfev': 4}]}
    def failed(*args,**kwargs):
        return SimpleNamespace(method='SCAR-TM-JACOBI',success=False,nfev=8,
                               message='budget',params=jacobi_params(1.,.3,.5),
                               log_likelihood=-12.,diagnostics=attempted)
    monkeypatch.setattr(_vine_fit,'_fit_with_strategy',failed)
    v=VineCopula.cvine(2,order=(0,1))
    v.fit(data(),method='scar-tm-jacobi',to_pobs=False)
    record=v.fit_result.diagnostics['fallback_edges'][0]
    assert record['attempted_log_likelihood']==-12.
    assert record['attempted_params']=={'kappa':1.,'m':.3,'xi':.5}
    assert record['attempted_diagnostics']['optimizer_retry_count']==1
    attempted['optimizer_attempts'][0]['nfev']=999
    assert record['attempted_diagnostics']['optimizer_attempts'][0]['nfev']==4


def test_retry_is_bounded_even_when_both_budgets_fail(monkeypatch):
    calls = []
    def optimizer(fun, x, **kwargs):
        calls.append(1)
        return OptimizeResult(x=x, fun=fun(x), nfev=304, success=False,
                              message='STOP: TOTAL NO. of f AND g EVALUATIONS EXCEEDS LIMIT')
    monkeypatch.setattr(scar_jacobi, 'minimize', optimizer)
    r = scar_jacobi.SCARJacobiStrategy().fit(
        GumbelCopula(), data(), alpha0=np.array([1., .3, .5]))
    assert len(calls) == 2
    assert r.nfev == 608
    assert not r.success
    assert r.diagnostics['optimizer_retry_count'] == 1

@pytest.mark.parametrize('rotation, kappa_bounds, xi_bounds, tolerance, expected', [
    (90, (.001, 100.), (.001, 5.), 1e-3, True),
    (0, (.5, 2.), (.05, .2), 0., False),
])
def test_native_boundary_candidate_checks_two_inward_scales(
        rotation, kappa_bounds, xi_bounds, tolerance, expected):
    from pyscarcopula._native import jacobi
    evaluator = jacobi.PreparedScarJacobiEvaluator(data(), ClaytonCopula(rotate=rotation))
    report = evaluator.near_independence_candidate(
        [.7, 3e-5, .1], kappa_bounds, xi_bounds, 1e-6, 1e6, tolerance)
    assert report['attempted']
    assert report['selected'] == expected
    assert report['stationarity_validated'] == expected
    assert report['nfev'] == 7
    assert report['stationarity_steps'] == [1e-3, 1e-2]
    base = evaluator.neg_loglik(*report['params'])
    for scale, step in enumerate(report['stationarity_steps']):
        for axis, direction in enumerate((1., 1., -1.)):
            probe = np.array(report['raw'])
            probe[axis] += step * direction
            value = evaluator.neg_loglik(*jacobi.raw_to_physical(probe))
            assert report['inward_slopes'][scale][axis] == pytest.approx(
                (value - base) / step, rel=1e-7, abs=1e-12)


def test_native_boundary_candidate_rejects_invalid_tolerance():
    from pyscarcopula._native import jacobi
    evaluator = jacobi.PreparedScarJacobiEvaluator(data(), ClaytonCopula(rotate=90))
    with pytest.raises(ValueError):
        evaluator.near_independence_candidate(
            [.7, 3e-5, .5], (.001, 100.), (.001, 5.), 1e-6, 1., -1.)

def test_native_boundary_uses_distinct_actual_steps_for_narrow_bounds():
    from pyscarcopula._native import jacobi
    evaluator = jacobi.PreparedScarJacobiEvaluator(data(), ClaytonCopula(rotate=90))
    report = evaluator.near_independence_candidate(
        [.0010005, 3e-5, 4.9995], (.001, .001001), (4.999, 5.),
        1e-6, 1e6, 1e-3)
    assert report['selected']
    steps = np.asarray(report['inward_steps'])
    assert np.all(steps[0] > 0.)
    assert np.all(steps[1] > steps[0])
    np.testing.assert_allclose(steps[1] / steps[0], 10., rtol=1e-8)
    assert steps[1, 0] < 1e-3
    assert steps[1, 2] < 1e-3
    base = evaluator.neg_loglik(*report['params'])
    for scale in range(2):
        for axis, direction in enumerate((1., 1., -1.)):
            probe = np.array(report['raw'])
            probe[axis] += steps[scale, axis] * direction
            objective = evaluator.neg_loglik(*jacobi.raw_to_physical(probe))
            assert report['inward_slopes'][scale][axis] == pytest.approx(
                (objective - base) / steps[scale, axis], abs=1e-12)


def test_native_boundary_cannot_validate_unrepresentable_two_scale_probes():
    from pyscarcopula._native import jacobi
    evaluator = jacobi.PreparedScarJacobiEvaluator(data(), ClaytonCopula(rotate=90))
    report = evaluator.near_independence_candidate(
        [.001, 3e-5, .5], (.001, np.nextafter(.001, np.inf)), (.001, 5.),
        1e-6, 1e6, 1e-3)
    assert report['attempted']
    assert not report['stationarity_validated']
    assert not report['selected']
