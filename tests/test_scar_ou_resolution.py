from dataclasses import replace

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from pyscarcopula import BivariateGaussianCopula, StochasticStudentCopula
from pyscarcopula.api import log_likelihood
from pyscarcopula._native.errors import NativeError
from pyscarcopula.strategy import _scar_ou_resolution as resolution
from pyscarcopula.strategy import scar_tm


@pytest.fixture
def fixed_point(monkeypatch):
    # Pin the returned candidate so real native probes can be compared across
    # modes without making the test depend on optimizer stopping criteria.
    def minimize(fun, x0, **kwargs):
        value, grad = fun(x0)
        return OptimizeResult(x=x0.copy(), fun=value, jac=grad, nfev=1,
                              message="fixed test candidate", success=True)
    monkeypatch.setattr(scar_tm, "minimize", minimize)


def run_fit(model, u, **kwargs):
    return model.fit(u, method="scar-tm-ou", alpha0=[8., .5, .8],
                     smart_init=False, **kwargs)


@pytest.mark.parametrize("corr, mode, status, count", [
    (None, "screen", "screen_passed", 3),
    ("cholesky", "full", "numerical_agreement", 5),
    ("shrinkage", "screen", "screen_passed", 3),
])
def test_native_fit_resolution_and_joint_score(fixed_point, corr, mode, status, count):
    u = np.random.default_rng(81).uniform(.12, .88, (22, 3 if corr else 2))
    model = (StochasticStudentCopula(d=3, corr_mode=corr) if corr
             else BivariateGaussianCopula())
    fit = run_fit(model, u, spectral_check_tolerances=(1e-5, 1e-4),
                  spectral_check_mode=mode)
    diag = fit.diagnostics
    assert fit.success and diag["last_backend"] == "spectral"
    assert diag["final_integration_status"] == status
    assert diag["final_integration_converged"] is (True if mode == "full" else None)
    probes = diag["spectral_resolution_check"]["probes"]
    assert len(probes) == count
    assert probes[0]["log_likelihood"] == fit.log_likelihood
    size = {None: 3, "shrinkage": 4, "cholesky": 6}[corr]
    assert all(len(row["score"]) == size for row in probes)
    assert np.all(np.ptp([row["score"] for row in probes], axis=0) <= 1e-4)
    assert fit.nfev == 1


def test_resolution_catches_score_with_constant_likelihood(fixed_point, monkeypatch):
    u = np.random.default_rng(46).uniform(.2, .8, (17, 2))
    reference = run_fit(BivariateGaussianCopula(), u)
    assert "spectral_resolution_check" not in reference.diagnostics
    calls = []
    def changed_score(u, copula, params, config):
        calls.append((config.basis_order, config.quad_order))
        return -reference.log_likelihood, np.full(3, 100.)
    monkeypatch.setattr(resolution, "_probe", changed_score)
    fit = run_fit(BivariateGaussianCopula(), u, spectral_check_tolerances=(1e-5, 1e-4))
    diag = fit.diagnostics
    assert calls == [(32, 100)]
    assert fit.success and fit.log_likelihood == reference.log_likelihood
    np.testing.assert_array_equal(fit.params.values, reference.params.values)
    assert diag["final_integration_status"] == "unresolved"
    assert diag["final_integration_converged"] is False
    assert diag["spectral_resolution_check"]["reason"] == "tolerance_exceeded"


def test_resolution_skip_and_native_failure(fixed_point, monkeypatch):
    u = np.random.default_rng(28).uniform(.15, .85, (17, 2))
    def failed(*args):
        raise NativeError("failed refined rule")
    monkeypatch.setattr(resolution, "_probe", failed)
    opts = dict(spectral_check_tolerances=(1e-5, 1e-4))
    skipped = run_fit(BivariateGaussianCopula(), u, transition_method="matrix", **opts)
    assert skipped.diagnostics["final_integration_status"] == "not_checked"
    assert skipped.diagnostics["spectral_resolution_check"]["reason"] == "final_backend_not_spectral"
    fit = run_fit(BivariateGaussianCopula(), u, transition_method="spectral", **opts)
    assert fit.success and fit.diagnostics["final_integration_converged"] is None
    check = fit.diagnostics["spectral_resolution_check"]
    assert check["reason"] == "probe_failure" and check["failed_orders"] == [32, 100]
    assert len(check["probes"]) == 1


def test_full_check_order_limit(fixed_point):
    u = np.random.default_rng(15).uniform(.15, .85, (17, 2))
    fit = run_fit(BivariateGaussianCopula(), u, spectral_basis_order=4,
                  spectral_quad_order=257, spectral_check_mode="full",
                  spectral_check_tolerances=(1., 1.))
    check = fit.diagnostics["spectral_resolution_check"]
    assert fit.success and fit.diagnostics["final_integration_status"] == "unresolved"
    assert check["pending_orders"] == [4, 1028] and check["reason"] == "order_limit"
    assert len(check["probes"]) == 2


def test_screen_detects_mode_sensitivity_at_minimal_quadrature(fixed_point, monkeypatch):
    u = np.random.default_rng(12).uniform(.25, .75, (17, 2))
    probe = resolution._probe
    def change_basis_only(u, copula, params, config):
        value, grad = probe(u, copula, params, config)
        if config.basis_order > 8:
            value += 10
        return value, grad
    monkeypatch.setattr(resolution, "_probe", change_basis_only)
    fit = run_fit(BivariateGaussianCopula(), u, spectral_basis_order=8,
                  spectral_quad_order=8, spectral_check_tolerances=(1., 1.))
    check = fit.diagnostics["spectral_resolution_check"]
    assert check["reason"] == "tolerance_exceeded"
    assert [(p["basis_order"], p["quad_order"]) for p in check["probes"]] == [
        (8, 8), (8, 10), (10, 10)]


@pytest.mark.parametrize("bad", [1e-4, (), (0., 1.), (1., np.inf), (True, 1.)])
def test_check_rejects_invalid_tolerance(bad):
    with pytest.raises(ValueError, match="spectral_check_tolerances"):
        scar_tm.SCARTMStrategy(spectral_check_tolerances=bad)

@pytest.mark.parametrize("method,corr", [("auto", None), ("spectral", "shrinkage")])
def test_optional_composite_restart(fixed_point, method, corr):
    u = np.random.default_rng(81).uniform(.12, .88, (22, 3 if corr else 2))
    model = (StochasticStudentCopula(d=3, corr_mode=corr) if corr
             else BivariateGaussianCopula())
    base = run_fit(model, u, transition_method=method)
    assert "refine_fit" not in base.diagnostics
    result = run_fit(model, u, transition_method=method, refine_fit=True,
                     refine_score_tol=100.)
    check = result.diagnostics["refine_fit"]
    assert check["status"] == "accepted", check
    assert check["basis_order"] > check["first_basis_order"]
    assert check["attempts"][-1]["error_per_obs"] <= 1e-8
    assert result.nfev >= 2
    strategy = scar_tm.SCARTMStrategy(transition_method=method)
    assert strategy.log_likelihood(model, u, result) == pytest.approx(result.log_likelihood)


@pytest.mark.parametrize("legacy", [False, True])
def test_refined_likelihood_honors_public_overrides(legacy):
    u = np.random.default_rng(81).uniform(.12, .88, (22, 2))
    model = BivariateGaussianCopula()
    fit = run_fit(model, u, transition_method="spectral", refine_fit=True)
    report = fit.diagnostics["refine_fit"]
    assert report["status"] == "accepted"
    if legacy:
        fit = replace(fit, diagnostics={"spectral_refit": report})
    assert log_likelihood(model, u, fit) == pytest.approx(fit.log_likelihood)

    plain = replace(fit, diagnostics={})
    override = dict(transition_method="matrix", K=15, adaptive=False, grid_range=2.)
    expected = log_likelihood(model, u, plain, **override)
    assert abs(expected - fit.log_likelihood) > 0.01
    assert scar_tm.SCARTMStrategy(**override).log_likelihood(model, u, fit) == pytest.approx(expected)
    assert log_likelihood(model, u, fit, **override) == pytest.approx(expected)

    override = dict(spectral_basis_order=12, spectral_quad_order=48)
    assert log_likelihood(model, u, fit, **override) == pytest.approx(
        log_likelihood(model, u, plain, **override))


def test_refined_auto_likelihood_preserves_unspecified_matrix_settings(fixed_point):
    u = np.random.default_rng(81).uniform(.2, .8, (22, 2))
    model = BivariateGaussianCopula()
    fit = run_fit(model, u, transition_method="auto", refine_fit=True, refine_score_tol=100.,
                  adaptive=False, K=51, refine_matrix_K=201, refine_matrix_grid_range=8.,
                  refine_matrix_max_K=501)
    assert fit.diagnostics["refine_fit"]["status"] == "accepted"
    plain = replace(fit, diagnostics={})
    expected = log_likelihood(model, u, plain, transition_method="matrix",
                              K=201, grid_range=3., max_K=501)
    assert log_likelihood(model, u, fit, transition_method="matrix",
                          grid_range=3.) == pytest.approx(expected)


def test_failed_refit_restores_joint_candidate(fixed_point):
    u = np.random.default_rng(82).uniform(.12, .88, (22, 3))
    model = StochasticStudentCopula(d=3, corr_mode="shrinkage")
    result = run_fit(model, u, refine_fit=True, refine_score_tol=1e-30)
    report = result.diagnostics["refine_fit"]
    assert report["status"] == "rejected"
    assert result.log_likelihood == report["first_log_likelihood"]
    np.testing.assert_allclose(model.R, result.diagnostics["corr_matrix"])

@pytest.mark.parametrize("shift", [-1., 1.])
def test_control_disagreement_keeps_first_fit(fixed_point, monkeypatch, shift):
    from pyscarcopula.strategy import _scar_ou_refit as refit

    probe = refit._probe
    calls = []
    def altered_probe(u, model, params, config):
        value, grad = probe(u, model, params, config)
        calls.append(config.basis_order)
        return value + (shift if len(calls) % 2 == 0 else 0.), grad
    monkeypatch.setattr(refit, "_probe", altered_probe)
    u = np.random.default_rng(80).uniform(.2, .8, (22, 2))
    fit = run_fit(BivariateGaussianCopula(), u, refine_fit=True,
                  refine_score_tol=100.)
    report = fit.diagnostics["refine_fit"]
    assert report["status"] == "rejected"
    assert report["first_log_likelihood"] == fit.log_likelihood
    assert all(a["error_per_obs"] > 1e-8 for a in report["attempts"])


@pytest.mark.parametrize("method,backend", [("auto", "local"), ("matrix", "matrix")])
def test_grid_restart_preserves_dispatch_and_warm_start(fixed_point, method, backend):
    u = np.random.default_rng(81).uniform(.2, .8, (22, 2))
    model = BivariateGaussianCopula()
    options = dict(transition_method=method, auto_small_kdt=1., adaptive=False, K=51)
    base = run_fit(model, u, **options)
    result = run_fit(model, u, **options, refine_fit=True, refine_score_tol=100.,
                     refine_matrix_K=201, refine_matrix_grid_range=8., refine_matrix_max_K=501)
    report = result.diagnostics["refine_fit"]
    assert report["status"] == "accepted", report
    assert result.nfev == 2
    assert result.transition_method == report["likelihood_config"]["transition_method"] == method
    assert report["final_backend"] == backend
    assert report["matrix_K"] == 201
    if method == "matrix":
        assert result.spectral_basis_order == base.spectral_basis_order
        assert result.spectral_quad_order == base.spectral_quad_order
    assert report["final_K_effective"] == (51 if method == "auto" else 201)
    if backend == "local":
        assert result.log_likelihood == base.log_likelihood
    strategy = scar_tm.SCARTMStrategy()
    assert strategy.log_likelihood(model, u, result) == pytest.approx(result.log_likelihood)


def test_matrix_refinement_default_spacing(fixed_point):
    u = np.random.default_rng(81).uniform(.2, .8, (22, 2))
    result = run_fit(BivariateGaussianCopula(), u, transition_method="matrix",
                     refine_fit=True, refine_score_tol=100., adaptive=False, K=51, grid_range=5.)
    report = result.diagnostics["refine_fit"]
    assert report["status"] == "accepted", report
    assert report["matrix_grid_range"] == 8.
    assert report["matrix_max_K"] == 10000
    assert report["final_grid_step"] == pytest.approx(report["first_grid_step"] / 2)


@pytest.mark.parametrize("mode,count", [("screen", 3), ("full", 5)])
def test_accepted_refit_checks_returned_candidate(mode, count):
    u = np.random.default_rng(81).uniform(.12, .88, (22, 2))
    result = run_fit(BivariateGaussianCopula(), u, transition_method="spectral",
                     refine_fit=True, spectral_check_tolerances=(1e-5, 1e-4),
                     spectral_check_mode=mode)
    assert result.diagnostics["refine_fit"]["status"] == "accepted"
    probes = result.diagnostics["spectral_resolution_check"]["probes"]
    assert len(probes) == count
    assert probes[0]["basis_order"] == result.spectral_basis_order
    assert probes[0]["log_likelihood"] == result.log_likelihood
    assert probes[-1]["basis_order"] > probes[0]["basis_order"]

@pytest.mark.parametrize("kwargs", [
    {"refine_matrix_K": 1}, {"refine_matrix_max_K": True},
    {"refine_matrix_grid_range": np.nan},
    {"refine_matrix_K": 101, "refine_matrix_max_K": 100},
])
def test_invalid_refined_grid(kwargs):
    with pytest.raises(ValueError):
        scar_tm.SCARTMStrategy(refine_fit=True, **kwargs)


def test_auto_local_refinement_uses_fitted_warm_start(monkeypatch):
    minimize = scar_tm.minimize
    starts, ends = [], []

    def record(fun, x0, **kwargs):
        starts.append(x0.copy())
        fit = minimize(fun, x0, **kwargs)
        ends.append(fit.x.copy())
        return fit

    monkeypatch.setattr(scar_tm, "minimize", record)
    u = np.random.default_rng(81).uniform(.12, .88, (22, 2))
    result = run_fit(BivariateGaussianCopula(), u, refine_fit=True, auto_small_kdt=1000.,
                     adaptive=False, K=41, log_stationary_scale_optimization=True)
    report = result.diagnostics["refine_fit"]
    assert report["first_backend"] == "local"
    assert len(starts) >= 2
    assert not np.allclose(starts[0], ends[0])
    for restart in starts[1:]:
        np.testing.assert_allclose(restart, ends[0])


@pytest.mark.parametrize("method", ["auto", "spectral"])
def test_custom_profile_reserves_finer_control(method):
    from pyscarcopula.strategy._scar_ou_refit import _profiles

    [(m, q, _, control_m, control_q, _)] = _profiles(method, 820, 1024)
    assert 820 < m < control_m <= 1024
    assert q == control_q == 1024
    assert _profiles(method, 1023, 1024) == []
    assert _profiles(method, 1024, 1024) == []
