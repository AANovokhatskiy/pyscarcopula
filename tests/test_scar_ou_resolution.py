import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from pyscarcopula import BivariateGaussianCopula, StochasticStudentCopula
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
