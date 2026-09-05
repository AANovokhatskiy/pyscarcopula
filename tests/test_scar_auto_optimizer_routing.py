"""Auto backend selection follows trial parameters inside optimizer callbacks."""
from types import SimpleNamespace

import numpy as np
import pytest

from pyscarcopula.copula.elliptical import BivariateGaussianCopula
from pyscarcopula.copula.multivariate import StochasticStudentCopula
from pyscarcopula.strategy import scar_tm


def _trace_trial_backends(monkeypatch, model, observations, *, mu, sigma,
                          **strategy_kwargs):
    threshold = .01 * (len(observations) - 1)
    trial_info = []
    in_optimizer = False
    record = scar_tm._record_backend_diagnostics

    def capture(diagnostics, info, engine):
        if in_optimizer:
            trial_info.append(dict(info))
        record(diagnostics, info, engine)

    def controlled_optimizer(fun, x0, **kwargs):
        nonlocal in_optimizer
        in_optimizer = True
        try:
            for relative_kappa in (.5, 2., .5, 2.):
                point = np.array(x0).copy()
                point[:3] = [np.log(threshold * relative_kappa), mu,
                             np.log(sigma)]
                value, gradient = fun(point)
                assert np.isfinite(value)
                assert np.all(np.isfinite(gradient))
        finally:
            in_optimizer = False
        return SimpleNamespace(x=point, fun=value, jac=gradient, success=False,
                               message="routing contract", nfev=4, nit=0)

    monkeypatch.setattr(scar_tm, "_record_backend_diagnostics", capture)
    monkeypatch.setattr(scar_tm, "minimize", controlled_optimizer)
    model.fit(observations, method="scar-tm-ou", to_pobs=False,
              alpha0=[threshold * 2, mu, sigma * np.sqrt(4 * threshold)],
              transition_method="auto", log_stationary_scale_optimization=True,
              K=300, max_K=300, adaptive=False, **strategy_kwargs)
    assert [item["selected_backend"] for item in trial_info] == [
        "local", "spectral", "local", "spectral"]
    return trial_info


@pytest.mark.parametrize("mode", ["fixed", "shrinkage", "factor"])
def test_student_optimizer_reselects_auto_backend_for_each_trial(monkeypatch, mode):
    observations = np.random.default_rng(857).uniform(.02, .98, (50, 3))
    model = StochasticStudentCopula(
        3, corr_mode=mode, **({"factor_rank": 1} if mode == "factor" else {}))
    info = _trace_trial_backends(monkeypatch, model, observations, mu=3., sigma=.1)
    assert [item["backend"] for item in info] == [
        "local", "spectral", "local", "spectral"]


def test_optimizer_auto_falls_back_to_matrix_for_failed_spectral_trial(monkeypatch):
    observations = np.random.default_rng(1).uniform(.001, .999, (50, 2))
    # Two spectral nodes see nearly singular Gaussian correlations, whereas
    # the matrix grid resolves finite-density states. Native kernels are real.
    info = _trace_trial_backends(
        monkeypatch, BivariateGaussianCopula(), observations, mu=0., sigma=10./np.sqrt(.2),
        spectral_basis_order=2, spectral_quad_order=2)
    assert [item["backend"] for item in info] == [
        "local", "matrix", "local", "matrix"]
    assert info[1]["fallback_chain"] == ["spectral"]
    assert info[3]["fallback_chain"] == ["spectral"]
