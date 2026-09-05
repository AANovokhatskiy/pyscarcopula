"""Regression checks for native trial failures and transition support."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from pyscarcopula import GumbelCopula
from pyscarcopula._native import scar_ou
from pyscarcopula._native.errors import NativeError, NativeUnsupported
from pyscarcopula.copula.multivariate import StochasticStudentCopula
from pyscarcopula.numerical._scar_ou_config import AutoTMConfig
from pyscarcopula.strategy import scar_tm


@pytest.mark.parametrize("analytical", [True, False])
@pytest.mark.parametrize("mode", ["fixed", "shrinkage"])
def test_overflowing_optimizer_trial_is_a_finite_penalty(
        monkeypatch, analytical, mode):
    seen = []

    def minimize(fun, x0, **kwargs):
        trial = np.asarray(x0).copy()
        trial[0] = 5195.2982765  # Captured crypto6 optimizer trial.
        penalty = fun(trial)
        value = penalty[0] if analytical else penalty
        assert np.isfinite(value) and value >= 1e9
        if analytical:
            assert np.all(np.isfinite(penalty[1]))
            assert np.shape(penalty[1]) == np.shape(trial)
        seen.append(value)
        initial = fun(x0)
        return SimpleNamespace(
            x=np.asarray(x0), fun=initial[0] if analytical else initial,
            success=False, message="test endpoint", nfev=2)

    monkeypatch.setattr(scar_tm, "minimize", minimize)
    u = np.random.default_rng(87).uniform(.02, .98, (20, 3))
    model = StochasticStudentCopula(
        3, corr_mode=mode, **({"R": np.eye(3)} if mode == "fixed" else {}))
    model.fit(
        u, method="SCAR-TM-OU", to_pobs=False,
        alpha0=np.array([2., 1., .5]), analytical_grad=analytical,
        smart_init=False, K=30, adaptive=False, transition_method="matrix")
    assert seen


def test_trial_conversion_does_not_hide_unsupported_or_final_failure():
    def unsupported(values):
        raise NativeUnsupported("missing kernel")

    with pytest.raises(NativeUnsupported, match="missing kernel"):
        scar_tm._trial_parameters(unsupported, np.ones(3), {})
    with pytest.raises(NativeError):
        scar_tm._ou_from_log_stationary(np.array([5195., 1., -6.9]))


def test_sparse_band_boundary_matches_dense_value_and_gradient():
    # Cross an integer 8-sigma support width; the former 5-sigma truncation
    # created objective jumps large enough to break a bootstrap line search.
    n, K = 120, 80
    kappa = -.5 * (n - 1) * np.log1p(-(2.5 * 10 / (K - 1)) ** 2)
    u = np.random.default_rng(724).uniform(.02, .98, (n, 2))
    copula = GumbelCopula(rotate=180)
    config = AutoTMConfig(transition_method="matrix", K=K,
                          adaptive=False, max_K=None, grid_method="sparse")
    values = []
    for delta in (-1e-5, 0., 1e-5):
        args = (kappa + delta, 1.7, 9.1, u, copula)
        sparse, gradient = scar_ou.neg_loglik_with_grad(*args, config)
        dense, dense_gradient = scar_ou.neg_loglik_with_grad(
            *args, replace(config, grid_method="dense"))
        assert sparse == pytest.approx(dense, abs=1e-9)
        np.testing.assert_allclose(gradient, dense_gradient, rtol=1e-9, atol=1e-9)
        values.append(sparse)
        if delta == 0:
            center_gradient = gradient[0]
    assert (values[2] - values[0]) / 2e-5 == pytest.approx(
        center_gradient, rel=2e-5, abs=2e-6)


def test_explicit_matrix_reports_grid_cap_without_fallback():
    u = np.random.default_rng(6).uniform(.1, .9, (20, 2))
    config = AutoTMConfig(transition_method="matrix", K=30, max_K=60)
    _, info = scar_ou.neg_loglik_info(
        .013, 1., .5, u, GumbelCopula(), config)
    assert info["K_requested"] > 60
    assert info["K_effective"] == 60
    assert info["grid_was_capped"] is True
    assert "matrix_fallback_reason" not in info


@pytest.mark.parametrize("retry_success", [True, False])
def test_bootstrap_retries_same_sample_and_rejects_failed_statistics(
        monkeypatch, retry_success):
    from pyscarcopula import stattests
    from pyscarcopula._types import DEFAULT_CONFIG

    sample = np.full((10, 2), .5)
    seen = []
    statistics = []
    fit_result = SimpleNamespace(method="MLE", success=True,
                                 log_likelihood=1., copula_param=2.)

    def refit(*args):
        seen.append((args[2], dict(args[4])))
        return object(), SimpleNamespace(
            method="MLE", copula_param=3., log_likelihood=2.,
            success=retry_success and len(seen) == 2, message="line search")

    def statistic(*args):
        statistics.append(1)
        return .1

    adapter = stattests._BootstrapAdapter(
        capture=lambda *args: None, prepare=lambda *args: object(),
        simulate=lambda *args: sample, refit=refit, statistic=statistic)
    monkeypatch.setitem(stattests._BOOTSTRAP_ADAPTERS, "test", adapter)
    task = (0, "test", object, {}, sample, fit_result, None, .2, 30, 5.,
            True, {"config": DEFAULT_CONFIG}, np.random.SeedSequence(6), 1)
    if retry_success:
        value, row = stattests._bootstrap_gof_worker(task)
        assert value == .1
        assert row["bootstrap_refit_retries"] == 1
        assert len(row["bootstrap_refit_attempts"]) == 2
        assert not row["bootstrap_refit_attempts"][0]["bootstrap_fit_success"]
        assert statistics == [1]
    else:
        with pytest.raises(RuntimeError, match="after 2 attempts"):
            stattests._bootstrap_gof_worker(task)
        assert not statistics
    assert len(seen) == 2
    assert all(item[0] is sample for item in seen)
    np.testing.assert_array_equal(seen[1][1]["alpha0"], [3.])
