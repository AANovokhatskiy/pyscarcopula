"""Regression checks for native trial failures and transition support."""

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


@pytest.mark.parametrize(
    ("delta", "expected", "expected_gradient"),
    [
        (-1e-5, 7.032874843067936,
         [1.3422102023226288, 2.371028126669939, -0.7199275160889503]),
        (0., 7.032888265172675,
         [1.3422107432369996, 2.3710338270584295, -0.7199291532891187]),
        (1e-5, 7.032892226477369,
         [1.342189053803724, 2.3710386400947225, -0.7199303878884512]),
    ],
)
def test_five_sigma_band_boundary_preserves_released_values(
        delta, expected, expected_gradient):
    # Released 0.20.1 references on both sides of band=ceil(5*2.4).
    # The contract retains the discrete band change; it does not require
    # equality to a full dense transition across this boundary.
    n, K = 120, 80
    kappa = -.5 * (n - 1) * np.log1p(-(2.4 * 10 / (K - 1)) ** 2)
    u = np.random.default_rng(724).uniform(.02, .98, (n, 2))
    copula = GumbelCopula(rotate=180)
    config = AutoTMConfig(transition_method="matrix", K=K,
                          adaptive=False, max_K=None, grid_method="sparse")
    args = (kappa + delta, 1.7, 9.1, u, copula)
    value, gradient = scar_ou.neg_loglik_with_grad(*args, config)
    assert value == pytest.approx(expected, rel=0, abs=1e-10)
    np.testing.assert_allclose(
        gradient, expected_gradient, rtol=1e-10, atol=1e-11)
    assert scar_ou.neg_loglik(*args, config) == pytest.approx(
        value, rel=0, abs=1e-10)


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


@pytest.mark.parametrize(
    "method,message,configured,explicit",
    [
        ("SCAR-TM-OU", "ABNORMAL: ", 20, None),
        ("SCAR-TM-OU", "ABNORMAL_TERMINATION_IN_LNSRCH", 250, None),
        ("SCAR-TM-OU", "ABNORMAL: ", 250, 150),
        ("SCAR-TM-OU", "ABNORMAL: ", 20, 50),
        ("SCAR-TM-OU", "ITERATION LIMIT", 20, None),
        ("GAS", "ABNORMAL: ", 20, None),
        ("MLE", "ABNORMAL: ", 20, None),
    ],
)
def test_bootstrap_retry_preserves_optimizer_settings(
        monkeypatch, method, message, configured, explicit):
    from pyscarcopula import stattests
    from pyscarcopula._types import (
        LBFGSBConfig, NumericalConfig, LatentResult, ou_params)

    config = NumericalConfig(
        bivariate_scar_optimizer=LBFGSBConfig(maxls=configured))
    sample = np.full((10, 2), .5)
    initial = LatentResult(
        method=method, copula_name="GumbelCopula", log_likelihood=1.,
        success=True, params=ou_params(2., 1., .5))
    calls = []

    def refit(*args):
        assert args[2] is sample
        assert args[8] is config
        calls.append(dict(args[4]))
        return GumbelCopula(), LatentResult(
            method=method, copula_name="GumbelCopula", log_likelihood=2.,
            success=len(calls) == 2, message=message,
            params=ou_params(3., 1., .5))

    adapter = stattests._BootstrapAdapter(
        capture=lambda *args: None, prepare=lambda *args: GumbelCopula(),
        simulate=lambda *args: sample, refit=refit,
        statistic=lambda *args: .1)
    monkeypatch.setitem(stattests._BOOTSTRAP_ADAPTERS, "test", adapter)
    kwargs = {"config": config, "maxls": explicit}
    task = (0, "test", GumbelCopula, {}, sample, initial, None, .2, 30, 5.,
            True, kwargs, np.random.SeedSequence(6), 1)
    _, row = stattests._bootstrap_gof_worker(task)
    assert calls[0] == {"maxls": explicit}
    assert calls[1].get("maxls") == explicit
    assert kwargs == {"config": config, "maxls": explicit}
    assert row["bootstrap_refit_retries"] == 1
    assert row["bootstrap_fit_success"]
