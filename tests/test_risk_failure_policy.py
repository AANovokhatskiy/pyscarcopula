"""Rolling-risk failure policy at the fit and portfolio optimizer boundaries."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from pyscarcopula import EquicorrGaussianCopula, IndependentCopula, VineCopula
from pyscarcopula.contrib import risk_metrics as risk
from pyscarcopula.contrib.marginal import MarginalModel


def _returns():
    return np.random.default_rng(704).normal(0.0, 0.01, (18, 2))


def _fitted_equicorr():
    model = EquicorrGaussianCopula(d=2)
    fitted = model.fit(np.random.default_rng(773).uniform(0.08, 0.92, (80, 2)))
    assert fitted.success
    return model, fitted


def _public_risk(model, optimize, **kwargs):
    return risk.risk_metrics(
        model, _returns(), window_len=17, gamma=0.9, N_mc=17,
        marginals_method="normal", optimize_portfolio=optimize, rng=1,
        **kwargs,
    )


@pytest.mark.parametrize("optimize", [False, True])
def test_default_rejects_real_failed_refit_before_stale_prediction(monkeypatch, optimize):
    model, accepted = _fitted_equicorr()
    attempted = []
    original_fit = EquicorrGaussianCopula.fit

    def observe_fit(self, *args, **kwargs):
        result = original_fit(self, *args, **kwargs)
        attempted.append(result)
        return result

    def forbid_prediction(*args, **kwargs):
        pytest.fail("Failed rolling fit reached prediction of retained state")

    monkeypatch.setattr(EquicorrGaussianCopula, "fit", observe_fit)
    monkeypatch.setattr(risk, "_predict_copula", forbid_prediction)
    with pytest.raises(RuntimeError, match="(?i)(fit|copula)") as failure:
        _public_risk(model, optimize, maxiter=1, maxfun=4)
    assert "window" in str(failure.value).lower()
    assert len(attempted) == 1 and not attempted[0].success
    assert model.fit_result is accepted


@pytest.mark.parametrize("optimize", [False, True])
def test_continue_preserves_legacy_stale_state_after_real_failed_refit(optimize):
    model, accepted = _fitted_equicorr()
    result = _public_risk(
        model, optimize, maxiter=1, maxfun=4, failure_policy="continue",
    )[0.9][17]
    assert model.fit_result is accepted
    assert np.isfinite(result["var"][16:]).all()
    assert np.isfinite(result["cvar"][16:]).all()


@pytest.mark.parametrize("optimize", [False, True])
@pytest.mark.parametrize("policy", ["raise", "continue"])
def test_real_spawn_failed_fit_policy_and_exception_propagation(optimize, policy):
    # Workers construct fresh models. Under continue, the original prediction
    # exception must survive; under raise, failure is reported at fit instead.
    expected = RuntimeError if policy == "raise" else (RuntimeError, ValueError)
    pattern = "(?i)(fit|copula).*" if policy == "raise" else "(?i)fit first"
    with pytest.raises(expected, match=pattern) as failure:
        _public_risk(
            EquicorrGaussianCopula(d=2), optimize, n_jobs=2,
            mp_start_method="spawn", maxiter=1, maxfun=4,
            failure_policy=policy,
        )
    assert failure.value.__cause__ is not None
    assert "RemoteTraceback" in type(failure.value.__cause__).__name__
    if policy == "raise":
        assert "window" in str(failure.value).lower()


@pytest.mark.parametrize("final_success", [False, True])
def test_fit_policy_uses_returned_attempt_not_retained_model_result(monkeypatch, final_success):
    model = IndependentCopula()
    seed_result = model.fit(np.random.default_rng(842).uniform(0.1, 0.9, (17, 2)))
    retained = replace(seed_result, success=not final_success, message="retained")
    attempt = replace(seed_result, success=final_success, message="current attempt")
    model.fit_result = retained
    monkeypatch.setattr(IndependentCopula, "fit", lambda self, *a, **kw: attempt)
    if final_success:
        risk._fit_copula(model, np.full((17, 2), 0.5), "mle", window_index=16)
    else:
        with pytest.raises(RuntimeError, match="current attempt"):
            risk._fit_copula(model, np.full((17, 2), 0.5), "mle", window_index=16)
    assert model.fit_result is retained


def test_vine_fit_returning_self_is_checked_via_its_final_result():
    model = VineCopula(candidates=[IndependentCopula])
    risk._fit_copula(
        model, np.random.default_rng(971).uniform(0.05, 0.95, (19, 2)),
        "mle", window_index=18,
    )
    assert model.fit_result.success


@pytest.mark.parametrize("final_success", [False, True])
def test_vine_final_result_controls_policy_not_failed_attempt_diagnostics(monkeypatch, final_success):
    model = VineCopula(candidates=[IndependentCopula])

    def return_self(self, *args, **kwargs):
        self.fit_result = OptimizeResult(
            success=final_success, message="final vine fit",
            diagnostics={
                "fallback_count": 1,
                "fallback_edges": [{"attempted_success": False, "actual_success": True}],
                "optimizer_refinement": {"first_success": False, "selected_stage": "refined"},
            },
        )
        return self

    monkeypatch.setattr(VineCopula, "fit", return_self)
    if final_success:
        risk._fit_copula(model, np.full((17, 2), 0.5), "mle", window_index=16)
    else:
        with pytest.raises(RuntimeError, match="final vine fit"):
            risk._fit_copula(model, np.full((17, 2), 0.5), "mle", window_index=16)


def _pipeline(optimize, execution, policy):
    data = _returns()
    marginal = MarginalModel.create("normal")
    params = marginal.fit_rolling(data, 17, n_jobs=1)
    seeds = np.random.SeedSequence(2008).spawn(2)
    weights = np.array([0.5, 0.5])
    if execution == "worker":
        common = (0, 2, data, "mle", IndependentCopula, {}, marginal, params, 0.9, 17, 17)
        if optimize:
            return risk._process_chunk_optimal(
                common + ({}, seeds), failure_policy=policy,
            )
        return risk._process_chunk_fixed(
            common + (weights, {}, seeds), failure_policy=policy,
        )
    common = (IndependentCopula(), data, "mle", marginal, params, 0.9, 17, 17)
    kwargs = dict(n_jobs=1, window_seed_sequences=seeds, failure_policy=policy)
    if optimize:
        return risk._calculate_cvar_optimal(*common, **kwargs)
    return risk._calculate_cvar_fixed(*common, weights, **kwargs)


@pytest.mark.parametrize("optimize", [False, True])
@pytest.mark.parametrize("execution", ["serial", "worker"])
def test_failed_portfolio_result_raises_before_reading_values_or_next_window(monkeypatch, optimize, execution):
    calls = []

    def failed_optimizer(*args, **kwargs):
        calls.append(1)
        # x/fun are absent: rejection must happen before reading a candidate
        # for publication or for the optimized portfolio's next warm start.
        return OptimizeResult(success=False, message="portfolio limit reached")

    monkeypatch.setattr(risk, "minimize", failed_optimizer)
    with pytest.raises(RuntimeError, match="portfolio limit reached") as failure:
        _pipeline(optimize, execution, "raise")
    assert "window" in str(failure.value).lower()
    assert len(calls) == 1


@pytest.mark.parametrize("optimize", [False, True])
@pytest.mark.parametrize("execution", ["serial", "worker"])
def test_continue_publishes_failed_portfolio_candidate_and_preserves_warm_start(monkeypatch, optimize, execution):
    seen_x0 = []
    candidate = np.array([0.25, 0.3, 0.7]) if optimize else np.array([0.25])

    def failed_optimizer(objective, x0, **kwargs):
        seen_x0.append(np.asarray(x0).copy())
        return OptimizeResult(success=False, message="portfolio limit reached", x=candidate.copy(), fun=0.4)

    monkeypatch.setattr(risk, "minimize", failed_optimizer)
    result = _pipeline(optimize, execution, "continue")
    assert len(seen_x0) == 2
    if execution == "serial":
        np.testing.assert_array_equal(result[0][16:], [0.25, 0.25])
        np.testing.assert_array_equal(result[1][16:], [0.4, 0.4])
    else:
        assert [(r[0], r[1], r[2]) for r in result] == [(16, 0.25, 0.4), (17, 0.25, 0.4)]
    if optimize:
        np.testing.assert_array_equal(seen_x0[1], candidate)


@pytest.mark.parametrize("owner", ["fit", "predict", "minimize"])
@pytest.mark.parametrize("policy", ["raise", "continue"])
@pytest.mark.parametrize("optimize", [False, True])
@pytest.mark.parametrize("execution", ["serial", "worker"])
def test_policy_never_swallows_pipeline_exceptions(monkeypatch, owner, policy, optimize, execution):
    failure = LookupError("owner failure is not a success flag")

    def fail(*args, **kwargs):
        raise failure

    if owner == "fit":
        monkeypatch.setattr(IndependentCopula, "fit", fail)
    else:
        monkeypatch.setattr(risk, "_predict_copula" if owner == "predict" else "minimize", fail)
    with pytest.raises(LookupError) as caught:
        _pipeline(optimize, execution, policy)
    assert caught.value is failure
