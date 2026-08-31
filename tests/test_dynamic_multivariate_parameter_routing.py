"""Regression coverage for dynamic multivariate public parameter routes."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from pyscarcopula import EquicorrGaussianCopula, StochasticStudentCopula, api
from pyscarcopula._native import scar_ou
from pyscarcopula._types import PredictConfig
from pyscarcopula.numerical._scar_ou_config import AutoTMConfig
from pyscarcopula.strategy.mle import MLEStrategy


U = np.random.default_rng(421).uniform(.08, .92, (32, 4))
R = np.eye(4) * .72 + .28
LOADINGS = np.array([[.4], [.3], [-.2], [.35]])
OU = dict(alpha0=[1.3, .15, .55], K=24, grid_range=3.7,
          grid_method="dense", adaptive=False, pts_per_sigma=5,
          transition_method="matrix", max_K=48, gh_order=7, r_gh=2.7,
          auto_small_kdt=.03, spectral_basis_order=12,
          spectral_quad_order=24, maxiter=2)


def make_model(kind):
    if kind.startswith("equicorr"):
        return EquicorrGaussianCopula(4)
    return StochasticStudentCopula(4, R=R)


def fit_model(kind, method, **overrides):
    model = make_model(kind)
    data = (model.prepare_sufficient_statistics(U)
            if kind == "equicorr-prepared" else U)
    options = dict(OU) if method == "SCAR-TM-OU" else (
        dict(gamma0=[.025, .018, .73], maxiter=2) if method == "GAS" else {})
    options.update(overrides)
    result = model.fit(data, method=method, **options)
    assert np.isfinite(result.log_likelihood)
    return model, data, result


@pytest.fixture(scope="module", params=[
    (kind, method) for kind in ("equicorr", "equicorr-prepared", "student")
    for method in ("MLE", "GAS", "SCAR-TM-OU")])
def fitted(request):
    return fit_model(*request.param)


@pytest.mark.parametrize("rotation", [90, 45, "0", 0j, False, None, np.nan])
def test_equicorr_rejects_unsupported_rotation(rotation):
    with pytest.raises((TypeError, ValueError), match="rotate"):
        EquicorrGaussianCopula(4, rotate=rotation)


@pytest.mark.parametrize("rotation", [0, 0., np.int64(0)])
def test_equicorr_zero_rotation_is_unchanged(rotation):
    actual = EquicorrGaussianCopula(4, rotate=rotation).log_pdf_rows(U, .2)
    np.testing.assert_array_equal(actual, EquicorrGaussianCopula(4).log_pdf_rows(U, .2))


@pytest.mark.parametrize("entry", ["predict", "predict_batches"])
@pytest.mark.parametrize("n", [0, 3])
@pytest.mark.parametrize("option,value", [
    ("dynamic_conditioning", "given_only"), ("return_diagnostics", True),
    ("mcmc_steps", 3), ("mcmc_burnin", 2)])
def test_object_prediction_rejects_vine_config(fitted, entry, n, option, value):
    model, _, _ = fitted
    with pytest.raises(TypeError, match=option):
        output = getattr(model, entry)(n, predict_config=PredictConfig(**{option: value}))
        if entry.endswith("batches"):
            list(output)


@pytest.mark.parametrize("entry", ["predict", "predict_batches"])
@pytest.mark.parametrize("option,value", [
    ("horizon", "past"), ("predictive_r_mode", "invalid")])
def test_prediction_controls_validated_without_config(fitted, entry, option, value):
    with pytest.raises(ValueError, match=option):
        output = getattr(fitted[0], entry)(3, **{option: value})
        if entry.endswith("batches"):
            list(output)


@pytest.mark.parametrize("entry", ["predict", "predict_batches", "sample_batches"])
@pytest.mark.parametrize("given", [{99: .5}, {0: 0.}, {True: .5}])
def test_zero_count_still_validates_given(fitted, entry, given):
    with pytest.raises((TypeError, ValueError), match="given"):
        output = getattr(fitted[0], entry)(0, given=given)
        if entry.endswith("batches"):
            list(output)


@pytest.mark.parametrize("entry", ["predict", "predict_batches", "sample_batches"])
def test_zero_and_all_given_prediction(fitted, entry):
    model, _, _ = fitted
    empty = getattr(model, entry)(0, given={1: .4})
    if entry.endswith("batches"):
        assert list(empty) == []
    else:
        assert empty.shape == (0, 4)
    output = getattr(model, entry)(3, given={0: .2, 1: .4, 2: .6, 3: .8})
    if entry.endswith("batches"):
        output = np.concatenate(list(output))
    np.testing.assert_array_equal(output, np.tile([.2, .4, .6, .8], (3, 1)))


@pytest.mark.parametrize("kind", ["equicorr", "equicorr-prepared", "student"])
@pytest.mark.parametrize("method", ["MLE", "GAS"])
def test_xt_rejects_non_ou_fit(kind, method, monkeypatch):
    model, data, _ = fit_model(kind, method)
    def no_native(*args, **kwargs):
        pytest.fail("incompatible fitted parameters reached an OU kernel")
    monkeypatch.setattr(scar_ou, "state_distribution", no_native)
    monkeypatch.setattr(scar_ou, "prepare_objective", no_native)
    with pytest.raises(ValueError, match="SCAR-TM-OU"):
        model.xT_distribution(data)


def test_smoothing_rejects_gas_but_accepts_explicit_ou_parameters():
    model, data, _ = fit_model("student", "GAS")
    with pytest.raises(ValueError, match="SCAR-TM-OU"):
        model.posterior_state_weights(data)
    weights = model.posterior_state_weights(data, params=[1.3, .15, .55],
                                             K=24, adaptive=False)
    np.testing.assert_allclose(weights.sum(axis=1), 1., atol=1e-12)


@pytest.mark.parametrize("kind", ["equicorr", "equicorr-prepared", "student"])
@pytest.mark.parametrize("transition", ["matrix", "local", "spectral", "auto"])
def test_xt_inherits_fit_settings_and_honors_explicit_grid(kind, transition):
    model, data, result = fit_model(kind, "SCAR-TM-OU", transition_method=transition)
    cfg = AutoTMConfig(K=24, grid_range=3.7, grid_method="dense", adaptive=False,
                       pts_per_sigma=5, transition_method=(
                           "auto" if transition == "spectral" else transition),
                       max_K=48, gh_order=7, r_gh=2.7, small_kdt=.03,
                       basis_order=12, quad_order=24)
    saved = deepcopy(result)
    for overrides, expected_cfg in [({}, cfg),
                                    (dict(K=28, grid_range=4.2),
                                     replace(cfg, K=28, grid_range=4.2))]:
        actual = model.xT_distribution(data, **overrides)
        expected = scar_ou.state_distribution(*result.params.values, U, model, expected_cfg)
        for a, b in zip(actual, expected):
            np.testing.assert_allclose(a, b, atol=1e-12, rtol=1e-12)
        assert len(actual[0]) == expected_cfg.K
    assert result.K == saved.K and result.grid_range == saved.grid_range
    with pytest.raises(ValueError, match="u is required"):
        model.xT_distribution(None)


@pytest.mark.parametrize("transition", ["matrix", "local", "spectral", "auto"])
def test_smoothing_inherits_all_settings_and_preserves_explicit_none(transition, monkeypatch):
    model, data, result = fit_model("student", "SCAR-TM-OU", transition_method=transition)
    original = scar_ou.smoothed_state_distribution
    calls = []
    def capture(*args, **kwargs):
        calls.append(args[5])
        return original(*args, **kwargs)
    monkeypatch.setattr(scar_ou, "smoothed_state_distribution", capture)
    implicit = model.posterior_state_weights(data)
    expected = AutoTMConfig(K=24, grid_range=3.7, grid_method="dense", adaptive=False,
                            pts_per_sigma=5, transition_method=(
                                "auto" if transition == "spectral" else transition),
                            max_K=48, gh_order=7, r_gh=2.7, small_kdt=.03,
                            basis_order=12, quad_order=24)
    assert calls[-1] == expected
    explicit = model.posterior_state_weights(data, params=result.params.values)
    np.testing.assert_allclose(implicit, explicit, rtol=0, atol=0)
    model.posterior_state_weights(data, K=28, grid_range=4.2, adaptive=False,
                                  transition_method="local", max_K=None,
                                  gh_order=9, r_gh=2.5)
    assert calls[-1] == replace(expected, K=28, grid_range=4.2,
                                transition_method="local", max_K=None,
                                gh_order=9, r_gh=2.5)
    assert result.max_K == 48 and result.K == 24


@pytest.mark.parametrize("transition", ["matrix", "local", "spectral", "auto"])
def test_fitted_ou_likelihood_accepts_prepared_data_and_overrides(transition):
    model, prepared, result = fit_model("equicorr-prepared", "SCAR-TM-OU",
                                        transition_method=transition)
    expected = api.log_likelihood(model, U, result)
    assert model.log_likelihood(prepared) == pytest.approx(expected, abs=1e-10)
    assert api.log_likelihood(model, prepared, result) == pytest.approx(expected, abs=1e-10)
    for data in (prepared, U):
        actual = api.log_likelihood(model, data, result, K=28, transition_method="local")
        dense = api.log_likelihood(model, U, result, K=28, transition_method="local")
        assert actual == pytest.approx(dense, abs=1e-10)


@pytest.mark.parametrize("estimation", ["two-stage", "joint"])
@pytest.mark.parametrize("entry", ["log_pdf_rows", "dlog_pdf_dr_rows",
    "log_pdf_and_dlog_dr_rows", "pdf_and_grad_on_grid_batch", "copula_grid_batch"])
@pytest.mark.parametrize("options", [dict(cache=object()), dict(t_index=-1),
    dict(t_index=True), dict(t_index=1.5), dict(t_index=1)])
def test_factor_emissions_reject_unsupported_cache_offsets(estimation, entry, options):
    model = StochasticStudentCopula(4, corr_mode="factor", factor_rank=1,
                                     factor_loadings=LOADINGS, factor_estimation=estimation)
    parameter = [.1, .5] if "grid" in entry else 7.
    with pytest.raises((TypeError, ValueError), match="cache|t_index"):
        getattr(model, entry)(U, parameter, **options)
    default = getattr(model, entry)(U, parameter)
    explicit = getattr(model, entry)(U, parameter, cache=None, t_index=0)
    np.testing.assert_array_equal(default, explicit)


@pytest.mark.parametrize("mode", ["fixed", "shrinkage", "cholesky"])
def test_joint_factor_policy_requires_factor_mode(mode):
    with pytest.raises(ValueError, match="factor_estimation.*factor"):
        StochasticStudentCopula(4, corr_mode=mode, factor_estimation="joint")


@pytest.mark.parametrize("entry", ["object", "api", "api-raw", "strategy"])
def test_joint_factor_mle_dispatch(entry):
    model = StochasticStudentCopula(4, corr_mode="factor", factor_rank=1,
                                     factor_loadings=LOADINGS, factor_estimation="joint")
    if entry == "strategy":
        result = MLEStrategy().fit(model, U)
    elif entry == "object":
        result = model.fit(U, method="MLE")
    else:
        result = api.fit(model, U, method="MLE", to_pobs=entry == "api-raw")
    assert result.method == "MLE" and result.success, result.message
    data = model._last_u
    assert model.log_likelihood(data) == pytest.approx(result.log_likelihood, abs=1e-7)
    for method in ("GAS", "SCAR-TM-OU"):
        with pytest.raises(NotImplementedError, match="joint"):
            api.fit(model, U, method=method)
