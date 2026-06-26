from types import SimpleNamespace

import numpy as np
import pytest

from pyscarcopula.api import fit
from pyscarcopula._types import LatentResult, MultivariateMLEResult, ou_params
from pyscarcopula._utils import pobs
from pyscarcopula.copula.multivariate.corr_param import (
    _corr_from_cholesky_params,
    _corr_gradient_to_raw_params,
    make_shrinkage_corr,
    pack_cholesky_corr,
    unpack_cholesky_corr,
    validate_corr_matrix,
)
from pyscarcopula.copula.multivariate.stochastic_student import (
    StochasticStudentCopula,
)
from pyscarcopula.copula.multivariate import stochastic_student
from pyscarcopula.numerical import _cpp_scar_ou
from pyscarcopula.numerical._scar_ou_config import AutoTMConfig
from pyscarcopula.numerical.tm_functions import tm_loglik
from pyscarcopula.strategy import scar_tm


def _u(T=60, d=3, seed=123):
    rng = np.random.default_rng(seed)
    return pobs(rng.standard_t(df=5.0, size=(T, d)))


def _R(d=3, rho=0.35):
    R = np.full((d, d), rho, dtype=np.float64)
    np.fill_diagonal(R, 1.0)
    return R


@pytest.mark.parametrize("corr_mode", ["shrinkage", "cholesky"])
def test_corr_gradient_chain_rule_matches_finite_difference(corr_mode):
    d = 4
    corr_gradient = np.array(
        [0.7, -0.4, 0.2, 0.9, -0.3, 0.5], dtype=np.float64)
    if corr_mode == "shrinkage":
        params = np.array([0.35], dtype=np.float64)
        corr_base = _R(d=d, rho=0.25)

        def make_corr(raw):
            return make_shrinkage_corr(raw[0], corr_base)
    else:
        params = np.array(
            [0.2, -0.1, 0.3, 0.15, -0.25, 0.4],
            dtype=np.float64,
        )
        corr_base = None

        def make_corr(raw):
            return _corr_from_cholesky_params(raw, d)

    R = make_corr(params)
    analytical = _corr_gradient_to_raw_params(
        corr_mode, params, R, corr_gradient, corr_base)
    finite_difference = np.empty_like(params)
    step = 1e-6

    def linear_functional(raw):
        trial_R = make_corr(raw)
        value = 0.0
        pos = 0
        for i in range(1, d):
            for j in range(i):
                value += corr_gradient[pos] * trial_R[i, j]
                pos += 1
        return value

    for index in range(params.size):
        plus = params.copy()
        minus = params.copy()
        plus[index] += step
        minus[index] -= step
        finite_difference[index] = (
            linear_functional(plus) - linear_functional(minus)
        ) / (2.0 * step)

    np.testing.assert_allclose(
        analytical, finite_difference, rtol=0.0, atol=2e-9)


def test_fixed_mode_keeps_canonical_behavior():
    u = _u()
    R = _R()
    copula = StochasticStudentCopula(d=3, R=R, corr_mode="fixed")

    assert copula._corr_num_params() == 0
    assert np.isfinite(copula.log_likelihood(u, r=5.0))

    mle_result = copula.fit(u, method="mle", maxiter=2)
    assert mle_result.n_params == 1
    assert mle_result.diagnostics["corr_mode"] == "fixed"
    assert mle_result.diagnostics["corr_n_params"] == 0
    np.testing.assert_allclose(
        mle_result.diagnostics["corr_matrix"], copula.R)

    scar = StochasticStudentCopula(d=3, R=R, corr_mode="fixed")
    scar_result = scar.fit(
        u, method="scar-tm-ou", K=8, max_K=8, maxiter=1)
    assert scar_result.n_params == 3
    assert scar_result.diagnostics["selected_engine"] == "cpp"


def test_fixed_mode_kendall_plugin_counts_correlation_parameters():
    u = _u(T=25)
    model = StochasticStudentCopula(d=3, corr_mode="fixed")

    mle_result = model.fit(u, method="mle", maxiter=2)

    assert mle_result.n_params == 4
    assert mle_result.diagnostics["corr_mode"] == "fixed"
    assert mle_result.diagnostics["corr_n_params"] == 0
    assert mle_result.diagnostics["corr_plugin_n_params"] == 3
    assert mle_result.diagnostics["corr_effective_n_params"] == 3
    assert (
        mle_result.diagnostics["corr_initialization_source"] == "kendall")

    scar = StochasticStudentCopula(d=3, corr_mode="fixed")
    scar_result = scar.fit(
        u,
        method="scar-tm-ou",
        K=8,
        max_K=8,
        adaptive=False,
        maxiter=1,
        maxfun=20,
        smart_init=False,
        alpha0=np.array([1.0, 1.0, 0.8]),
    )

    assert scar_result.n_params == 6
    assert scar_result.diagnostics["corr_mode"] == "fixed"
    assert scar_result.diagnostics["corr_n_params"] == 0
    assert scar_result.diagnostics["corr_plugin_n_params"] == 3
    assert scar_result.diagnostics["corr_effective_n_params"] == 3
    assert (
        scar_result.diagnostics["corr_initialization_source"] == "kendall")


def test_fixed_scar_default_auto_backend_selects_cpp():
    u = _u(T=25)
    model = StochasticStudentCopula(
        d=3,
        R=_R(),
        corr_mode="fixed",
    )

    result = model.fit(
        u,
        method="scar-tm-ou",
        alpha0=np.array([1.0, 1.0, 0.8]),
        K=8,
        max_K=8,
        adaptive=False,
        transition_method="matrix",
        maxiter=1,
        maxfun=20,
        analytical_grad=True,
        smart_init=False,
    )

    assert result.diagnostics["selected_engine"] == "cpp"
    assert result.diagnostics["cpp_evaluations"] > 0
    assert not hasattr(result, "backend")


@pytest.mark.parametrize("backend", ["cpp", "python", "auto"])
def test_fixed_scar_rejects_removed_backend_argument(backend):
    u = _u(T=25)
    model = StochasticStudentCopula(
        d=3,
        R=_R(),
        corr_mode="fixed",
    )

    with pytest.raises(TypeError, match="backend selection was removed"):
        model.fit(u, method="scar-tm-ou", backend=backend)


def test_constructor_rejects_aliases_and_invalid_fixed_base():
    R = _R()
    StochasticStudentCopula(d=3, R=R)
    assert StochasticStudentCopula(d=np.int64(3), R=R).d == 3
    assert StochasticStudentCopula(
        d=3, corr_mode="SHRINKAGE").corr_mode == "shrinkage"

    with pytest.raises(ValueError):
        StochasticStudentCopula(d=1)
    for invalid_d in (True, np.bool_(False), 3.0, 2.5, "3"):
        with pytest.raises(TypeError, match="integer"):
            StochasticStudentCopula(d=invalid_d)
    with pytest.raises(TypeError):
        StochasticStudentCopula(dim=3, R=R)
    with pytest.raises(TypeError):
        StochasticStudentCopula(d=3, corr=R)
    with pytest.raises(TypeError):
        StochasticStudentCopula(d=3, estimate_corr=True)
    with pytest.raises(TypeError):
        StochasticStudentCopula(d=3, R=R, rotate=0)
    with pytest.raises(ValueError):
        StochasticStudentCopula(d=3, corr_base=R)
    with pytest.raises(ValueError, match="corr_mode"):
        StochasticStudentCopula(d=3, corr_mode="invalid")
    for invalid_alpha in (0.0, 1.0, -0.1, 1.1):
        with pytest.raises(ValueError, match="corr_shrinkage_init"):
            StochasticStudentCopula(
                d=3,
                corr_mode="shrinkage",
                corr_shrinkage_init=invalid_alpha,
            )
    with pytest.raises(ValueError, match="limited"):
        StochasticStudentCopula(d=11, corr_mode="cholesky")
    assert StochasticStudentCopula(
        d=11,
        corr_mode="cholesky",
        allow_large_cholesky=True,
    ).d == 11


def test_df_transform_is_smooth_and_invertible():
    model = StochasticStudentCopula(d=2, R=np.eye(2))
    x = np.array([-20.0, -5.0, 0.0, 5.0, 50.0])

    df = model.transform(x)

    assert np.all(df > model._df_offset)
    np.testing.assert_allclose(
        model.inv_transform(df), x, rtol=0.0, atol=1e-6)

    points = np.array([-10.0, 0.0, 10.0])
    step = 1e-6
    finite_diff = (
        model.transform(points + step) - model.transform(points - step)
    ) / (2.0 * step)
    np.testing.assert_allclose(
        model.dtransform(points), finite_diff, rtol=2e-5, atol=1e-10)


def test_shrinkage_correlation_validity_and_limits():
    R0 = _R()

    near_i = make_shrinkage_corr(-40.0, R0)
    near_r0 = make_shrinkage_corr(40.0, R0)

    validate_corr_matrix(near_i)
    validate_corr_matrix(near_r0)
    np.testing.assert_allclose(near_i, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(near_r0, R0, atol=1e-12)


def test_cholesky_correlation_validity_and_pack_roundtrip():
    rng = np.random.default_rng(456)
    params = rng.normal(scale=0.2, size=3)

    R = unpack_cholesky_corr(params, d=3)
    validate_corr_matrix(R)

    with pytest.raises(ValueError):
        unpack_cholesky_corr(np.array([0.1, 0.2]), d=3)

    packed = pack_cholesky_corr(R)
    R_roundtrip = unpack_cholesky_corr(packed, d=3)
    validate_corr_matrix(R_roundtrip)
    np.testing.assert_allclose(R_roundtrip, R, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("corr_mode", ["shrinkage", "cholesky"])
def test_supplied_R_precedes_kendall_initialization(corr_mode):
    u = _u(T=30)
    supplied_R = _R(rho=0.55)
    model = StochasticStudentCopula(
        d=3,
        R=supplied_R,
        corr_mode=corr_mode,
    )

    initial = model._initial_corr_params(u)

    np.testing.assert_allclose(
        model._corr_base, supplied_R, rtol=0.0, atol=1e-12)
    if corr_mode == "shrinkage":
        expected = np.array([np.log(4.0)])
        expected_R = make_shrinkage_corr(expected[0], supplied_R)
    else:
        expected = pack_cholesky_corr(supplied_R)
        expected_R = supplied_R
    np.testing.assert_allclose(initial, expected, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(model.R, expected_R, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("corr_mode", ["shrinkage", "cholesky"])
def test_corr_base_precedes_supplied_R_initialization(corr_mode):
    u = _u(T=30)
    supplied_R = _R(rho=0.55)
    corr_base = _R(rho=0.15)
    model = StochasticStudentCopula(
        d=3,
        R=supplied_R,
        corr_mode=corr_mode,
        corr_base=corr_base,
    )

    initial = model._initial_corr_params(u)

    np.testing.assert_allclose(
        model._corr_base, corr_base, rtol=0.0, atol=1e-12)
    if corr_mode == "shrinkage":
        expected = np.array([np.log(4.0)])
        expected_R = make_shrinkage_corr(expected[0], corr_base)
    else:
        expected = pack_cholesky_corr(corr_base)
        expected_R = corr_base
    np.testing.assert_allclose(initial, expected, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(model.R, expected_R, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("corr_mode", ["shrinkage", "cholesky"])
def test_corr_base_initializes_model_without_supplied_R(corr_mode):
    u = _u(T=30)
    corr_base = _R(rho=0.15)
    model = StochasticStudentCopula(
        d=3,
        corr_mode=corr_mode,
        corr_base=corr_base,
    )

    initial = model._initial_corr_params(u)

    np.testing.assert_allclose(
        model._corr_base, corr_base, rtol=0.0, atol=1e-12)
    if corr_mode == "shrinkage":
        expected = np.array([np.log(4.0)])
        expected_R = make_shrinkage_corr(expected[0], corr_base)
    else:
        expected = pack_cholesky_corr(corr_base)
        expected_R = corr_base
    np.testing.assert_allclose(initial, expected, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(model.R, expected_R, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("corr_mode", ["shrinkage", "cholesky"])
def test_kendall_initialization_is_used_when_no_corr_is_supplied(corr_mode):
    u = _u(T=30)
    model = StochasticStudentCopula(d=3, corr_mode=corr_mode)
    expected_base = model._initial_corr(u)

    initial = model._initial_corr_params(u)

    np.testing.assert_allclose(
        model._corr_base, expected_base, rtol=0.0, atol=1e-12)
    if corr_mode == "shrinkage":
        expected = np.array([np.log(4.0)])
        expected_R = make_shrinkage_corr(expected[0], expected_base)
    else:
        expected = pack_cholesky_corr(expected_base)
        expected_R = expected_base
    np.testing.assert_allclose(initial, expected, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(model.R, expected_R, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize(
    ("corr_mode", "corr_params"),
    [
        ("shrinkage", np.array([0.35])),
        ("cholesky", np.array([0.1, -0.2, 0.3])),
    ],
)
def test_internal_corr_trial_skips_projection_and_uses_one_cholesky(
        corr_mode, corr_params, monkeypatch):
    R0 = _R()
    model = StochasticStudentCopula(
        d=3,
        R=R0,
        corr_mode=corr_mode,
        corr_base=R0 if corr_mode == "shrinkage" else None,
    )
    expected = (
        make_shrinkage_corr(float(corr_params[0]), model._corr_base)
        if corr_mode == "shrinkage"
        else unpack_cholesky_corr(corr_params, model.d)
    )
    original_cholesky = np.linalg.cholesky
    cholesky_calls = 0

    def fail_eigh(*args, **kwargs):
        raise AssertionError("correlation trial must not run SPD projection")

    def counting_cholesky(*args, **kwargs):
        nonlocal cholesky_calls
        cholesky_calls += 1
        return original_cholesky(*args, **kwargs)

    monkeypatch.setattr(np.linalg, "eigh", fail_eigh)
    monkeypatch.setattr(np.linalg, "cholesky", counting_cholesky)

    model._set_corr_from_params(corr_params)

    assert cholesky_calls == 1
    np.testing.assert_allclose(model.R, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize(
    ("corr_mode", "trial_params"),
    [
        ("shrinkage", np.array([-0.4])),
        ("cholesky", np.array([0.1, -0.2, 0.3])),
    ],
)
def test_initial_corr_params_reset_to_parameterization_default(
        corr_mode, trial_params):
    u = _u(T=20)
    R0 = _R()
    model = StochasticStudentCopula(
        d=3,
        R=R0,
        corr_mode=corr_mode,
        corr_base=R0,
    )
    model._set_corr_from_params(trial_params)

    initial = model._initial_corr_params(u)
    expected = (
        np.array([np.log(4.0)])
        if corr_mode == "shrinkage"
        else pack_cholesky_corr(R0)
    )

    np.testing.assert_allclose(initial, expected, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        model.corr_params(), expected, rtol=0.0, atol=1e-12)


def test_mle_shrinkage_fit_smoke_model_and_api():
    u = _u(T=50)

    model = StochasticStudentCopula(d=3, corr_mode="shrinkage")
    result = model.fit(u, method="mle", maxiter=3)

    assert np.isfinite(result.log_likelihood)
    assert result.n_params == 2
    validate_corr_matrix(model.R)
    assert 0.0 < model.corr_alpha() < 1.0
    assert result.diagnostics["corr_mode"] == "shrinkage"
    assert result.diagnostics["corr_n_params"] == 1
    assert result.diagnostics["gradient_mode"] == "analytical_joint"
    assert result.diagnostics["optimizer_gradient"] == "analytical"
    assert result.diagnostics["correlation_gradient"] == "analytical"
    assert result.diagnostics["corr_alpha"] == pytest.approx(
        model.corr_alpha())
    np.testing.assert_allclose(
        result.diagnostics["corr_params_raw"], model.corr_params())
    np.testing.assert_allclose(
        result.diagnostics["corr_matrix"], model.R)

    api_model = StochasticStudentCopula(d=3, corr_mode="shrinkage")
    api_result = fit(api_model, u, method="mle", maxiter=3)
    assert np.isfinite(api_result.log_likelihood)
    assert api_result.n_params == 2
    validate_corr_matrix(api_model.R)


def test_mle_cholesky_reports_all_static_parameters():
    u = _u(T=40)
    model = StochasticStudentCopula(d=3, corr_mode="cholesky")

    result = model.fit(u, method="mle", maxiter=3)

    assert np.isfinite(result.log_likelihood)
    assert result.n_params == 4
    assert result.diagnostics["corr_mode"] == "cholesky"
    assert result.diagnostics["corr_n_params"] == 3
    assert result.diagnostics["gradient_mode"] == "analytical_joint"
    assert result.diagnostics["optimizer_gradient"] == "analytical"
    assert result.diagnostics["correlation_gradient"] == "analytical"
    assert result.diagnostics["corr_alpha"] is None
    np.testing.assert_allclose(
        result.diagnostics["corr_params_raw"], model.corr_params())
    np.testing.assert_allclose(
        result.diagnostics["corr_matrix"], model.R)
    validate_corr_matrix(model.R)

    api_model = StochasticStudentCopula(d=3, corr_mode="cholesky")
    api_result = fit(api_model, u, method="mle", maxiter=3)
    assert np.isfinite(api_result.log_likelihood)
    assert api_result.n_params == 4
    validate_corr_matrix(api_model.R)


def test_mle_correlation_metadata_persistence_roundtrip(tmp_path):
    u = _u(T=30)
    model = StochasticStudentCopula(d=3, corr_mode="shrinkage")
    result = model.fit(u, method="mle", maxiter=2)
    path = tmp_path / "stochastic-student-mle.json"

    model.save(path)
    loaded = StochasticStudentCopula.load(path)
    loaded_result = loaded.fit_result

    assert loaded_result.n_params == result.n_params
    assert loaded_result.diagnostics["corr_mode"] == "shrinkage"
    assert loaded_result.diagnostics["corr_n_params"] == 1
    assert loaded_result.diagnostics["corr_alpha"] == pytest.approx(
        result.diagnostics["corr_alpha"])
    np.testing.assert_allclose(
        loaded_result.diagnostics["corr_params_raw"],
        result.diagnostics["corr_params_raw"],
    )
    np.testing.assert_allclose(
        loaded_result.diagnostics["corr_matrix"],
        result.diagnostics["corr_matrix"],
    )


def test_joint_scar_parameter_count_persistence_roundtrip(tmp_path):
    R = _R()
    model = StochasticStudentCopula(
        d=3,
        R=R,
        corr_mode="shrinkage",
        corr_base=R,
    )
    model.fit_result = LatentResult(
        log_likelihood=12.5,
        method="SCAR-TM-OU",
        copula_name=model.name,
        success=True,
        params=ou_params(1.2, 0.5, 0.8),
        parameter_count=4,
        diagnostics={"corr_mode": "shrinkage", "corr_n_params": 1},
    )
    path = tmp_path / "stochastic-student-scar.json"

    model.save(path)
    loaded = StochasticStudentCopula.load(path)

    assert loaded.fit_result.n_params == 4
    assert loaded.fit_result.parameter_count == 4
    assert loaded.fit_result.params.n_params == 3


@pytest.mark.parametrize(
    ("corr_mode", "corr_params"),
    [
        ("fixed", np.empty(0)),
        ("shrinkage", np.array([0.0])),
        ("cholesky", np.array([0.1, -0.2, 0.3])),
    ],
)
def test_posterior_state_weights_are_normalized(corr_mode, corr_params):
    u = _u(T=24)
    model = StochasticStudentCopula(d=3, corr_mode=corr_mode)
    params = np.concatenate([np.array([1.2, 0.5, 0.8]), corr_params])

    weights = model.posterior_state_weights(
        u, params=params, K=9, adaptive=False)

    assert weights.shape == (len(u), 9)
    assert np.all(np.isfinite(weights))
    assert np.all(weights >= 0.0)
    np.testing.assert_allclose(
        weights.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)
    validate_corr_matrix(model.R)
    if corr_mode == "shrinkage":
        assert model.corr_alpha() == pytest.approx(0.5)


def test_posterior_state_weights_validates_inputs_and_param_length():
    u = _u(T=12)
    model = StochasticStudentCopula(d=3, corr_mode="shrinkage")

    with pytest.raises(ValueError, match="3 or 4"):
        model.posterior_state_weights(
            u, params=np.array([1.2, 0.5, 0.8, 0.0, 1.0]),
            K=8, adaptive=False)
    with pytest.raises(ValueError, match="shape"):
        model.posterior_state_weights(
            u[:, :2], params=np.array([1.2, 0.5, 0.8]),
            K=8, adaptive=False)
    with pytest.raises(ValueError, match="at least two"):
        model.posterior_state_weights(
            u[:1], params=np.array([1.2, 0.5, 0.8]),
            K=8, adaptive=False)
    bad_u = u.copy()
    bad_u[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        model.posterior_state_weights(
            bad_u, params=np.array([1.2, 0.5, 0.8]),
            K=8, adaptive=False)


def test_posterior_state_weights_uses_fit_result_params_by_default():
    u = _u(T=16)
    model = StochasticStudentCopula(d=3, R=_R())
    model.fit_result = SimpleNamespace(params=ou_params(1.2, 0.5, 0.8))

    implicit = model.posterior_state_weights(u, K=8, adaptive=False)
    explicit = model.posterior_state_weights(
        u, params=np.array([1.2, 0.5, 0.8]), K=8, adaptive=False)

    np.testing.assert_allclose(implicit, explicit, rtol=0.0, atol=0.0)


def test_posterior_state_weights_falls_back_to_last_latent_result():
    u = _u(T=16)
    model = StochasticStudentCopula(d=3, R=_R())
    model._last_latent_result = LatentResult(
        log_likelihood=1.0,
        method="SCAR-TM-OU",
        copula_name=model.name,
        success=True,
        params=ou_params(1.2, 0.5, 0.8),
    )
    model.fit_result = MultivariateMLEResult(
        log_likelihood=1.0,
        method="MLE",
        copula_name=model.name,
        success=True,
        copula_param=5.0,
        parameter_count=1,
        n_observations=len(u),
    )

    implicit = model.posterior_state_weights(u, K=8, adaptive=False)
    explicit = model.posterior_state_weights(
        u, params=np.array([1.2, 0.5, 0.8]), K=8, adaptive=False)

    np.testing.assert_allclose(implicit, explicit, rtol=0.0, atol=0.0)


def test_scar_fixed_cpp_likelihood_matches_python_reference():
    u = _u(T=40)
    model = StochasticStudentCopula(d=3, R=_R(), corr_mode="fixed")
    config = AutoTMConfig(
        K=9, grid_range=3.0, adaptive=False, transition_method="matrix")

    cpp_ll, info = _cpp_scar_ou.loglik(
        1.2, 1.0, 0.8, u, model, config)
    py_nll = tm_loglik(
        1.2, 1.0, 0.8, u, model, K=9, grid_range=3.0,
        adaptive=False, transition_method="matrix")

    assert info["engine"] == "cpp"
    assert np.isfinite(cpp_ll)
    np.testing.assert_allclose(cpp_ll, -py_nll, atol=5e-4, rtol=5e-4)


def test_scar_fixed_cpp_gradient_is_finite():
    u = _u(T=30, d=2)
    model = StochasticStudentCopula(d=2, R=_R(d=2), corr_mode="fixed")
    config = AutoTMConfig(
        K=9, grid_range=3.0, adaptive=False, transition_method="matrix")

    value, grad, info = _cpp_scar_ou.neg_loglik_with_grad_info(
        1.1, 0.7, 0.9, u, model, config)

    assert info["engine"] == "cpp"
    assert np.isfinite(value)
    assert grad.shape == (3,)
    assert np.all(np.isfinite(grad))


def test_scar_fixed_cpp_rejects_dimension_mismatch():
    u = _u(T=12, d=2)
    model = StochasticStudentCopula(d=3, R=_R(d=3), corr_mode="fixed")
    config = AutoTMConfig(
        K=9, grid_range=3.0, adaptive=False, transition_method="matrix")

    with pytest.raises(ValueError, match="dimension"):
        _cpp_scar_ou.loglik(1.1, 0.7, 0.9, u, model, config)


@pytest.mark.parametrize(
    ("corr_mode", "corr_n_params"),
    [("shrinkage", 1), ("cholesky", 3)],
)
def test_scar_estimated_corr_modes_use_python_optimizer_with_native_likelihood(
        corr_mode, corr_n_params):
    u = _u(T=25)
    model = StochasticStudentCopula(d=3, corr_mode=corr_mode)

    result = model.fit(
        u,
        method="scar-tm-ou",
        alpha0=np.array([1.0, 1.0, 0.8]),
        K=8,
        max_K=8,
        adaptive=False,
        transition_method="matrix",
        maxiter=1,
        maxfun=20,
        analytical_grad=True,
        smart_init=False,
    )

    assert np.isfinite(result.log_likelihood)
    assert result.n_params == 3 + corr_n_params
    assert result.params.values.shape == (3,)
    assert result.diagnostics["selected_engine"] == "cpp"
    assert result.diagnostics["joint_static"] is True
    assert result.diagnostics["joint_optimizer"] == "python-lbfgsb"
    assert (
        result.diagnostics["correlation_parameterization_engine"] == "python")
    assert result.diagnostics["analytical_grad_requested"] is True
    assert result.diagnostics["analytical_grad_used"] is True
    assert result.diagnostics["optimizer_gradient"] == "analytical"
    assert result.diagnostics["gradient_kind"] == "analytical"
    assert result.diagnostics["setup_derivative"] == "analytical"
    assert result.diagnostics["filter_derivative"] == "analytical"
    assert result.diagnostics["ou_gradient"] == "analytical"
    assert result.diagnostics["hybrid_gradient_evaluations"] > 0
    assert result.diagnostics["correlation_gradient"] == "analytical"
    assert result.diagnostics["cpp_correlation_derivatives"] is True
    assert result.diagnostics["joint_gradient"] == "analytical"
    assert result.diagnostics["correlation_fd_scheme"] == "none"
    assert result.diagnostics["correlation_fd_evaluations"] == 0
    assert (
        result.diagnostics[
            "native_correlation_gradient_evaluations"] > 0)
    assert result.diagnostics["corr_mode"] == corr_mode
    assert result.diagnostics["corr_n_params"] == corr_n_params
    assert result.diagnostics["corr_params_raw"].shape == (corr_n_params,)
    assert result.diagnostics["cpp_evaluations"] > 0
    np.testing.assert_allclose(
        result.diagnostics["corr_matrix"], model.R)
    validate_corr_matrix(model.R)


def test_spectral_cholesky_fit_uses_native_correlation_gradient():
    u = _u(T=30, d=3, seed=20260706)
    model = StochasticStudentCopula(
        d=3, R=_R(), corr_mode="cholesky")

    result = model.fit(
        u,
        method="scar-tm-ou",
        alpha0=np.array([2.0, 0.5, 0.8]),
        transition_method="spectral",
        spectral_basis_order=16,
        spectral_quad_order=48,
        maxiter=1,
        maxfun=20,
        analytical_grad=True,
        smart_init=False,
    )

    assert np.isfinite(result.log_likelihood)
    assert result.diagnostics["last_backend"] == "spectral"
    assert result.diagnostics["correlation_gradient"] == "analytical"
    assert result.diagnostics["joint_gradient"] == "analytical"
    assert result.diagnostics["correlation_fd_evaluations"] == 0
    assert (
        result.diagnostics[
            "native_correlation_gradient_evaluations"] > 0)


@pytest.mark.parametrize(
    ("corr_mode", "corr_n_params"),
    [
        ("shrinkage", 1),
        ("cholesky", 3),
    ],
)
def test_joint_hybrid_jacobian_uses_one_plus_n_corr_evaluations(
        corr_mode, corr_n_params, monkeypatch):
    u = _u(T=12)
    model = StochasticStudentCopula(d=3, R=_R(), corr_mode=corr_mode)
    alpha0 = np.array([2.0, -0.5, 1.5])
    calls = {"gradient": 0, "objective": 0}
    captured = {}

    def value_for(kappa, mu, nu, copula):
        ou = np.array([kappa, mu, nu], dtype=np.float64)
        corr = np.asarray(copula.corr_params(), dtype=np.float64)
        return float(np.sum((ou - np.array([1.0, 0.25, 0.75])) ** 2)
                     + np.sum(corr ** 2) + 3.0)

    def info_for(kappa):
        return {
            "backend": "spectral",
            "transition_method": "auto",
            "kappa_dt": float(kappa) / (len(u) - 1),
            "n_obs": len(u),
            "basis_order": 32,
        }

    def fake_gradient(kappa, mu, nu, u_arg, copula, config):
        calls["gradient"] += 1
        grad = 2.0 * (
            np.array([kappa, mu, nu], dtype=np.float64)
            - np.array([1.0, 0.25, 0.75])
        )
        return value_for(kappa, mu, nu, copula), grad, info_for(kappa)

    def fake_objective(kappa, mu, nu, u_arg, copula, config):
        calls["objective"] += 1
        return value_for(kappa, mu, nu, copula), info_for(kappa)

    def fake_minimize(fun, x0, *, method, jac, bounds, options):
        assert method == "L-BFGS-B"
        assert jac is True
        value, gradient = fun(np.asarray(x0, dtype=np.float64))
        captured["gradient"] = gradient.copy()
        captured["x0"] = np.asarray(x0, dtype=np.float64).copy()
        return SimpleNamespace(
            x=np.asarray(x0, dtype=np.float64).copy(),
            fun=float(value),
            success=True,
            message="test optimizer",
            nfev=1,
            jac=gradient,
        )

    monkeypatch.setattr(
        _cpp_scar_ou,
        "neg_loglik_with_grad_and_corr_info",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            _cpp_scar_ou.CppUnsupported("test fallback")),
    )
    monkeypatch.setattr(
        _cpp_scar_ou, "neg_loglik_with_grad_info", fake_gradient)
    monkeypatch.setattr(_cpp_scar_ou, "neg_loglik_info", fake_objective)
    monkeypatch.setattr(scar_tm, "minimize", fake_minimize)

    result = scar_tm.SCARTMStrategy(
        analytical_grad=True,
        smart_init=False,
    ).fit(
        model,
        u,
        alpha0=alpha0,
        eps=1e-6,
    )

    scale = np.maximum(
        np.abs(np.concatenate([alpha0, model.corr_params()])), 1.0)
    expected_physical = np.concatenate([
        2.0 * (alpha0 - np.array([1.0, 0.25, 0.75])),
        2.0 * model.corr_params() + 1e-6,
    ])

    assert calls == {
        "gradient": 2,
        "objective": 2 * corr_n_params,
    }
    np.testing.assert_allclose(
        captured["gradient"], expected_physical * scale,
        rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(
        captured["x0"], np.concatenate([alpha0, model.corr_params()]) / scale)
    assert result.diagnostics["objective_evaluations"] == (
        2 * (1 + corr_n_params))
    assert result.diagnostics["hybrid_gradient_evaluations"] == 2
    assert result.diagnostics["correlation_fd_evaluations"] == (
        2 * corr_n_params)
    assert result.diagnostics["analytical_grad_used"] is True
    assert result.diagnostics["final_validation_passed"] is False
    assert (
        "projected gradient exceeds validation tolerance"
        in result.diagnostics["final_validation_reasons"]
    )


@pytest.mark.parametrize(
    ("corr_mode", "corr_n_params"),
    [
        ("shrinkage", 1),
        ("cholesky", 3),
    ],
)
def test_joint_scar_acceptance_with_task_grid(corr_mode, corr_n_params):
    """Acceptance smoke test from stochasticopula_task.txt."""
    u = _u(T=80)
    model = StochasticStudentCopula(d=3, corr_mode=corr_mode)

    result = model.fit(
        u,
        method="scar-tm-ou",
        K=16,
        max_K=16,
        adaptive=False,
        transition_method="matrix",
        maxiter=2,
        maxfun=40,
        analytical_grad=False,
    )

    assert np.isfinite(result.log_likelihood)
    assert result.n_params == 3 + corr_n_params
    assert result.diagnostics["selected_engine"] == "cpp"
    assert result.diagnostics["analytical_grad_used"] is False
    assert result.diagnostics["corr_mode"] == corr_mode
    assert result.diagnostics["corr_n_params"] == corr_n_params
    assert result.diagnostics["corr_params_raw"].shape == (corr_n_params,)
    validate_corr_matrix(model.R)
    np.testing.assert_allclose(
        result.diagnostics["corr_matrix"], model.R)
    assert "final_validation_passed" in result.diagnostics
    np.testing.assert_allclose(
        result.diagnostics["corr_params_raw"], model.corr_params())
    if corr_mode == "shrinkage":
        assert 0.0 < result.diagnostics["corr_alpha"] < 1.0
    else:
        assert result.diagnostics["corr_alpha"] is None


def test_joint_cpp_scar_reuses_ppf_cache_across_correlation_trials(monkeypatch):
    u = _u(T=30)
    model = StochasticStudentCopula(d=3, corr_mode="shrinkage")
    builds = 0
    ppf_table_cls = stochastic_student._PPFTable

    def counting_ppf_table(values):
        nonlocal builds
        builds += 1
        return ppf_table_cls(values)

    monkeypatch.setattr(
        stochastic_student, "_PPFTable", counting_ppf_table)

    result = model.fit(
        u,
        method="scar-tm-ou",
        alpha0=np.array([1.0, 1.0, 0.8]),
        K=8,
        max_K=8,
        adaptive=False,
        transition_method="matrix",
        maxiter=1,
        maxfun=20,
        analytical_grad=False,
        smart_init=False,
    )

    assert result.nfev > 1
    assert result.diagnostics["selected_engine"] == "cpp"
    assert result.diagnostics["cpp_evaluations"] > 1
    assert builds == 1


def test_unsupported_gas_joint_corr_path_is_explicit():
    u = _u(T=30)

    with pytest.raises(NotImplementedError, match="MLE and SCAR-TM-OU only"):
        StochasticStudentCopula(d=3, corr_mode="shrinkage").fit(
            u, method="gas")


@pytest.mark.parametrize(
    "factory",
    [
        lambda: StochasticStudentCopula(d=3),
        lambda: StochasticStudentCopula(d=3, corr_mode="shrinkage"),
        lambda: StochasticStudentCopula(d=3, corr_mode="cholesky"),
    ],
)
@pytest.mark.parametrize("method", ["gas", "scar-p-ou", "scar-m-ou"])
def test_data_estimated_corr_is_limited_to_mle_and_scar_tm_ou(
        factory, method):
    u = _u(T=20)
    kwargs = {
        "maxiter": 1,
        "maxfun": 5,
        "smart_init": False,
        "alpha0": np.array([1.0, 0.5, 0.8]),
        "gamma0": np.array([0.1, 0.05, 0.5]),
        "n_tr": 4,
        "seed": 7,
    }

    with pytest.raises(NotImplementedError, match="MLE and SCAR-TM-OU only"):
        factory().fit(u, method=method, **kwargs)

    with pytest.raises(NotImplementedError, match="MLE and SCAR-TM-OU only"):
        fit(factory(), u, method=method, **kwargs)
