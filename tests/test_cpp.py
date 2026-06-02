import numpy as np
import pytest
from pathlib import Path

from pyscarcopula import (
    BivariateGaussianCopula,
    ClaytonCopula,
    FrankCopula,
    GumbelCopula,
    JoeCopula,
)
from pyscarcopula._utils import pobs
from pyscarcopula._types import LatentResult, ou_params
from pyscarcopula.api import fit
from pyscarcopula.copula.independent import IndependentCopula
from pyscarcopula.numerical.auto_tm import (
    AutoTMConfig,
    auto_loglik,
    auto_loglik_with_info,
    auto_neg_loglik_with_grad,
)
from pyscarcopula.numerical.hermite_tm import (
    hermite_loglik,
    hermite_loglik_with_grad,
)
from pyscarcopula.numerical.tm_functions import (
    tm_forward_mixture_h,
    tm_forward_predictive_mean,
)
from pyscarcopula.numerical.predictive_tm import tm_state_distribution
from pyscarcopula.numerical import _cpp_scar_ou
from pyscarcopula.strategy.scar_tm import SCARTMStrategy
from pyscarcopula.vine._pair_copula import PairCopula
from pyscarcopula.vine._rvine_edges import _edge_h, _edge_h_inverse


pytestmark = pytest.mark.skipif(
    not _cpp_scar_ou.available(),
    reason="pyscarcopula C++ extension is not available",
)


def _data(seed=20260611, n=36):
    return np.random.default_rng(seed).uniform(0.08, 0.92, size=(n, 2))


def _basic_api_data():
    path = Path(__file__).resolve().parents[1] / "data" / "crypto_prices.csv"
    prices = np.genfromtxt(
        path,
        delimiter=";",
        skip_header=1,
        usecols=(1, 2),
        dtype=np.float64,
    )
    returns = np.log(prices[1:] / prices[:-1])
    return pobs(returns)


def _result(alpha=(0.9, 0.1, 1.1), copula_name="Clayton copula"):
    return LatentResult(
        log_likelihood=0.0,
        method="SCAR-TM-OU",
        copula_name=copula_name,
        success=True,
        params=ou_params(*alpha),
        K=38,
        grid_range=3.5,
        pts_per_sigma=4,
        transition_method="matrix",
        max_K=None,
        gh_order=5,
        backend="cpp",
    )


def test_cpp_loglik_matches_python_auto_spectral():
    u = _data()
    copula = ClaytonCopula(rotate=0, transform_type="softplus")
    cfg = AutoTMConfig(
        transition_method="auto",
        small_kdt=1e-3,
        basis_order=12,
        quad_order=40,
        K=38,
        grid_range=3.5,
        adaptive=False,
        max_K=None,
    )
    alpha = (1.1, 0.05, 0.9)

    got, info = _cpp_scar_ou.loglik(*alpha, u, copula, cfg)
    ref = auto_loglik(*alpha, u, copula, cfg)

    assert info["engine"] == "cpp"
    assert info["backend"] == "spectral"
    np.testing.assert_allclose(got, ref, rtol=2e-11, atol=2e-11)


def test_cpp_auto_loglik_falls_back_from_failed_spectral_to_matrix():
    u = np.random.default_rng(1).uniform(0.001, 0.999, size=(50, 2))
    copula = BivariateGaussianCopula()
    cfg = AutoTMConfig(
        transition_method="auto",
        small_kdt=1e-9,
        basis_order=2,
        quad_order=2,
        K=30,
        adaptive=False,
        max_K=None,
    )
    alpha = (0.1, 0.0, 10.0)

    got, info = _cpp_scar_ou.loglik(*alpha, u, copula, cfg)
    ref, ref_info = auto_loglik_with_info(*alpha, u, copula, cfg)

    assert info["backend"] == "matrix"
    assert info["fallback_from"] == "spectral"
    assert info["fallback_chain"] == ["spectral"]
    assert ref_info["backend"] == "matrix"
    assert ref_info["fallback_from"] == "spectral"
    assert ref_info["fallback_chain"] == ["spectral"]
    np.testing.assert_allclose(got, ref, rtol=2e-11, atol=2e-11)


def test_cpp_auto_gradient_falls_back_from_failed_spectral_to_matrix():
    u = np.random.default_rng(1).uniform(0.001, 0.999, size=(50, 2))
    copula = BivariateGaussianCopula()
    cfg = AutoTMConfig(
        transition_method="auto",
        small_kdt=1e-9,
        basis_order=2,
        quad_order=2,
        K=30,
        adaptive=False,
        max_K=None,
    )
    alpha = (0.1, 0.0, 10.0)

    got_val, got_grad, info = _cpp_scar_ou.neg_loglik_with_grad_info(
        *alpha, u, copula, cfg)
    ref_val, ref_grad = auto_neg_loglik_with_grad(*alpha, u, copula, cfg)

    assert info["backend"] == "matrix"
    assert info["fallback_from"] == "spectral"
    assert info["fallback_chain"] == ["spectral"]
    np.testing.assert_allclose(got_val, ref_val, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(got_grad, ref_grad, rtol=2e-8, atol=2e-8)


def test_cpp_spectral_matches_python_with_large_quad_order():
    u = _data(seed=20260621, n=80)
    copula = ClaytonCopula(rotate=0, transform_type="softplus")
    cfg = AutoTMConfig(
        transition_method="spectral",
        basis_order=128,
        quad_order=384,
    )
    alpha = (3.0, 0.1, 1.2)

    got, info = _cpp_scar_ou.loglik(*alpha, u, copula, cfg)
    ref = hermite_loglik(
        *alpha, u, copula,
        basis_order=cfg.basis_order,
        quad_order=cfg.quad_order,
    )

    assert info["backend"] == "spectral"
    np.testing.assert_allclose(got, ref, rtol=2e-10, atol=2e-10)


def test_cpp_spectral_gradient_matches_python_with_large_quad_order():
    u = _data(seed=20260622, n=80)
    copula = ClaytonCopula(rotate=0, transform_type="softplus")
    cfg = AutoTMConfig(
        transition_method="spectral",
        basis_order=128,
        quad_order=384,
    )
    alpha = (3.0, 0.1, 1.2)

    got_val, got_grad = _cpp_scar_ou.neg_loglik_with_grad(
        *alpha, u, copula, cfg)
    ref_val, ref_grad = hermite_loglik_with_grad(
        *alpha, u, copula,
        basis_order=cfg.basis_order,
        quad_order=cfg.quad_order,
    )

    np.testing.assert_allclose(got_val, ref_val, rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(got_grad, ref_grad, rtol=2e-8, atol=2e-8)


@pytest.mark.parametrize("transition_method", ["auto", "spectral", "local", "matrix"])
@pytest.mark.parametrize(
    "copula",
    [
        ClaytonCopula(rotate=0, transform_type="softplus"),
        GumbelCopula(rotate=180, transform_type="softplus"),
        FrankCopula(transform_type="softplus"),
        JoeCopula(rotate=90, transform_type="softplus"),
        BivariateGaussianCopula(),
    ],
)
def test_cpp_gradient_matches_python(transition_method, copula):
    u = _data(seed=20260620, n=32)
    cfg = AutoTMConfig(
        transition_method=transition_method,
        small_kdt=1e-3,
        basis_order=12,
        quad_order=40,
        K=38,
        grid_range=3.5,
        grid_method="dense" if transition_method == "matrix" else "auto",
        adaptive=False,
        max_K=None,
    )
    alpha = (1.2, 0.15, 0.8)

    got_val, got_grad = _cpp_scar_ou.neg_loglik_with_grad(
        *alpha, u, copula, cfg)
    ref_val, ref_grad = auto_neg_loglik_with_grad(*alpha, u, copula, cfg)

    np.testing.assert_allclose(got_val, ref_val, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(got_grad, ref_grad, rtol=2e-8, atol=2e-8)


@pytest.mark.data
def test_cpp_fit_matches_python_on_basic_api_example():
    u = _basic_api_data()
    copula = GumbelCopula(rotate=180)

    py_result = fit(copula, u, method="scar-tm-ou", backend="python")
    cpp_result = fit(copula, u, method="scar-tm-ou", backend="cpp")

    assert cpp_result.success
    np.testing.assert_allclose(
        cpp_result.log_likelihood,
        py_result.log_likelihood,
        rtol=0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        [
            cpp_result.params.kappa,
            cpp_result.params.mu,
            cpp_result.params.nu,
        ],
        [
            py_result.params.kappa,
            py_result.params.mu,
            py_result.params.nu,
        ],
        rtol=1e-4,
        atol=2e-3,
    )
    assert cpp_result.nfev == py_result.nfev


@pytest.mark.parametrize(
    "copula",
    [
        ClaytonCopula(rotate=0, transform_type="softplus"),
        GumbelCopula(rotate=180, transform_type="softplus"),
        FrankCopula(transform_type="softplus"),
        JoeCopula(rotate=90, transform_type="softplus"),
        BivariateGaussianCopula(),
    ],
)
def test_cpp_fit_matches_python_for_supported_families(copula):
    u = _data(seed=20260629, n=24)
    common = dict(
        method="scar-tm-ou",
        transition_method="matrix",
        K=30,
        adaptive=False,
        max_K=None,
        smart_init=False,
        alpha0=np.array([0.9, 0.1, 1.0], dtype=np.float64),
        maxiter=8,
        maxfun=24,
    )

    py_result = fit(copula, u, backend="python", **common)
    cpp_result = fit(copula, u, backend="cpp", **common)

    assert py_result.success == cpp_result.success
    np.testing.assert_allclose(
        cpp_result.log_likelihood,
        py_result.log_likelihood,
        rtol=0.0,
        atol=5e-6,
    )
    np.testing.assert_allclose(
        cpp_result.params.values,
        py_result.params.values,
        rtol=1e-4,
        atol=1e-4,
    )


@pytest.mark.parametrize(
    "copula",
    [
        ClaytonCopula(rotate=0, transform_type="softplus"),
        GumbelCopula(rotate=180, transform_type="softplus"),
        FrankCopula(transform_type="softplus"),
        JoeCopula(rotate=90, transform_type="softplus"),
        BivariateGaussianCopula(),
    ],
)
def test_cpp_loglik_matches_python_matrix_for_supported_families(copula):
    u = _data(seed=20260619, n=26)
    cfg = AutoTMConfig(
        transition_method="matrix",
        K=34,
        grid_range=3.0,
        adaptive=False,
        max_K=None,
    )
    alpha = (0.85, 0.05, 0.95)

    got, info = _cpp_scar_ou.loglik(*alpha, u, copula, cfg)
    ref = auto_loglik(*alpha, u, copula, cfg)

    assert info["engine"] == "cpp"
    assert info["backend"] == "matrix"
    np.testing.assert_allclose(got, ref, rtol=2e-7, atol=2e-7)


def test_cpp_strategy_forward_functions_match_python_matrix():
    u = _data(seed=20260612, n=30)
    copula = ClaytonCopula(rotate=180, transform_type="softplus")
    alpha = (0.9, 0.1, 1.1)
    strategy = SCARTMStrategy(
        backend="cpp",
        transition_method="matrix",
        K=38,
        grid_range=3.5,
        adaptive=False,
        max_K=None,
    )
    result = _result(alpha)

    pm = strategy.predictive_mean(copula, u, result)
    mh = strategy.mixture_h(copula, u, result)

    kwargs = {
        "K": strategy.K,
        "grid_range": strategy.grid_range,
        "grid_method": strategy.grid_method,
        "adaptive": strategy.adaptive,
        "pts_per_sigma": strategy.pts_per_sigma,
        "transition_method": strategy.transition_method,
        "max_K": strategy.max_K,
        "r_gh": strategy.r_gh,
        "gh_order": strategy.gh_order,
    }
    np.testing.assert_allclose(
        pm, tm_forward_predictive_mean(*alpha, u, copula, **kwargs),
        rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(
        mh, tm_forward_mixture_h(*alpha, u, copula, **kwargs),
        rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("transition_method", ["auto", "local"])
def test_cpp_forward_functions_match_python_for_selected_backend(transition_method):
    u = _data(seed=20260625, n=28)
    copula = GumbelCopula(rotate=180, transform_type="softplus")
    alpha = (0.07, -0.2, 0.8)
    cfg = AutoTMConfig(
        transition_method=transition_method,
        small_kdt=1e-3,
        K=34,
        grid_range=3.0,
        adaptive=False,
        max_K=None,
        gh_order=5,
    )
    kwargs = {
        "K": cfg.K,
        "grid_range": cfg.grid_range,
        "grid_method": cfg.grid_method,
        "adaptive": cfg.adaptive,
        "pts_per_sigma": cfg.pts_per_sigma,
        "transition_method": cfg.transition_method,
        "max_K": cfg.max_K,
        "r_gh": cfg.r_gh,
        "gh_order": cfg.gh_order,
    }

    np.testing.assert_allclose(
        _cpp_scar_ou.predictive_mean(*alpha, u, copula, cfg),
        tm_forward_predictive_mean(*alpha, u, copula, **kwargs),
        rtol=1e-8,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        _cpp_scar_ou.mixture_h(*alpha, u, copula, cfg),
        tm_forward_mixture_h(*alpha, u, copula, **kwargs),
        rtol=1e-8,
        atol=1e-8,
    )


@pytest.mark.parametrize("transition_method", ["matrix", "local"])
@pytest.mark.parametrize("horizon", ["current", "next"])
def test_cpp_state_distribution_matches_python(horizon, transition_method):
    u = _data(seed=20260616, n=30)
    copula = ClaytonCopula(rotate=90, transform_type="softplus")
    alpha = (0.9, -0.1, 1.0)
    cfg = AutoTMConfig(
        transition_method=transition_method,
        K=36,
        grid_range=3.0,
        adaptive=False,
        max_K=None,
    )

    z_cpp, prob_cpp = _cpp_scar_ou.state_distribution(
        *alpha, u, copula, cfg, horizon=horizon)
    z_py, prob_py = tm_state_distribution(
        *alpha, u, copula,
        K=cfg.K,
        grid_range=cfg.grid_range,
        grid_method=cfg.grid_method,
        adaptive=cfg.adaptive,
        pts_per_sigma=cfg.pts_per_sigma,
        transition_method=cfg.transition_method,
        max_K=cfg.max_K,
        r_gh=cfg.r_gh,
        gh_order=cfg.gh_order,
        horizon=horizon,
    )

    np.testing.assert_allclose(z_cpp, z_py, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(prob_cpp, prob_py, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(np.sum(prob_cpp), 1.0, rtol=1e-12, atol=1e-12)


def test_cpp_strategy_predictive_state_uses_cache():
    u = _data(seed=20260617, n=24)
    copula = ClaytonCopula(rotate=0, transform_type="softplus")
    result = _result(alpha=(0.8, 0.1, 1.0))
    strategy = SCARTMStrategy(
        backend="cpp",
        transition_method="matrix",
        K=result.K,
        grid_range=result.grid_range,
        adaptive=False,
        max_K=None,
    )
    cache = {}

    first = strategy.predictive_state(
        copula, u, result, horizon="next", state_cache=cache, cache_key="edge")
    second = strategy.predictive_state(
        copula, u, result, horizon="next", state_cache=cache, cache_key="edge")

    assert "edge" in cache
    np.testing.assert_allclose(first.z_grid, second.z_grid)
    np.testing.assert_allclose(first.prob, second.prob)


def test_cpp_strategy_mixture_h_populates_state_cache():
    u = _data(seed=20260618, n=24)
    copula = ClaytonCopula(rotate=180, transform_type="softplus")
    result = _result(alpha=(0.8, 0.1, 1.0))
    strategy = SCARTMStrategy(
        backend="cpp",
        transition_method="matrix",
        K=result.K,
        grid_range=result.grid_range,
        adaptive=False,
        max_K=None,
    )
    cache = {}

    h_mix = strategy.mixture_h(
        copula,
        u,
        result,
        state_cache=cache,
        current_cache_key="current",
        next_cache_key="next",
    )

    assert h_mix.shape == (len(u),)
    assert set(cache) == {"current", "next"}
    assert len(cache["current"][0]) == result.K
    assert len(cache["next"][0]) == result.K
    np.testing.assert_allclose(np.sum(cache["current"][1]), 1.0)
    np.testing.assert_allclose(np.sum(cache["next"][1]), 1.0)


@pytest.mark.parametrize(
    "copula,param",
    [
        (ClaytonCopula(rotate=0, transform_type="softplus"), 1.7),
        (ClaytonCopula(rotate=90, transform_type="xtanh"), 1.7),
        (GumbelCopula(rotate=180, transform_type="softplus"), 2.1),
        (GumbelCopula(rotate=270, transform_type="xtanh"), 2.1),
        (FrankCopula(transform_type="softplus"), 4.0),
        (JoeCopula(rotate=90, transform_type="softplus"), 2.3),
        (JoeCopula(rotate=180, transform_type="xtanh"), 2.3),
        (BivariateGaussianCopula(), 0.45),
    ],
)
def test_cpp_copula_h_and_inverse_match_python(copula, param):
    u = np.array([0.18, 0.42, 0.77], dtype=np.float64)
    given = np.array([0.23, 0.61, 0.84], dtype=np.float64)
    q = np.array([0.21, 0.55, 0.79], dtype=np.float64)
    r = np.full(len(u), param, dtype=np.float64)

    np.testing.assert_allclose(
        _cpp_scar_ou.copula_h(copula, u, given, r),
        copula.h(u, given, r),
        rtol=2e-6,
        atol=2e-6,
    )

    inv_cpp = _cpp_scar_ou.copula_h_inverse(copula, q, given, r)
    np.testing.assert_allclose(
        inv_cpp,
        copula.h_inverse(q, given, r),
        rtol=3e-6,
        atol=3e-6,
    )
    np.testing.assert_allclose(
        copula.h(inv_cpp, given, r),
        q,
        rtol=3e-6,
        atol=3e-6,
    )


def test_cpp_independent_copula_h_and_inverse_match_python():
    copula = IndependentCopula()
    u = np.array([0.18, 0.42, 0.77], dtype=np.float64)
    given = np.array([0.23, 0.61, 0.84], dtype=np.float64)
    q = np.array([0.21, 0.55, 0.79], dtype=np.float64)
    r = np.zeros(len(u), dtype=np.float64)

    np.testing.assert_allclose(
        _cpp_scar_ou.copula_h(copula, u, given, r),
        copula.h(u, given, r),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        _cpp_scar_ou.copula_h_inverse(copula, q, given, r),
        copula.h_inverse(q, given, r),
        rtol=0.0,
        atol=0.0,
    )


def test_vine_edge_can_route_point_h_and_inverse_to_cpp_backend():
    copula = GumbelCopula(rotate=180, transform_type="softplus")
    edge = PairCopula(copula=copula, param=2.1)
    u = np.array([0.2, 0.4, 0.7], dtype=np.float64)
    given = np.array([0.3, 0.6, 0.8], dtype=np.float64)
    q = np.array([0.25, 0.5, 0.75], dtype=np.float64)
    cfg = {"backend": "cpp"}

    np.testing.assert_allclose(
        _edge_h(edge, u, given, config=cfg),
        copula.h(u, given, np.full(len(u), edge.param)),
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        _edge_h_inverse(edge, q, given, config=cfg),
        copula.h_inverse(q, given, np.full(len(q), edge.param)),
        rtol=3e-6,
        atol=3e-6,
    )


def test_cpp_auto_falls_back_for_unsupported_copula():
    u = _data(seed=20260613, n=16)
    copula = GumbelCopula(rotate=180, transform_type="xtanh")
    alpha = np.array([0.8, 0.2, 1.0])

    strategy = SCARTMStrategy(
        backend="auto",
        transition_method="matrix",
        K=24,
        adaptive=False,
        max_K=None,
        analytical_grad=False,
    )
    python_strategy = SCARTMStrategy(
        backend="python",
        transition_method="matrix",
        K=24,
        adaptive=False,
        max_K=None,
        analytical_grad=False,
    )

    assert not strategy._uses_cpp(copula)
    assert strategy.objective(copula, u, alpha) == pytest.approx(
        python_strategy.objective(copula, u, alpha))


def test_cpp_explicit_unsupported_copula_fails_clearly():
    u = _data(seed=20260614, n=8)
    copula = GumbelCopula(rotate=0, transform_type="xtanh")
    strategy = SCARTMStrategy(backend="cpp", transition_method="matrix")

    with pytest.raises(_cpp_scar_ou.CppUnsupported, match="Gumbel"):
        strategy.log_likelihood(copula, u, _result(copula_name=copula.name))


@pytest.mark.parametrize(
    "alpha",
    [
        (-1.0, 0.0, 1.0),
        (1.0, np.nan, 1.0),
        (1.0, 0.0, np.inf),
    ],
)
def test_cpp_invalid_ou_parameter_raises_non_ok_status(alpha):
    u = _data(seed=20260623, n=8)
    copula = ClaytonCopula(rotate=0, transform_type="softplus")
    cfg = AutoTMConfig(
        transition_method="matrix",
        K=16,
        adaptive=False,
        max_K=None,
    )

    with pytest.raises(_cpp_scar_ou.CppError, match="invalid_parameter"):
        _cpp_scar_ou.loglik(*alpha, u, copula, cfg)


def test_cpp_invalid_numerical_config_raises_non_ok_status():
    u = _data(seed=20260624, n=8)
    copula = ClaytonCopula(rotate=0, transform_type="softplus")
    cfg = AutoTMConfig(
        transition_method="matrix",
        K=16,
        grid_range=np.nan,
        adaptive=False,
        max_K=None,
    )

    with pytest.raises(_cpp_scar_ou.CppError, match="invalid_parameter"):
        _cpp_scar_ou.loglik(1.0, 0.0, 1.0, u, copula, cfg)


def test_cpp_rejects_non_finite_observations_before_kernel_call():
    u = _data(seed=20260624, n=8)
    u[3, 0] = np.nan
    copula = ClaytonCopula(rotate=0, transform_type="softplus")
    cfg = AutoTMConfig(
        transition_method="matrix",
        K=16,
        adaptive=False,
        max_K=None,
    )

    with pytest.raises(ValueError, match="finite"):
        _cpp_scar_ou.loglik(1.0, 0.0, 1.0, u, copula, cfg)


def test_cpp_rejects_invalid_observation_shape_before_kernel_call():
    u = np.array([0.2, 0.4, 0.6], dtype=np.float64)
    copula = ClaytonCopula(rotate=0, transform_type="softplus")
    cfg = AutoTMConfig(
        transition_method="matrix",
        K=16,
        adaptive=False,
        max_K=None,
    )

    with pytest.raises(ValueError, match="shape"):
        _cpp_scar_ou.loglik(1.0, 0.0, 1.0, u, copula, cfg)


@pytest.mark.parametrize(
    "call",
    [
        lambda alpha, u, copula, cfg: _cpp_scar_ou.loglik(
            *alpha, u[:1], copula, cfg),
        lambda alpha, u, copula, cfg: _cpp_scar_ou.neg_loglik_with_grad(
            *alpha, u[:1], copula, cfg),
        lambda alpha, u, copula, cfg: _cpp_scar_ou.predictive_mean(
            *alpha, u, copula, cfg),
        lambda alpha, u, copula, cfg: _cpp_scar_ou.mixture_h(
            *alpha, u, copula, cfg),
        lambda alpha, u, copula, cfg: _cpp_scar_ou.state_distribution(
            *alpha, u[:1], copula, cfg),
    ],
)
def test_cpp_invalid_size_status_reaches_python_without_crashing(call):
    u = _data(seed=20260626, n=8)
    copula = ClaytonCopula(rotate=0, transform_type="softplus")
    cfg = AutoTMConfig(
        transition_method="matrix",
        K=1,
        adaptive=False,
        max_K=None,
    )

    with pytest.raises(_cpp_scar_ou.CppError, match="invalid_size"):
        call((1.0, 0.0, 1.0), u, copula, cfg)


@pytest.mark.parametrize(
    "entrypoint",
    [_cpp_scar_ou.predictive_mean, _cpp_scar_ou.mixture_h],
)
def test_cpp_spectral_forward_entrypoints_fail_clearly(entrypoint):
    u = _data(seed=20260627, n=8)
    copula = ClaytonCopula(rotate=0, transform_type="softplus")
    cfg = AutoTMConfig(transition_method="spectral")

    with pytest.raises(_cpp_scar_ou.CppUnsupported, match="spectral"):
        entrypoint(1.0, 0.0, 1.0, u, copula, cfg)


def test_cpp_spectral_state_distribution_fails_clearly():
    u = _data(seed=20260628, n=8)
    copula = ClaytonCopula(rotate=0, transform_type="softplus")
    cfg = AutoTMConfig(transition_method="spectral")

    with pytest.raises(_cpp_scar_ou.CppUnsupported, match="spectral"):
        _cpp_scar_ou.state_distribution(1.0, 0.0, 1.0, u, copula, cfg)


@pytest.mark.parametrize("entrypoint", [
    _cpp_scar_ou.copula_h,
    _cpp_scar_ou.copula_h_inverse,
])
def test_cpp_copula_ops_reject_non_finite_vector_inputs(entrypoint):
    copula = ClaytonCopula(rotate=0, transform_type="softplus")
    first = np.array([0.18, 0.42, 0.77], dtype=np.float64)
    given = np.array([0.23, 0.61, 0.84], dtype=np.float64)
    r = np.array([1.5, np.nan, 1.5], dtype=np.float64)

    with pytest.raises(ValueError, match="finite"):
        entrypoint(copula, first, given, r)


def test_cpp_fit_path_uses_cpp_objective_with_default_gradient_flag(monkeypatch):
    u = _data(seed=20260615, n=10)
    copula = ClaytonCopula(rotate=0, transform_type="softplus")
    calls = []

    def fake_neg_loglik_with_grad_info(kappa, mu, nu, u_arg, copula_arg, config):
        calls.append((float(kappa), float(mu), float(nu), len(u_arg), copula_arg))
        kappa = float(kappa)
        mu = float(mu)
        nu = float(nu)
        value = (
            (kappa - 0.8) ** 2
            + mu ** 2
            + (nu - 1.0) ** 2
            + 1.0
        )
        grad = np.array(
            [2.0 * (kappa - 0.8), 2.0 * mu, 2.0 * (nu - 1.0)],
            dtype=np.float64,
        )
        info = {
            "backend": "matrix",
            "transition_method": "matrix",
            "engine": "cpp",
            "kappa_dt": kappa / (len(u_arg) - 1),
            "n_obs": len(u_arg),
        }
        return value, grad, info

    monkeypatch.setattr(
        _cpp_scar_ou, "neg_loglik_with_grad_info",
        fake_neg_loglik_with_grad_info)

    result = SCARTMStrategy(
        backend="cpp",
        smart_init=False,
        K=20,
        adaptive=False,
        max_K=None,
    ).fit(
        copula,
        u,
        alpha0=np.array([1.0, 0.1, 1.2]),
        maxiter=2,
        maxfun=12,
    )

    assert calls
    assert result.backend == "cpp"


def test_default_auto_fit_path_uses_cpp_for_supported_copula(monkeypatch):
    u = _data(seed=20260630, n=10)
    copula = ClaytonCopula(rotate=0, transform_type="softplus")
    calls = []

    def fake_neg_loglik_with_grad_info(kappa, mu, nu, u_arg, copula_arg, config):
        calls.append((float(kappa), float(mu), float(nu), len(u_arg), copula_arg))
        kappa = float(kappa)
        mu = float(mu)
        nu = float(nu)
        value = (
            (kappa - 0.8) ** 2
            + mu ** 2
            + (nu - 1.0) ** 2
            + 1.0
        )
        grad = np.array(
            [2.0 * (kappa - 0.8), 2.0 * mu, 2.0 * (nu - 1.0)],
            dtype=np.float64,
        )
        info = {
            "backend": "matrix",
            "transition_method": "auto",
            "engine": "cpp",
            "kappa_dt": kappa / (len(u_arg) - 1),
            "n_obs": len(u_arg),
        }
        return value, grad, info

    monkeypatch.setattr(
        _cpp_scar_ou, "neg_loglik_with_grad_info",
        fake_neg_loglik_with_grad_info)

    result = SCARTMStrategy(
        smart_init=False,
        K=20,
        adaptive=False,
        max_K=None,
    ).fit(
        copula,
        u,
        alpha0=np.array([1.0, 0.1, 1.2]),
        maxiter=2,
        maxfun=12,
    )

    assert calls
    assert result.backend == "auto"
