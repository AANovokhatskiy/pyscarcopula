from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
from pathlib import Path
from types import SimpleNamespace

from scipy.stats import t as t_dist

from pyscarcopula import (
    BivariateGaussianCopula,
    ClaytonCopula,
    FrankCopula,
    GumbelCopula,
    JoeCopula,
)
from pyscarcopula._constants import PSEUDO_OBS_EPS
from pyscarcopula._utils import pobs
from pyscarcopula._types import LatentResult, ou_params
from pyscarcopula.api import fit
from pyscarcopula.copula.multivariate import StochasticStudentCopula
from pyscarcopula.copula.independent import IndependentCopula
from pyscarcopula.numerical._scar_ou_config import AutoTMConfig
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

auto_loglik = lambda *args, **kwargs: _cpp_scar_ou.loglik(
    *args, **kwargs)[0]
auto_loglik_with_info = _cpp_scar_ou.loglik
auto_neg_loglik_with_grad = _cpp_scar_ou.neg_loglik_with_grad


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
    )


@pytest.fixture
def hermite_cache_module():
    module = _cpp_scar_ou._cpp_extension.load()
    module._reset_hermite_rule_cache_limits_for_testing()
    try:
        yield module
    finally:
        module._reset_hermite_rule_cache_limits_for_testing()


@pytest.mark.parametrize("df", [2.0001, 2.1, 5.0, 30.0, 1000.0])
def test_cpp_student_quantile_matches_scipy(df):
    module = _cpp_scar_ou._cpp_extension.load()
    probabilities = np.array([
        0.0,
        PSEUDO_OBS_EPS / 10.0,
        PSEUDO_OBS_EPS,
        1e-6,
        1e-3,
        0.1,
        0.5,
        0.9,
        1.0 - 1e-3,
        1.0 - 1e-6,
        1.0 - PSEUDO_OBS_EPS,
        1.0 - PSEUDO_OBS_EPS / 10.0,
        1.0,
    ])
    got = np.array([
        module._student_quantile(float(probability), df)
        for probability in probabilities
    ])
    expected = t_dist.ppf(
        np.clip(
            probabilities,
            PSEUDO_OBS_EPS,
            1.0 - PSEUDO_OBS_EPS,
        ),
        df=df,
    )

    np.testing.assert_allclose(got, expected, rtol=2e-9, atol=2e-10)


def test_cpp_and_python_quantile_boundary_constants_match():
    module = _cpp_scar_ou._cpp_extension.load()
    assert module.PSEUDO_OBS_EPS == PSEUDO_OBS_EPS


@pytest.mark.parametrize("transition_method", ["local", "spectral"])
def test_cpp_stochastic_student_gradient_matches_python(transition_method):
    rng = np.random.default_rng(20260608)
    u = rng.uniform(0.01, 0.99, size=(18, 3))
    copula = StochasticStudentCopula(
        d=3,
        R=np.array([
            [1.0, 0.45, 0.2],
            [0.45, 1.0, 0.35],
            [0.2, 0.35, 1.0],
        ]),
    )
    cfg = AutoTMConfig(
        transition_method=transition_method,
        K=24,
        adaptive=False,
        max_K=24,
        gh_order=5,
        basis_order=12,
        quad_order=32,
    )
    alpha = (1.2, 0.3, 0.8)

    got_value, got_grad = _cpp_scar_ou.neg_loglik_with_grad(
        *alpha, u, copula, cfg)
    ref_value, ref_grad = auto_neg_loglik_with_grad(
        *alpha, u, copula, cfg)

    np.testing.assert_allclose(got_value, ref_value, rtol=0.0, atol=3e-4)
    np.testing.assert_allclose(got_grad, ref_grad, rtol=0.0, atol=3e-3)


def test_cpp_stochastic_student_spec_reuses_ppf_cache():
    rng = np.random.default_rng(20260609)
    u = rng.uniform(0.01, 0.99, size=(20, 3))
    copula = StochasticStudentCopula(
        d=3,
        R=np.array([
            [1.0, 0.4, 0.15],
            [0.4, 1.0, 0.25],
            [0.15, 0.25, 1.0],
        ]),
    )
    module = _cpp_scar_ou._cpp_extension.load()

    spec = _cpp_scar_ou._cpp_copula.make_spec(module, copula, u)
    repeated_spec = _cpp_scar_ou._cpp_copula.make_spec(
        module, copula, u.copy())
    cache = copula.prepare_emission_cache(u)

    assert repeated_spec is spec
    assert spec.ppf_n_obs == len(u)
    assert len(spec.ppf_nodes) == len(cache.ppf_nodes)
    assert len(spec.ppf_table) == cache.ppf_table.size
    assert copula.prepare_emission_cache(u) is cache


def test_cpp_stochastic_student_mutation_refreshes_likelihood_and_gradient():
    rng = np.random.default_rng(20260710)
    u = rng.uniform(0.05, 0.95, size=(18, 3))
    R = np.array([
        [1.0, 0.35, 0.15],
        [0.35, 1.0, 0.2],
        [0.15, 0.2, 1.0],
    ])
    copula = StochasticStudentCopula(d=3, R=R)
    cfg = AutoTMConfig(
        transition_method="matrix",
        K=10,
        adaptive=False,
        max_K=10,
    )
    alpha = (1.1, 0.4, 0.9)
    module = _cpp_scar_ou._cpp_extension.load()

    before_spec = _cpp_scar_ou._cpp_copula.make_spec(module, copula, u)
    before_cache = copula.prepare_emission_cache(u)
    before_value, before_grad = _cpp_scar_ou.neg_loglik_with_grad(
        *alpha, u, copula, cfg)

    u[0] = np.array([0.91, 0.08, 0.82])
    after_spec = _cpp_scar_ou._cpp_copula.make_spec(module, copula, u)
    after_cache = copula.prepare_emission_cache(u)
    after_value, after_grad = _cpp_scar_ou.neg_loglik_with_grad(
        *alpha, u, copula, cfg)

    fresh = StochasticStudentCopula(d=3, R=R)
    fresh_value, fresh_grad = _cpp_scar_ou.neg_loglik_with_grad(
        *alpha, u.copy(), fresh, cfg)
    python_value, python_grad = auto_neg_loglik_with_grad(
        *alpha, u, copula, cfg)

    assert after_cache is not before_cache
    assert after_cache.version > before_cache.version
    assert after_spec is not before_spec
    assert abs(after_value - before_value) > 1e-6
    assert np.max(np.abs(after_grad - before_grad)) > 1e-6
    np.testing.assert_allclose(after_value, fresh_value, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(after_grad, fresh_grad, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        after_value, python_value, rtol=0.0, atol=5e-4)
    np.testing.assert_allclose(
        after_grad, python_grad, rtol=0.0, atol=5e-3)


def test_cpp_stochastic_student_view_mutation_refreshes_spec():
    rng = np.random.default_rng(20260711)
    base = rng.uniform(0.05, 0.95, size=(16, 3))
    u = base[:, ::-1]
    R = np.array([
        [1.0, 0.25, 0.1],
        [0.25, 1.0, 0.2],
        [0.1, 0.2, 1.0],
    ])
    copula = StochasticStudentCopula(d=3, R=R)
    module = _cpp_scar_ou._cpp_extension.load()

    first_spec = _cpp_scar_ou._cpp_copula.make_spec(module, copula, u)
    first_cache = copula.prepare_emission_cache(u)
    base[2, 2] = 0.77
    second_spec = _cpp_scar_ou._cpp_copula.make_spec(module, copula, u)
    second_cache = copula.prepare_emission_cache(u)

    assert second_cache is not first_cache
    assert second_spec is not first_spec
    np.testing.assert_array_equal(second_cache.u_snapshot, u)


def test_cpp_student_ppf_buffer_setter_owns_copy_and_validates_shape():
    module = _cpp_scar_ou._cpp_extension.load()
    spec = module.CopulaSpec()
    spec.family = module.CopulaFamily.Student
    spec.dim = 3
    nodes = np.array([2.1, 3.0, 5.0], dtype=np.float64)
    table = np.arange(3 * 4 * 3, dtype=np.float64).reshape(3, 4, 3)
    expected_nodes = nodes.copy()
    expected_table = table.copy()

    spec.set_student_ppf_cache(nodes, table)
    nodes[:] = -1.0
    table[:] = -1.0

    assert spec.ppf_n_obs == 4
    np.testing.assert_array_equal(spec.ppf_nodes, expected_nodes)
    np.testing.assert_array_equal(
        spec.ppf_table, expected_table.reshape(-1))

    with pytest.raises(ValueError, match="1D"):
        spec.set_student_ppf_cache(expected_nodes[:, None], expected_table)
    with pytest.raises(ValueError, match="shape"):
        spec.set_student_ppf_cache(expected_nodes, expected_table[:, :, :2])


@pytest.mark.parametrize(
    ("nodes", "table", "message"),
    [
        (
            np.array([2.1, np.nan, 5.0]),
            np.zeros((3, 2, 3)),
            "finite",
        ),
        (
            np.array([2.1, 2.1, 5.0]),
            np.zeros((3, 2, 3)),
            "strictly increasing",
        ),
        (
            np.array([2.1, 5.0, 3.0]),
            np.zeros((3, 2, 3)),
            "strictly increasing",
        ),
        (
            np.array([2.1, 3.0, 5.0]),
            np.full((3, 2, 3), np.inf),
            "finite",
        ),
    ],
)
def test_cpp_student_ppf_buffer_setter_rejects_invalid_values(
        nodes, table, message):
    module = _cpp_scar_ou._cpp_extension.load()
    spec = module.CopulaSpec()
    spec.family = module.CopulaFamily.Student
    spec.dim = 3
    valid_nodes = np.array([2.1, 3.0, 5.0])
    valid_table = np.arange(18, dtype=np.float64).reshape(3, 2, 3)
    spec.set_student_ppf_cache(valid_nodes, valid_table)

    with pytest.raises(ValueError, match=message):
        spec.set_student_ppf_cache(nodes, table)

    assert spec.ppf_n_obs == 2
    np.testing.assert_array_equal(spec.ppf_nodes, valid_nodes)
    np.testing.assert_array_equal(
        spec.ppf_table, valid_table.reshape(-1))


def test_cpp_stochastic_student_ppf_transfer_uses_contiguous_numpy_buffers():
    calls = []

    class BufferSpec:
        def set_student_ppf_cache(self, nodes, table):
            calls.append((nodes, table))

    nodes_source = np.arange(8, dtype=np.float64)[::2]
    table_source = np.arange(
        4 * 3 * 2, dtype=np.float64).reshape(4, 3, 2)[:, ::-1, :]
    cache = SimpleNamespace(
        ppf_nodes=nodes_source,
        ppf_table=table_source,
    )

    _cpp_scar_ou._cpp_copula._set_student_ppf_cache(BufferSpec(), cache)

    assert len(calls) == 1
    nodes, table = calls[0]
    assert isinstance(nodes, np.ndarray)
    assert isinstance(table, np.ndarray)
    assert nodes.dtype == np.float64
    assert table.dtype == np.float64
    assert nodes.flags.c_contiguous
    assert table.flags.c_contiguous
    np.testing.assert_array_equal(nodes, nodes_source)
    np.testing.assert_array_equal(table, table_source)


def test_cpp_stochastic_student_corr_change_refreshes_spec_in_place():
    rng = np.random.default_rng(20260609)
    u = rng.uniform(0.01, 0.99, size=(20, 3))
    copula = StochasticStudentCopula(d=3, R=np.eye(3))
    module = _cpp_scar_ou._cpp_extension.load()

    first_spec = _cpp_scar_ou._cpp_copula.make_spec(module, copula, u)
    cache = copula.prepare_emission_cache(u)
    ppf_nodes = cache.ppf_nodes
    ppf_table = cache.ppf_table
    first_l_inv = list(first_spec.l_inv)
    first_log_det = first_spec.log_det

    copula._set_R(np.array([
        [1.0, 0.4, 0.15],
        [0.4, 1.0, 0.25],
        [0.15, 0.25, 1.0],
    ]))
    second_spec = _cpp_scar_ou._cpp_copula.make_spec(module, copula, u)

    assert second_spec is first_spec
    assert copula.prepare_emission_cache(u) is cache
    assert cache.ppf_nodes is ppf_nodes
    assert cache.ppf_table is ppf_table
    assert second_spec.ppf_nodes == list(ppf_nodes)
    assert second_spec.ppf_table == list(ppf_table.ravel())
    assert list(second_spec.l_inv) != first_l_inv
    assert second_spec.log_det != first_log_det


def test_cpp_stochastic_student_observation_then_corr_refresh_is_consistent():
    rng = np.random.default_rng(20260712)
    u = rng.uniform(0.05, 0.95, size=(18, 3))
    copula = StochasticStudentCopula(d=3, R=np.eye(3))
    module = _cpp_scar_ou._cpp_extension.load()

    first_spec = _cpp_scar_ou._cpp_copula.make_spec(module, copula, u)
    u[1, 1] = 0.87
    observation_spec = _cpp_scar_ou._cpp_copula.make_spec(
        module, copula, u)
    observation_cache = copula.prepare_emission_cache(u)

    R = np.array([
        [1.0, 0.3, 0.1],
        [0.3, 1.0, 0.2],
        [0.1, 0.2, 1.0],
    ])
    copula._set_R(R)
    final_spec = _cpp_scar_ou._cpp_copula.make_spec(module, copula, u)

    assert observation_spec is not first_spec
    assert final_spec is observation_spec
    assert copula.prepare_emission_cache(u) is observation_cache
    np.testing.assert_array_equal(
        final_spec.ppf_table, observation_cache.ppf_table.ravel())
    np.testing.assert_allclose(
        final_spec.l_inv,
        np.asarray(copula._L_inv).ravel(),
        rtol=0.0,
        atol=0.0,
    )


def test_cpp_stochastic_student_corr_then_observation_refresh_is_consistent():
    rng = np.random.default_rng(20260713)
    u = rng.uniform(0.05, 0.95, size=(18, 3))
    copula = StochasticStudentCopula(d=3, R=np.eye(3))
    module = _cpp_scar_ou._cpp_extension.load()

    first_spec = _cpp_scar_ou._cpp_copula.make_spec(module, copula, u)
    R = np.array([
        [1.0, 0.28, 0.12],
        [0.28, 1.0, 0.18],
        [0.12, 0.18, 1.0],
    ])
    copula._set_R(R)
    correlation_spec = _cpp_scar_ou._cpp_copula.make_spec(
        module, copula, u)

    u[2, 0] = 0.93
    final_spec = _cpp_scar_ou._cpp_copula.make_spec(module, copula, u)
    final_cache = copula.prepare_emission_cache(u)

    assert correlation_spec is first_spec
    assert final_spec is not correlation_spec
    np.testing.assert_array_equal(
        final_spec.ppf_table, final_cache.ppf_table.ravel())
    np.testing.assert_allclose(
        final_spec.l_inv,
        np.asarray(copula._L_inv).ravel(),
        rtol=0.0,
        atol=0.0,
    )


def test_cpp_stochastic_student_uses_python_df_transform_offset():
    u = np.full((4, 2), 0.5)
    copula = StochasticStudentCopula(d=2, R=np.eye(2))
    module = _cpp_scar_ou._cpp_extension.load()

    spec = _cpp_scar_ou._cpp_copula.make_spec(module, copula, u)

    assert spec.offset == copula._df_offset
    np.testing.assert_allclose(
        copula.transform([-100.0, -20.0, 0.0, 20.0]),
        spec.offset + np.logaddexp(0.0, [-100.0, -20.0, 0.0, 20.0]),
        rtol=0.0,
        atol=1e-15,
    )


def test_cpp_stochastic_student_ppf_cache_has_no_derived_size_limit(
        monkeypatch):
    u = np.full((9000, 3), 0.5)
    copula = StochasticStudentCopula(d=3, R=np.eye(3))
    module = _cpp_scar_ou._cpp_extension.load()
    cache = SimpleNamespace(
        ppf_nodes=np.array([2.1, 10.0]),
        ppf_table=np.zeros((2, len(u), 3), dtype=np.float64),
        version=987654321,
    )
    calls = 0

    def prepare_cache(values):
        nonlocal calls
        calls += 1
        assert values is u
        return cache

    monkeypatch.setattr(copula, "prepare_emission_cache", prepare_cache)

    spec = _cpp_scar_ou._cpp_copula.make_spec(module, copula, u)

    assert u.size * 199 > 5_000_000
    assert calls == 1
    assert spec.ppf_n_obs == len(u)
    assert len(spec.ppf_nodes) == 2
    assert len(spec.ppf_table) == 2 * u.size


def test_cpp_stochastic_student_large_dimension_cache_is_finite():
    T = 32
    d = 80
    rng = np.random.default_rng(20260701)
    u = rng.uniform(0.05, 0.95, size=(T, d))
    copula = StochasticStudentCopula(d=d, R=np.eye(d))
    cfg = AutoTMConfig(
        transition_method="matrix",
        K=6,
        adaptive=False,
        max_K=6,
    )
    module = _cpp_scar_ou._cpp_extension.load()

    spec = _cpp_scar_ou._cpp_copula.make_spec(module, copula, u)
    value, grad = _cpp_scar_ou.neg_loglik_with_grad(
        1.1, 0.7, 0.9, u, copula, cfg)

    assert spec.ppf_n_obs == T
    assert spec.ppf_nodes
    assert spec.ppf_table
    assert np.isfinite(value)
    assert np.all(np.isfinite(grad))


@pytest.mark.parametrize("rho", [0.99999999, -0.24999999])
def test_cpp_stochastic_student_ill_conditioned_corr_matches_python(rho):
    rng = np.random.default_rng(20260702)
    u = rng.uniform(0.02, 0.98, size=(30, 5))
    R = np.full((5, 5), rho, dtype=np.float64)
    np.fill_diagonal(R, 1.0)
    copula = StochasticStudentCopula(d=5, R=R)
    cfg = AutoTMConfig(
        transition_method="matrix",
        K=12,
        adaptive=False,
        max_K=12,
    )
    params = (1.1, -20.0, 0.7)

    python_value, python_grad = auto_neg_loglik_with_grad(
        *params, u, copula, cfg)
    cpp_value, cpp_grad = _cpp_scar_ou.neg_loglik_with_grad(
        *params, u, copula, cfg)

    assert np.linalg.cond(copula.R) > 1e7
    assert np.linalg.eigvalsh(copula.R)[0] > 0.0
    assert np.isfinite(python_value)
    assert np.all(np.isfinite(python_grad))
    np.testing.assert_allclose(
        cpp_value, python_value, rtol=0.0, atol=5e-10)
    np.testing.assert_allclose(
        cpp_grad, python_grad, rtol=0.0, atol=5e-10)


@pytest.mark.parametrize("mu", [-1000.0, 1000.0])
def test_cpp_stochastic_student_extreme_latent_mean_matches_python(mu):
    rng = np.random.default_rng(20260703)
    u = rng.uniform(0.01, 0.99, size=(24, 3))
    R = np.full((3, 3), 0.5, dtype=np.float64)
    np.fill_diagonal(R, 1.0)
    copula = StochasticStudentCopula(d=3, R=R)
    cfg = AutoTMConfig(
        transition_method="matrix",
        K=12,
        adaptive=False,
        max_K=12,
    )
    params = (1.1, mu, 0.1)

    python_value, python_grad = auto_neg_loglik_with_grad(
        *params, u, copula, cfg)
    cpp_value, cpp_grad = _cpp_scar_ou.neg_loglik_with_grad(
        *params, u, copula, cfg)

    assert np.isfinite(python_value)
    assert np.all(np.isfinite(python_grad))
    assert np.isfinite(cpp_value)
    assert np.all(np.isfinite(cpp_grad))
    np.testing.assert_allclose(
        cpp_value, python_value, rtol=0.0, atol=2e-8)
    np.testing.assert_allclose(
        cpp_grad, python_grad, rtol=0.0, atol=5e-6)


def test_cpp_stochastic_student_analytic_gradient_matches_finite_difference():
    rng = np.random.default_rng(20260610)
    u = rng.uniform(0.01, 0.99, size=(20, 3))
    copula = StochasticStudentCopula(
        d=3,
        R=np.array([
            [1.0, 0.35, 0.1],
            [0.35, 1.0, 0.3],
            [0.1, 0.3, 1.0],
        ]),
    )
    cfg = AutoTMConfig(
        transition_method="local",
        K=24,
        adaptive=False,
        max_K=24,
        gh_order=5,
    )
    alpha = np.array([1.2, 0.3, 0.8])
    value, grad = _cpp_scar_ou.neg_loglik_with_grad(
        *alpha, u, copula, cfg)
    finite_diff = np.empty(3)

    for i, step in enumerate((1e-5, 1e-5, 1e-5)):
        plus = alpha.copy()
        minus = alpha.copy()
        plus[i] += step
        minus[i] -= step
        finite_diff[i] = (
            _cpp_scar_ou.neg_loglik(*plus, u, copula, cfg)
            - _cpp_scar_ou.neg_loglik(*minus, u, copula, cfg)
        ) / (2.0 * step)

    assert np.isfinite(value)
    np.testing.assert_allclose(grad, finite_diff, rtol=0.0, atol=5e-5)


@pytest.mark.parametrize("transition_method", ["matrix", "local"])
def test_cpp_stochastic_student_corr_gradient_matches_finite_difference(
        transition_method):
    rng = np.random.default_rng(20260704)
    u = rng.uniform(0.03, 0.97, size=(30, 3))
    R = np.array([
        [1.0, 0.35, 0.1],
        [0.35, 1.0, 0.3],
        [0.1, 0.3, 1.0],
    ])
    copula = StochasticStudentCopula(d=3, R=R)
    cfg = AutoTMConfig(
        transition_method=transition_method,
        K=18,
        adaptive=False,
        max_K=18,
        gh_order=5,
    )
    alpha = np.array([1.2, 0.3, 0.8])

    value, ou_grad, corr_grad, info = (
        _cpp_scar_ou.neg_loglik_with_grad_and_corr_info(
            *alpha, u, copula, cfg))
    ref_value, ref_ou_grad = _cpp_scar_ou.neg_loglik_with_grad(
        *alpha, u, copula, cfg)
    finite_difference = np.empty(3, dtype=np.float64)
    step = 1e-6
    pos = 0
    for i in range(1, 3):
        for j in range(i):
            plus = R.copy()
            minus = R.copy()
            plus[i, j] += step
            plus[j, i] += step
            minus[i, j] -= step
            minus[j, i] -= step
            copula._set_R(plus)
            plus_value = _cpp_scar_ou.neg_loglik(
                *alpha, u, copula, cfg)
            copula._set_R(minus)
            minus_value = _cpp_scar_ou.neg_loglik(
                *alpha, u, copula, cfg)
            finite_difference[pos] = (
                plus_value - minus_value) / (2.0 * step)
            pos += 1
    copula._set_R(R)

    assert info["backend"] in {"matrix", "local"}
    assert corr_grad.shape == (3,)
    np.testing.assert_allclose(value, ref_value, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        ou_grad, ref_ou_grad, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(
        corr_grad, finite_difference, rtol=0.0, atol=2e-7)


def test_cpp_bivariate_stochastic_student_gradient_matches_finite_difference():
    rng = np.random.default_rng(20260612)
    u = rng.uniform(0.01, 0.99, size=(24, 2))
    copula = StochasticStudentCopula(
        d=2,
        R=np.array([
            [1.0, 0.55],
            [0.55, 1.0],
        ]),
    )
    cfg = AutoTMConfig(
        transition_method="local",
        K=26,
        adaptive=False,
        max_K=26,
        gh_order=5,
    )
    alpha = np.array([1.3, 0.4, 0.9])
    _, grad = _cpp_scar_ou.neg_loglik_with_grad(
        *alpha, u, copula, cfg)
    finite_diff = np.empty(3)

    for i in range(3):
        plus = alpha.copy()
        minus = alpha.copy()
        plus[i] += 1e-5
        minus[i] -= 1e-5
        finite_diff[i] = (
            _cpp_scar_ou.neg_loglik(*plus, u, copula, cfg)
            - _cpp_scar_ou.neg_loglik(*minus, u, copula, cfg)
        ) / 2e-5

    np.testing.assert_allclose(grad, finite_diff, rtol=0.0, atol=5e-5)


def test_cpp_stochastic_student_forward_state_matches_python():
    rng = np.random.default_rng(20260629)
    u = rng.uniform(0.03, 0.97, size=(16, 3))
    copula = StochasticStudentCopula(
        d=3,
        R=np.array([
            [1.0, 0.4, 0.15],
            [0.4, 1.0, 0.3],
            [0.15, 0.3, 1.0],
        ]),
    )
    alpha = (0.8, 0.1, 0.9)
    cfg = AutoTMConfig(
        transition_method="local",
        K=22,
        grid_range=3.0,
        adaptive=False,
        max_K=22,
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

    predictive_cpp = _cpp_scar_ou.predictive_mean(
        *alpha, u, copula, cfg)
    predictive_python = tm_forward_predictive_mean(
        *alpha, u, copula, **kwargs)
    z_cpp, prob_cpp = _cpp_scar_ou.state_distribution(
        *alpha, u, copula, cfg)
    z_python, prob_python = tm_state_distribution(
        *alpha, u, copula, **kwargs)

    np.testing.assert_allclose(
        predictive_cpp, predictive_python, rtol=0.0, atol=3e-6)
    np.testing.assert_allclose(z_cpp, z_python, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        prob_cpp, prob_python, rtol=0.0, atol=1e-6)


def test_cpp_stochastic_student_rejects_non_triangular_l_inv():
    rng = np.random.default_rng(20260630)
    u = rng.uniform(0.03, 0.97, size=(8, 3))
    copula = StochasticStudentCopula(d=3, R=np.eye(3))
    cfg = AutoTMConfig(
        transition_method="local",
        K=16,
        adaptive=False,
        max_K=16,
    )
    module, params, spec, obs, cpp_cfg, _ = _cpp_scar_ou._inputs(
        1.0, 0.0, 1.0, u, copula, cfg)
    l_inv = list(spec.l_inv)
    l_inv[1] = 0.1
    spec.l_inv = l_inv

    result = module.ScarOuEvaluator().loglik_local_gh(
        params, spec, obs, cpp_cfg)

    assert result["status"] == 5


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


def test_cpp_hermite_cache_reuses_same_rule(hermite_cache_module):
    module = hermite_cache_module

    first = module._hermite_rule_for_testing(32, 16)
    after_first = module._hermite_rule_cache_info()
    second = module._hermite_rule_for_testing(32, 16)
    after_second = module._hermite_rule_cache_info()

    assert after_first["entries"] == 1
    assert after_first["misses"] == 1
    assert after_first["insertions"] == 1
    assert after_second["entries"] == 1
    assert after_second["hits"] == 1
    assert after_second["bytes"] == after_first["bytes"]
    for first_values, second_values in zip(first, second):
        np.testing.assert_array_equal(first_values, second_values)


def test_cpp_hermite_cache_uses_lru_eviction(hermite_cache_module):
    module = hermite_cache_module
    module._set_hermite_rule_cache_limits_for_testing(
        2, module.HERMITE_RULE_CACHE_MAX_BYTES)

    module._hermite_rule_for_testing(24, 8)
    module._hermite_rule_for_testing(32, 8)
    module._hermite_rule_for_testing(24, 8)
    module._hermite_rule_for_testing(40, 8)
    before_reloading_evicted = module._hermite_rule_cache_info()
    module._hermite_rule_for_testing(32, 8)
    after_reloading_evicted = module._hermite_rule_cache_info()

    assert before_reloading_evicted["entries"] == 2
    assert before_reloading_evicted["hits"] == 1
    assert before_reloading_evicted["evictions"] == 1
    assert after_reloading_evicted["misses"] == (
        before_reloading_evicted["misses"] + 1)
    assert after_reloading_evicted["evictions"] == 2


def test_cpp_hermite_cache_respects_entry_and_byte_limits(
        hermite_cache_module):
    module = hermite_cache_module
    module._set_hermite_rule_cache_limits_for_testing(3, 25_000)

    for quad_order in (24, 32, 40, 48, 56):
        module._hermite_rule_for_testing(quad_order, 8)
        info = module._hermite_rule_cache_info()
        assert info["entries"] <= info["max_entries"]
        assert info["bytes"] <= info["max_bytes"]

    info = module._hermite_rule_cache_info()
    assert info["max_entries"] == 3
    assert info["max_bytes"] == 25_000
    assert info["evictions"] > 0


def test_cpp_hermite_cache_does_not_store_oversized_rule(
        hermite_cache_module):
    module = hermite_cache_module
    module._set_hermite_rule_cache_limits_for_testing(4, 1)

    first = module._hermite_rule_for_testing(16, 8)
    after_first = module._hermite_rule_cache_info()
    second = module._hermite_rule_for_testing(16, 8)
    after_second = module._hermite_rule_cache_info()

    assert after_first["entries"] == 0
    assert after_first["bytes"] == 0
    assert after_first["oversized_skips"] == 1
    assert after_second["misses"] == 2
    assert after_second["oversized_skips"] == 2
    for first_values, second_values in zip(first, second):
        np.testing.assert_array_equal(first_values, second_values)


def test_cpp_hermite_cache_concurrent_same_key_is_correct(
        hermite_cache_module):
    module = hermite_cache_module
    workers = 8

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(
            lambda _: module._hermite_rule_for_testing(96, 32),
            range(workers),
        ))

    info = module._hermite_rule_cache_info()
    assert info["entries"] == 1
    assert info["insertions"] == 1
    assert info["hits"] + info["misses"] == workers
    assert info["hits"] + info["duplicate_builds"] == workers - 1
    for result in results[1:]:
        for expected, actual in zip(results[0], result):
            np.testing.assert_array_equal(actual, expected)


def test_cpp_hermite_rule_is_unchanged_after_eviction_and_rebuild(
        hermite_cache_module):
    module = hermite_cache_module
    module._set_hermite_rule_cache_limits_for_testing(
        1, module.HERMITE_RULE_CACHE_MAX_BYTES)

    before = module._hermite_rule_for_testing(48, 16)
    module._hermite_rule_for_testing(56, 16)
    rebuilt = module._hermite_rule_for_testing(48, 16)
    info = module._hermite_rule_cache_info()

    assert info["entries"] == 1
    assert info["evictions"] == 2
    for expected, actual in zip(before, rebuilt):
        np.testing.assert_array_equal(actual, expected)


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

    cpp_result = fit(copula, u, method="scar-tm-ou")

    assert cpp_result.success
    assert np.isfinite(cpp_result.log_likelihood)
    assert cpp_result.diagnostics["selected_engine"] == "cpp"


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

    cpp_result = fit(copula, u, **common)

    assert np.isfinite(cpp_result.log_likelihood)
    assert np.all(np.isfinite(cpp_result.params.values))
    assert cpp_result.diagnostics["selected_engine"] == "cpp"


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


@pytest.mark.parametrize("transition_method", ["matrix", "local"])
@pytest.mark.parametrize("horizon", ["current", "next"])
def test_cpp_state_distribution_propagates_numerical_failure(
        transition_method, horizon):
    u = np.ones((4, 2), dtype=np.float64)
    copula = GumbelCopula(rotate=0, transform_type="softplus")
    cfg = AutoTMConfig(
        transition_method=transition_method,
        K=8,
        adaptive=False,
        max_K=8,
        gh_order=5,
    )

    with pytest.raises(_cpp_scar_ou.CppError, match="numerical_failure"):
        _cpp_scar_ou.state_distribution(
            1.0, 0.0, 1.0, u, copula, cfg, horizon=horizon)


@pytest.mark.parametrize("transition_method", ["matrix", "local"])
@pytest.mark.parametrize(
    "entrypoint",
    [_cpp_scar_ou.predictive_mean, _cpp_scar_ou.mixture_h],
)
def test_cpp_forward_filter_propagates_numerical_failure(
        transition_method, entrypoint):
    u = np.ones((4, 2), dtype=np.float64)
    copula = GumbelCopula(rotate=0, transform_type="softplus")
    cfg = AutoTMConfig(
        transition_method=transition_method,
        K=8,
        adaptive=False,
        max_K=8,
        gh_order=5,
    )

    with pytest.raises(_cpp_scar_ou.CppError, match="numerical_failure"):
        entrypoint(1.0, 0.0, 1.0, u, copula, cfg)


@pytest.mark.parametrize(
    ("method", "expected_backend"),
    [("matrix", 2), ("local_gh", 1)],
)
def test_direct_cpp_failed_state_is_empty(method, expected_backend):
    u = np.ones((4, 2), dtype=np.float64)
    copula = GumbelCopula(rotate=0, transform_type="softplus")
    cfg = AutoTMConfig(
        transition_method="matrix" if method == "matrix" else "local",
        K=8,
        adaptive=False,
        max_K=8,
        gh_order=5,
    )
    module, params, spec, obs, cpp_cfg, _ = _cpp_scar_ou._inputs(
        1.0, 0.0, 1.0, u, copula, cfg)

    result = getattr(
        module.ScarOuEvaluator(),
        f"state_distribution_{method}",
    )(params, spec, obs, cpp_cfg, False)

    assert result["status"] == 7
    assert int(result["backend"]) == expected_backend
    assert len(result["z_grid"]) == 0
    assert len(result["prob"]) == 0


def test_cpp_strategy_predictive_state_uses_cache():
    u = _data(seed=20260617, n=24)
    copula = ClaytonCopula(rotate=0, transform_type="softplus")
    result = _result(alpha=(0.8, 0.1, 1.0))
    strategy = SCARTMStrategy(
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


def test_cpp_supports_xtanh_copula_without_fallback():
    u = _data(seed=20260613, n=16)
    copula = GumbelCopula(rotate=180, transform_type="xtanh")
    alpha = np.array([0.8, 0.2, 1.0])

    strategy = SCARTMStrategy(
        transition_method="matrix",
        K=24,
        adaptive=False,
        max_K=None,
        analytical_grad=False,
    )
    assert strategy._uses_cpp(copula)
    assert np.isfinite(strategy.objective(copula, u, alpha))


def test_removed_backend_argument_fails_clearly():
    u = _data(seed=20260614, n=8)
    copula = GumbelCopula(rotate=0, transform_type="xtanh")
    with pytest.raises(TypeError, match="backend selection was removed"):
        SCARTMStrategy(backend="cpp", transition_method="matrix")


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

    with pytest.raises(ValueError, match="grid_range"):
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


def test_cpp_rejects_observation_dimension_mismatch_before_kernel_call():
    u = np.full((8, 3), 0.5, dtype=np.float64)
    copula = ClaytonCopula(rotate=0, transform_type="softplus")
    cfg = AutoTMConfig(
        transition_method="matrix",
        K=16,
        adaptive=False,
        max_K=None,
    )

    with pytest.raises(ValueError, match="copula dimension"):
        _cpp_scar_ou.loglik(1.0, 0.0, 1.0, u, copula, cfg)


@pytest.mark.parametrize(
    "convert",
    [
        np.asfortranarray,
        lambda values: values.astype(np.float32),
    ],
)
def test_cpp_observation_view_preserves_array_conversion_semantics(convert):
    u = _data(seed=20260625, n=8)
    copula = ClaytonCopula(rotate=0, transform_type="softplus")
    cfg = AutoTMConfig(
        transition_method="matrix",
        K=16,
        adaptive=False,
        max_K=None,
    )

    expected, _ = _cpp_scar_ou.loglik(1.0, 0.0, 1.0, u, copula, cfg)
    actual, _ = _cpp_scar_ou.loglik(
        1.0, 0.0, 1.0, convert(u), copula, cfg)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-7)


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
def test_cpp_invalid_size_is_rejected_before_kernel_call(call):
    u = _data(seed=20260626, n=8)
    copula = ClaytonCopula(rotate=0, transform_type="softplus")
    cfg = AutoTMConfig(
        transition_method="matrix",
        K=1,
        adaptive=False,
        max_K=None,
    )

    with pytest.raises(ValueError, match="K"):
        call((1.0, 0.0, 1.0), u, copula, cfg)


def test_cpp_resource_limits_match_extension_constants():
    module = _cpp_scar_ou._cpp_extension.load()

    assert module.MAX_GRID_SIZE == 100_000
    assert module.MAX_DENSE_GRID_SIZE == 10_000
    assert module.MAX_SPECTRAL_ORDER == 1_024
    assert not hasattr(module, "MAX_OBSERVATION_GRID_ELEMENTS")
    assert not hasattr(module, "MAX_SPECTRAL_ELEMENTS")
    assert not hasattr(module, "MAX_STUDENT_DIMENSION")
    assert not hasattr(module, "MAX_CORRELATION_SCORE_ELEMENTS")
    assert not hasattr(module, "MAX_PPF_TABLE_ELEMENTS")
    assert module.HERMITE_RULE_CACHE_MAX_ENTRIES == 16
    assert module.HERMITE_RULE_CACHE_MAX_BYTES == 8 * 1024 * 1024


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("K", True, "K"),
        ("K", 10_001, "K"),
        ("basis_order", 1_025, "basis_order"),
        ("basis_order", 505, "quad_order derived"),
        ("quad_order", 1_025, "quad_order"),
        ("gh_order", 1_025, "gh_order"),
    ],
)
def test_cpp_wrapper_rejects_unsafe_integer_config(field, value, message):
    u = _data(seed=20260706, n=8)
    copula = ClaytonCopula(rotate=0, transform_type="softplus")
    kwargs = {
        "transition_method": "matrix" if field == "K" else "spectral",
        field: value,
    }
    cfg = AutoTMConfig(**kwargs)

    with pytest.raises(ValueError, match=message):
        _cpp_scar_ou.loglik(1.0, 0.0, 1.0, u, copula, cfg)


def test_cpp_strategy_rejects_unsafe_config_at_construction():
    with pytest.raises(ValueError, match="K"):
        SCARTMStrategy(
            transition_method="matrix",
            K=10_001,
            max_K=None,
        )


def test_cpp_strategy_accepts_matrix_grid_above_old_limit():
    strategy = SCARTMStrategy(
        transition_method="matrix",
        K=4_097,
        max_K=4_097,
    )

    assert strategy.K == 4_097
    assert strategy.max_K == 4_097


def test_direct_cpp_accepts_student_dimension_above_old_limit():
    module = _cpp_scar_ou._cpp_extension.load()
    spec = module.CopulaSpec()
    spec.family = module.CopulaFamily.Student
    spec.rotation = module.Rotation.R0
    spec.transform = module.Transform.Softplus
    spec.offset = 2.0
    spec.dim = 1_025
    spec.l_inv = np.eye(1_025, dtype=np.float64).ravel()
    spec.log_det = 0.0
    params = module.OuParams()
    params.kappa = 1.0
    params.mu = 1.0
    params.nu = 0.5
    config = module.OuNumericalConfig()
    config.K = 4
    config.adaptive = False
    config.max_K = 4
    config.gh_order = 3
    u = np.random.default_rng(20260709).uniform(
        0.2, 0.8, size=(3, 1_025))

    result = module.ScarOuEvaluator().loglik_local_gh(
        params, spec, u, config)

    assert result["status"] == 0
    assert np.isfinite(result["log_likelihood"])


@pytest.mark.parametrize(
    ("method", "field", "value"),
    [
        ("matrix", "K", 10_001),
        ("spectral", "spectral_basis_order", 1_025),
        ("local", "gh_order", 1_025),
    ],
)
def test_direct_pybind_rejects_unsafe_config_without_allocation(
        method, field, value):
    module = _cpp_scar_ou._cpp_extension.load()
    u = _data(seed=20260707, n=8)
    copula = ClaytonCopula(rotate=0, transform_type="softplus")
    spec = _cpp_scar_ou._cpp_copula.make_spec(module, copula, u)
    params = module.OuParams()
    config = module.OuNumericalConfig()
    config.adaptive = False
    config.max_K = 0
    setattr(config, field, value)
    evaluator = module.ScarOuEvaluator()

    entrypoint = "loglik_local_gh" if method == "local" else f"loglik_{method}"
    result = getattr(evaluator, entrypoint)(
        params, spec, u, config)

    assert result["status"] == 2


def test_cpp_rejects_adaptive_grid_above_implementation_limit():
    u = _data(seed=20260708, n=8)
    copula = ClaytonCopula(rotate=0, transform_type="softplus")
    cfg = AutoTMConfig(
        transition_method="local",
        K=16,
        adaptive=True,
        max_K=None,
        pts_per_sigma=4,
    )

    with pytest.raises(_cpp_scar_ou.CppError, match="invalid_size"):
        _cpp_scar_ou.loglik(3.5e-7, 0.0, 1.0, u, copula, cfg)


@pytest.mark.parametrize(
    "entrypoint",
    [_cpp_scar_ou.predictive_mean, _cpp_scar_ou.mixture_h],
)
def test_cpp_spectral_forward_entrypoints_reconstruct_on_grid(entrypoint):
    u = _data(seed=20260627, n=8)
    copula = ClaytonCopula(rotate=0, transform_type="softplus")
    cfg = AutoTMConfig(transition_method="spectral")

    values = entrypoint(1.0, 0.0, 1.0, u, copula, cfg)
    assert values.shape == (len(u),)
    assert np.all(np.isfinite(values))


def test_cpp_spectral_state_distribution_reconstructs_on_grid():
    u = _data(seed=20260628, n=8)
    copula = ClaytonCopula(rotate=0, transform_type="softplus")
    cfg = AutoTMConfig(transition_method="spectral")

    z_grid, prob = _cpp_scar_ou.state_distribution(
        1.0, 0.0, 1.0, u, copula, cfg)
    assert z_grid.shape == prob.shape
    assert np.sum(prob) == pytest.approx(1.0)


@pytest.mark.parametrize("d", [2, 3, 6])
def test_cpp_spectral_corr_gradient_matches_finite_difference(d):
    rng = np.random.default_rng(20260705 + d)
    u = pobs(rng.standard_t(df=5.0, size=(30, d)))
    R = np.full((d, d), 0.2, dtype=np.float64)
    np.fill_diagonal(R, 1.0)
    copula = StochasticStudentCopula(d=d, R=R)
    cfg = AutoTMConfig(
        transition_method="spectral", basis_order=16, quad_order=48)

    value, ou_grad, corr_grad, info = (
        _cpp_scar_ou.neg_loglik_with_grad_and_corr_info(
            1.2, 0.5, 0.8, u, copula, cfg))

    finite_difference = []
    step = 1e-6
    for i in range(1, d):
        for j in range(i):
            plus = R.copy()
            minus = R.copy()
            plus[i, j] += step
            plus[j, i] += step
            minus[i, j] -= step
            minus[j, i] -= step
            copula._set_R(plus)
            plus_value = _cpp_scar_ou.neg_loglik(
                1.2, 0.5, 0.8, u, copula, cfg)
            copula._set_R(minus)
            minus_value = _cpp_scar_ou.neg_loglik(
                1.2, 0.5, 0.8, u, copula, cfg)
            finite_difference.append(
                (plus_value - minus_value) / (2.0 * step))
    copula._set_R(R)

    assert info["backend"] == "spectral"
    assert np.isfinite(value)
    assert np.all(np.isfinite(ou_grad))
    np.testing.assert_allclose(
        corr_grad,
        finite_difference,
        rtol=2e-8,
        atol=2e-8,
    )


def test_cpp_spectral_corr_gradient_is_finite_above_ppf_cache_range():
    u = np.array([
        [0.9169235897, 0.9500874927],
        [0.2, 0.8],
        [0.5, 0.55],
    ])
    R = np.array([[1.0, 0.736], [0.736, 1.0]])
    copula = StochasticStudentCopula(d=2, R=R)
    cfg = AutoTMConfig(
        transition_method="spectral", basis_order=16, quad_order=48)

    value, ou_grad, corr_grad, info = (
        _cpp_scar_ou.neg_loglik_with_grad_and_corr_info(
            1.0, 1_000_000.0, 10.0, u, copula, cfg))

    assert info["backend"] == "spectral"
    assert np.isfinite(value)
    assert np.all(np.isfinite(ou_grad))
    assert np.all(np.isfinite(corr_grad))


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
    monkeypatch.setattr(
        _cpp_scar_ou,
        "prepare_objective",
        lambda *args, **kwargs: (
            (_ for _ in ()).throw(
                _cpp_scar_ou.CppUnsupported("test fallback"))),
    )

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
    assert result.diagnostics["selected_engine"] == "cpp"


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
    monkeypatch.setattr(
        _cpp_scar_ou,
        "prepare_objective",
        lambda *args, **kwargs: (
            (_ for _ in ()).throw(
                _cpp_scar_ou.CppUnsupported("test fallback"))),
    )

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
    assert result.diagnostics["selected_engine"] == "cpp"
    assert not hasattr(result, "backend")
