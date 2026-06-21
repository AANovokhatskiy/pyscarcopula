"""Contracts for native SCAR-MC trajectory density evaluation."""

import numpy as np
import pytest

from pyscarcopula import (
    BivariateGaussianCopula,
    ClaytonCopula,
    FrankCopula,
    GumbelCopula,
    JoeCopula,
    StochasticStudentCopula,
)
from pyscarcopula._utils import pobs
from pyscarcopula.numerical import _cpp_extension, mc_native
from pyscarcopula.numerical.mc_samplers import (
    _copula_log_pdf_trajectory_grid,
    eis_find_auxiliary,
    m_sampler_loglik,
    p_sampler_loglik,
)
from pyscarcopula.numerical.ou_kernels import calculate_dwt


_U = np.array(
    [
        [0.12, 0.83],
        [0.71, 0.28],
        [0.44, 0.62],
        [0.91, 0.17],
    ],
    dtype=np.float64,
)
_X = np.linspace(-1.5, 1.5, 28).reshape(4, 7)


def _python_bivariate_trajectory_grid(copula, u, x):
    return np.vstack([
        copula.log_pdf(
            np.full(x.shape[1], row[0]),
            np.full(x.shape[1], row[1]),
            copula.transform(x[t]),
        )
        for t, row in enumerate(u)
    ])


def test_pybind_exports_mc_trajectory_density():
    assert "copula_log_pdf_trajectory_grid" in dir(_cpp_extension.load())


@pytest.mark.parametrize(
    "copula",
    [
        ClaytonCopula(rotate=90, transform_type="xtanh"),
        FrankCopula(),
        GumbelCopula(rotate=180),
        JoeCopula(rotate=270, transform_type="xtanh"),
        BivariateGaussianCopula(),
    ],
)
def test_native_bivariate_trajectory_grid_matches_point_route(copula):
    expected = _python_bivariate_trajectory_grid(copula, _U, _X)
    actual = mc_native.log_pdf_trajectory_grid(copula, _U, _X)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_native_student_trajectory_grid_matches_cached_grid_route():
    u = np.array(
        [
            [0.12, 0.83, 0.41],
            [0.71, 0.28, 0.64],
            [0.44, 0.62, 0.19],
            [0.91, 0.17, 0.52],
        ],
        dtype=np.float64,
    )
    correlation = np.array(
        [
            [1.0, 0.25, -0.1],
            [0.25, 1.0, 0.2],
            [-0.1, 0.2, 1.0],
        ],
        dtype=np.float64,
    )
    copula = StochasticStudentCopula(d=3, R=correlation)
    x = np.linspace(-1.0, 1.0, 24).reshape(4, 6)
    cache = copula.prepare_emission_cache(u)
    expected = np.vstack([
        np.log(copula.copula_grid_batch(
            u[t:t + 1],
            x[t],
            t_index=t,
            cache=cache,
        )[0])
        for t in range(len(u))
    ])

    actual = mc_native.log_pdf_trajectory_grid(copula, u, x)

    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize("n_obs", [3, 11])
def test_native_mc_call_count_is_independent_of_t(monkeypatch, n_obs):
    module = _cpp_extension.load()
    original = module.copula_log_pdf_trajectory_grid
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        module, "copula_log_pdf_trajectory_grid", counted)
    u = np.resize(_U, (n_obs, 2))
    x = np.resize(_X, (n_obs, _X.shape[1]))
    mc_native.log_pdf_trajectory_grid(GumbelCopula(), u, x)

    assert calls == 1


@pytest.mark.parametrize("n_obs", [6, 13])
def test_sampler_call_counts_are_independent_of_t(monkeypatch, n_obs):
    module = _cpp_extension.load()
    original = module.copula_log_pdf_trajectory_grid
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        module, "copula_log_pdf_trajectory_grid", counted)
    u = np.random.default_rng(n_obs).uniform(0.05, 0.95, (n_obs, 2))
    dwt = calculate_dwt(n_obs, 12, seed=n_obs)
    alpha = np.array([1.2, 0.3, 0.7])
    copula = GumbelCopula()

    p_sampler_loglik(*alpha, u, dwt, copula, True)
    assert calls == 1

    calls = 0
    zeros = np.zeros(n_obs)
    m_sampler_loglik(
        *alpha, u, dwt, zeros, zeros, copula, True)
    assert calls == 1

    calls = 0
    eis_find_auxiliary(alpha, u, 2, dwt, copula, True)
    assert calls == 4


def test_p_sampler_does_not_call_public_copula_numerics(monkeypatch):
    copula = GumbelCopula(rotate=180)
    u = pobs(np.random.default_rng(4).standard_normal((10, 2)))
    dwt = calculate_dwt(10, 24, seed=17)

    def fail(*args, **kwargs):
        raise AssertionError("public copula numerical method was called")

    monkeypatch.setattr(copula, "transform", fail)
    monkeypatch.setattr(copula, "log_pdf", fail)

    value = p_sampler_loglik(1.2, 0.35, 0.7, u, dwt, copula, True)

    assert np.isfinite(value)


def test_eis_does_not_call_public_copula_numerics(monkeypatch):
    copula = GumbelCopula(rotate=180)
    u = pobs(np.random.default_rng(5).standard_normal((10, 2)))
    dwt = calculate_dwt(10, 24, seed=19)

    def fail(*args, **kwargs):
        raise AssertionError("public copula numerical method was called")

    monkeypatch.setattr(copula, "transform", fail)
    monkeypatch.setattr(copula, "log_pdf", fail)

    a1t, a2t = eis_find_auxiliary(
        np.array([1.2, 0.35, 0.7]),
        u,
        1,
        dwt,
        copula,
        True,
    )
    value = m_sampler_loglik(
        1.2, 0.35, 0.7, u, dwt, a1t, a2t, copula, True)

    assert np.all(np.isfinite(a1t))
    assert np.all(np.isfinite(a2t))
    assert np.isfinite(value)


def test_synthetic_mc_copula_retains_python_fallback():
    class SyntheticCopula:
        def transform(self, x):
            return np.asarray(x, dtype=np.float64) + 2.0

        def log_pdf(self, u1, u2, r):
            return np.asarray(r) * 0.0 + np.asarray(u1) - np.asarray(u2)

    copula = SyntheticCopula()
    expected = np.repeat(
        (_U[:, 0] - _U[:, 1])[:, np.newaxis], _X.shape[1], axis=1)
    actual = _copula_log_pdf_trajectory_grid(_U, _X, copula)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_student_p_sampler_uses_native_trajectory_density():
    u = np.array(
        [
            [0.12, 0.83, 0.41],
            [0.71, 0.28, 0.64],
            [0.44, 0.62, 0.19],
            [0.91, 0.17, 0.52],
            [0.33, 0.76, 0.87],
        ],
        dtype=np.float64,
    )
    correlation = np.array(
        [
            [1.0, 0.25, -0.1],
            [0.25, 1.0, 0.2],
            [-0.1, 0.2, 1.0],
        ],
        dtype=np.float64,
    )
    copula = StochasticStudentCopula(d=3, R=correlation)
    dwt = calculate_dwt(5, 6, seed=17)

    value = p_sampler_loglik(
        1.2, 0.3, 0.7, u, dwt, copula, True)

    assert value == pytest.approx(
        1.2484329631093338, rel=2e-13, abs=2e-13)


def test_fixed_dwt_mc_and_eis_outputs_match_regression_values():
    u = pobs(
        np.random.default_rng(20260701).standard_normal((18, 2)))
    dwt = calculate_dwt(18, 64, seed=20260701)
    alpha = np.array([1.2, 0.35, 0.7])
    copula = GumbelCopula(rotate=180)

    p_value = p_sampler_loglik(*alpha, u, dwt, copula, True)
    a1t, a2t = eis_find_auxiliary(
        alpha, u, 2, dwt, copula, True)
    m_value = m_sampler_loglik(
        *alpha, u, dwt, a1t, a2t, copula, True)

    assert p_value == pytest.approx(
        3.069369272846249, rel=0.0, abs=0.0)
    assert m_value == pytest.approx(
        3.2062240751760727, rel=0.0, abs=0.0)
    np.testing.assert_allclose(
        a1t,
        [
            -1.3939143950823851,
            -1.7753980654961354,
            -1.9961869470118272,
            -2.0180716945666983,
            -1.9413785333353513,
            -2.3241382172287617,
            -2.7281707985016825,
            -2.6266973738622266,
            -1.7876811824491665,
            -1.0991186994045816,
            -1.0174443495918877,
            -1.036731215888142,
            -0.9933601347357597,
            -1.1714852573540142,
            -1.1904248459538787,
            -1.1474333871271591,
            -0.9888411476669061,
            -0.7488364742279616,
        ],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        a2t,
        [
            -0.0527215781968752,
            -0.18182225058204815,
            -0.2279085787360649,
            -0.1290007003255291,
            -0.1897021128572232,
            -0.552616209436509,
            -0.8358358833787543,
            -0.7894843438565038,
            -0.673848943676245,
            -0.6772934224595226,
            -0.5955501950462259,
            -0.41000239490731805,
            -0.31954146163177066,
            -0.3475706109667484,
            -0.3529550179047821,
            -0.3046888429219776,
            -0.2514343091706712,
            -0.18835142313547878,
        ],
        rtol=0.0,
        atol=0.0,
    )


def test_fixed_dwt_xtanh_outputs_match_regression_values():
    u = pobs(
        np.random.default_rng(20260701).standard_normal((18, 2)))
    dwt = calculate_dwt(18, 64, seed=20260701)
    alpha = np.array([1.2, 0.35, 0.7])
    copula = GumbelCopula(rotate=180, transform_type="xtanh")

    p_value = p_sampler_loglik(*alpha, u, dwt, copula, True)
    a1t, a2t = eis_find_auxiliary(
        alpha, u, 2, dwt, copula, True)
    m_value = m_sampler_loglik(
        *alpha, u, dwt, a1t, a2t, copula, True)

    assert p_value == pytest.approx(
        0.18021335665185778, rel=0.0, abs=0.0)
    assert m_value == pytest.approx(
        0.3248397150077815, rel=0.0, abs=0.0)
