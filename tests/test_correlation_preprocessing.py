"""Shared Kendall-correlation preprocessing contracts."""

from types import SimpleNamespace

import numpy as np

from pyscarcopula.copula.multivariate import (
    StochasticStudentCopula,
    StudentCopula,
)
from pyscarcopula.copula.multivariate import corr_param
from pyscarcopula.copula.multivariate.corr_param import (
    estimate_kendall_correlation,
    validate_corr_matrix,
)


def _ordinary_u(seed=20260623, n=80):
    rng = np.random.default_rng(seed)
    common = rng.normal(size=n)
    values = np.column_stack([
        common + 0.3 * rng.normal(size=n),
        0.5 * common + 0.7 * rng.normal(size=n),
        -0.3 * common + rng.normal(size=n),
    ])
    order = np.argsort(np.argsort(values, axis=0), axis=0)
    return (order + 1.0) / (n + 1.0)


def _assert_valid_correlation(result):
    validate_corr_matrix(result.correlation)
    assert np.all(np.isfinite(result.correlation))
    assert np.allclose(result.correlation, result.correlation.T)
    np.testing.assert_array_equal(
        np.diag(result.correlation),
        np.ones(result.correlation.shape[0]),
    )
    assert result.min_eigenvalue_after > 0.0


def test_static_and_stochastic_student_share_kendall_initialization():
    u = _ordinary_u()
    expected = estimate_kendall_correlation(u)

    static = StudentCopula()
    static.fit(u)
    stochastic = StochasticStudentCopula(d=3)
    stochastic_initial = stochastic._initial_corr(u)

    np.testing.assert_allclose(
        static.shape, expected.correlation, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        stochastic_initial, expected.correlation, rtol=0.0, atol=0.0)
    assert (
        static.fit_result.diagnostics["corr_projection_applied"]
        == expected.projection_applied
    )
    assert (
        stochastic.correlation_preprocessing_diagnostics()
        == expected.diagnostics()
    )


def test_constant_column_maps_unavailable_kendall_pair_to_independence():
    u = _ordinary_u()
    u[:, 1] = 0.5

    result = estimate_kendall_correlation(u)
    static = StudentCopula()
    static.fit(u)
    stochastic = StochasticStudentCopula(d=3)
    stochastic_initial = stochastic._initial_corr(u)

    _assert_valid_correlation(result)
    np.testing.assert_allclose(
        static.shape, result.correlation, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        stochastic_initial, result.correlation, rtol=0.0, atol=0.0)
    assert result.nonfinite_kendall_pairs == ((0, 1), (1, 2))
    assert result.input_correlation[0, 1] == 0.0
    assert result.input_correlation[1, 2] == 0.0
    assert result.correlation[0, 1] == 0.0
    assert result.correlation[1, 2] == 0.0


def test_nearly_singular_kendall_matrix_is_projected_and_reported():
    base = np.linspace(0.01, 0.99, 60)
    u = np.column_stack([base, base, base])

    result = estimate_kendall_correlation(u)

    _assert_valid_correlation(result)
    assert result.projection_applied is True
    assert result.min_eigenvalue_before <= 0.0
    assert result.min_eigenvalue_after > 0.0


def test_nonfinite_pairwise_statistic_is_recorded(monkeypatch):
    original = corr_param.kendalltau
    calls = 0

    def one_missing_pair(x, y):
        nonlocal calls
        calls += 1
        if calls == 2:
            return SimpleNamespace(statistic=np.nan)
        return original(x, y)

    monkeypatch.setattr(corr_param, "kendalltau", one_missing_pair)
    result = estimate_kendall_correlation(_ordinary_u())

    _assert_valid_correlation(result)
    assert result.nonfinite_kendall_pairs == ((0, 2),)
    assert result.input_correlation[0, 2] == 0.0


def test_stochastic_mle_exposes_kendall_projection_diagnostics():
    u = _ordinary_u(n=45)
    u[:, 1] = 0.5
    model = StochasticStudentCopula(d=3)

    result = model.fit(u, method="mle", maxiter=3, maxfun=12)

    diagnostics = result.diagnostics
    assert diagnostics["corr_initialization_source"] == "kendall"
    assert diagnostics["corr_nonfinite_kendall_pairs"] == ((0, 1), (1, 2))
    assert np.isfinite(diagnostics["corr_min_eigenvalue_before"])
    assert diagnostics["corr_min_eigenvalue_after"] > 0.0
    _assert_valid_correlation(model._corr_preprocessing)
