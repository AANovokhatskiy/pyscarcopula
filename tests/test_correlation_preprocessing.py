"""Shared Kendall-correlation preprocessing contracts."""

import numpy as np
import pytest

from pyscarcopula.copula.multivariate import (
    StochasticStudentCopula,
    StudentCopula,
)
from pyscarcopula.copula.multivariate import corr_param
from pyscarcopula.copula.multivariate.corr_param import (
    estimate_kendall_correlation,
    preprocess_correlation_matrix,
    project_to_corr,
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


def test_constant_column_preprocessing_is_available_but_static_fit_rejects_it():
    u = _ordinary_u()
    u[:, 1] = 0.5

    result = estimate_kendall_correlation(u)
    static = StudentCopula()
    with pytest.raises(ValueError, match="not identifiable"):
        static.fit(u)
    stochastic = StochasticStudentCopula(d=3)
    stochastic_initial = stochastic._initial_corr(u)

    _assert_valid_correlation(result)
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


def test_kendall_initialization_has_no_python_statistic_callback():
    assert not hasattr(corr_param, "kendalltau")
    result = estimate_kendall_correlation(_ordinary_u())
    _assert_valid_correlation(result)
    assert result.nonfinite_kendall_pairs == ()


def test_projection_and_diagnostics_do_not_use_numpy_eigensolvers(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("correlation eigendecomposition must run in C++")

    monkeypatch.setattr(np.linalg, "eigh", forbidden)
    monkeypatch.setattr(np.linalg, "eigvalsh", forbidden)
    matrix = np.array([
        [1.0, 1.2, -0.1],
        [1.2, 1.0, 0.3],
        [-0.1, 0.3, 1.0],
    ])

    projected = project_to_corr(matrix)
    result = preprocess_correlation_matrix(matrix, source="supplied")

    np.testing.assert_array_equal(projected, result.correlation)
    np.testing.assert_array_equal(result.input_correlation, matrix)
    assert result.source == "supplied"
    assert result.projection_applied is True
    assert result.min_eigenvalue_before < 0.0
    assert result.min_eigenvalue_after > 0.0
    validate_corr_matrix(result.correlation)


def test_stochastic_mle_exposes_kendall_projection_diagnostics():
    u = _ordinary_u(n=45)
    u[:, 1] = 0.5
    model = StochasticStudentCopula(d=3)

    result = model.fit(
        u, method="mle", gtol=1e-2, maxiter=100, maxfun=500)

    assert result.success
    diagnostics = result.diagnostics
    assert diagnostics["corr_initialization_source"] == "kendall"
    assert diagnostics["corr_nonfinite_kendall_pairs"] == ((0, 1), (1, 2))
    assert np.isfinite(diagnostics["corr_min_eigenvalue_before"])
    assert diagnostics["corr_min_eigenvalue_after"] > 0.0
    _assert_valid_correlation(model._corr_preprocessing)
