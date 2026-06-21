"""Contracts for native multivariate row and grid operations."""

import numpy as np
import pytest

from pyscarcopula import (
    EquicorrGaussianCopula,
    StochasticStudentCopula,
)
from pyscarcopula.copula.multivariate import equicorr, stochastic_student
from pyscarcopula.numerical import _cpp_extension, multivariate_native
from pyscarcopula.numerical._cpp_extension import CppError


def _observations(n=24, d=4):
    return np.random.default_rng(20260616).uniform(0.05, 0.95, (n, d))


def _correlation():
    return np.array(
        [
            [1.0, 0.30, -0.10, 0.15],
            [0.30, 1.0, 0.20, -0.05],
            [-0.10, 0.20, 1.0, 0.25],
            [0.15, -0.05, 0.25, 1.0],
        ],
        dtype=np.float64,
    )


def test_pybind_exports_multivariate_bulk_operations():
    module = _cpp_extension.load()
    assert hasattr(module, "multivariate_log_pdf_and_grad")
    assert hasattr(module, "multivariate_pdf_and_grad_grid")
    assert hasattr(module, "multivariate_gaussian_conditional")
    assert hasattr(module, "multivariate_student_conditional")


def _reference_conditional_latent(
        correlation, given_indices, given_latent, normal_draws,
        *, df=None, chi_square=None):
    d = correlation.shape[0]
    free_indices = np.array([
        index for index in range(d) if index not in given_indices])
    R_gg = correlation[np.ix_(given_indices, given_indices)]
    R_fg = correlation[np.ix_(free_indices, given_indices)]
    R_gf = correlation[np.ix_(given_indices, free_indices)]
    R_ff = correlation[np.ix_(free_indices, free_indices)]
    solved = np.linalg.solve(R_gg, given_latent)
    location = R_fg @ solved
    schur = R_ff - R_fg @ np.linalg.solve(R_gg, R_gf)
    radial = 1.0
    if df is not None:
        conditional_df = float(df) + len(given_indices)
        delta = float(given_latent @ solved)
        schur = ((float(df) + delta) / conditional_df) * schur
        radial = np.sqrt(conditional_df / float(chi_square))
    return location + radial * (np.linalg.cholesky(schur) @ normal_draws)


def test_native_gaussian_conditional_matches_reference_with_supplied_draws():
    correlation = _correlation()
    correlations = np.stack([correlation, correlation * 0.9 + np.eye(4) * 0.1])
    given_indices = np.array([0, 2], dtype=np.int32)
    given_latent = np.array([-0.7, 0.8])
    normal_draws = np.array([[0.2, -1.1], [0.5, 0.3]])

    actual = multivariate_native.gaussian_conditional_latent(
        correlations, given_indices, given_latent, normal_draws)
    expected = np.vstack([
        _reference_conditional_latent(
            correlations[row], given_indices, given_latent,
            normal_draws[row])
        for row in range(2)
    ])

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=3e-15)


def test_native_student_conditional_matches_reference_with_supplied_draws():
    correlation = _correlation()
    given_indices = np.array([0, 2], dtype=np.int32)
    given_latent = np.array([
        [-0.9, 0.6],
        [-1.1, 0.8],
    ])
    degrees = np.array([5.0, 9.0])
    normal_draws = np.array([[0.2, -1.1], [0.5, 0.3]])
    chi_square = np.array([4.2, 8.1])

    actual = multivariate_native.student_conditional_latent(
        correlation,
        given_indices,
        given_latent,
        degrees,
        normal_draws,
        chi_square,
    )
    expected = np.vstack([
        _reference_conditional_latent(
            correlation,
            given_indices,
            given_latent[row],
            normal_draws[row],
            df=degrees[row],
            chi_square=chi_square[row],
        )
        for row in range(2)
    ])

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=3e-15)


def test_native_conditional_reports_failure_index_without_python_fallback():
    invalid = np.array([
        [1.0, 2.0, 0.0],
        [2.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    with pytest.raises(CppError, match="failure_index=0"):
        multivariate_native.gaussian_conditional_latent(
            invalid,
            np.array([0], dtype=np.int32),
            np.array([0.2]),
            np.zeros((1, 2)),
        )


@pytest.mark.parametrize(
    "copula",
    [
        EquicorrGaussianCopula(d=4),
        StochasticStudentCopula(d=4, R=_correlation()),
    ],
)
def test_native_multivariate_transform_round_trip(copula):
    x = np.linspace(-3.0, 3.0, 13)
    parameter = copula.transform(x)

    np.testing.assert_allclose(
        copula.inv_transform(parameter), x, rtol=0.0, atol=2e-12)
    step = 1e-6
    finite_difference = (
        copula.transform(x + step) - copula.transform(x - step)
    ) / (2.0 * step)
    np.testing.assert_allclose(
        copula.dtransform(x), finite_difference, rtol=2e-9, atol=2e-10)


def test_student_inverse_transform_preserves_lower_boundary_clamp():
    copula = StochasticStudentCopula(d=4, R=_correlation())
    values = copula.inv_transform(
        np.array([copula._df_offset, copula._df_offset - 1.0]))
    np.testing.assert_allclose(
        values, np.full(2, np.log(1e-15)), rtol=0.0, atol=1e-14)


def test_equicorr_native_rows_and_grid_are_consistent():
    u = _observations()
    copula = EquicorrGaussianCopula(d=4)
    rho = np.linspace(-0.15, 0.55, len(u))

    log_pdf, dlog = copula.log_pdf_and_dlog_dr_rows(u, rho)
    step = 1e-6
    finite_difference = (
        copula.log_pdf_rows(u, rho + step)
        - copula.log_pdf_rows(u, rho - step)
    ) / (2.0 * step)
    np.testing.assert_allclose(
        dlog, finite_difference, rtol=2e-7, atol=2e-8)

    x_grid = np.linspace(-2.0, 2.0, 9)
    fi, dfi = copula.pdf_and_grad_on_grid_batch(u, x_grid)
    expected_fi = np.column_stack([
        np.exp(copula.log_pdf_rows(u, parameter))
        for parameter in copula.transform(x_grid)
    ])
    expected_dfi = np.column_stack([
        expected_fi[:, index]
        * copula.dlog_pdf_dr_rows(u, parameter)
        * copula.dtransform(x_grid)[index]
        for index, parameter in enumerate(copula.transform(x_grid))
    ])
    np.testing.assert_allclose(fi, expected_fi, rtol=3e-12, atol=2e-12)
    np.testing.assert_allclose(dfi, expected_dfi, rtol=3e-12, atol=2e-12)


def test_student_native_rows_and_grid_use_full_cache_block():
    u = _observations()
    copula = StochasticStudentCopula(d=4, R=_correlation())
    cache = copula.prepare_emission_cache(u)
    start, stop = 5, 19
    block = u[start:stop]
    df = np.linspace(3.5, 9.0, len(block))

    log_pdf, dlog = copula.log_pdf_and_dlog_dr_rows(
        block, df, t_index=start, cache=cache)
    expected_log = np.empty(len(block))
    expected_grad = np.empty(len(block))
    for index, df_value in enumerate(df):
        observation = block[index:index + 1]
        expected_log[index] = copula.log_pdf_rows(
            observation,
            df_value,
            t_index=start + index,
            cache=cache,
        )[0]
        expected_grad[index] = copula.dlog_pdf_dr_rows(
            observation,
            df_value,
            t_index=start + index,
            cache=cache,
        )[0]

    np.testing.assert_allclose(log_pdf, expected_log, rtol=0.0, atol=2e-12)
    np.testing.assert_allclose(dlog, expected_grad, rtol=2e-6, atol=2e-8)

    x_grid = np.linspace(-1.5, 1.5, 7)
    cached = copula.pdf_and_grad_on_grid_batch(
        block, x_grid, t_index=start, cache=cache)
    fresh = copula.pdf_and_grad_on_grid_batch(block, x_grid)
    np.testing.assert_allclose(cached[0], fresh[0], rtol=0.0, atol=2e-14)
    np.testing.assert_allclose(cached[1], fresh[1], rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("x_value", [-7.0, 500.0, 2_500_000.0])
def test_student_grid_uses_exact_quantiles_outside_ppf_cache_range(x_value):
    u = np.array([[0.9169235897008583, 0.9500874927089409]])
    R = np.array([
        [1.0, 0.7363755858397765],
        [0.7363755858397765, 1.0],
    ])
    copula = StochasticStudentCopula(d=2, R=R)
    x_grid = np.array([x_value])
    df = copula.transform(x_grid)

    log_pdf, dlog = copula.log_pdf_and_dlog_dr_rows(u, df)
    fi, dfi = copula.pdf_and_grad_on_grid_batch(u, x_grid)

    expected_pdf = np.exp(log_pdf)
    expected_grad = expected_pdf * dlog * copula.dtransform(x_grid)
    assert np.all(np.isfinite(fi))
    assert np.all(np.isfinite(dfi))
    np.testing.assert_allclose(fi[:, 0], expected_pdf, rtol=2e-10, atol=0.0)
    np.testing.assert_allclose(
        dfi[:, 0], expected_grad, rtol=2e-4, atol=1e-12)


def test_student_grid_mixed_cache_range_matches_exact_row_evaluator():
    u = np.array([
        [0.9169235897008583, 0.9500874927089409],
        [0.2, 0.8],
    ])
    R = np.array([
        [1.0, 0.7363755858397765],
        [0.7363755858397765, 1.0],
    ])
    copula = StochasticStudentCopula(d=2, R=R)
    x_grid = np.array([-7.0, 0.0, 500.0])
    df_grid = copula.transform(x_grid)

    fi, dfi = copula.pdf_and_grad_on_grid_batch(u, x_grid)
    expected_fi = np.empty_like(fi)
    expected_dfi = np.empty_like(dfi)
    for index, df in enumerate(df_grid):
        log_pdf, dlog = copula.log_pdf_and_dlog_dr_rows(u, df)
        expected_fi[:, index] = np.exp(log_pdf)
        expected_dfi[:, index] = (
            expected_fi[:, index]
            * dlog
            * copula.dtransform([x_grid[index]])[0]
        )

    np.testing.assert_allclose(
        fi[:, [0, 2]], expected_fi[:, [0, 2]], rtol=2e-10, atol=1e-13)
    np.testing.assert_allclose(
        dfi[:, [0, 2]], expected_dfi[:, [0, 2]], rtol=2e-4, atol=1e-11)

    cached_fi, cached_dfi = copula.pdf_and_grad_on_grid_batch(
        u, x_grid[1:2])
    np.testing.assert_allclose(fi[:, 1], cached_fi[:, 0], rtol=0.0, atol=2e-14)
    np.testing.assert_allclose(
        dfi[:, 1], cached_dfi[:, 0], rtol=1e-9, atol=1e-10)


def test_production_multivariate_methods_use_native_surface():
    for name in (
            "_equicorr_transform",
            "_equicorr_inv_transform",
            "_equicorr_dtransform",
            "_equicorr_log_pdf",
            "_equicorr_dlog_pdf_drho",
            "_equicorr_pdf_and_grad_batch"):
        assert not hasattr(equicorr, name)
    u = _observations(8)
    grid = np.linspace(-1.0, 1.0, 5)
    equicorr_copula = EquicorrGaussianCopula(d=4)
    student_copula = StochasticStudentCopula(d=4, R=_correlation())

    equicorr_copula.transform(grid)
    equicorr_copula.log_pdf_and_dlog_dr_rows(u, 0.25)
    equicorr_copula.pdf_and_grad_on_grid_batch(u, grid)
    student_copula.transform(grid)
    student_copula.log_pdf_and_dlog_dr_rows(u, 5.0)
    student_copula.pdf_and_grad_on_grid_batch(u, grid)


def test_native_multivariate_failure_is_translated():
    copula = EquicorrGaussianCopula(d=4)
    with pytest.raises(CppError, match="failure_index=0"):
        copula.log_pdf_rows(_observations(2), 1.0)


def test_native_student_rejects_invalid_factorization_state():
    module = _cpp_extension.load()
    spec = module.CopulaSpec()
    spec.family = module.CopulaFamily.Student
    spec.rotation = module.Rotation.R0
    spec.transform = module.Transform.Softplus
    spec.offset = 2.0 + 1e-6
    spec.dim = 4
    spec.l_inv = np.ones(16, dtype=np.float64).tolist()
    spec.log_det = 0.0

    result = dict(module.multivariate_log_pdf_and_grad(
        spec, _observations(2), np.array([5.0]), 0))
    assert result["status"] == module.SCAR_INVALID_FAMILY
