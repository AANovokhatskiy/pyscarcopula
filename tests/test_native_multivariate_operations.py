"""Contracts for native multivariate row and grid operations."""

import inspect

import numpy as np
import pytest
from scipy.stats import chi2
from scipy.stats import norm
from scipy.stats import t as t_dist

import pyscarcopula.stattests as statt
from pyscarcopula import (
    EquicorrGaussianCopula,
    FactorCorrelation,
    GaussianCopula,
    StochasticStudentCopula,
    StudentCopula,
)
from pyscarcopula.copula.multivariate import equicorr, stochastic_student
from pyscarcopula.copula.multivariate import conditional as conditional_module
from pyscarcopula.copula.multivariate import gaussian as gaussian_module
from pyscarcopula._native import _extension as _cpp_extension, multivariate as multivariate_native
from pyscarcopula._native.errors import NativeError


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


def test_gaussian_score_correlation_matches_frozen_scipy_formula():
    observations = _observations(128)
    expected = np.corrcoef(norm.ppf(observations).T)

    actual = multivariate_native.gaussian_score_correlation(observations)

    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)
    assert "np.corrcoef" not in inspect.getsource(gaussian_module)
    assert "norm.ppf" not in inspect.getsource(gaussian_module)


def test_gaussian_score_correlation_rejects_constant_columns():
    observations = _observations(16)
    observations[:, 2] = 0.5

    with pytest.raises(ValueError, match="failure_coordinate=2"):
        multivariate_native.gaussian_score_correlation(observations)


@pytest.mark.parametrize("student", [False, True])
def test_factor_conditional_raw_binding_rejects_scalar_factor_draws(student):
    module = _cpp_extension.load()
    factor = FactorCorrelation(np.array([
        [0.35, -0.10],
        [0.20, 0.25],
        [-0.15, 0.30],
        [0.22, 0.12],
    ])).prepare()
    native = factor._native
    indices = np.array([0, 2], dtype=np.int32)
    given = np.array([0.2, 0.8], dtype=np.float64)
    scalar_factors = np.array(0.0, dtype=np.float64)
    residuals = np.zeros((2, 4), dtype=np.float64)

    with pytest.raises(ValueError, match="invalid shapes"):
        if student:
            module.factor_student_conditional_from_uniforms(
                native,
                indices,
                given,
                np.array([6.0]),
                scalar_factors,
                residuals,
                np.ones(2),
                1,
            )
        else:
            module.factor_gaussian_conditional_from_uniforms(
                native,
                indices,
                given,
                scalar_factors,
                residuals,
                1,
            )


def test_dense_student_sampling_preserves_frozen_rng_draw_order():
    model = StudentCopula(d=4, R=_correlation())
    model._correlation = _correlation()
    model.df = 6.75
    actual_rng = np.random.default_rng(20260825)
    expected_rng = np.random.default_rng(20260825)

    actual = model.sample(3, rng=actual_rng)
    chi_square = expected_rng.chisquare(model.df, size=3)
    normal_draws = expected_rng.standard_normal((3, model.dimension))
    expected = multivariate_native.student_sample_from_draws(
        model._correlation, model.df, normal_draws, chi_square)

    np.testing.assert_array_equal(actual, expected)
    assert actual_rng.random() == expected_rng.random()


def test_pybind_exports_multivariate_bulk_operations():
    module = _cpp_extension.load()
    assert hasattr(module, "multivariate_log_pdf_and_grad")
    assert hasattr(module, "multivariate_pdf_and_grad_grid")
    assert hasattr(module, "multivariate_gaussian_conditional")
    assert hasattr(module, "multivariate_student_conditional")
    for name in (
            "multivariate_gaussian_sample_from_normals",
            "multivariate_student_sample_from_draws",
            "factor_gaussian_sample_from_normals",
            "factor_student_sample_from_draws",
            "multivariate_gaussian_conditional_from_uniforms",
            "multivariate_student_conditional_from_uniforms",
            "factor_gaussian_conditional_from_uniforms",
            "factor_student_conditional_from_uniforms",
            "dense_gaussian_rosenblatt_transform",
            "factor_gaussian_rosenblatt_transform",
            "factor_student_rosenblatt_transform",
            "radial_uniform_summary"):
        assert hasattr(module, name)


def _factor_operator():
    loadings = np.array([
        [0.30, 0.05],
        [0.20, -0.15],
        [-0.10, 0.25],
        [0.12, 0.18],
    ])
    return FactorCorrelation(loadings).prepare()


def test_native_dense_unconditional_sampling_matches_fixed_draw_oracle():
    correlation = _correlation()
    normal_draws = np.array([
        [0.2, -1.1, 0.4, 0.7],
        [0.5, 0.3, -0.8, 1.2],
        [-0.6, 0.9, 0.1, -0.2],
    ])
    latent = normal_draws @ np.linalg.cholesky(correlation).T

    gaussian = multivariate_native.gaussian_sample_from_normals(
        correlation, normal_draws)
    np.testing.assert_allclose(
        gaussian, norm.cdf(latent), rtol=0.0, atol=5e-15)

    df = 6.5
    chi_square = np.array([4.1, 7.3, 5.8])
    student = multivariate_native.student_sample_from_draws(
        correlation, df, normal_draws, chi_square)
    expected = t_dist.cdf(
        latent * np.sqrt(df / chi_square)[:, None], df=df)
    np.testing.assert_allclose(
        student, expected, rtol=0.0, atol=5e-12)


def test_native_factor_unconditional_sampling_matches_fixed_draw_oracle():
    correlation = _factor_operator()
    factors = np.array([[0.2, -1.1], [0.5, 0.3], [-0.6, 0.9]])
    residuals = np.array([
        [0.4, 0.7, -0.3, 0.1],
        [-0.8, 1.2, 0.6, -0.4],
        [0.1, -0.2, 0.8, 0.5],
    ])
    latent = (
        factors @ correlation.loadings.T
        + residuals * np.sqrt(correlation.uniqueness)[None, :]
    )

    gaussian = multivariate_native.factor_gaussian_sample_from_normals(
        correlation, factors, residuals)
    np.testing.assert_allclose(
        gaussian, norm.cdf(latent), rtol=0.0, atol=5e-15)

    df = 7.25
    chi_square = np.array([5.2, 8.4, 6.1])
    scale = np.sqrt(df / chi_square)[:, None]
    student = multivariate_native.factor_student_sample_from_draws(
        correlation, df, factors, residuals, chi_square)
    np.testing.assert_allclose(
        student,
        t_dist.cdf(latent * scale, df=df),
        rtol=0.0,
        atol=5e-12,
    )


def _conditional_reference(correlation, df, given, normal_draws, chi_square):
    given_indices = np.array(sorted(given))
    free_indices = np.array([
        index for index in range(len(correlation))
        if index not in given_indices
    ])
    given_uniforms = np.array([given[index] for index in given_indices])
    if df is None:
        given_latent = norm.ppf(given_uniforms)
    else:
        given_latent = t_dist.ppf(given_uniforms, df=df)
    R_gg = correlation[np.ix_(given_indices, given_indices)]
    R_fg = correlation[np.ix_(free_indices, given_indices)]
    schur = (
        correlation[np.ix_(free_indices, free_indices)]
        - R_fg @ np.linalg.solve(
            R_gg, correlation[np.ix_(given_indices, free_indices)])
    )
    solved = np.linalg.solve(R_gg, given_latent)
    mean = R_fg @ solved
    if df is None:
        latent = mean + normal_draws @ np.linalg.cholesky(schur).T
        return norm.cdf(latent)
    conditional_df = df + len(given_indices)
    shape = ((df + given_latent @ solved) / conditional_df) * schur
    latent = (
        mean
        + normal_draws @ np.linalg.cholesky(shape).T
        * np.sqrt(conditional_df / chi_square)[:, None]
    )
    return t_dist.cdf(latent, df=df)


def _factor_conditional_reference(
        operator, df, given, factor_draws, residual_draws, chi_square):
    given_indices = np.array(sorted(given))
    free_indices = np.array([
        index for index in range(operator.dimension)
        if index not in given_indices
    ])
    given_uniforms = np.array([given[index] for index in given_indices])
    loadings = operator.loadings
    uniqueness = operator.uniqueness
    given_loadings = loadings[given_indices]
    given_uniqueness = uniqueness[given_indices]
    precision = (
        np.eye(operator.rank)
        + given_loadings.T
        @ (given_loadings / given_uniqueness[:, None])
    )
    lower = np.linalg.cholesky(precision)
    root = np.linalg.solve(lower.T, np.eye(operator.rank))
    if df is None:
        given_latent = norm.ppf(given_uniforms)[None, :]
        projected = (
            given_latent / given_uniqueness[None, :]
        ) @ given_loadings
        mean = np.linalg.solve(precision, projected.T).T
        factors = factor_draws @ root.T + mean
        latent = (
            factors @ loadings.T
            + residual_draws * np.sqrt(uniqueness)[None, :]
        )
        return norm.cdf(latent[:, free_indices])

    given_latent = np.repeat(
        t_dist.ppf(given_uniforms, df=df)[None, :],
        len(factor_draws),
        axis=0,
    )
    projected = (
        given_latent / given_uniqueness[None, :]
    ) @ given_loadings
    mean = np.linalg.solve(precision, projected.T).T
    delta = (
        np.sum(given_latent**2 / given_uniqueness[None, :], axis=1)
        - np.sum(projected * mean, axis=1)
    )
    radial = np.sqrt((df + np.maximum(delta, 0.0)) / chi_square)
    factors = (factor_draws @ root.T) * radial[:, None] + mean
    latent = (
        factors @ loadings.T
        + residual_draws
        * np.sqrt(uniqueness)[None, :]
        * radial[:, None]
    )
    return t_dist.cdf(latent[:, free_indices], df=df)


@pytest.mark.parametrize("factor", [False, True], ids=["dense", "factor"])
def test_native_complete_conditionals_match_fixed_draw_oracles(factor):
    operator = _factor_operator()
    correlation = operator.to_dense() if factor else _correlation()
    given = {0: 0.17, 2: 0.81}
    given_indices = np.array(sorted(given), dtype=np.int32)
    given_uniforms = np.array([given[index] for index in given_indices])
    normal_draws = np.array([[0.2, -1.1], [0.5, 0.3], [-0.6, 0.9]])
    chi_square = np.array([4.2, 8.1, 5.7])

    if factor:
        factor_draws = np.array([[0.2, -1.1], [0.5, 0.3], [-0.6, 0.9]])
        residual_draws = np.array([
            [0.4, 0.7, -0.3, 0.1],
            [-0.8, 1.2, 0.6, -0.4],
            [0.1, -0.2, 0.8, 0.5],
        ])
        gaussian = (
            multivariate_native.factor_gaussian_conditional_from_uniforms(
                operator, given_indices, given_uniforms,
                factor_draws, residual_draws)
        )
        student = (
            multivariate_native.factor_student_conditional_from_uniforms(
                operator, given_indices, given_uniforms, 6.5,
                factor_draws, residual_draws, chi_square)
        )
        expected_gaussian = _factor_conditional_reference(
            operator, None, given, factor_draws, residual_draws, None)
        expected_student = _factor_conditional_reference(
            operator, 6.5, given,
            factor_draws, residual_draws, chi_square)
        np.testing.assert_allclose(
            gaussian, expected_gaussian, rtol=0.0, atol=5e-12)
        np.testing.assert_allclose(
            student, expected_student, rtol=0.0, atol=5e-12)
    else:
        gaussian = multivariate_native.gaussian_conditional_from_uniforms(
            correlation, given_indices, given_uniforms, normal_draws)
        expected_gaussian = _conditional_reference(
            correlation, None, given, normal_draws, None)
        np.testing.assert_allclose(
            gaussian, expected_gaussian, rtol=0.0, atol=5e-12)

        student = multivariate_native.student_conditional_from_uniforms(
            correlation, given_indices, given_uniforms, 6.5,
            normal_draws, chi_square)
        expected_student = _conditional_reference(
            correlation, 6.5, given, normal_draws, chi_square)
        np.testing.assert_allclose(
            student, expected_student, rtol=0.0, atol=5e-12)


def test_static_dense_and_factor_sampling_formulas_are_absent_from_python():
    conditional_sources = "\n".join(inspect.getsource(function) for function in (
        conditional_module.sample_gaussian_copula_conditional,
        conditional_module.sample_student_conditional,
        conditional_module.sample_factor_gaussian_conditional,
        conditional_module.sample_factor_student_conditional,
    ))
    for token in (
            "np.linalg", "norm.ppf", "t_dist.ppf", "norm.cdf",
            "t_dist.cdf", "conditional_factor_mean", "radial_scale"):
        assert token not in conditional_sources

    unconditional_sources = "\n".join(inspect.getsource(method) for method in (
        GaussianCopula.sample,
        StudentCopula.sample,
    ))
    for token in ("multivariate_normal", "multivariate_t.rvs", "norm.cdf"):
        assert token not in unconditional_sources


def test_native_gaussian_rosenblatt_matches_dense_and_factor_oracles():
    observations = np.array([
        [1e-10, 0.2, 0.8, 1.0 - 1e-10],
        [0.1, 0.5, 0.9, 0.35],
        [0.75, 0.6, 0.4, 0.25],
    ])

    def dense_oracle(correlation):
        quantiles = norm.ppf(np.clip(observations, 1e-10, 1.0 - 1e-10))
        whitened = np.linalg.solve(
            np.linalg.cholesky(correlation), quantiles.T).T
        return np.clip(norm.cdf(whitened), 1e-10, 1.0 - 1e-10)

    dense = multivariate_native.gaussian_rosenblatt(
        _correlation(), observations)
    np.testing.assert_allclose(
        dense, dense_oracle(_correlation()), rtol=0.0, atol=8e-15)

    factor = _factor_operator()
    compact = multivariate_native.factor_gaussian_rosenblatt(
        factor, observations)
    np.testing.assert_allclose(
        compact, dense_oracle(factor.to_dense()), rtol=0.0, atol=8e-15)


def test_native_factor_student_rosenblatt_matches_dense_native_reference():
    observations = np.array([
        [0.05, 0.2, 0.8, 0.95],
        [0.1, 0.5, 0.9, 0.35],
        [0.75, 0.6, 0.4, 0.25],
    ])
    df_path = np.array([2.5, 4.0, 17.0])
    factor = _factor_operator()
    compact = multivariate_native.factor_student_rosenblatt(
        factor, df_path, observations)
    dense = multivariate_native.dense_student_rosenblatt(
        factor.to_dense(), df_path, observations)
    np.testing.assert_allclose(compact, dense, rtol=0.0, atol=4e-12)


def test_native_radial_uniform_summary_matches_scipy_oracle():
    residuals = np.array([
        [1e-10, 0.2, 0.8, 1.0 - 1e-10],
        [0.1, 0.5, 0.9, 0.35],
        [0.75, 0.6, 0.4, 0.25],
    ])
    quantiles = norm.ppf(np.clip(residuals, 1e-10, 1.0 - 1e-10))
    expected = chi2.cdf(np.sum(quantiles * quantiles, axis=1), df=4)
    actual = multivariate_native.radial_uniform_summary(residuals)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-14)


def test_static_rosenblatt_and_radial_math_are_absent_from_python():
    assert not hasattr(
        multivariate_native, "_dense_student_rosenblatt_if_supported")
    assert not hasattr(
        multivariate_native, "_dense_student_rosenblatt_arrays_supported")
    wrappers = (
        statt.cvm_test,
        statt.gaussian_rosenblatt_transform,
        statt.factor_gaussian_rosenblatt_transform,
        statt.student_rosenblatt_transform,
        statt.factor_student_rosenblatt_transform,
    )
    forbidden_names = {
        "cholesky", "inv", "solve", "ppf", "cdf", "chi2", "norm",
        "t_dist", "dispatch_rvine_backend",
    }
    for wrapper in wrappers:
        assert forbidden_names.isdisjoint(wrapper.__code__.co_names)


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

    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)


def test_native_gaussian_conditional_shared_correlation_matches_reference():
    correlation = _correlation()
    given_indices = np.array([0, 2], dtype=np.int32)
    given_latent = np.array([
        [-0.7, 0.8],
        [-0.4, 0.2],
        [0.3, -0.5],
    ])
    normal_draws = np.array([
        [0.2, -1.1],
        [0.5, 0.3],
        [-0.6, 0.9],
    ])

    actual = multivariate_native.gaussian_conditional_latent(
        correlation, given_indices, given_latent, normal_draws)
    expected = np.vstack([
        _reference_conditional_latent(
            correlation, given_indices, given_latent[row],
            normal_draws[row])
        for row in range(len(normal_draws))
    ])

    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)


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

    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)


def test_conditional_bindings_accept_read_only_float64_views():
    module = _cpp_extension.load()
    correlation = _correlation()
    given_indices = np.array([0, 2], dtype=np.int32)
    given_latent = np.array([[-0.9, 0.6], [-1.1, 0.8]])
    degrees = np.array([5.0, 9.0])
    normal_draws = np.array([[0.2, -1.1], [0.5, 0.3]])
    chi_square = np.array([4.2, 8.1])
    for values in (
            correlation, given_latent, degrees,
            normal_draws, chi_square):
        values.setflags(write=False)

    result = dict(module.multivariate_student_conditional(
        correlation,
        given_indices,
        given_latent,
        degrees,
        normal_draws,
        chi_square,
    ))
    assert result["status"] == module.SCAR_OK
    expected = multivariate_native.student_conditional_latent(
        correlation,
        given_indices,
        given_latent,
        degrees,
        normal_draws,
        chi_square,
    )
    np.testing.assert_allclose(result["values"], expected, rtol=0.0, atol=0.0)


def test_conditional_bindings_preserve_forcecast_fallback():
    module = _cpp_extension.load()
    correlation = np.asfortranarray(_correlation().astype(np.float32))
    given_indices = np.array([0, 2], dtype=np.int64)
    given_latent = np.array([[-0.7, 99.0, 0.8]], dtype=np.float32)[:, ::2]
    normal_draws = np.array([[0.2, 99.0, -1.1]], dtype=np.float32)[:, ::2]

    result = dict(module.multivariate_gaussian_conditional(
        correlation, given_indices, given_latent, normal_draws))
    assert result["status"] == module.SCAR_OK
    expected = _reference_conditional_latent(
        _correlation(), np.array([0, 2]),
        np.array([-0.7, 0.8]), np.array([0.2, -1.1]))
    np.testing.assert_allclose(result["values"][0], expected, atol=2e-8)


@pytest.mark.parametrize(
    "argument_index,name",
    [
        (0, "correlations"),
        (2, "given_latent"),
        (3, "df"),
        (4, "normal_draws"),
        (5, "chi_square_draws"),
    ],
)
def test_conditional_views_keep_finite_validation(argument_index, name):
    module = _cpp_extension.load()
    arguments = [
        _correlation(),
        np.array([0, 2], dtype=np.int32),
        np.array([[-0.9, 0.6], [-1.1, 0.8]]),
        np.array([5.0, 9.0]),
        np.array([[0.2, -1.1], [0.5, 0.3]]),
        np.array([4.2, 8.1]),
    ]
    arguments[argument_index].flat[0] = np.nan

    with pytest.raises(ValueError, match=name):
        module.multivariate_student_conditional(*arguments)


def test_native_student_conditional_preserves_jitter_scaling_per_row():
    correlation = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ])
    given_indices = np.array([0], dtype=np.int32)
    given_latent = np.array([[0.2], [-0.4]])
    degrees = np.array([4.0, 12.0])
    normal_draws = np.array([[0.3, -0.8], [-0.2, 0.7]])
    chi_square = np.array([3.5, 10.5])

    actual = multivariate_native.student_conditional_latent(
        correlation,
        given_indices,
        given_latent,
        degrees,
        normal_draws,
        chi_square,
    )

    expected = []
    schur_base = np.ones((2, 2))
    for row in range(len(degrees)):
        conditional_df = degrees[row] + len(given_indices)
        delta = float(given_latent[row] @ given_latent[row])
        covariance_scale = (degrees[row] + delta) / conditional_df
        lower = np.linalg.cholesky(
            covariance_scale * schur_base + 1e-12 * np.eye(2))
        radial_scale = np.sqrt(conditional_df / chi_square[row])
        expected.append(radial_scale * (lower @ normal_draws[row]))

    # NumPy and the native scalar Cholesky differ slightly in the last
    # digits for this deliberately singular matrix. The tolerance remains
    # well below the 1e-8--1e-7 error produced by scaling a jittered base
    # factor instead of applying the fixed per-row jitter.
    np.testing.assert_allclose(
        actual, np.asarray(expected), rtol=0.0, atol=1e-9)


def test_native_conditional_reports_failure_index_without_python_fallback():
    invalid = np.array([
        [1.0, 2.0, 0.0],
        [2.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    with pytest.raises(NativeError, match="failure_index=0"):
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
def test_student_grid_controlled_quantiles_match_exact_row_evaluator(x_value):
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
    np.testing.assert_allclose(fi[:, 0], expected_pdf, rtol=5e-9, atol=1e-12)
    np.testing.assert_allclose(
        dfi[:, 0], expected_grad, rtol=2e-4, atol=2e-10)


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
        fi[:, [0, 2]], expected_fi[:, [0, 2]], rtol=5e-9, atol=1e-12)
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
    with pytest.raises(NativeError, match="failure_index=0"):
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
