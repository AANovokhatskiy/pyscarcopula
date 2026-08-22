"""Regression and oracle baselines for static elliptical fits.

The default fits intentionally characterize the pre-correlation-mode behavior:
Gaussian uses a normal-score plug-in correlation and Student uses a Kendall
plug-in correlation followed by a one-dimensional df optimization.  The
independent SciPy objectives below also preserve compact examples where these
plug-in points differ materially from a joint correlation optimum.
"""

import numpy as np
import pytest
from scipy.optimize import minimize_scalar
from scipy.stats import (
    multivariate_normal,
    multivariate_t,
    norm,
    t as t_dist,
)

from pyscarcopula import GaussianCopula, StudentCopula
from pyscarcopula._utils import pobs


_GAUSSIAN_BASELINES = (
    (
        24,
        np.array([[1.0, 0.65], [0.65, 1.0]]),
        9101,
        np.array(
            [
                [1.0, 0.5898729745568927],
                [0.5898729745568927, 1.0],
            ]
        ),
        5.131610719537965,
    ),
    (
        60,
        np.array(
            [
                [1.0, 0.45, -0.2],
                [0.45, 1.0, 0.25],
                [-0.2, 0.25, 1.0],
            ]
        ),
        9102,
        np.array(
            [
                [1.0, 0.3038362782649849, -0.3163402952211118],
                [0.3038362782649849, 1.0, 0.3832673419408427],
                [-0.3163402952211118, 0.3832673419408427, 1.0],
            ]
        ),
        15.979742157599304,
    ),
)


_STUDENT_BASELINES = (
    (
        80,
        np.array([[1.0, 0.6], [0.6, 1.0]]),
        4.5,
        9201,
        np.array(
            [
                [1.0, 0.6842571628866113],
                [0.6842571628866113, 1.0],
            ]
        ),
        4.413178961400477,
        24.3976351496382,
    ),
    (
        90,
        np.array(
            [
                [1.0, 0.4, -0.15],
                [0.4, 1.0, 0.2],
                [-0.15, 0.2, 1.0],
            ]
        ),
        5.5,
        9202,
        np.array(
            [
                [1.0, 0.19907967446132763, -0.17210296345587098],
                [0.19907967446132763, 1.0, 0.25180366789575515],
                [-0.17210296345587098, 0.25180366789575515, 1.0],
            ]
        ),
        7.042094001235549,
        6.8021981325736345,
    ),
)


def _correlated_normal_scores(n, correlation, seed):
    rng = np.random.default_rng(seed)
    factor = np.linalg.cholesky(np.asarray(correlation, dtype=np.float64))
    return rng.standard_normal((n, len(correlation))) @ factor.T


def _correlated_student_scores(n, correlation, df, seed):
    rng = np.random.default_rng(seed)
    factor = np.linalg.cholesky(np.asarray(correlation, dtype=np.float64))
    normal = rng.standard_normal((n, len(correlation))) @ factor.T
    scale = np.sqrt(rng.chisquare(df, size=n) / df)
    return normal / scale[:, None]


def _gaussian_copula_log_likelihood(u, rho):
    scores = norm.ppf(u)
    correlation = np.array([[1.0, rho], [rho, 1.0]])
    rows = (
        multivariate_normal.logpdf(
            scores, mean=np.zeros(2), cov=correlation)
        - np.sum(norm.logpdf(scores), axis=1)
    )
    return float(np.sum(rows))


def _student_copula_log_likelihood(u, rho, df):
    scores = t_dist.ppf(u, df=df)
    correlation = np.array([[1.0, rho], [rho, 1.0]])
    rows = (
        multivariate_t.logpdf(
            scores, loc=np.zeros(2), shape=correlation, df=df)
        - np.sum(t_dist.logpdf(scores, df=df), axis=1)
    )
    return float(np.sum(rows))


@pytest.mark.parametrize(
    "n,source_correlation,seed,expected_correlation,expected_log_likelihood",
    _GAUSSIAN_BASELINES,
    ids=["bivariate", "trivariate"],
)
def test_gaussian_score_plugin_regression_baseline(
        n, source_correlation, seed, expected_correlation,
        expected_log_likelihood):
    observations = pobs(
        _correlated_normal_scores(n, source_correlation, seed))
    model = GaussianCopula()

    result = model.fit(observations)

    np.testing.assert_allclose(
        model.corr, expected_correlation, rtol=0.0, atol=2e-14)
    assert result.log_likelihood == pytest.approx(
        expected_log_likelihood, rel=0.0, abs=2e-12)
    assert result.parameter_count == len(source_correlation) * (
        len(source_correlation) - 1) // 2
    assert result.success is True
    assert result.nfev == 0
    assert result.diagnostics["estimator"] == "gaussian_score_correlation"
    np.testing.assert_array_equal(
        result.diagnostics["corr_matrix"], result.correlation_matrix)


@pytest.mark.parametrize(
    "n,source_correlation,source_df,seed,expected_correlation,expected_df,"
    "expected_log_likelihood",
    _STUDENT_BASELINES,
    ids=["bivariate", "trivariate"],
)
def test_student_kendall_plugin_regression_baseline(
        n, source_correlation, source_df, seed, expected_correlation,
        expected_df, expected_log_likelihood):
    observations = pobs(_correlated_student_scores(
        n, source_correlation, source_df, seed))
    model = StudentCopula()

    result = model.fit(observations)

    np.testing.assert_allclose(
        model.shape, expected_correlation, rtol=0.0, atol=2e-14)
    assert model.df == pytest.approx(expected_df, rel=0.0, abs=2e-8)
    assert result.log_likelihood == pytest.approx(
        expected_log_likelihood, rel=0.0, abs=2e-9)
    assert result.parameter_count == 1 + len(source_correlation) * (
        len(source_correlation) - 1) // 2
    assert result.success is True
    assert result.nfev > 0
    assert result.diagnostics["optimizer_gradient"] == "analytical"
    assert result.diagnostics["df_gradient"] == "analytical"
    assert result.diagnostics["corr_initialization_source"] == "kendall"
    assert result.diagnostics["corr_nonfinite_kendall_pairs"] == ()
    np.testing.assert_array_equal(
        result.diagnostics["corr_matrix"], result.correlation_matrix)


@pytest.mark.parametrize("kind", ["gaussian", "student"])
def test_static_plugin_fit_is_permutation_invariant(kind):
    correlation = np.array(
        [
            [1.0, 0.45, -0.2],
            [0.45, 1.0, 0.25],
            [-0.2, 0.25, 1.0],
        ]
    )
    if kind == "gaussian":
        observations = pobs(
            _correlated_normal_scores(70, correlation, 9301))
        original = GaussianCopula()
        permuted = GaussianCopula()
    else:
        observations = pobs(
            _correlated_student_scores(70, correlation, 5.0, 9302))
        original = StudentCopula()
        permuted = StudentCopula()
    permutation = np.array([2, 0, 1])

    original_result = original.fit(observations)
    permuted_result = permuted.fit(observations[:, permutation])
    original_correlation = (
        original.corr if kind == "gaussian" else original.shape)
    permuted_correlation = (
        permuted.corr if kind == "gaussian" else permuted.shape)

    np.testing.assert_allclose(
        permuted_correlation,
        original_correlation[np.ix_(permutation, permutation)],
        rtol=0.0,
        atol=3e-14,
    )
    assert permuted_result.log_likelihood == pytest.approx(
        original_result.log_likelihood, rel=0.0, abs=3e-11)
    if kind == "student":
        assert permuted.df == pytest.approx(
            original.df, rel=0.0, abs=3e-11)


def test_gaussian_score_plugin_point_differs_from_joint_likelihood_optimum():
    source_correlation = np.array([[1.0, 0.7], [0.7, 1.0]])
    scores = _correlated_normal_scores(18, source_correlation, 26)
    observations = norm.cdf(scores)
    model = GaussianCopula()

    result = model.fit(observations)
    oracle = minimize_scalar(
        lambda rho: -_gaussian_copula_log_likelihood(observations, rho),
        bounds=(-0.999, 0.999),
        method="bounded",
        options={"xatol": 1e-13},
    )

    assert oracle.success
    assert model.corr[0, 1] == pytest.approx(
        0.8695276496015485, rel=0.0, abs=2e-14)
    assert oracle.x == pytest.approx(
        0.7746608437246573, rel=0.0, abs=2e-8)
    assert -oracle.fun - result.log_likelihood > 2.0
    assert result.log_likelihood == pytest.approx(
        12.295004659674941, rel=0.0, abs=2e-12)
    assert -oracle.fun == pytest.approx(
        14.305284005954745, rel=0.0, abs=2e-9)


def test_student_kendall_plugin_point_has_nonzero_correlation_score():
    source_correlation = np.array([[1.0, 0.65], [0.65, 1.0]])
    source_df = 4.5
    scores = _correlated_student_scores(
        35, source_correlation, source_df, 22)
    observations = t_dist.cdf(scores, df=source_df)
    model = StudentCopula()

    result = model.fit(observations)
    oracle = minimize_scalar(
        lambda rho: -_student_copula_log_likelihood(
            observations, rho, model.df),
        bounds=(-0.999, 0.999),
        method="bounded",
        options={"xatol": 1e-13},
    )
    step = 1e-6
    plugin_rho = float(model.shape[0, 1])
    plugin_score = (
        _student_copula_log_likelihood(
            observations, plugin_rho + step, model.df)
        - _student_copula_log_likelihood(
            observations, plugin_rho - step, model.df)
    ) / (2.0 * step)

    assert oracle.success
    assert plugin_rho == pytest.approx(
        0.8360247083564342, rel=0.0, abs=2e-14)
    assert model.df == pytest.approx(
        3.9728820690078828, rel=0.0, abs=2e-8)
    assert oracle.x == pytest.approx(
        0.7300581604906521, rel=0.0, abs=2e-8)
    assert -oracle.fun - result.log_likelihood > 1.75
    assert abs(plugin_score) > 40.0
    assert plugin_score == pytest.approx(
        -43.94064162838163, rel=0.0, abs=2e-5)
