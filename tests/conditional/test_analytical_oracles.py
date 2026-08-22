"""Self-tests and production checks for independent analytical oracles."""

from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest
from scipy.stats import multivariate_normal, multivariate_t, norm
from scipy.stats import t as t_dist

from pyscarcopula import BivariateGaussianCopula, IndependentCopula
from pyscarcopula.copula.multivariate.conditional import (
    sample_gaussian_copula_conditional,
    sample_student_conditional,
)

from . import _analytical_oracles as analytical_oracles_module
from . import _bivariate_oracles as bivariate_oracles_module
from . import _scar_tm_ou_oracle as scar_tm_ou_oracle_module
from . import _multivariate_scar_oracle as multivariate_scar_oracle_module
from . import _statistical_assertions as statistical_assertions_module
from ._analytical_oracles import (
    gaussian_conditional_cdf,
    gaussian_conditional_inverse,
    gaussian_conditional_parameters,
    gaussian_copula_log_density,
    gaussian_copula_parameter_from_state,
    student_conditional_parameters,
)
from ._runtime import build_runtime
from ._multivariate_scar_oracle import (
    equicorr_gaussian_log_density,
    equicorr_parameter_from_state,
    equicorrelation_matrix,
    student_copula_log_density,
    student_df_parameter_from_state,
)
from ._statistical_assertions import (
    assert_covariance_with_whitening,
    assert_mean_with_mc_error,
    assert_uniform_pit,
)


R4 = np.array([
    [1.00, 0.55, 0.20, 0.10],
    [0.55, 1.00, 0.35, 0.15],
    [0.20, 0.35, 1.00, 0.45],
    [0.10, 0.15, 0.45, 1.00],
])
GIVEN = {0: 0.2, 2: 0.8}


def test_reference_oracle_modules_do_not_import_production_package():
    modules = (
        analytical_oracles_module,
        bivariate_oracles_module,
        scar_tm_ou_oracle_module,
        multivariate_scar_oracle_module,
        statistical_assertions_module,
    )
    for module in modules:
        tree = ast.parse(inspect.getsource(module))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        assert not any(
            name == "pyscarcopula" or name.startswith("pyscarcopula.")
            for name in imported
        ), f"{module.__name__} imports production package: {imported}"


def test_gaussian_conditional_density_factorization_against_scipy():
    oracle = gaussian_conditional_parameters(R4, GIVEN)
    z_given = norm.ppf([GIVEN[int(i)] for i in oracle.given_indices])
    x_free = np.array([-0.4, 0.7])
    full = np.empty(4)
    full[oracle.given_indices] = z_given
    full[oracle.free_indices] = x_free

    ratio = (
        multivariate_normal.pdf(full, mean=np.zeros(4), cov=R4)
        / multivariate_normal.pdf(
            z_given,
            mean=np.zeros(len(z_given)),
            cov=R4[np.ix_(oracle.given_indices, oracle.given_indices)],
        )
    )
    expected = multivariate_normal.pdf(
        x_free, mean=oracle.mean, cov=oracle.covariance
    )
    assert ratio == pytest.approx(expected, rel=2e-12)


def test_student_conditional_density_factorization_against_scipy():
    df = 6.5
    oracle = student_conditional_parameters(R4, df, GIVEN)
    x_given = t_dist.ppf(
        [GIVEN[int(i)] for i in oracle.given_indices], df=df
    )
    x_free = np.array([-0.4, 0.7])
    full = np.empty(4)
    full[oracle.given_indices] = x_given
    full[oracle.free_indices] = x_free

    ratio = (
        multivariate_t.pdf(full, loc=np.zeros(4), shape=R4, df=df)
        / multivariate_t.pdf(
            x_given,
            loc=np.zeros(len(x_given)),
            shape=R4[np.ix_(oracle.given_indices, oracle.given_indices)],
            df=df,
        )
    )
    expected = multivariate_t.pdf(
        x_free,
        loc=oracle.location,
        shape=oracle.shape,
        df=oracle.conditional_df,
    )
    assert ratio == pytest.approx(expected, rel=3e-12)


def test_gaussian_copula_density_against_scipy_density_ratio():
    u = np.array([0.17, 0.83])
    rho = 0.63
    z = norm.ppf(u)
    ratio = (
        multivariate_normal.pdf(z, cov=[[1.0, rho], [rho, 1.0]])
        / np.prod(norm.pdf(z))
    )
    observed = np.exp(gaussian_copula_log_density(u, rho))
    assert observed == pytest.approx(ratio, rel=2e-13)


def test_gaussian_scar_link_is_independent_and_matches_public_transform():
    states = np.linspace(-8.0, 8.0, 41)
    production = BivariateGaussianCopula().transform(states)
    reference = gaussian_copula_parameter_from_state(states)
    np.testing.assert_allclose(production, reference, rtol=0.0, atol=2e-15)


def test_gaussian_conditional_cdf_inverse_roundtrip_closed_form():
    q = np.linspace(0.01, 0.99, 99)
    for given in (0.02, 0.3, 0.8, 0.98):
        for rho in (-0.8, -0.2, 0.0, 0.65, 0.9):
            sample = gaussian_conditional_inverse(q, given, rho)
            recovered = gaussian_conditional_cdf(sample, given, rho)
            np.testing.assert_allclose(recovered, q, rtol=0.0, atol=2e-12)


def test_production_gaussian_conditional_matches_full_mvn_oracle():
    n = 20_000
    samples = sample_gaussian_copula_conditional(
        n, R4, GIVEN, rng=np.random.default_rng(20260910)
    )
    oracle = gaussian_conditional_parameters(R4, GIVEN)
    latent = norm.ppf(samples[:, oracle.free_indices])
    assert_mean_with_mc_error(latent, oracle.mean, oracle.covariance)
    assert_covariance_with_whitening(latent, oracle.mean, oracle.covariance)
    for index, value in GIVEN.items():
        np.testing.assert_array_equal(samples[:, index], value)


def test_production_student_conditional_matches_full_t_oracle():
    n = 28_000
    df = 6.5
    samples = sample_student_conditional(
        n, R4, df, GIVEN, rng=np.random.default_rng(20260911)
    )
    oracle = student_conditional_parameters(R4, df, GIVEN)
    latent = oracle.latent_from_copula(samples)
    assert_mean_with_mc_error(latent, oracle.location, oracle.covariance)
    assert_covariance_with_whitening(
        latent, oracle.location, oracle.covariance,
        sigma=9.0, numerical_floor=0.025,
    )
    for index, value in GIVEN.items():
        np.testing.assert_array_equal(samples[:, index], value)


@pytest.mark.parametrize("rho", [-0.75, -0.2, 0.0, 0.55, 0.88])
def test_bivariate_gaussian_native_h_and_inverse_match_closed_form(rho):
    copula = BivariateGaussianCopula()
    given = np.linspace(0.03, 0.97, 97)
    q = np.linspace(0.01, 0.99, 97)
    free = gaussian_conditional_inverse(q, given, rho)
    r = np.full(len(q), rho)
    observed_q = copula.h(free, given, r)
    observed_free = copula.h_inverse(q, given, r)
    np.testing.assert_allclose(observed_q, q, rtol=0.0, atol=5e-9)
    np.testing.assert_allclose(observed_free, free, rtol=0.0, atol=5e-9)


@pytest.mark.parametrize("given_index", [0, 1])
def test_bivariate_gaussian_conditional_sample_has_uniform_reference_pit(
        given_index):
    runtime = build_runtime("bivariate-gaussian")
    free_index = 1 - given_index
    samples = runtime.model.predict(
        18_000,
        given={given_index: 0.91},
        rng=np.random.default_rng(20260912),
    )
    pit = gaussian_conditional_cdf(
        samples[:, free_index], 0.91, 0.45
    )
    assert_uniform_pit(pit)


def test_independent_conditional_free_coordinate_is_uniform():
    runtime = build_runtime("bivariate-independent")
    assert isinstance(runtime.model, IndependentCopula)
    samples = runtime.model.predict(
        18_000,
        given={0: 0.17},
        rng=np.random.default_rng(20260913),
    )
    assert_uniform_pit(samples[:, 1])
    np.testing.assert_array_equal(samples[:, 0], 0.17)


def test_equicorr_multivariate_density_oracle_matches_scipy_ratio():
    dimension = 5
    rho = 0.37
    observation = np.array([0.12, 0.31, 0.56, 0.78, 0.91])
    latent = norm.ppf(observation)
    matrix = equicorrelation_matrix(dimension, rho)
    expected = np.log(multivariate_normal.pdf(latent, cov=matrix)) - np.sum(
        np.log(norm.pdf(latent))
    )
    observed = equicorr_gaussian_log_density(observation, np.array([rho]))[0]
    assert observed == pytest.approx(expected, rel=3e-13, abs=3e-13)


def test_student_multivariate_density_oracle_matches_scipy_ratio():
    observation = np.array([0.17, 0.39, 0.66, 0.88])
    parameters = np.array([3.5, 7.0, 30.0])
    for df, observed in zip(
            parameters, student_copula_log_density(observation, parameters, R4)):
        latent = t_dist.ppf(observation, df=df)
        expected = np.log(multivariate_t.pdf(latent, shape=R4, df=df)) - np.sum(
            np.log(t_dist.pdf(latent, df=df))
        )
        assert observed == pytest.approx(expected, rel=2e-12, abs=2e-12)


def test_multivariate_dynamic_links_match_public_transforms():
    from pyscarcopula import EquicorrGaussianCopula, StochasticStudentCopula

    states = np.linspace(-7.0, 7.0, 31)
    equicorr = EquicorrGaussianCopula(d=5)
    student = StochasticStudentCopula(d=4, R=R4)
    np.testing.assert_allclose(
        equicorr.transform(states),
        equicorr_parameter_from_state(states, dimension=5),
        rtol=0.0,
        atol=2e-15,
    )
    np.testing.assert_allclose(
        student.transform(states),
        student_df_parameter_from_state(states),
        rtol=2e-15,
        atol=2e-15,
    )
