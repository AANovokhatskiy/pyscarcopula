"""Conditional short paths and Rosenblatt adapters validate supplied inputs."""

import numpy as np
import pytest
from scipy.stats import norm

from pyscarcopula import FactorCorrelation
from pyscarcopula.copula.multivariate import conditional
from pyscarcopula import stattests


HELPERS = ("gaussian", "gaussian_copula", "student", "factor_gaussian", "factor_student")


@pytest.fixture
def correlation():
    return FactorCorrelation(np.array([[.2], [.3], [.4]]))


def _arguments(name, correlation, **options):
    kwargs = dict(n=4, given={0: .2, 1: .3, 2: .4}, n_threads=1)
    if name == "gaussian":
        kwargs.update(d=3, rho=.2)
    elif name == "gaussian_copula":
        kwargs["R"] = correlation.to_dense()
    elif name == "student":
        kwargs.update(R_path=correlation.to_dense(), df=5.)
    elif name == "factor_gaussian":
        kwargs["correlation"] = correlation.prepare()
    else:
        kwargs.update(correlation=correlation.prepare(), df=5.)
    kwargs.update(options)
    return kwargs


def _sample(name, **kwargs):
    return getattr(conditional, f"sample_{name}_conditional")(**kwargs)


@pytest.mark.parametrize("name", HELPERS)
@pytest.mark.parametrize("n_threads", [0, -1, 1.8, True])
def test_all_given_rejects_invalid_thread_count(name, n_threads, correlation):
    kwargs = _arguments(name, correlation, n_threads=n_threads)
    with pytest.raises((TypeError, ValueError), match="n_threads"):
        _sample(name, **kwargs)


@pytest.mark.parametrize("name", HELPERS)
@pytest.mark.parametrize("n", [-1, 1.8, True])
def test_all_given_rejects_invalid_sample_count(name, n, correlation):
    with pytest.raises((TypeError, ValueError), match="n"):
        _sample(name, **_arguments(name, correlation, n=n))


@pytest.mark.parametrize("name", ["student", "factor_student"])
@pytest.mark.parametrize("df", [2., -3., np.nan, np.inf, np.complex128(5 + .1j)])
@pytest.mark.parametrize("n", [0, 4])
def test_all_given_validates_student_df_before_empty_broadcast(name, df, n, correlation):
    with pytest.raises((TypeError, ValueError), match="df"):
        _sample(name, **_arguments(name, correlation, n=n, df=df))


@pytest.mark.parametrize("name", ["gaussian_copula", "student"])
@pytest.mark.parametrize("matrix", [
    np.zeros((3, 3)), np.zeros((3, 2)), np.full((3, 3), np.nan),
    np.eye(3).astype(complex) + .1j,
])
def test_all_given_validates_dense_correlation(name, matrix, correlation):
    field = "R" if name == "gaussian_copula" else "R_path"
    with pytest.raises((TypeError, ValueError)):
        _sample(name, **_arguments(name, correlation, **{field: matrix}))


def test_all_given_student_checks_each_matrix_in_path(correlation):
    matrices = np.repeat(correlation.to_dense()[None], 4, axis=0)
    matrices[-1] = 0.
    with pytest.raises(ValueError):
        _sample("student", **_arguments("student", correlation, R_path=matrices))


@pytest.mark.parametrize("rho", [1., -.5, np.nan, np.complex128(.2 + .1j)])
def test_empty_all_given_equicorrelation_validates_scalar(rho, correlation):
    with pytest.raises((TypeError, ValueError)):
        _sample("gaussian", **_arguments("gaussian", correlation, n=0, rho=rho))


@pytest.mark.parametrize("name", HELPERS)
@pytest.mark.parametrize("n", [0, 4])
def test_all_given_preserves_values_and_does_not_consume_rng(name, n, correlation):
    rng = np.random.default_rng(163)
    control = np.random.default_rng(163)
    given = {0: np.nextafter(0., 1.), 1: .3, 2: np.nextafter(1., 0.)}
    result = _sample(name, **_arguments(
        name, correlation, n=n, given=given, rng=rng, n_threads=np.int64(2)))
    np.testing.assert_array_equal(result, np.tile(list(given.values()), (n, 1)))
    np.testing.assert_array_equal(rng.random(10), control.random(10))


@pytest.mark.parametrize("name", ["gaussian_copula", "student"])
def test_partial_given_rejects_complex_before_rng_draws(name, correlation):
    matrix = correlation.to_dense().astype(complex) + .1j
    field = "R" if name == "gaussian_copula" else "R_path"
    rng = np.random.default_rng(163)
    control = np.random.default_rng(163)
    with pytest.raises(TypeError, match="real"):
        _sample(name, **_arguments(
            name, correlation, given={0: .2}, rng=rng, **{field: matrix}))
    np.testing.assert_array_equal(rng.random(10), control.random(10))


@pytest.mark.parametrize("target", [
    "gaussian_correlation", "gaussian_observations",
    "factor_gaussian_observations", "factor_student_observations",
])
@pytest.mark.parametrize("representation", ["complex", "zero_imaginary", "object"])
def test_rosenblatt_rejects_complex_arrays(target, representation, correlation):
    observations = np.array([[.2, .4, .7], [.3, .8, .6], [.7, .6, .3], [.4, .2, .8]])
    values = correlation.to_dense() if target == "gaussian_correlation" else observations
    values = values.astype(complex)
    if representation != "zero_imaginary":
        values += .1j
    if representation == "object":
        values = values.astype(object)

    with pytest.raises(TypeError, match="real"):
        if target == "gaussian_correlation":
            stattests.gaussian_rosenblatt_transform(values, observations)
        elif target == "gaussian_observations":
            stattests.gaussian_rosenblatt_transform(correlation.to_dense(), values)
        elif target == "factor_gaussian_observations":
            stattests.factor_gaussian_rosenblatt_transform(correlation.prepare(), values)
        else:
            stattests.factor_student_rosenblatt_transform(correlation.prepare(), 5., values)


def test_gaussian_rosenblatt_real_strided_rows_match_conditional_normal_reference(correlation):
    source = np.linspace(.1, .9, 24).reshape(4, 6)
    u = source[:, ::2]
    u.setflags(write=False)
    matrix = correlation.to_dense()
    z = norm.ppf(u)
    expected = np.empty_like(u)
    expected[:, 0] = u[:, 0]
    for column in range(1, 3):
        weights = np.linalg.solve(matrix[:column, :column], matrix[:column, column])
        variance = 1 - matrix[column, :column] @ weights
        expected[:, column] = norm.cdf(
            (z[:, column] - z[:, :column] @ weights) / np.sqrt(variance))

    dense = stattests.gaussian_rosenblatt_transform(matrix, u)
    factor = stattests.factor_gaussian_rosenblatt_transform(correlation.prepare(), u)
    np.testing.assert_allclose(dense, expected, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(factor, expected, rtol=2e-13, atol=2e-13)
