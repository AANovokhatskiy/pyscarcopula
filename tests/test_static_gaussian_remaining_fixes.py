"""Gaussian initialization, real parameter, and dense output contracts."""

from types import SimpleNamespace
import warnings

import numpy as np
import pytest
from scipy.stats import multivariate_normal, norm, rankdata

from pyscarcopula import FactorCorrelation, GaussianCopula
from pyscarcopula.copula.multivariate.factor_correlation import (
    _validate_dense_materialization,
)


MODES = ("fixed", "shrinkage", "cholesky", "factor")
CORRELATION = np.array([
    [1.0, 0.4, 0.15, -0.1],
    [0.4, 1.0, 0.2, 0.05],
    [0.15, 0.2, 1.0, 0.3],
    [-0.1, 0.05, 0.3, 1.0],
])
LOADINGS = np.array([[0.52], [0.37], [-0.29], [0.41]])


def _observations(seed=731):
    draws = np.random.default_rng(seed).multivariate_normal(
        np.zeros(4), CORRELATION, size=120)
    return norm.cdf(draws)


def _model(mode, **kwargs):
    options = {"d": 4, "corr_mode": mode}
    if mode == "factor":
        options.update(factor_rank=1, factor_loadings=LOADINGS)
    else:
        options["R"] = CORRELATION
    options.update(kwargs)
    return GaussianCopula(**options)


@pytest.mark.parametrize("seed", [731, 947])
@pytest.mark.parametrize("operation", ["initialize_factor", "fit"])
def test_factor_raw_initialization_matches_independent_rank_transform(
        seed, operation):
    raw = norm.ppf(_observations(seed)) * np.array([2., 3., 4., 5.]) + 7.
    snapshot = raw.copy()
    ranked = rankdata(raw, axis=0) / (len(raw) + 1)
    actual = GaussianCopula(d=4, corr_mode="factor", factor_rank=1)
    expected = GaussianCopula(d=4, corr_mode="factor", factor_rank=1)

    getattr(actual, operation)(raw, to_pobs=True)
    getattr(expected, operation)(ranked)

    np.testing.assert_array_equal(raw, snapshot)
    np.testing.assert_allclose(
        actual.to_correlation_matrix(), expected.to_correlation_matrix(),
        rtol=0., atol=2e-14)
    scores = norm.ppf(ranked)
    independent_rows = (
        multivariate_normal.logpdf(
            scores, cov=actual.to_correlation_matrix())
        - norm.logpdf(scores).sum(axis=1)
    )
    np.testing.assert_allclose(
        actual.log_pdf_rows(ranked), independent_rows, rtol=0., atol=2e-8)
    if operation == "fit":
        np.testing.assert_array_equal(actual._last_u, ranked)
    else:
        assert actual.fit_result is None


@pytest.mark.parametrize("data", [
    np.full((10, 4), np.nan), np.full((10, 4), np.inf),
    np.zeros((0, 4)), np.zeros(4), np.zeros((4, 4, 1)),
    np.zeros((4, 1)), np.ones((10, 4), dtype=complex),
])
@pytest.mark.parametrize("initialized", [False, True])
def test_factor_raw_initialization_rejects_bad_data_without_replacing_state(
        data, initialized):
    model = GaussianCopula(d=4, corr_mode="factor", factor_rank=1)
    if initialized:
        model.initialize_factor(_observations())
    before = model.__dict__.copy()
    with pytest.raises((TypeError, ValueError)):
        model.initialize_factor(data, to_pobs=True)
    for key, value in before.items():
        assert model.__dict__[key] is value


@pytest.mark.parametrize("to_pobs", [False, True])
def test_factor_initialization_keeps_explicit_dimension_contract(to_pobs):
    model = GaussianCopula(d=3, corr_mode="factor", factor_rank=1)
    with pytest.raises(ValueError, match="3 columns"):
        model.initialize_factor(_observations(), to_pobs=to_pobs)
    assert model.factor_loadings_ is None


def test_factor_initialization_requires_rank_transform_for_raw_data():
    model = GaussianCopula(d=4, corr_mode="factor", factor_rank=1)
    with pytest.raises(ValueError, match="to_pobs=True"):
        model.initialize_factor(norm.ppf(_observations()) + 8.)
    assert model.factor_loadings_ is None


@pytest.fixture(scope="module", params=MODES)
def fitted_model(request):
    model = _model(request.param)
    assert model.fit(_observations()).success
    return model


@pytest.mark.parametrize("options", [
    {"max_dimension": 3}, {"memory_budget_bytes": 127},
    {"memory_budget_bytes": 0},
])
def test_dense_correlation_rejects_insufficient_output_limits(
        fitted_model, options):
    with pytest.raises(MemoryError):
        fitted_model.to_correlation_matrix(**options)


@pytest.mark.parametrize("options,error", [
    ({"max_dimension": True}, TypeError),
    ({"max_dimension": 4.0}, TypeError),
    ({"max_dimension": None}, TypeError),
    ({"max_dimension": 0}, ValueError),
    ({"memory_budget_bytes": True}, TypeError),
    ({"memory_budget_bytes": 128.0}, TypeError),
    ({"memory_budget_bytes": -1}, ValueError),
])
def test_dense_correlation_limit_types_are_consistent(
        fitted_model, options, error):
    with pytest.raises(error):
        fitted_model.to_correlation_matrix(**options)


def test_exact_dense_output_limits_return_an_owned_correlation(fitted_model):
    model = fitted_model
    value = model.to_correlation_matrix(
        max_dimension=np.int64(4), memory_budget_bytes=np.int64(128))
    assert value.nbytes == 128 and value.shape == (4, 4)
    if model.corr_mode == "factor":
        expected = LOADINGS @ LOADINGS.T + np.diag(
            1. - np.sum(LOADINGS ** 2, axis=1))
    else:
        expected = model.corr.copy()
    np.testing.assert_allclose(value, expected, rtol=0., atol=2e-15)
    value[:] = 0.
    np.testing.assert_allclose(
        model.to_correlation_matrix(), expected, rtol=0., atol=2e-15)


@pytest.mark.parametrize("prepared", [False, True])
def test_factor_operator_reuses_dense_output_limits(prepared):
    factor = FactorCorrelation(LOADINGS)
    owner = factor.prepare() if prepared else factor
    for options in ({"max_dimension": 3}, {"memory_budget_bytes": 127}):
        with pytest.raises(MemoryError):
            owner.to_dense(**options)
    dense = owner.to_dense(max_dimension=4, memory_budget_bytes=128)
    np.testing.assert_allclose(np.diag(dense), 1., rtol=0., atol=1e-15)


def test_dense_materialization_rejects_overflow_before_allocation():
    dimension = int(np.iinfo(np.intp).max)
    with pytest.raises(MemoryError, match="too large"):
        _validate_dense_materialization(
            dimension, max_dimension=dimension)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("parameter", [
    "corr_shrinkage_init", "factor_uniqueness_min",
])
@pytest.mark.parametrize("value", [
    complex(.2, 0.), np.complex64(.2 + .3j),
    np.complex128(complex(.2, np.nan)),
    np.array(complex(.2, np.inf)),
    np.array(complex(.2, 0.), dtype=object),
])
def test_gaussian_constructor_rejects_complex_scalar_controls(
        mode, parameter, value):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(TypeError, match="real values"):
            _model(mode, **{parameter: value})
    assert not caught


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("parameter", [
    "corr_shrinkage_init", "factor_uniqueness_min",
])
@pytest.mark.parametrize("value", [np.float32(.2), np.array(.2)])
def test_gaussian_constructor_accepts_real_scalar_controls(
        mode, parameter, value):
    model = _model(mode, **{parameter: value})
    assert getattr(model, "_" + parameter) == float(value)


@pytest.mark.parametrize("parameter", [
    "corr_shrinkage_init", "factor_uniqueness_min",
])
@pytest.mark.parametrize("value", [0., 1., np.nan, np.inf, -.5])
def test_gaussian_constructor_preserves_scalar_domains(parameter, value):
    with pytest.raises(ValueError):
        _model("factor", **{parameter: value})


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128, object])
@pytest.mark.parametrize("imaginary", [0., .3, np.nan, np.inf])
def test_gaussian_constructor_rejects_complex_factor_loadings(dtype, imaginary):
    value = (LOADINGS.astype(complex) + complex(0., imaginary)).astype(dtype)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(TypeError, match="real values"):
            _model("factor", factor_loadings=value)
    assert not caught


def test_gaussian_real_factor_loadings_remain_owned():
    values = LOADINGS.astype(np.float32)
    model = _model("factor", factor_loadings=values)
    expected = values.astype(np.float64)
    values[:] = 0.
    np.testing.assert_array_equal(model.factor_loadings_, expected)


@pytest.mark.parametrize("imaginary", [0., .3, np.nan, np.inf])
def test_gaussian_factor_fit_rejects_complex_failure_control_before_cast(
        imaginary):
    model = _model("factor")
    config = SimpleNamespace(
        n_threads=1, fail_value=np.complex128(complex(1e10, imaginary)))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(TypeError, match="real values"):
            model.fit(_observations(), config=config)
    assert not caught
    assert model.fit_result is None
