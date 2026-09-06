"""Static Student validation and accepted-result sampling contracts."""

from copy import deepcopy
from dataclasses import replace
import warnings

import numpy as np
import pytest
from scipy.stats import multivariate_t, norm, t

from pyscarcopula import StudentCopula
from pyscarcopula.copula.multivariate.factor_correlation import (
    PreparedFactorCorrelation,
)


MODES = ("fixed", "shrinkage", "cholesky", "factor", "factor-joint")
CORRELATION = np.array([
    [1.0, 0.43, 0.17, -0.12], [0.43, 1.0, 0.21, 0.08],
    [0.17, 0.21, 1.0, 0.3], [-0.12, 0.08, 0.3, 1.0],
])
LOADINGS = np.array([[0.52], [0.37], [-0.29], [0.41]])
DATA = norm.cdf(np.random.default_rng(317).multivariate_normal(
    np.zeros(4), CORRELATION, size=160))


def _model(mode, **overrides):
    options = dict(d=4, corr_mode="factor" if mode.startswith("factor") else mode)
    if mode.startswith("factor"):
        options.update(
            factor_rank=1, factor_loadings=LOADINGS,
            factor_estimation="joint" if mode == "factor-joint" else "two-stage")
    else:
        options["R"] = CORRELATION
        if mode != "fixed":
            options["corr_base"] = 0.6 * CORRELATION + 0.4 * np.eye(4)
    options.update(overrides)
    return StudentCopula(**options)


@pytest.fixture(scope="module")
def fitted_models():
    models = {}
    for mode in MODES:
        model = _model(mode)
        result = model.fit(DATA, maxiter=500, maxfun=3000)
        assert result.success, (mode, result.message)
        models[mode] = model
    return models


def _draw(model, operation, given, *, n=25):
    options = {} if given is None else {"given": given}
    return getattr(model, operation)(
        n, rng=np.random.default_rng(951), n_threads=2, **options)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("operation,given", [
    ("sample", None), ("predict", None), ("sample_conditional", {}),
    ("sample_conditional", {0: 0.4}), ("predict", {0: 0.4}),
])
def test_student_sampling_uses_accepted_result_after_property_changes(
        fitted_models, mode, operation, given):
    model = deepcopy(fitted_models[mode])
    expected = _draw(model, operation, given)
    accepted = model.fit_result
    model.df = 3.1
    if not mode.startswith("factor"):
        model.shape = np.eye(4)
    np.testing.assert_array_equal(_draw(model, operation, given), expected)
    assert model.fit_result is accepted


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("given", [None, {}, {0: 0.4}])
def test_student_result_survives_cleared_mutable_parameters(
        fitted_models, mode, given):
    model = deepcopy(fitted_models[mode])
    expected = _draw(model, "predict", given)
    model.df = None
    model.shape = None
    np.testing.assert_array_equal(_draw(model, "predict", given), expected)
    assert _draw(model, "predict", given, n=0).shape == (0, 4)


@pytest.mark.parametrize("mode", ["factor", "factor-joint"])
@pytest.mark.parametrize("given", [None, {}, {0: 0.4}])
def test_student_factor_result_loadings_are_owned_without_dense_materialization(
        fitted_models, mode, given, monkeypatch):
    model = deepcopy(fitted_models[mode])
    old_result = model.fit_result
    expected = _draw(model, "predict", given)
    new_data = DATA[:, ::-1].copy()
    assert model.fit(new_data, maxiter=500, maxfun=3000).success
    model.fit_result = old_result
    model._set_factor_loadings(0.25 * LOADINGS)
    model._factor_uniqueness_min = 0.99

    def no_dense(*args, **kwargs):
        raise AssertionError("factor result sampling must stay compact")

    monkeypatch.setattr(PreparedFactorCorrelation, "to_dense", no_dense)
    np.testing.assert_array_equal(_draw(model, "predict", given), expected)
    assert model.fit_result is old_result
    np.testing.assert_array_equal(model.factor_loadings_, 0.25 * LOADINGS)


@pytest.mark.parametrize("mode", MODES)
def test_student_result_replacement_controls_scalar_in_every_sampling_branch(
        fitted_models, mode):
    model = deepcopy(fitted_models[mode])
    result = replace(model.fit_result, copula_param=4.5)
    model.fit_result = result
    reference = _model(mode)
    if mode.startswith("factor"):
        reference._set_factor_loadings(result.model_parameters["factor_loadings"])
    else:
        reference.shape = result.correlation_matrix
    reference.df = 4.5
    for given in (None, {}, {1: 0.3}):
        np.testing.assert_array_equal(
            _draw(model, "predict", given), _draw(reference, "predict", given))


@pytest.mark.parametrize("mode", MODES)
def test_student_likelihood_keeps_current_parameter_semantics(fitted_models, mode):
    model = deepcopy(fitted_models[mode])
    accepted = model.fit_result
    model.df = 5.7
    if not mode.startswith("factor"):
        model.shape = 0.4 * CORRELATION + 0.6 * np.eye(4)
    correlation = model.to_correlation_matrix()
    latent = t.ppf(DATA[:20], model.df)
    expected = multivariate_t.logpdf(
        latent, shape=correlation, df=model.df) - t.logpdf(latent, model.df).sum(axis=1)
    np.testing.assert_allclose(model.log_pdf_rows(DATA[:20]), expected, atol=2e-8)
    assert model.log_likelihood(DATA[:20]) == pytest.approx(expected.sum(), abs=2e-7)
    assert model.fit_result is accepted


@pytest.mark.parametrize("invalid", [
    2 * CORRELATION,
    CORRELATION + np.diag(np.full(3, 0.2), k=1),
    np.eye(3), np.ones((4, 4)), np.full((4, 4), np.nan),
    np.full((4, 4), np.inf), np.ones(4), np.ones((4, 3)),
])
def test_student_shape_rejects_invalid_correlation_without_changing_state(invalid):
    model = StudentCopula(d=4)
    model.shape = CORRELATION
    model.df = 6.7
    expected = _draw(model, "sample", None)
    with pytest.raises((TypeError, ValueError, FloatingPointError)):
        model.shape = invalid
    np.testing.assert_array_equal(model.shape, CORRELATION)
    assert model.dimension == 4
    np.testing.assert_array_equal(_draw(model, "sample", None), expected)


def test_student_manual_shape_infers_dimension_and_copies_valid_matrix():
    model = StudentCopula()
    value = CORRELATION.copy()
    model.shape = value
    model.df = 6.7
    value[...] = 0.0
    assert model.dimension == 4
    np.testing.assert_array_equal(model.shape, CORRELATION)
    for given in (None, {}, {0: 0.4}):
        assert np.all(np.isfinite(_draw(model, "predict", given)))
    model.shape = None
    assert model.shape is None


@pytest.mark.parametrize("dimension", [0, 1])
def test_student_shape_cannot_infer_dimension_below_two(dimension):
    model = StudentCopula()
    with pytest.raises(ValueError, match="dimension"):
        model.shape = np.eye(dimension)
    assert model.dimension is None and model.shape is None


@pytest.mark.parametrize("imaginary", [0.0, 0.25, np.nan, np.inf])
@pytest.mark.parametrize("representation", ["complex", "object"])
def test_student_shape_rejects_complex_before_cast(imaginary, representation):
    model = StudentCopula(d=4)
    model.shape = CORRELATION
    value = CORRELATION.astype(complex)
    value[0, 1] = complex(value[0, 1].real, imaginary)
    if representation == "object":
        value = value.astype(object)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(TypeError, match="real|complex"):
            model.shape = value
    assert not caught
    np.testing.assert_array_equal(model.shape, CORRELATION)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name,value", [
    ("corr_shrinkage_init", 0.73), ("factor_uniqueness_min", 1e-5),
    ("factor_joint_penalty", 2e-6), ("factor_joint_condition_max", 1e6),
])
def test_student_constructor_rejects_lossy_complex_controls(mode, name, value):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(TypeError, match=name):
            _model(mode, **{name: np.complex128(complex(value, 0.2))})
    assert not caught
    assert getattr(_model(mode, **{name: np.array(value)}), "_" + name) == value


@pytest.mark.parametrize("mode", ["factor", "factor-joint"])
@pytest.mark.parametrize("representation", ["complex", "object"])
def test_student_constructor_rejects_complex_loadings(mode, representation):
    value = LOADINGS.astype(complex) + 0.2j
    if representation == "object":
        value = value.astype(object)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(TypeError, match="factor_loadings"):
            _model(mode, factor_loadings=value)
    assert not caught


@pytest.mark.parametrize("name", ["R", "corr_base"])
def test_student_constructor_correlation_rejects_complex_before_preprocessing(name):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(TypeError, match=name):
            _model("shrinkage", **{name: CORRELATION.astype(complex) + 0.2j})
    assert not caught


@pytest.mark.parametrize("mode", MODES)
def test_student_dense_materialization_limits_apply_before_copy(fitted_models, mode):
    model = deepcopy(fitted_models[mode])
    expected = model.to_correlation_matrix()
    for options in ({"max_dimension": 3}, {"memory_budget_bytes": 127}):
        with pytest.raises(MemoryError):
            model.to_correlation_matrix(**options)
    actual = model.to_correlation_matrix(max_dimension=4, memory_budget_bytes=128)
    np.testing.assert_array_equal(actual, expected)
    actual[...] = 0
    np.testing.assert_array_equal(model.to_correlation_matrix(), expected)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("given", [None, {}, {0: 0.4}])
def test_student_sampling_snapshot_keeps_validation_before_rng(fitted_models, mode, given):
    model = deepcopy(fitted_models[mode])
    for options in ({"n": -1}, {"n": True}, {"n": 0, "n_threads": 0}):
        rng = np.random.default_rng(721)
        before = deepcopy(rng.bit_generator.state)
        with pytest.raises((TypeError, ValueError)):
            model.predict(given=given, rng=rng, **options)
        assert rng.bit_generator.state == before
