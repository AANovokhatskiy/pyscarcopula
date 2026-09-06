"""Real-valued inputs and integer options for prepared correlation objects."""

import numpy as np
import pytest

from pyscarcopula import EquicorrPreparedData, FactorCorrelation, FactorStudentEvaluator
from pyscarcopula.copula.multivariate import corr_param
from pyscarcopula.copula.multivariate.correlation_policy import CorrelationPolicy
from pyscarcopula.copula.multivariate.factor_estimation import (
    FactorLoadingParameterization,
    estimate_factor_loadings,
)


@pytest.fixture
def prepared_inputs():
    u = np.array([[.2, .4, .7], [.3, .8, .6], [.7, .6, .3], [.4, .2, .8]])
    loadings = np.array([[.2], [.3], [.4]])
    factor = FactorCorrelation(loadings, uniqueness_min=.1)
    parameterization, raw = FactorLoadingParameterization.from_loadings(
        loadings, uniqueness_min=.1)
    return u, loadings, factor, factor.prepare(), parameterization, raw


def _complex(values, representation):
    values = np.asarray(values, dtype=np.complex128)
    if representation != "zero_imaginary":
        values += .13j
    return values.astype(object) if representation == "object" else values


@pytest.mark.parametrize("factory", [CorrelationPolicy, CorrelationPolicy.create])
@pytest.mark.parametrize("source", [None, "explicit-source"])
def test_policy_diagnostics_keep_selected_initialization_source(factory, source):
    preprocessing = corr_param.preprocess_correlation_matrix(
        np.eye(3), source="preprocessing-source")
    policy = factory(
        mode="fixed", estimator="supplied", dimension=3,
        supplied_correlation=np.eye(3), preprocessing=preprocessing,
        initialization_source=source)

    expected = "preprocessing-source" if source is None else source
    assert policy.initialization_source == expected
    assert policy.diagnostics()["corr_initialization_source"] == expected


@pytest.mark.parametrize("representation", ["complex", "zero_imaginary", "object"])
@pytest.mark.parametrize("constructor", [FactorCorrelation, FactorCorrelation.from_unconstrained])
def test_factor_loadings_reject_complex(constructor, representation, prepared_inputs):
    _, loadings, *_ = prepared_inputs
    with pytest.raises(TypeError, match="real"):
        constructor(_complex(loadings, representation))


@pytest.mark.parametrize("operation", ["matvec", "solve", "quadratic_form", "quadratic_forms"])
@pytest.mark.parametrize("representation", ["complex", "object"])
def test_factor_operator_rejects_complex_rows(operation, representation, prepared_inputs):
    u, _, _, prepared, *_ = prepared_inputs
    values = u[0] if operation == "quadratic_form" else u
    with pytest.raises(TypeError, match="real"):
        getattr(prepared, operation)(_complex(values, representation))


@pytest.mark.parametrize("target", ["factor_draws", "residual_draws"])
def test_factor_transform_rejects_complex_draws(target, prepared_inputs):
    _, _, _, prepared, *_ = prepared_inputs
    draws = {"factor_draws": np.ones((4, 1)), "residual_draws": np.ones((4, 3))}
    draws[target] = _complex(draws[target], "complex")
    with pytest.raises(TypeError, match="real"):
        prepared.transform_normal_draws(**draws)


@pytest.mark.parametrize("method", [
    "evaluate_grid", "log_pdf_and_dlog_ddf_grid", "pdf_and_grad_on_grid",
    "evaluate_grid_batches", "pdf_and_grad_on_grid_batches",
    "stochastic_pdf_and_gradient_grid",
])
@pytest.mark.parametrize("representation", ["complex", "object"])
def test_student_grid_rejects_complex_before_evaluation(method, representation, prepared_inputs):
    u, _, factor, *_ = prepared_inputs
    evaluator = FactorStudentEvaluator(factor, u)
    grid = _complex([5., 7.], representation)
    kwargs = {"offset": 3.} if method == "stochastic_pdf_and_gradient_grid" else {}
    with pytest.raises(TypeError, match="real"):
        result = getattr(evaluator, method)(grid, **kwargs)
        if method.endswith("batches"):
            list(result)


@pytest.mark.parametrize("offset", [np.complex128(3 + .1j), np.array(3 + 0j)])
def test_student_stochastic_grid_rejects_complex_offset(offset, prepared_inputs):
    u, _, factor, *_ = prepared_inputs
    evaluator = FactorStudentEvaluator(factor, u)
    with pytest.raises(TypeError, match="real"):
        evaluator.stochastic_pdf_and_gradient_grid([1., 2.], offset=offset)


@pytest.mark.parametrize("target", ["loadings", "pullback_parameters", "pullback_gradient"])
@pytest.mark.parametrize("representation", ["complex", "object"])
def test_factor_parameterization_rejects_complex(target, representation, prepared_inputs):
    _, loadings, _, _, parameterization, raw = prepared_inputs
    gradient = np.ones_like(loadings)
    if target != "pullback_gradient":
        raw = _complex(raw, representation)
    else:
        gradient = _complex(gradient, representation)
    with pytest.raises(TypeError, match="real"):
        if target == "loadings":
            parameterization.loadings(raw)
        else:
            parameterization.pullback(raw, gradient)


@pytest.mark.parametrize("operation", [
    "project", "preprocess", "kendall", "validate", "pack", "unpack",
    "policy_supplied", "policy_trial", "policy_gradient",
])
def test_correlation_inputs_reject_complex(operation, prepared_inputs):
    u, _, factor, *_ = prepared_inputs
    matrix = factor.to_dense()
    complex_matrix = _complex(matrix, "complex")
    policy = CorrelationPolicy.create(
        mode="shrinkage", estimator="joint_mle", dimension=3,
        base_correlation=matrix)
    calls = {
        "project": lambda: corr_param.project_to_corr(complex_matrix),
        "preprocess": lambda: corr_param.preprocess_correlation_matrix(
            complex_matrix, source="supplied"),
        "kendall": lambda: corr_param.estimate_kendall_correlation(_complex(u, "complex")),
        "validate": lambda: corr_param.validate_corr_matrix(complex_matrix),
        "pack": lambda: corr_param.pack_cholesky_corr(complex_matrix),
        "unpack": lambda: corr_param.unpack_cholesky_corr(_complex([.1, .2, .3], "complex"), 3),
        "policy_supplied": lambda: CorrelationPolicy.create(
            mode="fixed", estimator="supplied", dimension=3,
            supplied_correlation=complex_matrix),
        "policy_trial": lambda: policy.trial_correlation(_complex([.3], "complex")),
        "policy_gradient": lambda: policy.raw_gradient(
            _complex([.3], "complex"), matrix, [.2, .3, .4]),
    }
    with pytest.raises(TypeError, match="real"):
        calls[operation]()


@pytest.mark.parametrize("field", ["n_obs", "dimension"])
@pytest.mark.parametrize("bad", [1.8, True, np.float64(3), -1])
def test_equicorr_counts_require_integers_in_range(field, bad):
    kwargs = dict(sum_z=[0.], sum_z2=[1.], n_obs=1, dimension=3)
    kwargs[field] = bad
    with pytest.raises((TypeError, ValueError), match=field):
        EquicorrPreparedData(**kwargs)


@pytest.mark.parametrize("field", ["sum_z", "sum_z2"])
def test_equicorr_statistics_reject_complex(field):
    kwargs = dict(sum_z=[0.], sum_z2=[1.], n_obs=1, dimension=3)
    kwargs[field] = np.asarray(kwargs[field], dtype=complex)
    with pytest.raises(TypeError, match="real"):
        EquicorrPreparedData(**kwargs)


@pytest.mark.parametrize("field", ["rank", "seed", "oversampling", "dimension_tile"])
@pytest.mark.parametrize("bad", [1.8, True, np.float64(1), -1])
def test_factor_initialization_integer_options_are_strict(field, bad, prepared_inputs):
    u, *_ = prepared_inputs
    kwargs = dict(rank=1, uniqueness_min=.1, dimension_tile=2, seed=3, oversampling=1)
    kwargs[field] = bad
    with pytest.raises((TypeError, ValueError), match=field):
        estimate_factor_loadings(u, **kwargs)


def test_numpy_integer_options_preserve_initialization_and_diagnostics(prepared_inputs):
    u, *_ = prepared_inputs
    kwargs = dict(rank=1, uniqueness_min=.1, dimension_tile=2, seed=0, oversampling=0)
    expected, expected_metadata = estimate_factor_loadings(u, **kwargs)
    actual, metadata = estimate_factor_loadings(u, **{
        name: value if name == "uniqueness_min" else np.int64(value)
        for name, value in kwargs.items()
    })
    np.testing.assert_array_equal(actual, expected)
    assert metadata["random_seed"] == expected_metadata["random_seed"] == 0
    assert metadata["oversampling"] == expected_metadata["oversampling"] == 0
    assert metadata["configured_dimension_tile"] == 2

    prepared = EquicorrPreparedData([0.], [1.], np.int64(1), np.int64(2))
    assert prepared.n_obs == 1
    assert prepared.dimension == 2


def test_real_readonly_noncontiguous_factor_rows_match_dense_reference(prepared_inputs):
    u, _, factor, prepared, *_ = prepared_inputs
    storage = np.zeros((len(u), 6))
    storage[:, ::2] = u
    rows = storage[:, ::2]
    rows.setflags(write=False)
    expected = np.linalg.solve(factor.to_dense(), rows.T).T
    np.testing.assert_allclose(prepared.solve(rows), expected, rtol=5e-14, atol=5e-14)
    np.testing.assert_array_equal(rows, u)
