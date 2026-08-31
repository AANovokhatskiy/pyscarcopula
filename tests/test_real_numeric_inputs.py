"""Shared real-input contracts across scalar and array adapter boundaries."""

from copy import deepcopy
from types import SimpleNamespace
import warnings

import numpy as np
import pytest
from scipy.stats import multivariate_t, t

from pyscarcopula import (
    BivariateGaussianCopula, EquicorrGaussianCopula,
    StochasticStudentCopula, StudentCopula,
)
from pyscarcopula._native import multivariate, static, vine
from pyscarcopula.copula.multivariate.conditional import as_path
from pyscarcopula.copula.multivariate.factor_correlation import FactorCorrelation
from pyscarcopula.copula.multivariate.factor_estimation import FactorLoadingParameterization
from pyscarcopula.copula.multivariate.factor_student import FactorStudentEvaluator
from pyscarcopula.numerical import _arrays as arrays
from pyscarcopula.vine._helpers import _prepared_open_unit_draws
from pyscarcopula.vine._rvine_edges import _edge_log_likelihood


DATA = np.random.default_rng(5801).uniform(.04, .96, (32, 4))
LOADINGS = np.array([[.4], [.3], [-.2], [.5]])
CORRELATION = LOADINGS @ LOADINGS.T
np.fill_diagonal(CORRELATION, 1.)
COMPLEX_VALUES = [
    complex(5, 0), complex(5, 2),
    np.complex64(5 + 2j), np.complex128(5 + 0j),
    np.complex128(complex(5, np.nan)), np.complex128(complex(5, np.inf)),
    np.array(5 + 2j), np.array(np.complex128(5 + 2j), dtype=object),
]


@pytest.fixture(autouse=True)
def no_lossy_coercion_warnings():
    # A rejected input must not lose information before its exception either.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield
    assert not [w for w in caught if issubclass(
        w.category, (np.exceptions.ComplexWarning, DeprecationWarning))]


@pytest.mark.parametrize("value", COMPLEX_VALUES)
def test_scalar_and_array_normalizers_reject_complex(value):
    for normalize in (arrays.as_float64_scalar, arrays.as_float64_array):
        with pytest.raises(TypeError, match="trial.*real.*complex"):
            normalize(value, name="trial")


@pytest.mark.parametrize("value", [5, 5., np.float16(5), np.float32(5), np.float64(5),
                                  np.longdouble(5), np.int64(5),
                                  np.array(5.), np.array(5, dtype=object)])
def test_scalar_normalizer_accepts_real_scalars(value):
    result = arrays.as_float64_scalar(value)
    assert type(result) is float
    assert result == 5.


@pytest.mark.parametrize("value", [[], [5.], [[5.]], np.ones((1, 1))])
def test_scalar_normalizer_rejects_arrays_even_with_one_element(value):
    with pytest.raises(ValueError, match="scalar"):
        arrays.as_float64_scalar(value)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, -3., 0.])
def test_normalization_preserves_domain_validation_ownership(value):
    result = arrays.as_float64_scalar(np.array(value))
    assert np.isnan(result) if np.isnan(value) else result == value


def test_array_normalizer_preserves_views_and_existing_real_object_coercion():
    source = np.arange(24., dtype=np.float64).reshape(6, 4)[::2, ::2]
    source.setflags(write=False)
    assert arrays.as_float64_array(source) is source
    assert not arrays.as_float64_array(source).flags.writeable
    np.testing.assert_array_equal(
        arrays.as_float64_array(np.array([.5, 1, ".25"], dtype=object)),
        [.5, 1, .25])


@pytest.fixture(scope="module", params=["fixed", "shrinkage", "cholesky", "factor", "factor-joint"])
def student(request):
    mode = request.param
    if mode.startswith("factor"):
        model = StudentCopula(d=4, corr_mode="factor", factor_rank=1,
                              factor_loadings=LOADINGS,
                              factor_estimation="joint" if mode == "factor-joint" else "two-stage")
    else:
        model = StudentCopula(d=4, R=CORRELATION, corr_mode=mode)
    result = model.fit(DATA)
    assert result.success, result.message
    return model


@pytest.mark.parametrize("value", COMPLEX_VALUES)
@pytest.mark.parametrize("method", ["log_likelihood", "log_pdf_rows"])
def test_student_scalar_entries_reject_complex(student, value, method):
    with pytest.raises(TypeError):
        getattr(student, method)(DATA, parameter=value)


def test_real_scalar_forms_preserve_student_likelihood_and_state(student):
    df_before = student.df
    correlation = student.to_correlation_matrix().copy()
    z = t.ppf(DATA, 5.)
    expected = multivariate_t.logpdf(z, shape=correlation, df=5.) - t.logpdf(z, 5.).sum(axis=1)
    for df in (5., np.float64(5), np.array(5.), np.array(5., dtype=object)):
        np.testing.assert_allclose(student.log_pdf_rows(DATA, parameter=df), expected, atol=2e-8)
        assert student.log_likelihood(DATA, parameter=df) == pytest.approx(expected.sum(), abs=2e-8)
    assert student.df == df_before
    np.testing.assert_array_equal(student.to_correlation_matrix(), correlation)


@pytest.fixture(scope="module", params=["equicorr", "student-dense", "student-factor"])
def dynamic_model(request):
    if request.param == "equicorr":
        return EquicorrGaussianCopula(d=4), .2
    if request.param == "student-factor":
        return StochasticStudentCopula(d=4, corr_mode="factor", factor_rank=1,
                                       factor_loadings=LOADINGS), 5.
    return StochasticStudentCopula(d=4, R=CORRELATION), 5.


@pytest.mark.parametrize("value", COMPLEX_VALUES)
def test_dynamic_explicit_likelihood_rejects_complex(dynamic_model, value):
    model, _ = dynamic_model
    with pytest.raises(TypeError):
        model.log_likelihood(DATA, r=value)


@pytest.mark.parametrize("batches", [False, True])
@pytest.mark.parametrize("representation", ["scalar", "array", "object"])
def test_sampling_rejects_complex_before_consuming_rng(dynamic_model, batches, representation):
    model, real = dynamic_model
    value = np.complex128(complex(real, 0))
    if representation != "scalar":
        value = np.full(7, value, dtype=object if representation == "object" else complex)
    rng = np.random.default_rng(923)
    before = deepcopy(rng.bit_generator.state)
    with pytest.raises(TypeError):
        if batches:
            list(model.sample_at_parameter_batches(7, value, batch_rows=3, rng=rng))
        else:
            model.sample_at_parameter(7, value, rng=rng)
    assert rng.bit_generator.state == before


@pytest.mark.parametrize("value", COMPLEX_VALUES)
@pytest.mark.parametrize("method", [
    "result", "value_result", "joint_result", "log_pdf_rows", "log_likelihood",
    "objective_and_gradient", "validated_objective_and_gradient",
    "objective_and_joint_gradient", "transformed_objective_and_gradient",
])
def test_static_facade_rejects_complex_without_optimizer_penalty(value, method):
    evaluator = (static.prepare(EquicorrGaussianCopula(d=4), DATA)
                 if method == "transformed_objective_and_gradient"
                 else static.prepare_student(CORRELATION, DATA))
    with pytest.raises(TypeError):
        getattr(evaluator, method)(value)


@pytest.fixture(scope="module")
def factor():
    return FactorStudentEvaluator(FactorCorrelation(LOADINGS), DATA)


@pytest.mark.parametrize("value", COMPLEX_VALUES)
@pytest.mark.parametrize("method", ["evaluate", "log_likelihood_and_gradient", "joint_likelihood_and_gradient"])
def test_factor_student_rejects_complex_df(factor, value, method):
    with pytest.raises(TypeError):
        getattr(factor, method)(value)


@pytest.mark.parametrize("argument", ["df", "parameters", "penalty", "condition_max"])
def test_factor_parameterized_objective_rejects_complex(factor, argument):
    parameterization, parameters = FactorLoadingParameterization.from_loadings(
        LOADINGS, uniqueness_min=1e-8)
    kwargs = dict(df=5., parameters=parameters, parameterization=parameterization,
                  penalty=1e-5, condition_max=1e12)
    kwargs[argument] = np.asarray(kwargs[argument], dtype=complex)
    with pytest.raises(TypeError):
        factor.penalized_parameterized_objective_and_gradient(**kwargs)


@pytest.mark.parametrize("field", ["df", "observations", "nodes", "table"])
def test_ppf_table_rejects_complex_and_preserves_real_quantiles(field):
    nodes = np.array([4., 5., 6.])
    table = t.ppf(DATA[None, :, :], nodes[:, None, None])
    kwargs = dict(observations=DATA, nodes=nodes, table=table, df=np.array(5.))
    np.testing.assert_allclose(multivariate.evaluate_student_ppf_table(**kwargs), table[1], atol=2e-8)
    kwargs[field] = np.asarray(kwargs[field], dtype=complex)
    with pytest.raises(TypeError):
        multivariate.evaluate_student_ppf_table(**kwargs)
    if field != "observations":
        kwargs.pop("observations")
        with pytest.raises(TypeError):
            multivariate.interpolate_student_ppf_table(**kwargs)
    else:
        with pytest.raises(TypeError):
            multivariate.prepare_student_ppf_table(kwargs[field])


@pytest.mark.parametrize("object_dtype", [False, True])
@pytest.mark.parametrize("field", ["correlation", "df", "u"])
def test_dense_student_rosenblatt_rejects_complex(field, object_dtype):
    kwargs = dict(correlation=CORRELATION, df=np.full(len(DATA), 5.), u=DATA)
    kwargs[field] = np.asarray(kwargs[field], dtype=complex)
    if object_dtype:
        kwargs[field] = kwargs[field].astype(object)
    with pytest.raises(TypeError):
        multivariate.dense_student_rosenblatt(**kwargs)


@pytest.mark.parametrize("object_dtype", [False, True])
def test_shared_paths_and_replay_draws_reject_complex(object_dtype):
    values = np.array([np.complex128(.2), np.complex128(.4)],
                      dtype=object if object_dtype else complex)
    for operation in (
        lambda: as_path(values, 2, "r"),
        lambda: vine._parameter_path(values, 2, (0, 1)),
        lambda: _prepared_open_unit_draws(values, (2,), name="draws"),
    ):
        with pytest.raises(TypeError):
            operation()


@pytest.mark.parametrize("fitted", [False, True])
def test_vine_edge_parameter_reaches_shared_scalar_validation(fitted):
    value = np.complex128(.3 + 0j)
    edge = SimpleNamespace(copula=BivariateGaussianCopula(),
                           param=None if fitted else value,
                           fit_result=SimpleNamespace(copula_param=value) if fitted else None)
    with pytest.raises(TypeError):
        _edge_log_likelihood(edge, DATA[:, :2])
