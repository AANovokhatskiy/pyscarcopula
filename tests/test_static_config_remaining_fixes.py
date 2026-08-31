"""Real numeric controls and explicit-result boundaries for static MLE."""

from copy import deepcopy
from dataclasses import replace
from functools import wraps
import warnings

import numpy as np
import pytest
from scipy.stats import multivariate_t, norm, t

from pyscarcopula import GaussianCopula, StudentCopula, api
from pyscarcopula._types import LBFGSBConfig, NumericalConfig
from pyscarcopula.strategy import multivariate_mle


CORRELATION = np.array([
    [1., .43, .17, -.12], [.43, 1., .21, .08],
    [.17, .21, 1., .3], [-.12, .08, .3, 1.],
])
LOADINGS = np.array([[.52], [.37], [-.29], [.41]])
DATA = norm.cdf(np.random.default_rng(317).multivariate_normal(
    np.zeros(4), CORRELATION, size=160))
OPTIONS = {
    "gtol": 2e-4, "ftol": 2e-9, "maxfun": 200, "maxiter": 100,
    "maxls": 31, "eps": 3e-6, "maxcor": 7, "finite_diff_rel_step": 4e-6,
}
VARIANTS = [
    (family, mode)
    for family in (GaussianCopula, StudentCopula)
    for mode in ("fixed", "shrinkage", "cholesky", "factor", "factor-joint")
    if family is StudentCopula or mode != "factor-joint"
]
OPTIMIZING = [
    (family, mode) for family, mode in VARIANTS
    if family is StudentCopula or mode in ("shrinkage", "cholesky")
]
COMPLEX_SCALARS = [
    complex(3, 0), np.complex64(3 + 7j), np.complex128(3 + 7j),
    np.complex128(complex(3, np.nan)), np.array(3 + 7j),
    np.array(np.complex128(3 + 7j), dtype=object),
]


def make_model(family, mode):
    options = dict(d=4, corr_mode="factor" if mode == "factor-joint" else mode)
    if mode.startswith("factor"):
        options.update(factor_rank=1, factor_loadings=LOADINGS,
                       factor_estimation="joint" if mode == "factor-joint" else "two-stage")
    else:
        options["R"] = CORRELATION
        if mode != "fixed":
            options["corr_base"] = .6 * CORRELATION + .4 * np.eye(4)
    return family(**options)


@pytest.fixture(autouse=True)
def no_complex_coercion_warnings():
    # Rejecting after a lossy conversion is too late, even if a later check fails.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield
    assert not [warning for warning in caught
                if warning.category.__name__ == "ComplexWarning"]


@pytest.mark.parametrize("name", OPTIONS)
@pytest.mark.parametrize("value", COMPLEX_SCALARS)
def test_optimizer_controls_reject_complex_before_conversion(name, value):
    with pytest.raises(TypeError, match="real|complex"):
        LBFGSBConfig(**{name: value}).options()
    with pytest.raises(TypeError, match="real|complex"):
        LBFGSBConfig(**OPTIONS).options(**{name: value})


@pytest.mark.parametrize("name", OPTIONS)
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, 0., -1.])
def test_optimizer_controls_require_finite_positive_values(name, value):
    with pytest.raises(ValueError):
        LBFGSBConfig(**{name: value}).options()
    with pytest.raises(ValueError):
        LBFGSBConfig(**OPTIONS).options(**{name: value})


def test_real_optimizer_controls_preserve_conversion_none_and_override_precedence():
    config = LBFGSBConfig(
        gtol=np.float32(.125), ftol=np.array(2e-9), maxfun=np.float64(200.75),
        maxiter=np.int64(100), maxls=31, eps=3e-6, maxcor=7,
        finite_diff_rel_step=np.array(4e-6, dtype=object))
    expected = dict(OPTIONS, gtol=.125)
    assert config.options() == expected
    assert config.options(**dict.fromkeys(OPTIONS)) == expected
    assert config.options(gtol=.25, maxiter=51.75) == dict(expected, gtol=.25, maxiter=51)
    with pytest.raises(TypeError, match="maxiterr"):
        config.options(maxiter=3, maxiterr=3)


@pytest.mark.parametrize("family,mode", OPTIMIZING)
@pytest.mark.parametrize("channel", ["direct", "config"])
@pytest.mark.parametrize("name,value", [
    ("gtol", np.inf), ("ftol", np.nan),
    ("maxiter", np.complex128(9 + 7j)),
    ("eps", np.complex128(3e-6 + 7j)),
    ("finite_diff_rel_step", np.inf),
])
def test_api_fit_rejects_invalid_active_optimizer_control(family, mode, channel, name, value):
    owner = "mle_optimizer" if family is GaussianCopula else "static_student_optimizer"
    with pytest.raises((TypeError, ValueError)):
        config = (NumericalConfig(**{owner: LBFGSBConfig(**{name: value})})
                  if channel == "config" else None)
        kwargs = {name: value} if channel == "direct" else {}
        api.fit(make_model(family, mode), DATA, method="mle", config=config, **kwargs)


@pytest.mark.parametrize("family,mode", OPTIMIZING)
def test_api_fit_preserves_owner_none_and_direct_options(family, mode, monkeypatch):
    owner = "mle_optimizer" if family is GaussianCopula else "static_student_optimizer"
    inactive = "static_student_optimizer" if family is GaussianCopula else "mle_optimizer"
    config = NumericalConfig(**{
        owner: LBFGSBConfig(**OPTIONS),
        inactive: LBFGSBConfig(gtol=.75, maxiter=1),
    })
    calls = []
    original = multivariate_mle.minimize

    @wraps(original)
    def minimize(*args, **kwargs):
        calls.append((dict(kwargs["options"]), kwargs["jac"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(multivariate_mle, "minimize", minimize)
    result = api.fit(make_model(family, mode), DATA, method="mle", config=config,
                     gtol=None, maxiter=101, maxls=107)
    assert result.success, result.message
    assert calls
    # Analytic gradients leave eps/finite_diff_rel_step inactive, but their
    # supplied values must still survive validation and reach SciPy unchanged.
    expected = dict(OPTIONS, maxiter=101, maxls=107)
    assert all(options == expected and jac is True for options, jac in calls)


@pytest.mark.parametrize("family,mode", VARIANTS)
@pytest.mark.parametrize("value", [
    np.complex128(1e10 + 0j), np.complex128(1e10 + 7j),
    np.nan, np.inf, -np.inf, 0., -1.,
])
def test_static_fit_rejects_invalid_failure_policy(family, mode, value):
    with pytest.raises((ValueError, TypeError)):
        api.fit(make_model(family, mode), DATA, method="mle",
                config=NumericalConfig(fail_value=value))


@pytest.fixture(scope="module", params=["fixed", "factor"])
def fitted_student(request):
    model = make_model(StudentCopula, request.param)
    result = api.fit(model, DATA, method="mle")
    assert result.success, result.message
    return model, result


@pytest.mark.parametrize("family,mode", [(GaussianCopula, "fixed"), (StudentCopula, "fixed")])
@pytest.mark.parametrize("representation", ["complex", "object"])
def test_result_correlation_rejects_complex_before_copy(family, mode, representation):
    model = make_model(family, mode)
    result = api.fit(model, DATA, method="mle")
    correlation_before = model.to_correlation_matrix().copy()
    correlation = CORRELATION.astype(complex)
    correlation[0, 1] += 2j
    correlation[1, 0] += 2j
    if representation == "object":
        correlation = correlation.astype(object)
    with pytest.raises(TypeError, match="real|complex"):
        replace(result, correlation_matrix=correlation)
    np.testing.assert_array_equal(model.to_correlation_matrix(), correlation_before)


@pytest.mark.parametrize("operation", ["log_likelihood", "sample", "predict"])
@pytest.mark.parametrize("value", [np.complex128(4.3 + 0j), np.complex128(4.3 + 2j)])
def test_explicit_student_result_rejects_complex_df_before_rng(fitted_student, operation, value):
    model, result = fitted_student
    invalid = replace(result, copula_param=value)
    rng = np.random.default_rng(452)
    before = deepcopy(rng.bit_generator.state)
    with pytest.raises(TypeError, match="real|complex"):
        if operation == "log_likelihood":
            api.log_likelihood(model, DATA, invalid)
        else:
            getattr(api, operation)(model, DATA, invalid, 13, rng=rng)
    assert rng.bit_generator.state == before
    assert model.fit_result is result


def test_explicit_real_student_result_preserves_likelihood_sampling_and_state(fitted_student):
    model, fitted = fitted_student
    correlation = model.to_correlation_matrix().copy()
    df_before = model.df
    scalar_result = replace(fitted, copula_param=4.3)
    array_result = replace(fitted, copula_param=np.array(4.3))
    quantiles = t.ppf(DATA, 4.3)
    expected = np.sum(multivariate_t.logpdf(quantiles, shape=correlation, df=4.3)
                      - t.logpdf(quantiles, 4.3).sum(axis=1))
    assert api.log_likelihood(model, DATA, array_result) == pytest.approx(expected, abs=3e-7)
    for operation in (api.sample, api.predict):
        reference = operation(model, DATA, scalar_result, 17, given={1: .3},
                              rng=np.random.default_rng(319))
        actual = operation(model, DATA, array_result, 17, given={1: .3},
                           rng=np.random.default_rng(319))
        np.testing.assert_array_equal(actual, reference)
        np.testing.assert_array_equal(actual[:, 1], .3)
    assert model.df == df_before
    assert model.fit_result is fitted
    np.testing.assert_array_equal(model.to_correlation_matrix(), correlation)


def test_result_correlation_keeps_owned_copy_of_real_input(fitted_student):
    _, fitted = fitted_student
    correlation = CORRELATION.copy()
    result = replace(fitted, correlation_matrix=correlation)
    correlation[0, 1] = .01
    np.testing.assert_array_equal(result.correlation_matrix, CORRELATION)
    assert not np.shares_memory(result.correlation_matrix, correlation)
