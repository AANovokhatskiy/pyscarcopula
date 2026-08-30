"""Dynamic multivariate API, fitted-state and bounded-sampling regressions."""

import copy
import warnings

import numpy as np
import pytest
from scipy.stats import norm

from pyscarcopula import EquicorrGaussianCopula, StochasticStudentCopula
from pyscarcopula import api
from pyscarcopula._native import gas, scar_ou
from pyscarcopula._types import GASResult, LatentResult, gas_params, ou_params
from pyscarcopula.numerical._scar_ou_config import AutoTMConfig
from pyscarcopula.numerical.ou_kernels import (
    sample_ou_trajectory,
    sample_ou_trajectory_batches,
)
from pyscarcopula.strategy import gas as gas_strategy


def _model(kind, *, supplied=True):
    if kind == "equicorr":
        return EquicorrGaussianCopula(4)
    if kind == "factor":
        return StochasticStudentCopula(
            4, corr_mode="factor", factor_rank=1,
            factor_loadings=np.full((4, 1), .45) if supplied else None)
    return StochasticStudentCopula(
        4, corr_mode=kind,
        R=.3 * np.ones((4, 4)) + .7 * np.eye(4) if supplied else None)


def _data():
    return np.random.default_rng(121).uniform(.07, .93, (21, 4))


def _attach(model, method, data=None):
    if method == "gas":
        result = GASResult(
            method="GAS", copula_name=model.name, success=True,
            log_likelihood=0., params=gas_params(.15, .08, .75),
            scaling="unit", r_last=.3 if isinstance(
                model, EquicorrGaussianCopula) else 4.)
    else:
        result = LatentResult(
            method="SCAR-TM-OU", copula_name=model.name, success=True,
            log_likelihood=0., params=ou_params(2., .4, 1.3),
            K=80, adaptive=False, grid_method="dense", grid_range=7.,
            transition_method="matrix")
    model.fit_result = result
    model._last_u = (_data() if data is None else data).copy()
    return model


DYNAMIC_CASES = [
    (kind, method)
    for method in ("gas", "scar-tm-ou")
    for kind in ("equicorr", "fixed", "shrinkage", "cholesky", "factor")
    if not (method == "gas" and kind == "cholesky")
]


@pytest.mark.parametrize("kind,method", DYNAMIC_CASES)
def test_implicit_likelihood_evaluates_the_fitted_dynamic_strategy(kind, method):
    model = _attach(_model(kind), method)
    observations = model._last_u
    expected = api.log_likelihood(model, observations, model.fit_result)
    actual = model.log_likelihood(observations)
    assert actual == pytest.approx(expected, abs=1e-12)
    if method == "scar-tm-ou":
        static = model.log_likelihood(
            observations, float(model.transform(np.array([.4]))[0]))
        assert abs(actual - static) > 1e-4


def test_implicit_equicorr_scar_likelihood_matches_independent_integral():
    from conditional._multivariate_scar_oracle import (
        ScalarScarOuReference,
        equicorr_gaussian_log_density,
        equicorr_parameter_from_state,
    )

    model = _attach(_model("equicorr"), "scar-tm-ou")
    oracle = ScalarScarOuReference(
        2., .4, 1.3, len(model._last_u),
        lambda states: equicorr_parameter_from_state(states, 4),
        equicorr_gaussian_log_density, n_nodes=301, range_sigma=8.)
    expected = oracle.filter(model._last_u).log_evidence
    assert model.log_likelihood(model._last_u) == pytest.approx(
        expected, abs=1e-9)


@pytest.mark.parametrize("kind,method", DYNAMIC_CASES)
def test_implicit_conditional_sample_uses_next_predictive_distribution(kind, method):
    model = _attach(_model(kind), method)
    options = {"given": {0: .2, 2: .7}}
    actual = model.sample_conditional(
        29, rng=np.random.default_rng(42), **options)
    expected = model.predict(29, rng=np.random.default_rng(42), **options)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual[:, 0], .2)
    np.testing.assert_array_equal(actual[:, 2], .7)


def _fit(model, observations, method, entry, *, automatic=False):
    options = dict(method=method, maxiter=1, maxfun=20)
    if method == "gas":
        if not automatic:
            options["gamma0"] = np.array([.2, .03, .7])
    else:
        options.update(
            alpha0=np.array([2., .4, 1.3]), K=24, max_K=24,
            transition_method="matrix", adaptive=False, smart_init=False)
    if entry == "model":
        return model.fit(observations, **options)
    return api.fit(model, observations, **options)


def _correlation(model):
    return model.to_correlation_matrix() if model.corr_mode == "factor" else model.R


def _two_histories():
    rng = np.random.default_rng(18)
    return tuple(
        norm.cdf(rng.multivariate_normal(
            np.zeros(4), rho * np.ones((4, 4)) + (1 - rho) * np.eye(4),
            size=100))
        for rho in (.65, -.2)
    )


@pytest.mark.parametrize("kind", ["fixed", "factor"])
@pytest.mark.parametrize("method", ["gas", "scar-tm-ou"])
@pytest.mark.parametrize("entry", ["model", "api"])
def test_refit_reestimates_data_derived_correlation(kind, method, entry):
    first, second = _two_histories()
    model = _model(kind, supplied=False)
    _fit(model, first, method, entry)
    old = _correlation(model).copy()
    refitted = _fit(model, second, method, entry)
    fresh = _model(kind, supplied=False)
    expected = _fit(fresh, second, method, entry)
    assert np.max(np.abs(_correlation(model) - old)) > .1
    np.testing.assert_allclose(_correlation(model), _correlation(fresh), atol=1e-14)
    assert refitted.log_likelihood == pytest.approx(expected.log_likelihood, abs=1e-10)
    assert refitted.n_params == expected.n_params


@pytest.mark.parametrize("kind", ["fixed", "factor"])
@pytest.mark.parametrize("automatic", [False, True])
def test_dynamic_refit_retains_explicit_correlation(kind, automatic):
    first, second = _two_histories()
    model = _model(kind)
    original = _correlation(model).copy()
    for observations in (first, second):
        _fit(model, observations, "gas", "model", automatic=automatic)
        # Correlation preprocessing may introduce a few rounding ulps.
        np.testing.assert_allclose(_correlation(model), original, rtol=0., atol=2e-15)


def test_explicit_factor_initialization_remains_fixed_across_dynamic_fits():
    first, second = _two_histories()
    model = _model("factor", supplied=False)
    model.initialize_factor(first)
    original = model.factor_loadings_.copy()
    for automatic in (False, True):
        _fit(model, second, "gas", "model", automatic=automatic)
        np.testing.assert_array_equal(model.factor_loadings_, original)


@pytest.mark.parametrize("kind", ["equicorr", "fixed", "factor"])
@pytest.mark.parametrize("method", ["gas", "scar-tm-ou"])
@pytest.mark.parametrize("entry", ["model", "api"])
def test_dynamic_training_history_is_owned(kind, method, entry):
    observations = _data()
    snapshot = observations.copy()
    model = _model(kind)
    _fit(model, observations, method, entry)
    prediction = model.predict(13, rng=np.random.default_rng(23))
    assert not np.shares_memory(model._last_u, observations)
    observations[:] = .4
    np.testing.assert_array_equal(model._last_u, snapshot)
    np.testing.assert_array_equal(
        model.predict(13, rng=np.random.default_rng(23)), prediction)


@pytest.mark.parametrize("kind", ["equicorr", "fixed", "factor"])
@pytest.mark.parametrize("entry", ["model", "api"])
def test_invalid_gas_scaling_preserves_fitted_state(kind, entry):
    model = _attach(_model(kind), "gas")
    previous = model.__dict__.copy()
    operation = model.fit if entry == "model" else lambda u, **kw: api.fit(model, u, **kw)
    with pytest.raises(ValueError, match="scaling"):
        operation(_data()[::-1].copy(), method="gas", scaling="invalid")
    assert model.__dict__.keys() == previous.keys()
    assert all(model.__dict__[key] is value for key, value in previous.items())


@pytest.mark.parametrize("kind", ["equicorr", "fixed", "factor"])
@pytest.mark.parametrize("entry", ["model", "api"])
def test_exception_after_initial_mle_rolls_back_all_fitted_owners(kind, entry, monkeypatch):
    model = _attach(_model(kind), "gas")
    previous = model.__dict__.copy()
    original_start = gas_strategy._automatic_gas_start

    def interrupt_after_mle(copula, *args, **kwargs):
        original_start(copula, *args, **kwargs)
        assert copula.fit_result.method == "MLE"
        raise RuntimeError("interrupted after MLE initialization")

    monkeypatch.setattr(gas_strategy, "_automatic_gas_start", interrupt_after_mle)
    with pytest.raises(RuntimeError, match="interrupted after MLE"):
        _fit(model, _data()[::-1].copy(), "gas", entry, automatic=True)
    assert model.__dict__.keys() == previous.keys()
    assert all(model.__dict__[key] is value for key, value in previous.items())


@pytest.mark.parametrize("method", ["gas", "scar-tm-ou"])
@pytest.mark.parametrize("operation", ["sample", "sample_at_parameter", "api", "batches"])
def test_dense_student_budget_rejects_before_consuming_rng(method, operation):
    model = _attach(_model("fixed"), method)
    rng = np.random.default_rng(10)
    before = copy.deepcopy(rng.bit_generator.state)
    options = dict(rng=rng, memory_budget_bytes=1)
    if operation == "sample_at_parameter":
        options["r"] = 4.
    with pytest.raises(MemoryError):
        if operation == "api":
            api.sample(model, model._last_u, model.fit_result, 8, **options)
        elif operation == "batches":
            next(model.sample_batches(8, batch_rows=3, **options))
        else:
            getattr(model, operation)(8, **options)
    assert rng.bit_generator.state == before


@pytest.mark.parametrize("kind", ["equicorr", "fixed", "factor"])
def test_implicit_mle_likelihood_remains_static(kind):
    model = _model(kind)
    result = model.fit(_data(), method="mle")
    assert result.success
    assert model.log_likelihood(_data()) == pytest.approx(
        model.log_likelihood(_data(), r=result.copula_param), abs=1e-12)


def test_equicorr_all_coordinates_given_needs_no_fitted_parameter():
    model = _model("equicorr")
    given = {0: .2, 1: .4, 2: .6, 3: .8}
    actual = model.sample_conditional(3, given=given)
    np.testing.assert_array_equal(actual, np.tile(list(given.values()), (3, 1)))


class _BoundedRng:
    """Prevent an accidental full-path allocation in the streaming regression."""

    def __init__(self):
        self.generator = np.random.default_rng(29)
        self.maximum = 0

    def standard_normal(self, size):
        self.maximum = max(self.maximum, int(np.prod(size)))
        assert self.maximum <= 32, "sampler requested a path-sized RNG buffer"
        return self.generator.standard_normal(size)

    def uniform(self, low, high, size):
        assert int(np.prod(size)) <= 32
        return self.generator.uniform(low, high, size)


@pytest.mark.parametrize("kind", ["equicorr", "fixed", "factor"])
def test_scar_batches_do_not_allocate_the_full_requested_path(kind):
    model = _attach(_model(kind), "scar-tm-ou")
    rng = _BoundedRng()
    blocks = model.sample_batches(
        10**9, batch_rows=3, memory_budget_bytes=4096, rng=rng)
    try:
        for _ in range(2):
            block = next(blocks)
            assert block.shape == (3, 4)
            assert np.all(np.isfinite(block))
            assert np.all((block >= 0.) & (block <= 1.))
    finally:
        blocks.close()


@pytest.mark.parametrize("n", [1, 2, 19, 257])
@pytest.mark.parametrize("batch_rows", [1, 3, 64])
def test_ou_block_sampler_preserves_full_path_time_step_and_normal_stream(n, batch_rows):
    parameters = (1.4, .2, .9)
    expected = sample_ou_trajectory(*parameters, n, np.random.default_rng(17))
    actual = np.concatenate(list(sample_ou_trajectory_batches(
        *parameters, n, np.random.default_rng(17), batch_rows=batch_rows)))
    np.testing.assert_array_equal(actual, expected)


def test_native_ou_blocks_match_independent_exact_recurrence():
    kappa, mu, nu = 1.4, .2, .9
    draws = np.random.default_rng(5).standard_normal(11)
    rho = np.exp(-kappa / (len(draws) - 1))
    sigma = nu / np.sqrt(2 * kappa)
    innovation_scale = sigma * np.sqrt(1 - rho**2)
    expected = [mu + sigma * draws[0]]
    for draw in draws[1:]:
        expected.append(mu + rho * (expected[-1] - mu) + innovation_scale * draw)
    actual = []
    previous = 0.
    for start in range(0, len(draws), 3):
        block = scar_ou.sample_trajectory_block(
            kappa, mu, nu, len(draws), previous, start == 0, draws[start:start + 3])
        actual.extend(block)
        previous = block[-1]
    np.testing.assert_allclose(actual, expected, rtol=0., atol=3e-16)


@pytest.mark.parametrize("kind", ["equicorr", "fixed", "factor"])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128, object])
def test_native_dynamic_observation_inputs_reject_complex_before_casting(kind, dtype):
    model = _model(kind)
    u = _data()
    invalid = (u + 1j).astype(dtype)
    module, params, spec, _, config = gas._inputs(.1, .03, .7, u, model, "unit", 1e-4)
    _, ou_spec, _, ou_config, method, _ = scar_ou._prepared_inputs(
        u, model, AutoTMConfig(K=20, max_K=20, adaptive=False, transition_method="matrix"))
    ou_parameters = scar_ou._params(module, 2., .4, 1.3)
    operations = [
        lambda: module.GasEvaluator().filter(params, spec, invalid, config),
        lambda: module.ScarOuEvaluator().loglik_matrix(
            ou_parameters, ou_spec, invalid, ou_config),
        lambda: module.PreparedScarOuEvaluator(ou_spec, invalid, ou_config, method),
    ]
    for operation in operations:
        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter("always")
            with pytest.raises(TypeError, match="complex"):
                operation()
        assert not emitted


@pytest.mark.parametrize("layout", ["float32", "object", "strided", "readonly", "list"])
def test_native_dynamic_observation_inputs_keep_real_coercion(layout):
    model = _model("equicorr")
    u = _data()
    if layout in {"float32", "object"}:
        u = u.astype(layout)
    elif layout == "strided":
        u = np.repeat(u, 2, axis=1)[:, ::2]
    elif layout == "readonly":
        u.flags.writeable = False
    elif layout == "list":
        u = u.tolist()
    module, params, spec, _, config = gas._inputs(.1, .03, .7, u, model, "unit", 1e-4)
    result = module.GasEvaluator().filter(params, spec, u, config)
    assert result["status"] == 0
    assert np.isfinite(result["log_likelihood"])


@pytest.mark.parametrize("column", [0, 1])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128, object])
def test_native_prepared_equicorr_inputs_reject_complex(column, dtype):
    model = _model("equicorr")
    prepared = model.prepare_sufficient_statistics(_data())
    module, params, spec, _, config = gas._inputs(
        .1, .03, .7, _data(), model, "unit", 1e-4)
    _, ou_spec, _, ou_config, method, _ = scar_ou._prepared_inputs(
        _data(), model, AutoTMConfig(K=20, max_K=20, adaptive=False,
                                   transition_method="matrix"))
    statistics = [prepared.sum_z, prepared.sum_z2]
    statistics[column] = (statistics[column] + 1j).astype(dtype)
    operations = [
        lambda: module.GasEvaluator().filter_equicorr_prepared(
            params, spec, *statistics, config),
        lambda: module.PreparedScarOuEvaluator(
            ou_spec, *statistics, ou_config, method),
    ]
    for operation in operations:
        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter("always")
            with pytest.raises(TypeError, match="complex"):
                operation()
        assert not emitted
