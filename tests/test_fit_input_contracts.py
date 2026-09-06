"""Public fit input ownership, rank policy and minimum observation contracts."""

import numpy as np
import pytest
from scipy.stats import rankdata

from pyscarcopula import (
    GumbelCopula, IndependentCopula, GaussianCopula, StudentCopula,
    EquicorrGaussianCopula, StochasticStudentCopula, VineCopula, api,
)


def _fit(model, observations, entry, **options):
    if entry == "api":
        return api.fit(model, observations, **options)
    return model.fit(observations, **options)


@pytest.mark.parametrize("entry", ["object", "api"])
@pytest.mark.parametrize("method", ["MLE", "GAS", "SCAR-TM-OU", "SCAR-TM-JACOBI"])
@pytest.mark.parametrize("layout", ["c", "f", "readonly", "strided"])
def test_bivariate_fit_owns_history_used_by_prediction(entry, method, layout):
    model = GumbelCopula()
    values = model.sample_at_parameter(120, 2., rng=np.random.default_rng(5))
    backing = np.repeat(values, 2, axis=1) if layout == "strided" else values.copy()
    if layout == "f":
        backing = np.asfortranarray(backing)
    observations = backing[:, ::2] if layout == "strided" else backing.view()
    if layout == "readonly":
        observations.flags.writeable = False
    result = _fit(model, observations, entry, method=method)
    assert result.success, result.message
    np.testing.assert_array_equal(observations, values)
    assert not np.shares_memory(model._last_u, observations)
    before = model.predict(8, rng=np.random.default_rng(10))
    backing[:] = .5
    np.testing.assert_array_equal(model._last_u, values)
    np.testing.assert_array_equal(
        model.predict(8, rng=np.random.default_rng(10)), before)


@pytest.mark.parametrize("entry", ["object", "api"])
def test_independent_fit_retains_owned_training_data(entry):
    observations = np.random.default_rng(22).uniform(.1, .9, (20, 2))
    expected = observations.copy()
    model = IndependentCopula()
    _fit(model, observations, entry, method="MLE")
    observations[:] = .5
    np.testing.assert_array_equal(model._last_u, expected)
    assert not np.shares_memory(observations, model._last_u)


@pytest.mark.parametrize("entry", ["object", "api"])
@pytest.mark.parametrize("factory", [IndependentCopula, GumbelCopula, GaussianCopula, StudentCopula])
def test_public_fit_ranks_ties_in_input_order_without_mutating_raw_data(entry, factory):
    raw = np.array([[3., 1.], [1., 3.], [3., 2.], [2., 3.], [1., 2.], [2., 1.]])
    before = raw.copy()
    expected = rankdata(raw, method="ordinal", axis=0) / (len(raw) + 1.)
    model = factory()
    _fit(model, raw, entry, method="MLE", to_pobs=True)
    np.testing.assert_array_equal(raw, before)
    np.testing.assert_array_equal(model._last_u, expected)


@pytest.mark.parametrize("entry", ["object", "api"])
@pytest.mark.parametrize("factory,dimension", [
    (GumbelCopula, 2), (IndependentCopula, 2), (GaussianCopula, 3),
    (StudentCopula, 3), (lambda: EquicorrGaussianCopula(d=3), 3),
    (lambda: StochasticStudentCopula(d=3, R=np.eye(3)), 3),
    (lambda: VineCopula.cvine(d=3), 3),
])
def test_public_fit_empty_data_is_rejected_without_publishing_state(entry, factory, dimension):
    model = factory()
    with pytest.raises(ValueError):
        _fit(model, np.empty((0, dimension)), entry, method="MLE")
    assert model.fit_result is None
    assert getattr(model, "_last_u", None) is None


@pytest.mark.parametrize("entry", ["object", "api"])
@pytest.mark.parametrize("factory,dimension,method,accepted", [
    (GumbelCopula, 2, "MLE", True),
    (GumbelCopula, 2, "GAS", True),
    (GumbelCopula, 2, "SCAR-TM-OU", False),
    (GumbelCopula, 2, "SCAR-TM-JACOBI", False),
    (IndependentCopula, 2, "MLE", True),
    (GaussianCopula, 3, "MLE", False),
    (StudentCopula, 3, "MLE", False),
    (lambda: EquicorrGaussianCopula(d=3), 3, "MLE", True),
    (lambda: StochasticStudentCopula(d=3, R=np.eye(3)), 3, "MLE", True),
    (lambda: VineCopula.cvine(d=3), 3, "MLE", False),
])
def test_public_fit_singleton_follows_model_and_method_contract(
        entry, factory, dimension, method, accepted):
    model = factory()
    observations = np.array([[.2, .6, .4]])[:, :dimension]
    before = observations.copy()
    if accepted:
        result = _fit(model, observations, entry, method=method)
        assert result.success, result.message
        assert np.isfinite(result.log_likelihood)
        assert model.fit_result is result
    else:
        with pytest.raises(ValueError):
            _fit(model, observations, entry, method=method)
        assert model.fit_result is None
        assert getattr(model, "_last_u", None) is None
    np.testing.assert_array_equal(observations, before)


@pytest.mark.parametrize("entry", ["object", "api"])
@pytest.mark.parametrize("factory,dimension", [
    (IndependentCopula, 2), (GaussianCopula, 3), (StudentCopula, 3),
    (lambda: EquicorrGaussianCopula(d=3), 3),
    (lambda: StochasticStudentCopula(d=3, R=np.eye(3)), 3),
    (lambda: VineCopula.cvine(d=3), 3),
])
@pytest.mark.parametrize("boundary", ["endpoints", "nextafter"])
def test_public_mle_fit_interval_contract(entry, factory, dimension, boundary):
    model = factory()
    observations = np.random.default_rng(55).uniform(.1, .9, (24, dimension))
    observations[0, 0], observations[1, 0] = (
        (0., 1.) if boundary == "endpoints" else
        (np.nextafter(0., 1.), np.nextafter(1., 0.)))
    before = observations.copy()
    if isinstance(model, VineCopula) and boundary == "endpoints":
        with pytest.raises(ValueError, match="pseudo-observations"):
            _fit(model, observations, entry, method="MLE")
        assert model.fit_result is None
    else:
        _fit(model, observations, entry, method="MLE")
        assert model.fit_result is not None
        assert np.isfinite(model.fit_result.log_likelihood)
    np.testing.assert_array_equal(observations, before)
