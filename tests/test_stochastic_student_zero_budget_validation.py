"""Sampling budgets must be validated before zero-size Student shortcuts."""

from copy import deepcopy

import numpy as np
import pytest

from pyscarcopula import NumericalConfig, StochasticStudentCopula


CORRELATION = np.eye(4) * 0.8 + 0.2
LOADINGS = np.array([[0.4], [0.3], [-0.2], [0.35]])


@pytest.fixture(scope="module", params=[
    (correlation, method)
    for correlation in ("fixed", "factor")
    for method in ("MLE", "GAS", "SCAR-TM-OU")
])
def fitted(request):
    correlation, method = request.param
    source = StochasticStudentCopula(4, R=CORRELATION)
    observations = source.sample_at_parameter(
        80, 5.0, rng=np.random.default_rng(773))
    model = (
        StochasticStudentCopula(4, R=CORRELATION)
        if correlation == "fixed" else StochasticStudentCopula(
            4, corr_mode="factor", factor_rank=1,
            factor_loadings=LOADINGS))
    if method == "MLE":
        options = dict(maxiter=300, maxfun=1000)
    elif method == "GAS":
        options = dict(gamma0=[0.02, 0.04, 0.7], maxiter=1, maxfun=12)
    else:
        options = dict(
            alpha0=[1.0, 0.2, 0.7], K=12, max_K=12, adaptive=False,
            smart_init=False, transition_method="matrix",
            maxiter=1, maxfun=12)
    result = model.fit(
        observations, method=method, config=NumericalConfig(n_threads=1),
        **options)
    assert model.fit_result is result
    return model


def sampling_call(model, entry, n, budget, rng):
    options = dict(memory_budget_bytes=budget, rng=rng, n_threads=1)
    if entry == "sample_at_parameter_batches":
        return list(model.sample_at_parameter_batches(
            n, 5.0, batch_rows=2, **options))
    if entry == "predict_batches":
        return list(model.predict_batches(n, batch_rows=2, **options))
    return model.predict(n, **options)


@pytest.mark.parametrize("entry", [
    "sample_at_parameter_batches", "predict_batches", "predict",
])
@pytest.mark.parametrize("budget, error", [
    (-1, ValueError),
    (True, TypeError),
    (np.bool_(False), TypeError),
    (1.5, TypeError),
    ("bad", TypeError),
])
def test_invalid_budget_rejected_before_zero_output(fitted, entry, budget, error):
    rng = np.random.default_rng(927)
    before = deepcopy(rng.bit_generator.state)

    with pytest.raises(error, match="memory_budget_bytes"):
        sampling_call(fitted, entry, 0, budget, rng)

    assert rng.bit_generator.state == before


@pytest.mark.parametrize("entry", [
    "sample_at_parameter_batches", "predict_batches", "predict",
])
@pytest.mark.parametrize("budget", [None, 0, np.int64(0)])
def test_empty_output_accepts_absent_or_zero_budget(fitted, entry, budget):
    rng = np.random.default_rng(928)
    before = deepcopy(rng.bit_generator.state)

    actual = sampling_call(fitted, entry, 0, budget, rng)

    if entry.endswith("batches"):
        assert actual == []
    else:
        assert actual.shape == (0, 4)
    assert rng.bit_generator.state == before


@pytest.mark.parametrize("entry", [
    "sample_at_parameter_batches", "predict_batches", "predict",
])
def test_positive_output_still_requires_sufficient_budget(fitted, entry):
    with pytest.raises(MemoryError):
        sampling_call(fitted, entry, 3, 0, np.random.default_rng(929))

    expected = sampling_call(
        fitted, entry, 3, None, np.random.default_rng(929))
    actual = sampling_call(
        fitted, entry, 3, 100000, np.random.default_rng(929))
    if entry.endswith("batches"):
        expected = np.concatenate(expected)
        actual = np.concatenate(actual)
    np.testing.assert_array_equal(actual, expected)


def test_factor_batches_keep_workspace_guard():
    model = StochasticStudentCopula(
        4, corr_mode="factor", factor_rank=1, factor_loadings=LOADINGS)
    rng = np.random.default_rng(930)
    before = deepcopy(rng.bit_generator.state)
    # Enough for a 2x4 output, but not for factor RNG/native workspaces.
    with pytest.raises(MemoryError):
        list(model.sample_at_parameter_batches(
            3, 5.0, batch_rows=2, memory_budget_bytes=2 * 4 * 8, rng=rng))
    assert rng.bit_generator.state == before

    peak = model._factor_sampling_peak_bytes(2)
    with pytest.raises(MemoryError):
        list(model.sample_at_parameter_batches(
            3, 5.0, batch_rows=2, memory_budget_bytes=peak - 1, rng=rng))
    batches = list(model.sample_at_parameter_batches(
        3, 5.0, batch_rows=2, memory_budget_bytes=peak, rng=rng))
    assert [batch.shape for batch in batches] == [(2, 4), (1, 4)]
