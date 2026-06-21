"""Tests for the thin Python adapter around the compiled GAS evaluator."""

from types import SimpleNamespace

import numpy as np
import pytest

from pyscarcopula.copula.gumbel import GumbelCopula
from pyscarcopula.copula.multivariate import (
    EquicorrGaussianCopula,
    StochasticStudentCopula,
)
from pyscarcopula.numerical import _cpp_gas
from pyscarcopula.numerical._cpp_extension import (
    CppError,
    CppUnavailable,
    CppUnsupported,
)
from pyscarcopula.numerical.gas_filter import (
    gas_filter,
    gas_loglik,
)


OBSERVATIONS = np.array(
    [
        [0.12, 0.83],
        [0.71, 0.28],
        [0.44, 0.62],
        [0.91, 0.17],
        [0.33, 0.76],
        [0.58, 0.39],
    ],
    dtype=np.float64,
)
PARAMS = (0.07, 0.35, 0.62)


@pytest.mark.parametrize("scaling", ["unit", "fisher"])
def test_cpp_gas_wrapper_filter_and_likelihood_are_internally_consistent(
        scaling):
    copula = GumbelCopula(rotate=90)

    result = _cpp_gas.filter_result(
        *PARAMS, OBSERVATIONS, copula, scaling=scaling)
    tuple_result = _cpp_gas.filter(
        *PARAMS, OBSERVATIONS, copula, scaling=scaling)

    assert isinstance(result, _cpp_gas.GasFilterOutput)
    assert result.score_path.shape == (len(OBSERVATIONS) - 1,)
    np.testing.assert_allclose(tuple_result[0], result.g_path)
    np.testing.assert_allclose(tuple_result[1], result.r_path)
    assert tuple_result[2] == result.log_likelihood
    assert _cpp_gas.log_likelihood(
        *PARAMS, OBSERVATIONS, copula, scaling=scaling
    ) == pytest.approx(result.log_likelihood, rel=2e-9, abs=2e-10)
    assert _cpp_gas.negative_log_likelihood(
        *PARAMS, OBSERVATIONS, copula, scaling=scaling
    ) == pytest.approx(-result.log_likelihood, rel=2e-9, abs=2e-10)


@pytest.mark.parametrize("scaling", ["unit", "fisher"])
def test_cpp_gas_wrapper_update_prediction_and_h_are_consistent(scaling):
    copula = GumbelCopula(rotate=90)
    initial = _cpp_gas.initial_state(
        *PARAMS, copula, scaling=scaling)
    filtered = _cpp_gas.filter_result(
        *PARAMS, OBSERVATIONS, copula, scaling=scaling)

    assert initial.g == pytest.approx(PARAMS[0] / (1.0 - PARAMS[2]))
    assert initial.parameter == pytest.approx(filtered.r_path[0])

    update = _cpp_gas.update_one(
        *PARAMS,
        filtered.g_path[-1],
        OBSERVATIONS[-1],
        copula,
        scaling=scaling,
    )

    assert isinstance(update, _cpp_gas.GasUpdateOutput)
    assert update.r == pytest.approx(filtered.r_path[-1])
    assert update.r_next == pytest.approx(
        copula.transform(np.array([update.g_next]))[0])
    for horizon in ("current", "next", 0, 1):
        normalized = "current" if horizon in ("current", 0) else "next"
        value = _cpp_gas.predict_parameter(
            *PARAMS,
            OBSERVATIONS,
            copula,
            scaling=scaling,
            horizon=horizon,
        )
        expected = (
            filtered.r_path[-1]
            if normalized == "current"
            else update.r_next
        )
        assert value == pytest.approx(expected, rel=2e-9, abs=2e-10)
    np.testing.assert_allclose(
        _cpp_gas.h_path(
            *PARAMS, OBSERVATIONS, copula, scaling=scaling),
        copula.h(
            OBSERVATIONS[:, 1],
            OBSERVATIONS[:, 0],
            filtered.r_path,
        ),
        rtol=2e-9,
        atol=2e-10,
    )


def test_public_bivariate_filter_and_loglik_route_to_cpp(monkeypatch):
    copula = GumbelCopula()
    calls = []

    def fake_filter(*args, **kwargs):
        calls.append(("filter", args, kwargs))
        return np.array([1.0]), np.array([2.0]), 3.0

    def fake_loglik(*args, **kwargs):
        calls.append(("loglik", args, kwargs))
        return 4.0

    monkeypatch.setattr(_cpp_gas, "filter", fake_filter)
    monkeypatch.setattr(_cpp_gas, "log_likelihood", fake_loglik)

    filtered = gas_filter(*PARAMS, OBSERVATIONS[:1], copula)
    likelihood = gas_loglik(*PARAMS, OBSERVATIONS[:1], copula)

    np.testing.assert_allclose(filtered[0], [1.0])
    np.testing.assert_allclose(filtered[1], [2.0])
    assert filtered[2] == 3.0
    assert likelihood == 4.0
    assert [call[0] for call in calls] == ["filter", "loglik"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"omega": np.nan}, "omega"),
        ({"gamma": np.inf}, "gamma"),
        ({"beta": "bad"}, "beta"),
        ({"scaling": "unknown"}, "scaling"),
        ({"score_eps": 0.0}, "score_eps"),
        ({"score_eps": np.nan}, "score_eps"),
    ],
)
def test_cpp_gas_wrapper_rejects_invalid_scalar_inputs(kwargs, message):
    values = {
        "omega": PARAMS[0],
        "gamma": PARAMS[1],
        "beta": PARAMS[2],
        "scaling": "unit",
        "score_eps": 1e-4,
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=message):
        _cpp_gas.filter(
            values["omega"],
            values["gamma"],
            values["beta"],
            OBSERVATIONS,
            GumbelCopula(),
            values["scaling"],
            values["score_eps"],
        )


@pytest.mark.parametrize(
    "u",
    [
        np.empty((0, 2)),
        np.ones((3, 1)),
        np.ones((2, 3)),
        np.array([[0.2, np.nan]]),
        [0.2, 0.4],
    ],
)
def test_cpp_gas_wrapper_rejects_invalid_filter_observations(u):
    with pytest.raises(ValueError):
        _cpp_gas.filter(*PARAMS, u, GumbelCopula())


def test_cpp_gas_wrapper_update_requires_exactly_one_pair():
    with pytest.raises(ValueError, match="exactly one"):
        _cpp_gas.update_one(
            *PARAMS, 0.1, OBSERVATIONS[:2], GumbelCopula())
    with pytest.raises(ValueError, match="shape"):
        _cpp_gas.update_one(
            *PARAMS, 0.1, np.array([0.2]), GumbelCopula())


def test_cpp_gas_wrapper_rejects_custom_python_copula():
    custom = SimpleNamespace(name="custom-python-copula", rotate=0)

    assert not _cpp_gas.supported(custom)
    with pytest.raises(CppUnsupported, match="custom-python-copula"):
        _cpp_gas.ensure_supported(custom)
    with pytest.raises(CppUnsupported, match="custom-python-copula"):
        _cpp_gas.filter(*PARAMS, OBSERVATIONS, custom)


def test_cpp_gas_wrapper_has_no_extension_fallback(monkeypatch):
    def unavailable():
        raise CppUnavailable("compiled extension missing")

    monkeypatch.setattr(_cpp_gas._cpp_extension, "load", unavailable)

    with pytest.raises(CppUnavailable, match="compiled extension missing"):
        _cpp_gas.filter(*PARAMS, OBSERVATIONS, GumbelCopula())


@pytest.mark.parametrize(
    ("status", "error"),
    [
        (1, CppError),
        (2, ValueError),
        (3, CppUnsupported),
        (4, CppUnsupported),
        (5, CppUnsupported),
        (6, ValueError),
        (7, FloatingPointError),
        (999, CppError),
    ],
)
def test_cpp_gas_wrapper_translates_status_codes(status, error):
    with pytest.raises(error, match=f"status={status}"):
        _cpp_gas._raise_status(
            {"status": status, "failure_index": 4}, "test")


def test_cpp_gas_wrapper_rejects_invalid_horizon():
    with pytest.raises(ValueError, match="horizon"):
        _cpp_gas.predict_parameter(
            *PARAMS,
            OBSERVATIONS,
            GumbelCopula(),
            horizon="later",
        )


@pytest.mark.parametrize(
    "copula",
    [
        EquicorrGaussianCopula(d=3),
        StochasticStudentCopula(d=3, R=np.eye(3)),
    ],
)
def test_cpp_gas_wrapper_accepts_multivariate_observations(copula):
    observations = np.random.default_rng(812).uniform(
        0.05, 0.95, size=(12, 3))

    filtered = _cpp_gas.filter_result(
        *PARAMS, observations, copula, scaling="unit")
    update = _cpp_gas.update_one(
        *PARAMS,
        float(filtered.g_path[-1]),
        observations[-1],
        copula,
        scaling="unit",
    )

    assert filtered.g_path.shape == (len(observations),)
    assert filtered.r_path.shape == (len(observations),)
    assert filtered.score_path.shape == (len(observations) - 1,)
    assert np.isfinite(filtered.log_likelihood)
    assert np.all(np.isfinite([
        update.g_next,
        update.r,
        update.r_next,
        update.log_likelihood,
        update.score,
    ]))
