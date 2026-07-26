"""Phase 9.8 convergence and static joint factor MLE contracts."""

import inspect
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from pyscarcopula import (
    FactorCorrelation,
    FactorStudentEvaluator,
    NumericalConfig,
    StochasticStudentCopula,
)
from pyscarcopula.api import fit as api_fit
from pyscarcopula.contrib.risk_metrics import _get_copula_constructor
from pyscarcopula.copula.multivariate.factor_estimation import (
    FactorLoadingParameterization,
)


def _problem(rows=180, d=6, rank=2, seed=9801):
    rng = np.random.default_rng(seed)
    loadings = rng.normal(scale=0.16, size=(d, rank))
    source = StochasticStudentCopula(
        d,
        corr_mode="factor",
        factor_rank=rank,
        factor_loadings=loadings,
    )
    observations = source.sample_at_parameter(
        rows, 6.5, rng=rng)
    return observations, loadings


def test_native_joint_gradient_matches_finite_differences():
    rng = np.random.default_rng(9802)
    loadings = rng.normal(scale=0.08, size=(6, 2))
    observations = rng.uniform(0.05, 0.95, size=(25, 6))
    df = 5.7
    epsilon = 1e-6
    evaluator = FactorStudentEvaluator(
        FactorCorrelation(loadings), observations)

    actual = evaluator.joint_likelihood_and_gradient(
        df, n_threads=4)
    numerical = np.empty_like(loadings)
    for row in range(loadings.shape[0]):
        for column in range(loadings.shape[1]):
            plus = loadings.copy()
            minus = loadings.copy()
            plus[row, column] += epsilon
            minus[row, column] -= epsilon
            plus_value = FactorStudentEvaluator(
                FactorCorrelation(plus), observations).evaluate(
                    df).log_likelihood
            minus_value = FactorStudentEvaluator(
                FactorCorrelation(minus), observations).evaluate(
                    df).log_likelihood
            numerical[row, column] = (
                plus_value - minus_value) / (2.0 * epsilon)

    plus_df = evaluator.evaluate(df + epsilon).log_likelihood
    minus_df = evaluator.evaluate(df - epsilon).log_likelihood
    numerical_df = (plus_df - minus_df) / (2.0 * epsilon)
    one_thread = evaluator.joint_likelihood_and_gradient(
        df, n_threads=1)

    np.testing.assert_allclose(
        actual.dlog_likelihood_dloadings,
        numerical,
        rtol=2e-7,
        atol=2e-8,
    )
    assert actual.dlog_likelihood_ddf == pytest.approx(
        numerical_df, rel=3e-8, abs=2e-8)
    np.testing.assert_array_equal(
        actual.dlog_likelihood_dloadings,
        one_thread.dlog_likelihood_dloadings,
    )
    assert (
        actual.dlog_likelihood_ddf
        == one_thread.dlog_likelihood_ddf
    )
    assert actual.diagnostics["gradient_kind"] == "analytical"


def test_identifiable_parameterization_and_pullback():
    rng = np.random.default_rng(9803)
    loadings = rng.normal(scale=0.1, size=(8, 3))
    parameterization, values = (
        FactorLoadingParameterization.from_loadings(
            loadings, uniqueness_min=1e-8))
    canonical = parameterization.loadings(values)
    gradient = rng.normal(size=canonical.shape)
    analytical = parameterization.pullback(values, gradient)
    epsilon = 1e-6
    numerical = np.empty_like(values)
    for index in range(len(values)):
        plus = values.copy()
        minus = values.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        numerical[index] = (
            np.sum(parameterization.loadings(plus) * gradient)
            - np.sum(parameterization.loadings(minus) * gradient)
        ) / (2.0 * epsilon)

    assert parameterization.n_parameters == 8 * 3 - 3
    for order, row in enumerate(parameterization.anchors):
        assert np.all(canonical[row, order + 1:] == 0.0)
        assert canonical[row, order] > 0.0
    assert np.min(
        1.0 - np.sum(canonical * canonical, axis=1)) >= 1e-8
    np.testing.assert_allclose(
        analytical, numerical, rtol=2e-8, atol=2e-9)


@pytest.mark.parametrize("n_threads", [1, 4])
def test_joint_static_mle_improves_objective_and_stays_compact(n_threads):
    observations, loadings = _problem(seed=9804)
    start = loadings + np.random.default_rng(9805).normal(
        scale=0.06, size=loadings.shape)
    model = StochasticStudentCopula(
        6,
        corr_mode="factor",
        factor_rank=2,
        factor_loadings=start,
        factor_estimation="joint",
        factor_joint_penalty=1e-6,
    )

    result = model.fit(
        observations,
        method="mle",
        maxiter=180,
        config=NumericalConfig(n_threads=n_threads),
    )

    assert result.success
    assert result.correlation_matrix is None
    assert result.parameter_count == 1 + 6 * 2 - 1
    assert result.diagnostics["gradient_mode"] == (
        "analytical_joint_factor")
    assert result.diagnostics["joint_identification"] == (
        "pivoted_lower_triangular_positive_diag")
    assert result.diagnostics["joint_final_objective"] < (
        result.diagnostics["joint_initial_objective"])
    assert result.diagnostics["joint_gradient_inf_norm"] <= (
        result.diagnostics["joint_gradient_gate"])
    assert result.diagnostics["n_threads"] == n_threads
    assert model._R is None
    assert model._L is None
    assert model._L_inv is None
    assert np.all(model.factor_uniqueness_ >= 1e-8)


def test_joint_fit_without_supplied_loadings_uses_tiled_start():
    observations, _ = _problem(rows=160, d=5, seed=9806)
    model = StochasticStudentCopula(
        5,
        corr_mode="factor",
        factor_rank=2,
        factor_estimation="joint",
        factor_tile_size=2,
        factor_seed=17,
    )

    result = model.fit(observations, method="mle", maxiter=120)

    assert np.isfinite(result.log_likelihood)
    assert (
        result.diagnostics["initialization_joint_start_source"]
        == "joint_randomized_svd_start"
    )
    assert result.model_parameters["factor_estimation"] == "joint"
    assert model._R is None


def test_optimizer_success_is_rejected_when_joint_gradient_is_large(
        monkeypatch):
    observations, loadings = _problem(rows=60, d=5, seed=9810)

    class DummyResult:
        success = True
        message = "synthetic success"
        nfev = 1

    def fake_minimize(fun, x0, **kwargs):
        del kwargs
        fun(x0)
        result = DummyResult()
        result.x = np.asarray(x0).copy()
        return result

    monkeypatch.setattr(
        "pyscarcopula.copula.multivariate.stochastic_student.minimize",
        fake_minimize,
    )
    model = StochasticStudentCopula(
        5,
        corr_mode="factor",
        factor_rank=2,
        factor_loadings=loadings,
        factor_estimation="joint",
    )

    result = model.fit(observations, method="mle")

    assert not result.success
    assert "rejected by joint factor gradient gate" in result.message
    assert result.diagnostics["joint_gradient_inf_norm"] > (
        result.diagnostics["joint_gradient_gate"])


@pytest.mark.parametrize(
    "method", ["gas", "scar-tm-ou", "scar-p-ou", "scar-m-ou"])
@pytest.mark.parametrize("entrypoint", ["model", "api"])
def test_dynamic_joint_factor_fit_is_rejected_before_state_mutation(
        method, entrypoint):
    observations, _ = _problem(rows=30, d=5, seed=9807)
    model = StochasticStudentCopula(
        5,
        corr_mode="factor",
        factor_rank=2,
        factor_estimation="joint",
    )

    with pytest.raises(
            NotImplementedError,
            match="factor_estimation='joint'.*static MLE"):
        if entrypoint == "model":
            model.fit(observations, method=method)
        else:
            api_fit(model, observations, method=method)

    assert model.fit_result is None
    assert model.factor_loadings_ is None
    assert getattr(model, "_last_u", None) is None


def test_joint_policy_and_safety_gates_survive_rolling_reconstruction():
    observations, loadings = _problem(rows=40, d=5, seed=9808)
    del observations
    source = StochasticStudentCopula(
        5,
        corr_mode="factor",
        factor_rank=2,
        factor_loadings=loadings,
        factor_estimation="joint",
        factor_joint_max_params=25,
        factor_joint_penalty=2e-5,
        factor_joint_condition_max=1e8,
        factor_seed=13,
    )

    model_type, kwargs = _get_copula_constructor(source)
    rebuilt = model_type(**kwargs)

    assert rebuilt.factor_estimation == "joint"
    assert rebuilt.factor_joint_max_params == 25
    assert rebuilt.factor_joint_penalty == 2e-5
    assert rebuilt.factor_joint_condition_max == 1e8
    np.testing.assert_array_equal(
        rebuilt.factor_loadings_, loadings)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"factor_joint_penalty": -1.0}, "penalty"),
        ({"factor_joint_condition_max": 1.0}, "condition"),
    ],
)
def test_joint_safety_configuration_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        StochasticStudentCopula(
            5,
            corr_mode="factor",
            factor_rank=2,
            factor_estimation="joint",
            **kwargs,
        )


def test_joint_fit_persistence_remains_compact(tmp_path):
    observations, loadings = _problem(rows=120, d=5, seed=9809)
    model = StochasticStudentCopula(
        5,
        corr_mode="factor",
        factor_rank=2,
        factor_loadings=loadings,
        factor_estimation="joint",
    )
    result = model.fit(observations, method="mle", maxiter=100)
    target = tmp_path / "joint-factor.json"

    model.save(target)
    restored = StochasticStudentCopula.load(target)

    assert restored.fit_result.correlation_matrix is None
    assert restored.factor_estimation == "joint"
    assert restored._R is None
    assert restored.log_likelihood(
        observations, restored.fit_result.copula_param
    ) == pytest.approx(result.log_likelihood, rel=0.0, abs=1e-13)


def test_joint_evaluator_default_is_one_thread_and_does_not_start_pool():
    parameter = inspect.signature(
        FactorStudentEvaluator.joint_likelihood_and_gradient
    ).parameters["n_threads"]
    assert parameter.default == 1

    root = Path(__file__).resolve().parents[1]
    code = (
        "import json, numpy as np\n"
        "from pyscarcopula import FactorCorrelation, FactorStudentEvaluator\n"
        "from pyscarcopula.numerical import _cpp_extension\n"
        "m = _cpp_extension.load()\n"
        "before = dict(m._parallel_runtime_info())\n"
        "factor = FactorCorrelation(np.full((1024, 4), 0.01))\n"
        "e = FactorStudentEvaluator("
        "factor, np.full((8, 1024), 0.5))\n"
        "e.joint_likelihood_and_gradient(6.0)\n"
        "after = dict(m._parallel_runtime_info())\n"
        "print(json.dumps({'before': before, 'after': after}))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    payload = json.loads(completed.stdout)
    assert payload["before"]["initialized"] is False
    assert payload["after"]["initialized"] is False


def test_joint_gradient_large_dimension_has_no_dense_state():
    dimension = 10000
    rank = 2
    factor = FactorCorrelation(
        np.full((dimension, rank), 0.002))
    observations = np.full((4, dimension), 0.5)

    result = FactorStudentEvaluator(
        factor, observations).joint_likelihood_and_gradient(
            6.0, n_threads=2)

    assert result.dlog_likelihood_dloadings.shape == (
        dimension, rank)
    assert np.all(np.isfinite(
        result.dlog_likelihood_dloadings))
    assert result.diagnostics["reduction_workspace_bytes"] == (
        len(observations) * dimension * rank * 8)
