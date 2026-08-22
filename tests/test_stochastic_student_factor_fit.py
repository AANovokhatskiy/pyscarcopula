"""End-to-end fitting contracts for factor Student models."""

import numpy as np
import pytest

from pyscarcopula import NumericalConfig, StochasticStudentCopula
from pyscarcopula.contrib.risk_metrics import _get_copula_constructor
from pyscarcopula.numerical import (
    _cpp_copula,
    _cpp_extension,
    _cpp_scar_ou,
    static_likelihood,
)
from pyscarcopula.numerical._scar_ou_config import AutoTMConfig
from pyscarcopula.numerical.gas_filter import gas_loglik
from pyscarcopula.numerical.mc_native import log_pdf_trajectory_grid


def _problem(rows=24, d=4, k=2):
    rng = np.random.default_rng(9501)
    observations = rng.uniform(0.08, 0.92, size=(rows, d))
    loadings = rng.normal(scale=0.09, size=(d, k))
    return observations, loadings


def _models(rows=24):
    observations, loadings = _problem(rows=rows)
    factor = StochasticStudentCopula(
        4,
        corr_mode="factor",
        factor_rank=2,
        factor_loadings=loadings,
        factor_tile_size=2,
    )
    dense = StochasticStudentCopula(
        4, R=factor.to_correlation_matrix())
    return observations, factor, dense


def test_native_factor_spec_shares_operator_without_dense_cholesky():
    observations, factor, _ = _models()
    module = _cpp_extension.load()

    spec = _cpp_copula.make_spec(module, factor, u=observations)

    assert spec.correlation_kind == module.CorrelationKind.Factor
    assert spec.factor_correlation.dimension == factor.d
    assert spec.factor_correlation.rank == factor.factor_rank
    assert spec.factor_dimension_tile == factor.factor_tile_size
    assert len(spec.l_inv) == 0
    assert factor._R is None
    assert factor._L is None
    assert factor._L_inv is None


def test_factor_static_mle_returns_compact_result():
    observations, factor, _ = _models(rows=40)

    result = factor.fit(observations, method="mle", maxiter=50)

    assert result.success
    assert np.isfinite(result.log_likelihood)
    assert result.correlation_matrix is None
    assert result.n_params == 8
    assert result.diagnostics["corr_mode"] == "factor"
    assert result.diagnostics["factor_n_params"] == 7
    assert "corr_matrix" not in result.diagnostics
    assert "correlation_matrix" not in result.model_parameters
    np.testing.assert_allclose(
        result.model_parameters["factor_loadings"],
        factor.factor_loadings_,
    )
    assert factor._R is None


def test_generic_native_static_evaluator_accepts_factor_spec():
    observations, factor, _ = _models(rows=16)
    expected_log_pdf, expected_gradient = (
        factor.log_pdf_and_dlog_dr_rows(
            observations, 5.5, n_threads=2))

    evaluator = static_likelihood.prepare(
        factor, observations, n_threads=2)
    value, gradient = evaluator.objective_and_gradient(5.5)

    assert value == pytest.approx(
        -np.sum(expected_log_pdf), rel=2e-12, abs=2e-12)
    np.testing.assert_allclose(
        gradient,
        [-np.sum(expected_gradient)],
        rtol=2e-11,
        atol=2e-11,
    )


def test_factor_static_mle_performs_two_stage_initialization():
    observations, _ = _problem(rows=36, d=6, k=2)
    model = StochasticStudentCopula(
        6,
        corr_mode="factor",
        factor_rank=2,
        factor_tile_size=3,
        factor_seed=19,
        factor_oversampling=3,
    )

    result = model.fit(observations, method="mle", maxiter=100)

    assert result.success
    assert np.isfinite(result.log_likelihood)
    assert model.factor_loadings_ is not None
    assert model._R is None
    assert (
        model.factor_diagnostics()["initialization_source"]
        == "two_stage_randomized_svd"
    )


def test_fitted_factor_model_persistence_stays_compact(tmp_path):
    observations, factor, _ = _models(rows=32)
    fitted = factor.fit(observations, method="mle", maxiter=30)
    path = tmp_path / "fitted-factor-student.json"

    factor.save(path)
    restored = StochasticStudentCopula.load(path)

    assert restored.fit_result.correlation_matrix is None
    assert restored.fit_result.copula_param == pytest.approx(
        fitted.copula_param)
    assert restored._R is None
    np.testing.assert_allclose(
        restored.factor_loadings_, factor.factor_loadings_)
    assert restored.log_likelihood(
        observations, restored.fit_result.copula_param
    ) == pytest.approx(
        factor.log_likelihood(observations, fitted.copula_param),
        rel=0.0,
        abs=0.0,
    )


def test_single_thread_rolling_workers_keep_factor_fit_state_isolated():
    observations, model, _ = _models(rows=28)
    model_type, constructor = _get_copula_constructor(model)
    results = []

    for start in range(3):
        worker = model_type(**constructor)
        results.append(worker.fit(
            observations[start:start + 20],
            method="mle",
            maxiter=10,
            config=NumericalConfig(n_threads=1),
        ))
        assert worker._R is None
        np.testing.assert_allclose(
            worker.factor_loadings_, model.factor_loadings_)

    assert model.fit_result is None
    assert all(np.isfinite(result.log_likelihood) for result in results)
    assert all(
        result.diagnostics["n_threads"] == 1 for result in results)


@pytest.mark.parametrize("scaling", ["unit", "fisher"])
def test_factor_gas_likelihood_matches_dense_reference(scaling):
    observations, factor, dense = _models()
    parameters = (0.08, 0.04, 0.82)

    factor_value = gas_loglik(
        *parameters, observations, factor, scaling=scaling)
    dense_value = gas_loglik(
        *parameters, observations, dense, scaling=scaling)

    assert factor_value == pytest.approx(
        dense_value, rel=3e-6, abs=3e-6)
    assert factor._R is None


def test_factor_gas_fit_is_native_and_counts_estimated_loadings():
    observations, factor, _ = _models()

    result = factor.fit(
        observations,
        method="gas",
        gamma0=np.array([0.08, 0.04, 0.82]),
        maxiter=2,
        maxfun=30,
        config=NumericalConfig(n_threads=2),
    )

    assert np.isfinite(result.log_likelihood)
    assert result.log_likelihood > -1e9
    assert result.n_params == 10
    assert result.diagnostics["model_score"] == "native"
    assert result.diagnostics["corr_mode"] == "factor"
    assert result.diagnostics["n_threads"] == 2
    assert "corr_matrix" not in result.diagnostics
    assert factor._R is None


@pytest.mark.parametrize(
    "transition_method",
    ["matrix", "local", "spectral"],
)
def test_factor_scar_objective_and_gradient_match_dense_reference(
        transition_method):
    observations, factor, dense = _models(rows=12)
    config = AutoTMConfig(
        transition_method=transition_method,
        K=10,
        max_K=10,
        adaptive=False,
        basis_order=10,
        quad_order=10,
        n_threads=2,
    )
    parameters = (1.1, 0.25, 0.75)

    factor_value, factor_gradient, _ = (
        _cpp_scar_ou.neg_loglik_with_grad_info(
            *parameters, observations, factor, config))
    dense_value, dense_gradient, _ = (
        _cpp_scar_ou.neg_loglik_with_grad_info(
            *parameters, observations, dense, config))

    assert factor_value == pytest.approx(
        dense_value, rel=2e-5, abs=2e-5)
    np.testing.assert_allclose(
        factor_gradient,
        dense_gradient,
        rtol=4e-4,
        atol=4e-5,
    )
    assert factor._R is None


def test_factor_scar_fit_uses_prepared_native_evaluator():
    observations, factor, _ = _models(rows=20)

    result = factor.fit(
        observations,
        method="scar-tm-ou",
        alpha0=np.array([1.0, 0.25, 0.8]),
        K=8,
        max_K=8,
        adaptive=False,
        transition_method="matrix",
        analytical_grad=True,
        smart_init=False,
        maxiter=1,
        maxfun=20,
        config=NumericalConfig(n_threads=2),
    )

    assert np.isfinite(result.log_likelihood)
    assert result.log_likelihood > -1e9
    assert result.n_params == 10
    assert result.diagnostics["selected_engine"] == "cpp"
    assert result.diagnostics["prepared_native_evaluator"] is True
    assert result.diagnostics["corr_mode"] == "factor"
    assert result.diagnostics["n_threads"] == 2
    assert factor._R is None


def test_factor_scar_mc_trajectory_density_matches_dense_reference():
    observations, factor, dense = _models(rows=10)
    latent_paths = np.random.default_rng(9502).normal(
        0.2, 0.4, size=(len(observations), 7))

    factor_values = log_pdf_trajectory_grid(
        factor, observations, latent_paths, n_threads=2)
    dense_values = log_pdf_trajectory_grid(
        dense, observations, latent_paths, n_threads=2)

    np.testing.assert_allclose(
        factor_values,
        dense_values,
        rtol=3e-6,
        atol=3e-6,
    )
    assert factor._R is None


def test_factor_scar_p_fit_uses_native_trajectory_kernel():
    observations, factor, _ = _models(rows=8)

    result = factor.fit(
        observations,
        method="scar-p-ou",
        alpha0=np.array([1.0, 0.2, 0.7]),
        n_tr=4,
        maxiter=1,
        maxfun=10,
        seed=9503,
        config=NumericalConfig(n_threads=2),
    )

    assert np.isfinite(result.log_likelihood)
    assert result.log_likelihood > -1e9
    assert result.method == "SCAR-P-OU"
    assert result.n_params == 10
    assert result.diagnostics["corr_mode"] == "factor"
    assert result.diagnostics["n_threads"] == 2
    assert factor._R is None


def test_factor_scar_results_are_deterministic_across_thread_counts():
    observations, factor, _ = _models(rows=16)
    parameters = (1.0, 0.2, 0.7)

    def evaluate(n_threads):
        config = AutoTMConfig(
            transition_method="matrix",
            K=8,
            max_K=8,
            adaptive=False,
            n_threads=n_threads,
        )
        return _cpp_scar_ou.neg_loglik_with_grad_info(
            *parameters, observations, factor, config)

    sequential = evaluate(1)
    parallel = evaluate(2)

    assert parallel[0] == pytest.approx(
        sequential[0], rel=0.0, abs=2e-13)
    np.testing.assert_allclose(
        parallel[1], sequential[1], rtol=0.0, atol=2e-13)


def test_factor_scar_large_dimension_uses_scaled_emissions():
    rng = np.random.default_rng(9510)
    dimension = 10_000
    observations = rng.uniform(0.08, 0.92, size=(6, dimension))
    loadings = rng.normal(scale=0.01, size=(dimension, 4))
    factor = StochasticStudentCopula(
        dimension,
        corr_mode="factor",
        factor_rank=4,
        factor_loadings=loadings,
        factor_tile_size=512,
    )
    config = AutoTMConfig(
        transition_method="matrix",
        K=8,
        max_K=8,
        adaptive=False,
        n_threads=2,
    )
    parameters = np.array([1.1, 0.2, 0.7])

    value, gradient, info = _cpp_scar_ou.neg_loglik_with_grad_info(
        *parameters, observations, factor, config)

    assert info["status"] == 0
    assert np.isfinite(value)
    assert np.all(np.isfinite(gradient))
    assert factor._R is None

    step = 1e-5
    finite_difference = np.empty(3)
    for index in range(3):
        delta = np.zeros(3)
        delta[index] = step
        plus = _cpp_scar_ou.neg_loglik_info(
            *(parameters + delta), observations, factor, config)[0]
        minus = _cpp_scar_ou.neg_loglik_info(
            *(parameters - delta), observations, factor, config)[0]
        finite_difference[index] = (plus - minus) / (2.0 * step)

    np.testing.assert_allclose(
        gradient, finite_difference, rtol=5e-7, atol=5e-5)
