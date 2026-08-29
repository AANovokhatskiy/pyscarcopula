from pathlib import Path

import numpy as np
import pytest

from pyscarcopula import GumbelCopula
from pyscarcopula._native import jacobi as jacobi_native


PARAMS = np.array([1.2, 0.4, 0.25], dtype=np.float64)
OBSERVATIONS = np.array([
    [0.12, 0.23],
    [0.21, 0.74],
    [0.33, 0.45],
    [0.48, 0.61],
    [0.57, 0.32],
    [0.69, 0.83],
    [0.77, 0.54],
    [0.88, 0.91],
], dtype=np.float64)


def _evaluator(**kwargs):
    options = {
        "basis_order": 4,
        "quad_order": 16,
        "gh_order": 3,
    }
    options.update(kwargs)
    return jacobi_native.PreparedScarJacobiEvaluator(
        OBSERVATIONS, GumbelCopula(), **options)


def test_prepared_evaluator_reuses_setup_filter_and_observation_cache():
    evaluator = _evaluator(
        transition_method="local_fixed", storage="dense")

    assert evaluator.preparation_count == 0
    state = evaluator.filter(*PARAMS)
    loglik = evaluator.loglik(*PARAMS)
    mean = evaluator.predictive_mean(*PARAMS)
    mixture = evaluator.mixture_h_pair(*PARAMS)

    assert evaluator.preparation_count == 1
    assert loglik == pytest.approx(
        state["diagnostics"]["log_likelihood"], rel=0.0, abs=1e-14)
    assert mean.shape == (len(OBSERVATIONS),)
    assert mixture[0].shape == mixture[1].shape == (len(OBSERVATIONS),)

    evaluator.loglik(PARAMS[0] + 0.01, PARAMS[1], PARAMS[2])
    assert evaluator.preparation_count == 2


def test_prepared_loglik_preserves_native_invalid_parameter_sentinel():
    evaluator = _evaluator(
        transition_method="local_fixed", storage="dense")

    assert evaluator.loglik(-1.0, 0.4, 0.25) == -np.inf
    assert evaluator.preparation_count == 0


def test_prepared_state_distribution_preserves_frozen_invalid_domain_error():
    evaluator = _evaluator(
        transition_method="local_fixed", storage="dense")

    with pytest.raises(
            ValueError,
            match="^Jacobi stationary shape is outside supported range$") as error:
        evaluator.state_distribution(-1.0, 0.4, 0.25)
    assert error.value.status == 6
    assert error.value.operation == "prepared state distribution"
    assert error.value.failure_index == -1
    assert error.value.context.diagnostics["preparation_generation"] == 0


def test_prepared_filter_smoother_state_and_diagnostics_contract():
    evaluator = _evaluator(transition_method="auto", storage="dense")
    state = evaluator.filter(*PARAMS)

    assert state["tau"].shape == state["theta"].shape == (16,)
    assert state["emissions"].shape == (8, 16)
    assert state["predicted"].shape == (8, 16)
    assert state["filtered"].shape == (8, 16)
    assert state["smoothed"].shape == (8, 16)
    np.testing.assert_allclose(state["predicted"].sum(axis=1), 1.0, atol=2e-12)
    np.testing.assert_allclose(state["filtered"].sum(axis=1), 1.0, atol=2e-12)
    np.testing.assert_allclose(state["smoothed"].sum(axis=1), 1.0, atol=2e-12)
    np.testing.assert_allclose(state["current_probability"].sum(), 1.0)
    np.testing.assert_allclose(state["next_probability"].sum(), 1.0)
    assert state["diagnostics"]["transition_method_requested"] == "auto"
    assert state["diagnostics"]["transition_method"] == "local"
    assert state["diagnostics"]["preparation_generation"] == 1

    current_tau, current = evaluator.state_distribution(
        *PARAMS, horizon="current")
    next_tau, next_probability = evaluator.state_distribution(
        *PARAMS, horizon="next")
    np.testing.assert_array_equal(current_tau, next_tau)
    np.testing.assert_allclose(current, state["current_probability"])
    np.testing.assert_allclose(next_probability, state["next_probability"])


@pytest.mark.parametrize(
    "options",
    [
        {"transition_method": "spectral_coeff", "storage": "dense"},
        {"transition_method": "auto", "storage": "dense"},
        {
            "transition_method": "spectral_matrix",
            "storage": "dense",
            "clip_negative": True,
        },
        {"transition_method": "local", "storage": "dense"},
        {"transition_method": "local_fixed", "storage": "dense"},
        {"transition_method": "local", "storage": "sparse"},
        {
            "transition_method": "local",
            "storage": "sparse",
            "correction": "mh",
        },
        {
            "transition_method": "local",
            "storage": "sparse",
            "correction": "ipfp",
        },
        {"transition_method": "local_fixed", "storage": "sparse"},
    ],
    ids=[
        "spectral-coeff",
        "auto-dense",
        "spectral-matrix",
        "local-dense",
        "local-fixed-dense",
        "local-sparse",
        "local-sparse-mh",
        "local-sparse-ipfp",
        "local-fixed-sparse",
    ],
)
def test_prepared_gradient_matches_physical_finite_difference(options):
    evaluator = _evaluator(**options)
    value, gradient = evaluator.neg_loglik_with_grad(*PARAMS)
    expected = np.empty(3, dtype=np.float64)
    for parameter in range(3):
        step = 1e-5 * max(abs(PARAMS[parameter]), 1.0)
        if parameter == 1:
            step = min(
                step,
                0.49 * PARAMS[parameter],
                0.49 * (1.0 - PARAMS[parameter]),
            )
        else:
            step = min(step, 0.49 * PARAMS[parameter])
        plus = PARAMS.copy()
        minus = PARAMS.copy()
        plus[parameter] += step
        minus[parameter] -= step
        expected[parameter] = -(
            evaluator.loglik(*plus) - evaluator.loglik(*minus)
        ) / (2.0 * step)

    assert np.isfinite(value)
    assert value == pytest.approx(-_evaluator(**options).loglik(*PARAMS))
    np.testing.assert_allclose(gradient, expected, rtol=3e-3, atol=3e-4)


def test_prepared_rosenblatt_and_conditioning_are_native_state_operations():
    evaluator = _evaluator(transition_method="local", storage="sparse")
    residual = evaluator.rosenblatt(*PARAMS)
    gaussian = evaluator.rosenblatt(*PARAMS, gaussian=True)

    assert residual.shape == gaussian.shape == OBSERVATIONS.shape
    np.testing.assert_array_equal(residual[:, 0], OBSERVATIONS[:, 0])
    assert np.all((residual > 0.0) & (residual < 1.0))
    assert np.all(np.isfinite(gaussian))

    tau, probability = evaluator.state_distribution(*PARAMS)
    conditioned_tau, conditioned = evaluator.condition_state(
        tau, probability, np.array([0.31, 0.79]))
    np.testing.assert_array_equal(conditioned_tau, tau)
    np.testing.assert_allclose(conditioned.sum(), 1.0)
    assert not np.array_equal(conditioned, probability)


@pytest.mark.parametrize(
    "operation",
    [
        "filter",
        "loglik",
        "neg_loglik",
        "neg_loglik_with_grad",
        "predictive_mean",
        "mixture_h",
        "mixture_h_pair",
        "rosenblatt",
        "gaussian_rosenblatt",
        "state_distribution",
    ],
)
def test_prepared_evaluator_reports_combined_workspace_failure(operation):
    evaluator = _evaluator(
        transition_method="local",
        storage="sparse",
        memory_budget_bytes=4096,
    )
    with pytest.raises(MemoryError, match="memory_budget_bytes"):
        if operation == "gaussian_rosenblatt":
            evaluator.rosenblatt(*PARAMS, gaussian=True)
        else:
            getattr(evaluator, operation)(*PARAMS)


def test_python_numerical_modules_do_not_own_jacobi_filter_math():
    root = Path(__file__).resolve().parents[1]
    dense = (root / "pyscarcopula/numerical/jacobi_tm.py").read_text(
        encoding="utf-8")
    sparse = (root / "pyscarcopula/numerical/jacobi_sparse.py").read_text(
        encoding="utf-8")

    for forbidden in (
        "_iter_matrix_filter",
        "_matrix_setup_fd_derivatives",
        "_normalize_prob_mass_with_derivatives",
        "_iter_coeff_filter",
    ):
        assert forbidden not in dense
    for forbidden in (
        "_sparse_filter_loglik_kernel",
        "_sparse_neg_loglik_grad_kernel",
        "_iter_sparse_filter",
        "_sparse_filter_setup",
    ):
        assert forbidden not in sparse
