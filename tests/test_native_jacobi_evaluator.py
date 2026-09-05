import warnings

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


@pytest.mark.parametrize("raw_binding", [False, True])
@pytest.mark.parametrize("bad_value", [-0.1, 1.1, np.nan, np.inf, -np.inf])
def test_prepared_evaluator_rejects_invalid_observations(raw_binding, bad_value):
    observations = OBSERVATIONS.copy()
    observations[0, 0] = bad_value
    with pytest.raises(ValueError):
        if raw_binding:
            from pyscarcopula._native import _descriptors

            module = jacobi_native.load()
            config = module.JacobiEvaluatorConfig()
            config.transition.numerical.n_obs = len(observations)
            spec = _descriptors.make_copula_ops_spec(module, GumbelCopula())
            module.PreparedScarJacobiEvaluator(spec, observations, config)
        else:
            jacobi_native.PreparedScarJacobiEvaluator(
                observations, GumbelCopula())


@pytest.mark.parametrize("raw_binding", [False, True])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128, object])
def test_prepared_evaluator_rejects_lossy_complex_coercion(raw_binding, dtype):
    observations = np.array([[np.complex64(0.5 + 1j)] * 2], dtype=dtype)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(TypeError, match="complex"):
            if raw_binding:
                from pyscarcopula._native import _descriptors

                module = jacobi_native.load()
                spec = _descriptors.make_copula_ops_spec(module, GumbelCopula())
                module.PreparedScarJacobiEvaluator(
                    spec, observations, module.JacobiEvaluatorConfig())
            else:
                jacobi_native.PreparedScarJacobiEvaluator(
                    observations, GumbelCopula())
    assert not caught


@pytest.mark.parametrize("raw_binding", [False, True])
@pytest.mark.parametrize("argument", ["tau", "probability", "observation"])
@pytest.mark.parametrize("dtype", [np.complex64, object])
def test_conditioning_rejects_complex_before_cast(raw_binding, argument, dtype):
    values = dict(tau=np.array([0.2, 0.8]), probability=np.array([0.5, 0.5]),
                  observation=np.array([0.5, 0.5]))
    values[argument] = np.array(
        [np.complex64(value + 1j) for value in values[argument]], dtype=dtype)
    evaluator = _evaluator()
    target = evaluator._native if raw_binding else evaluator
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(TypeError, match="complex"):
            target.condition_state(**values)
    assert not caught


@pytest.mark.parametrize("raw_binding", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.int64, object])
def test_conditioning_preserves_real_input_coercion(raw_binding, dtype):
    evaluator = _evaluator()
    tau = np.array([0.2, 0.8], dtype=np.float32)
    probability = np.array([1, 1], dtype=dtype)
    observation = np.array([0, 1], dtype=dtype)
    if raw_binding:
        result = evaluator._native.condition_state(tau, probability, observation)
        assert result["status"] == 0
        actual = result["probability"]
    else:
        _, actual = evaluator.condition_state(tau, probability, observation)
    np.testing.assert_array_equal(actual, [0.5, 0.5])


@pytest.mark.parametrize("layout", ["strided", "fortran", "read_only"])
def test_prepared_evaluator_preserves_supported_observation_buffers(layout):
    observations = {
        "strided": np.repeat(OBSERVATIONS, 2, axis=0)[::2],
        "fortran": np.asfortranarray(OBSERVATIONS),
        "read_only": OBSERVATIONS.copy(),
    }[layout]
    if layout == "read_only":
        observations.flags.writeable = False
    before = observations.copy()
    evaluator = jacobi_native.PreparedScarJacobiEvaluator(
        observations, GumbelCopula(), basis_order=4, quad_order=16, gh_order=3)
    assert evaluator.loglik(*PARAMS) == _evaluator().loglik(*PARAMS)
    np.testing.assert_array_equal(observations, before)


@pytest.mark.parametrize("raw_binding", [False, True])
@pytest.mark.parametrize("tau,probability,observation", [
    ([-0.1, 0.8], [0.5, 0.5], [0.5, 0.5]),
    ([0.2, 1.1], [0.5, 0.5], [0.5, 0.5]),
    ([0.8, 0.2], [0.5, 0.5], [0.5, 0.5]),
    ([0.2, 0.2], [0.5, 0.5], [0.5, 0.5]),
    ([0.2, 0.8], [0.0, 0.0], [0.5, 0.5]),
    ([0.2, 0.8], [-0.1, 1.1], [0.5, 0.5]),
    ([0.2, 0.8], [np.finfo(float).max] * 2, [0.5, 0.5]),
    ([0.2, 0.8], [0.5, 0.5], [-0.1, 0.5]),
    ([0.2, 0.8], [0.5, 0.5], [0.5, 1.1]),
])
def test_conditioning_rejects_invalid_state_or_observation(
        raw_binding, tau, probability, observation):
    evaluator = _evaluator()
    if raw_binding:
        result = evaluator._native.condition_state(
            np.array(tau), np.array(probability), np.array(observation))
        assert result["status"] != 0
        assert len(result["probability"]) == 0
    else:
        with pytest.raises(ValueError):
            evaluator.condition_state(tau, probability, observation)


@pytest.mark.parametrize("scale", [0.2, 2.0, 1e-200, 1e200, 1e-320])
def test_singleton_evaluator_can_condition_scaled_state_without_mutation(scale):
    evaluator = jacobi_native.PreparedScarJacobiEvaluator(
        [[0.5, 0.5]], GumbelCopula())
    tau = np.array([0.2, 0.8])
    probability = np.array([0.5, 0.5]) * scale
    before = probability.copy()
    _, reference = evaluator.condition_state(tau, [0.5, 0.5], [0.5, 0.5])
    actual_tau, actual = evaluator.condition_state(tau, probability, [0.5, 0.5])
    np.testing.assert_array_equal(actual_tau, tau)
    np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(actual.sum(), 1.0, rtol=0.0, atol=2e-15)
    np.testing.assert_array_equal(probability, before)


def test_conditioning_zero_likelihood_keeps_a_normalized_prior():
    evaluator = _evaluator()
    _, probability = evaluator.condition_state([0.2, 0.8], [2.0, 2.0], [0.0, 1.0])
    np.testing.assert_array_equal(probability, [0.5, 0.5])


@pytest.mark.parametrize("scale", [1.0, 1e300, 1e-320])
@pytest.mark.parametrize("observation", [[0.5, 0.5], [0.31, 0.79]])
def test_conditioning_matches_closed_form_gumbel_bayes_update(scale, observation):
    tau = np.array([0.2, 0.8, 0.9])
    probability = np.array([0.3, 0.7, 0.0]) * scale
    # Differentiate C(u,v) = exp(-((-log(u))**theta
    #                            + (-log(v))**theta)**(1/theta)).
    # This oracle does not call production density or tau-mapping kernels.
    theta = 1.0 / (1.0 - tau)
    x, y = -np.log(observation)
    power_sum = x**theta + y**theta
    radius = power_sum**(1.0 / theta)
    log_density = (
        -radius + x + y
        + (theta - 1.0) * (np.log(x) + np.log(y))
        + (2.0 / theta - 2.0) * np.log(power_sum)
        + np.log1p((theta - 1.0) / radius)
    )
    expected = (probability / probability.sum()) * np.exp(log_density)
    expected /= expected.sum()

    _, actual = _evaluator().condition_state(tau, probability, observation)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
    assert actual[-1] == 0.0


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
