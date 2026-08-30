"""Acceptance tests for the native-only SCAR-TM-OU strategy."""

import ast
import inspect

import numpy as np
import pytest

from pyscarcopula.copula.clayton import ClaytonCopula
from pyscarcopula.copula.frank import FrankCopula
from pyscarcopula.copula.gumbel import GumbelCopula
from pyscarcopula.copula.independent import IndependentCopula
from pyscarcopula.copula.joe import JoeCopula
from pyscarcopula.copula.elliptical import BivariateGaussianCopula
from pyscarcopula.copula.multivariate.equicorr import (
    EquicorrGaussianCopula,
)
from pyscarcopula.copula.multivariate import StochasticStudentCopula
from pyscarcopula._native import scar_ou as _cpp_scar_ou
from pyscarcopula._native import NativeError
from pyscarcopula.numerical._scar_ou_config import AutoTMConfig
from pyscarcopula import stattests
from pyscarcopula.strategy import scar_tm
from pyscarcopula.strategy.scar_tm import SCARTMStrategy


_MATRIX_CONFIG = AutoTMConfig(
    transition_method="matrix",
    K=24,
    adaptive=False,
    max_K=24,
)


@pytest.mark.parametrize(
    "transition_method", ["auto", "spectral", "matrix", "local"])
def test_native_ou_objectives_require_two_observations(transition_method):
    copula = BivariateGaussianCopula()
    u = np.array([[0.25, 0.75]])
    config = AutoTMConfig(
        transition_method=transition_method,
        K=16,
        adaptive=False,
        max_K=16,
    )

    with pytest.raises(NativeError, match="invalid_size"):
        _cpp_scar_ou.neg_loglik(1.0, 0.0, 0.5, u, copula, config)
    prepared = _cpp_scar_ou.prepare_objective(u, copula, config)
    with pytest.raises(NativeError, match="invalid_size"):
        prepared.neg_loglik_info(1.0, 0.0, 0.5)


@pytest.mark.parametrize(
    "copula",
    [
        family(transform_type=transform)
        for transform in ("softplus", "xtanh", "exp", "logistic")
        for family in (ClaytonCopula, FrankCopula, GumbelCopula, JoeCopula)
    ] + [
        IndependentCopula(),
    ],
)
def test_native_ou_support_covers_bivariate_family_and_transform_combinations(copula):
    u = np.random.default_rng(20260618).uniform(0.05, 0.95, size=(12, 2))

    value, gradient, info = _cpp_scar_ou.neg_loglik_with_grad_info(
        1.2, 0.1, 0.7, u, copula, _MATRIX_CONFIG)

    assert _cpp_scar_ou.supported(copula)
    assert np.isfinite(value)
    assert np.all(np.isfinite(gradient))
    assert info["engine"] == "cpp"


@pytest.mark.parametrize(
    "factory",
    [ClaytonCopula, FrankCopula, GumbelCopula, JoeCopula],
)
@pytest.mark.parametrize("transform_type", ["exp", "logistic"])
def test_new_transform_ou_gradient_matches_finite_difference(
        factory, transform_type):
    copula = factory(transform_type=transform_type)
    u = np.random.default_rng(20260730).uniform(0.08, 0.92, size=(12, 2))
    params = np.array([1.2, 0.1, 0.7])
    value, gradient = _cpp_scar_ou.neg_loglik_with_grad(
        *params, u, copula, _MATRIX_CONFIG)
    finite_difference = np.empty(3)

    for index in range(3):
        step = 1e-5 * max(1.0, abs(params[index]))
        plus = params.copy()
        minus = params.copy()
        plus[index] += step
        minus[index] -= step
        finite_difference[index] = (
            _cpp_scar_ou.neg_loglik(*plus, u, copula, _MATRIX_CONFIG)
            - _cpp_scar_ou.neg_loglik(*minus, u, copula, _MATRIX_CONFIG)
        ) / (2.0 * step)

    assert np.isfinite(value)
    np.testing.assert_allclose(
        gradient, finite_difference, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("transition_method", ["matrix", "local"])
def test_grid_gradient_streams_long_observation_history(transition_method):
    copula = IndependentCopula()
    u = np.full((100_000, 2), 0.5)
    config = AutoTMConfig(
        transition_method=transition_method,
        K=8,
        adaptive=False,
        max_K=8,
        gh_order=5,
    )

    expected = _cpp_scar_ou.neg_loglik(
        2.0, 0.0, 0.5, u, copula, config)
    value, gradient = _cpp_scar_ou.neg_loglik_with_grad(
        2.0, 0.0, 0.5, u, copula, config)

    assert value == pytest.approx(expected, rel=1e-12, abs=1e-8)
    assert np.all(np.isfinite(gradient))


@pytest.mark.parametrize("transition_method", ["matrix", "local"])
def test_student_d10_all_correlation_gradients_match_finite_differences(
        transition_method):
    d = 10
    u = np.random.default_rng(20260830).uniform(0.05, 0.95, (36, d))
    correlation = np.full((d, d), 0.2)
    np.fill_diagonal(correlation, 1.0)
    copula = StochasticStudentCopula(d=d, R=correlation)
    config = AutoTMConfig(
        transition_method=transition_method, K=32, max_K=32,
        adaptive=False, grid_method="dense")
    prepared = _cpp_scar_ou.prepare_objective(u, copula, config)
    params = np.array([1.2, 0.4, 0.8])
    value, ou_gradient, gradient, _ = (
        prepared.neg_loglik_with_grad_and_corr_info(*params))
    expected_value, expected_ou, _ = prepared.neg_loglik_with_grad_info(*params)
    assert gradient.shape == (45,)
    np.testing.assert_allclose(value, expected_value, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(ou_gradient, expected_ou, rtol=0.0, atol=1e-11)
    lower = np.tril_indices(d, -1)
    finite_difference = []
    step = 1e-5
    for i, j in zip(*lower):
        values = []
        for sign in (1, -1):
            trial = correlation.copy()
            trial[i, j] += sign * step
            trial[j, i] += sign * step
            copula._set_R(trial)
            prepared.update_copula(copula)
            values.append(prepared.neg_loglik_info(*params)[0])
        finite_difference.append((values[0] - values[1]) / (2 * step))
    np.testing.assert_allclose(gradient, finite_difference, rtol=2e-7, atol=2e-7)


@pytest.mark.validation
@pytest.mark.parametrize("d,T", [(2, 4097), (10, 8195)])
@pytest.mark.parametrize("transition_method", ["matrix", "local"])
def test_student_checkpoint_gradient_across_emission_blocks(d, T, transition_method):
    from pyscarcopula.copula.multivariate.student_ppf_cache import (
        StudentPPFTable, prepare_student_ppf_cache,
    )

    rng = np.random.default_rng(20260830 + d)
    u = rng.uniform(0.08, 0.92, (T, d))
    correlation = np.full((d, d), 0.15)
    np.fill_diagonal(correlation, 1.0)
    copula = StochasticStudentCopula(d=d, R=correlation)
    # A small table keeps this long-history test about gradient workspace,
    # rather than the separately budgeted PPF cache.
    copula._ppf_cache = prepare_student_ppf_cache(
        None, u, u, d,
        table_factory=lambda values: StudentPPFTable(
            values, n_boundary=8, n_lo=24, n_hi=16))
    config = AutoTMConfig(
        transition_method=transition_method, K=256, max_K=256,
        adaptive=False, grid_method="dense", n_threads=4)
    prepared = _cpp_scar_ou.prepare_objective(u, copula, config)
    params = np.array([2.0, 0.4, 0.9])
    value, ou_gradient, gradient, _ = (
        prepared.neg_loglik_with_grad_and_corr_info(*params))
    lower = np.tril_indices(d, -1)
    direction = rng.normal(size=len(lower[0]))
    direction /= np.linalg.norm(direction)
    directional_value, directional_ou, directional_gradient, _ = (
        prepared.neg_loglik_with_grad_and_corr_directional_info(
            *params, direction))
    ordinary_value, ordinary_ou, _ = prepared.neg_loglik_with_grad_info(*params)
    np.testing.assert_allclose([value, directional_value], ordinary_value, atol=1e-9)
    np.testing.assert_allclose(ou_gradient, ordinary_ou, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(directional_ou, ordinary_ou, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(directional_gradient, [gradient @ direction], rtol=1e-10)

    step = 1e-5
    finite_ou = []
    for p in range(3):
        delta = np.zeros(3)
        delta[p] = step
        plus = prepared.neg_loglik_info(*(params + delta))[0]
        minus = prepared.neg_loglik_info(*(params - delta))[0]
        finite_ou.append((plus - minus) / (2 * step))
    np.testing.assert_allclose(ou_gradient, finite_ou, rtol=2e-5, atol=2e-5)

    values = []
    for sign in (1, -1):
        trial = correlation.copy()
        trial[lower] += sign * step * direction
        trial[(lower[1], lower[0])] += sign * step * direction
        copula._set_R(trial)
        prepared.update_copula(copula)
        values.append(prepared.neg_loglik_info(*params)[0])
    finite_corr = (values[0] - values[1]) / (2 * step)
    np.testing.assert_allclose(directional_gradient, [finite_corr], rtol=2e-6, atol=2e-5)


def test_native_ou_supports_equicorr_forward_and_state():
    copula = EquicorrGaussianCopula(4)
    u = np.random.default_rng(20260619).uniform(0.05, 0.95, size=(12, 4))

    value, gradient, _ = _cpp_scar_ou.neg_loglik_with_grad_info(
        1.2, 0.1, 0.7, u, copula, _MATRIX_CONFIG)
    predictive = _cpp_scar_ou.predictive_mean(
        1.2, 0.1, 0.7, u, copula, _MATRIX_CONFIG)
    z_grid, probability = _cpp_scar_ou.state_distribution(
        1.2, 0.1, 0.7, u, copula, _MATRIX_CONFIG, horizon="next")

    assert np.isfinite(value)
    assert np.all(np.isfinite(gradient))
    assert predictive.shape == (len(u),)
    assert z_grid.shape == probability.shape == (24,)
    assert np.sum(probability) == pytest.approx(1.0)


def test_prepared_equicorr_repeated_calls_match_stateless_paths():
    copula = EquicorrGaussianCopula(12)
    u = np.random.default_rng(20260719).uniform(
        0.001, 0.999, size=(40, 12))
    config = AutoTMConfig(
        transition_method="matrix",
        K=28,
        adaptive=False,
        max_K=28,
    )
    params = (1.2, 0.1, 0.7)
    prepared = _cpp_scar_ou.prepare_objective(u, copula, config)

    expected_value, _ = _cpp_scar_ou.neg_loglik_info(
        *params, u, copula, config)
    expected_grad_value, expected_gradient, _ = (
        _cpp_scar_ou.neg_loglik_with_grad_info(
            *params, u, copula, config))
    first_value, _ = prepared.neg_loglik_info(*params)
    second_value, _ = prepared.neg_loglik_info(*params)
    actual_grad_value, actual_gradient, _ = (
        prepared.neg_loglik_with_grad_info(*params))

    assert first_value == expected_value
    assert second_value == expected_value
    assert actual_grad_value == expected_grad_value
    np.testing.assert_array_equal(actual_gradient, expected_gradient)
    np.testing.assert_array_equal(
        prepared.predictive_mean(*params),
        _cpp_scar_ou.predictive_mean(*params, u, copula, config),
    )
    for horizon in ("current", "next"):
        actual_z, actual_probability = prepared.state_distribution(
            *params, horizon=horizon)
        expected_z, expected_probability = _cpp_scar_ou.state_distribution(
            *params, u, copula, config, horizon=horizon)
        np.testing.assert_array_equal(actual_z, expected_z)
        np.testing.assert_array_equal(
            actual_probability, expected_probability)


@pytest.mark.parametrize("transition_method", ["spectral", "matrix", "local"])
def test_prepared_gaussian_objective_matches_stateless_exactly(
        transition_method):
    copula = BivariateGaussianCopula()
    u = np.random.default_rng(20260801).uniform(
        0.001, 0.999, size=(80, 2))
    config = AutoTMConfig(
        transition_method=transition_method,
        basis_order=32,
        K=32,
        adaptive=False,
        max_K=32,
    )
    params = (8.0, 0.2, 1.4)
    prepared = _cpp_scar_ou.prepare_objective(u, copula, config)

    expected_value, expected_gradient, _ = (
        _cpp_scar_ou.neg_loglik_with_grad_info(
            *params, u, copula, config))
    actual_value, actual_gradient, _ = (
        prepared.neg_loglik_with_grad_info(*params))

    assert actual_value == expected_value
    np.testing.assert_array_equal(actual_gradient, expected_gradient)


def test_spectral_forward_and_state_use_native_grid_reconstruction():
    copula = ClaytonCopula(transform_type="xtanh")
    u = np.random.default_rng(20260620).uniform(0.05, 0.95, size=(10, 2))
    config = AutoTMConfig(
        transition_method="spectral",
        basis_order=16,
        K=20,
        adaptive=False,
        max_K=20,
    )

    predictive = _cpp_scar_ou.predictive_mean(
        1.0, 0.0, 0.8, u, copula, config)
    z_grid, probability = _cpp_scar_ou.state_distribution(
        1.0, 0.0, 0.8, u, copula, config)

    assert predictive.shape == (len(u),)
    assert z_grid.shape == probability.shape == (20,)
    assert np.sum(probability) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "transition_method",
    ["matrix", "local", "auto", "spectral"],
)
def test_prepared_forward_helpers_match_stateless_wrappers(transition_method):
    copula = ClaytonCopula(transform_type="xtanh")
    u = np.random.default_rng(20260730).uniform(0.05, 0.95, size=(12, 2))
    config = AutoTMConfig(
        transition_method=transition_method,
        K=20,
        adaptive=False,
        max_K=20,
        basis_order=16,
        gh_order=5,
    )
    params = (1.1, 0.2, 0.8)
    prepared = _cpp_scar_ou.prepare_objective(u, copula, config)

    np.testing.assert_allclose(
        prepared.predictive_mean(*params),
        _cpp_scar_ou.predictive_mean(*params, u, copula, config),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        prepared.mixture_h(*params),
        _cpp_scar_ou.mixture_h(*params, u, copula, config),
        rtol=0.0,
        atol=0.0,
    )
    pair_prepared = prepared.mixture_h_pair(*params)
    pair_expected = _cpp_scar_ou.mixture_h_pair(
        *params, u, copula, config)
    np.testing.assert_allclose(
        pair_prepared[0], pair_expected[0], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        pair_prepared[1], pair_expected[1], rtol=0.0, atol=0.0)
    for horizon in ("current", "next"):
        z_prepared, p_prepared = prepared.state_distribution(
            *params, horizon=horizon)
        z_expected, p_expected = _cpp_scar_ou.state_distribution(
            *params, u, copula, config, horizon=horizon)
        np.testing.assert_allclose(
            z_prepared, z_expected, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            p_prepared, p_expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("transition_method", ["matrix", "local", "auto"])
def test_prepared_gaussian_mixture_h_pair_matches_stateless_exactly(
        transition_method):
    copula = BivariateGaussianCopula()
    u = np.random.default_rng(20260802).uniform(0.001, 0.999, size=(80, 2))
    config = AutoTMConfig(
        transition_method=transition_method,
        K=32,
        adaptive=False,
        max_K=32,
        gh_order=7,
    )
    params = (8.0, 0.2, 1.4)
    prepared = _cpp_scar_ou.prepare_objective(u, copula, config)

    actual = prepared.mixture_h_pair(*params)
    expected = _cpp_scar_ou.mixture_h_pair(*params, u, copula, config)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


def test_prepared_forward_helpers_use_updated_student_factor():
    rng = np.random.default_rng(20260731)
    u = rng.uniform(0.05, 0.95, size=(12, 3))
    base_R = np.full((3, 3), 0.2, dtype=np.float64)
    np.fill_diagonal(base_R, 1.0)
    copula = StochasticStudentCopula(d=3, R=base_R)
    config = AutoTMConfig(
        transition_method="matrix",
        K=18,
        adaptive=False,
        max_K=18,
    )
    params = (1.0, 0.1, 0.9)
    prepared = _cpp_scar_ou.prepare_objective(u, copula, config)
    before = prepared.predictive_mean(*params)

    updated_R = np.array(
        [
            [1.0, 0.35, 0.12],
            [0.35, 1.0, 0.18],
            [0.12, 0.18, 1.0],
        ],
        dtype=np.float64,
    )
    copula._set_R(updated_R, source="test")
    prepared.update_copula(copula)

    after = prepared.predictive_mean(*params)
    expected = _cpp_scar_ou.predictive_mean(*params, u, copula, config)
    assert not np.allclose(before, after, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(after, expected, rtol=0.0, atol=0.0)


def test_strategy_has_no_python_ou_production_imports():
    def dotted_name(node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            prefix = dotted_name(node.value)
            return f"{prefix}.{node.attr}" if prefix else node.attr
        return ""

    trees = [
        ast.parse(inspect.getsource(module), filename=module.__file__)
        for module in (scar_tm, stattests)
    ]
    references = {
        dotted_name(node)
        for tree in trees
        for node in ast.walk(tree)
        if isinstance(node, (ast.Name, ast.Attribute))
    }
    for symbol in (
        "tm_loglik",
        "tm_loglik_with_grad",
        "tm_forward_predictive_mean",
        "tm_forward_rosenblatt",
        "tm_forward_mixture_h",
        "auto_neg_loglik",
        "predictive_tm.tm_state_distribution",
    ):
        assert symbol not in references


def test_missing_native_extension_fails_before_optimizer(monkeypatch):
    copula = ClaytonCopula()
    u = np.random.default_rng(20260621).uniform(0.05, 0.95, size=(8, 2))
    optimizer_called = False

    def unavailable():
        raise RuntimeError("native extension unavailable")

    def forbidden_optimizer(*args, **kwargs):
        nonlocal optimizer_called
        optimizer_called = True
        raise AssertionError("optimizer must not run")

    monkeypatch.setattr(_cpp_scar_ou, "require_available", unavailable)
    monkeypatch.setattr(scar_tm, "minimize", forbidden_optimizer)

    with pytest.raises(RuntimeError, match="native extension unavailable"):
        SCARTMStrategy(smart_init=False).fit(
            copula,
            u,
            alpha0=np.array([1.0, 0.0, 1.0]),
        )
    assert optimizer_called is False
