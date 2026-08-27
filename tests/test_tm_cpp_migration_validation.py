"""Final edge and resource contracts for the native TM migration."""

from __future__ import annotations

import numpy as np
import pytest

from pyscarcopula import ClaytonCopula, EquicorrGaussianCopula
from pyscarcopula.copula.multivariate import StochasticStudentCopula
from pyscarcopula._native import scar_ou as _cpp_scar_ou
from pyscarcopula._native.errors import NativeError
from pyscarcopula.numerical._scar_ou_config import AutoTMConfig


def _student(d=4):
    correlation = np.full((d, d), 0.18, dtype=np.float64)
    np.fill_diagonal(correlation, 1.0)
    return StochasticStudentCopula(d=d, R=correlation)


def _uniform(T, d, seed):
    return np.random.default_rng(seed).uniform(0.02, 0.98, size=(T, d))


@pytest.mark.parametrize(
    "call",
    [
        lambda cfg: _cpp_scar_ou.forward_rosenblatt(
            0.8, 0.0, 1.0, np.array([[0.3, 0.7]]),
            ClaytonCopula(), cfg),
        lambda cfg: _cpp_scar_ou.gaussian_rosenblatt(
            0.8, 0.0, 1.0, np.array([[0.2, 0.4, 0.8]]),
            EquicorrGaussianCopula(d=3), cfg),
        lambda cfg: _cpp_scar_ou.student_rosenblatt(
            0.8, 0.0, 1.0, np.array([[0.2, 0.4, 0.8]]),
            _student(d=3), cfg),
        lambda cfg: _cpp_scar_ou.smoothed_state_distribution(
            0.8, 0.0, 1.0, np.array([[0.2, 0.4, 0.8]]),
            _student(d=3), cfg),
    ],
)
def test_native_migration_endpoints_reject_single_observation(call):
    config = AutoTMConfig(
        K=9,
        adaptive=False,
        transition_method="matrix",
        max_K=None,
    )
    with pytest.raises(NativeError, match=r"invalid_size"):
        call(config)


@pytest.mark.parametrize(
    ("params", "transition_method", "grid_method"),
    [
        ((0.001, -2.0, 0.02), "local", "auto"),
        ((80.0, 1.5, 8.0), "matrix", "sparse"),
    ],
)
def test_native_migration_endpoints_are_finite_at_extreme_ou_parameters(
        params, transition_method, grid_method):
    pair_u = _uniform(48, 2, 20260810)
    multi_u = _uniform(48, 4, 20260811)
    config = AutoTMConfig(
        K=73,
        grid_range=4.0,
        grid_method=grid_method,
        adaptive=False,
        transition_method=transition_method,
        max_K=None,
        gh_order=9,
    )

    pair = _cpp_scar_ou.forward_rosenblatt(
        *params, pair_u, ClaytonCopula(), config)
    gaussian = _cpp_scar_ou.gaussian_rosenblatt(
        *params, multi_u, EquicorrGaussianCopula(d=4), config)
    student = _cpp_scar_ou.student_rosenblatt(
        *params, multi_u, _student(), config)
    grid, smoothed = _cpp_scar_ou.smoothed_state_distribution(
        *params, multi_u, _student(), config)

    for values in (pair, gaussian, student, grid, smoothed):
        assert np.all(np.isfinite(values))
    np.testing.assert_allclose(
        smoothed.sum(axis=1), 1.0, rtol=0.0, atol=2e-15)


@pytest.mark.parametrize(
    ("factory", "u"),
    [
        (
            lambda config, values: _cpp_scar_ou.forward_rosenblatt(
                0.001, 0.2, 0.02, values, ClaytonCopula(), config),
            _uniform(80, 2, 20260812),
        ),
        (
            lambda config, values: _cpp_scar_ou.gaussian_rosenblatt(
                0.001, 0.2, 0.02, values,
                EquicorrGaussianCopula(d=4), config),
            _uniform(80, 4, 20260813),
        ),
        (
            lambda config, values: _cpp_scar_ou.student_rosenblatt(
                0.001, 0.2, 0.02, values, _student(), config),
            _uniform(80, 4, 20260814),
        ),
    ],
)
def test_adaptive_max_k_matches_explicit_capped_local_grid(factory, u):
    capped = AutoTMConfig(
        K=31,
        grid_range=3.5,
        adaptive=True,
        pts_per_sigma=6,
        transition_method="auto",
        max_K=65,
        gh_order=7,
    )
    explicit = AutoTMConfig(
        K=65,
        grid_range=3.5,
        adaptive=False,
        transition_method="local",
        max_K=None,
        gh_order=7,
    )

    np.testing.assert_allclose(
        factory(capped, u),
        factory(explicit, u),
        rtol=0.0,
        atol=0.0,
    )


def test_native_migration_large_t_and_k_stress_shapes_and_finiteness():
    pair_u = _uniform(1024, 2, 20260815)
    pair_config = AutoTMConfig(
        K=1025,
        grid_range=4.0,
        adaptive=False,
        transition_method="local",
        max_K=None,
        gh_order=7,
    )
    pair = _cpp_scar_ou.forward_rosenblatt(
        0.03, 0.0, 1.0, pair_u, ClaytonCopula(), pair_config)

    multi_u = _uniform(256, 4, 20260816)
    multi_config = AutoTMConfig(
        K=513,
        grid_range=4.0,
        adaptive=False,
        transition_method="local",
        max_K=None,
        gh_order=7,
    )
    gaussian = _cpp_scar_ou.gaussian_rosenblatt(
        0.03, 0.0, 1.0, multi_u,
        EquicorrGaussianCopula(d=4), multi_config)
    student_model = _student()
    student = _cpp_scar_ou.student_rosenblatt(
        0.03, 0.0, 1.0, multi_u, student_model, multi_config)
    grid, smoothed = _cpp_scar_ou.smoothed_state_distribution(
        0.03, 0.0, 1.0, multi_u, student_model, multi_config)

    assert pair.shape == pair_u.shape
    assert gaussian.shape == multi_u.shape
    assert student.shape == multi_u.shape
    assert grid.shape == (513,)
    assert smoothed.shape == (len(multi_u), 513)
    for values in (pair, gaussian, student, grid, smoothed):
        assert np.all(np.isfinite(values))
    np.testing.assert_allclose(
        smoothed.sum(axis=1), 1.0, rtol=0.0, atol=3e-15)
