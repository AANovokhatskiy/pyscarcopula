"""Regression contracts for saved OU spectral settings in native vines."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from pyscarcopula import BivariateGaussianCopula
from pyscarcopula._native import (
    _extension as native_extension,
    scar_ou as native_ou,
    vine as native_vine,
)
from pyscarcopula._types import LatentResult, ou_params
from pyscarcopula.vine._edge_adapter import strategy_for_result
from pyscarcopula.vine._pair_copula import PairCopula
from rvine_runtime_cases import configured_static_dvine


def _spectral_vine(observations):
    vine = configured_static_dvine(2)
    copula = BivariateGaussianCopula()
    result = LatentResult(
        log_likelihood=0.0,
        method="SCAR-TM-OU",
        copula_name=copula.name,
        success=True,
        params=ou_params(0.12, 0.15, 0.55),
        K=31,
        grid_range=3.0,
        grid_method="dense",
        adaptive=False,
        pts_per_sigma=4,
        transition_method="spectral",
        max_K=31,
        r_gh=3.0,
        gh_order=7,
        # Force auto posterior reconstruction onto its local-grid branch.
        # This makes a likelihood-side spectral-to-auto regression observable.
        auto_small_kdt=10.0,
        spectral_basis_order=12,
        spectral_quad_order=24,
    )
    key = next(iter(vine.pair_copulas))
    vine.pair_copulas[key] = PairCopula(
        copula=copula,
        param=None,
        log_likelihood=0.0,
        nfev=0,
        tau=0.0,
        fit_result=result,
    )
    vine.method = result.method
    vine._last_u = np.ascontiguousarray(observations, dtype=np.float64)
    vine._last_u.flags.writeable = False
    vine._T = len(observations)
    return vine, key


def _saved_config(result, n_rows):
    strategy = strategy_for_result(result)
    return strategy._auto_config(
        strategy.transition_method,
        kappa=result.params.kappa,
        n_obs=n_rows,
    )


def test_saved_ou_spectral_descriptor_drives_likelihood_but_h_uses_auto_grid():
    observations = np.random.default_rng(2026090101).uniform(
        0.04, 0.96, size=(24, 2))
    vine, key = _spectral_vine(observations)
    result = vine.pair_copulas[key].fit_result
    params = result.params
    config = _saved_config(result, len(observations))
    module = native_extension.load()
    active_keys = native_vine.density_active_keys(
        vine.trees, vine._edge_map)

    edges, _ = native_vine.compile_dynamic_rosenblatt_edges(
        module, vine.pair_copulas, active_keys, len(observations))

    assert len(edges) == 1
    assert edges[0].ou_method == "spectral"
    assert edges[0].ou_config.spectral_basis_order == 12
    assert edges[0].ou_config.spectral_quad_order == 24

    expected_log_likelihood, diagnostics = native_ou.loglik(
        params.kappa,
        params.mu,
        params.nu,
        observations,
        vine.pair_copulas[key].copula,
        config,
    )
    assert diagnostics["backend"] == "spectral"
    assert vine.log_likelihood(observations) == pytest.approx(
        expected_log_likelihood, rel=0.0, abs=1e-12)

    # The prepared C++ evaluator intentionally treats spectral as auto only
    # for posterior h-reconstruction. The compiled dynamic trace must keep
    # that operation-specific behavior after preserving spectral likelihood.
    expected_second_given_first, expected_first_given_second = (
        native_ou.mixture_h_pair(
            params.kappa,
            params.mu,
            params.nu,
            observations,
            vine.pair_copulas[key].copula,
            config,
        )
    )
    pseudo = native_vine.pseudo_observations(
        module,
        vine.pair_copulas,
        vine.d,
        vine.trees,
        vine._edge_map,
        vine.matrix,
        observations,
    )
    np.testing.assert_array_equal(
        pseudo[(1, frozenset({0}))], expected_second_given_first)
    np.testing.assert_array_equal(
        pseudo[(0, frozenset({1}))], expected_first_given_second)


def test_saved_ou_spectral_posterior_sample_and_predict_match_auto_grid():
    observations = np.random.default_rng(2026090102).uniform(
        0.04, 0.96, size=(20, 2))
    spectral_vine, key = _spectral_vine(observations)
    spectral_result = spectral_vine.pair_copulas[key].fit_result
    params = spectral_result.params
    spectral_config = _saved_config(spectral_result, len(observations))
    auto_config = replace(spectral_config, transition_method="auto")

    spectral_state = native_ou.state_distribution(
        params.kappa,
        params.mu,
        params.nu,
        observations,
        spectral_vine.pair_copulas[key].copula,
        spectral_config,
        horizon="next",
    )
    auto_state = native_ou.state_distribution(
        params.kappa,
        params.mu,
        params.nu,
        observations,
        spectral_vine.pair_copulas[key].copula,
        auto_config,
        horizon="next",
    )
    for spectral_values, auto_values in zip(spectral_state, auto_state):
        np.testing.assert_array_equal(spectral_values, auto_values)

    auto_vine = deepcopy(spectral_vine)
    auto_vine.pair_copulas[key].fit_result = replace(
        spectral_result, transition_method="auto")

    np.testing.assert_array_equal(
        spectral_vine.sample(12, rng=np.random.default_rng(2026090103)),
        auto_vine.sample(12, rng=np.random.default_rng(2026090103)),
    )
    np.testing.assert_array_equal(
        spectral_vine.predict(12, rng=np.random.default_rng(2026090104)),
        auto_vine.predict(12, rng=np.random.default_rng(2026090104)),
    )
