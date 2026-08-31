"""Mandatory native R-vine Rosenblatt contracts."""

from __future__ import annotations

import threading
from types import SimpleNamespace
import warnings

import numpy as np
import pytest

from pyscarcopula import (
    BivariateGaussianCopula,
    ClaytonCopula,
    FrankCopula,
    GumbelCopula,
    IndependentCopula,
    JoeCopula,
)
from pyscarcopula._native import _extension as _cpp_extension, vine as _cpp_rvine
from pyscarcopula._native.errors import NativeUnsupported
import pyscarcopula.stattests as stattests
from pyscarcopula.stattests import rvine_rosenblatt_transform
from rvine_runtime_cases import (
    configured_mixed_family_vine,
    configured_mixed_gas_vine,
    configured_mixed_jacobi_vine,
    configured_mixed_scar_vine,
    configured_static_dvine,
    fitted_pair,
)


def _observations(n=17, d=3, seed=2026082255):
    return np.random.default_rng(seed).uniform(0.01, 0.99, size=(n, d))


def _configured_all_dynamic_vine():
    vine = configured_static_dvine(4)
    for key, factory in zip(vine.pair_copulas, (
            configured_mixed_gas_vine, configured_mixed_scar_vine,
            configured_mixed_jacobi_vine)):
        vine.pair_copulas[key] = factory().pair_copulas[(0, 1)]
    vine.method = 'MIXED'
    return vine


@pytest.mark.parametrize('factory', [
    configured_mixed_gas_vine, configured_mixed_jacobi_vine,
    configured_mixed_scar_vine, _configured_all_dynamic_vine,
])
@pytest.mark.parametrize('overrides', [{}, {'K': 23, 'grid_range': 2.75}])
def test_gof_routes_ou_grid_options_in_dynamic_vines(factory, overrides):
    vine = factory()
    result = stattests.gof_test(
        vine, _observations(d=vine.d), to_pobs=False, **overrides)
    assert np.isfinite(result.statistic)
    assert 0 <= result.pvalue <= 1


def test_dynamic_rosenblatt_preserves_ou_grid_and_other_fitted_settings():
    vine = _configured_all_dynamic_vine()
    module = _cpp_extension.load()
    active = _cpp_rvine.density_active_keys(vine._trees, vine._edge_map)
    edges, _ = _cpp_rvine.compile_dynamic_rosenblatt_edges(
        module, vine.pair_copulas, active, 17,
        strategy_kwargs={'K': 23, 'grid_range': 2.75})
    by_kind = {edge.dynamics: edge for edge in edges}
    ou = by_kind[module.DynamicRvineKind.SCAR_OU]
    assert ou.ou_config.K == 23
    assert ou.ou_config.grid_range == 2.75
    jacobi = by_kind[module.DynamicRvineKind.SCAR_JACOBI]
    assert jacobi.jacobi_config.transition.numerical.basis_order == 8
    assert jacobi.jacobi_config.transition.numerical.quad_order == 32
    assert module.DynamicRvineKind.GAS in by_kind
    assert module.DynamicRvineKind.STATIC in by_kind


@pytest.mark.parametrize('factory', [
    configured_mixed_gas_vine, configured_mixed_jacobi_vine,
    configured_mixed_scar_vine,
])
def test_dynamic_rosenblatt_rejects_unknown_non_grid_options(factory):
    vine = factory()
    module = _cpp_extension.load()
    active = _cpp_rvine.density_active_keys(vine._trees, vine._edge_map)
    with pytest.raises(TypeError, match='definitely_unknown'):
        _cpp_rvine.compile_dynamic_rosenblatt_edges(
            module, vine.pair_copulas, active, 17,
            strategy_kwargs={'K': 23, 'grid_range': 2.75, 'definitely_unknown': 1})


def _native_request(vine, observations):
    module = _cpp_extension.load()
    active_keys = _cpp_rvine.density_active_keys(
        vine._trees, vine._edge_map)
    layout = _cpp_rvine.static_rosenblatt_parameter_layout(
        vine.pair_copulas, active_keys)
    assert layout is not None
    parameter_paths, parameter_sources = layout
    residual_node_keys = _cpp_rvine.rosenblatt_residual_node_keys(
        vine.matrix)
    context, pack = vine._native_density_context(
        module,
        vine.pair_copulas,
        vine._edge_map,
        parameter_paths,
        len(observations),
        active_keys=active_keys,
        normalized_paths=parameter_paths,
        parameter_sources=parameter_sources,
        residual_node_keys=residual_node_keys,
        cache_slot="rosenblatt",
    )
    assert context is not None
    return module, context, pack, residual_node_keys


def test_direct_native_diagnostics_and_thread_request_are_deterministic():
    vine = configured_mixed_family_vine()
    observations = _observations(n=11, d=vine.d)
    module, context, pack, _ = _native_request(vine, observations)

    sequential = module.rvine_rosenblatt_transform(
        context["plan"],
        context["edges"],
        pack.scalar_parameters,
        pack.row_parameters,
        observations,
        1,
    )
    requested_many = module.rvine_rosenblatt_transform(
        context["plan"],
        context["edges"],
        pack.scalar_parameters,
        pack.row_parameters,
        observations,
        4,
    )
    _cpp_rvine.raise_for_status(sequential, "test Rosenblatt")
    _cpp_rvine.raise_for_status(requested_many, "test Rosenblatt")
    np.testing.assert_array_equal(
        sequential["residuals"], requested_many["residuals"])
    diagnostics = dict(requested_many["diagnostics"])
    assert diagnostics["n_threads_requested"] == 4
    assert diagnostics["n_threads_used"] == 1
    assert diagnostics["h_pair_operations"] == len(observations) * 3
    assert diagnostics["independence_fast_paths"] == len(observations)


def test_direct_native_rejects_missing_residuals_and_accepts_valid_row_pack():
    vine = configured_mixed_family_vine()
    observations = _observations(n=7, d=vine.d)
    module, context, pack, _ = _native_request(vine, observations)

    malformed = _cpp_rvine.compile_density_plan(
        module,
        vine.d,
        vine._trees,
        vine._edge_map,
        context["active_keys"],
    )
    invalid_plan = module.rvine_rosenblatt_transform(
        malformed,
        context["edges"],
        pack.scalar_parameters,
        pack.row_parameters,
        observations,
    )
    assert int(invalid_plan["status"]) == 2

    row_path_edges = list(context["edges"])
    row_path_edges[0].parameter_source = module.RVineParameterSource.ROW_PATH
    row_path_edges[0].parameter_index = 0
    invalid_parameters = module.rvine_rosenblatt_transform(
        context["plan"],
        row_path_edges,
        pack.scalar_parameters,
        np.full((len(observations), 1), 0.8),
        observations,
    )
    # The low-level common runtime supports paths, but the public dynamic
    # adapter never sends one. Native validation still accepts a coherent
    # pack and produces a well-formed result.
    _cpp_rvine.raise_for_status(invalid_parameters, "test row-path pack")
    assert np.asarray(invalid_parameters["residuals"]).size == observations.size


def test_custom_builtin_subclass_is_rejected():
    calls = []

    class CustomClayton(ClaytonCopula):
        def h_pair(self, u, v, r):
            calls.append("h_pair")
            first = np.full_like(np.asarray(u, dtype=np.float64), 0.314)
            second = np.full_like(np.asarray(v, dtype=np.float64), 0.271)
            return first, second

    vine = configured_mixed_family_vine()
    vine.pair_copulas[(0, 0)] = fitted_pair(CustomClayton(rotate=90), 0.8)
    observations = _observations(n=9, d=vine.d)
    with pytest.raises(NativeUnsupported, match="exact registered"):
        rvine_rosenblatt_transform(vine, observations)
    with pytest.raises(NativeUnsupported, match="exact registered"):
        vine.log_likelihood(observations)
    assert calls == []


def test_bootstrap_refit_uses_fitted_worker_vine(monkeypatch):
    vine = configured_static_dvine(2)
    observations = _observations(n=19, d=vine.d, seed=2026082260)
    original_edges = tuple(vine.pair_copulas.values())

    result = stattests.gof_test(
        vine,
        observations,
        to_pobs=False,
        bootstrap=True,
        n_bootstrap=1,
        bootstrap_refit=True,
        bootstrap_fit_kwargs={"maxiter": 5},
        rng=2026082261,
        n_jobs=1,
    )

    assert result.n_bootstrap == 1
    assert result.bootstrap_statistics.shape == (1,)
    assert result.bootstrap_diagnostics[0]["bootstrap_refit"] is True
    assert result.bootstrap_diagnostics[0]["bootstrap_fit_method"] == "MLE"
    assert tuple(vine.pair_copulas.values()) == original_edges


def test_rosenblatt_context_reuses_and_invalidates_semantic_cache(
        monkeypatch):
    vine = configured_mixed_family_vine()
    observations = _observations(n=12, d=vine.d)

    rvine_rosenblatt_transform(vine, observations)
    first = vine._native_rvine_cache["rosenblatt"]
    first_plan = first["plan"]
    first_edges = first["edges"]
    rvine_rosenblatt_transform(vine, observations[:3])
    reused = vine._native_rvine_cache["rosenblatt"]
    assert reused["plan"] is first_plan
    assert all(
        current is previous
        for current, previous in zip(reused["edges"], first_edges)
    )

    vine.pair_copulas[(0, 0)].copula._rotate = 180
    rvine_rosenblatt_transform(vine, observations)
    refreshed = vine._native_rvine_cache["rosenblatt"]
    assert refreshed["plan"] is not first_plan
    assert refreshed["edges"][0] is not first_edges[0]


def test_native_rosenblatt_releases_the_gil():
    vine = configured_static_dvine(25)
    vine.pair_copulas = {
        key: fitted_pair(BivariateGaussianCopula(), 0.05)
        for key in vine.pair_copulas
    }
    observations = _observations(n=3000, d=vine.d, seed=2026082256)
    module, context, pack, _ = _native_request(vine, observations)
    started = threading.Event()
    stop = threading.Event()
    counter = [0]

    def worker():
        started.set()
        while not stop.is_set():
            counter[0] += 1

    thread = threading.Thread(target=worker)
    thread.start()
    assert started.wait(timeout=2.0)
    before = counter[0]
    try:
        result = module.rvine_rosenblatt_transform(
            context["plan"],
            context["edges"],
            pack.scalar_parameters,
            pack.row_parameters,
            observations,
        )
    finally:
        stop.set()
        thread.join()
    _cpp_rvine.raise_for_status(result, "test Rosenblatt GIL release")
    assert counter[0] > before
