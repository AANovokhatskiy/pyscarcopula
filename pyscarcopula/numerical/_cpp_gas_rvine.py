"""Native sequential R-vine sampler for fitted GAS edge states."""

from __future__ import annotations

import numpy as np

from pyscarcopula._types import GASResult
from pyscarcopula.numerical import _cpp_extension, _cpp_gas, _cpp_rvine
from pyscarcopula.numerical._cpp_extension import (
    CppUnsupported,
)
from pyscarcopula.vine._edge_adapter import edge_copula, edge_result
from pyscarcopula.vine._helpers import _open_unit_uniform
from pyscarcopula.vine._rvine_edges import (
    _edge_r_for_sample,
    _edge_requires_stepwise_sample,
)
from pyscarcopula.vine._rvine_sampling_plan import (
    build_rvine_sampling_plan,
)


def _native_edges(module, vine, active_keys):
    native_edges = []
    dynamic = []
    for key in active_keys:
        edge = vine.pair_copulas[key]
        result = edge_result(edge)
        copula = edge_copula(edge)
        native = module.GasRvineEdge()
        native.copula = _cpp_rvine.compile_copula_spec(module, copula)
        is_dynamic = isinstance(result, GASResult)
        native.dynamic = is_dynamic
        if is_dynamic:
            params = result.params
            native.gas_params = _cpp_gas._params(
                module, params.omega, params.gamma, params.beta)
            native.gas_config = _cpp_gas._config(
                module,
                result.scaling,
                result.score_eps,
            )
        native_edges.append(native)
        dynamic.append(is_dynamic)
    return native_edges, dynamic


def sample(
        vine,
        n,
        rng,
        active_keys,
        max_active_tree,
        traversal_plan=None):
    """Return a native GAS R-vine sample, or ``None`` when unsupported."""
    module = _cpp_extension.load()
    if not hasattr(module, "gas_rvine_sample"):
        return None
    for key in active_keys:
        edge = vine.pair_copulas[key]
        if (
            not isinstance(edge_result(edge), GASResult)
            and _edge_requires_stepwise_sample(edge)
        ):
            return None
    try:
        native_edges, dynamic = _native_edges(module, vine, active_keys)
    except CppUnsupported:
        return None
    if not any(dynamic):
        return None

    parameter_paths = np.zeros((n, len(active_keys)), dtype=np.float64)
    for edge_index, key in enumerate(active_keys):
        if not dynamic[edge_index]:
            parameter_paths[:, edge_index] = _edge_r_for_sample(
                vine.pair_copulas[key], n, rng)

    uniforms = np.ascontiguousarray(
        _open_unit_uniform(rng, size=(n, vine.d)),
        dtype=np.float64,
    )
    if traversal_plan is None:
        traversal_plan = build_rvine_sampling_plan(
            vine.d,
            vine.matrix,
            vine.trees,
            vine._edge_map,
            active_keys,
            max_active_tree,
        )
    elif traversal_plan.active_keys != tuple(active_keys):
        raise ValueError(
            "R-vine traversal plan does not match active edge keys")
    plan = _cpp_rvine.compile_traversal_plan(module, traversal_plan)
    result = module.gas_rvine_sample(
        native_edges,
        plan,
        uniforms,
        parameter_paths,
    )
    _cpp_rvine.raise_for_status(result, "GAS R-vine sample")
    values = np.asarray(result["values"], dtype=np.float64)
    return values.reshape(n, vine.d)


__all__ = ["sample"]
