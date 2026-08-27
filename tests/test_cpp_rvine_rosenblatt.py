"""Stage 8.4 mandatory native R-vine Rosenblatt contracts."""

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
from rvine_candidate_harness import (
    _log_likelihood_python,
    _rvine_rosenblatt_transform_python,
)

from rvine_runtime_cases import (
    configured_mixed_family_vine,
    configured_mixed_gas_vine,
    configured_mixed_jacobi_vine,
    configured_mixed_scar_vine,
    configured_static_dvine,
    fitted_pair,
)


def _python_public_rosenblatt(
        vine, observations, K=300, grid_range=5.0, *, vine_type=None):
    if vine_type is None:
        vine_type = getattr(vine, "vine_type", "rvine")
    prepared = stattests._prepare_rvine_rosenblatt_observations(
        vine, observations)
    return _rvine_rosenblatt_transform_python(
        vine,
        prepared,
        K=K,
        grid_range=grid_range,
        vine_type=vine_type,
    )


def _observations(n=17, d=3, seed=2026082255):
    return np.random.default_rng(seed).uniform(0.01, 0.99, size=(n, d))


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


@pytest.mark.parametrize(
    ("family", "rotation", "parameter"),
    [
        *[(ClaytonCopula, rotation, 0.8) for rotation in (0, 90, 180, 270)],
        *[(GumbelCopula, rotation, 1.6) for rotation in (0, 90, 180, 270)],
        *[(JoeCopula, rotation, 1.7) for rotation in (0, 90, 180, 270)],
        (FrankCopula, 0, 2.5),
        (BivariateGaussianCopula, 0, -0.4),
        (IndependentCopula, 0, 0.0),
    ],
)
def test_static_two_dimensional_family_rotation_matrix_is_exact(
        monkeypatch, family, rotation, parameter):
    vine = configured_static_dvine(2)
    copula = family() if family is IndependentCopula else family(
        rotate=rotation)
    vine.pair_copulas[(0, 0)] = fitted_pair(copula, parameter)
    observations = np.array([
        [2e-10, 0.2],
        [0.35, 0.65],
        [0.8, 1.0 - 2e-10],
    ])

    expected = _rvine_rosenblatt_transform_python(
        vine, observations)
    actual = rvine_rosenblatt_transform(vine, observations)

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual[:, 0], np.clip(
        observations[:, 0], 1e-6, 1.0 - 1e-6))


def test_static_nontrivial_peel_order_and_noncontiguous_input_are_exact(
        monkeypatch):
    vine = configured_static_dvine(6, order=[5, 1, 4, 0, 3, 2])
    base = _observations(n=23, d=12)
    observations = base[:, ::2]
    assert not observations.flags.c_contiguous

    expected = _rvine_rosenblatt_transform_python(
        vine, observations)
    actual = rvine_rosenblatt_transform(vine, observations)

    np.testing.assert_array_equal(actual, expected)
    assert actual.flags.c_contiguous


@pytest.mark.parametrize("n", [0, 1, 19])
def test_static_empty_singleton_and_batch_contract(monkeypatch, n):
    vine = configured_mixed_family_vine()
    observations = _observations(n=n, d=vine.d)
    expected = _rvine_rosenblatt_transform_python(
        vine, observations)
    actual = rvine_rosenblatt_transform(vine, observations)
    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == (n, vine.d)
    assert actual.dtype == np.float64


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
    # The low-level common runtime supports paths, but the public Stage 5
    # adapter never sends one. Native validation still accepts a coherent
    # pack and produces a well-formed result.
    _cpp_rvine.raise_for_status(invalid_parameters, "test row-path pack")
    assert np.asarray(invalid_parameters["residuals"]).size == observations.size


def test_public_native_adapter_accepts_explicit_row_path_layout():
    vine = configured_mixed_family_vine()
    observations = _observations(n=5, d=vine.d)
    module = _cpp_extension.load()
    active_keys = _cpp_rvine.density_active_keys(
        vine._trees, vine._edge_map)
    parameters = {
        key: np.full(len(observations), edge.param)
        for key, edge in vine.pair_copulas.items()
        if not isinstance(edge.copula, IndependentCopula)
    }
    sources = {key: "row_path" for key in parameters}
    actual = _cpp_rvine.rosenblatt(
        module,
        vine.pair_copulas,
        vine.d,
        vine._trees,
        vine._edge_map,
        vine.matrix,
        observations,
        active_keys=active_keys,
        parameter_paths=parameters,
        parameter_sources=sources,
    )
    expected = _rvine_rosenblatt_transform_python(
        vine, observations)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "factory",
    [
        configured_mixed_gas_vine,
        configured_mixed_scar_vine,
        configured_mixed_jacobi_vine,
    ],
    ids=["gas", "scar-ou", "scar-jacobi"],
)
def test_dynamic_edges_use_native_traversal_without_python_fallback(
        monkeypatch, factory):
    vine = factory()
    observations = _observations(n=13, d=vine.d)
    expected = _rvine_rosenblatt_transform_python(
        vine, observations)

    assert not hasattr(stattests, "_rvine_rosenblatt_transform_python")
    actual = rvine_rosenblatt_transform(vine, observations)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "factory",
    [
        configured_mixed_gas_vine,
        configured_mixed_scar_vine,
        configured_mixed_jacobi_vine,
    ],
    ids=["gas", "scar-ou", "scar-jacobi"],
)
def test_dynamic_log_likelihood_uses_native_traversal(factory):
    vine = factory()
    observations = _observations(n=13, d=vine.d, seed=2026082261)

    expected = _log_likelihood_python(vine, observations)
    actual = vine.log_likelihood(observations)

    assert actual == pytest.approx(expected, rel=2e-12, abs=2e-12)


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


@pytest.mark.parametrize("observations", [
    np.ones((3, 2)),
    np.ones(3),
    np.full((3, 3), np.nan),
])
def test_public_input_errors_match_python_oracle(
        monkeypatch, observations):
    vine = configured_mixed_family_vine()

    def outcome(transform):
        try:
            transform(vine, observations)
        except Exception as exc:
            return type(exc), str(exc)
        pytest.fail("transform unexpectedly accepted invalid observations")

    expected = outcome(_python_public_rosenblatt)
    actual = outcome(rvine_rosenblatt_transform)
    assert actual == expected


@pytest.mark.parametrize("observations", [
    np.full((3, 3), np.inf),
    np.full((3, 3), 0.5 + 0.1j),
])
def test_public_coercion_and_clipping_match_python_oracle(
        monkeypatch, observations):
    vine = configured_mixed_family_vine()

    def outcome(transform):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = transform(vine, observations)
        return result, [
            (item.category, str(item.message))
            for item in caught
        ]

    expected, expected_warnings = outcome(_python_public_rosenblatt)
    actual, actual_warnings = outcome(rvine_rosenblatt_transform)
    np.testing.assert_array_equal(actual, expected)
    assert actual_warnings == expected_warnings


def test_missing_rosenblatt_symbol_has_operation_specific_contract(
        monkeypatch):
    vine = configured_mixed_family_vine()
    observations = _observations(n=5, d=vine.d)
    monkeypatch.setattr(_cpp_extension, "load", lambda: SimpleNamespace())
    with pytest.raises(NativeUnsupported, match="rvine_rosenblatt_transform"):
        rvine_rosenblatt_transform(vine, observations)
    assert not hasattr(stattests, "_rvine_rosenblatt_transform_python")


def test_gof_result_is_exact_across_python_and_native_backends(monkeypatch):
    vine = configured_mixed_family_vine()
    observations = _observations(n=31, d=vine.d, seed=2026082257)

    with monkeypatch.context() as oracle:
        oracle.setattr(
            stattests, "rvine_rosenblatt_transform",
            _python_public_rosenblatt)
        expected = stattests.gof_test(vine, observations, to_pobs=False)
    actual = stattests.gof_test(vine, observations, to_pobs=False)

    assert actual.statistic == expected.statistic
    assert actual.pvalue == expected.pvalue


def test_bootstrap_is_reproducible_across_python_and_native_backends(
        monkeypatch):
    vine = configured_mixed_family_vine()
    observations = _observations(n=11, d=vine.d, seed=2026082258)

    def outcome(n_jobs):
        return stattests.gof_test(
            vine,
            observations,
            to_pobs=False,
            bootstrap=True,
            n_bootstrap=3,
            bootstrap_refit=False,
            rng=2026082259,
            n_jobs=n_jobs,
        )

    with monkeypatch.context() as oracle:
        oracle.setattr(
            stattests, "rvine_rosenblatt_transform",
            _python_public_rosenblatt)
        expected = outcome(1)
    actual = outcome(2)
    assert actual.statistic == expected.statistic
    assert actual.pvalue == expected.pvalue
    assert actual.n_bootstrap == expected.n_bootstrap == 3
    assert expected.n_jobs == 1
    assert actual.n_jobs == 2
    np.testing.assert_array_equal(
        actual.bootstrap_statistics, expected.bootstrap_statistics)

    def deterministic_diagnostics(result):
        return tuple({
            key: value
            for key, value in row.items()
            if not key.endswith("_time_sec")
        } for row in result.bootstrap_diagnostics)

    assert deterministic_diagnostics(actual) == deterministic_diagnostics(
        expected)


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
