import json
import os

import numpy as np
import pytest

from tools.benchmark_timing import interleaved_timings

DIMENSIONS = (20, 80, 150, 300)


def _module():
    import pyscarcopula._scar_cpp as module

    return module


def test_dependency_free_linalg_backend_contract():
    info = dict(_module()._linalg_info())

    assert info == {
        "selected_backend": "portable",
        "scalar_fallback": True,
        "external_dependencies": False,
        "accumulators": 4,
        "portable_min_elements": 32,
    }


@pytest.mark.parametrize("dimension", DIMENSIONS)
def test_portable_matvec_matches_scalar_and_numpy(dimension):
    rng = np.random.default_rng(920 + dimension)
    matrix = rng.normal(size=(dimension, dimension))
    vector = rng.normal(size=dimension)
    module = _module()

    scalar = module._linalg_matvec_probe(matrix, vector, scalar=True)
    portable = module._linalg_matvec_probe(matrix, vector, scalar=False)
    reference = matrix @ vector

    np.testing.assert_allclose(portable, scalar, rtol=2e-14, atol=2e-13)
    np.testing.assert_allclose(portable, reference, rtol=2e-14, atol=2e-13)
    if dimension == 20:
        np.testing.assert_array_equal(portable, scalar)


@pytest.mark.parametrize("dimension", DIMENSIONS)
def test_portable_cholesky_solve_matches_scalar(dimension):
    rng = np.random.default_rng(930 + dimension)
    design = rng.normal(size=(dimension, dimension))
    matrix = design @ design.T + np.eye(dimension)
    rhs = rng.normal(size=(dimension, 3))
    module = _module()

    scalar_lower, scalar_solution = module._linalg_cholesky_solve_probe(
        matrix, rhs, scalar=True)
    portable_lower, portable_solution = module._linalg_cholesky_solve_probe(
        matrix, rhs, scalar=False)

    np.testing.assert_allclose(
        portable_lower, scalar_lower, rtol=2e-14, atol=2e-13)
    np.testing.assert_allclose(
        portable_solution, scalar_solution, rtol=5e-13, atol=5e-13)
    np.testing.assert_allclose(
        matrix @ portable_solution, rhs, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize(
    ("matrix", "vector"),
    [
        (np.eye(2), np.ones(3)),
        (np.array([[1.0, np.nan], [0.0, 1.0]]), np.ones(2)),
    ],
)
def test_linalg_probe_rejects_invalid_input(matrix, vector):
    with pytest.raises(ValueError):
        _module()._linalg_matvec_probe(matrix, vector)


def _benchmark_enabled():
    if os.environ.get("PYSCA_RUN_BENCHMARKS") != "1":
        pytest.skip("set PYSCA_RUN_BENCHMARKS=1 to run benchmark baselines")


@pytest.mark.benchmark
def test_dependency_free_linalg_benchmark():
    _benchmark_enabled()
    rng = np.random.default_rng(940)
    module = _module()
    payload = {}

    for dimension in DIMENSIONS:
        matrix = rng.normal(size=(dimension, dimension))
        vector = rng.normal(size=dimension)
        repeats = max(10, 20_000_000 // (dimension * dimension))
        matvec_calls = {
            "scalar": lambda: module._linalg_matvec_probe(
                matrix, vector, scalar=True, repeat=repeats),
            "portable": lambda: module._linalg_matvec_probe(
                matrix, vector, scalar=False, repeat=repeats),
        }
        for call in matvec_calls.values():
            call()
        matvec = interleaved_timings(matvec_calls, repeats=5)

        spd = matrix @ matrix.T + np.eye(dimension)
        rhs = rng.normal(size=(dimension, 4))
        solve_calls = {
            "scalar": lambda: module._linalg_cholesky_solve_probe(
                spd, rhs, scalar=True),
            "portable": lambda: module._linalg_cholesky_solve_probe(
                spd, rhs, scalar=False),
        }
        for call in solve_calls.values():
            call()
        solve = interleaved_timings(solve_calls, repeats=5)
        matvec_medians = matvec.medians
        solve_medians = solve.medians
        payload[str(dimension)] = {
            "matvec_scalar_seconds": matvec_medians["scalar"],
            "matvec_portable_seconds": matvec_medians["portable"],
            "matvec_speedup": matvec.median_ratio("scalar", "portable"),
            "cholesky_solve_scalar_seconds": solve_medians["scalar"],
            "cholesky_solve_portable_seconds": solve_medians["portable"],
            "cholesky_solve_speedup": solve.median_ratio(
                "scalar", "portable"),
        }

    print("PYSCA_BENCHMARK " + json.dumps(
        payload, sort_keys=True), flush=True)
