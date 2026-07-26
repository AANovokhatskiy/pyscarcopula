"""Opt-in Stage 8 workload reports for the VineCopula unification."""

from __future__ import annotations

import gc
import os
from time import perf_counter

import numpy as np
import pytest
from scipy.stats import rankdata

from pyscarcopula import (
    BivariateGaussianCopula,
    IndependentCopula,
    VineCopula,
)


pytestmark = pytest.mark.benchmark


def _require_benchmarks():
    if os.environ.get("PYSCA_RUN_VINE_BENCHMARKS") != "1":
        pytest.skip("set PYSCA_RUN_VINE_BENCHMARKS=1")


def _data(rows, dimension, seed):
    rng = np.random.default_rng(seed)
    common = rng.standard_normal((rows, 1))
    loadings = np.linspace(0.2, 0.65, dimension)
    raw = (
        loadings * common
        + np.sqrt(1.0 - loadings**2)
        * rng.standard_normal((rows, dimension))
    )
    return np.apply_along_axis(
        rankdata, 0, raw
    ) / (rows + 1)


def _specs(trees):
    return [
        [(BivariateGaussianCopula, 0) for _ in level]
        for level in trees
    ]


def _edge_identity(edge):
    pair, conditioning = edge
    return frozenset(pair), frozenset(conditioning)


@pytest.mark.parametrize("dimension", [5, 10, 20])
def test_stage8_auto_versus_fixed_fit_profile(dimension):
    _require_benchmarks()
    u = _data(80, dimension, 20260800 + dimension)

    started = perf_counter()
    auto = VineCopula(
        candidates=[BivariateGaussianCopula],
        allow_rotations=False,
    ).fit(u, method="mle")
    auto_seconds = perf_counter() - started

    auto_pairs_by_identity = {
        _edge_identity(edge): auto.pair_copulas[
            auto._matrix_key(tree, edge_index)
        ]
        for tree, level in enumerate(auto.trees)
        for edge_index, edge in enumerate(level)
    }
    fixed_trees = auto.structure.to_trees()
    fixed_specs = [
        [
            (
                type(auto_pairs_by_identity[_edge_identity(edge)].copula),
                int(auto_pairs_by_identity[
                    _edge_identity(edge)
                ].copula.rotate),
            )
            for edge in level
        ]
        for level in fixed_trees
    ]
    started = perf_counter()
    fixed = VineCopula(structure=auto.structure).fit(
        u,
        method="mle",
        copulas=fixed_specs,
    )
    fixed_seconds = perf_counter() - started

    assert [
        {_edge_identity(edge) for edge in level}
        for level in fixed.trees
    ] == [
        {_edge_identity(edge) for edge in level}
        for level in auto.trees
    ]
    assert fixed.log_likelihood() == pytest.approx(
        auto.log_likelihood(),
        abs=2e-5 * auto.structure.n_edges(),
    )
    print(
        "BENCH stage8_fit"
        f" d={dimension} T={len(u)}"
        f" auto_ms={1e3 * auto_seconds:.3f}"
        f" fixed_ms={1e3 * fixed_seconds:.3f}"
        f" auto_over_fixed={auto_seconds / fixed_seconds:.3f}",
        flush=True,
    )


def test_stage8_static_sampling_scaling_profile():
    _require_benchmarks()
    for dimension in (5, 10, 20):
        u = _data(80, dimension, 20260820 + dimension)
        model = VineCopula.dvine(dimension).fit(
            u,
            method="mle",
            copulas=_specs(VineCopula.dvine(dimension).structure.to_trees()),
        )
        previous_seconds = None
        previous_n = None
        for n in (1_000, 10_000, 100_000):
            started = perf_counter()
            samples = model.sample(
                n,
                batch_rows=2_048,
                rng=np.random.default_rng(20260830 + n),
            )
            elapsed = perf_counter() - started
            assert samples.shape == (n, dimension)
            assert np.all(np.isfinite(samples))
            ratio = (
                elapsed / previous_seconds
                if previous_seconds is not None
                else float("nan")
            )
            size_ratio = (
                n / previous_n
                if previous_n is not None
                else float("nan")
            )
            print(
                "BENCH stage8_sample"
                f" d={dimension} n={n}"
                f" elapsed_ms={1e3 * elapsed:.3f}"
                f" time_ratio={ratio:.3f}"
                f" size_ratio={size_ratio:.1f}",
                flush=True,
            )
            previous_seconds = elapsed
            previous_n = n
            del samples
            gc.collect()


def test_stage8_suffix_sampling_typical_and_stress_profile():
    _require_benchmarks()
    dimension = 10
    u = _data(100, dimension, 20260840)
    model = VineCopula.dvine(dimension).fit(
        u,
        method="mle",
        copulas=_specs(VineCopula.dvine(dimension).structure.to_trees()),
    )
    for n in (1_000, 20_000):
        started = perf_counter()
        samples, diagnostics = model.predict(
            n,
            u=u,
            given={0: 0.25, 1: 0.75},
            rng=np.random.default_rng(20260841),
            return_diagnostics=True,
        )
        elapsed = perf_counter() - started
        assert samples.shape == (n, dimension)
        assert diagnostics["conditional_method"] == "suffix"
        print(
            "BENCH stage8_suffix"
            f" d={dimension} n={n}"
            f" elapsed_ms={1e3 * elapsed:.3f}",
            flush=True,
        )


def test_stage8_dag_mcmc_given_free_scaling_profile():
    _require_benchmarks()
    u = np.array([
        [1, 1, 8, 2],
        [2, 3, 7, 1],
        [3, 2, 5, 4],
        [4, 5, 6, 3],
        [5, 4, 3, 6],
        [6, 7, 4, 5],
        [7, 6, 2, 8],
        [8, 8, 1, 7],
    ], dtype=np.float64) / 9.0
    model = VineCopula(
        candidates=[IndependentCopula],
        allow_rotations=False,
    ).fit(u)
    for given in ({0: 0.25, 3: 0.75}, {0: 0.25, 1: 0.5, 3: 0.75}):
        started = perf_counter()
        samples, diagnostics = model.predict(
            1_000,
            given=given,
            mcmc_steps=5,
            mcmc_burnin=3,
            rng=np.random.default_rng(20260850 + len(given)),
            return_diagnostics=True,
        )
        elapsed = perf_counter() - started
        assert samples.shape == (1_000, 4)
        assert diagnostics["conditional_method"] == "dag_mcmc"
        print(
            "BENCH stage8_dag_mcmc"
            f" given={len(given)} free={4 - len(given)}"
            f" n=1000 elapsed_ms={1e3 * elapsed:.3f}",
            flush=True,
        )
