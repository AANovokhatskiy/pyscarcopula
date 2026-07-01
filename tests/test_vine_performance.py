"""Optional performance regression checks for vine workloads."""

import os
from collections import Counter
from contextlib import contextmanager
import time

import numpy as np
import pandas as pd
import pytest

from pyscarcopula._utils import pobs
from pyscarcopula.copula.elliptical import BivariateGaussianCopula
from pyscarcopula.copula.independent import IndependentCopula
from pyscarcopula.strategy.scar_tm import SCARTMStrategy
from pyscarcopula.vine.rvine import RVineCopula


def _example_u():
    crypto_prices = pd.read_csv("data/crypto_prices.csv", index_col=0, sep=";")
    tickers = [
        "BTC-USD",
        "ETH-USD",
        "BNB-USD",
        "ADA-USD",
        "XRP-USD",
        "DOGE-USD",
    ]
    returns = np.log(crypto_prices[tickers] / crypto_prices[tickers].shift(1))
    return pobs(returns[1:251].values)


def _skip_unless_enabled():
    if os.environ.get("PYSCA_RUN_BENCHMARKS") != "1":
        pytest.skip("set PYSCA_RUN_BENCHMARKS=1 to run benchmark checks")


def _skip_unless_vine_enabled():
    if (
            os.environ.get("PYSCA_RUN_VINE_BENCHMARKS") != "1"
            and os.environ.get("PYSCA_RUN_BENCHMARKS") != "1"):
        pytest.skip(
            "set PYSCA_RUN_VINE_BENCHMARKS=1 or PYSCA_RUN_BENCHMARKS=1 "
            "to run vine workload benchmarks")


def _print_benchmark(name, **fields):
    values = " ".join(f"{key}={value}" for key, value in fields.items())
    print(f"BENCH {name} {values}", flush=True)


def _synthetic_u(T, d, seed):
    rng = np.random.default_rng(seed)
    common = rng.standard_normal((T, 1))
    loadings = np.linspace(0.15, 0.65, d)
    noise = rng.standard_normal((T, d))
    raw = loadings * common + np.sqrt(1.0 - loadings ** 2) * noise
    raw[:, 1:] += 0.25 * raw[:, :-1]
    return pobs(raw)


def _fixed_gaussian_copulas(d):
    return [
        [(BivariateGaussianCopula, 0) for _ in range(d - 1 - tree)]
        for tree in range(d - 1)
    ]


def _edge_summary(vine):
    methods = Counter()
    families = Counter()
    scar_nfev = 0
    prepared_edges = 0
    dynamic_edges = 0
    total_nfev = 0
    for edge in vine.pair_copulas.values():
        result = getattr(edge, "fit_result", None)
        method = str(getattr(result, "method", None)).upper()
        if method == "NONE":
            method = "STATIC"
        methods[method] += 1
        families[type(getattr(edge, "copula", None)).__name__] += 1
        nfev = int(getattr(result, "nfev", 0) or getattr(edge, "nfev", 0) or 0)
        total_nfev += nfev
        if method == "SCAR-TM-OU":
            dynamic_edges += 1
            scar_nfev += nfev
            diagnostics = getattr(result, "diagnostics", {}) or {}
            if diagnostics.get("prepared_native_evaluator"):
                prepared_edges += 1
    return {
        "edges": len(vine.pair_copulas),
        "methods": ",".join(
            f"{name}:{count}" for name, count in sorted(methods.items())),
        "families": ",".join(
            f"{name}:{count}" for name, count in sorted(families.items())),
        "dynamic_edges": dynamic_edges,
        "scar_edges": methods.get("SCAR-TM-OU", 0),
        "scar_nfev": scar_nfev,
        "total_nfev": total_nfev,
        "prepared_scar_edges": prepared_edges,
    }


@contextmanager
def _count_scar_tm_posterior_calls():
    original_predictive_state = SCARTMStrategy.predictive_state
    original_predictive_params = SCARTMStrategy.predictive_params
    original_mixture_h = SCARTMStrategy.mixture_h
    counts = Counter()
    elapsed = Counter()

    def timed(name, original):
        def wrapper(self, *args, **kwargs):
            counts[name] += 1
            start = time.perf_counter()
            try:
                return original(self, *args, **kwargs)
            finally:
                elapsed[name] += time.perf_counter() - start
        return wrapper

    SCARTMStrategy.predictive_state = timed(
        "predictive_state", original_predictive_state)
    SCARTMStrategy.predictive_params = timed(
        "predictive_params", original_predictive_params)
    SCARTMStrategy.mixture_h = timed("mixture_h", original_mixture_h)
    try:
        yield counts, elapsed
    finally:
        SCARTMStrategy.predictive_state = original_predictive_state
        SCARTMStrategy.predictive_params = original_predictive_params
        SCARTMStrategy.mixture_h = original_mixture_h


_SYNTHETIC_FIT_WORKLOADS = [
    pytest.param("mle-static", "mle", 5, 80, [BivariateGaussianCopula],
                 {}, False, id="fit-mle-d5-T80"),
    pytest.param(
        "scar-heavy-short",
        "scar-tm-ou",
        5,
        40,
        [BivariateGaussianCopula],
        {
            "K": 10,
            "max_K": 10,
            "adaptive": False,
            "analytical_grad": True,
            "maxiter": 20,
            "maxfun": 40,
            "smart_init": False,
        },
        True,
        id="fit-scar-d5-T40",
    ),
    pytest.param(
        "scar-heavy-medium",
        "scar-tm-ou",
        8,
        80,
        [BivariateGaussianCopula],
        {
            "K": 12,
            "max_K": 12,
            "adaptive": False,
            "analytical_grad": True,
            "maxiter": 20,
            "maxfun": 40,
            "smart_init": False,
        },
        True,
        id="fit-scar-d8-T80",
    ),
    pytest.param(
        "independent-heavy",
        "mle",
        10,
        80,
        [IndependentCopula, BivariateGaussianCopula],
        {"threshold": 0.15},
        False,
        id="fit-independent-d10-T80",
    ),
]

_SYNTHETIC_PREDICT_WORKLOADS = [
    pytest.param("unconditional", None, "ignore", "next",
                 id="predict-unconditional"),
    pytest.param("suffix-given", {0: 0.25, 1: 0.75}, "ignore", "next",
                 id="predict-suffix-given"),
    pytest.param("given-only-current", {0: 0.25, 1: 0.75}, "given_only",
                 "current", id="predict-given-only-current"),
    pytest.param("given-only-next", {0: 0.25, 1: 0.75}, "given_only",
                 "next", id="predict-given-only-next"),
]


@pytest.mark.data
@pytest.mark.benchmark
def test_rvine_mle_conditional_suffix_predict_speed_smoke():
    _skip_unless_enabled()
    u = _example_u()
    vine = RVineCopula()
    vine.fit(u, method="mle")

    t0 = time.perf_counter()
    out, diagnostics = vine.predict(
        1000,
        u=u,
        given={0: 0.2, 1: 0.8},
        horizon="next",
        rng=np.random.default_rng(20260602),
        return_diagnostics=True,
    )
    elapsed = time.perf_counter() - t0

    assert out.shape == (1000, 6)
    assert diagnostics["conditional_method"] == "suffix"
    assert elapsed < 2.0


@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("name", "method", "d", "T", "candidates", "fit_kwargs",
     "fixed_gaussian"),
    _SYNTHETIC_FIT_WORKLOADS,
)
def test_rvine_synthetic_fit_profile(
        name, method, d, T, candidates, fit_kwargs, fixed_gaussian):
    _skip_unless_vine_enabled()
    u = _synthetic_u(T=T, d=d, seed=20260710 + 31 * d + T)
    vine = RVineCopula(candidates=candidates, allow_rotations=False)
    copulas = _fixed_gaussian_copulas(d) if fixed_gaussian else None

    start = time.perf_counter()
    vine.fit(u, method=method, copulas=copulas, **fit_kwargs)
    elapsed = time.perf_counter() - start

    summary = _edge_summary(vine)
    assert summary["edges"] == d * (d - 1) // 2
    _print_benchmark(
        "rvine_fit",
        workload=name,
        method=method,
        d=d,
        T=T,
        elapsed_ms=f"{1e3 * elapsed:.3f}",
        edges=summary["edges"],
        dynamic_edges=summary["dynamic_edges"],
        scar_edges=summary["scar_edges"],
        total_nfev=summary["total_nfev"],
        scar_nfev=summary["scar_nfev"],
        prepared_scar_edges=summary["prepared_scar_edges"],
        methods=summary["methods"],
        families=summary["families"],
    )


@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("name", "given", "dynamic_conditioning", "horizon"),
    _SYNTHETIC_PREDICT_WORKLOADS,
)
def test_rvine_scar_synthetic_predict_profile(
        name, given, dynamic_conditioning, horizon):
    _skip_unless_vine_enabled()
    d = 5
    T = 50
    n = 300
    u = _synthetic_u(T=T, d=d, seed=20260720)
    vine = RVineCopula(
        candidates=[BivariateGaussianCopula],
        allow_rotations=False,
    )
    vine.fit(
        u,
        method="scar-tm-ou",
        copulas=_fixed_gaussian_copulas(d),
        K=10,
        max_K=10,
        adaptive=False,
        analytical_grad=True,
        maxiter=20,
        maxfun=40,
        smart_init=False,
    )
    summary = _edge_summary(vine)

    with _count_scar_tm_posterior_calls() as (counts, elapsed_by_call):
        start = time.perf_counter()
        out, diagnostics = vine.predict(
            n,
            u=u,
            given=given,
            horizon=horizon,
            dynamic_conditioning=dynamic_conditioning,
            return_diagnostics=True,
            rng=np.random.default_rng(20260721),
        )
        elapsed = time.perf_counter() - start

    assert out.shape == (n, d)
    timings = diagnostics.get("timings_ms", {})
    updated_edges = len(diagnostics.get("updated_edges", ()))
    skipped_edges = len(diagnostics.get("skipped_edges", ()))
    _print_benchmark(
        "rvine_predict",
        workload=name,
        d=d,
        T=T,
        n=n,
        horizon=horizon,
        dynamic_conditioning=dynamic_conditioning,
        given_count=0 if given is None else len(given),
        elapsed_ms=f"{1e3 * elapsed:.3f}",
        edges=summary["edges"],
        scar_edges=summary["scar_edges"],
        conditional_method=diagnostics.get("conditional_method"),
        matrix_rebuilt=diagnostics.get("matrix_rebuilt"),
        updated_edges=updated_edges,
        skipped_edges=skipped_edges,
        predictive_params_calls=counts["predictive_params"],
        predictive_state_calls=counts["predictive_state"],
        mixture_h_calls=counts["mixture_h"],
        total_ms=f"{timings.get('total', 0.0):.3f}",
        compute_pseudo_obs_ms=(
            f"{timings.get('compute_pseudo_obs', 0.0):.3f}"),
        predict_r_for_edges_ms=(
            f"{timings.get('predict_r_for_edges', 0.0):.3f}"),
        dynamic_update_ms=f"{timings.get('dynamic_update', 0.0):.3f}",
        suffix_sample_ms=f"{timings.get('suffix_sample', 0.0):.3f}",
        unconditional_sample_ms=(
            f"{timings.get('unconditional_sample', 0.0):.3f}"),
        predictive_params_ms=(
            f"{1e3 * elapsed_by_call['predictive_params']:.3f}"),
        predictive_state_ms=(
            f"{1e3 * elapsed_by_call['predictive_state']:.3f}"),
        mixture_h_ms=f"{1e3 * elapsed_by_call['mixture_h']:.3f}",
    )


@pytest.mark.data
@pytest.mark.benchmark
def test_rvine_scar_conditional_suffix_cached_predict_speed_smoke():
    _skip_unless_enabled()
    u = _example_u()
    vine = RVineCopula()
    vine.fit(u, method="scar-tm-ou")
    vine.predict(
        10,
        u=u,
        given={0: 0.2, 1: 0.8},
        horizon="next",
        rng=np.random.default_rng(20260602),
    )

    t0 = time.perf_counter()
    out, diagnostics = vine.predict(
        1000,
        u=u,
        given={0: 0.2, 1: 0.8},
        horizon="next",
        rng=np.random.default_rng(20260603),
        return_diagnostics=True,
    )
    elapsed = time.perf_counter() - t0

    assert out.shape == (1000, 6)
    assert diagnostics["conditional_method"] == "suffix"
    assert elapsed < 2.0
