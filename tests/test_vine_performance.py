"""Opt-in benchmark reports and structural fast-path checks for vines."""

import os
from collections import Counter
from contextlib import contextmanager
from functools import partial
import time

import numpy as np
import pandas as pd
import pytest

from tools.benchmark_timing import interleaved_timings
from pyscarcopula._utils import clip_pseudo_observations, pobs
from pyscarcopula.copula.elliptical import BivariateGaussianCopula
from pyscarcopula.copula.independent import IndependentCopula
from pyscarcopula.strategy.scar_tm import SCARTMStrategy
from pyscarcopula.stattests import (
    rvine_rosenblatt_transform,
    student_rosenblatt_transform,
)
from pyscarcopula.vine._rvine_dag import (
    build_runtime_rvine_dag,
    plan_conditional_sample,
)
from pyscarcopula.vine.rvine import RVineCopula

from rvine_runtime_cases import (
    configured_static_dvine,
    fitted_pair,
    scalar_parameters,
)


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


def _execute_repeated(execute, repetitions=1):
    result = None
    for _ in range(repetitions):
        result = execute()
    return result


def _backend_calls(reference, native_executor, repetitions=1):
    """Build paired timings from explicit, independently named callables."""
    return {
        "reference": lambda: _execute_repeated(
            reference, repetitions),
        "native": lambda: _execute_repeated(
            native_executor, repetitions),
    }


def _student_rosenblatt_reference(correlation, df, observations):
    """Preserved SciPy formula used only as an opt-in benchmark oracle."""
    from scipy.stats import t as t_dist

    df_values = np.asarray(df)
    if df_values.ndim != 0:
        df_path = np.asarray(df_values, dtype=np.float64).ravel()
        if df_path.size == 1:
            df = float(df_path[0])
        else:
            return np.vstack([
                _student_rosenblatt_reference(
                    correlation, float(row_df), observations[row:row + 1])
                for row, row_df in enumerate(df_path)
            ])

    clipped = clip_pseudo_observations(observations)
    x = t_dist.ppf(clipped, df=df)
    result = np.empty_like(x)
    result[:, 0] = t_dist.cdf(x[:, 0], df=df)
    for coordinate in range(1, x.shape[1]):
        leading = correlation[:coordinate, :coordinate]
        cross = correlation[coordinate, :coordinate]
        inverse = np.linalg.inv(leading)
        beta = cross @ inverse
        conditional_variance = (
            correlation[coordinate, coordinate]
            - cross @ inverse @ cross
        )
        previous = x[:, :coordinate]
        mean = previous @ beta
        quadratic = np.sum(previous @ inverse * previous, axis=1)
        conditional_df = df + coordinate
        scale = (df + quadratic) / conditional_df
        z = (
            (x[:, coordinate] - mean)
            / np.sqrt(max(conditional_variance, 1e-12) * scale)
        )
        result[:, coordinate] = t_dist.cdf(z, df=conditional_df)
    return clip_pseudo_observations(result)


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
    fit_diagnostics = getattr(vine.fit_result, "diagnostics", {}) or {}
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
        "fallback_count": int(fit_diagnostics.get("fallback_count", 0)),
        "dynamic_attempted_count": int(
            fit_diagnostics.get("dynamic_attempted_count", 0)),
        "dynamic_success_count": int(
            fit_diagnostics.get("dynamic_success_count", 0)),
        "selection_nfev": int(
            fit_diagnostics.get("selection_nfev_total", 0)),
        "dynamic_attempted_nfev": int(
            fit_diagnostics.get("dynamic_attempted_nfev_total", 0)),
        "fallback_discarded_nfev": int(
            fit_diagnostics.get("fallback_discarded_nfev", 0)),
    }


@contextmanager
def _count_scar_tm_posterior_calls():
    original_predictive_state = SCARTMStrategy.predictive_state
    original_predictive_params = SCARTMStrategy.predictive_params
    original_mixture_h = SCARTMStrategy.mixture_h
    original_mixture_h_pair = SCARTMStrategy.mixture_h_pair
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
    SCARTMStrategy.mixture_h_pair = timed(
        "mixture_h_pair", original_mixture_h_pair)
    try:
        yield counts, elapsed
    finally:
        SCARTMStrategy.predictive_state = original_predictive_state
        SCARTMStrategy.predictive_params = original_predictive_params
        SCARTMStrategy.mixture_h = original_mixture_h
        SCARTMStrategy.mixture_h_pair = original_mixture_h_pair


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
def test_rvine_mle_conditional_suffix_predict_benchmark_report():
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
    _print_benchmark(
        "rvine_mle_conditional_suffix_predict",
        n=len(out),
        d=out.shape[1],
        elapsed_ms=f"{1e3 * elapsed:.3f}",
    )


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
    assert summary["dynamic_success_count"] + summary["fallback_count"] == (
        summary["dynamic_attempted_count"])
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
        fallback_count=summary["fallback_count"],
        dynamic_attempted=summary["dynamic_attempted_count"],
        dynamic_success=summary["dynamic_success_count"],
        selection_nfev=summary["selection_nfev"],
        dynamic_attempted_nfev=summary["dynamic_attempted_nfev"],
        fallback_discarded_nfev=summary["fallback_discarded_nfev"],
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
        mixture_h_pair_calls=counts["mixture_h_pair"],
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
        mixture_h_pair_ms=(
            f"{1e3 * elapsed_by_call['mixture_h_pair']:.3f}"),
    )


@pytest.mark.data
@pytest.mark.benchmark
def test_rvine_scar_conditional_suffix_cached_predict_benchmark_report():
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
    _print_benchmark(
        "rvine_scar_conditional_suffix_cached_predict",
        n=len(out),
        d=out.shape[1],
        elapsed_ms=f"{1e3 * elapsed:.3f}",
    )


@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("free_variable", "minimum_speedup", "maximum_ratio", "case"),
    [
        pytest.param(19, 1.5, None, "edge-free", id="edge-free-target"),
        pytest.param(10, None, 1.1, "central-free", id="large-closure"),
    ],
)
def test_rvine_incremental_mcmc_relative_gate(
        monkeypatch, free_variable, minimum_speedup, maximum_ratio, case):
    """Gate incremental density updates against the exact full traversal."""
    _skip_unless_vine_enabled()
    d = 20
    n = 100
    coordinate_steps = 30
    vine = configured_static_dvine(d)
    vine.pair_copulas = {
        key: fitted_pair(BivariateGaussianCopula(), 0.05)
        for key in vine.pair_copulas
    }
    parameters = scalar_parameters(vine)
    given = {
        variable: 0.2 + 0.6 * variable / (d - 1)
        for variable in range(d)
        if variable != free_variable
    }
    initial = np.random.default_rng(2026082270 + free_variable).uniform(
        0.02, 0.98, size=(n, d))
    for variable, value in given.items():
        initial[:, variable] = value
    draws = np.random.default_rng(2026082290 + free_variable).uniform(
        0.01, 0.99, size=(coordinate_steps, n, 2))
    def execute(algorithm):
        return vine._sample_arbitrary_given_mcmc(
            n,
            parameters,
            np.random.default_rng(1),
            given,
            initial=initial,
            n_steps=coordinate_steps,
            burnin_steps=0,
            random_draws=draws,
            density_algorithm=algorithm,
        )

    calls = {
        algorithm: lambda algorithm=algorithm: execute(algorithm)
        for algorithm in ("full_recompute", "incremental")
    }
    for call in calls.values():
        call()
    measured = interleaved_timings(calls, repeats=5)
    outputs = measured.results
    np.testing.assert_array_equal(
        outputs["incremental"][0], outputs["full_recompute"][0])
    assert outputs["incremental"][1] == outputs["full_recompute"][1]

    speedup = measured.median_ratio("full_recompute", "incremental")
    incremental_ratio = measured.median_ratio(
        "incremental", "full_recompute")
    _print_benchmark(
        "rvine_incremental_mcmc",
        case=case,
        d=d,
        n=n,
        coordinate_steps=coordinate_steps,
        full_ms=f"{1e3 * measured.medians['full_recompute']:.3f}",
        incremental_ms=f"{1e3 * measured.medians['incremental']:.3f}",
        speedup=f"{speedup:.3f}",
    )
    if minimum_speedup is not None:
        assert speedup >= minimum_speedup
    if maximum_ratio is not None:
        assert incremental_ratio <= maximum_ratio


@pytest.mark.benchmark
def test_dense_student_df_path_native_speedup(monkeypatch):
    """Guard the dense GAS path against returning to Python row loops."""
    _skip_unless_vine_enabled()
    dimension = 10
    rows = 1_000
    correlation = np.fromfunction(
        lambda i, j: 0.35 ** np.abs(i - j),
        (dimension, dimension),
    )
    observations = np.random.default_rng(20260822).uniform(
        0.01, 0.99, size=(rows, dimension))
    df_path = np.linspace(0.5, 20.0, rows)

    calls = _backend_calls(
        lambda: _student_rosenblatt_reference(
            correlation, df_path, observations),
        lambda: student_rosenblatt_transform(
            correlation, df_path, observations),
    )
    for call in calls.values():
        call()
    measured = interleaved_timings(calls, repeats=5)
    outputs = measured.results
    np.testing.assert_allclose(
        outputs["native"],
        outputs["reference"],
        rtol=2e-10,
        atol=2e-11,
    )
    ratio = measured.median_ratio("native", "reference")
    _print_benchmark(
        "dense_student_df_path",
        d=dimension,
        n=rows,
        python_ms=f"{1e3 * measured.medians['reference']:.3f}",
        native_ms=f"{1e3 * measured.medians['native']:.3f}",
        native_to_python=f"{ratio:.3f}",
    )
    assert ratio <= 0.2


@pytest.mark.benchmark
@pytest.mark.parametrize(("rows", "repetitions"), [(1, 100), (32, 30)])
def test_dense_student_small_input_adapter_overhead(
        monkeypatch, rows, repetitions):
    _skip_unless_vine_enabled()
    dimension = 10
    correlation = np.fromfunction(
        lambda i, j: 0.2 ** np.abs(i - j),
        (dimension, dimension),
    )
    observations = np.random.default_rng(20260823 + rows).uniform(
        0.01, 0.99, size=(rows, dimension))
    df_path = np.linspace(0.5, 10.0, rows)

    calls = _backend_calls(
        lambda: _student_rosenblatt_reference(
            correlation, df_path, observations),
        lambda: student_rosenblatt_transform(
            correlation, df_path, observations),
        repetitions,
    )
    for call in calls.values():
        call()
    measured = interleaved_timings(calls, repeats=5)
    np.testing.assert_allclose(
        measured.results["native"],
        measured.results["reference"],
        rtol=2e-10,
        atol=2e-11,
    )
    ratio = measured.median_ratio("native", "reference")
    _print_benchmark(
        "dense_student_small_input",
        d=dimension,
        n=rows,
        native_to_python=f"{ratio:.3f}",
    )
    assert ratio <= 1.5
