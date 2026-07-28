"""Compare TM-grid and Lamperti--Euler Jacobi samplers.

Example
-------
python tools/validate_jacobi_sampling.py --paths 2000 --n 33
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from scipy.stats import beta as beta_distribution
from scipy.stats import kstest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyscarcopula.numerical.jacobi_sampling import (
    sample_jacobi_lamperti_trajectory,
)
from pyscarcopula.numerical.jacobi_tm import jacobi_transition_matrix
from pyscarcopula import GumbelCopula
from pyscarcopula.numerical.jacobi_sparse import (
    jacobi_sparse_local_transition,
    jacobi_sparse_matrix_loglik,
    sparse_jacobi_full_horizon_diagnostics,
)
from pyscarcopula.numerical.jacobi_tm import jacobi_matrix_loglik


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kappa", type=float, default=1.2)
    parser.add_argument("--m", type=float, default=0.4)
    parser.add_argument("--xi", type=float, default=0.25)
    parser.add_argument("--n", type=int, default=33)
    parser.add_argument("--paths", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--basis-order", type=int, default=16)
    parser.add_argument(
        "--tm-orders", type=int, nargs="+", default=[48, 80, 128])
    parser.add_argument(
        "--lamperti-substeps", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument(
        "--lamperti-engines",
        choices=["numba", "python"],
        nargs="+",
        default=["numba"],
    )
    parser.add_argument("--chunk-observations", type=int, default=4096)
    parser.add_argument(
        "--benchmark-updates",
        type=int,
        default=0,
        help="also benchmark each Lamperti engine on at least this many updates",
    )
    parser.add_argument(
        "--crossover-n", type=int, nargs="+", default=[])
    parser.add_argument(
        "--crossover-substeps", type=int, nargs="+", default=[])
    parser.add_argument("--crossover-repeats", type=int, default=5)
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument(
        "--sparse-benchmark-observations",
        type=int,
        default=0,
        help="benchmark dense versus sparse local filtering when positive",
    )
    parser.add_argument("--sparse-benchmark-repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    return parser


def _stationary_metrics(starts, endpoints, *, alpha, beta, bins):
    target = beta_distribution(alpha, beta)
    edges = np.linspace(0.0, 1.0, bins + 1)
    observed, _ = np.histogram(endpoints, bins=edges)
    observed = observed / endpoints.size
    expected = np.diff(target.cdf(edges))
    return {
        "initial_mean": float(np.mean(starts)),
        "endpoint_mean": float(np.mean(endpoints)),
        "target_mean": float(target.mean()),
        "endpoint_variance": float(np.var(endpoints)),
        "target_variance": float(target.var()),
        "variance_ratio": float(np.var(endpoints) / target.var()),
        "ks_statistic": float(kstest(endpoints, target.cdf).statistic),
        "tv_distance": float(0.5 * np.sum(np.abs(observed - expected))),
    }


def _conditional_mean_rmse(starts, endpoints, *, kappa, m, dt, bins):
    edges = np.quantile(starts, np.linspace(0.0, 1.0, bins + 1))
    edges = np.unique(edges)
    if edges.size < 3:
        return float("nan")
    assignments = np.searchsorted(edges[1:-1], starts, side="right")
    errors = []
    for index in range(edges.size - 1):
        selected = assignments == index
        if np.count_nonzero(selected) < 5:
            continue
        start_mean = float(np.mean(starts[selected]))
        empirical = float(np.mean(endpoints[selected]))
        expected = m + (start_mean - m) * np.exp(-kappa * dt)
        errors.append(empirical - expected)
    return float(np.sqrt(np.mean(np.square(errors))))


def _sample_tm_ensemble(
        *, kappa, m, xi, n, paths, seed, basis_order, quad_order):
    tau, stationary, transition, diagnostics = jacobi_transition_matrix(
        kappa,
        m,
        xi,
        n_obs=n,
        basis_order=basis_order,
        quad_order=quad_order,
        transition_method="auto",
        return_diagnostics=True,
    )
    stationary_cdf = np.cumsum(stationary)
    stationary_cdf[-1] = 1.0
    transition_cdf = np.cumsum(transition, axis=1)
    transition_cdf[:, -1] = 1.0
    rng = np.random.default_rng(seed)
    indices = np.searchsorted(
        stationary_cdf, rng.random(paths), side="right")
    starts = tau[indices].copy()
    for _ in range(n - 1):
        uniforms = rng.random(paths)
        indices = np.fromiter(
            (
                np.searchsorted(
                    transition_cdf[index], draw, side="right")
                for index, draw in zip(indices, uniforms)
            ),
            dtype=np.intp,
            count=paths,
        )
    return starts, tau[indices], diagnostics


def _sample_lamperti_ensemble(
        *, kappa, m, xi, n, paths, seed, substeps, engine,
        chunk_observations):
    rng = np.random.default_rng(seed)
    starts = np.empty(paths, dtype=np.float64)
    endpoints = np.empty(paths, dtype=np.float64)
    interventions = 0
    euler_steps = 0
    for index in range(paths):
        path, diagnostics = sample_jacobi_lamperti_trajectory(
            kappa,
            m,
            xi,
            n,
            rng=rng,
            substeps=substeps,
            engine=engine,
            chunk_observations=chunk_observations,
            return_diagnostics=True,
        )
        starts[index] = path[0]
        endpoints[index] = path[-1]
        interventions += diagnostics["boundary_interventions"]
        euler_steps += diagnostics["euler_steps"]
    return starts, endpoints, {
        "boundary_interventions": interventions,
        "boundary_intervention_rate": (
            interventions / euler_steps if euler_steps else 0.0),
    }


def _benchmark_lamperti_engines(args):
    if args.benchmark_updates <= 0:
        return []
    substeps = max(args.lamperti_substeps)
    intervals = max(
        1, (args.benchmark_updates + substeps - 1) // substeps)
    n = intervals + 1
    records = []
    for engine in args.lamperti_engines:
        cold_rng = np.random.default_rng(args.seed)
        cold_started = perf_counter()
        sample_jacobi_lamperti_trajectory(
            args.kappa,
            args.m,
            args.xi,
            2,
            rng=cold_rng,
            substeps=substeps,
            engine=engine,
            chunk_observations=args.chunk_observations,
        )
        cold_seconds = perf_counter() - cold_started

        warm_rng = np.random.default_rng(args.seed)
        warm_started = perf_counter()
        _, diagnostics = sample_jacobi_lamperti_trajectory(
            args.kappa,
            args.m,
            args.xi,
            n,
            rng=warm_rng,
            substeps=substeps,
            engine=engine,
            chunk_observations=args.chunk_observations,
            return_diagnostics=True,
        )
        warm_seconds = perf_counter() - warm_started
        records.append({
            "engine": engine,
            "cold_start_seconds": cold_seconds,
            "warm_seconds": warm_seconds,
            "euler_updates": diagnostics["euler_steps"],
            "updates_per_second": (
                diagnostics["euler_steps"] / warm_seconds),
            "chunk_observations": diagnostics["chunk_observations"],
            "boundary_interventions": diagnostics[
                "boundary_interventions"],
        })
    timings = {
        record["engine"]: record["warm_seconds"]
        for record in records
    }
    if "numba" in timings and "python" in timings:
        speedup = timings["python"] / timings["numba"]
        for record in records:
            record["numba_speedup_vs_python"] = speedup
    return records


def _benchmark_lamperti_crossover(args):
    if not args.crossover_n or not args.crossover_substeps:
        return []
    if args.crossover_repeats <= 0:
        raise ValueError("--crossover-repeats must be positive")
    # Compile/load the sequential kernel outside warm timing.
    sample_jacobi_lamperti_trajectory(
        args.kappa,
        args.m,
        args.xi,
        2,
        rng=np.random.default_rng(args.seed),
        substeps=1,
        engine="numba",
        chunk_observations=args.chunk_observations,
    )
    records = []
    for n in args.crossover_n:
        for substeps in args.crossover_substeps:
            timings = {}
            for engine in args.lamperti_engines:
                samples = []
                for repeat in range(args.crossover_repeats):
                    rng = np.random.default_rng(args.seed + repeat)
                    started = perf_counter()
                    sample_jacobi_lamperti_trajectory(
                        args.kappa,
                        args.m,
                        args.xi,
                        n,
                        rng=rng,
                        substeps=substeps,
                        engine=engine,
                        chunk_observations=args.chunk_observations,
                    )
                    samples.append(perf_counter() - started)
                timings[engine] = float(np.median(samples))
            record = {
                "n": n,
                "substeps": substeps,
                "euler_updates": max(n - 1, 0) * substeps,
                "median_seconds": timings,
            }
            if "numba" in timings and "python" in timings:
                record["numba_speedup_vs_python"] = (
                    timings["python"] / timings["numba"])
            records.append(record)
    return records


def _sparse_transition_diagnostics(args):
    records = []
    for order in args.tm_orders:
        for correction in ("none", "mh"):
            tau, weights, transition, construction = (
                jacobi_sparse_local_transition(
                    args.kappa,
                    args.m,
                    args.xi,
                    n_obs=args.n,
                    basis_order=min(args.basis_order, order),
                    quad_order=order,
                    correction=correction,
                    return_diagnostics=True,
                )
            )
            horizon = sparse_jacobi_full_horizon_diagnostics(
                tau,
                weights,
                transition,
                steps=args.n - 1,
                kappa=args.kappa,
                m=args.m,
            )
            records.append({
                "quad_order": order,
                "correction": correction,
                **construction,
                **horizon,
            })
    return records


def _benchmark_sparse_filter(args):
    n_obs = args.sparse_benchmark_observations
    if n_obs <= 0:
        return []
    if args.sparse_benchmark_repeats <= 0:
        raise ValueError("--sparse-benchmark-repeats must be positive")
    u = np.random.default_rng(args.seed).uniform(
        0.05, 0.95, size=(n_obs, 2))
    copula = GumbelCopula()
    records = []
    for order in args.tm_orders:
        common = {
            "basis_order": min(args.basis_order, order),
            "quad_order": order,
            "gh_order": 5,
        }
        dense_kwargs = {**common, "transition_method": "local"}
        # Warm both code paths before timing.
        dense_value = jacobi_matrix_loglik(
            args.kappa, args.m, args.xi, u, copula, **dense_kwargs)
        sparse_value = jacobi_sparse_matrix_loglik(
            args.kappa, args.m, args.xi, u, copula, **common)
        timings = {"dense": [], "sparse": []}
        for _ in range(args.sparse_benchmark_repeats):
            started = perf_counter()
            jacobi_matrix_loglik(
                args.kappa, args.m, args.xi, u, copula, **dense_kwargs)
            timings["dense"].append(perf_counter() - started)
            started = perf_counter()
            jacobi_sparse_matrix_loglik(
                args.kappa, args.m, args.xi, u, copula, **common)
            timings["sparse"].append(perf_counter() - started)
        _, _, sparse_transition, diagnostics = (
            jacobi_sparse_local_transition(
                args.kappa,
                args.m,
                args.xi,
                n_obs=n_obs,
                quad_order=order,
                gh_order=5,
                return_diagnostics=True,
            )
        )
        dense_seconds = float(np.median(timings["dense"]))
        sparse_seconds = float(np.median(timings["sparse"]))
        records.append({
            "quad_order": order,
            "n_obs": n_obs,
            "dense_median_seconds": dense_seconds,
            "sparse_median_seconds": sparse_seconds,
            "speedup": dense_seconds / sparse_seconds,
            "loglik_absolute_difference": abs(
                dense_value - sparse_value),
            "dense_transition_bytes": diagnostics["dense_bytes"],
            "sparse_transition_bytes": sparse_transition.retained_bytes,
            "transition_memory_reduction": (
                diagnostics["dense_bytes"]
                / sparse_transition.retained_bytes),
        })
    return records


def run_validation(args):
    if args.n < 2:
        raise ValueError("--n must be at least 2")
    if args.paths <= 0:
        raise ValueError("--paths must be positive")
    alpha = 2.0 * args.kappa * args.m / args.xi ** 2
    beta = 2.0 * args.kappa * (1.0 - args.m) / args.xi ** 2
    dt = 1.0 / (args.n - 1)
    report = {
        "contract": {
            "kappa": args.kappa,
            "m": args.m,
            "xi": args.xi,
            "stationary_alpha": alpha,
            "stationary_beta": beta,
            "n": args.n,
            "paths": args.paths,
            "horizon": 1.0,
            "seed": args.seed,
        },
        "tm_grid": [],
        "lamperti_euler": [],
        "engine_benchmark": _benchmark_lamperti_engines(args),
        "crossover_benchmark": _benchmark_lamperti_crossover(args),
        "sparse_transition": _sparse_transition_diagnostics(args),
        "sparse_filter_benchmark": _benchmark_sparse_filter(args),
    }
    for order in args.tm_orders:
        started = perf_counter()
        starts, endpoints, diagnostics = _sample_tm_ensemble(
            kappa=args.kappa,
            m=args.m,
            xi=args.xi,
            n=args.n,
            paths=args.paths,
            seed=args.seed,
            basis_order=min(args.basis_order, order),
            quad_order=order,
        )
        metrics = _stationary_metrics(
            starts, endpoints, alpha=alpha, beta=beta, bins=args.bins)
        metrics.update({
            "quad_order": order,
            "conditional_mean_rmse": _conditional_mean_rmse(
                starts,
                endpoints,
                kappa=args.kappa,
                m=args.m,
                dt=1.0,
                bins=min(args.bins, 10),
            ),
            "elapsed_seconds": perf_counter() - started,
            "transition_method": diagnostics.get("transition_method"),
        })
        report["tm_grid"].append(metrics)

    for engine in args.lamperti_engines:
        for substeps in args.lamperti_substeps:
            started = perf_counter()
            starts, endpoints, diagnostics = _sample_lamperti_ensemble(
                kappa=args.kappa,
                m=args.m,
                xi=args.xi,
                n=args.n,
                paths=args.paths,
                seed=args.seed,
                substeps=substeps,
                engine=engine,
                chunk_observations=args.chunk_observations,
            )
            metrics = _stationary_metrics(
                starts, endpoints, alpha=alpha, beta=beta, bins=args.bins)
            metrics.update({
                "engine": engine,
                "substeps": substeps,
                "conditional_mean_rmse": _conditional_mean_rmse(
                    starts,
                    endpoints,
                    kappa=args.kappa,
                    m=args.m,
                    dt=1.0,
                    bins=min(args.bins, 10),
                ),
                "elapsed_seconds": perf_counter() - started,
                **diagnostics,
            })
            report["lamperti_euler"].append(metrics)
    return report


def main():
    args = _parser().parse_args()
    report = run_validation(args)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
