"""Smoke test for the bundled native dynamic-model evaluators."""

from __future__ import annotations

import argparse
import importlib.util
import json

import numpy as np

from pyscarcopula._native import _extension
from pyscarcopula._native.threads import validate_n_threads


def parallel_runtime_child_probe(queue, n_threads: int) -> None:
    """Exercise the native runtime from an importable multiprocessing target."""
    module = _extension.load()
    module._parallel_for_blocks_probe(16, 1, n_threads)
    queue.put(dict(module._parallel_runtime_info()))


def run_native_smoke(n_threads: int = 1) -> dict:
    """Exercise native dynamic models and the requested parallel runtime."""
    from pyscarcopula.api import fit
    from pyscarcopula.copula.elliptical import BivariateGaussianCopula
    from pyscarcopula._native import jacobi

    n_threads = validate_n_threads(n_threads)
    module = _extension.load()
    if module.__name__ != "pyscarcopula._native._scar_cpp":
        raise RuntimeError(
            "native extension loaded from an unexpected import path: "
            f"{module.__name__}"
        )
    if importlib.util.find_spec("pyscarcopula._scar_cpp") is not None:
        raise RuntimeError(
            "removed raw extension path pyscarcopula._scar_cpp is importable"
        )
    parallel = dict(module._parallel_for_blocks_probe(
        max(32, 4 * n_threads), 1, n_threads))

    u = np.array([
        [0.20, 0.70],
        [0.60, 0.30],
        [0.40, 0.80],
        [0.75, 0.25],
    ], dtype=np.float64)
    result = fit(
        BivariateGaussianCopula(),
        u,
        method="gas",
        gamma0=np.array([0.0, 0.02, 0.7]),
        maxiter=2,
        maxfun=12,
    )
    if hasattr(result, "backend"):
        raise RuntimeError("GASResult must not expose backend selection")
    if not np.isfinite(result.log_likelihood):
        raise RuntimeError("native GAS fit returned non-finite logL")

    evaluator = jacobi.PreparedScarJacobiEvaluator(
        u,
        BivariateGaussianCopula(),
        basis_order=4,
        quad_order=16,
        gh_order=3,
        transition_method="local_fixed",
    )
    objective, gradient = evaluator.neg_loglik_with_grad(1.2, 0.4, 0.25)
    state = evaluator.filter(1.2, 0.4, 0.25)
    residual = evaluator.rosenblatt(1.2, 0.4, 0.25)
    sampled, diagnostics = jacobi.sample_grid_trajectory_fixed_draws(
        1.2,
        0.4,
        0.25,
        np.array([0.13, 0.37, 0.61, 0.89], dtype=np.float64),
        basis_order=4,
        quad_order=16,
        gh_order=3,
        method="local_fixed",
    )
    if not np.isfinite(objective) or not np.all(np.isfinite(gradient)):
        raise RuntimeError("native Jacobi evaluator returned non-finite output")
    if state["smoothed"].shape != (len(u), 16):
        raise RuntimeError("native Jacobi smoother returned an invalid shape")
    if residual.shape != u.shape or sampled.shape != (len(u),):
        raise RuntimeError("native Jacobi residual/sampler shape is invalid")
    if diagnostics["draws_used"] != len(u):
        raise RuntimeError("native Jacobi fixed-draw contract was violated")
    return {
        "n_threads_requested": n_threads,
        "parallel_runtime": dict(parallel["runtime"]),
        "parallel_block_count": len(parallel["block_ids"]),
        "jacobi": {
            "objective": float(objective),
            "gradient_size": int(np.asarray(gradient).size),
            "state_rows": int(state["smoothed"].shape[0]),
            "draws_used": int(diagnostics["draws_used"]),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-threads", type=int, default=1)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = run_native_smoke(args.n_threads)
    if args.json:
        print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
