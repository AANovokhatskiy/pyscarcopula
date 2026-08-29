"""Smoke test for the bundled native dynamic-model evaluators."""

from __future__ import annotations

import argparse
from importlib import metadata
import importlib.util
import json
import sys

import numpy as np

from pyscarcopula._native import _extension
from pyscarcopula._native.threads import validate_n_threads


_REMOVED_DISTRIBUTION_PATHS = frozenset({
    "pyscarcopula/_native/gof.py",
    "pyscarcopula/copula/_protocol.py",
    "pyscarcopula/numerical/_cpp_copula.py",
    "pyscarcopula/numerical/_cpp_extension.py",
    "pyscarcopula/numerical/_cpp_gas.py",
    "pyscarcopula/numerical/_cpp_gas_rvine.py",
    "pyscarcopula/numerical/_cpp_rvine.py",
    "pyscarcopula/numerical/_cpp_scar_ou.py",
    "pyscarcopula/numerical/copula_native.py",
    "pyscarcopula/numerical/gof_blocks.py",
    "pyscarcopula/numerical/mc_native.py",
    "pyscarcopula/numerical/mc_samplers.py",
    "pyscarcopula/numerical/multivariate_native.py",
    "pyscarcopula/numerical/static_likelihood.py",
    "pyscarcopula/numerical/student_gof.py",
    "pyscarcopula/numerical/tm_grid.py",
    "pyscarcopula/strategy/scar_mc.py",
    "pyscarcopula/vine/_conditional_cvine.py",
    "pyscarcopula/vine/_rvine_conditional_runtime.py",
    "pyscarcopula/vine/cvine.py",
})
_REMOVED_IMPORTS = frozenset({
    "pyscarcopula._scar_cpp",
    "pyscarcopula._native.gof",
    "pyscarcopula.copula._protocol",
    "pyscarcopula.numerical._cpp_copula",
    "pyscarcopula.numerical._cpp_extension",
    "pyscarcopula.numerical._cpp_gas",
    "pyscarcopula.numerical._cpp_gas_rvine",
    "pyscarcopula.numerical._cpp_rvine",
    "pyscarcopula.numerical._cpp_scar_ou",
    "pyscarcopula.numerical.copula_native",
    "pyscarcopula.numerical.gof_blocks",
    "pyscarcopula.numerical.mc_native",
    "pyscarcopula.numerical.mc_samplers",
    "pyscarcopula.numerical.multivariate_native",
    "pyscarcopula.numerical.static_likelihood",
    "pyscarcopula.numerical.student_gof",
    "pyscarcopula.numerical.tm_grid",
    "pyscarcopula.strategy.scar_mc",
    "pyscarcopula.vine._conditional_cvine",
    "pyscarcopula.vine._rvine_conditional_runtime",
    "pyscarcopula.vine.cvine",
})
_AUDIT_ONLY_PARTS = frozenset({
    "benchmark_artifacts",
    "fixtures",
    "oracle",
    "oracles",
    "reference",
    "references",
    "tests",
})


def validate_distribution_boundary(files, imported_modules) -> dict:
    """Reject build, oracle, fallback, and removed paths from a wheel run."""
    normalized_files = tuple(
        str(value).replace("\\", "/").lstrip("./") for value in files
    )
    normalized_imports = tuple(str(value) for value in imported_modules)
    violations = []

    for path in normalized_files:
        lowered = path.lower()
        parts = tuple(part for part in lowered.split("/") if part)
        filename = parts[-1] if parts else ""
        if lowered.startswith("pyscarcopula/_cpp/"):
            violations.append(f"build-only path in distribution: {path}")
        if lowered.startswith("pyscarcopula/_scar_cpp."):
            violations.append(f"removed raw extension path in distribution: {path}")
        if lowered in _REMOVED_DISTRIBUTION_PATHS:
            violations.append(f"removed compatibility path in distribution: {path}")
        if (
            lowered.startswith("pyscarcopula/numerical/")
            and (
                filename.startswith("_cpp_")
                or filename.endswith("_backend.py")
                or filename.endswith("_native.py")
            )
        ):
            violations.append(f"retired numerical adapter in distribution: {path}")
        if any(part in _AUDIT_ONLY_PARTS for part in parts):
            violations.append(f"audit-only path in distribution: {path}")
        if lowered.endswith((".cpp", ".hpp", ".h", ".obj", ".o")):
            violations.append(f"build source in distribution: {path}")

    package_imports = tuple(
        name for name in normalized_imports
        if name == "pyscarcopula" or name.startswith("pyscarcopula.")
    )
    for name in package_imports:
        parts = tuple(part.lower() for part in name.split("."))
        leaf = parts[-1] if parts else ""
        if name in _REMOVED_IMPORTS:
            violations.append(f"removed compatibility import is loaded: {name}")
        if (
            name.startswith("pyscarcopula.numerical.")
            and (
                leaf.startswith("_cpp_")
                or leaf.endswith("_backend")
                or leaf.endswith("_native")
            )
        ):
            violations.append(f"retired numerical adapter is loaded: {name}")
        if any(part in _AUDIT_ONLY_PARTS for part in parts):
            violations.append(f"audit-only import is loaded: {name}")

    if violations:
        raise RuntimeError(
            "installed distribution boundary violation: "
            + "; ".join(sorted(set(violations)))
        )
    return {
        "distribution_files_checked": len(normalized_files),
        "package_imports_checked": len(package_imports),
    }


def installed_distribution_boundary() -> dict:
    """Validate wheel contents when wheel metadata is available."""
    distribution = metadata.distribution("pyscarcopula")
    is_wheel_install = distribution.read_text("WHEEL") is not None
    files = distribution.files or () if is_wheel_install else ()
    result = validate_distribution_boundary(files, sys.modules)
    result["wheel_metadata"] = is_wheel_install
    return result


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
    result["distribution_boundary"] = installed_distribution_boundary()
    if args.json:
        print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
