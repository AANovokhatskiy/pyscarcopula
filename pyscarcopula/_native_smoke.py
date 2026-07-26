"""Smoke test for the bundled native extension and GAS evaluator."""

from __future__ import annotations

import numpy as np


def parallel_runtime_child_probe(queue, n_threads: int) -> None:
    """Exercise the native runtime from an importable multiprocessing target."""
    from pyscarcopula.numerical import _cpp_extension

    module = _cpp_extension.load()
    module._parallel_for_blocks_probe(16, 1, n_threads)
    queue.put(dict(module._parallel_runtime_info()))


def run_native_smoke() -> None:
    import pyscarcopula._scar_cpp  # noqa: F401
    from pyscarcopula.api import fit
    from pyscarcopula.copula.elliptical import BivariateGaussianCopula
    from pyscarcopula.numerical import _cpp_gas

    if not _cpp_gas.available():
        raise RuntimeError("pyscarcopula native GAS evaluator is unavailable")

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


if __name__ == "__main__":
    run_native_smoke()
