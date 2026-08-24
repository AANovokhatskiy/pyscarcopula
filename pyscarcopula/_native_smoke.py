"""Compatibility entry point for :mod:`pyscarcopula._native.smoke`."""

from pyscarcopula._native.smoke import (
    parallel_runtime_child_probe,
    run_native_smoke,
)


if __name__ == "__main__":
    run_native_smoke()
