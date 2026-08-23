"""Canonical source manifests for the native extension and C++ tests.

Paths are relative to ``pyscarcopula/_cpp/src``.  Keep this module free of
setuptools, pybind11, and package imports so build tools can load it before the
Python package or extension exists.
"""

from __future__ import annotations

import re
from pathlib import Path


_PAIR_FAMILY_DEFINITION = (
    Path(__file__).resolve().parents[1]
    / "include" / "scar" / "copula" / "pair" / "families.def"
)
_PAIR_FAMILY_PATTERN = re.compile(
    r"^SCAR_PAIR_FAMILY\(\s*[A-Za-z][A-Za-z0-9_]*\s*,\s*"
    r"([a-z][a-z0-9_]*)\s*,"
)


def _pair_family_sources() -> tuple[str, ...]:
    """Derive pair implementation paths from the C++ registry manifest."""

    sources = []
    for raw_line in _PAIR_FAMILY_DEFINITION.read_text(
            encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue
        match = _PAIR_FAMILY_PATTERN.match(line)
        if match is None:
            raise RuntimeError(
                f"invalid pair-family registry entry: {raw_line!r}")
        sources.append(f"copula/pair/{match.group(1)}.cpp")
    if not sources:
        raise RuntimeError("pair-family registry must not be empty")
    return tuple(sources)


PAIR_FAMILY_SOURCES = _pair_family_sources()


SCAR_COMPUTE_SOURCES = (
    "copula/core.cpp",
    "copula/common.cpp",
    "copula/rotation.cpp",
    "copula/transforms.cpp",
    "copula/dispatch.cpp",
    "copula/pair/runtime_registry.cpp",
    *PAIR_FAMILY_SOURCES,
    "copula/kendall.cpp",
    "copula/families/student.cpp",
    "copula/multivariate.cpp",
    "copula/student_rosenblatt.cpp",
    "factor/operator.cpp",
    "factor/grid.cpp",
    "factor/student.cpp",
    "math/normal.cpp",
    "parallel/runtime.cpp",
    "likelihood/static.cpp",
    "gas/evaluator.cpp",
    "vine/executor.cpp",
    "vine/density.cpp",
    "vine/mcmc.cpp",
    "vine/rosenblatt.cpp",
    "gas/rvine_sampler.cpp",
    "scar_ou/monte_carlo.cpp",
    "scar_ou/validation.cpp",
    "scar_ou/likelihood.cpp",
    "scar_ou/gradient.cpp",
    "scar_ou/prediction.cpp",
    "scar_ou/gaussian_rosenblatt.cpp",
    "scar_ou/student_rosenblatt.cpp",
    "scar_ou/state_distribution.cpp",
    "scar_ou/evaluator.cpp",
    "scar_ou/prepared.cpp",
    "scar_ou/grid.cpp",
    "scar_ou/quadrature.cpp",
    "scar_ou/transition.cpp",
)

PYTHON_BINDING_SOURCES = (
    "bindings/common.cpp",
    "bindings/parallel.cpp",
    "bindings/copula.cpp",
    "bindings/factor.cpp",
    "bindings/multivariate.cpp",
    "bindings/scar_ou_types.cpp",
    "bindings/rvine.cpp",
    "bindings/gas.cpp",
    "bindings/scar_ou.cpp",
    "bindings/module.cpp",
)


def extension_sources() -> tuple[str, ...]:
    """Return the complete extension source set in deterministic order."""

    return (*SCAR_COMPUTE_SOURCES, *PYTHON_BINDING_SOURCES)
