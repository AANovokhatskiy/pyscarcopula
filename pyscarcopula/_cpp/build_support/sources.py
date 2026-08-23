"""Canonical source manifests for the native extension and C++ tests.

Paths are relative to ``pyscarcopula/_cpp/src``.  Keep this module free of
setuptools, pybind11, and package imports so build tools can load it before the
Python package or extension exists.
"""

from __future__ import annotations


SCAR_COMPUTE_SOURCES = (
    "copula/core.cpp",
    "copula/common.cpp",
    "copula/dispatch.cpp",
    "copula/families/clayton.cpp",
    "copula/families/gumbel.cpp",
    "copula/families/frank.cpp",
    "copula/families/joe.cpp",
    "copula/families/gaussian.cpp",
    "copula/kendall.cpp",
    "copula/families/student.cpp",
    "copula/multivariate.cpp",
    "copula/student_rosenblatt.cpp",
    "factor/operator.cpp",
    "factor/grid.cpp",
    "factor/student.cpp",
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

