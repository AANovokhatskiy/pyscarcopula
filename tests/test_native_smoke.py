import numpy as np
import pytest

from pyscarcopula import BivariateGaussianCopula
from pyscarcopula._native import _scar_cpp
from pyscarcopula._native.smoke import (
    installed_distribution_boundary,
    run_native_smoke,
    validate_distribution_boundary,
)
from pyscarcopula.api import fit


def test_native_distribution_smoke():
    observations = np.array([
        [0.20, 0.70],
        [0.60, 0.30],
        [0.40, 0.80],
        [0.75, 0.25],
    ])
    result = fit(
        BivariateGaussianCopula(),
        observations,
        method="gas",
        gamma0=np.array([0.0, 0.02, 0.7]),
        maxiter=2,
        maxfun=12,
    )

    assert not hasattr(result, "backend")
    assert np.isfinite(result.log_likelihood)


def test_native_types_and_gas_methods_are_self_documenting():
    expected_class_docs = {
        _scar_cpp.CopulaSpec: "Native copula family",
        _scar_cpp.OuParams: "Ornstein-Uhlenbeck",
        _scar_cpp.OuNumericalConfig: "backend-dispatch settings",
        _scar_cpp.GasParams: "score-driven GAS recursion",
        _scar_cpp.GasConfig: "GAS score scaling",
        _scar_cpp.GasEvaluator: "GAS copula dynamics",
        _scar_cpp.RVineTraversalPlan: "Model-independent execution plan",
    }
    for native_type, phrase in expected_class_docs.items():
        assert phrase in (native_type.__doc__ or "")

    expected_parameters = {
        "initial_state": ("params", "copula", "config"),
        "filter": ("params", "copula", "u", "config"),
        "update_one": ("params", "copula", "g", "u1", "u2", "config"),
        "update_observation": (
            "params", "copula", "g", "observation", "config"),
        "predict_parameter": (
            "params", "copula", "u", "config", "horizon_next"),
    }
    for method_name, parameter_names in expected_parameters.items():
        docstring = getattr(_scar_cpp.GasEvaluator, method_name).__doc__ or ""
        for parameter_name in parameter_names:
            assert f"{parameter_name}:" in docstring

    assert _scar_cpp.GasRvinePlan is _scar_cpp.RVineTraversalPlan


def test_native_smoke_reports_distribution_and_import_boundary():
    result = run_native_smoke(n_threads=1)
    boundary = installed_distribution_boundary()

    assert result["jacobi"]["draws_used"] == 4
    assert boundary["package_imports_checked"] > 0
    assert isinstance(boundary["wheel_metadata"], bool)
    if boundary["wheel_metadata"]:
        assert boundary["distribution_files_checked"] > 0


@pytest.mark.parametrize(
    "path",
    [
        "pyscarcopula/_cpp/src/copula/core.cpp",
        "pyscarcopula/_scar_cpp.cp312-win_amd64.pyd",
        "pyscarcopula/copula/_protocol.py",
        "pyscarcopula/numerical/tm_grid.py",
        "pyscarcopula/vine/cvine.py",
        "tests/reference/oracle.py",
        "pyscarcopula/fixtures/frozen.json",
    ],
)
def test_distribution_boundary_rejects_build_oracle_and_removed_files(path):
    with pytest.raises(RuntimeError, match="distribution boundary violation"):
        validate_distribution_boundary([path], ["pyscarcopula"])


@pytest.mark.parametrize(
    "module_name",
    [
        "pyscarcopula._scar_cpp",
        "pyscarcopula.copula._protocol",
        "pyscarcopula.numerical._rvine_backend",
        "pyscarcopula.numerical.tm_grid",
        "pyscarcopula.vine.cvine",
        "pyscarcopula.tests.reference_oracle",
    ],
)
def test_distribution_boundary_rejects_removed_or_audit_only_imports(
    module_name,
):
    with pytest.raises(RuntimeError, match="distribution boundary violation"):
        validate_distribution_boundary([], ["pyscarcopula", module_name])


def test_distribution_boundary_accepts_native_wheel_layout():
    result = validate_distribution_boundary(
        [
            "pyscarcopula/__init__.py",
            "pyscarcopula/_native/_scar_cpp.cp312-win_amd64.pyd",
            "pyscarcopula/_native/pair.py",
            "pyscarcopula-1.2.3.dist-info/METADATA",
        ],
        [
            "pyscarcopula",
            "pyscarcopula._native",
            "pyscarcopula._native._scar_cpp",
        ],
    )

    assert result == {
        "distribution_files_checked": 4,
        "package_imports_checked": 3,
    }
