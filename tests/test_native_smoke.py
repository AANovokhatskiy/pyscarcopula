import numpy as np

from pyscarcopula import BivariateGaussianCopula
from pyscarcopula._native import _scar_cpp
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
