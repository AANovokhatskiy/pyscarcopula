from pyscarcopula._native_smoke import run_native_smoke
from pyscarcopula import _scar_cpp


def test_native_distribution_smoke():
    run_native_smoke()


def test_native_types_and_gas_methods_are_self_documenting():
    expected_class_docs = {
        _scar_cpp.CopulaSpec: "Native copula family",
        _scar_cpp.OuParams: "Ornstein-Uhlenbeck",
        _scar_cpp.OuNumericalConfig: "backend-dispatch settings",
        _scar_cpp.GasParams: "score-driven GAS recursion",
        _scar_cpp.GasConfig: "GAS score scaling",
        _scar_cpp.GasEvaluator: "GAS copula dynamics",
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
