"""Loader and shared errors for the bundled pyscarcopula C++ extension."""

from __future__ import annotations

import importlib


class CppError(RuntimeError):
    """Base error for bundled C++ kernel failures."""


class CppUnavailable(CppError):
    """Raised when the compiled C++ extension cannot be imported."""


class CppUnsupported(CppError):
    """Raised when a model combination is not implemented in C++."""


_CPP_STATUS_NAMES = {
    0: "ok",
    1: "null_pointer",
    2: "invalid_size",
    3: "invalid_family",
    4: "invalid_rotation",
    5: "invalid_transform",
    6: "invalid_parameter",
    7: "numerical_failure",
}

_MODULE = None
_MODULE_ERROR = None


def cpp_status_name(status: int) -> str:
    """Return the shared name for native kernel status codes."""
    return _CPP_STATUS_NAMES.get(int(status), "unknown")


def load():
    global _MODULE, _MODULE_ERROR
    if _MODULE is not None:
        return _MODULE
    if _MODULE_ERROR is not None:
        raise CppUnavailable(str(_MODULE_ERROR)) from _MODULE_ERROR

    try:
        _MODULE = importlib.import_module("pyscarcopula._scar_cpp")
    except ImportError as exc:
        _MODULE_ERROR = exc
        raise CppUnavailable(
            "pyscarcopula native extension 'pyscarcopula._scar_cpp' is "
            "unavailable. Official wheels include it; source installs require "
            "a C++17 compiler. Reinstall pyscarcopula or rebuild with "
            "'python setup.py build_ext --inplace'. "
            f"Original import error: {exc}"
        ) from exc
    return _MODULE


def available() -> bool:
    try:
        load()
    except CppUnavailable:
        return False
    return True
