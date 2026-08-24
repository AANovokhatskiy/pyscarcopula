"""The sole production loader for the current raw extension path."""

from __future__ import annotations

import importlib

from pyscarcopula._native.errors import NativeUnavailable


_MODULE = None
_MODULE_ERROR = None


def load():
    """Load and cache the bundled extension.

    Stage 8.1 deliberately loads the existing top-level binary.  The binary
    moves below this package in Stage 8.7 without changing facade callers.
    """
    global _MODULE, _MODULE_ERROR
    if _MODULE is not None:
        return _MODULE
    if _MODULE_ERROR is not None:
        raise NativeUnavailable(str(_MODULE_ERROR)) from _MODULE_ERROR

    try:
        _MODULE = importlib.import_module("pyscarcopula._scar_cpp")
    except ImportError as exc:
        _MODULE_ERROR = exc
        raise NativeUnavailable(
            "pyscarcopula native extension 'pyscarcopula._scar_cpp' is "
            "unavailable. Official wheels include it; source installs require "
            "a C++17 compiler. Reinstall pyscarcopula or rebuild with "
            "'python setup.py build_ext --inplace'. "
            f"Original import error: {exc}"
        ) from exc
    return _MODULE


def available() -> bool:
    """Return whether the transitional extension path can be loaded."""
    try:
        load()
    except NativeUnavailable:
        return False
    return True
