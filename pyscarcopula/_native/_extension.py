"""The sole production loader for the bundled raw extension."""

from __future__ import annotations

import importlib

from pyscarcopula._native.errors import NativeUnavailable


_MODULE = None
_MODULE_ERROR = None


def load():
    """Load and cache the bundled extension.

    Production callers use the stable ``pyscarcopula._native`` facade rather
    than importing this implementation module directly.
    """
    global _MODULE, _MODULE_ERROR
    if _MODULE is not None:
        return _MODULE
    if _MODULE_ERROR is not None:
        raise NativeUnavailable(str(_MODULE_ERROR)) from _MODULE_ERROR

    try:
        _MODULE = importlib.import_module("pyscarcopula._native._scar_cpp")
    except ImportError as exc:
        _MODULE_ERROR = exc
        raise NativeUnavailable(
            "pyscarcopula native extension "
            "'pyscarcopula._native._scar_cpp' is "
            "unavailable. Official wheels include it; source installs require "
            "a C++17 compiler. Reinstall pyscarcopula or rebuild with "
            "'python setup.py build_ext --inplace'. "
            f"Original import error: {exc}"
        ) from exc
    return _MODULE
