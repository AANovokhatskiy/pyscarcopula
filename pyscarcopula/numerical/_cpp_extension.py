"""Loader and shared errors for the optional pyscarcopula C++ extension."""

from __future__ import annotations

import importlib


class CppError(RuntimeError):
    """Base error for optional C++ kernel failures."""


class CppUnavailable(CppError):
    """Raised when the compiled C++ extension cannot be imported."""


class CppUnsupported(CppError):
    """Raised when a model combination is not implemented in C++."""


_MODULE = None
_MODULE_ERROR = None


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
        raise CppUnavailable(str(exc)) from exc
    return _MODULE


def available() -> bool:
    try:
        load()
    except CppUnavailable:
        return False
    return True
