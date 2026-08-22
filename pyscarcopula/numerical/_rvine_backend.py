"""Private backend selection for R-vine runtime validation.

The selector is intentionally controlled only through a project-prefixed
test environment variable. Public sampling and GoF signatures do not expose
a backend argument; tests use it for explicit native/Python parity checks.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TypeVar

from pyscarcopula.numerical import _cpp_extension
from pyscarcopula.numerical._cpp_extension import CppUnsupported


_RVINE_BACKEND_ENV = "PYSCARCOPULA_TEST_RVINE_BACKEND"
_RVINE_BACKENDS = frozenset({"auto", "native_strict", "python_executor"})

_T = TypeVar("_T")


def rvine_backend_mode() -> str:
    """Return the private R-vine backend mode requested by the test process."""
    value = os.environ.get(_RVINE_BACKEND_ENV, "auto").strip().lower()
    if value not in _RVINE_BACKENDS:
        expected = ", ".join(sorted(_RVINE_BACKENDS))
        raise ValueError(
            f"{_RVINE_BACKEND_ENV} must be one of {expected}, got {value!r}"
        )
    return value


def native_rvine_symbol_available(symbol: str) -> bool:
    """Return whether the mandatory extension exposes a runtime symbol.

    Loading is deliberate even for the Python traversal oracle: built-in pair
    operations still require the base extension, and a completely missing
    ``_scar_cpp`` must remain distinguishable from a missing entry point.
    """
    return hasattr(_cpp_extension.load(), symbol)


def dispatch_rvine_backend(
        *,
        capability: str,
        native_symbol: str,
        python_executor: Callable[[], _T],
        native_executor: Callable[[object], _T | None] | None = None,
) -> _T:
    """Dispatch one internal operation without hiding native runtime errors.

    ``None`` from a native adapter means that a pre-call capability gate
    rejected the model combination. Only ``auto`` may fall back in that case.
    Once the callback is entered, every exception propagates; in particular a
    native status translated to ``CppUnsupported`` is not silently retried.
    """
    mode = rvine_backend_mode()
    module = _cpp_extension.load()

    if mode == "python_executor":
        return python_executor()

    if not hasattr(module, native_symbol) or native_executor is None:
        if mode == "native_strict":
            raise CppUnsupported(
                f"native R-vine capability {capability!r} requires "
                f"_scar_cpp.{native_symbol}"
            )
        return python_executor()

    result = native_executor(module)
    if result is None:
        if mode == "native_strict":
            raise CppUnsupported(
                f"native R-vine capability {capability!r} does not support "
                "this model configuration"
            )
        return python_executor()
    return result


__all__: list[str] = []
