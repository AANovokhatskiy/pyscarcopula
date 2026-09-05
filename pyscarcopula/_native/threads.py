"""Single Python owner of the public native-thread validation contract."""

from __future__ import annotations

import operator

import numpy as np


MIN_NATIVE_THREADS = 1
MAX_NATIVE_THREADS = 256


def validate_n_threads(value) -> int:
    """Return an integer thread count in the native supported interval."""
    # NumPy < 2.3 lets bool_ pass operator.index (with a warning).
    # Reject it explicitly so the public contract is version-independent.
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("n_threads must be an integer in [1, 256]")
    try:
        resolved = operator.index(value)
    except TypeError as exc:
        raise ValueError(
            "n_threads must be an integer in [1, 256]") from exc
    resolved = int(resolved)
    if resolved < MIN_NATIVE_THREADS or resolved > MAX_NATIVE_THREADS:
        raise ValueError(
            f"n_threads must be in [1, 256], got {resolved}")
    return resolved


__all__ = ["MAX_NATIVE_THREADS", "MIN_NATIVE_THREADS", "validate_n_threads"]
