"""Base class for multivariate copulas."""

from __future__ import annotations

from functools import wraps
import threading
from typing import Callable, ParamSpec, TypeVar

import numpy as np

from pyscarcopula.copula.base import CopulaBase


_P = ParamSpec("_P")
_R = TypeVar("_R")


def model_state_locked(method: Callable[_P, _R]) -> Callable[_P, _R]:
    """Serialize operations that read or publish mutable fitted state."""
    @wraps(method)
    def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        self = args[0]
        with self._state_lock:
            return method(*args, **kwargs)
    return wrapped


class MultivariateCopula(CopulaBase):
    """Common contract for copulas that are not vine pair copulas."""

    def __init__(
            self, dimension: int | None = None, *, name: str = "Copula") -> None:
        super().__init__(name=name)
        self._state_lock = threading.RLock()
        self._dimension = self._validate_dimension_value(dimension)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_state_lock", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._state_lock = threading.RLock()

    @staticmethod
    def _validate_dimension_value(dimension):
        if dimension is None:
            return None
        if isinstance(dimension, (bool, np.bool_)) or not isinstance(
                dimension, (int, np.integer)):
            raise TypeError(
                f"dimension must be an integer >= 2 or None, got {dimension!r}")
        dimension = int(dimension)
        if dimension < 2:
            raise ValueError(f"dimension must be >= 2, got {dimension}")
        return dimension

    def _set_dimension(self, dimension, *, allow_change=False):
        dimension = self._validate_dimension_value(dimension)
        current = getattr(self, "_dimension", None)
        if current is not None and dimension != current and not allow_change:
            raise ValueError(
                f"{type(self).__name__} dimension is {current}, got {dimension}")
        self._dimension = dimension

    @property
    def dimension(self) -> int | None:
        dimension = getattr(self, "_dimension", None)
        if dimension is not None:
            return dimension
        for attr in ("corr", "shape", "_R"):
            matrix = getattr(self, attr, None)
            if isinstance(matrix, np.ndarray) and matrix.ndim == 2:
                return int(matrix.shape[0])
        return None

    @property
    def d(self) -> int | None:
        return self.dimension
