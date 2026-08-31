"""Base class for multivariate copulas."""

from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
import threading
from typing import Callable, ParamSpec, TypeVar

import numpy as np

from pyscarcopula.copula.base import CopulaBase


_P = ParamSpec("_P")
_R = TypeVar("_R")


def as_real_array(data):
    """Normalize real numeric observations without accepting lossy coercions."""
    raw = np.asarray(data)
    if np.iscomplexobj(raw):
        raise ValueError("data must be real-valued")
    if raw.dtype.kind in {"O", "S", "U", "V", "b"}:
        raise TypeError("data must have a real numeric dtype")
    return np.asarray(raw, dtype=np.float64)


def factor_uniqueness(instance):
    """Shared read-only factor uniqueness property implementation."""
    if instance._factor_correlation is None:
        return None
    return instance._factor_correlation.uniqueness.copy()


def fitted_ou_state_distribution(instance, u, K=None, grid_range=None):
    """Shared OU posterior query for scalar dynamic multivariate models."""
    from pyscarcopula.strategy._base import get_ou_strategy_for_result
    overrides = {key: value for key, value in (
        ("K", K), ("grid_range", grid_range)) if value is not None}
    strategy = get_ou_strategy_for_result(instance.fit_result, **overrides)
    if u is None:
        raise ValueError(
            "u is required for the distribution at the last observation")
    state = strategy.predictive_state(
        instance, u, instance.fit_result, horizon="current")
    return state.z_grid, state.prob


def factor_copula_getstate(instance):
    """Drop derived factor caches from a multivariate copula pickle."""
    state = MultivariateCopula.__getstate__(instance)
    state["_factor_correlation"] = None
    state["_factor_operator"] = None
    return state


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

    @contextmanager
    def _fit_transaction(self):
        """Restore fitted state if orchestration or an initializer raises.

        Model fitting replaces correlation arrays and prepared caches rather
        than mutating published buffers, so retaining their owners is enough
        to roll back without duplicating a training matrix or native cache.
        """
        with self._state_lock:
            previous = self.__dict__.copy()
            try:
                yield
            except BaseException:
                self.__dict__.clear()
                self.__dict__.update(previous)
                raise

    def _prepare_dynamic_fit(self, u):
        """Prepare model-specific state for a new dynamic fit."""

    def _finalize_dynamic_fit(self, result):
        """Attach model-specific metadata before publishing a fit."""
        return result

    def _fitted_log_likelihood(self, u, *, n_threads=1):
        from pyscarcopula.api import log_likelihood
        from pyscarcopula._types import NumericalConfig

        if self.fit_result is None:
            raise ValueError("Fit first or supply an explicit parameter r")
        return log_likelihood(
            self, u, self.fit_result,
            config=NumericalConfig(n_threads=n_threads))

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
