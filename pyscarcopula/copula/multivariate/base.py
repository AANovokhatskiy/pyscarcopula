"""Base class for multivariate copulas."""

from __future__ import annotations

import numpy as np

from pyscarcopula.copula.base import CopulaBase, CopulaCapabilities


class MultivariateCopula(CopulaBase):
    """Common contract for copulas that are not vine pair copulas."""

    _capabilities = CopulaCapabilities()

    def __init__(self, dimension: int | None = None, *, name="Copula"):
        super().__init__(name=name)
        self._dimension = self._validate_dimension_value(dimension)

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
    def dimension(self):
        dimension = getattr(self, "_dimension", None)
        if dimension is not None:
            return dimension
        for attr in ("corr", "shape", "_R"):
            matrix = getattr(self, attr, None)
            if isinstance(matrix, np.ndarray) and matrix.ndim == 2:
                return int(matrix.shape[0])
        return None

    @property
    def d(self):
        return self.dimension
