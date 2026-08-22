"""Shared coordinate-transposition helpers for bivariate copulas."""

from __future__ import annotations

from copy import copy


def transposed_bivariate_copula(copula):
    """Return a non-mutating coordinate-transposed copula view."""

    rotation = int(getattr(copula, "rotate", 0))
    if rotation not in (90, 270):
        return copula
    transposed = copy(copula)
    transposed._rotate = 360 - rotation
    return transposed
