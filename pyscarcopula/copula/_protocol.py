"""Structural contracts for common, pair, and multivariate copulas."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from pyscarcopula.copula.base import CopulaCapabilities


@runtime_checkable
class CommonCopulaProtocol(Protocol):
    @property
    def name(self) -> str:
        ...

    @property
    def dimension(self) -> int | None:
        ...

    @property
    def capabilities(self) -> CopulaCapabilities:
        ...

    def sample(self, n: int, *args, rng=None, **kwargs) -> np.ndarray:
        ...


@runtime_checkable
class BivariateCopulaProtocol(CommonCopulaProtocol, Protocol):
    @property
    def rotate(self) -> int:
        ...

    @property
    def bounds(self) -> list[tuple[float, float]]:
        ...

    def transform(self, x: np.ndarray) -> np.ndarray:
        ...

    def inv_transform(self, r: np.ndarray) -> np.ndarray:
        ...

    def dtransform(self, x: np.ndarray) -> np.ndarray:
        ...

    def tau_to_param(self, tau: np.ndarray) -> np.ndarray:
        ...

    def param_to_tau(self, r: np.ndarray) -> np.ndarray:
        ...

    def pdf(self, u1: np.ndarray, u2: np.ndarray,
            r: np.ndarray) -> np.ndarray:
        ...

    def log_pdf(self, u1: np.ndarray, u2: np.ndarray,
                r: np.ndarray) -> np.ndarray:
        ...

    def h(self, u: np.ndarray, v: np.ndarray,
          r: np.ndarray) -> np.ndarray:
        ...

    def h_pair(self, u: np.ndarray, v: np.ndarray,
               r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ...

    def h_inverse(self, u: np.ndarray, v: np.ndarray,
                  r: np.ndarray) -> np.ndarray:
        ...

    def sample_at_parameter(
            self, n: int, r, rng=None) -> np.ndarray:
        ...


@runtime_checkable
class MultivariateCopulaProtocol(CommonCopulaProtocol, Protocol):
    def log_pdf_rows(self, u: np.ndarray, parameter,
                     **kwargs) -> np.ndarray:
        ...

    def log_likelihood(self, u: np.ndarray, parameter=None) -> float:
        ...


# Backward-compatible typing name. New code should choose the dimensional
# protocol explicitly.
CopulaProtocol = CommonCopulaProtocol

__all__ = (
    "CommonCopulaProtocol",
    "BivariateCopulaProtocol",
    "MultivariateCopulaProtocol",
    "CopulaProtocol",
)
