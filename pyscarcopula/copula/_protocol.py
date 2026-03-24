"""
pyscarcopula.copula._protocol — copula interface.

A copula is a pure mathematical object: it computes PDF, CDF, h-functions,
samples, and transforms between R and the copula parameter domain.

It does NOT:
  - hold fit results (no self.fit_result)
  - know about estimation methods (no fit(), no mlog_likelihood())
  - have mutable state

The estimation logic lives in `strategy/` modules, which accept a copula
as an argument and return an immutable FitResult.
"""

from __future__ import annotations
from typing import Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class CopulaProtocol(Protocol):
    """Interface that every bivariate copula must implement.

    Subclass contract:
        MUST implement:
            name, rotate, bounds
            pdf_unrotated(u1, u2, r)
            log_pdf_unrotated(u1, u2, r)
            transform(x) — R -> copula parameter domain
            inv_transform(r) — copula parameter domain -> R
            psi(t, r) — inverse generator (for Marshall-Olkin sampling)
            V(n, r) — frailty sample

        RECOMMENDED (enable analytical TM gradient):
            dtransform(x) — d Psi / dx
            dlog_pdf_dr_unrotated(u1, u2, r) — d(log c) / dr

        OPTIONAL (fused numba kernels for TM speed):
            copula_grid_batch(u, x_grid) — evaluate PDF on full grid
    """

    @property
    def name(self) -> str:
        """Human-readable name, e.g. 'Gumbel copula (rot=180)'."""
        ...

    @property
    def rotate(self) -> int:
        """Rotation angle: 0, 90, 180, or 270."""
        ...

    @property
    def bounds(self) -> list[tuple[float, float]]:
        """Parameter bounds for L-BFGS-B (MLE)."""
        ...

    # ── Transform between R and copula parameter domain ──────────

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Psi: R -> copula parameter domain. Vectorized."""
        ...

    def inv_transform(self, r: np.ndarray) -> np.ndarray:
        """Psi^{-1}: copula parameter domain -> R."""
        ...

    def dtransform(self, x: np.ndarray) -> np.ndarray:
        """d Psi(x) / dx. For analytical TM gradient."""
        ...

    # ── PDF / log-PDF ────────────────────────────────────────────

    def pdf_unrotated(self, u1: np.ndarray, u2: np.ndarray,
                      r: np.ndarray) -> np.ndarray:
        """Copula density c(u1, u2; r) without rotation."""
        ...

    def log_pdf_unrotated(self, u1: np.ndarray, u2: np.ndarray,
                          r: np.ndarray) -> np.ndarray:
        """log c(u1, u2; r) without rotation."""
        ...

    def pdf(self, u1: np.ndarray, u2: np.ndarray,
            r: np.ndarray) -> np.ndarray:
        """Copula density with rotation applied."""
        ...

    def log_pdf(self, u1: np.ndarray, u2: np.ndarray,
                r: np.ndarray) -> np.ndarray:
        """log c(u1, u2; r) with rotation applied."""
        ...

    def dlog_pdf_dr_unrotated(self, u1: np.ndarray, u2: np.ndarray,
                               r: np.ndarray) -> np.ndarray:
        """d(log c)/dr. Default: central finite differences."""
        ...

    def dlog_pdf_dr(self, u1: np.ndarray, u2: np.ndarray,
                     r: np.ndarray) -> np.ndarray:
        """d(log c)/dr with rotation applied."""
        ...

    # ── h-functions (conditional distribution) ───────────────────

    def h(self, u: np.ndarray, v: np.ndarray,
          r: np.ndarray) -> np.ndarray:
        """h(u | v; r) = dC(u, v; r) / dv. With rotation."""
        ...

    def h_inverse(self, u: np.ndarray, v: np.ndarray,
                  r: np.ndarray) -> np.ndarray:
        """Inverse of h: find t such that h(t, v; r) = u."""
        ...

    # ── Sampling ─────────────────────────────────────────────────

    def psi(self, t: np.ndarray, r: np.ndarray) -> np.ndarray:
        """Inverse generator (Laplace-Stieltjes transform)."""
        ...

    def V(self, n: int, r: np.ndarray) -> np.ndarray:
        """Sample from F = LS^{-1}(psi). For Marshall-Olkin."""
        ...

    def sample(self, n: int, r, rng=None) -> np.ndarray:
        """Sample (n, 2) pseudo-observations at parameter r."""
        ...

    # ── Grid evaluation (optional, for TM speed) ────────────────

    def pdf_on_grid(self, u_row: np.ndarray,
                    x_grid: np.ndarray) -> np.ndarray:
        """c(u1, u2; Psi(x)) for each x in x_grid.
        u_row: (2,), x_grid: (K,). Returns (K,).
        """
        ...

    def copula_grid_batch(self, u: np.ndarray,
                          x_grid: np.ndarray) -> np.ndarray:
        """Batch version: fi[t, k] = c(u1t, u2t; Psi(x_grid[k])).
        u: (T, 2), x_grid: (K,). Returns (T, K).
        Default: loop over pdf_on_grid. Override with numba for speed.
        """
        ...
