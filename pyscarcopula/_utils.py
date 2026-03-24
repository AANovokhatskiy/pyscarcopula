"""
pyscarcopula._utils — shared utility functions.

Single source of truth for:
  - broadcast()   — array shape alignment (was duplicated in base.py and elliptical.py)
  - pobs()        — pseudo-observations via rank transform
  - clip_unit()   — clip to (eps, 1-eps)
"""

import numpy as np
from numba import njit


# ══════════════════════════════════════════════════════════════════
# Broadcasting helper (was duplicated across copula/base.py and
# copula/elliptical.py — now one canonical copy)
# ══════════════════════════════════════════════════════════════════

def broadcast(u1, u2, r):
    """Ensure all inputs are 1D float64 arrays of the same length.

    Scalars and length-1 arrays are broadcast to match the longest input.

    Parameters
    ----------
    u1, u2, r : array_like
        Inputs to align.

    Returns
    -------
    u1a, u2a, ra : ndarray (n,)
    """
    u1a = np.atleast_1d(np.asarray(u1, dtype=np.float64)).ravel()
    u2a = np.atleast_1d(np.asarray(u2, dtype=np.float64)).ravel()
    ra = np.atleast_1d(np.asarray(r, dtype=np.float64)).ravel()
    n = max(len(u1a), len(u2a), len(ra))
    if len(u1a) == 1 and n > 1:
        u1a = np.full(n, u1a[0])
    if len(u2a) == 1 and n > 1:
        u2a = np.full(n, u2a[0])
    if len(ra) == 1 and n > 1:
        ra = np.full(n, ra[0])
    return u1a, u2a, ra


# ══════════════════════════════════════════════════════════════════
# Pseudo-observations
# ══════════════════════════════════════════════════════════════════

@njit(cache=True)
def _rank_col(x):
    """Rank a single column. Returns float64 ranks in [1, n]."""
    n = len(x)
    order = np.argsort(x)
    ranks = np.empty(n, dtype=np.float64)
    for i in range(n):
        ranks[order[i]] = float(i + 1)
    return ranks


@njit(cache=True)
def pobs(data):
    """Pseudo-observations via rank transform.

    u_ij = rank(x_ij) / (n + 1), so u in (0, 1).

    Parameters
    ----------
    data : ndarray (T, d)

    Returns
    -------
    u : ndarray (T, d), values in (0, 1)
    """
    n, d = data.shape
    u = np.empty((n, d), dtype=np.float64)
    for j in range(d):
        u[:, j] = _rank_col(data[:, j]) / (n + 1.0)
    return u


# ══════════════════════════════════════════════════════════════════
# Clipping
# ══════════════════════════════════════════════════════════════════

def clip_unit(x, eps=1e-10):
    """Clip array to (eps, 1-eps). Used for pseudo-obs safety."""
    return np.clip(x, eps, 1.0 - eps)


# ══════════════════════════════════════════════════════════════════
# Linear algebra helper (used in EIS)
# ══════════════════════════════════════════════════════════════════

@njit(cache=True)
def linear_least_squares(A, b, ridge_alpha=0.0, pseudo_inverse=False):
    """Solve Ax = b with optional Tikhonov regularization.

    Parameters
    ----------
    A : (m, n) array
    b : (m,) array
    ridge_alpha : float, regularization strength
    pseudo_inverse : bool, use pinv instead of normal equations

    Returns
    -------
    x : (n,) array
    """
    if pseudo_inverse:
        return np.linalg.pinv(A) @ b
    else:
        I = np.eye(A.shape[1])
        I[0, 0] = 0.0
        return np.linalg.inv(A.T @ A + ridge_alpha * I) @ A.T @ b
