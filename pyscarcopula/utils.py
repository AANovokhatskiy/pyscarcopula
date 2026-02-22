import numpy as np
from numba import njit, prange


@njit(cache=True)
def _rank_col(x):
    n = len(x)
    order = np.argsort(x)
    ranks = np.empty(n, dtype=np.float64)
    for i in range(n):
        ranks[order[i]] = float(i + 1)
    return ranks


@njit(cache=True)
def pobs(data):
    """Pseudo-observations via rank transform. data: (T, d)."""
    n, d = data.shape
    u = np.empty((n, d), dtype=np.float64)
    for j in range(d):
        u[:, j] = _rank_col(data[:, j]) / (n + 1.0)
    return u


@njit(cache=True)
def linear_least_squares(A, b, ridge_alpha=0.0, pseudo_inverse=False):
    if pseudo_inverse:
        return np.linalg.pinv(A) @ b
    else:
        I = np.eye(A.shape[1])
        I[0, 0] = 0.0
        return np.linalg.inv(A.T @ A + ridge_alpha * I) @ A.T @ b
