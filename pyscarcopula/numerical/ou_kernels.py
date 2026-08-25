"""Numba-assisted OU trajectory sampling used by SCAR-TM-OU."""

import numpy as np
from numba import njit


_NUMBA_OU_SAMPLE_MIN_SIZE = 10_000

@njit(cache=True)
def ou_sample_trajectory_from_innovations(
        x0, mu, rho, sigma_cond, innovations):
    """Build one exact OU trajectory from pre-generated N(0, 1) draws."""
    n = innovations.size + 1
    x = np.empty(n, dtype=np.float64)
    x[0] = x0
    for t in range(1, n):
        x[t] = (
            mu + rho * (x[t - 1] - mu)
            + sigma_cond * innovations[t - 1]
        )
    return x


def sample_ou_trajectory(kappa, mu, nu, n, rng):
    """Sample one stationary exact-OU path while keeping RNG in NumPy.

    Drawing all innovations before entering the Numba kernel preserves the
    scalar-draw random stream and removes Python overhead from the recurrence.
    """
    if isinstance(n, (bool, np.bool_)) or not isinstance(
            n, (int, np.integer)):
        raise TypeError("n must be a non-negative integer")
    n = int(n)
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return np.empty(0, dtype=np.float64)

    dt = 1.0 / (n - 1) if n > 1 else 1.0
    rho = np.exp(-kappa * dt)
    sigma_cond = np.sqrt(
        nu ** 2 / (2.0 * kappa) * (1.0 - rho ** 2)
    )
    x0 = rng.normal(mu, nu / np.sqrt(2.0 * kappa))
    if n <= _NUMBA_OU_SAMPLE_MIN_SIZE:
        x = np.empty(n, dtype=np.float64)
        x[0] = x0
        for t in range(1, n):
            x[t] = (
                mu + rho * (x[t - 1] - mu)
                + sigma_cond * rng.standard_normal()
            )
        return x

    innovations = rng.standard_normal(max(n - 1, 0))
    return ou_sample_trajectory_from_innovations(
        x0, mu, rho, sigma_cond, innovations)
