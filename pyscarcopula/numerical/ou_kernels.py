"""Raw-draw orchestration for native SCAR-TM-OU trajectory sampling."""

import numpy as np
from pyscarcopula._native import scar_ou


def ou_sample_trajectory_from_innovations(
        x0, mu, rho, sigma_cond, innovations):
    """Build one exact OU trajectory from pre-generated N(0, 1) draws."""
    return scar_ou.trajectory_from_innovations(
        x0, mu, rho, sigma_cond, innovations)


def sample_ou_trajectory(kappa, mu, nu, n, rng):
    """Sample one stationary exact-OU path while keeping RNG in NumPy.

    The first raw standard normal initializes the stationary native state.
    Vector draws preserve the scalar NumPy normal stream and its consumption.
    """
    if isinstance(n, (bool, np.bool_)) or not isinstance(
            n, (int, np.integer)):
        raise TypeError("n must be a non-negative integer")
    n = int(n)
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return np.empty(0, dtype=np.float64)

    scar_ou.validate_trajectory_parameters(kappa, mu, nu, n)
    return scar_ou.sample_trajectory(
        kappa, mu, nu, rng.standard_normal(n))
