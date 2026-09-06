"""Raw-draw orchestration for native SCAR-TM-OU trajectory sampling."""

import numpy as np
from pyscarcopula._native import scar_ou
from pyscarcopula.numerical._arrays import validate_integer


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


def sample_ou_trajectory_batches(kappa, mu, nu, n, rng, *, batch_rows):
    """Stream exact OU states without retaining a path-sized draw buffer.

    The native kernel uses the full path length to set dt, even for a final
    short block. With no intervening RNG use the blocks reproduce the
    monolithic normal stream exactly.
    """
    n = validate_integer(n, "n")
    batch_rows = validate_integer(batch_rows, "batch_rows", minimum=1)
    scar_ou.validate_trajectory_parameters(kappa, mu, nu, n)
    previous_state = 0.0
    for start in range(0, n, batch_rows):
        count = min(batch_rows, n - start)
        values = scar_ou.sample_trajectory_block(
            kappa, mu, nu, n, previous_state, start == 0,
            rng.standard_normal(count))
        previous_state = float(values[-1])
        yield values
