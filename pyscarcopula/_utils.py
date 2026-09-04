"""
pyscarcopula._utils — shared utility functions.

Single source of truth for:
  - broadcast()   — array shape alignment (was duplicated in base.py and elliptical.py)
  - pobs()        — pseudo-observations via rank transform
  - clip_unit()   — clip to (eps, 1-eps)
"""

import numpy as np

from pyscarcopula._native import validation as native_validation
from pyscarcopula._constants import (
    H_FUNCTION_EPS,
    PSEUDO_OBS_EPS,
    ROSENBLATT_OUTPUT_EPS,
)


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
    # Imported lazily because numerical.__init__ imports utilities used by
    # native modules while this module itself is still being initialized.
    from pyscarcopula.numerical._arrays import as_float64_array

    u1a = np.atleast_1d(as_float64_array(u1, name="u1")).ravel()
    u2a = np.atleast_1d(as_float64_array(u2, name="u2")).ravel()
    ra = np.atleast_1d(as_float64_array(r, name="r")).ravel()
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

def pobs(data):
    """Pseudo-observations via rank transform.

    u_ij = rank(x_ij) / (n + 1), so u in (0, 1).
    Ranks are ordinal: ties receive successive ranks in input row order.
    NaNs sort after all other values. Computation is performed in C++.

    Parameters
    ----------
    data : ndarray (T, d)

    Returns
    -------
    u : ndarray (T, d), values in (0, 1)
    """
    return native_validation.pobs(data)


# ══════════════════════════════════════════════════════════════════
# Clipping
# ══════════════════════════════════════════════════════════════════

def clip_unit(x, eps=PSEUDO_OBS_EPS):
    """Clip array to (eps, 1-eps). Used for pseudo-obs safety."""
    return native_validation.clip_open_unit(x, eps)


def clip_pseudo_observations(x):
    """Clip pseudo-observations before an inverse Gaussian/Student CDF."""
    return native_validation.clip_open_unit(x, PSEUDO_OBS_EPS)


def clip_pseudo_observations_no_copy(x):
    """Return a float64 pseudo-observation array, copying only if clipped."""
    if type(x) is np.ndarray and x.dtype == np.float64:
        values = x
    else:
        values = np.asarray(x, dtype=np.float64)
    if not native_validation.open_unit_clip_required(
            values, PSEUDO_OBS_EPS):
        return values
    return native_validation.clip_open_unit(values, PSEUDO_OBS_EPS)


def clip_h_function_values(x):
    """Clip h/inverse-h values to the native numerical safety interval."""
    return native_validation.clip_open_unit(x, H_FUNCTION_EPS)


def clip_rosenblatt_output(x):
    """Clip final Rosenblatt values before GoF normal quantiles."""
    return native_validation.clip_open_unit(x, ROSENBLATT_OUTPUT_EPS)
