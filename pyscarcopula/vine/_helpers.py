"""Shared utility functions for vine copulas."""

import numpy as np


def _clip_unit(x):
    """Clip to (eps, 1-eps) to avoid NaN in h-functions."""
    eps = 1e-10
    return np.clip(x, eps, 1.0 - eps)
