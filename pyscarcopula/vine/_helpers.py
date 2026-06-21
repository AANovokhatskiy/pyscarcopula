"""Shared utility functions for vine copulas."""

from pyscarcopula._constants import PSEUDO_OBS_EPS
from pyscarcopula._utils import clip_pseudo_observations


def _clip_unit(x):
    """Clip vine pseudo-observations before h-function evaluation."""
    return clip_pseudo_observations(x)


def _open_unit_uniform(rng, size):
    """Draw vine pseudo-observations inside the shared safe unit interval."""
    return rng.uniform(PSEUDO_OBS_EPS, 1.0 - PSEUDO_OBS_EPS, size=size)
