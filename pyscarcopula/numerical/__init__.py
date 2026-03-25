"""
pyscarcopula.numerical — computational kernels for latent process models.

Modules:
  ou_kernels    — Numba kernels for OU path generation
  tm_grid       — TMGrid class (adaptive grid, dense/sparse transfer operator)
  tm_functions  — TM log-likelihood and forward-pass computations
  tm_gradient   — Analytical gradient in xi-coordinates
  mc_samplers   — Monte Carlo p-sampler, m-sampler, EIS
"""

from pyscarcopula.numerical.tm_grid import TMGrid
from pyscarcopula.numerical.tm_functions import (
    tm_loglik,
    tm_forward_smoothed,
    tm_forward_rosenblatt,
    tm_forward_mixture_h,
    tm_xT_distribution,
)
from pyscarcopula.numerical.tm_gradient import tm_loglik_with_grad
from pyscarcopula.numerical.ou_kernels import calculate_dwt

__all__ = [
    'TMGrid',
    'tm_loglik', 'tm_loglik_with_grad',
    'tm_forward_smoothed', 'tm_forward_rosenblatt',
    'tm_forward_mixture_h', 'tm_xT_distribution',
    'calculate_dwt',
]
