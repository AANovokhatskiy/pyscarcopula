"""
pyscarcopula.numerical — computational kernels for latent process models.

Modules:
  ou_kernels    — Numba kernels for OU path generation
  tm_grid       — TMGrid class (adaptive grid, dense/sparse transfer operator)
  tm_functions  — TM log-likelihood and forward-pass computations
  tm_gradient   — Analytical gradient in xi-coordinates
  mc_samplers   — Monte Carlo p-sampler, m-sampler, EIS
  gas_filter    — GAS filter, Rosenblatt, h-functions
"""

from pyscarcopula.numerical.tm_grid import TMGrid
from pyscarcopula.numerical.tm_functions import (
    tm_loglik,
    tm_forward_predictive_mean,
    tm_forward_rosenblatt,
    tm_forward_mixture_h,
    tm_xT_distribution,
)
from pyscarcopula.numerical.tm_gradient import tm_loglik_with_grad
from pyscarcopula.numerical.hermite_tm import (
    hermite_loglik,
    hermite_loglik_with_grad,
    hermite_neg_loglik,
)
from pyscarcopula.numerical.jacobi_tm import (
    jacobi_rule,
    jacobi_transition_matrix,
    jacobi_spectral_transition_matrix,
    jacobi_local_transition_matrix,
    jacobi_fixed_grid_transition_matrix,
    jacobi_matrix_loglik,
    jacobi_matrix_neg_loglik,
    jacobi_matrix_neg_loglik_with_grad,
    jacobi_matrix_forward_predictive_mean,
    jacobi_matrix_forward_mixture_h,
    jacobi_matrix_state_distribution,
    jacobi_loglik,
    jacobi_neg_loglik,
    jacobi_forward_predictive_mean,
    jacobi_forward_mixture_h,
    jacobi_state_distribution,
)
from pyscarcopula.numerical.auto_tm import (
    AutoTMConfig,
    select_auto_backend,
    auto_loglik,
    auto_loglik_with_info,
    auto_neg_loglik,
    auto_neg_loglik_info,
    auto_neg_loglik_with_grad,
    auto_neg_loglik_with_grad_info,
)
from pyscarcopula.numerical.ou_kernels import calculate_dwt
from pyscarcopula.numerical.predictive_tm import sample_grid_distribution
from pyscarcopula.numerical.gas_filter import (
    gas_filter, gas_predict_param, gas_negloglik, gas_rosenblatt,
    gas_mixture_h,
)

__all__ = [
    'TMGrid',
    'tm_loglik', 'tm_loglik_with_grad',
    'hermite_loglik', 'hermite_loglik_with_grad', 'hermite_neg_loglik',
    'jacobi_rule', 'jacobi_transition_matrix',
    'jacobi_spectral_transition_matrix',
    'jacobi_local_transition_matrix',
    'jacobi_fixed_grid_transition_matrix',
    'jacobi_matrix_loglik', 'jacobi_matrix_neg_loglik',
    'jacobi_matrix_neg_loglik_with_grad',
    'jacobi_matrix_forward_predictive_mean',
    'jacobi_matrix_forward_mixture_h',
    'jacobi_matrix_state_distribution',
    'jacobi_loglik', 'jacobi_neg_loglik',
    'jacobi_forward_predictive_mean', 'jacobi_forward_mixture_h',
    'jacobi_state_distribution',
    'AutoTMConfig', 'select_auto_backend',
    'auto_loglik', 'auto_loglik_with_info',
    'auto_neg_loglik', 'auto_neg_loglik_info',
    'auto_neg_loglik_with_grad', 'auto_neg_loglik_with_grad_info',
    'tm_forward_predictive_mean',
    'tm_forward_rosenblatt', 'tm_forward_mixture_h',
    'tm_xT_distribution',
    'calculate_dwt', 'sample_grid_distribution',
    'gas_filter', 'gas_predict_param', 'gas_negloglik', 'gas_rosenblatt',
    'gas_mixture_h',
]
