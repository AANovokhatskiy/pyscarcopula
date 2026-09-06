"""Numerical compatibility adapters over native model implementations.

SCAR-TM-OU, GAS, and SCAR-TM-Jacobi model, transition, filtering, state,
residual, and sampling math is native-only. Python retains public argument,
fit, result, and NumPy RNG orchestration plus compatibility wrappers.
"""

from pyscarcopula.numerical.tm_functions import (
    tm_loglik,
    tm_forward_predictive_mean,
    tm_forward_rosenblatt,
    tm_forward_mixture_h,
    tm_xT_distribution,
)
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
    jacobi_matrix_forward_mixture_h_pair,
    jacobi_matrix_state_distribution,
    jacobi_loglik,
    jacobi_neg_loglik,
    jacobi_forward_predictive_mean,
    jacobi_forward_mixture_h,
    jacobi_forward_mixture_h_pair,
    jacobi_state_distribution,
)
from pyscarcopula.numerical.jacobi_sampling import (
    sample_jacobi_lamperti_trajectory,
)
from pyscarcopula.numerical.jacobi_sparse import (
    SparseJacobiTransition,
    compare_sparse_jacobi_corrections,
    jacobi_sparse_fixed_grid_transition,
    jacobi_sparse_matrix_forward_mixture_h,
    jacobi_sparse_matrix_forward_mixture_h_pair,
    jacobi_sparse_matrix_forward_predictive_mean,
    jacobi_sparse_matrix_loglik,
    jacobi_sparse_matrix_neg_loglik_with_grad,
    jacobi_sparse_matrix_state_distribution,
    jacobi_sparse_local_transition,
    sample_sparse_jacobi_trajectory,
    select_sparse_jacobi_order,
    sparse_jacobi_full_horizon_diagnostics,
)
from pyscarcopula.numerical._scar_ou_config import (
    AutoTMConfig,
    select_auto_backend,
)
from pyscarcopula.numerical.predictive_tm import sample_grid_distribution
from pyscarcopula.numerical.gas_filter import (
    gas_filter, gas_loglik, gas_predict_param, gas_negloglik, gas_rosenblatt,
    gas_mixture_h,
    gas_mixture_h_pair,
)

__all__ = [
    'tm_loglik',
    'hermite_loglik', 'hermite_loglik_with_grad', 'hermite_neg_loglik',
    'jacobi_rule', 'jacobi_transition_matrix',
    'jacobi_spectral_transition_matrix',
    'jacobi_local_transition_matrix',
    'jacobi_fixed_grid_transition_matrix',
    'jacobi_matrix_loglik', 'jacobi_matrix_neg_loglik',
    'jacobi_matrix_neg_loglik_with_grad',
    'jacobi_matrix_forward_predictive_mean',
    'jacobi_matrix_forward_mixture_h',
    'jacobi_matrix_forward_mixture_h_pair',
    'jacobi_matrix_state_distribution',
    'jacobi_loglik', 'jacobi_neg_loglik',
    'jacobi_forward_predictive_mean', 'jacobi_forward_mixture_h',
    'jacobi_forward_mixture_h_pair',
    'jacobi_state_distribution',
    'sample_jacobi_lamperti_trajectory',
    'SparseJacobiTransition',
    'compare_sparse_jacobi_corrections',
    'jacobi_sparse_fixed_grid_transition',
    'jacobi_sparse_matrix_forward_mixture_h',
    'jacobi_sparse_matrix_forward_mixture_h_pair',
    'jacobi_sparse_matrix_forward_predictive_mean',
    'jacobi_sparse_matrix_loglik',
    'jacobi_sparse_matrix_neg_loglik_with_grad',
    'jacobi_sparse_matrix_state_distribution',
    'jacobi_sparse_local_transition',
    'sample_sparse_jacobi_trajectory',
    'select_sparse_jacobi_order',
    'sparse_jacobi_full_horizon_diagnostics',
    'AutoTMConfig', 'select_auto_backend',
    'tm_forward_predictive_mean',
    'tm_forward_rosenblatt', 'tm_forward_mixture_h',
    'tm_xT_distribution',
    'sample_grid_distribution',
    'gas_filter', 'gas_loglik', 'gas_predict_param', 'gas_negloglik',
    'gas_rosenblatt',
    'gas_mixture_h',
    'gas_mixture_h_pair',
]
