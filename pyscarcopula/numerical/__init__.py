"""Numerical adapters and retained Python orchestration.

SCAR-TM-OU and GAS likelihood, gradient, filtering, and forward operations
are native-only. Python retains Jacobi algorithms, SCAR-MC/EIS, sampling,
goodness-of-fit orchestration, and explicit native adapters.
"""

from pyscarcopula.numerical.tm_grid import TMGrid
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
    jacobi_sparse_matrix_forward_mixture_h,
    jacobi_sparse_matrix_forward_mixture_h_pair,
    jacobi_sparse_matrix_forward_predictive_mean,
    jacobi_sparse_matrix_loglik,
    jacobi_sparse_matrix_state_distribution,
    jacobi_sparse_local_transition,
    sample_sparse_jacobi_trajectory,
    sparse_jacobi_full_horizon_diagnostics,
)
from pyscarcopula.numerical._scar_ou_config import (
    AutoTMConfig,
    select_auto_backend,
)
from pyscarcopula.numerical.ou_kernels import calculate_dwt
from pyscarcopula.numerical.predictive_tm import sample_grid_distribution
from pyscarcopula.numerical.gas_filter import (
    gas_filter, gas_loglik, gas_predict_param, gas_negloglik, gas_rosenblatt,
    gas_mixture_h,
    gas_mixture_h_pair,
)

__all__ = [
    'TMGrid',
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
    'jacobi_sparse_matrix_forward_mixture_h',
    'jacobi_sparse_matrix_forward_mixture_h_pair',
    'jacobi_sparse_matrix_forward_predictive_mean',
    'jacobi_sparse_matrix_loglik',
    'jacobi_sparse_matrix_state_distribution',
    'jacobi_sparse_local_transition',
    'sample_sparse_jacobi_trajectory',
    'sparse_jacobi_full_horizon_diagnostics',
    'AutoTMConfig', 'select_auto_backend',
    'tm_forward_predictive_mean',
    'tm_forward_rosenblatt', 'tm_forward_mixture_h',
    'tm_xT_distribution',
    'calculate_dwt', 'sample_grid_distribution',
    'gas_filter', 'gas_loglik', 'gas_predict_param', 'gas_negloglik',
    'gas_rosenblatt',
    'gas_mixture_h',
    'gas_mixture_h_pair',
]
