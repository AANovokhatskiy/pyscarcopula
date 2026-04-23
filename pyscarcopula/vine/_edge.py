"""
vine._edge — VineEdge dataclass and edge-level operations.

Shared by CVineCopula and RVineCopula. All edge-level logic lives here:
  - VineEdge: stores one fitted bivariate copula
  - _edge_h: compute h-function dispatching by method (MLE/GAS/SCAR)
  - _edge_log_likelihood: compute log-likelihood dispatching by method
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class VineEdge:
    """One edge in the vine: a fitted bivariate copula."""
    tree: int             # tree level (0-indexed)
    idx: int              # edge index within tree
    copula: object = None # fitted BivariateCopula instance
    fit_result: object = None

    @property
    def method(self):
        if self.fit_result is None:
            return None
        return self.fit_result.method

    def get_r(self, u_pair, T=None):
        """
        Get copula parameter r for given data.
        MLE: constant (uses fit_result.copula_param).
        GAS: deterministic score-driven path.
        SCAR: smoothed_params via TM forward pass.
        """
        from pyscarcopula._types import IndependentResult

        method = self.method
        if method is None:
            raise ValueError("Edge not fitted")

        if isinstance(self.fit_result, IndependentResult):
            n = T if T is not None else len(u_pair)
            return np.zeros(n)

        from pyscarcopula.strategy._base import get_strategy_for_result
        strategy = get_strategy_for_result(self.fit_result)
        return strategy.smoothed_params(self.copula, u_pair, self.fit_result)

    def get_r_predict(self, n):
        """
        Get copula parameter r for prediction (sampling from x_T).
        MLE: constant (uses fit_result.copula_param).
        GAS: one-step-ahead value cached in fit_result.r_last.
        SCAR: sample from stationary OU.
        """
        from pyscarcopula._types import IndependentResult

        if self.fit_result is None:
            raise ValueError("Edge not fitted")
        if isinstance(self.fit_result, IndependentResult):
            return np.zeros(n)

        from pyscarcopula.strategy._base import get_strategy_for_result
        strategy = get_strategy_for_result(self.fit_result)
        return strategy.predictive_params(self.copula, None, self.fit_result, n)


def _get_alpha(fit_result):
    """Extract (theta, mu, nu) from fit_result."""
    return fit_result.params.values


def _get_gas_params(fit_result):
    """Extract (omega, alpha, beta) from fit_result."""
    return fit_result.params.values


def _edge_h(edge, u2, u1, u_pair, K=300, grid_range=5.0,
            state_cache=None, current_cache_key=None, next_cache_key=None):
    """
    Compute h(u2 | u1; r) for a vine edge using the correct method.

    MLE:  h(u2, u1; theta_mle) — constant parameter.
    GAS:  h(u2, u1; Psi(f_t)) — along deterministic GAS path.
    SCAR: E[h(u2, u1; Psi(x)) | data] — mixture over predictive
          distribution via transfer matrix forward pass.
    Independent: h(u2 | u1) = u2 — trivial pass-through.
    """
    from pyscarcopula.copula.independent import IndependentCopula
    if isinstance(edge.copula, IndependentCopula):
        return u2.copy()

    method = edge.method.upper() if edge.method else 'MLE'

    if method == 'MLE':
        r = edge.get_r(u_pair)
        return edge.copula.h(u2, u1, r)

    elif method == 'GAS':
        from pyscarcopula.numerical.gas_filter import gas_mixture_h
        alpha = _get_gas_params(edge.fit_result)
        scaling = getattr(edge.fit_result, 'scaling', 'unit')
        return gas_mixture_h(alpha[0], alpha[1], alpha[2],
                              u_pair, edge.copula, scaling)

    else:
        # SCAR-TM-OU: mixture h via transfer matrix
        from pyscarcopula.numerical.tm_functions import (
            tm_forward_mixture_h as _tm_forward_mixture_h,
        )
        alpha = _get_alpha(edge.fit_result)
        theta, mu, nu = alpha
        return _tm_forward_mixture_h(theta, mu, nu, u_pair,
                                      edge.copula, K, grid_range,
                                      state_cache=state_cache,
                                      current_cache_key=current_cache_key,
                                      next_cache_key=next_cache_key)


def _edge_log_likelihood(edge, u_pair, K=300, grid_range=5.0):
    """
    Compute log-likelihood for one edge using the correct method.

    MLE:  sum log c(u1, u2; theta_mle)
    GAS:  sum log c(u1, u2; Psi(f_t))  (score-driven filter)
    SCAR: log integral (transfer matrix likelihood)
    """
    from pyscarcopula.copula.independent import IndependentCopula
    if isinstance(edge.copula, IndependentCopula):
        return 0.0

    method = edge.method.lower() if edge.method else 'mle'
    cop = edge.copula

    if method == 'mle':
        r = edge.fit_result.copula_param
        u1 = u_pair[:, 0]
        u2 = u_pair[:, 1]
        return np.sum(cop.log_pdf(u1, u2, np.full(len(u1), float(r))))

    elif method == 'gas':
        from pyscarcopula.numerical.gas_filter import gas_negloglik
        alpha = _get_gas_params(edge.fit_result)
        scaling = getattr(edge.fit_result, 'scaling', 'unit')
        return -gas_negloglik(alpha[0], alpha[1], alpha[2],
                              u_pair, cop, scaling)

    else:
        from pyscarcopula.numerical.tm_functions import tm_loglik
        alpha = _get_alpha(edge.fit_result)
        theta, mu, nu = alpha
        return -tm_loglik(theta, mu, nu, u_pair, cop, K, grid_range)
