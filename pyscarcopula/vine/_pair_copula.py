"""Shared pair-copula edge container for vine models."""

from dataclasses import dataclass

import numpy as np

from pyscarcopula.copula.independent import IndependentCopula


@dataclass
class PairCopula:
    """One fitted pair-copula edge.

    This is the shared edge container for R-vines and C-vines. RVine uses
    ``param/log_likelihood/nfev/tau`` for matrix summaries; CVine additionally
    stores ``tree`` and ``idx`` for its level-indexed edge layout.

    Attributes
    ----------
    copula : BivariateCopula instance
        Family + rotation. Parameter is stored separately in ``param``.
    param : float
        MLE copula parameter (0.0 for IndependentCopula).
    log_likelihood : float
        Edge log-likelihood at the fitted parameter.
    nfev : int
        Optimizer function evaluations (0 for closed-form / Independent).
    tau : float
        Empirical Kendall's tau on the pseudo-observations used to fit
        this edge (for diagnostics only).
    fit_result : FitResult
        Strategy result for this edge.
    tree : int or None
        Optional tree level for CVine-style layouts.
    idx : int or None
        Optional edge index within a CVine tree.
    """
    copula: object = None
    param: float | None = None
    log_likelihood: float = 0.0
    nfev: int = 0
    tau: float = 0.0
    fit_result: object = None
    tree: int | None = None
    idx: int | None = None

    @property
    def method(self):
        if self.fit_result is None:
            return None
        return getattr(self.fit_result, 'method', None)

    @property
    def n_params(self) -> int:
        if self.fit_result is not None:
            return int(getattr(self.fit_result, 'n_params', 0))
        return 0 if isinstance(self.copula, IndependentCopula) else 1

    def h(self, u_conditioned, u_given):
        """h(u_conditioned | u_given) using this edge's fit result."""
        from pyscarcopula.vine._rvine_edges import _edge_h
        return _edge_h(self, u_conditioned, u_given)

    def get_r(self, u_pair, T=None):
        """Return the fitted strategy parameter path for observed pairs."""
        if self.fit_result is None:
            raise ValueError("Edge not fitted")
        if isinstance(self.copula, IndependentCopula):
            n = T if T is not None else len(u_pair)
            return np.zeros(n)
        from pyscarcopula.strategy._base import get_strategy_for_result
        strategy = get_strategy_for_result(self.fit_result)
        return strategy.predictive_mean(self.copula, u_pair, self.fit_result)

    def get_r_predict(self, n):
        """Return the fitted strategy parameter path for prediction."""
        if self.fit_result is None:
            raise ValueError("Edge not fitted")
        if isinstance(self.copula, IndependentCopula):
            return np.zeros(n)
        from pyscarcopula.strategy._base import get_strategy_for_result
        strategy = get_strategy_for_result(self.fit_result)
        return strategy.predictive_params(self.copula, None, self.fit_result, n)


__all__ = ['PairCopula']
