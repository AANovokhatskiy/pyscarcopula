"""
pyscarcopula.contrib — optional modules for risk metrics and marginal models.

These modules are not part of the core copula API. They provide a
convenience pipeline for rolling VaR/CVaR estimation and marginal
distribution fitting, useful for end-to-end validation and research.

Usage:
    from pyscarcopula.contrib.risk_metrics import risk_metrics
    from pyscarcopula.contrib.marginal import MarginalModel
    from pyscarcopula.contrib.empirical import cvar_emp_window
    from pyscarcopula.contrib.parallel_fit import fit_independent
"""

from pyscarcopula.contrib.parallel_fit import (
    IndependentFit,
    IndependentFitBatch,
    fit_independent,
)

__all__ = [
    "IndependentFit",
    "IndependentFitBatch",
    "fit_independent",
]
