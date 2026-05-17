"""
pyscarcopula.vine — vine copula models.

Submodules:
    _pair_copula — shared PairCopula edge container
    _rvine_edges — shared pair-edge runtime operations
    _selection  — copula family selection (itau + refinement)
    _helpers    — shared utility functions
    _structure  — R-vine matrix representation, Dissmann tree selection
    cvine       — CVineCopula
    rvine       — RVineCopula
"""

from pyscarcopula.vine.cvine import CVineCopula
from pyscarcopula.vine.rvine import RVineCopula
from pyscarcopula.vine._pair_copula import PairCopula
from pyscarcopula.vine._selection import select_best_copula
from pyscarcopula.vine._structure import RVineMatrix

__all__ = [
    'CVineCopula',
    'RVineCopula',
    'PairCopula',
    'RVineMatrix',
    'select_best_copula',
]
