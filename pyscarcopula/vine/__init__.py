"""
pyscarcopula.vine — vine copula models.

Submodules:
    _pair_copula — shared PairCopula edge container
    _rvine_edges — shared pair-edge runtime operations
    _selection  — copula family selection (itau + refinement)
    _helpers    — shared utility functions
    _structure  — R-vine matrix representation, Dissmann tree selection
    vine        — generic VineCopula runtime
    rvine       — compatibility name for VineCopula
"""

from pyscarcopula.vine.vine import VineCopula
from pyscarcopula.vine.rvine import RVineCopula
from pyscarcopula.vine._pair_copula import PairCopula
from pyscarcopula.vine._selection import SelectedCopula, select_best_copula
from pyscarcopula.vine._structure import (
    RVineMatrix,
    cvine_structure,
    dvine_structure,
)

__all__ = [
    'VineCopula',
    'RVineCopula',
    'PairCopula',
    'RVineMatrix',
    'cvine_structure',
    'dvine_structure',
    'SelectedCopula',
    'select_best_copula',
]
