"""
pyscarcopula.vine — vine copula models.

Submodules:
    _edge       — VineEdge dataclass and edge-level operations
    _selection  — copula family selection (itau + refinement)
    _helpers    — shared utilities (r-generation, clipping)
    _structure  — R-vine matrix representation, Dissmann tree selection
    cvine       — CVineCopula
    rvine       — RVineCopula
"""

from pyscarcopula.vine.cvine import CVineCopula
from pyscarcopula.vine.rvine import RVineCopula
from pyscarcopula.vine._edge import VineEdge, _edge_h, _edge_log_likelihood
from pyscarcopula.vine._selection import select_best_copula
from pyscarcopula.vine._structure import RVineMatrix

__all__ = [
    'CVineCopula',
    'RVineCopula',
    'VineEdge',
    'RVineMatrix',
    'select_best_copula',
    '_edge_h',
    '_edge_log_likelihood',
]
