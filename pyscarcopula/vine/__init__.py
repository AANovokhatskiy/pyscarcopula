"""
pyscarcopula.vine — vine copula models.

Submodules:
    _edge       — VineEdge dataclass and edge-level operations
    _selection  — copula family selection (itau + refinement)
    _helpers    — shared utilities (r-generation, clipping)
    cvine       — CVineCopula
    rvine       — RVineCopula (future)
"""

from pyscarcopula.vine.cvine import CVineCopula
from pyscarcopula.vine._edge import VineEdge, _edge_h, _edge_log_likelihood
from pyscarcopula.vine._selection import select_best_copula

__all__ = [
    'CVineCopula',
    'VineEdge',
    'select_best_copula',
    '_edge_h',
    '_edge_log_likelihood',
]
