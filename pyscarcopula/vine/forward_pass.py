"""
pyscarcopula.vine.forward_pass — unified C-vine tree traversal.

This module eliminates the code duplication where the same nested loop
pattern (for j in trees, for i in edges, clip, column_stack, h-function)
was repeated in CVineCopula.fit(), log_likelihood(), sample(), and
vine_rosenblatt_transform().

The core is vine_forward_iter() — a generator that yields VineStep
objects for each edge in order, while maintaining the pseudo-observation
matrix v[j][i] internally. The caller only needs to process each step
and optionally provide an h-function for pseudo-obs propagation.

Usage patterns:

    # Pattern 1: Iterate over edges (fit, log_likelihood)
    for step in vine_forward_iter(u, edges, h_func):
        edge = edges[step.tree][step.edge_idx]
        total_ll += compute_ll(edge, step.u_pair)

    # Pattern 2: Build Rosenblatt residuals
    e = vine_rosenblatt(u, edges, h_func)

    # Pattern 3: Just get pseudo-obs at each level
    for step in vine_forward_iter(u, edges, h_func):
        pass  # pseudo-obs are maintained internally
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, Generator, Any


EPS = 1e-10


def _clip(x):
    """Clip to (eps, 1-eps)."""
    return np.clip(x, EPS, 1.0 - EPS)


@dataclass
class VineStep:
    """One edge visit during vine traversal.

    Attributes
    ----------
    tree : int — tree level (0-indexed)
    edge_idx : int — edge index within tree
    u1 : (T,) — first pseudo-observation (root node at this tree level)
    u2 : (T,) — second pseudo-observation (this edge's variable)
    u_pair : (T, 2) — column_stack(u1, u2) ready for copula methods
    """
    tree: int
    edge_idx: int
    u1: np.ndarray
    u2: np.ndarray
    u_pair: np.ndarray


def vine_forward_iter(
    u: np.ndarray,
    edges: list[list],
    h_func: Callable,
) -> Generator[VineStep, None, None]:
    """Iterate over all edges of a C-vine, maintaining pseudo-observations.

    This is the single source of truth for C-vine tree traversal.
    It replaces the duplicated nested loops in fit(), log_likelihood(),
    sample(), and vine_rosenblatt_transform().

    At each tree level j:
      1. Yields VineStep for each edge (j, i)
      2. After all edges in tree j are yielded, computes pseudo-obs
         for tree j+1 using h_func

    Parameters
    ----------
    u : (T, d) array
        Input pseudo-observations.
    edges : list of lists of VineEdge
        Fitted vine edges. edges[j][i] is the edge at tree j, position i.
    h_func : callable(edge, u2, u1, u_pair) -> (T,)
        Computes h(u2 | u1; params) for the given edge.
        This is the function that differs across MLE/GAS/SCAR:
          MLE:  h(u2, u1; theta_mle)
          GAS:  h along GAS-filtered path
          SCAR: mixture h via transfer matrix forward pass

    Yields
    ------
    VineStep for each edge, in order (tree 0 first, then tree 1, etc.)
    """
    T, d = u.shape

    # v[j][i] = (T,) pseudo-obs: variable i at tree level j
    v = [[None] * d for _ in range(d)]
    for i in range(d):
        v[0][i] = _clip(u[:, i].copy())

    for j in range(d - 1):
        n_edges = d - j - 1

        # Yield each edge at this tree level
        for i in range(n_edges):
            u1 = _clip(v[j][0])
            u2 = _clip(v[j][i + 1])
            u_pair = np.column_stack((u1, u2))

            yield VineStep(
                tree=j, edge_idx=i,
                u1=u1, u2=u2, u_pair=u_pair,
            )

        # Compute pseudo-obs for next tree level
        if j < d - 2:
            for i in range(n_edges):
                u1 = _clip(v[j][0])
                u2 = _clip(v[j][i + 1])
                u_pair = np.column_stack((u1, u2))

                edge = edges[j][i]
                v[j + 1][i] = _clip(h_func(edge, u2, u1, u_pair))


def vine_rosenblatt(
    u: np.ndarray,
    edges: list[list],
    h_func: Callable,
) -> np.ndarray:
    """Compute Rosenblatt transform for a fitted C-vine.

    e_0 = u_0
    e_{j+1} = h(v[j][1] | v[j][0]; edge_{j,0})

    This is the vine GoF version: we need e[j+1] from the FIRST edge
    of each tree level (not all edges).

    Parameters
    ----------
    u : (T, d)
    edges : fitted vine edges
    h_func : callable(edge, u2, u1, u_pair) -> (T,)

    Returns
    -------
    e : (T, d) — Rosenblatt-transformed, should be iid U[0,1]^d
    """
    T, d = u.shape

    v = [[None] * d for _ in range(d)]
    for i in range(d):
        v[0][i] = _clip(u[:, i].copy())

    e = np.empty((T, d))
    e[:, 0] = v[0][0]

    for j in range(d - 1):
        n_edges = d - j - 1

        # Rosenblatt residual: first edge of tree j
        u1 = _clip(v[j][0])
        u2 = _clip(v[j][1])
        u_pair = np.column_stack((u1, u2))
        edge = edges[j][0]
        e[:, j + 1] = _clip(h_func(edge, u2, u1, u_pair))

        # Propagate pseudo-obs for next tree
        if j < d - 2:
            for i in range(n_edges):
                u1 = _clip(v[j][0])
                u2 = _clip(v[j][i + 1])
                u_pair = np.column_stack((u1, u2))
                edge_i = edges[j][i]
                v[j + 1][i] = _clip(h_func(edge_i, u2, u1, u_pair))

    return _clip(e)
