"""
vine._rvine_matrix_builder — natural-order R-vine matrix construction.

Given a valid R-vine (a list of trees indexed by level), build the
R-vine matrix in the **natural-order convention** of Czado (2019,
Algorithm 5.4), which is the convention used by ``pyvinecopulib`` and
described in Joe (2014, Ch. 5).

Any regular vine has a natural-order matrix representation, and this
representation is reached by iteratively peeling a leaf of the current
top tree ("peel-to-leaf" algorithm, Czado 2019 Alg. 5.4; Dissmann 2010,
§3.3). The proximity condition then holds by construction — no
backtracking is required.

References
----------
* Bedford, T. and Cooke, R. M. (2001). Probability density decomposition
  for conditionally dependent random variables modeled by vines.
  *Annals of Mathematics and Artificial Intelligence*, 32(1-4):245-268.
* Bedford, T. and Cooke, R. M. (2002). Vines - a new graphical model
  for dependent random variables. *The Annals of Statistics*,
  30(4):1031-1068.
* Dissmann, J. F. (2010). *Statistical inference for regular vines and
  application.* Diploma Thesis, Technische Universitaet Muenchen.
* Dissmann, J., Brechmann, E. C., Czado, C. and Kurowicka, D. (2013).
  Selecting and estimating regular vine copulae and application to
  financial returns. *Computational Statistics & Data Analysis*,
  59:52-69.
* Joe, H. (2014). *Dependence Modeling with Copulas.* CRC Press,
  Chapter 5.
* Czado, C. (2019). *Analyzing Dependent Data with Vine Copulas - A
  Practical Guide with R.* Springer, Chapter 5 (Algorithm 5.4).

Matrix convention (natural order)
---------------------------------
``M`` is a ``d x d`` integer matrix. Non-zero entries occupy the
*upper-left* anti-triangle (rows + columns satisfy ``r + c <= d - 1``);
the lower-right anti-triangle is zero-padded. Column ``c`` carries
``d - c`` non-zero entries at rows ``0..d-1-c``:

    row d-1-c           : the leaf variable peeled at step c
                          (anti-diagonal entry)
    row d-2-c           : tree-0 other endpoint
    row d-3-c           : tree-1 other endpoint
    ...
    row 0               : top-tree other endpoint

For tree level ``t`` (``0 <= t <= d-2-c``) at column ``c``:

    conditioned pair = { M[d-1-c, c], M[d-2-c-t, c] }
    conditioning set = { M[r, c] : r = d-1-c-t, ..., d-2-c }

The last column ``c = d-1`` carries only ``M[0, d-1]`` — the final
variable left over after ``d-1`` peelings.

Input format
------------
``trees`` : list of lists.
    ``trees[t]`` is the edge list at tree level ``t`` (0-indexed).
    Each edge is ``(conditioned, conditioning)`` where

        ``conditioned``  : frozenset of exactly 2 variables
        ``conditioning`` : frozenset of exactly ``t`` variables

    Tree ``t`` must have exactly ``d - 1 - t`` edges.

Algorithm (iterative peel)
--------------------------
for c = 0, 1, ..., d - 2:
    k = d - 2 - c                               # current top tree level
    find the unique unclaimed edge in trees[k]
    let (conditioned_top, conditioning_top) = that edge
    pick v = min(conditioned_top) as the leaf (deterministic choice)
    M[d-1-c, c] = v                             # anti-diagonal leaf
    cond_accum = {}
    for t = 0, 1, ..., k - 1:
        find the unique unclaimed edge e in trees[t] with
            v in e.conditioned and e.conditioning == cond_accum
        let other = e.conditioned - {v}
        M[d-2-c-t, c] = other
        cond_accum = cond_accum | {other}
        claim e
    verify cond_accum == conditioning_top
    M[0, c] = other endpoint of the top edge      # tree-k entry
    claim the top edge

The step ``for t`` walks exactly once through each level 0..k-1, and
the verification at the top ensures the nested chain closes on the
top-tree edge. If the input is malformed a ``RuntimeError`` is raised
with a descriptive message.
"""

import numpy as np


# ──────────────────────────────────────────────────────────────
# Input normalization
# ──────────────────────────────────────────────────────────────


def _normalize_trees(d, trees):
    """Validate and normalize ``trees`` to a list of list of
    ``(frozenset, frozenset)`` tuples.
    """
    if len(trees) != d - 1:
        raise RuntimeError(
            f"build_rvine_matrix: expected {d - 1} trees for d={d}, "
            f"got {len(trees)}"
        )

    normalized = []
    for t, tree in enumerate(trees):
        expected_n_edges = d - 1 - t
        if len(tree) != expected_n_edges:
            raise RuntimeError(
                f"build_rvine_matrix: tree {t} must have "
                f"{expected_n_edges} edges, got {len(tree)}"
            )

        level = []
        for edge in tree:
            if len(edge) != 2:
                raise RuntimeError(
                    f"build_rvine_matrix: tree {t} edge must be "
                    f"(conditioned, conditioning), got {edge!r}"
                )
            conditioned = frozenset(edge[0])
            conditioning = frozenset(edge[1])

            if len(conditioned) != 2:
                raise RuntimeError(
                    f"build_rvine_matrix: tree {t} conditioned set must "
                    f"have exactly 2 variables, got {conditioned}"
                )
            if len(conditioning) != t:
                raise RuntimeError(
                    f"build_rvine_matrix: tree {t} conditioning set must "
                    f"have exactly {t} variables, got {conditioning}"
                )
            if conditioned & conditioning:
                raise RuntimeError(
                    f"build_rvine_matrix: tree {t} conditioned "
                    f"{conditioned} and conditioning {conditioning} "
                    f"overlap"
                )

            level.append((conditioned, conditioning))
        normalized.append(level)

    return normalized


def _all_variables(trees_norm):
    """Collect the variable set from the tree-0 edges."""
    result = set()
    for conditioned, _ in trees_norm[0]:
        result.update(conditioned)
    return result


# ──────────────────────────────────────────────────────────────
# Natural-order proximity check
# ──────────────────────────────────────────────────────────────


def validate_natural_order_matrix(M):
    """Check that a matrix is a valid natural-order R-vine matrix.

    Convention (Czado 2019 Alg. 5.4):
        * Non-zero entries occupy the upper-left anti-triangle
          (``r + c <= d - 1``); the lower-right anti-triangle is zero.
        * The anti-diagonal ``M[d-1-c, c]`` is the leaf peeled at column
          ``c``; these ``d`` values must be a permutation of ``0..d-1``.
        * Each column's non-zero entries are distinct.
        * The decoded tree-by-tree edge lists form a valid regular vine:
          tree ``t`` has exactly ``d-1-t`` distinct edges, and every
          tree-``(t+1)`` edge's conditioned+conditioning set is obtained
          by merging two adjacent tree-``t`` edges sharing a node
          (proximity condition; Joe 2014 Ch. 5).

    Returns
    -------
    bool
        True if the matrix is a valid natural-order R-vine matrix.
    """
    M = np.asarray(M, dtype=int)
    if M.ndim != 2:
        return False
    d = M.shape[0]
    if M.shape != (d, d):
        return False
    if d <= 0:
        return False
    if d == 1:
        return bool(M[0, 0] == 0)

    # (1) Anti-diagonal is a permutation of 0..d-1.
    leaves = [int(M[d - 1 - c, c]) for c in range(d)]
    if set(leaves) != set(range(d)):
        return False

    # (2) Column non-zero entries are distinct and in 0..d-1.
    for c in range(d):
        col_vals = [int(M[r, c]) for r in range(d - c)]
        if len(set(col_vals)) != len(col_vals):
            return False
        if any(v < 0 or v >= d for v in col_vals):
            return False

    # (3) Lower-right anti-triangle is zero-padded.
    for c in range(d):
        for r in range(d - c, d):
            if M[r, c] != 0:
                return False

    # (4) Decoded trees must form a valid regular vine (proximity between
    #     consecutive trees).
    try:
        trees = decode_matrix_to_trees(M)
    except Exception:
        return False

    # Shape of decoded trees.
    if len(trees) != d - 1:
        return False
    for t, level in enumerate(trees):
        if len(level) != d - 1 - t:
            return False
        for conditioned, conditioning in level:
            if len(conditioned) != 2 or len(conditioning) != t:
                return False
            if conditioned & conditioning:
                return False
        # Tree t must have d-1-t distinct edges identified by
        # (conditioned, conditioning) as a pair of sets.
        edge_keys = {(cp, cs) for cp, cs in level}
        if len(edge_keys) != d - 1 - t:
            return False

    # (5) Cross-tree proximity: every tree-(t+1) edge must be obtainable
    #     from two adjacent tree-t edges (sharing a node of size t+1).
    for t in range(d - 2):
        tree_t_nodes = [cp | cs for cp, cs in trees[t]]  # each size t+2
        for cp_next, cs_next in trees[t + 1]:
            full_next = cp_next | cs_next        # size t+3
            # Find a pair of tree-t edges whose "node sets" are subsets
            # of full_next and whose union equals full_next.
            candidates = [i for i, node in enumerate(tree_t_nodes)
                          if node <= full_next]
            found = False
            for i in candidates:
                for j in candidates:
                    if i < j and tree_t_nodes[i] | tree_t_nodes[j] == full_next \
                            and len(tree_t_nodes[i] & tree_t_nodes[j]) == t + 1:
                        found = True
                        break
                if found:
                    break
            if not found:
                return False
    return True


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────


def build_rvine_matrix(d, trees):
    """Build the natural-order R-vine matrix from a valid tree list.

    Parameters
    ----------
    d : int
        Number of variables.
    trees : list of lists
        Tree-by-tree edge lists; see the module docstring.

    Returns
    -------
    M : (d, d) int ndarray
        Natural-order R-vine matrix (Czado 2019 Alg. 5.4 /
        pyvinecopulib convention). The lower-right anti-triangle is
        zero-padded.
    """
    M, _ = build_rvine_matrix_with_edge_map(d, trees)
    return M


def build_rvine_matrix_with_edge_map(d, trees):
    """Build the matrix and the ``(tree, col) -> input_edge_index`` map.

    The edge map is what later stages need to attach fitted pair
    copulas to matrix positions without any search.

    Returns
    -------
    M : (d, d) int ndarray
        Natural-order R-vine matrix.
    edge_map : dict
        Keys ``(tree, col)`` for ``0 <= tree <= d-2`` and
        ``0 <= col <= d-2-tree``; values are indices into
        ``trees[tree]``.
    """
    if not isinstance(d, int) or d < 1:
        raise RuntimeError(f"build_rvine_matrix: d must be int >= 1, got {d!r}")
    if d == 1:
        if len(trees) != 0:
            raise RuntimeError(
                f"build_rvine_matrix: d=1 requires 0 trees, got {len(trees)}"
            )
        return np.array([[0]], dtype=int), {}

    trees_norm = _normalize_trees(d, trees)
    all_vars = _all_variables(trees_norm)
    if all_vars != set(range(d)):
        raise RuntimeError(
            f"build_rvine_matrix: tree 0 must cover variables "
            f"{set(range(d))}, got {all_vars}"
        )

    M = np.zeros((d, d), dtype=int)
    claimed = [set() for _ in range(d - 1)]
    edge_map = {}
    peeled = []

    for c in range(d - 1):
        k = d - 2 - c

        # Locate the unique unclaimed edge in the current top tree.
        top_candidates = [
            idx for idx in range(len(trees_norm[k]))
            if idx not in claimed[k]
        ]
        if len(top_candidates) != 1:
            raise RuntimeError(
                f"build_rvine_matrix: expected exactly one unclaimed edge "
                f"in tree {k} at column {c}, got {len(top_candidates)}"
            )
        idx_top = top_candidates[0]
        conditioned_top, conditioning_top = trees_norm[k][idx_top]

        # Deterministic leaf: pick the smaller endpoint. Both endpoints
        # are valid leaves of the current sub-vine — the choice only
        # affects column order inside the natural-order matrix.
        v = min(conditioned_top)
        other_top = max(conditioned_top)

        # Anti-diagonal placement.
        M[d - 1 - c, c] = v
        peeled.append(v)

        # Walk tree levels 0 .. k-1 matching conditioning == cond_accum.
        cond_accum = set()
        walk_claims = []
        for t in range(k):
            target_cc = frozenset(cond_accum)
            hits = [
                idx
                for idx, (conditioned, conditioning)
                in enumerate(trees_norm[t])
                if (idx not in claimed[t]
                    and v in conditioned
                    and conditioning == target_cc)
            ]
            if len(hits) != 1:
                raise RuntimeError(
                    f"build_rvine_matrix: expected exactly one tree-{t} "
                    f"edge with leaf {v} and conditioning "
                    f"{sorted(target_cc)} at column {c}, got {len(hits)}"
                )
            idx_t = hits[0]
            conditioned_t, _ = trees_norm[t][idx_t]
            other = next(iter(conditioned_t - {v}))
            M[d - 2 - c - t, c] = other
            edge_map[(t, c)] = idx_t
            walk_claims.append((t, idx_t))
            cond_accum.add(other)

        # Consistency check: the accumulated conditioning of the walk
        # must equal the top-edge conditioning (this is the natural-order
        # nesting guarantee). If it fails, the input is not a valid R-vine.
        if cond_accum != conditioning_top:
            raise RuntimeError(
                f"build_rvine_matrix: inconsistent walk at column {c}: "
                f"accumulated conditioning {sorted(cond_accum)} != top "
                f"edge conditioning {sorted(conditioning_top)}"
            )

        # Place the top-tree entry at row 0 and claim all walk edges.
        M[0, c] = other_top
        edge_map[(k, c)] = idx_top
        claimed[k].add(idx_top)
        for cl_t, cl_idx in walk_claims:
            claimed[cl_t].add(cl_idx)

    # Final column: the single unpeeled variable.
    remaining = all_vars - set(peeled)
    if len(remaining) != 1:
        raise RuntimeError(
            f"build_rvine_matrix: expected 1 remaining variable for final "
            f"column, got {len(remaining)}"
        )
    M[0, d - 1] = remaining.pop()

    # Defensive validation (should never fail for a valid R-vine input).
    if not validate_natural_order_matrix(M):
        raise RuntimeError(
            "build_rvine_matrix: produced matrix fails the natural-order "
            "proximity check. This indicates the input trees do not form a "
            "valid regular vine."
        )

    return M, edge_map


# ──────────────────────────────────────────────────────────────
# Decode for roundtrip tests
# ──────────────────────────────────────────────────────────────


def decode_matrix_to_trees(M):
    """Decode a natural-order R-vine matrix back into a tree list.

    Returns
    -------
    trees : list of lists
        ``trees[t]`` contains ``(conditioned_fz, conditioning_fz)`` tuples
        for the edges encoded by column positions at tree level ``t``.
    """
    M = np.asarray(M, dtype=int)
    d = M.shape[0]
    if M.shape != (d, d):
        raise ValueError(f"decode_matrix_to_trees: expected square, got {M.shape}")
    trees = []
    for t in range(d - 1):
        level = []
        for col in range(d - 1 - t):
            conditioned = frozenset({
                int(M[d - 1 - col, col]),
                int(M[d - 2 - col - t, col]),
            })
            conditioning = frozenset(
                int(M[r, col])
                for r in range(d - 1 - col - t, d - 1 - col)
            )
            level.append((conditioned, conditioning))
        trees.append(level)
    return trees
