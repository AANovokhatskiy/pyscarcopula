"""
vine._structure — R-vine structure representation and tree selection.

R-vine matrix (lower-triangular, Bedford & Cooke convention):
    M[i][i] = diagonal element = variable index at node i in the vine
    M[j][i] for j > i = the conditioning variable added at tree (j - i)
    for the edge involving variable M[i][i].

Dissmann algorithm (2013):
    Greedy tree-by-tree construction:
    1. Tree 1: maximum spanning tree on |tau_ij| for all pairs.
    2. Tree k: compute pseudo-observations via h-functions from tree k-1.
       Build graph with proximity condition, weight = |tau| on pseudo-obs.
       Find MST.

This module is independent of copula fitting — it only determines which
pairs are connected at each tree level.
"""

import warnings

import numpy as np
from scipy.stats import kendalltau


# ══════════════════════════════════════════════════════════════
# R-vine matrix
# ══════════════════════════════════════════════════════════════

class RVineMatrix:
    """
    R-vine structure encoded as a d x d lower-triangular matrix.

    Convention (Dißmann et al. 2013, Joe 2014 Ch.6):
        - M is d x d, 0-indexed.
        - Diagonal M[i,i] = variable label at position i.
        - Below diagonal: M[k,i] for k > i gives the conditioned/conditioning
          structure at tree level (k - i).

    For tree level t (1-indexed, t = 1..d-1):
        Edge at column i (i = 0..d-t-1):
            conditioned pair: (M[i,i], M[i+t, i])
            conditioning set: {M[i+1,i], M[i+2,i], ..., M[i+t-1,i]}

    Parameters
    ----------
    matrix : (d, d) int array
        R-vine matrix in the convention above.
    """

    def __init__(self, matrix):
        matrix = np.asarray(matrix, dtype=int)
        d = matrix.shape[0]
        if matrix.shape != (d, d):
            raise ValueError(f"Matrix must be square, got {matrix.shape}")
        self._matrix = matrix.copy()
        self._d = d
        self._validate()

    @property
    def d(self):
        return self._d

    @property
    def matrix(self):
        return self._matrix.copy()

    def _validate(self):
        """Validate R-vine matrix structure.

        Checks:
        1. Diagonal is a permutation of 0..d-1.
        2. Off-diagonal values are in range 0..d-1.
        3. Each column has no duplicate variable entries.
        4. Proximity condition (Joe 2014, Ch.6) holds.
        """
        d = self._d
        M = self._matrix
        # 1. Diagonal = permutation of 0..d-1
        diag = set(M[i, i] for i in range(d))
        if diag != set(range(d)):
            raise ValueError(
                f"Diagonal must be a permutation of 0..{d-1}, got {sorted(diag)}")
        # 2. Off-diagonal values in valid range
        for i in range(d):
            for k in range(i + 1, d):
                val = M[k, i]
                if val < 0 or val >= d:
                    raise ValueError(
                        f"M[{k},{i}]={val} out of range 0..{d-1}")
        # 3. Column uniqueness: all entries from diagonal down must be distinct
        for i in range(d):
            col_vals = [M[k, i] for k in range(i, d)]
            if len(col_vals) != len(set(col_vals)):
                raise ValueError(
                    f"Column {i} has duplicate variable entries: {col_vals}")
        # 4. Proximity condition
        if not validate_rvine_matrix(M):
            raise ValueError(
                "R-vine matrix fails the proximity condition "
                "(Joe 2014, Ch.6)")

    def n_trees(self):
        return self._d - 1

    def n_edges(self):
        return self._d * (self._d - 1) // 2

    def edge(self, tree, edge_idx):
        """
        Return the edge at (tree, edge_idx).

        tree: 0-indexed tree level (0 = first tree)
        edge_idx: 0-indexed edge within tree

        Returns
        -------
        (var1, var2, cond_set) where:
            var1, var2 : int — conditioned variables
            cond_set : tuple of int — conditioning variables (empty for tree 0)
        """
        t = tree + 1   # 1-indexed tree level in matrix
        i = edge_idx   # column index
        if i < 0 or i >= self._d - t:
            raise IndexError(
                f"Tree {tree} has {self._d - t} edges, got index {edge_idx}")

        M = self._matrix
        var1 = M[i, i]
        var2 = M[i + t, i]
        cond_set = tuple(M[i + s, i] for s in range(1, t))
        return var1, var2, cond_set

    def edges_at_tree(self, tree):
        """All edges at a given tree level."""
        t = tree + 1
        n = self._d - t
        return [self.edge(tree, i) for i in range(n)]

    def __repr__(self):
        return f"RVineMatrix(d={self._d})\n{self._matrix}"


def validate_rvine_matrix(M):
    """
    Check the proximity condition on an R-vine matrix (Joe 2014, Ch.6).

    For every entry M[k, i] with k - i >= 2, the variable M[k, i]
    must appear in column (i+1) at some row j with i+1 <= j <= k:

        M[k, i] ∈ {M[i+1, i+1], M[i+2, i+1], ..., M[k, i+1]}

    This ensures that the edge at tree level (k-i) in column i is
    reachable from an edge at tree level (k-i-1) via the proximity
    condition.

    Parameters
    ----------
    M : ndarray (d, d) or RVineMatrix
        R-vine matrix (0-indexed, lower-triangular).

    Returns
    -------
    bool
        True if the proximity condition holds, False otherwise.
    """
    if isinstance(M, RVineMatrix):
        M = M.matrix
    d = M.shape[0]
    for i in range(d - 2):
        for k in range(i + 2, d):
            target = M[k, i]
            if not any(M[j, i + 1] == target for j in range(i + 1, k + 1)):
                return False
    return True


# ══════════════════════════════════════════════════════════════
# Maximum spanning tree
# ══════════════════════════════════════════════════════════════

def _maximum_spanning_tree(nodes, edges, weights):
    """
    Kruskal's algorithm for maximum spanning tree.

    Parameters
    ----------
    nodes : list of node labels
    edges : list of (node_a, node_b)
    weights : list of float (same length as edges)

    Returns
    -------
    mst_edges : list of (node_a, node_b) in the MST
    """
    # Sort by weight descending
    order = np.argsort(-np.array(weights))

    # Union-Find
    parent = {n: n for n in nodes}
    rank = {n: 0 for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1
        return True

    mst_edges = []
    for idx in order:
        a, b = edges[idx]
        if union(a, b):
            mst_edges.append((a, b))
            if len(mst_edges) == len(nodes) - 1:
                break

    return mst_edges


# ══════════════════════════════════════════════════════════════
# Dissmann algorithm
# ══════════════════════════════════════════════════════════════

def _proximity_condition(edge_a, edge_b):
    """
    Check proximity condition and return the shared node.

    Two edges in tree k share exactly one node (a set of variables).
    If they do, they can be connected in tree k+1.

    edge_a, edge_b : each is (conditioned_set, conditioning_set)
        where conditioned_set is a frozenset of 2 variables,
        and conditioning_set is a frozenset.

    The "node" of an edge is the union of conditioned + conditioning.
    Two edges share a node if they have exactly one "endpoint" in common.

    Returns
    -------
    (shared_node, new_conditioned, new_conditioning) or None
    """
    ca, da = edge_a  # conditioned frozenset, conditioning frozenset
    cb, db = edge_b

    # Strict input validation
    if len(ca) != 2:
        raise ValueError(
            f"_proximity_condition: edge_a conditioned set must have "
            f"exactly 2 variables, got {ca}")
    if len(cb) != 2:
        raise ValueError(
            f"_proximity_condition: edge_b conditioned set must have "
            f"exactly 2 variables, got {cb}")
    if ca & da:
        raise ValueError(
            f"_proximity_condition: edge_a conditioned {ca} and "
            f"conditioning {da} sets overlap")
    if cb & db:
        raise ValueError(
            f"_proximity_condition: edge_b conditioned {cb} and "
            f"conditioning {db} sets overlap")

    # An edge's two "endpoints" (nodes in the tree graph) are:
    # node_left = {var1} | cond_set
    # node_right = {var2} | cond_set
    node_a1 = frozenset([min(ca)]) | da
    node_a2 = frozenset([max(ca)]) | da
    node_b1 = frozenset([min(cb)]) | db
    node_b2 = frozenset([max(cb)]) | db

    nodes_a = {node_a1, node_a2}
    nodes_b = {node_b1, node_b2}

    shared = nodes_a & nodes_b
    if len(shared) != 1:
        return None

    shared_node = shared.pop()
    # The new edge connects the two non-shared nodes
    diff_a = (nodes_a - {shared_node}).pop()
    diff_b = (nodes_b - {shared_node}).pop()

    # New conditioned variables: the "unique" variable from each side
    new_cond_vars = (diff_a | diff_b) - shared_node
    # New conditioning set: the shared node
    new_conditioning = shared_node

    # Validate output invariants
    if len(new_cond_vars) != 2:
        raise ValueError(
            f"_proximity_condition: resulting conditioned set must have "
            f"exactly 2 variables, got {new_cond_vars}")
    if len(new_conditioning) != len(da) + 1:
        raise ValueError(
            f"_proximity_condition: conditioning set should grow by 1, "
            f"got size {len(new_conditioning)} (expected {len(da) + 1})")

    return shared_node, frozenset(new_cond_vars), new_conditioning


def _build_tree_0(u):
    """
    Build the first tree (tree 0) of Dissmann's algorithm.

    Parameters
    ----------
    u : (T, d) pseudo-observations

    Returns
    -------
    tree_0 : list of (var1, var2, cond_set)
    edge_repr_0 : list of (conditioned_frozenset, conditioning_frozenset)
    """
    T, d = u.shape
    nodes = list(range(d))
    edges = []
    weights = []
    for i in range(d):
        for j in range(i + 1, d):
            tau, _ = kendalltau(u[:, i], u[:, j])
            edges.append((i, j))
            weights.append(abs(tau))

    mst = _maximum_spanning_tree(nodes, edges, weights)

    tree_0 = [(min(a, b), max(a, b), ()) for a, b in mst]
    edge_repr_0 = [
        (frozenset([v1, v2]), frozenset())
        for v1, v2, _ in tree_0
    ]
    return tree_0, edge_repr_0


def _build_next_tree(tree_level, prev_edge_repr, pseudo_obs,
                     truncation_level=None):
    """
    Build tree at the given level using pseudo-observations for tau weights.

    Parameters
    ----------
    tree_level : int (>= 1)
    prev_edge_repr : list of (conditioned_frozenset, conditioning_frozenset)
    pseudo_obs : dict mapping (var, frozenset_cond) -> (T,) array
        h-transformed pseudo-observations from fitting previous trees.
    truncation_level : int or None

    Returns
    -------
    new_tree : list of (var1, var2, cond_set)
    new_edge_repr : list of (conditioned_frozenset, conditioning_frozenset)
        Returns (None, None) if no valid edges can be built.
    """
    n_prev = len(prev_edge_repr)

    if truncation_level is not None and tree_level >= truncation_level:
        d_minus_1 = n_prev  # n_prev = d - 1 - (tree_level - 1)
        n_needed = n_prev - 1
        remaining = _complete_structure_above_truncation(
            prev_edge_repr, n_needed)
        if remaining is None:
            return None, None
        new_tree = [
            (min(cv), max(cv), tuple(sorted(cs)))
            for cv, cs in remaining
        ]
        return new_tree, remaining

    candidate_edges = []
    candidate_weights = []
    candidate_repr = []

    for i in range(n_prev):
        for j in range(i + 1, n_prev):
            result = _proximity_condition(
                prev_edge_repr[i], prev_edge_repr[j])
            if result is None:
                continue

            shared_node, new_cond_vars, new_conditioning = result

            # Compute |tau| on h-transformed pseudo-observations
            v_list = sorted(new_cond_vars)
            if len(v_list) != 2:
                warnings.warn(
                    f"_build_next_tree: proximity returned {len(v_list)} "
                    f"conditioned vars (expected 2) at tree {tree_level}, "
                    f"edges ({i}, {j}). Skipping candidate.",
                    stacklevel=2)
                continue

            key_a = (v_list[0], frozenset(new_conditioning))
            key_b = (v_list[1], frozenset(new_conditioning))
            obs_a = pseudo_obs.get(key_a)
            obs_b = pseudo_obs.get(key_b)

            if obs_a is None or obs_b is None:
                warnings.warn(
                    f"_build_next_tree: missing pseudo-observations for "
                    f"edge ({v_list[0]},{v_list[1]}|"
                    f"{set(new_conditioning)}) at tree {tree_level}. "
                    f"Skipping candidate.",
                    stacklevel=2)
                continue

            tau_val, _ = kendalltau(obs_a, obs_b)
            if np.isnan(tau_val):
                tau_val = 0.0
                warnings.warn(
                    f"_build_next_tree: NaN Kendall's tau for "
                    f"({v_list[0]},{v_list[1]}|"
                    f"{set(new_conditioning)}) at tree {tree_level}, "
                    f"using 0.0",
                    stacklevel=2)

            candidate_edges.append((i, j))
            candidate_weights.append(abs(tau_val))
            candidate_repr.append(
                (frozenset(new_cond_vars), new_conditioning))

    if len(candidate_edges) == 0:
        return None, None

    tree_nodes = list(range(n_prev))
    mst_idx = _maximum_spanning_tree(
        tree_nodes, candidate_edges, candidate_weights)

    new_tree = []
    new_repr = []
    edge_lookup = {
        (e[0], e[1]): idx for idx, e in enumerate(candidate_edges)
    }
    edge_lookup.update({
        (e[1], e[0]): idx for idx, e in enumerate(candidate_edges)
    })

    for a, b in mst_idx:
        key = (a, b) if (a, b) in edge_lookup else (b, a)
        idx = edge_lookup[key]
        cv, cs = candidate_repr[idx]
        new_tree.append((
            min(cv), max(cv), tuple(sorted(cs))
        ))
        new_repr.append((cv, cs))

    return new_tree, new_repr



def _complete_structure_above_truncation(prev_edges, n_needed):
    """Complete vine structure above the truncation level.

    Even when the vine is truncated (edges above truncation_level are
    treated as independent or fitted with static MLE only), a valid
    R-vine matrix requires a complete tree structure at every level.
    This function builds the minimal structure satisfying the proximity
    condition, using uniform weights so the MST selection is arbitrary.

    The *copula fitting* policy for these edges is controlled by the
    ``truncation_fill`` parameter in ``RVineCopula.fit()``.
    """
    n_prev = len(prev_edges)
    candidate_edges = []
    candidate_repr = []

    for i in range(n_prev):
        for j in range(i + 1, n_prev):
            prox = _proximity_condition(prev_edges[i], prev_edges[j])
            if prox is not None:
                _, new_cv, new_cs = prox
                candidate_edges.append((i, j))
                candidate_repr.append((new_cv, new_cs))

    if len(candidate_edges) < n_needed:
        return None

    nodes = list(range(n_prev))
    weights = [1.0] * len(candidate_edges)
    mst_idx = _maximum_spanning_tree(nodes, candidate_edges, weights)

    if len(mst_idx) < n_needed:
        return None

    edge_lookup = {}
    for idx, (i, j) in enumerate(candidate_edges):
        edge_lookup[(i, j)] = idx
        edge_lookup[(j, i)] = idx

    result = []
    for a, b in mst_idx:
        key = (a, b) if (a, b) in edge_lookup else (b, a)
        result.append(candidate_repr[edge_lookup[key]])
    return result


def _trees_to_matrix(d, trees, n_strict_levels=None):
    """
    Convert a list of tree edges to an R-vine matrix.

    Uses backtracking search over diagonal orderings and edge assignments
    (Czado 2019, Joe 2014).  For each column the algorithm:

    1. Picks a candidate diagonal variable, prioritising leaf nodes in
       the current tree-0 subgraph (fewest connections among remaining
       nodes).
    2. Fills the column by matching edges from each tree level via their
       full variable set (conditioned ∪ conditioning), tracking which
       edges are already used.
    3. Checks the proximity condition (Joe 2014, Ch.6) between adjacent
       columns.  If it fails, the algorithm backtracks and tries the
       next candidate or edge assignment.

    Parameters
    ----------
    n_strict_levels : int or None
        Number of tree levels that must be matched exactly from the
        input trees.  Levels at or above this threshold may use free
        variable choice when no matching edge exists (useful for
        truncated vines).  Default: all levels are strict.
    """
    M = np.zeros((d, d), dtype=int)

    if len(trees) == 0 or len(trees[0]) == 0:
        for i in range(d):
            M[i, i] = i
        return M

    n_strict = n_strict_levels if n_strict_levels is not None else len(trees)

    # Build adjacency for tree 0 (used for leaf-priority heuristic).
    adj0 = {i: set() for i in range(d)}
    for v1, v2, _ in trees[0]:
        adj0[v1].add(v2)
        adj0[v2].add(v1)

    # Index edges with unique IDs and their full variable sets.
    # full_index[t] = [(edge_id, frozenset_of_all_vars), ...]
    full_index = []
    eid = 0
    for tree in trees:
        level_edges = []
        for v1, v2, cond in tree:
            full_vars = frozenset({v1, v2}) | frozenset(cond)
            level_edges.append((eid, full_vars))
            eid += 1
        full_index.append(level_edges)

    def _column_options(col, leaf, used_edges, remaining_vars):
        """Return all valid column fills as (entries_list, used_edges_set)."""
        results = []

        def _fill(t, current_cond, entries, used):
            row = col + t + 1
            if row >= d:
                results.append((list(entries), set(used)))
                return
            target = frozenset({leaf}) | current_cond

            # For tree levels with known edges, match strictly.
            if t < n_strict:
                for edge_id, full_vars in full_index[t]:
                    if edge_id in used:
                        continue
                    if target < full_vars and len(full_vars) == len(target) + 1:
                        other = next(iter(full_vars - target))
                        _fill(t + 1, current_cond | {other},
                               entries + [other], used | {edge_id})
                return

            # Above strict levels: try matching first, fall back to
            # free variable choice for truncated/incomplete levels.
            if t < len(full_index):
                for edge_id, full_vars in full_index[t]:
                    if edge_id in used:
                        continue
                    if target < full_vars and len(full_vars) == len(target) + 1:
                        other = next(iter(full_vars - target))
                        _fill(t + 1, current_cond | {other},
                               entries + [other], used | {edge_id})
                if results:
                    return  # found at least one match, skip free choice

            # No matching edge — allow any remaining variable.
            used_in_col = frozenset({leaf}) | frozenset(entries)
            for other in sorted(remaining_vars - used_in_col):
                _fill(t + 1, current_cond | {other},
                       entries + [other], used)

        _fill(0, frozenset(), [], used_edges)
        return results

    def _proximity_ok(col):
        """Check proximity of column *col* against column col+1."""
        if col >= d - 2:
            return True
        for k in range(col + 2, d):
            target = M[k, col]
            if not any(M[j, col + 1] == target
                       for j in range(col + 1, k + 1)):
                return False
        return True

    def _solve(col, remaining, used_edges, adj):
        if len(remaining) <= 1:
            if remaining:
                M[col, col] = next(iter(remaining))
            return True

        # Prioritise leaf nodes (degree <= 1 in remaining tree-0 subgraph),
        # then sort by degree ascending (most constrained first).
        candidates = sorted(remaining,
                            key=lambda v: (len(adj[v] & remaining), v))

        for leaf in candidates:
            new_remaining = remaining - {leaf}
            for entries, new_used in _column_options(
                    col, leaf, used_edges, new_remaining):
                M[col, col] = leaf
                for idx, val in enumerate(entries):
                    M[col + idx + 1, col] = val

                # Check proximity of previous column now that this
                # column is filled (needed for backtracking to work).
                if col > 0 and not _proximity_ok(col - 1):
                    M[col, col] = 0
                    for idx in range(len(entries)):
                        M[col + idx + 1, col] = 0
                    continue

                # Update adjacency: remove leaf from tree-0 subgraph.
                new_adj = {k: set(v) for k, v in adj.items()}
                for nb in new_adj.get(leaf, []):
                    new_adj[nb].discard(leaf)
                new_adj.pop(leaf, None)

                if _solve(col + 1, new_remaining, new_used, new_adj):
                    return True

                # Backtrack
                M[col, col] = 0
                for idx in range(len(entries)):
                    M[col + idx + 1, col] = 0

        return False

    adj = {k: set(v) for k, v in adj0.items()}
    if not _solve(0, set(range(d)), set(), adj):
        raise RuntimeError(
            "_trees_to_matrix: backtracking failed to produce a valid "
            "R-vine matrix. The input trees may not form a proper vine.")

    return M


def cvine_structure(d, order=None):
    """
    Construct a C-vine R-vine matrix.

    Parameters
    ----------
    d : int — dimension
    order : list of int or None
        Variable ordering. order[0] is root of tree 0, etc.
        Default: [0, 1, 2, ..., d-1].

    Returns
    -------
    RVineMatrix
    """
    if order is None:
        order = list(range(d))

    M = np.zeros((d, d), dtype=int)
    for i in range(d):
        M[i, i] = order[i]
        for j in range(i):
            M[i, j] = order[i]

    # Correct C-vine matrix: column j has order[j] on diagonal,
    # and below it the remaining variables in order.
    M = np.zeros((d, d), dtype=int)
    for j in range(d):
        M[j, j] = order[j]
        for k in range(j + 1, d):
            M[k, j] = order[k]

    return RVineMatrix(M)
