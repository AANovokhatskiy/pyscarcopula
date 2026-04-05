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
        """Check that all variables 0..d-1 appear on the diagonal."""
        d = self._d
        diag = set(self._matrix[i, i] for i in range(d))
        if diag != set(range(d)):
            raise ValueError(
                f"Diagonal must be a permutation of 0..{d-1}, got {sorted(diag)}")

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
    # An edge's two "endpoints" (nodes in the tree graph) are:
    # node_left = {var1} | cond_set
    # node_right = {var2} | cond_set
    ca, da = edge_a  # conditioned frozenset, conditioning frozenset
    cb, db = edge_b

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

    return shared_node, frozenset(new_cond_vars), new_conditioning


def build_rvine_structure(u, truncation_level=None):
    """
    Build R-vine structure using Dissmann's sequential MST algorithm.

    Parameters
    ----------
    u : (T, d) pseudo-observations
    truncation_level : int or None
        If set, trees beyond this level use a trivial (arbitrary) structure.

    Returns
    -------
    RVineMatrix
    trees : list of lists of (var1, var2, cond_set) — the edge list per tree,
            in the order they should be fitted.
    """
    T, d = u.shape

    # ── Tree 1: MST on |tau_ij| ─────────────────────────────
    # Nodes = variable indices 0..d-1
    # Edges = all pairs, weight = |tau|
    nodes = list(range(d))
    edges = []
    weights = []
    for i in range(d):
        for j in range(i + 1, d):
            tau, _ = kendalltau(u[:, i], u[:, j])
            edges.append((i, j))
            weights.append(abs(tau))

    mst = _maximum_spanning_tree(nodes, edges, weights)

    # Build tree structure: list of (var1, var2, cond_set)
    # For tree 0: cond_set = ()
    trees = []
    tree_0 = [(min(a, b), max(a, b), ()) for a, b in mst]
    trees.append(tree_0)

    # Track edges as abstract objects for proximity condition
    # edge_repr[k] = list of (conditioned_frozenset, conditioning_frozenset)
    edge_repr = []
    edge_repr.append([
        (frozenset([v1, v2]), frozenset())
        for v1, v2, _ in tree_0
    ])

    # ── Trees 2..d-1: MST on pseudo-obs with proximity condition ──
    for tree_level in range(1, d - 1):
        prev_edges = edge_repr[tree_level - 1]
        n_prev = len(prev_edges)

        if truncation_level is not None and tree_level >= truncation_level:
            # Build arbitrary valid structure for remaining trees
            remaining = _build_trivial_remaining(
                prev_edges, d - 1 - tree_level)
            if remaining is None:
                break
            trees.append([
                (min(cv), max(cv), tuple(sorted(cs)))
                for cv, cs in remaining
            ])
            edge_repr.append(remaining)
            continue

        # Find all valid pairs (proximity condition)
        candidate_edges = []
        candidate_weights = []
        candidate_repr = []

        for i in range(n_prev):
            for j in range(i + 1, n_prev):
                result = _proximity_condition(prev_edges[i], prev_edges[j])
                if result is None:
                    continue

                shared_node, new_cond_vars, new_conditioning = result

                # Compute |tau| on pseudo-observations for this pair
                # We use a heuristic: |tau| from the original data
                # between the two "new" conditioned variables.
                # (Proper pseudo-obs will be computed during fit.)
                v_list = sorted(new_cond_vars)
                if len(v_list) == 2:
                    tau_val, _ = kendalltau(u[:, v_list[0]], u[:, v_list[1]])
                else:
                    tau_val = 0.0

                candidate_edges.append((i, j))
                candidate_weights.append(abs(tau_val))
                candidate_repr.append(
                    (frozenset(new_cond_vars), new_conditioning))

        if len(candidate_edges) == 0:
            break

        # MST on candidate edges
        # Nodes for this tree = indices into prev_edges
        tree_nodes = list(range(n_prev))
        mst_idx = _maximum_spanning_tree(
            tree_nodes, candidate_edges, candidate_weights)

        # Map MST edges back to edge representations
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

        trees.append(new_tree)
        edge_repr.append(new_repr)

    # Build R-vine matrix from trees
    matrix = _trees_to_matrix(d, trees)
    return RVineMatrix(matrix), trees


def _build_trivial_remaining(prev_edges, n_needed):
    """Build trivial edges satisfying proximity for truncated trees."""
    n_prev = len(prev_edges)
    result = []

    for i in range(n_prev):
        for j in range(i + 1, n_prev):
            prox = _proximity_condition(prev_edges[i], prev_edges[j])
            if prox is not None:
                _, new_cv, new_cs = prox
                result.append((new_cv, new_cs))
                if len(result) == n_needed:
                    return result
    return result if len(result) == n_needed else None


def _trees_to_matrix(d, trees):
    """
    Convert a list of tree edges to an R-vine matrix.

    This is the inverse of the edge extraction: given the tree structure,
    produce the d x d lower-triangular matrix M.

    Uses the natural order construction (Dißmann §3.2):
    columns correspond to variables in the order they appear
    as "leaf" nodes through the vine.
    """
    # For small d, use a direct construction approach.
    # Build an ordering of variables and fill the matrix column by column.

    M = np.zeros((d, d), dtype=int)

    # Collect all edges with their full conditioned/conditioning info
    all_edges = []
    for tree_level, tree in enumerate(trees):
        for v1, v2, cond in tree:
            all_edges.append((tree_level, v1, v2, set(cond)))

    # Determine variable ordering: use the first tree's structure.
    # Start from the node with highest degree in tree 0.
    if len(trees) == 0 or len(trees[0]) == 0:
        # Trivial case
        for i in range(d):
            M[i, i] = i
        return M

    # Build adjacency for tree 0
    adj = {i: [] for i in range(d)}
    for v1, v2, _ in trees[0]:
        adj[v1].append(v2)
        adj[v2].append(v1)

    # DFS ordering from highest-degree node
    start = max(adj, key=lambda x: len(adj[x]))
    visited = set()
    order = []

    def dfs(node):
        visited.add(node)
        order.append(node)
        for nb in sorted(adj[node], key=lambda x: -len(adj[x])):
            if nb not in visited:
                dfs(nb)

    dfs(start)

    # Reverse order gives the column assignment:
    # column 0 gets order[-1], column 1 gets order[-2], etc.
    col_var = list(reversed(order))

    # Fill diagonal
    for i in range(d):
        M[i, i] = col_var[i]

    # Fill below diagonal using the tree edges
    # For each column i, we need to find the edges at successive tree levels
    # that involve variable col_var[i].
    for i in range(d - 1):
        var_i = col_var[i]
        current_set = {var_i}  # grows as we go down rows

        for t in range(len(trees)):
            if i + t + 1 >= d:
                break
            # Find edge at tree level t that involves var_i
            # and whose conditioning set is a subset of current_set - {var_i}
            found = False
            for v1, v2, cond in trees[t]:
                cond_s = set(cond)
                pair = {v1, v2}
                if var_i in pair and cond_s <= (current_set - {var_i}):
                    other = (pair - {var_i}).pop()
                    M[i + t + 1, i] = other
                    current_set.add(other)
                    found = True
                    break

            if not found:
                # Fill with first available variable not yet in column
                used = set(M[k, i] for k in range(i + t + 1))
                for v in range(d):
                    if v not in used and v not in current_set:
                        M[i + t + 1, i] = v
                        current_set.add(v)
                        break

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
