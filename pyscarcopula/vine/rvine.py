"""
R-vine copula model.

Unlike C-vine (fixed star structure), R-vine uses a data-driven tree
structure selected via Dissmann's sequential MST algorithm.

Each edge stores a fitted BivariateCopula (with rotation).
Pseudo-observations are tracked as (variable, conditioning_set) pairs
through the vine tree structure.

Usage:
    from pyscarcopula.vine.rvine import RVineCopula

    vine = RVineCopula()
    vine.fit(u, method='mle', to_pobs=True)
    vine.summary()

    vine.log_likelihood(u, to_pobs=True)
    samples = vine.sample(10000)
    predictions = vine.predict(10000)
"""

import numpy as np
from scipy.optimize import OptimizeResult

from pyscarcopula._utils import pobs
from pyscarcopula.vine._edge import (
    VineEdge, _edge_h, _edge_log_likelihood, _get_alpha, _get_gas_params,
)
from pyscarcopula.vine._selection import (
    select_best_copula, _default_candidates,
)
from pyscarcopula.vine._helpers import (
    _clip_unit, generate_r_for_sample, generate_r_for_predict,
)
from pyscarcopula.vine._structure import (
    RVineMatrix, build_rvine_structure,
    _build_tree_0, _build_next_tree, _trees_to_matrix,
)


# ══════════════════════════════════════════════════════════════
# Internal: build edge lookup for sampling
# ══════════════════════════════════════════════════════════════

def _build_edge_index(trees, edges_dict):
    """
    Build a lookup: (var1, var2, cond_set) -> (tree_level, edge_idx, edge).

    Also build reverse lookup by conditioned pair for fast access:
        pair_lookup[(frozenset({v1,v2}), cond_set)] -> edge, v1, v2
    """
    pair_lookup = {}
    for tree_level, tree_edges in enumerate(trees):
        for edge_idx, (v1, v2, cond) in enumerate(tree_edges):
            cond_set = frozenset(cond)
            edge = edges_dict[(tree_level, edge_idx)]
            pair_lookup[(frozenset({v1, v2}), cond_set)] = (edge, v1, v2)
    return pair_lookup


def _build_sampling_order(trees, d):
    """
    Build the variable ordering and per-variable h-inverse chain
    for Rosenblatt inverse sampling.

    For each variable var (except the first), determine the chain of
    edges to apply h_inverse through, from the deepest conditioning
    down to the unconditional level.

    Returns
    -------
    var_order : list of int — variable ordering (first has no conditioning)
    chains : dict var -> list of (edge_key, partner_var, cond_set)
        where partner_var is the variable to condition on,
        and the list is ordered from deepest (most conditions) to shallowest.
    """
    # Determine variable ordering from tree structure:
    # Start with variables that appear most in tree 0 (the "hub" variables)
    if not trees or not trees[0]:
        return list(range(d)), {}

    # Build adjacency for tree 0
    adj = {i: set() for i in range(d)}
    for v1, v2, _ in trees[0]:
        adj[v1].add(v2)
        adj[v2].add(v1)

    # BFS from highest-degree node
    start = max(adj, key=lambda x: len(adj[x]))
    visited = []
    seen = set()
    queue = [start]
    seen.add(start)
    while queue:
        node = queue.pop(0)
        visited.append(node)
        for nb in sorted(adj[node], key=lambda x: -len(adj[x])):
            if nb not in seen:
                seen.add(nb)
                queue.append(nb)

    # Add any isolated variables
    for v in range(d):
        if v not in seen:
            visited.append(v)

    return visited


class RVineCopula:
    """
    R-vine copula for d dimensions.

    Decomposes d-dimensional dependence into d(d-1)/2 bivariate copulas
    arranged in a data-driven tree structure. The structure is selected
    via Dissmann's algorithm (MST on |Kendall's tau|) and can be
    optionally truncated.

    Parameters
    ----------
    candidates : list of copula classes, or None (default: 5 families)
    allow_rotations : bool (default True)
    criterion : 'aic', 'bic', or 'loglik'
    structure : RVineMatrix or None
        If provided, use this structure instead of Dissmann selection.
    """

    def __init__(self, candidates=None, allow_rotations=True,
                 criterion='aic', structure=None):
        self.candidates = candidates
        self.allow_rotations = allow_rotations
        self.criterion = criterion

        self._structure = structure
        self.trees = None
        self.edges = None       # dict: (tree, edge_idx) -> VineEdge
        self.d = None
        self.method = None

    def _get_candidates(self):
        if self.candidates is not None:
            return self.candidates
        return _default_candidates()

    # ── Fit ────────────────────────────────────────────────────────

    def fit(self, data, method='mle', to_pobs=False,
            K=300, grid_range=5.0,
            truncation_level=None, min_edge_logL=None,
            transform_type='xtanh',
            **kwargs):
        """Fit the R-vine copula."""
        u = np.asarray(data, dtype=np.float64)
        if to_pobs:
            u = pobs(u)

        T, d = u.shape
        self.d = d
        self.method = method.upper()

        pseudo_obs = {}
        for i in range(d):
            pseudo_obs[(i, frozenset())] = u[:, i].copy()

        self.edges = {}

        if self._structure is not None:
            # Pre-defined structure: extract all trees, fit sequentially
            vine_matrix = self._structure
            self.trees = []
            for t in range(d - 1):
                self.trees.append(vine_matrix.edges_at_tree(t))

            for tree_level, tree_edges in enumerate(self.trees):
                self._fit_tree(tree_level, tree_edges, pseudo_obs,
                               d, K, grid_range, truncation_level,
                               min_edge_logL, transform_type,
                               method, kwargs)
        else:
            # Incremental Dissmann: build each tree using tau on
            # h-transformed pseudo-obs from fitted previous trees.
            tree_0, edge_repr_0 = _build_tree_0(u)
            self.trees = [tree_0]
            edge_repr = [edge_repr_0]

            # Fit tree 0
            self._fit_tree(0, tree_0, pseudo_obs,
                           d, K, grid_range, truncation_level,
                           min_edge_logL, transform_type,
                           method, kwargs)

            # Build and fit subsequent trees
            for tree_level in range(1, d - 1):
                new_tree, new_repr = _build_next_tree(
                    tree_level, edge_repr[tree_level - 1],
                    pseudo_obs, truncation_level=truncation_level)
                if new_tree is None:
                    break
                self.trees.append(new_tree)
                edge_repr.append(new_repr)

                self._fit_tree(tree_level, new_tree, pseudo_obs,
                               d, K, grid_range, truncation_level,
                               min_edge_logL, transform_type,
                               method, kwargs)

            # Build R-vine matrix from the final trees
            matrix = _trees_to_matrix(d, self.trees)
            self._structure = RVineMatrix(matrix)

        # Aggregate
        total_ll = sum(e.fit_result.log_likelihood
                       for e in self.edges.values())
        total_nfev = sum(getattr(e.fit_result, 'nfev', 0)
                         for e in self.edges.values())

        self.fit_result = OptimizeResult()
        self.fit_result.log_likelihood = total_ll
        self.fit_result.method = method
        self.fit_result.name = f"R-vine ({d}d, {len(self.edges)} edges)"
        self.fit_result.nfev = total_nfev
        self.fit_result.success = True
        self._last_u = u
        self._pair_lookup = _build_edge_index(self.trees, self.edges)

        return self

    def _fit_tree(self, tree_level, tree_edges, pseudo_obs,
                  d, K, grid_range, truncation_level,
                  min_edge_logL, transform_type, method, kwargs):
        """Fit all edges at one tree level and propagate pseudo-obs."""
        from pyscarcopula.copula.independent import IndependentCopula

        for edge_idx, (v1, v2, cond) in enumerate(tree_edges):
            cond_set = frozenset(cond)

            u1 = _clip_unit(pseudo_obs[(v1, cond_set)])
            u2 = _clip_unit(pseudo_obs[(v2, cond_set)])
            u_pair = np.column_stack((u1, u2))

            edge = VineEdge(tree=tree_level, idx=edge_idx)

            cop, result = select_best_copula(
                u1, u2, self._get_candidates(),
                self.allow_rotations, self.criterion,
                transform_type=transform_type)

            skip_dynamic = (
                self.method == 'MLE'
                or isinstance(cop, IndependentCopula)
                or (truncation_level is not None
                    and tree_level >= truncation_level)
                or (min_edge_logL is not None
                    and result.log_likelihood < min_edge_logL)
            )

            if not skip_dynamic:
                from pyscarcopula.api import fit as _api_fit
                scar_kwargs = {kk: vv for kk, vv in kwargs.items()
                               if kk != 'alpha0'}
                result = _api_fit(cop, u_pair, method=method,
                                  alpha0=kwargs.get('alpha0'),
                                  **scar_kwargs)

            edge.copula = cop
            edge.fit_result = result
            self.edges[(tree_level, edge_idx)] = edge

            # Propagate pseudo-obs for next trees
            if tree_level < d - 2:
                h_2given1 = _clip_unit(
                    _edge_h(edge, u2, u1, u_pair, K, grid_range))
                pseudo_obs[(v2, cond_set | {v1})] = h_2given1

                u_pair_rev = np.column_stack((u2, u1))
                h_1given2 = _clip_unit(
                    _edge_h(edge, u1, u2, u_pair_rev, K, grid_range))
                pseudo_obs[(v1, cond_set | {v2})] = h_1given2

    # ── Log-likelihood ────────────────────────────────────────────

    def log_likelihood(self, data, to_pobs=False, K=300, grid_range=5.0):
        """Compute total log-likelihood of the R-vine."""
        if self.edges is None:
            raise ValueError("Fit first")

        u = np.asarray(data, dtype=np.float64)
        if to_pobs:
            u = pobs(u)

        d = self.d
        pseudo_obs = {}
        for i in range(d):
            pseudo_obs[(i, frozenset())] = u[:, i].copy()

        total_ll = 0.0

        for tree_level, tree_edges in enumerate(self.trees):
            for edge_idx, (v1, v2, cond) in enumerate(tree_edges):
                cond_set = frozenset(cond)

                u1 = _clip_unit(pseudo_obs[(v1, cond_set)])
                u2 = _clip_unit(pseudo_obs[(v2, cond_set)])
                u_pair = np.column_stack((u1, u2))

                edge = self.edges[(tree_level, edge_idx)]
                total_ll += _edge_log_likelihood(edge, u_pair, K, grid_range)

                if tree_level < d - 2:
                    h_2given1 = _clip_unit(
                        _edge_h(edge, u2, u1, u_pair, K, grid_range))
                    pseudo_obs[(v2, cond_set | {v1})] = h_2given1

                    u_pair_rev = np.column_stack((u2, u1))
                    h_1given2 = _clip_unit(
                        _edge_h(edge, u1, u2, u_pair_rev, K, grid_range))
                    pseudo_obs[(v1, cond_set | {v2})] = h_1given2

        return total_ll

    # ── Sampling ─────────────────────────────────────────────────

    def sample(self, n, K=300, grid_range=5.0):
        """
        Sample from fitted R-vine via forward simulation.

        For each sample: generate d independent U[0,1], then transform
        through the vine tree structure using h_inverse to introduce
        the dependence.

        Strategy: simulate forward through the trees. At each tree level,
        for each edge (v1, v2 | D), we can compute the conditional
        samples using h and h_inverse.
        """
        if self.edges is None:
            raise ValueError("Fit first")

        d = self.d
        rng = np.random.default_rng()

        # Generate r for each edge
        r_all = {}
        for key, edge in self.edges.items():
            r_all[key] = generate_r_for_sample(edge, n, rng)

        return self._sample_with_r(n, r_all, rng)

    def _sample_with_r(self, n, r_all, rng):
        """Core sampling logic used by both sample() and predict()."""
        d = self.d
        w = rng.uniform(0, 1, (n, d))

        # We use the variable ordering from the vine structure.
        var_order = _build_sampling_order(self.trees, d)

        # pseudo_obs_samp[(var, cond_set)] = (n,) array of sampled values
        pseudo_obs_samp = {}

        # First variable: no conditioning
        first_var = var_order[0]
        pseudo_obs_samp[(first_var, frozenset())] = w[:, 0]

        # For each subsequent variable in the ordering
        for idx in range(1, d):
            var = var_order[idx]
            # Start with independent uniform
            val = w[:, idx]

            # Find all edges that involve this variable, ordered by tree
            # level (deepest first for h_inverse application).
            # We need to build the chain: for variable `var`, find the
            # sequence of edges from deepest tree down to tree 0 that
            # "condition" var on previously sampled variables.
            chain = self._build_hinverse_chain(var, var_order[:idx])

            # Apply h_inverse from deepest to shallowest
            for edge_key, partner, cond_set in reversed(chain):
                edge = self.edges[edge_key]
                r = r_all[edge_key]
                # partner pseudo-obs at the right conditioning level
                partner_val = pseudo_obs_samp.get((partner, cond_set))
                if partner_val is None:
                    continue
                val = _clip_unit(edge.copula.h_inverse(val, partner_val, r))

            # val is now the sample for this variable
            pseudo_obs_samp[(var, frozenset())] = val

            # Compute h-transforms for this variable needed by later variables
            for edge_key, partner, cond_set in chain:
                edge = self.edges[edge_key]
                r = r_all[edge_key]
                var_at_cond = pseudo_obs_samp.get((var, cond_set))
                partner_at_cond = pseudo_obs_samp.get((partner, cond_set))
                if var_at_cond is None or partner_at_cond is None:
                    continue
                # h(var | partner; D) — var conditioned on partner
                h_val = _clip_unit(
                    edge.copula.h(var_at_cond, partner_at_cond, r))
                pseudo_obs_samp[(var, cond_set | {partner})] = h_val

                # h(partner | var; D) — partner conditioned on var
                h_val_rev = _clip_unit(
                    edge.copula.h(partner_at_cond, var_at_cond, r))
                pseudo_obs_samp[(partner, cond_set | {var})] = h_val_rev

        # Extract results
        x = np.zeros((n, d))
        for var in range(d):
            x[:, var] = pseudo_obs_samp[(var, frozenset())]

        return x

    def _build_hinverse_chain(self, var, already_sampled):
        """
        Build the chain of h_inverse operations needed to sample `var`.

        Returns list of (edge_key, partner_var, cond_set), ordered from
        shallowest (tree 0) to deepest. The caller reverses this for
        h_inverse application.
        """
        chain = []
        already = set(already_sampled)

        for tree_level, tree_edges in enumerate(self.trees):
            for edge_idx, (v1, v2, cond) in enumerate(tree_edges):
                cond_set = frozenset(cond)
                # Does this edge involve our variable?
                if var == v1 and v2 in already and cond_set <= already:
                    chain.append(((tree_level, edge_idx), v2, cond_set))
                elif var == v2 and v1 in already and cond_set <= already:
                    chain.append(((tree_level, edge_idx), v1, cond_set))

        return chain

    # ── Prediction ───────────────────────────────────────────────

    def predict(self, n, u=None, K=300, grid_range=5.0):
        """Conditional predict: sample for next-step prediction."""
        if self.edges is None:
            raise ValueError("Fit first")

        u_data = u if u is not None else getattr(self, '_last_u', None)
        d = self.d
        rng = np.random.default_rng()

        # Build training pseudo-obs if needed
        train_pseudo = None
        from pyscarcopula._types import LatentResult
        needs_train = any(
            isinstance(e.fit_result, LatentResult)
            for e in self.edges.values())

        if u_data is not None and needs_train:
            train_pseudo = self._compute_pseudo_obs(u_data, K, grid_range)

        # Generate r for each edge
        r_all = {}
        for key, edge in self.edges.items():
            tree_level, edge_idx = key
            v1, v2, cond = self.trees[tree_level][edge_idx]
            cond_set = frozenset(cond)

            v_pair = None
            if train_pseudo is not None:
                u1_key = (v1, cond_set)
                u2_key = (v2, cond_set)
                if u1_key in train_pseudo and u2_key in train_pseudo:
                    v_pair = np.column_stack((
                        _clip_unit(train_pseudo[u1_key]),
                        _clip_unit(train_pseudo[u2_key])))

            r_all[key] = generate_r_for_predict(
                edge, n, v_pair, K, grid_range)

        return self._sample_with_r(n, r_all, rng)

    def _compute_pseudo_obs(self, u, K=300, grid_range=5.0):
        """Compute pseudo-observations for all edges from data."""
        d = self.d
        pseudo_obs = {}
        for i in range(d):
            pseudo_obs[(i, frozenset())] = u[:, i].copy()

        for tree_level, tree_edges in enumerate(self.trees):
            for edge_idx, (v1, v2, cond) in enumerate(tree_edges):
                cond_set = frozenset(cond)
                edge = self.edges[(tree_level, edge_idx)]

                u1 = _clip_unit(pseudo_obs[(v1, cond_set)])
                u2 = _clip_unit(pseudo_obs[(v2, cond_set)])
                u_pair = np.column_stack((u1, u2))

                if tree_level < d - 2:
                    h_2given1 = _clip_unit(
                        _edge_h(edge, u2, u1, u_pair, K, grid_range))
                    pseudo_obs[(v2, cond_set | {v1})] = h_2given1

                    u_pair_rev = np.column_stack((u2, u1))
                    h_1given2 = _clip_unit(
                        _edge_h(edge, u1, u2, u_pair_rev, K, grid_range))
                    pseudo_obs[(v1, cond_set | {v2})] = h_1given2

        return pseudo_obs

    # ── Summary ──────────────────────────────────────────────────

    def summary(self):
        """Print vine structure summary."""
        if self.edges is None:
            print("Not fitted")
            return

        print(f"R-Vine Copula (d={self.d}, method={self.method})")
        print("=" * 60)
        total_ll = 0.0
        for tree_level, tree_edges in enumerate(self.trees):
            print(f"\nTree {tree_level}:")
            for edge_idx, (v1, v2, cond) in enumerate(tree_edges):
                edge = self.edges[(tree_level, edge_idx)]
                cop = edge.copula
                name = cop.name
                rot = getattr(cop, '_rotate', 0)

                if edge.method.upper() == 'MLE':
                    param = f"r={edge.fit_result.copula_param:.4f}"
                else:
                    alpha = _get_alpha(edge.fit_result)
                    param = f"alpha={alpha}"

                ll = edge.fit_result.log_likelihood
                total_ll += ll
                rot_str = f" rot={rot}" if rot != 0 else ""
                cond_str = f"|{','.join(map(str, cond))}" if cond else ""
                print(f"  ({v1},{v2}{cond_str}): "
                      f"{name}{rot_str}, {param}, logL={ll:.2f}")

        print(f"\nTotal logL (sum of edges): {total_ll:.2f}")

    def sample_model(self, n, u=None, rng=None):
        """Alias for sample."""
        return self.sample(n)
