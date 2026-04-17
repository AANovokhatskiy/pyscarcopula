"""
R-vine copula model.

Unlike C-vine (fixed star structure), R-vine uses a data-driven tree
structure selected via Dissmann's sequential MST algorithm.

Each edge stores a fitted BivariateCopula (with rotation).
Pseudo-observations are tracked as (variable, conditioning_set) pairs
through the vine tree structure.

The arbitrary-conditioning sampler uses a vine computational graph view:
known nodes are propagated through h-functions, graph-compatible variables are
sampled by inverse h-functions, and non-graph-compatible patterns fall back to
posterior grid or exact integration. This implements the conditional-sampling
and conditioning-structure ideas from Cheng et al. (2025), "Vine Copulas as
Differentiable Computational Graphs", arXiv:2506.13318, without the PyTorch /
autograd layer from that paper.

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
from pyscarcopula.vine._conditional_rvine import (
    ensure_rvine_conditional_supported,
    rvine_flexible_graph_plan,
    rvine_conditional_plan,
    sample_rvine_conditional_with_r,
    validate_rvine_given,
)
from pyscarcopula.vine._structure import (
    cvine_structure,
    RVineMatrix,
    _build_tree_0, _build_tree_0_conditional,
    _build_next_tree, _build_next_tree_conditional, _trees_to_matrix,
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


def _build_matrix_edge_map(structure, trees, edges_dict):
    """Map R-vine matrix positions (tree, col) to fitted edge keys.

    For each matrix position (tree_level=t, column=s), the matrix
    encodes an edge with:
        conditioned = {M[s,s], M[s+t+1,s]}
        conditioning = {M[s+1,s], ..., M[s+t,s]}

    This function finds the corresponding (tree_level, edge_idx) key
    in edges_dict by matching conditioned/conditioning sets.

    Returns
    -------
    dict : (tree_level, column) -> (tree_level, edge_idx)
    """
    # Index self.trees edges by (conditioned_set, conditioning_set)
    tree_idx = {}
    for tree_level, tree_edges in enumerate(trees):
        for edge_idx, (v1, v2, cond) in enumerate(tree_edges):
            key = (frozenset({v1, v2}), frozenset(cond))
            tree_idx[(tree_level, key)] = (tree_level, edge_idx)

    # Map matrix positions to edge keys
    edge_map = {}
    d = structure.d
    for t in range(d - 1):
        matrix_edges = structure.edges_at_tree(t)
        for s, (var1, var2, cond) in enumerate(matrix_edges):
            key = (frozenset({var1, var2}), frozenset(cond))
            edge_key = tree_idx.get((t, key))
            if edge_key is not None:
                edge_map[(t, s)] = edge_key

    return edge_map


class RVineCopula:
    """
    R-vine copula for d dimensions.

    Decomposes d-dimensional dependence into d(d-1)/2 bivariate copulas
    arranged in a data-driven tree structure. The structure is selected
    via Dissmann's algorithm (MST on |Kendall's tau|) and can be
    optionally truncated.

    Conditional prediction uses a computational-graph interpretation of the
    vine, following the conditional-sampling and conditioning-variable
    structure ideas of Cheng et al. (2025), arXiv:2506.13318. The
    implementation here is NumPy/SciPy based and is not a differentiable
    PyTorch graph.

    Parameters
    ----------
    candidates : list of copula classes, or None (default: 5 families)
    allow_rotations : bool (default True)
    criterion : 'aic', 'bic', or 'loglik'
    structure : RVineMatrix or None
        If provided, use this structure instead of Dissmann selection.
    structure_mode : {'dissmann', 'conditional'}, default 'dissmann'
        Structure selection rule used when ``structure`` is not provided.
        ``'conditional'`` prioritises edges inside ``conditional_vars``.
    conditional_vars : iterable of int or None
        Variables expected to be fixed in later conditional sampling.
    conditional_structure_policy : {'min_posterior', 'priority'}
        ``'min_posterior'`` falls back to a conditional C-vine when the
        priority structure has a larger posterior dimension for
        ``conditional_vars``. ``'priority'`` keeps any encodable priority
        structure.
    """

    def __init__(self, candidates=None, allow_rotations=True,
                 criterion='aic', structure=None,
                 structure_mode='dissmann', conditional_vars=None,
                 conditional_structure_policy='min_posterior'):
        if structure_mode not in ('dissmann', 'conditional'):
            raise ValueError(
                "structure_mode must be 'dissmann' or 'conditional'")
        if conditional_structure_policy not in ('min_posterior', 'priority'):
            raise ValueError(
                "conditional_structure_policy must be 'min_posterior' "
                "or 'priority'")
        if structure is not None and structure_mode == 'conditional':
            raise ValueError(
                "structure_mode='conditional' cannot be used with an "
                "explicit structure")
        if conditional_vars is not None and structure_mode != 'conditional':
            raise ValueError(
                "conditional_vars requires structure_mode='conditional'")

        self.candidates = candidates
        self.allow_rotations = allow_rotations
        self.criterion = criterion

        self._structure = structure
        self.structure_mode = structure_mode
        self.conditional_structure_policy = conditional_structure_policy
        self.conditional_vars = (
            None if conditional_vars is None
            else frozenset(int(v) for v in conditional_vars)
        )
        self._conditional_structure_applied = False
        self.conditional_structure_status = None
        self.trees = None
        self.edges = None       # dict: (tree, edge_idx) -> VineEdge
        self.d = None
        self.method = None

    def _get_candidates(self):
        if self.candidates is not None:
            return self.candidates
        return _default_candidates()

    def _conditional_vars_for_fit(self, d):
        """Validate conditional_vars once the data dimension is known."""
        if self.structure_mode != 'conditional':
            return None
        if self.conditional_vars is None or len(self.conditional_vars) == 0:
            raise ValueError(
                "conditional_vars must be a non-empty iterable when "
                "structure_mode='conditional'")
        bad = [v for v in self.conditional_vars if v < 0 or v >= d]
        if bad:
            raise ValueError(
                f"conditional_vars must be in [0, {d - 1}], got {bad}")
        if len(self.conditional_vars) >= d:
            raise ValueError(
                "conditional_vars must be a proper subset of variables")
        return self.conditional_vars

    def is_conditioning_optimized_for(self, given):
        """Return True when this vine was structured for the given variables."""
        if (self.structure_mode != 'conditional'
                or self.conditional_vars is None
                or not getattr(self, '_conditional_structure_applied', False)):
            return False
        if isinstance(given, dict):
            given_vars = frozenset(int(v) for v in given)
        else:
            given_vars = frozenset(int(v) for v in given)
        return given_vars <= self.conditional_vars

    def _conditional_cvine_fallback(self, d, conditional_vars):
        """Matrix-encodable fallback with conditioning variables sampled first."""
        cond_order = sorted(int(v) for v in conditional_vars)
        rest_order = [v for v in range(d) if v not in conditional_vars]
        return cvine_structure(d, order=rest_order + cond_order)

    def _cvine_fallback(self, d):
        """Matrix-encodable fallback for non-encodable Dissmann structures."""
        if self.trees and len(self.trees) > 0:
            degree = {v: 0 for v in range(d)}
            for v1, v2, _ in self.trees[0]:
                degree[int(v1)] += 1
                degree[int(v2)] += 1
            order = sorted(range(d), key=lambda v: (-degree[v], v))
        else:
            order = list(range(d))
        return cvine_structure(d, order=order)

    def _posterior_dim_for_structure(self, structure, conditional_vars):
        """Posterior dimension for the intended full conditioning pattern."""
        original = self._structure
        try:
            self._structure = structure
            given = {int(v): 0.5 for v in conditional_vars}
            plan = rvine_conditional_plan(self, given, quad_order=2)
            return int(plan['posterior_dim'])
        finally:
            self._structure = original

    def conditional_plan(self, given, quad_order=10):
        """
        Describe the current conditional sampling workload.

        The plan reports the matrix-based sampling order and the latent
        Rosenblatt variables that must be sampled from a posterior because
        fixed variables appear later in that order.  It is an introspection
        helper; it does not change the fitted model.

        The reported graph/pseudo-observation workload is based on the vine
        computational graph view used for conditional sampling in
        arXiv:2506.13318.
        """
        if self.edges is None:
            raise ValueError("Fit first")
        given = validate_rvine_given(given, self.d)
        return rvine_conditional_plan(self, given, quad_order=quad_order)

    def flexible_graph_plan(self, given):
        """
        Reachability plan for flexible graph conditional sampling.

        This helper reports which pseudo-observation nodes are known from
        `given`, reachable by deterministic h-propagation, or sampleable by
        inverse h-functions. It follows the computational-graph scheduling idea
        from arXiv:2506.13318. The current executor enables this path for
        graph-compatible single-given patterns; other patterns fall back to
        matrix-order grid or exact integration.
        """
        if self.edges is None:
            raise ValueError("Fit first")
        given = validate_rvine_given(given, self.d)
        return rvine_flexible_graph_plan(self, given)

    # ── Fit ────────────────────────────────────────────────────────

    def fit(self, data, method='mle', to_pobs=False,
            K=300, grid_range=5.0,
            truncation_level=None, truncation_fill='mle',
            min_edge_logL=None,
            transform_type='xtanh',
            **kwargs):
        """Fit the R-vine copula.

        Parameters
        ----------
        truncation_fill : {'independent', 'mle'}, default 'independent'
            Policy for edges above ``truncation_level``:
            - ``'independent'``: force IndependentCopula (logL = 0).
            - ``'mle'``: run static MLE copula selection (skip dynamic
              methods only).
        """
        if truncation_fill not in ('independent', 'mle'):
            raise ValueError(
                f"truncation_fill must be 'independent' or 'mle', "
                f"got {truncation_fill!r}")

        u = np.asarray(data, dtype=np.float64)
        if to_pobs:
            u = pobs(u)

        T, d = u.shape
        self.d = d
        self.method = method.upper()
        self.truncation_fill = truncation_fill
        self._predict_cache = {}
        active_structure_mode = (
            getattr(self, '_fit_structure_mode_override', None)
            or self.structure_mode
        )
        if active_structure_mode not in ('dissmann', 'conditional'):
            raise ValueError(
                "structure_mode must be 'dissmann' or 'conditional'")
        self._conditional_structure_applied = (
            active_structure_mode == 'conditional')
        self.conditional_structure_status = (
            'pending' if active_structure_mode == 'conditional' else None)
        conditional_vars = (
            self._conditional_vars_for_fit(d)
            if active_structure_mode == 'conditional'
            else None
        )

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
                               truncation_fill, min_edge_logL,
                               transform_type, method, kwargs)
        else:
            # Incremental Dissmann: build each tree using tau on
            # h-transformed pseudo-obs from fitted previous trees.
            if active_structure_mode == 'conditional':
                tree_0, edge_repr_0 = _build_tree_0_conditional(
                    u, conditional_vars)
            else:
                tree_0, edge_repr_0 = _build_tree_0(u)
            self.trees = [tree_0]
            edge_repr = [edge_repr_0]

            # Fit tree 0
            self._fit_tree(0, tree_0, pseudo_obs,
                           d, K, grid_range, truncation_level,
                           truncation_fill, min_edge_logL,
                           transform_type, method, kwargs)

            # Build and fit subsequent trees
            for tree_level in range(1, d - 1):
                if active_structure_mode == 'conditional':
                    new_tree, new_repr = _build_next_tree_conditional(
                        tree_level, edge_repr[tree_level - 1],
                        pseudo_obs, conditional_vars,
                        truncation_level=truncation_level)
                else:
                    new_tree, new_repr = _build_next_tree(
                        tree_level, edge_repr[tree_level - 1],
                        pseudo_obs, truncation_level=truncation_level)
                if new_tree is None:
                    break
                self.trees.append(new_tree)
                edge_repr.append(new_repr)

                self._fit_tree(tree_level, new_tree, pseudo_obs,
                               d, K, grid_range, truncation_level,
                               truncation_fill, min_edge_logL,
                               transform_type, method, kwargs)

            # Build R-vine matrix from the final trees.
            # When truncated, allow free variable choice above
            # truncation level so the matrix is always encodable.
            n_strict = (truncation_level
                        if truncation_level is not None
                        else None)
            try:
                matrix = _trees_to_matrix(d, self.trees,
                                          n_strict_levels=n_strict)
            except RuntimeError:
                if active_structure_mode != 'conditional':
                    import warnings
                    warnings.warn(
                        "Dissmann R-vine structure could not be encoded as "
                        "an RVineMatrix; falling back to a matrix-encodable "
                        "C-vine structure.",
                        RuntimeWarning,
                        stacklevel=2)
                    self._structure = self._cvine_fallback(d)
                    try:
                        result = self.fit(
                            data, method=method, to_pobs=to_pobs, K=K,
                            grid_range=grid_range,
                            truncation_level=truncation_level,
                            truncation_fill=truncation_fill,
                            min_edge_logL=min_edge_logL,
                            transform_type=transform_type,
                            **kwargs)
                    finally:
                        self._fit_structure_mode_override = None
                    self._conditional_structure_applied = False
                    self.conditional_structure_status = None
                    return result
                import warnings
                warnings.warn(
                    "Conditional R-vine structure could not be encoded as an "
                    "RVineMatrix; falling back to a conditional C-vine "
                    "structure.",
                    RuntimeWarning,
                    stacklevel=2)
                self._structure = self._conditional_cvine_fallback(
                    d, conditional_vars)
                self.conditional_structure_status = 'cvine_fallback'
                try:
                    result = self.fit(
                        data, method=method, to_pobs=to_pobs, K=K,
                        grid_range=grid_range,
                        truncation_level=truncation_level,
                        truncation_fill=truncation_fill,
                        min_edge_logL=min_edge_logL,
                        transform_type=transform_type,
                        **kwargs)
                finally:
                    self._fit_structure_mode_override = None
                self._conditional_structure_applied = True
                self.conditional_structure_status = 'cvine_fallback'
                return result
            self._structure = RVineMatrix(matrix)
            if active_structure_mode == 'conditional':
                self.conditional_structure_status = 'priority'
                if self.conditional_structure_policy == 'min_posterior':
                    priority_structure = self._structure
                    fallback_structure = self._conditional_cvine_fallback(
                        d, conditional_vars)
                    priority_dim = self._posterior_dim_for_structure(
                        priority_structure, conditional_vars)
                    fallback_dim = self._posterior_dim_for_structure(
                        fallback_structure, conditional_vars)
                    if fallback_dim < priority_dim:
                        import warnings
                        warnings.warn(
                            "Conditional priority structure increases the "
                            "matrix-order posterior dimension; falling back "
                            "to a conditional C-vine structure.",
                            RuntimeWarning,
                            stacklevel=2)
                        self._structure = fallback_structure
                        self.conditional_structure_status = 'cvine_fallback'
                        try:
                            result = self.fit(
                                data, method=method, to_pobs=to_pobs, K=K,
                                grid_range=grid_range,
                                truncation_level=truncation_level,
                                truncation_fill=truncation_fill,
                                min_edge_logL=min_edge_logL,
                                transform_type=transform_type,
                                **kwargs)
                        finally:
                            self._fit_structure_mode_override = None
                        self._conditional_structure_applied = True
                        self.conditional_structure_status = 'cvine_fallback'
                        return result

            # Roundtrip check: matrix edges must match self.trees.
            # For truncated levels, the matrix may have rebuilt the
            # structure to ensure encodability, so only check levels
            # below truncation (or all if no truncation).
            check_levels = len(self.trees)
            if truncation_level is not None:
                check_levels = min(check_levels, truncation_level)

            for tl in range(check_levels):
                tree_edges = self.trees[tl]
                matrix_edges = self._structure.edges_at_tree(tl)
                tree_edge_sets = {
                    (frozenset({v1, v2}), frozenset(cond))
                    for v1, v2, cond in tree_edges
                }
                matrix_edge_sets = {
                    (frozenset({v1, v2}), frozenset(cond))
                    for v1, v2, cond in matrix_edges
                }
                if tree_edge_sets != matrix_edge_sets:
                    raise RuntimeError(
                        f"trees->matrix roundtrip failed at tree {tl}: "
                        f"tree edges {tree_edge_sets} != "
                        f"matrix edges {matrix_edge_sets}")

            # For truncated levels, update self.trees and edge mapping
            # to match the matrix (the matrix is authoritative).
            if truncation_level is not None:
                for tl in range(truncation_level, len(self.trees)):
                    old_edges = self.trees[tl]
                    new_edges = self._structure.edges_at_tree(tl)
                    self.trees[tl] = new_edges
                    # Remap edges: take any existing edge at this level
                    # (all are IndependentCopula or identical MLE) and
                    # assign to new positions.
                    template = self.edges.get((tl, 0))
                    for old_idx in range(len(old_edges)):
                        self.edges.pop((tl, old_idx), None)
                    for new_idx in range(len(new_edges)):
                        if template is not None:
                            from copy import copy
                            edge = copy(template)
                            edge.idx = new_idx
                            self.edges[(tl, new_idx)] = edge

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
        self._edge_map = _build_matrix_edge_map(
            self._structure, self.trees, self.edges)

        return self

    def _fit_tree(self, tree_level, tree_edges, pseudo_obs,
                  d, K, grid_range, truncation_level,
                  truncation_fill, min_edge_logL,
                  transform_type, method, kwargs):
        """Fit all edges at one tree level and propagate pseudo-obs."""
        from pyscarcopula.copula.independent import IndependentCopula

        is_truncated = (truncation_level is not None
                        and tree_level >= truncation_level)

        for edge_idx, (v1, v2, cond) in enumerate(tree_edges):
            cond_set = frozenset(cond)

            u1 = _clip_unit(pseudo_obs[(v1, cond_set)])
            u2 = _clip_unit(pseudo_obs[(v2, cond_set)])
            u_pair = np.column_stack((u1, u2))

            edge = VineEdge(tree=tree_level, idx=edge_idx)

            if is_truncated and truncation_fill == 'independent':
                # True truncation: force independence
                cop = IndependentCopula()
                result = OptimizeResult()
                result.log_likelihood = 0.0
                result.copula_param = 0.0
                result.method = 'MLE'
                result.success = True
                skip_dynamic = True
            else:
                cop, result = select_best_copula(
                    u1, u2, self._get_candidates(),
                    self.allow_rotations, self.criterion,
                    transform_type=transform_type)

                skip_dynamic = (
                    self.method == 'MLE'
                    or isinstance(cop, IndependentCopula)
                    or is_truncated  # truncation_fill == 'mle'
                    or (min_edge_logL is not None
                        and result.log_likelihood < min_edge_logL)
                )

            if not skip_dynamic:
                from pyscarcopula.api import fit as _api_fit
                scar_kwargs = {kk: vv for kk, vv in kwargs.items()
                               if kk not in ('alpha0', 'K', 'grid_range')}
                result = _api_fit(cop, u_pair, method=method,
                                  K=K, grid_range=grid_range,
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
        Sample from fitted R-vine via inverse Rosenblatt transform.

        Uses the R-vine matrix to determine variable ordering and
        h-inverse chains (Czado 2019, Algorithm 6.1), guaranteeing
        correct sampling for any valid R-vine structure.
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
        """Core sampling via inverse Rosenblatt on the R-vine matrix.

        Implements the matrix-based algorithm (Czado 2019, Algorithm 6.1):
        process columns right-to-left, applying h_inverse from the deepest
        tree level down to tree 0, then forward h-transforms for later
        columns.

        Notation
        --------
        v_direct[(s, m)]  : pseudo-obs for the diagonal variable M[s,s]
            at conditioning level m in column s.
        v_indirect[(s, m)]: pseudo-obs for the "other" variable at
            conditioning level m in column s (used by earlier columns).
        """
        if self.edges is None:
            raise ValueError("Fit first")

        d = self.d
        M = self._structure.matrix
        edge_map = getattr(self, '_edge_map', None)
        if edge_map is None:
            edge_map = _build_matrix_edge_map(self._structure, self.trees, self.edges)

        eps = 1e-10
        pseudo = {}
        w = rng.uniform(eps, 1.0 - eps, size=(n, d))

        for s in range(d - 1, -1, -1):
            var = M[s, s]
            u_ind = _clip_unit(w[:, d - 1 - s])

            if s == d - 1:
                pseudo[(var, frozenset())] = u_ind
                continue

            n_levels = d - s - 1

            val = u_ind
            for m in range(n_levels - 1, -1, -1):
                edge_key = edge_map.get((m, s))
                if edge_key is None:
                    raise RuntimeError(
                        f"Missing edge_map entry for tree={m}, column={s}"
                    )

                edge = self.edges[edge_key]
                r = r_all[edge_key]

                partner_var = M[s + m + 1, s]
                cond_set = frozenset(M[s + 1:s + m + 1, s])

                partner_val = pseudo.get((partner_var, cond_set))
                if partner_val is None:
                    raise RuntimeError(
                        "Missing partner pseudo-observation during sampling: "
                        f"var={partner_var}, cond_set={sorted(cond_set)}, "
                        f"column={s}, level={m}"
                    )

                val = _clip_unit(edge.copula.h_inverse(val, partner_val, r))

            pseudo[(var, frozenset())] = val

            cur = val
            for m in range(n_levels):
                edge_key = edge_map.get((m, s))
                if edge_key is None:
                    raise RuntimeError(
                        f"Missing edge_map entry for tree={m}, column={s}"
                    )

                edge = self.edges[edge_key]
                r = r_all[edge_key]

                partner_var = M[s + m + 1, s]
                cond_set = frozenset(M[s + 1:s + m + 1, s])
                next_cond = frozenset(M[s + 1:s + m + 2, s])

                partner_val = pseudo.get((partner_var, cond_set))
                if partner_val is None:
                    raise RuntimeError(
                        "Missing partner pseudo-observation during forward "
                        f"propagation: var={partner_var}, "
                        f"cond_set={sorted(cond_set)}, column={s}, level={m}"
                    )

                var_at_cond = pseudo.get((var, cond_set), pseudo[(var, frozenset())])

                cur = _clip_unit(edge.copula.h(var_at_cond, partner_val, r))
                pseudo[(var, next_cond)] = cur

                rev = _clip_unit(edge.copula.h(partner_val, var_at_cond, r))
                pseudo[(partner_var, cond_set | {var})] = rev

        x = np.zeros((n, d))
        for var in range(d):
            key = (var, frozenset())
            if key not in pseudo:
                raise RuntimeError(f"Missing sampled variable {var}")
            x[:, var] = pseudo[key]

        return x

    # ── Prediction ───────────────────────────────────────────────

    def predict(self, n, u=None, K=300, grid_range=5.0,
                given=None, horizon='next', quad_order=10,
                conditional_method='auto'):
        """Conditional predict: sample for next-step prediction.

        `given` fixes selected variables in pseudo-observation space,
        e.g. ``given={1: 0.4}``. For SCAR-TM-OU edges, `horizon`
        selects between p(x_T | data) and p(x_{T+1} | data).
        `quad_order` controls Gauss-Legendre integration accuracy for
        arbitrary R-vine conditional sampling with non-prefix `given`.
        `conditional_method` selects the arbitrary-conditioning algorithm:
        'auto' uses the graph/no-posterior path or joint-grid fast path when
        suitable, 'graph' requires no posterior latent variables, 'exact' uses
        recursive quadrature, and 'grid' requires the joint-grid fast path.

        The graph paths implement the conditional-sampling idea from the vine
        computational graph formulation in arXiv:2506.13318. This method does
        not expose differentiable/autograd behavior.
        """
        if self.edges is None:
            raise ValueError("Fit first")

        u_data = u if u is not None else getattr(self, '_last_u', None)
        given = validate_rvine_given(given, self.d)
        d = self.d
        rng = np.random.default_rng()

        if len(given) == d:
            out = np.empty((n, d), dtype=np.float64)
            for i in range(d):
                out[:, i] = given[i]
            return out

        # Build training pseudo-obs if needed
        train_pseudo = None
        from pyscarcopula._types import LatentResult
        needs_train = any(
            isinstance(e.fit_result, LatentResult)
            for e in self.edges.values())

        predict_cache = getattr(self, '_predict_cache', None)
        if predict_cache is None:
            predict_cache = {}
            self._predict_cache = predict_cache
        u_token = None

        if u_data is not None and needs_train:
            u_arr = np.asarray(u_data)
            u_token = (id(u_arr), u_arr.shape, str(u_arr.dtype))
            train_key = ('train_pseudo', u_token, int(K), float(grid_range))
            train_pseudo = predict_cache.get(train_key)
            if train_pseudo is None:
                train_pseudo = self._compute_pseudo_obs(
                    u_data, K, grid_range, state_cache=predict_cache,
                    u_token=u_token)
                predict_cache[train_key] = train_pseudo

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

            state_key = None
            if v_pair is not None and u_token is not None:
                state_key = (
                    'state_dist', key, u_token, int(K), float(grid_range),
                    horizon,
                )

            r_all[key] = generate_r_for_predict(
                edge, n, v_pair, K, grid_range, horizon=horizon,
                state_cache=predict_cache, cache_key=state_key)

        if given:
            ensure_rvine_conditional_supported(self)
            return sample_rvine_conditional_with_r(
                self, n, r_all, given, rng, quad_order=quad_order,
                conditional_method=conditional_method)

        return self._sample_with_r(n, r_all, rng)

    def _compute_pseudo_obs(self, u, K=300, grid_range=5.0,
                            state_cache=None, u_token=None):
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
                    current_key = None
                    next_key = None
                    if state_cache is not None and u_token is not None:
                        current_key = (
                            'state_dist', (tree_level, edge_idx), u_token,
                            int(K), float(grid_range), 'current',
                        )
                        next_key = (
                            'state_dist', (tree_level, edge_idx), u_token,
                            int(K), float(grid_range), 'next',
                        )
                    h_2given1 = _clip_unit(
                        _edge_h(
                            edge, u2, u1, u_pair, K, grid_range,
                            state_cache=state_cache,
                            current_cache_key=current_key,
                            next_cache_key=next_key))
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
