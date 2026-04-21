"""
vine.rvine — MLE-only R-vine copula (Block 1 refactor).

Layout
------
Build path:
    u  ──►  select_rvine (Dissmann)  ──►  (trees_repr, fitted pair copulas)
                                                  │
                                                  ▼
                  build_rvine_matrix_with_edge_map  ──►  (M, edge_map)
                                                  │
                                                  ▼
                     ``RVineCopula`` stores M, trees, pair_copulas (by (t, col))

The R-vine *matrix* is the single source of truth for structure; pair
copulas are stored in a dict keyed by matrix position ``(tree, col)``.
The matrix follows the natural-order convention (Czado 2019, Alg. 5.4;
pyvinecopulib) — non-zero entries fill the upper-left anti-triangle,
the anti-diagonal ``M[d-1-col, col]`` holds the leaf peeled at column
``col``, and tree-``t`` edges at column ``col`` have their "other"
endpoint at row ``d-2-col-t``.

Current scope
-------------
Structure selection is still Dissmann-based. Edge fitting delegates to
the strategy registry (MLE/GAS/SCAR-TM-OU). Unconditional ``sample`` and
predictive ``predict`` use the natural-order matrix.

Usage
-----
    from pyscarcopula import RVineCopula

    vine = RVineCopula().fit(u)
    print(vine)                       # summary
    total_ll = vine.log_likelihood()  # cached fitted total
    new_ll = vine.log_likelihood(u_new)  # re-evaluate on new data
"""

import numpy as np

from pyscarcopula._utils import pobs
from pyscarcopula._types import GASResult, LatentResult, MLEResult
from pyscarcopula.copula.independent import IndependentCopula
from pyscarcopula.vine._conditional_rvine import validate_rvine_given
from pyscarcopula.vine._rvine_dissmann import PairCopula, select_rvine
from pyscarcopula.vine._rvine_edges import (
    _edge_h,
    _edge_h_inverse,
    _edge_initial_gas_state,
    _edge_r_for_sample,
    _edge_r_for_predict,
    _edge_update_gas_state,
    _is_gas_edge,
    _strategy_for_result,
)
from pyscarcopula.vine._rvine_matrix_builder import (
    build_rvine_matrix_with_edge_map,
)


_EPS = 1e-10


def _clip(u):
    return np.clip(u, _EPS, 1.0 - _EPS)


def _summary_family_name(copula):
    name = type(copula).__name__
    if name == 'BivariateGaussianCopula':
        return 'GaussianCopula'
    return name


def _summary_float(value):
    value = float(value)
    if abs(value) < 5e-4:
        value = 0.0
    return f"{value:.3g}"


def _summary_named_float(value):
    value = float(value)
    if abs(value) < 5e-4:
        value = 0.0
    return f"{value:7.3f}"


def _summary_dynamic_params(pc):
    if isinstance(pc.copula, IndependentCopula):
        return ''
    result = getattr(pc, 'fit_result', None)
    if isinstance(result, LatentResult):
        p = result.params
        return (
            f"theta={_summary_named_float(p.theta)}, "
            f"mu={_summary_named_float(p.mu)}, "
            f"nu={_summary_named_float(p.nu)}"
        )
    if isinstance(result, GASResult):
        p = result.params
        return (
            f"omega={_summary_named_float(p.omega)}, "
            f"alpha={_summary_named_float(p.alpha)}, "
            f"beta={_summary_named_float(p.beta)}"
        )
    return ''


def _summary_scalar_param(pc):
    if isinstance(pc.copula, IndependentCopula):
        return f"{pc.param:.4f}"
    result = getattr(pc, 'fit_result', None)
    if isinstance(result, MLEResult):
        return f"{result.copula_param:.4f}"
    if isinstance(result, (LatentResult, GASResult)):
        return ''
    return f"{pc.param:.4f}"


class RVineCopula:
    """Regular vine copula.

    Parameters
    ----------
    candidates : list of copula classes or None
        Family pool for per-edge selection. ``None`` uses the package
        default from ``_selection._default_candidates``.
    allow_rotations : bool, default True
        Whether to search over rotations for rotatable Archimedean
        families.
    criterion : {'aic', 'bic', 'loglik'}, default 'aic'
        Model selection criterion used within and between families.
    truncation_level : int or None
        If set, tree levels ``>= truncation_level`` use
        ``truncation_fill``.
    truncation_fill : {'mle', 'independent'}, default 'independent'
        For truncated trees, either fit edges with MLE only or force
        ``IndependentCopula``.
    threshold : float or None, default 0.0
        Pre-fit Kendall's tau threshold. If ``abs(tau) < threshold``,
        an edge is set to ``IndependentCopula`` without fitting.
    min_edge_logL : float or None
        If set, any fitted edge with log-likelihood strictly below this
        threshold is replaced by ``IndependentCopula``.
    transform_type : str, default 'xtanh'
        Parameter transform passed through to candidate copulas.

    Attributes (after ``fit``)
    --------------------------
    d : int
        Number of variables.
    matrix : (d, d) int ndarray
        Natural-order R-vine matrix (Czado 2019 Alg. 5.4; matches
        ``pyvinecopulib``). Non-zero entries occupy the upper-left
        anti-triangle; the anti-diagonal ``M[d-1-col, col]`` is the
        leaf peeled at column ``col``.
    trees : list of ``(d - 1)`` lists
        ``trees[t][i]`` = ``(conditioned_frozenset, conditioning_frozenset)``
        returned by the Dissmann selection step.
    pair_copulas : dict
        ``pair_copulas[(t, col)]`` = ``PairCopula`` for the edge encoded
        at matrix tree level ``t`` and column ``col``
        (``0 <= col <= d-2-t``).
    """

    def __init__(
        self,
        candidates=None,
        allow_rotations=True,
        criterion='aic',
        truncation_level=None,
        truncation_fill='independent',
        threshold=0.0,
        min_edge_logL=None,
        transform_type='xtanh',
    ):
        if criterion not in ('aic', 'bic', 'loglik'):
            raise ValueError(
                f"criterion must be 'aic', 'bic' or 'loglik', got {criterion!r}"
            )
        if truncation_level is not None:
            if not isinstance(truncation_level, (int, np.integer)):
                raise TypeError(
                    f"truncation_level must be int or None, "
                    f"got {type(truncation_level).__name__}"
                )
            if truncation_level < 0:
                raise ValueError(
                    f"truncation_level must be >= 0, got {truncation_level}"
                )
        if truncation_fill not in ('mle', 'independent'):
            raise ValueError(
                "truncation_fill must be 'mle' or 'independent', "
                f"got {truncation_fill!r}"
            )
        if threshold is not None and threshold < 0:
            raise ValueError(f"threshold must be >= 0 or None, got {threshold}")

        self.candidates = candidates
        self.allow_rotations = bool(allow_rotations)
        self.criterion = criterion
        self.truncation_level = truncation_level
        self.truncation_fill = truncation_fill
        self.threshold = threshold
        self.min_edge_logL = min_edge_logL
        self.transform_type = transform_type

        self.d = None
        self.matrix = None
        self.trees = None
        self.pair_copulas = None
        self._edge_map = None  # (t, col) -> orig_idx in trees[t]
        self._T = None
        self._log_likelihood = None
        self.method = None

    # ── Fit ────────────────────────────────────────────────────

    def fit(self, data, method='mle', *, to_pobs=False, copulas=None,
            config=None, **kwargs):
        """Fit the R-vine on pseudo-observations.

        Parameters
        ----------
        data : (T, d) array-like
            Observations in ``(0, 1)`` unless ``to_pobs=True``.
        method : {'mle', 'gas', 'scar-tm-ou'}, default 'mle'
            Estimation strategy for each selected pair copula.
        to_pobs : bool, default False
            If True, transform rows of ``data`` to pseudo-observations
            via the empirical distribution function.
        copulas : list-of-lists or None
            Optional fixed edge families as ``(copula_class, rotation)`` in
            the Dissmann edge order for each tree.
        config : NumericalConfig or None
            Optional numerical configuration passed to strategies.
        **kwargs
            Forwarded to the selected strategy.

        Returns
        -------
        self : RVineCopula
            Enables chained calls, e.g. ``RVineCopula().fit(u).summary()``.
        """
        u = np.asarray(data, dtype=np.float64)
        if u.ndim != 2:
            raise ValueError(f"RVineCopula.fit: data must be 2D, got shape {u.shape}")
        if to_pobs:
            u = pobs(u)

        T, d = u.shape
        if d < 2:
            raise ValueError(f"RVineCopula.fit: need d >= 2, got d={d}")

        truncation_level = kwargs.pop('truncation_level', self.truncation_level)
        truncation_fill = kwargs.pop('truncation_fill', self.truncation_fill)
        threshold = kwargs.pop('threshold', self.threshold)
        min_edge_logL = kwargs.pop('min_edge_logL', self.min_edge_logL)
        transform_type = kwargs.pop('transform_type', self.transform_type)

        if truncation_level is not None:
            if not isinstance(truncation_level, (int, np.integer)):
                raise TypeError(
                    f"truncation_level must be int or None, "
                    f"got {type(truncation_level).__name__}"
                )
            if truncation_level < 0:
                raise ValueError(
                    f"truncation_level must be >= 0, got {truncation_level}"
                )
        if truncation_fill not in ('mle', 'independent'):
            raise ValueError(
                "truncation_fill must be 'mle' or 'independent', "
                f"got {truncation_fill!r}"
            )
        if threshold is not None and threshold < 0:
            raise ValueError(f"threshold must be >= 0 or None, got {threshold}")

        trees_repr, fitted = select_rvine(
            u,
            candidates=self.candidates,
            allow_rotations=self.allow_rotations,
            criterion=self.criterion,
            method=method,
            copulas=copulas,
            config=config,
            truncation_level=truncation_level,
            truncation_fill=truncation_fill,
            threshold=threshold,
            min_edge_logL=min_edge_logL,
            transform_type=transform_type,
            **kwargs,
        )
        M, edge_map = build_rvine_matrix_with_edge_map(d, trees_repr)

        pair_copulas = {}
        for (t, col), orig_idx in edge_map.items():
            pair_copulas[(t, col)] = fitted[t][orig_idx]

        self.d = d
        self.matrix = M
        self.trees = trees_repr
        self.pair_copulas = pair_copulas
        self._edge_map = dict(edge_map)
        self._T = int(T)
        self._log_likelihood = float(sum(
            pc.log_likelihood for pc in pair_copulas.values()
        ))
        self.method = method.upper()
        self._last_u = u
        return self

    # ── Convenience predicates ─────────────────────────────────

    def _require_fit(self):
        if self.matrix is None:
            raise RuntimeError(
                "RVineCopula: call fit(...) before accessing fitted state"
            )

    # ── Log-likelihood ─────────────────────────────────────────

    def log_likelihood(self, data=None, to_pobs=False):
        """Total log-likelihood.

        With no argument returns the cached fitted log-likelihood
        (sum of per-edge MLE log-likelihoods). With an explicit
        ``data`` array, walks the fitted vine and evaluates the
        log-likelihood on the new observations using the stored pair
        copulas and the h-function propagation rule.
        """
        self._require_fit()
        if data is None:
            return self._log_likelihood

        u = np.asarray(data, dtype=np.float64)
        if to_pobs:
            u = pobs(u)
        if u.ndim != 2 or u.shape[1] != self.d:
            raise ValueError(
                f"RVineCopula.log_likelihood: data must be (T, {self.d}), "
                f"got {u.shape}"
            )

        pseudo_obs = {(i, frozenset()): u[:, i].copy() for i in range(self.d)}
        total = 0.0
        for t, level in enumerate(self.trees):
            for orig_idx, (conditioned, conditioning) in enumerate(level):
                pc = self.pair_copulas[self._matrix_key(t, orig_idx)]
                v1, v2 = sorted(conditioned)
                u1 = _clip(pseudo_obs[(v1, conditioning)])
                u2 = _clip(pseudo_obs[(v2, conditioning)])

                if not isinstance(pc.copula, IndependentCopula):
                    u_pair = np.column_stack((u1, u2))
                    if pc.fit_result is not None:
                        strategy = _strategy_for_result(pc.fit_result)
                        total += strategy.log_likelihood(
                            pc.copula, u_pair, pc.fit_result)
                    else:
                        r = np.full(len(u1), pc.param, dtype=np.float64)
                        total += float(np.sum(pc.copula.log_pdf(u1, u2, r)))

                if t < self.d - 2:
                    pseudo_obs[(v2, conditioning | {v1})] = _clip(pc.h(u2, u1))
                    pseudo_obs[(v1, conditioning | {v2})] = _clip(pc.h(u1, u2))
        return total

    def _matrix_key(self, tree_level, orig_idx):
        """Invert edge_map: (tree, orig_idx) -> (tree, col)."""
        for (t, col), idx in self._edge_map.items():
            if t == tree_level and idx == orig_idx:
                return (t, col)
        raise KeyError(
            f"RVineCopula: no matrix column for tree {tree_level}, edge {orig_idx}"
        )

    # ── Introspection ──────────────────────────────────────────

    @property
    def n_parameters(self):
        """Total number of fitted parameters across all pair copulas."""
        self._require_fit()
        return sum(pc.n_params for pc in self.pair_copulas.values())

    @property
    def aic(self):
        """AIC = -2 logL + 2 k."""
        self._require_fit()
        return -2.0 * self._log_likelihood + 2.0 * self.n_parameters

    @property
    def bic(self):
        """BIC = -2 logL + k log T."""
        self._require_fit()
        return -2.0 * self._log_likelihood + self.n_parameters * np.log(self._T)

    def family_matrix(self):
        """(d, d) object array with copula family names at edge positions.

        In the natural-order convention, the edge at tree ``t``, column
        ``col`` is encoded by the pair ``(M[d-1-col, col], M[d-2-col-t, col])``.
        This method puts the family name at position ``(d-2-col-t, col)``
        so that each column reads "top-tree at row 0 → tree-0 at row
        d-2-col → leaf at anti-diagonal row d-1-col". All other cells
        are empty strings.
        """
        self._require_fit()
        d = self.d
        M = np.full((d, d), "", dtype=object)
        for (t, col), pc in self.pair_copulas.items():
            M[d - 2 - col - t, col] = type(pc.copula).__name__
        return M

    def parameter_matrix(self):
        """(d, d) float array with fitted copula parameters (NaN elsewhere).

        Position ``(d-2-col-t, col)`` carries the parameter of the
        tree-``t`` edge at column ``col`` (natural-order convention).
        """
        self._require_fit()
        d = self.d
        M = np.full((d, d), np.nan, dtype=np.float64)
        for (t, col), pc in self.pair_copulas.items():
            M[d - 2 - col - t, col] = pc.param
        return M

    def rotation_matrix(self):
        """(d, d) int array with copula rotations (-1 elsewhere).

        Position ``(d-2-col-t, col)`` carries the rotation of the
        tree-``t`` edge at column ``col`` (natural-order convention).
        """
        self._require_fit()
        d = self.d
        M = np.full((d, d), -1, dtype=int)
        for (t, col), pc in self.pair_copulas.items():
            M[d - 2 - col - t, col] = int(getattr(pc.copula, 'rotate', 0))
        return M

    def tau_matrix(self):
        """(d, d) float array with empirical Kendall's tau per edge.

        Position ``(d-2-col-t, col)`` carries tau of the tree-``t`` edge
        at column ``col`` (natural-order convention).
        """
        self._require_fit()
        d = self.d
        M = np.full((d, d), np.nan, dtype=np.float64)
        for (t, col), pc in self.pair_copulas.items():
            M[d - 2 - col - t, col] = pc.tau
        return M

    # ── Summary / repr ─────────────────────────────────────────

    def summary(self, as_string=False):
        """Print R-vine structure summary.

        Matches ``CVineCopula.summary()`` behavior: by default the summary is
        printed and ``None`` is returned. Use ``summary(as_string=True)`` when
        a string value is needed.

        Returns
        -------
        text : str or None
        """
        if self.matrix is None:
            text = "RVineCopula (unfitted)"
            if as_string:
                return text
            print(text)
            return None

        lines = []
        lines.append(
            f"RVineCopula(d={self.d}, T={self._T}, criterion={self.criterion!r})"
        )
        lines.append(f"  log_likelihood = {self._log_likelihood:.4f}")
        lines.append(f"  n_parameters   = {self.n_parameters}")
        lines.append(f"  AIC = {self.aic:.4f}   BIC = {self.bic:.4f}")
        if self.truncation_level is not None:
            lines.append(f"  truncation_level = {self.truncation_level}")
            lines.append(f"  truncation_fill  = {self.truncation_fill}")
        if self.threshold not in (None, 0.0):
            lines.append(f"  threshold        = {self.threshold}")
        if self.min_edge_logL is not None:
            lines.append(f"  min_edge_logL   = {self.min_edge_logL}")

        lines.append("")
        lines.append(
            "Structure matrix (natural order, Czado 2019 Alg. 5.4; "
            "anti-diagonal = leaf peeled at each column):"
        )
        lines.append(np.array2string(self.matrix, separator=" "))

        lines.append("")
        lines.append("Edges (tree t, column col):")
        has_dynamic = any(
            isinstance(getattr(pc, 'fit_result', None), (LatentResult, GASResult))
            for pc in self.pair_copulas.values()
        )
        if has_dynamic:
            header = f"  {'t':>2} {'col':>4} {'pair':>10} {'cond':>14}  "\
                     f"{'family':<18} {'rot':>4}  {'dyn_params':<45}"\
                     f"{'param':>9} {'tau':>7} {'logL':>10}"
        else:
            header = f"  {'t':>2} {'col':>4} {'pair':>10} {'cond':>14}  "\
                     f"{'family':<18} {'rot':>4} {'param':>9} "\
                     f"{'tau':>7} {'logL':>10}"
        lines.append(header)
        d = self.d
        for t in range(d - 1):
            for col in range(d - 1 - t):
                pc = self.pair_copulas[(t, col)]
                leaf = int(self.matrix[d - 1 - col, col])
                tail = int(self.matrix[d - 2 - col - t, col])
                cond = sorted(
                    int(self.matrix[r, col])
                    for r in range(d - 1 - col - t, d - 1 - col)
                )
                pair_str = f"({leaf},{tail})"
                cond_str = ",".join(str(c) for c in cond) if cond else "-"
                fam = _summary_family_name(pc.copula)
                rot = int(getattr(pc.copula, 'rotate', 0))
                param = _summary_scalar_param(pc)
                base = (
                    f"  {t:>2} {col:>4} {pair_str:>10} {cond_str:>14}  "
                    f"{fam:<18} {rot:>4} "
                )
                if has_dynamic:
                    dyn_params = _summary_dynamic_params(pc)
                    lines.append(
                        f"{base} {dyn_params:<45} {param:>9} "
                        f"{pc.tau:>7.3f} {pc.log_likelihood:>10.3f}"
                    )
                else:
                    lines.append(
                        f"{base} {param:>9} "
                        f"{pc.tau:>7.3f} {pc.log_likelihood:>10.3f}"
                    )
        text = "\n".join(lines)
        if as_string:
            return text
        print(text)
        return None

    def __str__(self):
        return self.summary(as_string=True)

    def __repr__(self):
        if self.matrix is None:
            return "RVineCopula(unfitted)"
        return (
            f"RVineCopula(d={self.d}, T={self._T}, "
            f"logL={self._log_likelihood:.3f}, "
            f"n_params={self.n_parameters})"
        )

    # ── Sampling ───────────────────────────────────────────────

    def sample(self, n, u_train=None, rng=None):
        """Unconditional sampling from the fitted vine.

        Samples in natural-order matrix order: columns are processed from
        right to left, and each new anti-diagonal leaf is recovered by
        applying inverse h-functions from the top tree down to tree 0.
        """
        self._require_fit()
        if not isinstance(n, (int, np.integer)) or n <= 0:
            raise ValueError(f"RVineCopula.sample: n must be positive int, got {n!r}")
        if rng is None:
            rng = np.random.default_rng()

        n = int(n)
        if any(_is_gas_edge(edge) for edge in self.pair_copulas.values()):
            return self._sample_stepwise_gas(n, rng)

        r_all = {
            key: _edge_r_for_sample(edge, n, rng)
            for key, edge in self.pair_copulas.items()
        }
        return self._sample_with_r(n, r_all, rng)

    def _sample_with_r(self, n, r_all, rng, return_pseudo=False):
        d = self.d
        M = self.matrix
        w = rng.uniform(_EPS, 1.0 - _EPS, size=(n, d))
        pseudo_obs = {}

        last_var = int(M[0, d - 1])
        pseudo_obs[(last_var, frozenset())] = w[:, d - 1].copy()

        for col in range(d - 2, -1, -1):
            leaf = int(M[d - 1 - col, col])
            top_tree = d - 2 - col
            current = w[:, col].copy()

            for t in range(top_tree, -1, -1):
                row = d - 2 - col - t
                partner = int(M[row, col])
                conditioning = frozenset(
                    int(M[r, col])
                    for r in range(row + 1, d - 1 - col)
                )
                partner_val = pseudo_obs[(partner, conditioning)]
                edge = self.pair_copulas[(t, col)]
                current = _clip(_edge_h_inverse(
                    edge,
                    current,
                    partner_val,
                    config={'r': r_all[(t, col)]},
                ))
                pseudo_obs[(leaf, conditioning)] = current

            for t in range(top_tree + 1):
                row = d - 2 - col - t
                partner = int(M[row, col])
                conditioning = frozenset(
                    int(M[r, col])
                    for r in range(row + 1, d - 1 - col)
                )
                next_leaf_cond = conditioning | {partner}
                next_partner_cond = conditioning | {leaf}
                edge = self.pair_copulas[(t, col)]
                r = r_all[(t, col)]

                leaf_val = pseudo_obs[(leaf, conditioning)]
                partner_val = pseudo_obs[(partner, conditioning)]
                pseudo_obs[(leaf, next_leaf_cond)] = _clip(
                    _edge_h(edge, leaf_val, partner_val, config={'r': r})
                )
                pseudo_obs[(partner, next_partner_cond)] = _clip(
                    _edge_h(edge, partner_val, leaf_val, config={'r': r})
                )

        out = np.empty((n, d), dtype=np.float64)
        for var in range(d):
            out[:, var] = pseudo_obs[(var, frozenset())]
        if return_pseudo:
            return out, pseudo_obs
        return out

    def _given_suffix_start_col(self, given, matrix=None):
        matrix = self.matrix if matrix is None else matrix
        if not given:
            return self.d
        peel_order = [
            int(matrix[self.d - 1 - col, col])
            for col in range(self.d)
        ]
        k = len(given)
        suffix = set(peel_order[self.d - k:])
        if set(given) == suffix:
            return self.d - k
        return None

    def _suffix_sampling_state(self, given):
        start_col = self._given_suffix_start_col(given)
        if start_col is not None:
            return start_col, self.matrix, self._edge_map, self.pair_copulas

        given_vars = set(given)
        peel_order = self._find_peel_order_for_given_suffix(given_vars)
        if peel_order is None:
            return None

        to_perm = {var: idx for idx, var in enumerate(peel_order)}
        from_perm = {idx: var for var, idx in to_perm.items()}

        relabeled_trees = []
        for level in self.trees:
            relabeled_level = []
            for conditioned, conditioning in level:
                relabeled_level.append((
                    frozenset(to_perm[v] for v in conditioned),
                    frozenset(to_perm[v] for v in conditioning),
                ))
            relabeled_trees.append(relabeled_level)

        try:
            perm_matrix, edge_map = build_rvine_matrix_with_edge_map(
                self.d, relabeled_trees)
        except RuntimeError:
            return None

        matrix = np.zeros_like(perm_matrix)
        for col in range(self.d):
            for row in range(self.d - col):
                matrix[row, col] = from_perm[int(perm_matrix[row, col])]

        start_col = self._given_suffix_start_col(given, matrix=matrix)
        if start_col is None:
            return None

        pair_by_orig = {
            (t, orig_idx): self.pair_copulas[self._matrix_key(t, orig_idx)]
            for t, level in enumerate(self.trees)
            for orig_idx in range(len(level))
        }
        pair_copulas = {
            key: pair_by_orig[(key[0], orig_idx)]
            for key, orig_idx in edge_map.items()
        }
        return start_col, matrix, edge_map, pair_copulas

    def _find_peel_order_for_given_suffix(self, given_vars):
        prefix_len = self.d - len(given_vars)

        def search(col, claimed, peeled):
            if col == self.d - 1:
                remaining = set(range(self.d)) - set(peeled)
                if len(remaining) == 1 and remaining <= given_vars:
                    return peeled + [remaining.pop()]
                return None

            tree_level = self.d - 2 - col
            top_candidates = [
                idx for idx in range(len(self.trees[tree_level]))
                if idx not in claimed[tree_level]
            ]
            if len(top_candidates) != 1:
                return None

            idx_top = top_candidates[0]
            conditioned_top, conditioning_top = self.trees[tree_level][idx_top]
            candidates = sorted(conditioned_top)
            if col < prefix_len:
                candidates = [v for v in candidates if v not in given_vars]
            if not candidates:
                return None

            for leaf in candidates:
                cond_accum = set()
                walk_claims = []
                valid = True
                for t in range(tree_level):
                    target_cc = frozenset(cond_accum)
                    hits = [
                        idx
                        for idx, (conditioned, conditioning)
                        in enumerate(self.trees[t])
                        if (idx not in claimed[t]
                            and leaf in conditioned
                            and conditioning == target_cc)
                    ]
                    if len(hits) != 1:
                        valid = False
                        break
                    idx_t = hits[0]
                    conditioned_t, _ = self.trees[t][idx_t]
                    other = next(iter(conditioned_t - {leaf}))
                    cond_accum.add(other)
                    walk_claims.append((t, idx_t))

                if not valid or cond_accum != set(conditioning_top):
                    continue

                next_claimed = [set(level) for level in claimed]
                next_claimed[tree_level].add(idx_top)
                for t, idx_t in walk_claims:
                    next_claimed[t].add(idx_t)

                result = search(
                    col + 1,
                    tuple(frozenset(level) for level in next_claimed),
                    peeled + [leaf],
                )
                if result is not None:
                    return result
            return None

        empty_claimed = tuple(frozenset() for _ in range(self.d - 1))
        return search(0, empty_claimed, [])

    def _sample_suffix_given_with_r(self, n, r_all, rng, given, start_col,
                                    matrix=None, pair_copulas=None):
        d = self.d
        M = self.matrix if matrix is None else matrix
        pair_copulas = self.pair_copulas if pair_copulas is None else pair_copulas
        w = rng.uniform(_EPS, 1.0 - _EPS, size=(n, d))
        pseudo_obs = {}

        last_var = int(M[0, d - 1])
        if d - 1 >= start_col:
            pseudo_obs[(last_var, frozenset())] = np.full(
                n, given[last_var], dtype=np.float64)
        else:
            pseudo_obs[(last_var, frozenset())] = w[:, d - 1].copy()

        for col in range(d - 2, start_col - 1, -1):
            leaf = int(M[d - 1 - col, col])
            top_tree = d - 2 - col
            pseudo_obs[(leaf, frozenset())] = np.full(
                n, given[leaf], dtype=np.float64)
            for t in range(top_tree + 1):
                row = d - 2 - col - t
                partner = int(M[row, col])
                conditioning = frozenset(
                    int(M[r, col])
                    for r in range(row + 1, d - 1 - col)
                )
                next_leaf_cond = conditioning | {partner}
                next_partner_cond = conditioning | {leaf}
                edge = pair_copulas[(t, col)]
                r = r_all[(t, col)]

                leaf_val = pseudo_obs[(leaf, conditioning)]
                partner_val = pseudo_obs[(partner, conditioning)]
                pseudo_obs[(leaf, next_leaf_cond)] = _clip(
                    _edge_h(edge, leaf_val, partner_val, config={'r': r})
                )
                pseudo_obs[(partner, next_partner_cond)] = _clip(
                    _edge_h(edge, partner_val, leaf_val, config={'r': r})
                )

        for col in range(start_col - 1, -1, -1):
            leaf = int(M[d - 1 - col, col])
            top_tree = d - 2 - col
            current = w[:, col].copy()

            for t in range(top_tree, -1, -1):
                row = d - 2 - col - t
                partner = int(M[row, col])
                conditioning = frozenset(
                    int(M[r, col])
                    for r in range(row + 1, d - 1 - col)
                )
                partner_val = pseudo_obs[(partner, conditioning)]
                edge = pair_copulas[(t, col)]
                current = _clip(_edge_h_inverse(
                    edge,
                    current,
                    partner_val,
                    config={'r': r_all[(t, col)]},
                ))
                pseudo_obs[(leaf, conditioning)] = current

            for t in range(top_tree + 1):
                row = d - 2 - col - t
                partner = int(M[row, col])
                conditioning = frozenset(
                    int(M[r, col])
                    for r in range(row + 1, d - 1 - col)
                )
                next_leaf_cond = conditioning | {partner}
                next_partner_cond = conditioning | {leaf}
                edge = pair_copulas[(t, col)]
                r = r_all[(t, col)]

                leaf_val = pseudo_obs[(leaf, conditioning)]
                partner_val = pseudo_obs[(partner, conditioning)]
                pseudo_obs[(leaf, next_leaf_cond)] = _clip(
                    _edge_h(edge, leaf_val, partner_val, config={'r': r})
                )
                pseudo_obs[(partner, next_partner_cond)] = _clip(
                    _edge_h(edge, partner_val, leaf_val, config={'r': r})
                )

        out = np.empty((n, d), dtype=np.float64)
        for var in range(d):
            out[:, var] = pseudo_obs[(var, frozenset())]
        return out

    def _sample_stepwise_gas(self, n, rng):
        gas_state = {
            key: _edge_initial_gas_state(edge)
            for key, edge in self.pair_copulas.items()
            if _is_gas_edge(edge)
        }
        non_gas_r = {
            key: _edge_r_for_sample(edge, n, rng)
            for key, edge in self.pair_copulas.items()
            if key not in gas_state
        }

        out = np.empty((n, self.d), dtype=np.float64)
        for i in range(n):
            r_i = {}
            for key in self.pair_copulas:
                if key in gas_state:
                    r_i[key] = np.array([gas_state[key][1]], dtype=np.float64)
                else:
                    r_i[key] = non_gas_r[key][i:i + 1]

            row, pseudo_obs = self._sample_with_r(
                1, r_i, rng, return_pseudo=True)
            out[i, :] = row[0]

            for key, state in gas_state.items():
                edge = self.pair_copulas[key]
                u_pair = self._edge_pair_from_pseudo(key, pseudo_obs)
                gas_state[key] = _edge_update_gas_state(edge, state[0], u_pair)

        return out

    def _edge_pair_from_pseudo(self, key, pseudo_obs):
        return self._edge_pair_from_pseudo_map(key, pseudo_obs, self._edge_map)

    def _edge_pair_from_pseudo_map(self, key, pseudo_obs, edge_map):
        t, col = key
        orig_idx = edge_map[(t, col)]
        conditioned, conditioning = self.trees[t][orig_idx]
        v1, v2 = sorted(conditioned)
        return np.column_stack((
            _clip(pseudo_obs[(v1, conditioning)]),
            _clip(pseudo_obs[(v2, conditioning)]),
        ))

    def _compute_pseudo_obs(self, u):
        pseudo_obs = {
            (i, frozenset()): u[:, i].copy()
            for i in range(self.d)
        }
        for t, level in enumerate(self.trees):
            for orig_idx, (conditioned, conditioning) in enumerate(level):
                pc = self.pair_copulas[self._matrix_key(t, orig_idx)]
                v1, v2 = sorted(conditioned)
                u1 = _clip(pseudo_obs[(v1, conditioning)])
                u2 = _clip(pseudo_obs[(v2, conditioning)])
                if t < self.d - 2:
                    pseudo_obs[(v2, conditioning | {v1})] = _clip(
                        _edge_h(pc, u2, u1))
                    pseudo_obs[(v1, conditioning | {v2})] = _clip(
                        _edge_h(pc, u1, u2))
        return pseudo_obs

    def _predict_r_for_edges(self, edge_keys, pair_copulas, edge_map, n,
                             train_pseudo, horizon, rng):
        edge_horizon = 1 if horizon == 'next' else horizon
        r_all = {}
        for key in edge_keys:
            edge = pair_copulas[key]
            u_pair = None
            if train_pseudo is not None:
                u_pair = self._edge_pair_from_pseudo_map(
                    key, train_pseudo, edge_map)
            r_all[key] = _edge_r_for_predict(
                edge,
                n,
                u_train_pair=u_pair,
                horizon=edge_horizon,
                rng=rng,
            )
        return r_all

    def predict(self, n, u_train=None, horizon='next', rng=None, given=None,
                u=None):
        """Predictive sampling from fitted edge states.

        ``given`` fixes variables in pseudo-observation space. Conditional
        sampling is supported when the fixed variables can be placed at the
        end of the R-vine variable order, read from the anti-diagonal of the
        natural-order matrix. This can be true in the fitted matrix itself or
        after rebuilding the same fitted tree structure into an equivalent
        natural-order matrix with those variables last.

        For GAS edges, ``horizon='current'`` uses Psi(f_T) and ``'next'`` uses
        one score update to Psi(f_{T+1}). For SCAR-TM edges, the same argument
        selects p(x_T | data) or p(x_{T+1} | data) before sampling the
        posterior mixture path.
        """
        self._require_fit()
        if u is not None:
            if u_train is not None:
                raise ValueError("Pass only one of u_train or u")
            u_train = u
        if not isinstance(n, (int, np.integer)) or n <= 0:
            raise ValueError(f"RVineCopula.predict: n must be positive int, got {n!r}")
        horizon = str(horizon).lower()
        if horizon not in ('current', 'next'):
            raise ValueError("horizon must be 'current' or 'next'")
        if rng is None:
            rng = np.random.default_rng()

        n = int(n)
        given = validate_rvine_given(given, self.d)
        if len(given) == self.d:
            out = np.empty((n, self.d), dtype=np.float64)
            for i in range(self.d):
                out[:, i] = given[i]
            return out
        suffix_state = self._suffix_sampling_state(given) if given else None

        u_ref = self._last_u if u_train is None else np.asarray(
            u_train, dtype=np.float64)
        if u_ref is not None and (u_ref.ndim != 2 or u_ref.shape[1] != self.d):
            raise ValueError(
                f"RVineCopula.predict: u_train must be (T, {self.d}), "
                f"got {u_ref.shape}"
            )

        train_pseudo = self._compute_pseudo_obs(u_ref) if u_ref is not None else None
        if suffix_state is None:
            suffix_start_col = None
            matrix = self.matrix
            edge_map = self._edge_map
            pair_copulas = self.pair_copulas
        else:
            suffix_start_col, matrix, edge_map, pair_copulas = suffix_state

        if given:
            if suffix_start_col is None:
                raise ValueError(
                    "RVineCopula.predict: fixed variables cannot be placed "
                    "last in the R-vine variable order; arbitrary "
                    "conditional sampling is not supported"
                )
            r_all = self._predict_r_for_edges(
                pair_copulas.keys(),
                pair_copulas,
                edge_map,
                n,
                train_pseudo,
                horizon,
                rng,
            )
            return self._sample_suffix_given_with_r(
                n,
                r_all,
                rng,
                given,
                suffix_start_col,
                matrix=matrix,
                pair_copulas=pair_copulas,
            )
        r_all = self._predict_r_for_edges(
            pair_copulas.keys(),
            pair_copulas,
            edge_map,
            n,
            train_pseudo,
            horizon,
            rng,
        )
        return self._sample_with_r(n, r_all, rng)
