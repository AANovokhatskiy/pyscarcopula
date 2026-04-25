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

from copy import deepcopy

import numpy as np

from pyscarcopula._utils import pobs
from pyscarcopula._types import (
    GASResult,
    LatentResult,
    MLEResult,
    PredictConfig,
    PredictiveState,
)
from pyscarcopula.copula.independent import IndependentCopula
from pyscarcopula.vine._conditional_rvine import (
    find_rvine_peel_order_for_given_suffix,
    validate_rvine_given_vars,
    validate_rvine_given,
)
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
from pyscarcopula.vine._rvine_dag import (
    build_runtime_rvine_dag,
    execute_conditional_plan,
    plan_conditional_sample,
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
        self._target_given_vars = ()
        self._conditional_fit_supported = None
        self._conditional_mode = None
        self._fit_diagnostics = None

    # ── Fit ────────────────────────────────────────────────────

    def fit(self, data, method='mle', *, to_pobs=False, copulas=None,
            config=None, given_vars=None, conditional_strict=True,
            conditional_mode='suffix', **kwargs):
        """Fit the R-vine and its pair-copula edge models.

        The input must already be in pseudo-observation space unless
        ``to_pobs=True``. Structure selection uses the instance-level family
        pool and selection options, while several structure and strategy
        options can be overridden for this call via ``**kwargs``.

        Parameters
        ----------
        data : (T, d) array-like
            Pseudo-observations in ``(0, 1)``. If ``to_pobs=True``, raw
            observations are converted column-wise with empirical ranks.
        method : str, default 'mle'
            Estimation strategy for every non-independent selected pair
            copula. Common built-in values are ``'mle'``, ``'gas'`` and
            ``'scar-tm-ou'``; any method registered in the strategy registry
            may be used.
        to_pobs : bool, default False
            If True, transform ``data`` to pseudo-observations before fitting.
        copulas : list-of-lists or None
            Optional fixed edge families as ``(copula_class, rotation)`` in
            the Dissmann edge order for each tree. If ``None``, the best
            family is selected for each edge from the candidate pool.
        config : NumericalConfig or None
            Optional numerical configuration passed to pair-copula strategies.
        given_vars : iterable[int] or None
            Optional target set of variable indices for later conditional
            prediction. When provided, structure search prefers vines where
            these variables can be fixed exactly by the current suffix sampler.
        conditional_strict : bool, default True
            If True and ``given_vars`` is set, raise ``ValueError`` when the
            selected structure cannot support exact conditional sampling for
            that target set. If False, fit succeeds and the result is reported
            through ``fit_diagnostics``.
        conditional_mode : {'suffix'}, default 'suffix'
            Conditioning support mode enforced during fit. Currently only
            ``'suffix'`` is supported.
        truncation_level : int or None, optional
            ``**kwargs`` option overriding the instance setting for this fit.
            Tree levels ``>= truncation_level`` use ``truncation_fill``.
        truncation_fill : {'mle', 'independent'}, optional
            ``**kwargs`` option overriding how truncated trees are handled:
            fit MLE-only edges or force ``IndependentCopula``.
        threshold : float or None, optional
            ``**kwargs`` option overriding the pre-fit Kendall's tau filter.
            Edges with ``abs(tau) < threshold`` are made independent. ``0.0``
            disables filtering; ``None`` also disables filtering.
        min_edge_logL : float or None, optional
            ``**kwargs`` option overriding the post-fit edge log-likelihood
            filter. Non-independent edges with log-likelihood below this value
            are replaced by ``IndependentCopula``.
        transform_type : str, optional
            ``**kwargs`` option overriding the parameter transform passed when
            constructing candidate copulas.
        structure_search : {'beam', 'multi-start'}, optional
            ``**kwargs`` option controlling conditional structure search.
            Mainly relevant when ``given_vars`` is provided. The default in
            the selector is ``'beam'``.
        beam_width : int, optional
            ``**kwargs`` option giving the number of partial structures kept
            per level by beam search. Must be a positive integer.
        **kwargs
            Remaining keyword arguments are forwarded to the selected
            pair-copula strategy. Built-in strategy options include
            ``alpha0``, ``tol`` and ``verbose`` for fitting; ``scaling`` for
            GAS; ``K``, ``grid_range``, ``grid_method``, ``adaptive``,
            ``pts_per_sigma``, ``analytical_grad`` and ``smart_init`` for
            SCAR-TM; and ``n_tr`` / ``M_iterations`` for SCAR-MC strategies.

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
        given_vars = validate_rvine_given_vars(given_vars, d)
        conditional_mode = str(conditional_mode).lower()
        if conditional_mode != 'suffix':
            raise ValueError(
                f"conditional_mode must be 'suffix', got {conditional_mode!r}"
            )

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

        select_result = select_rvine(
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
            given_vars=given_vars,
            return_diagnostics=True,
            **kwargs,
        )
        if len(select_result) == 3:
            trees_repr, fitted, selection_diagnostics = select_result
        else:
            trees_repr, fitted = select_result
            selection_diagnostics = {
                'target_given_vars': tuple(given_vars),
                'selected_mode': None,
                'selected_index': None,
                'selected_candidate': {
                    'mode': None,
                    'exact_supported': False,
                    'dag_complete': False,
                    'fit_score': 0.0,
                    'missing_base_vars': tuple(given_vars),
                    'reachable_base_vars': (),
                    'n_known_nodes': 0,
                    'n_steps': 0,
                    'n_inverse_steps': 0,
                },
                'candidates': (),
            }
        M, edge_map = build_rvine_matrix_with_edge_map(d, trees_repr)

        pair_copulas = {}
        for (t, col), orig_idx in edge_map.items():
            pair_copulas[(t, col)] = fitted[t][orig_idx]

        conditional_supported = True
        reject_reason = None
        if given_vars:
            probe = self.__class__()
            probe.d = d
            probe.matrix = M
            probe.trees = trees_repr
            probe.pair_copulas = pair_copulas
            probe._edge_map = dict(edge_map)
            conditional_supported = probe._suffix_sampling_state({
                var: 0.5 for var in given_vars
            }) is not None
            if not conditional_supported:
                reject_reason = 'unsupported_given_vars'
            self._fit_diagnostics = self._build_fit_diagnostics(
                given_vars,
                conditional_mode,
                conditional_strict,
                selection_diagnostics,
                conditional_supported,
                reject_reason=reject_reason,
            )
            if conditional_strict and not conditional_supported:
                missing_base_vars = ()
                selected_mode = None
                if selection_diagnostics['selected_candidate'] is not None:
                    missing_base_vars = selection_diagnostics[
                        'selected_candidate']['missing_base_vars']
                    selected_mode = selection_diagnostics['selected_mode']
                raise ValueError(
                    "RVineCopula.fit: could not construct an R-vine structure "
                    "supporting exact conditional sampling for "
                    f"given_vars={list(given_vars)}; "
                    f"selected_mode={selected_mode}; "
                    "missing_base_vars="
                    f"{list(missing_base_vars)}"
                )
        else:
            self._fit_diagnostics = self._build_fit_diagnostics(
                given_vars,
                None,
                conditional_strict,
                selection_diagnostics,
                conditional_supported,
                reject_reason=reject_reason,
            )

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
        self._target_given_vars = given_vars
        self._conditional_fit_supported = conditional_supported
        self._conditional_mode = conditional_mode if given_vars else None
        return self

    def _build_fit_diagnostics(
            self,
            given_vars,
            conditional_mode,
            conditional_strict,
            selection_diagnostics,
            conditional_supported,
            *,
            reject_reason):
        return {
            'target_given_vars': tuple(given_vars),
            'conditional_mode': conditional_mode,
            'conditional_strict': bool(conditional_strict),
            'conditional_fit_supported': bool(conditional_supported),
            'reject_reason': reject_reason,
            'selection': deepcopy(selection_diagnostics),
        }

    # ── Convenience predicates ─────────────────────────────────

    def _require_fit(self):
        if self.matrix is None:
            raise RuntimeError(
                "RVineCopula: call fit(...) before accessing fitted state"
            )

    @property
    def fit_diagnostics(self):
        """Fit-time structure-selection diagnostics, if available."""
        if self._fit_diagnostics is None:
            return None
        return deepcopy(self._fit_diagnostics)

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
        pair_copulas = {}
        for key, orig_idx in edge_map.items():
            # edge_map indices refer to relabeled_trees[t][orig_idx]. Because
            # relabeled_trees preserves self.trees order within each level,
            # the same orig_idx points back to the fitted pair-copula edge.
            t = key[0]
            assert 0 <= orig_idx < len(self.trees[t])
            pair_copulas[key] = pair_by_orig[(t, orig_idx)]
        return start_col, matrix, edge_map, pair_copulas

    def _find_peel_order_for_given_suffix(self, given_vars):
        return find_rvine_peel_order_for_given_suffix(
            self.trees, self.d, given_vars)

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

    def _given_suffix_edge_observations_with_r(
            self, n, r_all, given, start_col, matrix=None, pair_copulas=None,
            edge_map=None):
        """Return edge observations fully determined by fixed suffix values."""
        d = self.d
        M = self.matrix if matrix is None else matrix
        pair_copulas = self.pair_copulas if pair_copulas is None else pair_copulas
        edge_map = self._edge_map if edge_map is None else edge_map
        pseudo_obs = {}
        observed = {}

        last_var = int(M[0, d - 1])
        if d - 1 >= start_col:
            pseudo_obs[(last_var, frozenset())] = np.full(
                n, given[last_var], dtype=np.float64)
        else:
            return observed

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
                observed[(t, col)] = self._edge_pair_from_pseudo_map(
                    (t, col), pseudo_obs, edge_map)
                pseudo_obs[(leaf, next_leaf_cond)] = _clip(
                    _edge_h(edge, leaf_val, partner_val, config={'r': r})
                )
                pseudo_obs[(partner, next_partner_cond)] = _clip(
                    _edge_h(edge, partner_val, leaf_val, config={'r': r})
                )

        return observed

    def _scar_tm_given_update_r(self, edge, u_train_pair, u_observed_pair,
                                n, horizon, rng, predictive_r_mode,
                                state_cache=None, cache_key=None):
        result = getattr(edge, 'fit_result', None)
        if not isinstance(result, LatentResult):
            return None
        if str(result.method).upper() != 'SCAR-TM-OU':
            return None
        if u_train_pair is None or len(u_train_pair) == 0:
            return None

        strategy = _strategy_for_result(result)
        state = strategy.predictive_state(
            edge.copula,
            u_train_pair,
            result,
            horizon=horizon,
            predictive_r_mode=predictive_r_mode,
            state_cache=state_cache,
            cache_key=cache_key,
        )
        conditioned = strategy.condition_state(
            edge.copula,
            state,
            u_observed_pair,
            result,
        )
        if (
                conditioned.kind != 'grid'
                or conditioned.prob is None
                or np.array_equal(
                    np.asarray(conditioned.prob, dtype=np.float64),
                    np.asarray(state.prob, dtype=np.float64),
                )):
            return None
        return strategy.sample_params(
            edge.copula,
            conditioned,
            n,
            rng=rng,
            predictive_r_mode=predictive_r_mode,
        )

    def _dynamic_edge_update_from_observation(
            self, key, edge, r_current, u_pair, edge_map, train_pseudo,
            horizon, rng, predictive_r_mode, state_cache=None):
        result = getattr(edge, 'fit_result', None)
        if _is_gas_edge(edge):
            if str(horizon).lower() == 'next':
                return None
            strategy = _strategy_for_result(result)
            state = PredictiveState(
                method=result.method,
                horizon=horizon,
                kind='point',
                r=np.array([float(np.asarray(r_current)[0])], dtype=np.float64),
            )
            conditioned_state = strategy.condition_state(
                edge.copula,
                state,
                u_pair,
                result,
            )
            return strategy.sample_params(
                edge.copula,
                conditioned_state,
                len(r_current),
                rng=rng,
                predictive_r_mode=predictive_r_mode,
            )

        if isinstance(result, LatentResult) and str(result.method).upper() == 'SCAR-TM-OU':
            if train_pseudo is None:
                return None
            u_train_pair = self._edge_pair_from_pseudo_map(
                key, train_pseudo, edge_map)
            return self._scar_tm_given_update_r(
                edge,
                u_train_pair,
                u_pair,
                len(r_current),
                horizon,
                rng,
                predictive_r_mode,
                state_cache=state_cache,
                cache_key=('predictive_state', key, horizon),
            )

        return None

    def _dynamic_edge_skip_reason(self, edge, train_pseudo, horizon):
        if _is_gas_edge(edge) and str(horizon).lower() == 'next':
            return 'gas_next_horizon_would_advance_filter'
        if (
                isinstance(getattr(edge, 'fit_result', None), LatentResult)
                and train_pseudo is None):
            return 'no_training_history'
        return 'unsupported_or_noop'

    def _dynamic_update_record(
            self, key, edge, edge_map, r_before, r_after, status, reason=None):
        orig_idx = edge_map[key]
        conditioned, conditioning = self.trees[key[0]][orig_idx]
        record = {
            'key': tuple(int(v) for v in key),
            'tree': int(key[0]),
            'col': int(key[1]),
            'conditioned': tuple(sorted(int(v) for v in conditioned)),
            'conditioning': tuple(sorted(int(v) for v in conditioning)),
            'method': str(getattr(getattr(edge, 'fit_result', None), 'method', '')),
            'family': type(edge.copula).__name__,
            'status': status,
        }
        if reason is not None:
            record['reason'] = reason
        if r_before is not None:
            r_before = np.asarray(r_before, dtype=np.float64)
            record['r_before_mean'] = float(np.mean(r_before))
        if r_after is not None:
            r_after = np.asarray(r_after, dtype=np.float64)
            record['r_after_mean'] = float(np.mean(r_after))
        return record

    def _apply_given_only_dynamic_updates_ordered(
            self, n, r_all, given, start_col, matrix, pair_copulas, edge_map,
            train_pseudo, horizon, rng, predictive_r_mode, state_cache=None):
        d = self.d
        M = matrix
        updated = {
            key: value.copy()
            for key, value in r_all.items()
        }
        pseudo_obs = {}
        diagnostics = {
            'mode': 'given_only',
            'updated_edges': [],
            'skipped_edges': [],
        }

        last_var = int(M[0, d - 1])
        if d - 1 >= start_col:
            pseudo_obs[(last_var, frozenset())] = np.full(
                n, given[last_var], dtype=np.float64)
        else:
            return updated, diagnostics

        for col in range(d - 2, start_col - 1, -1):
            leaf = int(M[d - 1 - col, col])
            top_tree = d - 2 - col
            pseudo_obs[(leaf, frozenset())] = np.full(
                n, given[leaf], dtype=np.float64)
            for t in range(top_tree + 1):
                key = (t, col)
                row = d - 2 - col - t
                partner = int(M[row, col])
                conditioning = frozenset(
                    int(M[r, col])
                    for r in range(row + 1, d - 1 - col)
                )
                next_leaf_cond = conditioning | {partner}
                next_partner_cond = conditioning | {leaf}
                edge = pair_copulas[key]
                leaf_val = pseudo_obs[(leaf, conditioning)]
                partner_val = pseudo_obs[(partner, conditioning)]
                u_pair = self._edge_pair_from_pseudo_map(
                    key, pseudo_obs, edge_map)

                r_before = updated[key]
                r_new = self._dynamic_edge_update_from_observation(
                    key,
                    edge,
                    r_before,
                    u_pair,
                    edge_map,
                    train_pseudo,
                    horizon,
                    rng,
                    predictive_r_mode,
                    state_cache=state_cache,
                )
                if r_new is not None:
                    updated[key] = np.asarray(r_new, dtype=np.float64)
                    diagnostics['updated_edges'].append(
                        self._dynamic_update_record(
                            key, edge, edge_map, r_before, updated[key],
                            'updated'))
                elif _is_gas_edge(edge) or isinstance(
                        getattr(edge, 'fit_result', None), LatentResult):
                    reason = self._dynamic_edge_skip_reason(
                        edge, train_pseudo, horizon)
                    diagnostics['skipped_edges'].append(
                        self._dynamic_update_record(
                            key, edge, edge_map, r_before, None,
                            'skipped', reason=reason))

                r = updated[key]
                pseudo_obs[(leaf, next_leaf_cond)] = _clip(
                    _edge_h(edge, leaf_val, partner_val, config={'r': r})
                )
                pseudo_obs[(partner, next_partner_cond)] = _clip(
                    _edge_h(edge, partner_val, leaf_val, config={'r': r})
                )

        return updated, diagnostics

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
                             train_pseudo, horizon, rng,
                             predictive_r_mode=None, state_cache=None):
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
                predictive_r_mode=predictive_r_mode,
                state_cache=state_cache,
                cache_key=('predictive_state', key, edge_horizon),
            )
        return r_all

    def _sample_dag_given_with_r(self, n, r_all, rng, given, plan, pair_copulas):
        missing = sorted(set(plan.edges_used) - set(r_all))
        if missing:
            raise KeyError(
                "RVineCopula._sample_dag_given_with_r: missing predicted "
                f"parameters for DAG edges {missing}"
            )
        r_payload = {
            key: {
                'edge': pair_copulas[key],
                'r': r_all[key],
            }
            for key in plan.edges_used
        }
        return execute_conditional_plan(plan, r_payload, given, n, rng)

    def _log_pdf_rows_with_r(self, u, r_all, pair_copulas=None, edge_map=None):
        pair_copulas = self.pair_copulas if pair_copulas is None else pair_copulas
        edge_map = self._edge_map if edge_map is None else edge_map
        pseudo_obs = {
            (i, frozenset()): u[:, i].copy()
            for i in range(self.d)
        }
        logp = np.zeros(len(u), dtype=np.float64)
        for t, level in enumerate(self.trees):
            for orig_idx, (conditioned, conditioning) in enumerate(level):
                key = self._matrix_key_from_map(t, orig_idx, edge_map)
                pc = pair_copulas[key]
                v1, v2 = sorted(conditioned)
                u1 = _clip(pseudo_obs[(v1, conditioning)])
                u2 = _clip(pseudo_obs[(v2, conditioning)])
                r = np.asarray(r_all[key], dtype=np.float64)
                # MLE/GAS may provide a scalar parameter path; SCAR-TM
                # predictive sampling normally provides one parameter per row.
                if len(r) == 1 and len(u) != 1:
                    r = np.full(len(u), float(r[0]), dtype=np.float64)
                elif len(r) != len(u):
                    raise ValueError(
                        "RVineCopula._log_pdf_rows_with_r: parameter path "
                        f"for edge {key} has length {len(r)}, expected 1 "
                        f"or {len(u)}"
                    )
                if not isinstance(pc.copula, IndependentCopula):
                    logp += pc.copula.log_pdf(u1, u2, r)
                if t < self.d - 2:
                    pseudo_obs[(v2, conditioning | {v1})] = _clip(
                        _edge_h(pc, u2, u1, config={'r': r}))
                    pseudo_obs[(v1, conditioning | {v2})] = _clip(
                        _edge_h(pc, u1, u2, config={'r': r}))
        return logp

    def _matrix_key_from_map(self, tree_level, orig_idx, edge_map):
        for key, mapped_idx in edge_map.items():
            if key[0] == tree_level and mapped_idx == orig_idx:
                return key
        raise KeyError((tree_level, orig_idx))

    def _sample_arbitrary_given_mcmc(
            self, n, r_all, rng, given, initial=None, n_steps=None,
            burnin_steps=None):
        free_vars = [var for var in range(self.d) if var not in given]
        if not free_vars:
            out = np.empty((n, self.d), dtype=np.float64)
            for var in range(self.d):
                out[:, var] = given[var]
            return out, {
                'accepted': {},
                'proposed': {},
                'acceptance_rate': {},
                'acceptance_min': None,
                'acceptance_mean': None,
                'acceptance_max': None,
                'low_acceptance_warning': False,
                'n_steps': 0,
                'burnin_steps': 0,
                'total_steps': 0,
            }

        if initial is None:
            current = rng.uniform(_EPS, 1.0 - _EPS, size=(n, self.d))
            for var, value in given.items():
                current[:, var] = value
        else:
            current = np.asarray(initial, dtype=np.float64).copy()
            for var, value in given.items():
                current[:, var] = value

        current_logp = self._log_pdf_rows_with_r(current, r_all)
        n_steps = (
            max(80, 30 * len(free_vars))
            if n_steps is None else int(n_steps)
        )
        burnin_steps = (
            max(40, 10 * len(free_vars))
            if burnin_steps is None else int(burnin_steps)
        )
        total_steps = burnin_steps + n_steps
        accepted = {int(var): 0 for var in free_vars}
        proposed = {int(var): 0 for var in free_vars}

        for step_idx in range(total_steps):
            var = free_vars[step_idx % len(free_vars)]
            proposal = current.copy()
            proposal[:, var] = rng.uniform(_EPS, 1.0 - _EPS, size=n)
            proposal_logp = self._log_pdf_rows_with_r(proposal, r_all)
            log_alpha = proposal_logp - current_logp
            accept = np.log(rng.uniform(_EPS, 1.0, size=n)) < log_alpha
            if np.any(accept):
                current[accept, var] = proposal[accept, var]
                current_logp[accept] = proposal_logp[accept]
            accepted[int(var)] += int(np.sum(accept))
            proposed[int(var)] += int(n)

        rates = {
            var: accepted[var] / proposed[var] if proposed[var] else 0.0
            for var in free_vars
        }
        rate_values = np.array(list(rates.values()), dtype=np.float64)
        has_proposals = any(proposed[var] > 0 for var in free_vars)
        acceptance_min = float(np.min(rate_values)) if has_proposals else None
        acceptance_mean = float(np.mean(rate_values)) if has_proposals else None
        acceptance_max = float(np.max(rate_values)) if has_proposals else None
        low_acceptance_warning = (
            bool(has_proposals)
            and acceptance_min is not None
            and acceptance_min < 0.02
        )
        return _clip(current), {
            'accepted': accepted,
            'proposed': proposed,
            'acceptance_rate': rates,
            'acceptance_min': acceptance_min,
            'acceptance_mean': acceptance_mean,
            'acceptance_max': acceptance_max,
            'low_acceptance_warning': low_acceptance_warning,
            'n_steps': n_steps,
            'burnin_steps': burnin_steps,
            'total_steps': total_steps,
        }

    def predict(self, n, u_train=None, horizon='next', rng=None, given=None,
                u=None, predictive_r_mode=None, dynamic_conditioning='ignore',
                predict_config=None, return_diagnostics=False,
                mcmc_steps=None, mcmc_burnin=None):
        """Draw predictive samples from the fitted R-vine.

        ``given`` fixes variables in pseudo-observation space. Conditional
        sampling is supported when the fixed variables can be placed at the
        end of the R-vine variable order, read from the anti-diagonal of the
        natural-order matrix. This can be true in the fitted matrix itself or
        after rebuilding the same fitted tree structure into an equivalent
        natural-order matrix with those variables last.

        When the model was fitted with ``given_vars=...``, that exact
        target set is treated as the supported conditioning contract for the
        current exact sampler. Other ``given`` patterns still follow the usual
        best-effort check for whether the fixed variables can be placed at the
        end of the R-vine variable order.

        For GAS edges, ``horizon='current'`` uses Psi(f_T) and ``'next'`` uses
        one score update to Psi(f_{T+1}). For SCAR-TM edges, the same argument
        selects p(x_T | data) or p(x_{T+1} | data) before sampling the
        posterior mixture path.

        ``dynamic_conditioning='given_only'`` additionally lets fixed suffix
        observations update dynamic edge states when an edge pair is fully
        determined before any free variable is sampled. Per-strategy behaviour:

        * **GAS** updates apply only with ``horizon='current'``; with
          ``'next'`` they are skipped, because another score update would
          advance the filter again rather than condition the same predictive
          state.
        * **SCAR-TM-OU** updates apply at both horizons via Bayes-reweighting
          of the transfer-matrix predictive grid by the observation
          likelihood ``p(u_pair | r=Psi(z_grid))``.
        * **MLE** edges have no dynamic state, so ``'given_only'`` is a no-op.

        Parameters
        ----------
        n : int
            Number of predictive samples to draw.
        u_train : (T, d) array-like or None, default None
            Reference pseudo-observations used to build current predictive
            edge states. If ``None``, uses the data stored by the last
            ``fit`` call.
        horizon : {'current', 'next'}, default 'next'
            Predictive state timing for dynamic edges. Static MLE edges ignore
            this option.
        rng : numpy.random.Generator or None, default None
            Random number generator. If ``None``, a fresh default generator is
            created.
        given : dict[int, float] or None, default None
            Fixed variable values in pseudo-observation space, keyed by
            zero-based variable index. Values must be in ``(0, 1)``.
        u : (T, d) array-like or None, default None
            Deprecated alias for ``u_train``. Pass only one of ``u`` and
            ``u_train``.
        predictive_r_mode : {'grid', 'histogram'} or None, default None
            SCAR-TM predictive parameter sampling mode. ``None`` uses the
            strategy default. MLE and GAS edges ignore this option.
        dynamic_conditioning : {'ignore', 'given_only'}, default 'ignore'
            Whether fixed suffix observations may update eligible dynamic edge
            states before sampling free variables.
        predict_config : PredictConfig or None, default None
            Optional bundled prediction options. Explicit non-default
            arguments passed to this method override the corresponding fields.
        return_diagnostics : bool, default False
            If True, return ``(samples, diagnostics)`` instead of only
            samples.
        mcmc_steps : int or None, default None
            Number of Metropolis-within-Gibbs sampling sweeps used after the
            DAG initializer for arbitrary non-suffix ``given`` patterns. If
            ``None``, a dimension-based default is used.
        mcmc_burnin : int or None, default None
            Number of burn-in sweeps for the arbitrary-``given`` MCMC fallback.
            If ``None``, a dimension-based default is used.

        Returns
        -------
        samples : (n, d) ndarray
            Predictive pseudo-observations.
        samples, diagnostics : tuple
            Returned when ``return_diagnostics=True``. Diagnostics include the
            conditioning method, suffix position, dynamic edge updates and
            MCMC acceptance information when applicable.
        """
        self._require_fit()
        if u is not None:
            if u_train is not None:
                raise ValueError("Pass only one of u_train or u")
            u_train = u
        if not isinstance(n, (int, np.integer)) or n <= 0:
            raise ValueError(f"RVineCopula.predict: n must be positive int, got {n!r}")
        if predict_config is None:
            pcfg = PredictConfig(
                given=given,
                horizon=horizon,
                predictive_r_mode=predictive_r_mode,
                dynamic_conditioning=dynamic_conditioning,
                return_diagnostics=return_diagnostics,
                mcmc_steps=mcmc_steps,
                mcmc_burnin=mcmc_burnin,
            ).validated()
        elif isinstance(predict_config, PredictConfig):
            pcfg = predict_config.validated()
            if given is not None:
                pcfg = pcfg.replace(given=given)
            if str(horizon).lower() != 'next':
                pcfg = pcfg.replace(horizon=horizon)
            if predictive_r_mode is not None:
                pcfg = pcfg.replace(predictive_r_mode=predictive_r_mode)
            if str(dynamic_conditioning).lower() != 'ignore':
                pcfg = pcfg.replace(dynamic_conditioning=dynamic_conditioning)
            if return_diagnostics:
                pcfg = pcfg.replace(return_diagnostics=True)
            if mcmc_steps is not None:
                pcfg = pcfg.replace(mcmc_steps=mcmc_steps)
            if mcmc_burnin is not None:
                pcfg = pcfg.replace(mcmc_burnin=mcmc_burnin)
        else:
            raise TypeError("predict_config must be PredictConfig or None")
        horizon = pcfg.horizon
        dynamic_conditioning = pcfg.dynamic_conditioning
        predictive_r_mode = pcfg.predictive_r_mode
        if rng is None:
            rng = np.random.default_rng()

        n = int(n)
        given = validate_rvine_given(pcfg.given, self.d)
        target_given = (
            bool(given)
            and self._target_given_vars
            and tuple(sorted(given)) == self._target_given_vars
        )
        if target_given and not self._conditional_fit_supported:
            raise ValueError(
                "RVineCopula.predict: model was fitted with "
                f"given_vars={list(self._target_given_vars)}, "
                "but the fitted structure does not support exact conditional "
                "sampling for that target set"
            )
        if len(given) == self.d:
            out = np.empty((n, self.d), dtype=np.float64)
            for i in range(self.d):
                out[:, i] = given[i]
            if pcfg.return_diagnostics:
                return out, {
                    'given': dict(given),
                    'dynamic_conditioning': dynamic_conditioning,
                    'suffix_start_col': 0,
                    'updated_edges': [],
                    'skipped_edges': [],
                    'all_variables_given': True,
                }
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

        diagnostics = {
            'given': dict(given),
            'dynamic_conditioning': dynamic_conditioning,
            'suffix_start_col': suffix_start_col,
            'matrix_rebuilt': (
                suffix_state is not None
                and not np.array_equal(matrix, self.matrix)
            ),
            'conditional_method': 'unconditional' if not given else 'suffix',
            'updated_edges': [],
            'skipped_edges': [],
        }
        state_cache = {}

        if given:
            if suffix_start_col is None:
                dag = build_runtime_rvine_dag(self.matrix, self._edge_map)
                plan = plan_conditional_sample(dag, given, self.d)
                r_all = self._predict_r_for_edges(
                    self.pair_copulas.keys(),
                    self.pair_copulas,
                    self._edge_map,
                    n,
                    train_pseudo,
                    horizon,
                    rng,
                    predictive_r_mode=predictive_r_mode,
                    state_cache=state_cache,
                )
                initial = self._sample_dag_given_with_r(
                    n,
                    r_all,
                    rng,
                    given,
                    plan,
                    self.pair_copulas,
                )
                samples, mcmc_diag = self._sample_arbitrary_given_mcmc(
                    n,
                    r_all,
                    rng,
                    given,
                    initial=initial,
                    n_steps=pcfg.mcmc_steps,
                    burnin_steps=pcfg.mcmc_burnin,
                )
                if pcfg.return_diagnostics:
                    diagnostics['conditional_method'] = 'dag_mcmc'
                    diagnostics['dag_steps'] = tuple(dict(step) for step in plan)
                    diagnostics['dag_edges_used'] = tuple(plan.edges_used)
                    diagnostics['mcmc'] = mcmc_diag
                    return samples, diagnostics
                return samples
            r_all = self._predict_r_for_edges(
                pair_copulas.keys(),
                pair_copulas,
                edge_map,
                n,
                train_pseudo,
                horizon,
                rng,
                predictive_r_mode=predictive_r_mode,
                state_cache=state_cache,
            )
            if dynamic_conditioning == 'given_only':
                r_all, dynamic_diag = self._apply_given_only_dynamic_updates_ordered(
                    n,
                    r_all,
                    given,
                    suffix_start_col,
                    matrix=matrix,
                    pair_copulas=pair_copulas,
                    edge_map=edge_map,
                    train_pseudo=train_pseudo,
                    horizon=horizon,
                    rng=rng,
                    predictive_r_mode=predictive_r_mode,
                    state_cache=state_cache,
                )
                diagnostics['updated_edges'] = dynamic_diag['updated_edges']
                diagnostics['skipped_edges'] = dynamic_diag['skipped_edges']
            samples = self._sample_suffix_given_with_r(
                n,
                r_all,
                rng,
                given,
                suffix_start_col,
                matrix=matrix,
                pair_copulas=pair_copulas,
            )
            if pcfg.return_diagnostics:
                return samples, diagnostics
            return samples
        r_all = self._predict_r_for_edges(
            pair_copulas.keys(),
            pair_copulas,
            edge_map,
            n,
            train_pseudo,
            horizon,
            rng,
            predictive_r_mode=predictive_r_mode,
            state_cache=state_cache,
        )
        samples = self._sample_with_r(n, r_all, rng)
        if pcfg.return_diagnostics:
            return samples, diagnostics
        return samples
