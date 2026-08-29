"""
vine.vine - generic vine copula with strategy-backed edge models.

Layout
------
Build path:
    u -> select_rvine_structure (Dissmann) -> VineStructureSelection
                                                  │
                                                  ▼
                       fit_vine_edges(canonical decoded trees)
                                                  │
                                                  ▼
                     ``VineCopula`` stores M, trees, pair_copulas (by (t, col))

The R-vine *matrix* is the single source of truth for structure; pair
copulas are stored in a dict keyed by matrix position ``(tree, col)``.
The matrix follows the zero-based natural-order runtime convention derived
from Czado (2019, Alg. 5.4); non-zero entries fill the upper-left anti-triangle,
the anti-diagonal ``M[d-1-col, col]`` holds the leaf peeled at column
``col``, and tree-``t`` edges at column ``col`` have their "other"
endpoint at row ``d-2-col-t``.

Current scope
-------------
Structure selection is Dissmann-based. Edge fitting delegates to the strategy
registry. Unconditional ``sample`` and predictive ``predict`` use the
natural-order matrix.

Usage
-----
    from pyscarcopula import VineCopula

    vine = VineCopula().fit(u)
    print(vine)                       # summary
    total_ll = vine.log_likelihood()  # cached fitted total
    new_ll = vine.log_likelihood(u_new)  # re-evaluate on new data
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import OptimizeResult

from pyscarcopula._utils import pobs
from pyscarcopula.numerical._arrays import (
    as_float64_array,
    as_pseudo_observation_array,
    validate_positive_int,
)
from pyscarcopula._types import (
    PredictConfig,
)
from pyscarcopula._native import _extension as _cpp_extension, statistics
from pyscarcopula._native.errors import NativeUnsupported
from pyscarcopula._native.registry import registry_entry_for
from pyscarcopula.vine._conditional_rvine import (
    validate_rvine_given_vars,
    validate_rvine_given,
)
from pyscarcopula.vine._rvine_dissmann import (
    select_rvine,
    select_rvine_structure,
)
from pyscarcopula.vine._vine_fit import fit_vine_edges
from pyscarcopula.vine._structure import (
    RVineMatrix,
    cvine_structure,
    dvine_structure,
)
from pyscarcopula.vine._rvine_edges import (
    _edge_h,
    _edge_h_inverse,
    _edge_h_inverse_for_variables,
    _edge_initial_model_state,
    _edge_requires_stepwise_sample,
    _edge_r_for_sample,
    _edge_r_for_predict,
    _edge_state_r,
    _edge_update_model_state,
    _strategy_for_result,
)
from pyscarcopula.vine._rvine_dag import (
    build_runtime_rvine_dag,
    plan_conditional_sample,
)
from pyscarcopula.vine._rvine_matrix_builder import (
    build_rvine_matrix_with_edge_map,
)
from pyscarcopula.vine._rvine_sampling_plan import (
    build_rvine_sampling_plan,
    rvine_sampling_edge_keys,
)
from pyscarcopula.vine._edge_adapter import (
    edge_copula,
    edge_has_dynamic_params,
    edge_is_independent,
    edge_param,
    edge_result,
)
from pyscarcopula.vine._dynamic_conditioning import (
    dynamic_edge_skip_reason,
    dynamic_edge_update_from_observation,
    dynamic_skip_records,
    dynamic_update_record,
    normalize_predict_horizon,
    predictive_given_update_r,
    predictive_state_cache_key,
)
from pyscarcopula.vine._rvine_summary import format_rvine_summary
from pyscarcopula.vine._helpers import (
    _clip_unit,
    _open_unit_uniform,
    _prepared_open_unit_draws,
)
from pyscarcopula.vine._rvine_suffix import (
    build_suffix_conditional_plan,
    edge_pair_from_pseudo_map,
    given_suffix_start_col,
    suffix_sampling_state,
)

_normalize_predict_horizon = normalize_predict_horizon
_predictive_state_cache_key = predictive_state_cache_key

_DEFAULT_STATIC_SAMPLE_BATCH_ROWS = 8192


def _as_rvine_observations(
        data, *, operation, expected_dimension=None, to_pobs=False):
    """Validate RVine observations before selection or numerical work."""
    u = as_float64_array(data, name="data")
    if u.ndim != 2:
        raise ValueError(
            f"VineCopula.{operation}: data must be 2D, got shape {u.shape}")
    if u.shape[0] == 0:
        raise ValueError(
            f"VineCopula.{operation}: data must contain at least one row")
    if expected_dimension is not None and u.shape[1] != expected_dimension:
        raise ValueError(
            f"VineCopula.{operation}: data must be "
            f"(T, {expected_dimension}), got {u.shape}")
    if to_pobs:
        if not np.all(np.isfinite(u)):
            raise ValueError("data must contain only finite values")
        return pobs(u)
    return as_pseudo_observation_array(
        u, name="data", allow_boundary=False)


def _validated_memory_budget(memory_budget_bytes):
    if memory_budget_bytes is None:
        return None
    if (
            isinstance(memory_budget_bytes, (bool, np.bool_))
            or not isinstance(memory_budget_bytes, (int, np.integer))):
        raise TypeError("memory_budget_bytes must be an integer or None")
    if int(memory_budget_bytes) < 0:
        raise ValueError("memory_budget_bytes must be non-negative")
    return int(memory_budget_bytes)


def _freeze_trees(trees):
    """Own a vine tree representation without retaining mutable containers."""
    if trees is None:
        return None
    return tuple(
        tuple(
            (frozenset(conditioned), frozenset(conditioning))
            for conditioned, conditioning in level
        )
        for level in trees
    )


def _copy_trees(trees):
    """Return the historical nested-list tree representation defensively."""
    if trees is None:
        return None
    return [
        [
            (frozenset(conditioned), frozenset(conditioning))
            for conditioned, conditioning in level
        ]
        for level in trees
    ]


def _edge_identity(edge):
    conditioned, conditioning = edge
    return frozenset(conditioned), frozenset(conditioning)


def _canonicalize_fitted_levels(structure, source_trees, fitted_levels):
    """Reindex fitted edges from a construction order to canonical tree order."""
    fitted_by_identity = {
        _edge_identity(edge): fitted_levels[tree][edge_index]
        for tree, level in enumerate(source_trees)
        for edge_index, edge in enumerate(level)
    }
    canonical_trees = structure.to_trees()
    canonical_levels = [
        [fitted_by_identity[_edge_identity(edge)] for edge in level]
        for level in canonical_trees
    ]
    return canonical_trees, canonical_levels


def _structure_kinds(structure):
    """Return all standard vine kinds compatible with a fixed structure."""
    trees = structure.to_trees()
    is_cvine = all(
        len(level) <= 1
        or bool(set.intersection(*(
            set(conditioned) for conditioned, _ in level
        )))
        for level in trees
    )
    degrees = {}
    for conditioned, _ in trees[0]:
        for variable in conditioned:
            degrees[variable] = degrees.get(variable, 0) + 1
    is_dvine = max(degrees.values(), default=0) <= 2
    kinds = set()
    if is_cvine:
        kinds.add("cvine")
    if is_dvine:
        kinds.add("dvine")
    if not kinds:
        kinds.add("rvine")
    if structure.d == 2:
        kinds.add("rvine")
    return frozenset(kinds)


class VineCopula:
    """Generic vine copula with automatic or fixed structure.

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
    transform_type : str, default 'softplus'
        Parameter transform passed through to candidate copulas.
    structure : RVineMatrix or None, default None
        Fixed regular-vine structure.  If omitted, the structure is selected
        automatically with the Dissmann procedure on every fit.
    vine_type : {'cvine', 'dvine', 'rvine'} or None, default None
        Explicit structural mode for integrations such as GoF. Factories set
        this value automatically. With a direct fixed ``structure``, ``None``
        derives the mode from the structure.

    Attributes (after ``fit``)
    --------------------------
    d : int
        Number of variables.
    matrix : (d, d) int ndarray
        Zero-based natural-order runtime R-vine matrix derived from Czado
        (2019 Alg. 5.4). ``pyvinecopulib`` uses the opposite tree-level order
        above the anti-diagonal and one-based labels. Non-zero entries occupy
        the upper-left
        anti-triangle; the anti-diagonal ``M[d-1-col, col]`` is the
        leaf peeled at column ``col``.
    trees : list of ``(d - 1)`` lists
        ``trees[t][i]`` = ``(conditioned_frozenset, conditioning_frozenset)``
        in the canonical order decoded from ``structure``. Each access returns
        a defensive nested-list copy.
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
        transform_type='softplus',
        structure=None,
        vine_type=None,
    ):
        """Initialize a configurable C-, D-, or R-vine model."""
        from pyscarcopula.vine._selection import validate_pair_candidates
        validate_pair_candidates(candidates)
        if structure is not None and not isinstance(structure, RVineMatrix):
            raise TypeError(
                "structure must be an RVineMatrix or None, "
                f"got {type(structure).__name__}")
        if vine_type is not None:
            vine_type = str(vine_type).lower()
            if vine_type not in {"cvine", "dvine", "rvine"}:
                raise ValueError(
                    "vine_type must be 'cvine', 'dvine', 'rvine' or None, "
                    f"got {vine_type!r}")
            if structure is None and vine_type != "rvine":
                raise ValueError(
                    f"vine_type={vine_type!r} requires a fixed structure")
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
        self._configured_structure = (
            None if structure is None else RVineMatrix(structure.matrix))
        if (
                self._configured_structure is not None
                and vine_type is not None
                and vine_type not in _structure_kinds(
                    self._configured_structure)):
            raise ValueError(
                f"vine_type={vine_type!r} does not match the fixed "
                "RVineMatrix structure")
        self._configured_vine_type = vine_type
        self._structure = None

        self.d = None
        self._natural_order_matrix = None
        self._trees = None
        self.pair_copulas = None
        self._edge_map = None  # (t, col) -> orig_idx in trees[t]
        self._orig_edge_key = None  # (t, orig_idx) -> (t, col)
        self._T = None
        self._log_likelihood = None
        self.fit_result = None
        self.method = None
        self._target_given_vars = ()
        self._conditional_fit_supported = None
        self._conditional_mode = None
        self._fit_diagnostics = None
        self._suffix_state_cache = {}
        self._predict_history_cache = {}
        self._native_rvine_cache = {}
        self._native_rvine_generation = 0

    @classmethod
    def cvine(
            cls,
            d: int,
            order: Sequence[int] | None = None,
            **kwargs: Any) -> VineCopula:
        """Create a generic vine configured with a fixed C-vine structure."""
        return cls(
            structure=cvine_structure(d, order),
            vine_type="cvine",
            **kwargs,
        )

    @classmethod
    def dvine(
            cls,
            d: int,
            order: Sequence[int] | None = None,
            **kwargs: Any) -> VineCopula:
        """Create a generic vine configured with a fixed D-vine structure."""
        return cls(
            structure=dvine_structure(d, order),
            vine_type="dvine",
            **kwargs,
        )

    @classmethod
    def rvine(cls, **kwargs: Any) -> VineCopula:
        """Create a generic vine with automatic R-vine selection."""
        return cls(vine_type="rvine", **kwargs)

    def __getstate__(self):
        """Return persistent model state without transient prediction caches."""
        state = self.__dict__.copy()
        state['trees'] = _copy_trees(state.pop('_trees', None))
        state.pop('_predict_history_cache', None)
        state.pop('_suffix_state_cache', None)
        state.pop('_native_rvine_cache', None)
        state.pop('_native_rvine_generation', None)
        state['_structure_source'] = self.structure_source
        return state

    def __setstate__(self, state):
        """Validate and restore persisted generic vine state."""
        state = dict(state)
        required_fields = {
            '_configured_structure',
            '_configured_vine_type',
            '_edge_map',
            '_natural_order_matrix',
            '_orig_edge_key',
            '_structure_source',
        }
        missing_fields = required_fields.difference(state)
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(
                f"Persisted VineCopula state is missing fields: {missing}")
        persisted_source = state.pop('_structure_source')

        configured_type = state['_configured_vine_type']
        if configured_type not in {None, "cvine", "dvine", "rvine"}:
            raise ValueError(
                "Persisted VineCopula has invalid vine_type "
                f"{configured_type!r}")

        def normalized_structure(value, *, name):
            if value is None:
                return None
            if not isinstance(value, RVineMatrix):
                raise TypeError(
                    f"Persisted VineCopula {name} must be an RVineMatrix")
            matrix = value.matrix
            if value.d != matrix.shape[0]:
                raise ValueError(
                    f"Persisted VineCopula {name} dimension is inconsistent")
            try:
                return RVineMatrix(matrix)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Persisted VineCopula {name} is invalid") from exc

        configured_structure = normalized_structure(
            state['_configured_structure'],
            name="configured structure",
        )
        state['_configured_structure'] = configured_structure
        expected_source = (
            "fixed" if configured_structure is not None else "auto")
        if persisted_source != expected_source:
            raise ValueError(
                "Persisted VineCopula structure source does not match "
                "configured structure")

        d = state.get('d')
        trees = state.get('trees', state.get('_trees'))
        natural_matrix = state.get('_natural_order_matrix')
        serialized_structure = normalized_structure(
            state.get('_structure'),
            name="fitted structure",
        )
        if d is None and trees is None and natural_matrix is None:
            if serialized_structure is not None:
                raise ValueError(
                    "Persisted unfitted VineCopula contains fitted structure")
            state['_structure'] = None
        elif d is None or trees is None or natural_matrix is None:
            raise ValueError(
                "Persisted VineCopula fitted structure state is incomplete")
        else:
            from pyscarcopula.vine._rvine_matrix_builder import (
                build_rvine_matrix_with_edge_map,
            )

            canonical_structure = RVineMatrix.from_trees(d, trees)
            expected_matrix, expected_edge_map = (
                build_rvine_matrix_with_edge_map(
                    d, trees, validate=False))
            try:
                natural_structure = RVineMatrix.from_natural_order(
                    natural_matrix)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Persisted VineCopula natural-order matrix is invalid"
                ) from exc
            if (
                    not np.array_equal(natural_matrix, expected_matrix)
                    or natural_structure != canonical_structure):
                raise ValueError(
                    "Persisted VineCopula natural-order matrix does not "
                    "match trees")
            if (
                    serialized_structure is not None
                    and serialized_structure != canonical_structure):
                raise ValueError(
                    "Persisted VineCopula fitted structure does not "
                    "match trees")
            if (
                    configured_structure is not None
                    and configured_structure != canonical_structure):
                raise ValueError(
                    "Persisted VineCopula fixed structure does not "
                    "match fitted trees")

            expected_edge_map = dict(expected_edge_map)
            stored_edge_map = state['_edge_map']
            if dict(stored_edge_map) != expected_edge_map:
                raise ValueError(
                    "Persisted VineCopula edge map does not match trees")
            expected_orig_edge_key = {
                (tree, orig_index): (tree, column)
                for (tree, column), orig_index
                in expected_edge_map.items()
            }
            stored_orig_edge_key = state['_orig_edge_key']
            if dict(stored_orig_edge_key) != expected_orig_edge_key:
                raise ValueError(
                    "Persisted VineCopula reverse edge map does not "
                    "match trees")
            pair_copulas = state.get('pair_copulas')
            if (
                    pair_copulas is None
                    or set(pair_copulas) != set(expected_edge_map)):
                raise ValueError(
                    "Persisted VineCopula pair copulas do not match "
                    "fitted structure")

            state['_natural_order_matrix'] = expected_matrix.copy()
            state['_structure'] = canonical_structure
            state['_edge_map'] = expected_edge_map
            state['_orig_edge_key'] = expected_orig_edge_key

        if (
                configured_structure is not None
                and configured_type is not None
                and configured_type not in _structure_kinds(
                    configured_structure)):
            raise ValueError(
                "Persisted VineCopula vine_type does not match configured "
                "structure")

        last_u = state.get('_last_u')
        if last_u is not None:
            last_u = np.asarray(last_u, dtype=np.float64).copy()
            if d is not None and (
                    last_u.ndim != 2 or last_u.shape[1] != d):
                raise ValueError(
                    "Persisted VineCopula training data dimension does "
                    "not match fitted structure")
            last_u.flags.writeable = False
            state['_last_u'] = last_u

        state.pop('_suffix_state_cache', None)
        state.pop('_predict_history_cache', None)
        state.pop('_native_rvine_cache', None)
        state.pop('_native_rvine_generation', None)
        state.pop('trees', None)
        state['_trees'] = _freeze_trees(trees)
        self.__dict__.update(state)
        self._suffix_state_cache = {}
        self._predict_history_cache = {}
        self._native_rvine_cache = {}
        self._native_rvine_generation = 0

    # Fit

    def fit(
            self,
            data: Any,
            method: str = 'mle',
            *,
            to_pobs: bool = False,
            copulas: Any = None,
            config: Any = None,
            given_vars: Sequence[int] | None = None,
            conditional_strict: bool = True,
            conditional_mode: str = 'suffix',
            **kwargs: Any) -> VineCopula:
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
            the canonical order returned by ``structure.to_trees()`` for each
            tree. If ``None``, the best
            family is selected for each edge from the candidate pool. Use
            ``candidates=`` on the constructor for automatic family pools;
            ``copulas=`` is for pre-specified edge family/rotation specs, not
            copula instances.
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
        **kwargs
            Supported structure options include ``truncation_level``,
            ``truncation_fill``, ``threshold``, ``min_edge_logL``,
            ``transform_type``, ``structure_search``, ``beam_width`` and
            ``dynamic_failure_policy`` (``'fallback'``, ``'keep'`` or
            ``'raise'``).
            Remaining keyword arguments are forwarded to the selected
            pair-copula strategy. Common strategy options include ``alpha0``,
            ``gtol``, ``ftol``, ``maxfun``, ``maxiter``, ``maxls``, ``eps``,
            ``verbose``, ``scaling``, ``K``, ``grid_range``, ``grid_method``,
            ``adaptive``, ``pts_per_sigma``, ``analytical_grad`` and
            ``smart_init``.

        Returns
        -------
        self : VineCopula
            Enables chained calls, e.g. ``VineCopula().fit(u).summary()``.
        """
        from pyscarcopula.strategy._base import validate_strategy_method
        method = validate_strategy_method(method)
        u = _as_rvine_observations(
            data, operation="fit", to_pobs=to_pobs)

        T, d = u.shape
        if d < 2:
            raise ValueError(f"VineCopula.fit: need d >= 2, got d={d}")
        from pyscarcopula.vine._selection import validate_fixed_copula_specs
        validate_fixed_copula_specs(copulas, d)
        given_vars = validate_rvine_given_vars(given_vars, d)
        conditional_mode = str(conditional_mode).lower()
        if conditional_mode != 'suffix':
            raise ValueError(
                f"conditional_mode must be 'suffix', got {conditional_mode!r}"
            )

        truncation_level = kwargs.pop('truncation_level', self.truncation_level)
        truncation_fill = kwargs.pop('truncation_fill', self.truncation_fill)
        threshold = kwargs.pop('threshold', self.threshold)
        dynamic_failure_policy = kwargs.pop(
            'dynamic_failure_policy', 'fallback')
        min_edge_logL = kwargs.pop('min_edge_logL', self.min_edge_logL)
        transform_type = kwargs.pop('transform_type', self.transform_type)
        structure_search_supplied = 'structure_search' in kwargs
        beam_width_supplied = 'beam_width' in kwargs
        structure_search = kwargs.pop('structure_search', 'beam')
        beam_width = kwargs.pop('beam_width', 4)

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

        if self._configured_structure is None:
            selection, working_fits = select_rvine_structure(
                u,
                _selector=select_rvine,
                _return_working_fits=True,
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
                structure_search=structure_search,
                beam_width=beam_width,
                **kwargs,
            )
            selected_structure = selection.structure
            selection_diagnostics = deepcopy(selection.diagnostics)
            trees_for_fit = _copy_trees(selection.trees)
        else:
            if self._configured_structure.d != d:
                raise ValueError(
                    "VineCopula.fit: fixed structure dimension "
                    f"{self._configured_structure.d} does not match "
                    f"data dimension {d}")
            if structure_search_supplied or beam_width_supplied:
                raise ValueError(
                    "structure_search and beam_width apply only to "
                    "automatic structure selection")
            selected_structure = RVineMatrix(
                self._configured_structure.matrix)
            trees_for_fit = selected_structure.to_trees()
            working_fits = None

        fixed_fit = fit_vine_edges(
            u,
            trees_for_fit,
            candidates=self.candidates,
            copulas=copulas,
            method=method,
            criterion=self.criterion,
            allow_rotations=self.allow_rotations,
            truncation_level=truncation_level,
            truncation_fill=truncation_fill,
            threshold=threshold,
            dynamic_failure_policy=dynamic_failure_policy,
            min_edge_logL=min_edge_logL,
            transform_type=transform_type,
            config=config,
            fit_kwargs=kwargs,
            _pre_fitted=working_fits,
        )
        trees_repr, fitted = _canonicalize_fitted_levels(
            selected_structure,
            trees_for_fit,
            fixed_fit.as_levels(),
        )
        if self._configured_structure is not None:
            selection_diagnostics = {
                'target_given_vars': tuple(given_vars),
                'selected_mode': 'fixed',
                'selected_index': None,
                'selected_candidate': {
                    'mode': 'fixed',
                    'exact_supported': True,
                    'dag_complete': True,
                    'fit_score': fixed_fit.log_likelihood,
                    'missing_base_vars': tuple(given_vars),
                    'reachable_base_vars': (),
                    'n_known_nodes': 0,
                    'n_steps': 0,
                    'n_inverse_steps': 0,
                },
                'candidates': (),
            }
        M, edge_map = build_rvine_matrix_with_edge_map(
            d, trees_repr, validate=False)

        pair_copulas = {}
        for (t, col), orig_idx in edge_map.items():
            pair_copulas[(t, col)] = fitted[t][orig_idx]
        orig_edge_key = {
            (t, orig_idx): (t, col)
            for (t, col), orig_idx in edge_map.items()
        }

        conditional_supported = True
        reject_reason = None
        if given_vars:
            probe = self.__class__()
            probe.d = d
            probe._natural_order_matrix = M.copy()
            probe._trees = _freeze_trees(trees_repr)
            probe.pair_copulas = pair_copulas
            probe._edge_map = dict(edge_map)
            probe._orig_edge_key = orig_edge_key
            conditional_supported = probe._suffix_sampling_state({
                var: 0.5 for var in given_vars
            }) is not None
            if self._configured_structure is not None:
                fixed_candidate = selection_diagnostics[
                    'selected_candidate']
                fixed_candidate['exact_supported'] = conditional_supported
                fixed_candidate['missing_base_vars'] = (
                    () if conditional_supported else tuple(given_vars))
                fixed_candidate['reachable_base_vars'] = (
                    tuple(given_vars) if conditional_supported else ())
            if not conditional_supported:
                reject_reason = 'unsupported_given_vars'
            fit_diagnostics = self._build_fit_diagnostics(
                given_vars,
                conditional_mode,
                conditional_strict,
                selection_diagnostics,
                conditional_supported,
                reject_reason=reject_reason,
            )
            if conditional_strict and not conditional_supported:
                if self.fit_result is None:
                    # An unfitted instance has no successful state to preserve;
                    # retain rejection diagnostics for the caller. A failed
                    # refit must leave the previous fitted snapshot untouched.
                    self._fit_diagnostics = deepcopy(fit_diagnostics)
                missing_base_vars = ()
                selected_mode = None
                if selection_diagnostics['selected_candidate'] is not None:
                    missing_base_vars = selection_diagnostics[
                        'selected_candidate']['missing_base_vars']
                    selected_mode = selection_diagnostics['selected_mode']
                raise ValueError(
                    "VineCopula.fit: could not construct an R-vine structure "
                    "supporting exact conditional sampling for "
                    f"given_vars={list(given_vars)}; "
                    f"selected_mode={selected_mode}; "
                    "missing_base_vars="
                    f"{list(missing_base_vars)}"
                )
        else:
            fit_diagnostics = self._build_fit_diagnostics(
                given_vars,
                None,
                conditional_strict,
                selection_diagnostics,
                conditional_supported,
                reject_reason=reject_reason,
            )

        self.d = d
        self._natural_order_matrix = M.copy()
        self._structure = RVineMatrix.from_trees(d, trees_repr)
        self._trees = _freeze_trees(trees_repr)
        self.pair_copulas = pair_copulas
        self._edge_map = dict(edge_map)
        self._orig_edge_key = orig_edge_key
        self._T = int(T)
        self._log_likelihood = statistics.sum_values(
            pc.log_likelihood for pc in pair_copulas.values()
        )
        self.method = method
        edge_fit_summary = self._build_edge_fit_summary(
            pair_copulas, requested_method=method)
        fit_diagnostics['edge_fits'] = deepcopy(edge_fit_summary)
        self._fit_diagnostics = fit_diagnostics
        total_nfev = statistics.sum_int64(
            int(getattr(edge_result(pc), 'nfev', 0) or 0)
            for pc in pair_copulas.values()
        )
        n_edges_total = len(pair_copulas)
        n_params = statistics.sum_int64(
            pc.n_params for pc in pair_copulas.values())
        self.fit_result = OptimizeResult()
        self.fit_result.log_likelihood = self._log_likelihood
        self.fit_result.method = method
        self.fit_result.name = (
            f"{self.structure_label} ({d}d, {n_edges_total} edges)")
        self.fit_result.nfev = total_nfev
        self.fit_result.n_params = n_params
        self.fit_result.parameter_count = n_params
        self.fit_result.diagnostics = deepcopy(edge_fit_summary)
        self.fit_result.actual_methods = dict(
            edge_fit_summary['actual_methods'])
        self.fit_result.fallback_count = edge_fit_summary['fallback_count']
        self.fit_result.fallback_edges = deepcopy(
            edge_fit_summary['fallback_edges'])
        self.fit_result.success = all(
            bool(getattr(edge_result(pc), 'success', True))
            for pc in pair_copulas.values()
        )
        # Prediction caches are valid only for this owned, immutable snapshot.
        # Explicit ``predict(..., u=...)`` inputs deliberately bypass them.
        self._last_u = u.copy()
        self._last_u.flags.writeable = False
        self._target_given_vars = given_vars
        self._conditional_fit_supported = conditional_supported
        self._conditional_mode = conditional_mode if given_vars else None
        self._suffix_state_cache = {}
        self._predict_history_cache = {}
        self._invalidate_native_rvine_cache()
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

    @staticmethod
    def _build_edge_fit_summary(pair_copulas, *, requested_method):
        actual_methods = {}
        family_counts = {}
        edge_records = []
        fallback_records = []
        dynamic_attempted_count = 0
        dynamic_success_count = 0
        selection_nfev_total = 0
        dynamic_attempted_nfev_total = 0
        fallback_discarded_nfev = 0

        for key in sorted(pair_copulas):
            edge = pair_copulas[key]
            result = edge_result(edge)
            provenance = dict(
                getattr(edge, 'fit_diagnostics', {}) or {})
            actual_method = str(
                getattr(result, 'method', None) or 'STATIC').upper()
            family = type(edge_copula(edge)).__name__
            actual_methods[actual_method] = (
                actual_methods.get(actual_method, 0) + 1)
            family_counts[family] = family_counts.get(family, 0) + 1

            dynamic_attempted = bool(
                provenance.get('dynamic_attempted', False))
            attempted_success = provenance.get('attempted_success')
            if dynamic_attempted:
                dynamic_attempted_count += 1
                dynamic_attempted_nfev_total += int(
                    provenance.get('attempted_nfev', 0) or 0)
                if attempted_success:
                    dynamic_success_count += 1
            selection_nfev_total += int(
                provenance.get('selection_nfev', 0) or 0)

            record = {
                'key': tuple(key),
                'family': family,
                'rotation': int(getattr(edge_copula(edge), 'rotate', 0)),
                'requested_method': str(
                    provenance.get('requested_method', requested_method)
                ).upper(),
                'actual_method': actual_method,
                'actual_success': bool(
                    getattr(result, 'success', True)),
                'actual_nfev': int(getattr(result, 'nfev', 0) or 0),
                'selection_nfev': int(
                    provenance.get('selection_nfev', 0) or 0),
                'dynamic_attempted': dynamic_attempted,
                'fallback_used': bool(
                    provenance.get('fallback_used', False)),
                'fallback_reason': provenance.get('fallback_reason'),
                'attempted_method': provenance.get('attempted_method'),
                'attempted_success': attempted_success,
                'attempted_nfev': int(
                    provenance.get('attempted_nfev', 0) or 0),
                'attempted_message': provenance.get('attempted_message'),
                'timings_ms': {
                    'selection': float(
                        provenance.get('selection_ms', 0.0) or 0.0),
                    'dynamic_fit': float(
                        provenance.get('dynamic_fit_ms', 0.0) or 0.0),
                    'total_fit': float(
                        provenance.get('fit_ms', 0.0) or 0.0),
                },
            }
            edge_records.append(record)
            if record['fallback_used']:
                fallback_records.append(record)
                fallback_discarded_nfev += record['attempted_nfev']

        return {
            'requested_method': str(requested_method).upper(),
            'edge_count': len(edge_records),
            'actual_methods': dict(sorted(actual_methods.items())),
            'family_counts': dict(sorted(family_counts.items())),
            'dynamic_attempted_count': dynamic_attempted_count,
            'dynamic_success_count': dynamic_success_count,
            'selection_nfev_total': selection_nfev_total,
            'dynamic_attempted_nfev_total': dynamic_attempted_nfev_total,
            'fallback_discarded_nfev': fallback_discarded_nfev,
            'fallback_count': len(fallback_records),
            'fallback_edges': tuple(fallback_records),
            'edges': tuple(edge_records),
        }

    # Convenience predicates

    def _require_fit(self):
        if self._natural_order_matrix is None:
            raise RuntimeError(
                "VineCopula: call fit(...) before accessing fitted state"
            )

    @property
    def fit_diagnostics(self) -> dict[str, Any] | None:
        """Fit-time structure-selection diagnostics, if available."""
        if self._fit_diagnostics is None:
            return None
        return deepcopy(self._fit_diagnostics)

    @property
    def structure(self) -> RVineMatrix | None:
        """Configured or fitted structure as a defensive ``RVineMatrix``."""
        value = (
            self._structure
            if self._structure is not None
            else self._configured_structure
        )
        return None if value is None else RVineMatrix(value.matrix)

    @property
    def trees(self) -> list[list[tuple[frozenset[int], frozenset[int]]]] | None:
        """Decoded semantic edges as a defensive nested-list copy."""
        return _copy_trees(self._trees)

    @property
    def structure_source(self) -> str:
        """Whether structure selection is automatic or fixed."""
        return "fixed" if self._configured_structure is not None else "auto"

    @property
    def structure_label(self) -> str:
        """Informational label derived from the current vine structure."""
        structure = self._structure or self._configured_structure
        if structure is None:
            return "regular vine"
        kinds = _structure_kinds(structure)
        if "cvine" in kinds:
            return "C-vine"
        if "dvine" in kinds:
            return "D-vine"
        return "regular vine"

    @property
    def vine_type(self) -> str:
        """Structural mode passed to type-sensitive integrations such as GoF."""
        if self._configured_vine_type is not None:
            return self._configured_vine_type
        if self.structure_source == "auto":
            return "rvine"
        return {
            "C-vine": "cvine",
            "D-vine": "dvine",
            "regular vine": "rvine",
        }[self.structure_label]

    @property
    def matrix(self) -> np.ndarray | None:
        """Compatibility copy of the fitted natural-order matrix."""
        if self._natural_order_matrix is None:
            return None
        return self._natural_order_matrix.copy()

    @matrix.setter
    def matrix(self, value: Any) -> None:
        """Set legacy hand-built runtime state using an owned matrix copy."""
        self._natural_order_matrix = (
            None if value is None else np.asarray(value, dtype=int).copy())
        self._invalidate_native_rvine_cache()

    @property
    def natural_order_matrix(self) -> np.ndarray:
        """Copy of the fitted natural-order R-vine matrix."""
        self._require_fit()
        return self._natural_order_matrix.copy()

    def to_rvine_matrix(self) -> RVineMatrix:
        """Return this fitted structure as lower-triangular ``RVineMatrix``."""
        self._require_fit()
        from pyscarcopula.vine._structure import RVineMatrix

        return RVineMatrix(self._structure.matrix)

    # Log-likelihood

    def save(
            self, path: str | Path, *, include_data: bool = True) -> None:
        """Save this fitted R-vine model to disk."""
        from pyscarcopula.io import save_model

        save_model(self, path, include_data=include_data)

    @classmethod
    def load(cls, path: str | Path) -> VineCopula:
        """Load a saved R-vine model from disk."""
        from pyscarcopula.io import load_model

        return load_model(path, expected_type=cls)

    def log_likelihood(
            self, data: Any = None, to_pobs: bool = False) -> float:
        """Total log-likelihood.

        With no argument returns the cached fitted log-likelihood. With an
        explicit ``data`` array, evaluates the complete fitted R-vine in the
        native traversal runtime.
        """
        registry_entry_for(self)
        self._require_fit()
        if data is None:
            return self._log_likelihood

        u = _as_rvine_observations(
            data,
            operation="log_likelihood",
            expected_dimension=self.d,
            to_pobs=to_pobs,
        )

        from pyscarcopula._native import vine as native_vine

        active_keys = native_vine.density_active_keys(
            self._trees, self._edge_map)
        if not native_vine.native_edges_supported(
                self.pair_copulas, active_keys):
            raise NativeUnsupported(
                "native R-vine likelihood requires exact registered "
                "built-in edge copulas")

        static_layout = native_vine.static_rosenblatt_parameter_layout(
            self.pair_copulas, active_keys)
        if static_layout is not None:
            parameter_paths, _parameter_sources = static_layout
            return statistics.sum_values(self._log_pdf_rows_with_r(
                u, parameter_paths))

        module = _cpp_extension.load()
        return float(native_vine.rosenblatt(
            module,
            self.pair_copulas,
            self.d,
            self._trees,
            self._edge_map,
            self.matrix,
            u,
            active_keys=active_keys,
            return_log_likelihood=True,
        ))

    def _matrix_key(self, tree_level, orig_idx):
        """Invert edge_map: (tree, orig_idx) -> (tree, col)."""
        lookup = self._ensure_orig_edge_key()
        try:
            return lookup[(tree_level, orig_idx)]
        except KeyError as exc:
            raise KeyError(
                "VineCopula: no matrix column for tree "
                f"{tree_level}, edge {orig_idx}"
            ) from exc

    def _ensure_orig_edge_key(self):
        lookup = getattr(self, '_orig_edge_key', None)
        if lookup is None:
            lookup = {
                (t, orig_idx): (t, col)
                for (t, col), orig_idx in self._edge_map.items()
            }
            self._orig_edge_key = lookup
        return lookup

    def _max_non_independent_tree_level(self):
        """Highest tree level that can affect static loglik/sample paths."""
        self._require_fit()
        max_level = -1
        for (t, _), edge in self.pair_copulas.items():
            if not edge_is_independent(edge):
                max_level = max(max_level, int(t))
        return max_level

    def _sample_active_edge_keys(self, max_active_tree=None):
        """Edge keys visited by the unconditional sampling recursion."""
        self._require_fit()
        if max_active_tree is None:
            max_active_tree = self._max_non_independent_tree_level()
        return rvine_sampling_edge_keys(self.d, max_active_tree)

    # Introspection

    @property
    def n_parameters(self) -> int:
        """Total number of fitted parameters across all pair copulas."""
        self._require_fit()
        return sum(pc.n_params for pc in self.pair_copulas.values())

    @property
    def aic(self) -> float:
        """AIC = -2 logL + 2 k."""
        self._require_fit()
        return statistics.information_criterion(
            self._log_likelihood, self.n_parameters, self._T, "aic")

    @property
    def bic(self) -> float:
        """BIC = -2 logL + k log T."""
        self._require_fit()
        return statistics.information_criterion(
            self._log_likelihood, self.n_parameters, self._T, "bic")

    def family_matrix(self) -> np.ndarray:
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
            M[d - 2 - col - t, col] = type(edge_copula(pc)).__name__
        return M

    def parameter_matrix(self) -> np.ndarray:
        """(d, d) float array with fitted copula parameters (NaN elsewhere).

        Position ``(d-2-col-t, col)`` carries the parameter of the
        tree-``t`` edge at column ``col`` (natural-order convention).
        """
        self._require_fit()
        d = self.d
        M = np.full((d, d), np.nan, dtype=np.float64)
        for (t, col), pc in self.pair_copulas.items():
            M[d - 2 - col - t, col] = edge_param(pc, default=np.nan)
        return M

    def rotation_matrix(self) -> np.ndarray:
        """(d, d) int array with copula rotations (-1 elsewhere).

        Position ``(d-2-col-t, col)`` carries the rotation of the
        tree-``t`` edge at column ``col`` (natural-order convention).
        """
        self._require_fit()
        d = self.d
        M = np.full((d, d), -1, dtype=int)
        for (t, col), pc in self.pair_copulas.items():
            M[d - 2 - col - t, col] = int(
                getattr(edge_copula(pc), 'rotate', 0))
        return M

    def tau_matrix(self) -> np.ndarray:
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

    # Summary / repr

    def summary(self, as_string: bool = False) -> str | None:
        """Print R-vine structure summary.

        By default the summary is
        printed and ``None`` is returned. Use ``summary(as_string=True)`` when
        a string value is needed.

        Returns
        -------
        text : str or None
        """
        text = format_rvine_summary(self)
        if as_string:
            return text
        print(text)
        return None

    def __str__(self):
        return self.summary(as_string=True)

    def __repr__(self):
        if self._natural_order_matrix is None:
            return "VineCopula(unfitted)"
        return (
            f"VineCopula(d={self.d}, T={self._T}, "
            f"logL={self._log_likelihood:.3f}, "
            f"n_params={self.n_parameters})"
        )

    def _invalidate_native_rvine_cache(self):
        """Discard transient compiled native plans after structural changes."""
        self._native_rvine_cache = {}
        self._native_rvine_generation = (
            int(getattr(self, '_native_rvine_generation', 0)) + 1)

    def _native_unconditional_fingerprint(
            self, module, traversal_plan, parameter_sources):
        """Return a semantic cache key, including mutable edge metadata."""
        matrix = np.ascontiguousarray(self._natural_order_matrix, dtype=int)
        structure = (
            int(self.d),
            matrix.shape,
            matrix.dtype.str,
            matrix.tobytes(),
            tuple(
                tuple((
                    tuple(sorted(int(value) for value in conditioned)),
                    tuple(sorted(int(value) for value in conditioning)),
                ) for conditioned, conditioning in level)
                for level in self._trees
            ),
            tuple(sorted(
                (tuple(int(value) for value in key), int(original))
                for key, original in self._edge_map.items()
            )),
        )
        edge_fingerprint = []
        for key in traversal_plan.active_keys:
            edge = self.pair_copulas[key]
            copula = edge_copula(edge)
            result = edge_result(edge)
            edge_fingerprint.append((
                tuple(key),
                type(edge),
                id(edge),
                type(copula),
                id(copula),
                int(getattr(copula, 'rotate', 0)),
                str(getattr(copula, '_transform_type', '')),
                bool(edge_is_independent(edge)),
                type(result),
                id(result),
            ))
        plan_fingerprint = tuple(
            tuple(getattr(traversal_plan, name))
            for name in (
                'output_nodes',
                'column_uniforms',
                'inverse_offsets',
                'inverse_edges',
                'inverse_partner_nodes',
                'inverse_output_nodes',
                'inverse_transposed',
                'forward_offsets',
                'forward_edges',
                'forward_leaf_nodes',
                'forward_partner_nodes',
                'forward_leaf_output_nodes',
                'forward_partner_output_nodes',
                'forward_transposed',
                'update_u1_nodes',
                'update_u2_nodes',
            )
        )
        return (
            id(module),
            int(getattr(self, '_native_rvine_generation', 0)),
            structure,
            tuple(edge_fingerprint),
            tuple(sorted(parameter_sources.items())),
            tuple(traversal_plan.active_keys),
            tuple(traversal_plan.node_keys),
            int(traversal_plan.last_uniform_column),
            int(traversal_plan.last_output_node),
            plan_fingerprint,
        )

    def _native_unconditional_context(
            self, module, traversal_plan, r_all, n):
        """Compile/cache plan and edge metadata, never request-owned buffers."""
        from pyscarcopula._native import vine as _cpp_rvine
        active_keys = tuple(traversal_plan.active_keys)
        if not _cpp_rvine.native_edges_supported(
                self.pair_copulas, active_keys):
            raise NativeUnsupported(
                "native R-vine sampling requires exact built-in edge copulas")
        parameter_sources = {}
        for key in active_keys:
            if edge_is_independent(self.pair_copulas[key]):
                continue
            if key not in r_all:
                # Let the common packer produce the stable missing-edge error.
                continue
            raw = np.asarray(r_all[key])
            if raw.ndim == 0 or raw.size == 1:
                parameter_sources[key] = 'scalar'
            elif raw.ndim == 1 and len(raw) == int(n):
                parameter_sources[key] = 'row_path'

        fingerprint = self._native_unconditional_fingerprint(
            module, traversal_plan, parameter_sources)
        cache = getattr(self, '_native_rvine_cache', None)
        if cache is None:
            cache = {}
            self._native_rvine_cache = cache
        cached = cache.get('unconditional')
        if cached is not None and cached['fingerprint'] == fingerprint:
            return cached, None

        edges, parameters = _cpp_rvine.compile_edge_specs(
            module,
            self.pair_copulas,
            active_keys,
            r_all,
            int(n),
            parameter_sources=parameter_sources,
        )
        context = {
            'fingerprint': fingerprint,
            'parameter_sources': parameter_sources,
            'plan': _cpp_rvine.compile_traversal_plan(
                module, traversal_plan),
            'edges': tuple(edges),
        }
        cache['unconditional'] = context
        return context, parameters

    def _native_conditional_fingerprint(
            self, module, plan, active_keys, pair_copulas, given,
            parameter_sources):
        """Return a cache key for one structure/given-set native program.

        Plans are immutable and carry a digest of their full node/opcode/
        orientation signature.  The digest keeps warm comparison O(1) while
        preventing topology-compatible edge sets from reusing a stale plan.
        """
        edge_fingerprint = []
        for key in active_keys:
            edge = pair_copulas[key]
            copula = edge_copula(edge)
            result = edge_result(edge)
            edge_fingerprint.append((
                tuple(key),
                type(edge),
                id(edge),
                type(copula),
                id(copula),
                int(getattr(copula, 'rotate', 0)),
                str(getattr(copula, '_transform_type', '')),
                bool(edge_is_independent(edge)),
                type(result),
                id(result),
            ))
        return (
            id(module),
            int(getattr(self, '_native_rvine_generation', 0)),
            type(plan),
            int(plan.d),
            tuple(sorted(int(variable) for variable in given)),
            bytes(plan.native_signature_digest),
            tuple(active_keys),
            tuple(edge_fingerprint),
            tuple(sorted(parameter_sources.items())),
        )

    def _native_suffix_conditional_plan(
            self, start_col, matrix, given):
        """Return one immutable suffix program per structure/given set."""
        normalized_matrix = np.ascontiguousarray(matrix, dtype=int)
        fingerprint = (
            int(getattr(self, '_native_rvine_generation', 0)),
            int(self.d),
            int(start_col),
            tuple(sorted(int(variable) for variable in given)),
            normalized_matrix.shape,
            normalized_matrix.dtype.str,
            normalized_matrix.tobytes(),
        )
        cache = getattr(self, '_native_rvine_cache', None)
        if cache is None:
            cache = {}
            self._native_rvine_cache = cache
        plans = cache.setdefault('conditional_python_plans', {})
        key = (
            'suffix',
            tuple(sorted(int(variable) for variable in given)),
        )
        cached = plans.get(key)
        if cached is not None and cached['fingerprint'] == fingerprint:
            return cached['plan']
        plan = build_suffix_conditional_plan(
            self.d, start_col, normalized_matrix, given)
        plans[key] = {'fingerprint': fingerprint, 'plan': plan}
        return plan

    def _native_conditional_context(
            self, module, plan, pair_copulas, r_all, given, n, *,
            active_keys=None, normalized_paths=None,
            parameter_sources=None):
        """Compile/cache conditional topology and immutable edge metadata."""
        from pyscarcopula._native import vine as _cpp_rvine
        active_keys = (
            _cpp_rvine.conditional_active_keys(plan)
            if active_keys is None else tuple(active_keys)
        )
        if not _cpp_rvine.native_edges_supported(pair_copulas, active_keys):
            raise NativeUnsupported(
                "native conditional R-vine sampling requires exact built-in "
                "edge copulas")
        if normalized_paths is None or parameter_sources is None:
            normalized_paths, parameter_sources = (
                _cpp_rvine.conditional_parameter_layout(
                    pair_copulas, active_keys, r_all, int(n))
            )
        fingerprint = self._native_conditional_fingerprint(
            module,
            plan,
            active_keys,
            pair_copulas,
            given,
            parameter_sources,
        )
        cache = getattr(self, '_native_rvine_cache', None)
        if cache is None:
            cache = {}
            self._native_rvine_cache = cache
        conditional_cache = cache.setdefault('conditional', {})
        cache_key = (
            type(plan).__module__,
            type(plan).__qualname__,
            tuple(sorted(int(variable) for variable in given)),
        )
        cached = conditional_cache.get(cache_key)
        if cached is not None and cached['fingerprint'] == fingerprint:
            scalar_signature = None
            if all(source == 'scalar'
                   for source in parameter_sources.values()):
                scalar_signature = tuple(
                    (key, float(np.asarray(
                        normalized_paths[key]).reshape(-1)[0]))
                    for key in active_keys
                    if parameter_sources.get(key) == 'scalar'
                )
            if (
                    scalar_signature is not None
                    and cached.get('scalar_parameter_signature')
                    == scalar_signature):
                parameters = _cpp_rvine.RVineParameterPack(
                    scalar_parameters=cached['scalar_parameters'],
                    row_parameters=np.empty((int(n), 0), dtype=np.float64),
                    n_rows=int(n),
                )
                return cached, parameters
            _edges, parameters = _cpp_rvine.compile_edge_specs(
                module,
                pair_copulas,
                active_keys,
                normalized_paths,
                int(n),
                parameter_sources=parameter_sources,
                native_edges=cached['edges'],
            )
            if scalar_signature is not None:
                cached['scalar_parameter_signature'] = scalar_signature
                cached['scalar_parameters'] = parameters.scalar_parameters
            return cached, parameters

        edges, parameters = _cpp_rvine.compile_edge_specs(
            module,
            pair_copulas,
            active_keys,
            normalized_paths,
            int(n),
            parameter_sources=parameter_sources,
        )
        context = {
            'fingerprint': fingerprint,
            'active_keys': active_keys,
            'parameter_sources': parameter_sources,
            'plan': _cpp_rvine.compile_conditional_plan(
                module, plan, active_keys, given),
            'edges': tuple(edges),
        }
        if all(source == 'scalar'
               for source in parameter_sources.values()):
            context['scalar_parameter_signature'] = tuple(
                (key, float(np.asarray(
                    normalized_paths[key]).reshape(-1)[0]))
                for key in active_keys
                if parameter_sources.get(key) == 'scalar'
            )
            context['scalar_parameters'] = parameters.scalar_parameters
        conditional_cache[cache_key] = context
        return context, parameters

    @staticmethod
    def _native_cached_scalar_layout(
            cached, fingerprint, active_keys, parameter_paths):
        """Reuse a validated scalar layout only while every value is unchanged."""
        if cached is None or cached.get('fingerprint') != fingerprint:
            return None
        sources = cached.get('parameter_sources', {})
        if any(source != 'scalar' for source in sources.values()):
            return None
        signature = []
        try:
            for key in active_keys:
                if sources.get(key) != 'scalar':
                    continue
                raw = np.asarray(parameter_paths[key])
                if np.iscomplexobj(raw) or raw.size != 1:
                    return None
                value = float(np.asarray(raw, dtype=np.float64).reshape(-1)[0])
                if not np.isfinite(value):
                    return None
                signature.append((key, value))
        except (KeyError, TypeError, ValueError):
            return None
        if tuple(signature) != cached.get('scalar_parameter_signature'):
            return None
        return parameter_paths, sources, cached

    def _native_cached_conditional_layout(
            self, module, plan, pair_copulas, r_all, given, active_keys):
        cache = getattr(self, '_native_rvine_cache', {})
        conditional_cache = cache.get('conditional', {})
        cache_key = (
            type(plan).__module__,
            type(plan).__qualname__,
            tuple(sorted(int(variable) for variable in given)),
        )
        cached = conditional_cache.get(cache_key)
        if cached is None:
            return None
        fingerprint = self._native_conditional_fingerprint(
            module,
            plan,
            active_keys,
            pair_copulas,
            given,
            cached.get('parameter_sources', {}),
        )
        return self._native_cached_scalar_layout(
            cached, fingerprint, active_keys, r_all)

    def _native_cached_density_layout(
            self, module, pair_copulas, edge_map, r_all, active_keys, *,
            cache_slot='density', residual_node_keys=()):
        cache = getattr(self, '_native_rvine_cache', {})
        cached = cache.get(cache_slot)
        if cached is None:
            return None
        fingerprint = self._native_density_fingerprint(
            module,
            pair_copulas,
            edge_map,
            active_keys,
            cached.get('parameter_sources', {}),
            residual_node_keys,
        )
        return self._native_cached_scalar_layout(
            cached, fingerprint, active_keys, r_all)

    def _native_conditional_executor(
            self, plan, pair_copulas, r_all, given, n, rng, *, uniforms,
            active_keys, normalized_paths, parameter_sources,
            native_context=None, parameter_pack=None):
        """Build one shared native callback for suffix and DAG programs."""
        from pyscarcopula._native import vine as _cpp_rvine

        def execute(module):
            """Execute the compiled native conditional request."""
            if native_context is None:
                context, parameters = self._native_conditional_context(
                    module,
                    plan,
                    pair_copulas,
                    r_all,
                    given,
                    n,
                    active_keys=active_keys,
                    normalized_paths=normalized_paths,
                    parameter_sources=parameter_sources,
                )
            else:
                context, parameters = native_context, parameter_pack
            return _cpp_rvine.conditional_sample(
                module,
                pair_copulas,
                plan,
                n,
                rng,
                given,
                r_all,
                uniforms=uniforms,
                active_keys=context['active_keys'],
                parameter_sources=context['parameter_sources'],
                native_plan=context['plan'],
                native_edges=context['edges'],
                parameter_pack=parameters,
            )

        return execute

    def _native_density_fingerprint(
            self, module, pair_copulas, edge_map, active_keys,
            parameter_sources, residual_node_keys=()):
        """Return a semantic key for the fused density/MCMC program."""
        edge_fingerprint = []
        for key in active_keys:
            edge = pair_copulas[key]
            copula = edge_copula(edge)
            result = edge_result(edge)
            edge_fingerprint.append((
                tuple(key),
                type(edge),
                id(edge),
                type(copula),
                id(copula),
                int(getattr(copula, 'rotate', 0)),
                str(getattr(copula, '_transform_type', '')),
                bool(edge_is_independent(edge)),
                type(result),
                id(result),
            ))
        tree_fingerprint = tuple(
            tuple((
                tuple(sorted(int(value) for value in conditioned)),
                tuple(sorted(int(value) for value in conditioning)),
            ) for conditioned, conditioning in level)
            for level in self._trees
        )
        return (
            id(module),
            int(getattr(self, '_native_rvine_generation', 0)),
            int(self.d),
            id(pair_copulas),
            tree_fingerprint,
            tuple(sorted(
                (tuple(int(value) for value in key), int(original))
                for key, original in edge_map.items()
            )),
            tuple(active_keys),
            tuple(edge_fingerprint),
            tuple(sorted(parameter_sources.items())),
            tuple(
                (int(variable), tuple(sorted(int(value) for value in cond)))
                for variable, cond in residual_node_keys
            ),
        )

    def _native_density_context(
            self, module, pair_copulas, edge_map, r_all, n, *,
            active_keys=None, normalized_paths=None,
            parameter_sources=None, residual_node_keys=(),
            cache_slot='density'):
        """Compile/cache the shared density plan and immutable edge specs."""
        from pyscarcopula._native import vine as _cpp_rvine
        active_keys = (
            _cpp_rvine.density_active_keys(self._trees, edge_map)
            if active_keys is None else tuple(active_keys)
        )
        if not _cpp_rvine.native_edges_supported(pair_copulas, active_keys):
            raise NativeUnsupported(
                "native R-vine density requires exact built-in edge copulas")
        if normalized_paths is None or parameter_sources is None:
            normalized_paths, parameter_sources = (
                _cpp_rvine.density_parameter_layout(
                    pair_copulas, active_keys, r_all, int(n))
            )
        fingerprint = self._native_density_fingerprint(
            module,
            pair_copulas,
            edge_map,
            active_keys,
            parameter_sources,
            residual_node_keys,
        )
        cache = getattr(self, '_native_rvine_cache', None)
        if cache is None:
            cache = {}
            self._native_rvine_cache = cache
        cached = cache.get(cache_slot)
        if cached is not None and cached['fingerprint'] == fingerprint:
            scalar_signature = None
            if all(source == 'scalar'
                   for source in parameter_sources.values()):
                scalar_signature = tuple(
                    (key, float(np.asarray(normalized_paths[key]).reshape(-1)[0]))
                    for key in active_keys
                    if parameter_sources.get(key) == 'scalar'
                )
            if (
                    scalar_signature is not None
                    and cached.get('scalar_parameter_signature')
                    == scalar_signature):
                parameters = _cpp_rvine.RVineParameterPack(
                    scalar_parameters=cached['scalar_parameters'],
                    row_parameters=np.empty((int(n), 0), dtype=np.float64),
                    n_rows=int(n),
                )
                return cached, parameters
            _edges, parameters = _cpp_rvine.compile_edge_specs(
                module,
                pair_copulas,
                active_keys,
                normalized_paths,
                int(n),
                parameter_sources=parameter_sources,
                native_edges=cached['edges'],
            )
            if scalar_signature is not None:
                cached['scalar_parameter_signature'] = scalar_signature
                cached['scalar_parameters'] = parameters.scalar_parameters
            return cached, parameters

        edges, parameters = _cpp_rvine.compile_edge_specs(
            module,
            pair_copulas,
            active_keys,
            normalized_paths,
            int(n),
            parameter_sources=parameter_sources,
        )
        context = {
            'fingerprint': fingerprint,
            'active_keys': active_keys,
            'parameter_sources': parameter_sources,
            'plan': _cpp_rvine.compile_density_plan(
                module,
                self.d,
                self._trees,
                edge_map,
                active_keys,
                residual_node_keys=residual_node_keys,
            ),
            'edges': tuple(edges),
        }
        if all(source == 'scalar'
               for source in parameter_sources.values()):
            context['scalar_parameter_signature'] = tuple(
                (key, float(np.asarray(normalized_paths[key]).reshape(-1)[0]))
                for key in active_keys
                if parameter_sources.get(key) == 'scalar'
            )
            context['scalar_parameters'] = parameters.scalar_parameters
        cache[cache_slot] = context
        return context, parameters

    # Sampling

    def sample(
            self,
            n: int,
            u: Any = None,
            rng: np.random.Generator | None = None,
            *,
            batch_rows: int | None = None,
            memory_budget_bytes: int | None = None) -> np.ndarray:
        """Unconditional sampling from the fitted vine.

        Samples in natural-order matrix order: columns are processed from
        right to left, and each new anti-diagonal leaf is recovered by
        applying inverse h-functions from the top tree down to tree 0.

        Static vines are evaluated in bounded row batches. ``batch_rows``
        controls the temporary vectorized workspace; the default is 8192.
        Dynamic edge trajectories retain their sequential full-path semantics
        and therefore are not split across batches.
        """
        self._require_fit()
        if not isinstance(n, (int, np.integer)) or n <= 0:
            raise ValueError(
                f"VineCopula.sample: n must be positive int, got {n!r}")
        n = int(n)
        if batch_rows is None:
            batch_rows = min(n, _DEFAULT_STATIC_SAMPLE_BATCH_ROWS)
        else:
            batch_rows = min(n, validate_positive_int(
                batch_rows, "batch_rows"))
        memory_budget_bytes = _validated_memory_budget(memory_budget_bytes)
        if rng is None:
            rng = np.random.default_rng()

        max_active_tree = self._max_non_independent_tree_level()
        active_keys = self._sample_active_edge_keys(max_active_tree)
        requires_stepwise = any(
            _edge_requires_stepwise_sample(self.pair_copulas[key])
            for key in active_keys
        )
        traversal_plan = (
            build_rvine_sampling_plan(
                self.d,
                self._natural_order_matrix,
                self._trees,
                self._edge_map,
                active_keys,
                max_active_tree,
            )
            if active_keys
            else None
        )
        is_static = all(
            not edge_has_dynamic_params(self.pair_copulas[key])
            for key in active_keys
        )
        working_rows = batch_rows if is_static else n
        self._check_sample_memory_budget(
            n,
            working_rows,
            len(active_keys),
            memory_budget_bytes,
        )
        if requires_stepwise:
            from pyscarcopula._native._gas_vine import sample as gas_vine_sample

            return gas_vine_sample(
                self,
                n,
                rng,
                active_keys,
                max_active_tree,
                traversal_plan=traversal_plan,
            )

        if is_static:
            r_all = {
                key: np.asarray(
                    _edge_r_for_sample(self.pair_copulas[key], 1, rng),
                    dtype=np.float64,
                )
                for key in active_keys
            }
            native_request = {}
            if batch_rows < n:
                out = np.empty((n, self.d), dtype=np.float64)
                for start in range(0, n, batch_rows):
                    stop = min(start + batch_rows, n)
                    out[start:stop] = self._sample_with_r(
                        stop - start,
                        r_all,
                        rng,
                        max_active_tree=max_active_tree,
                        traversal_plan=traversal_plan,
                        native_request=native_request,
                    )
                return out
            return self._sample_with_r(
                n,
                r_all,
                rng,
                max_active_tree=max_active_tree,
                traversal_plan=traversal_plan,
                native_request=native_request,
            )

        r_all = {
            key: _edge_r_for_sample(self.pair_copulas[key], n, rng)
            for key in active_keys
        }
        return self._sample_with_r(
            n,
            r_all,
            rng,
            max_active_tree=max_active_tree,
            traversal_plan=traversal_plan,
        )

    def _check_sample_memory_budget(
            self, n, working_rows, active_edge_count, memory_budget_bytes):
        if memory_budget_bytes is None:
            return
        itemsize = np.dtype(np.float64).itemsize
        output_bytes = int(n) * int(self.d) * itemsize
        # Conservative Python-workspace estimate: base uniforms and output
        # plus conditional values and native point-operation temporaries for
        # each active edge in the current row batch.
        workspace_vectors = 4 * int(self.d) + 6 * int(active_edge_count)
        required = (
            output_bytes
            + int(working_rows) * workspace_vectors * itemsize
        )
        if required > memory_budget_bytes:
            raise MemoryError(
                "VineCopula.sample requires an estimated "
                f"{required} bytes, exceeding memory_budget_bytes="
                f"{memory_budget_bytes}; reduce n or batch_rows, or increase "
                "memory_budget_bytes"
            )

    def _sample_with_r(self, n, r_all, rng,
                       max_active_tree=None, traversal_plan=None, *,
                       uniforms=None, native_request=None):
        """Run a sampling request through the mandatory native runtime."""
        native_max_tree = max_active_tree
        if native_max_tree is None:
            native_max_tree = self._max_non_independent_tree_level()
        native_traversal_plan = traversal_plan
        if native_traversal_plan is None:
            active_keys = self._sample_active_edge_keys(native_max_tree)
            native_traversal_plan = build_rvine_sampling_plan(
                self.d,
                self._natural_order_matrix,
                self._trees,
                self._edge_map,
                active_keys,
                native_max_tree,
            )
        active_keys = tuple(native_traversal_plan.active_keys)
        from pyscarcopula._native import vine as _cpp_rvine

        if not _cpp_rvine.native_edges_supported(
                self.pair_copulas, active_keys):
            raise NativeUnsupported(
                "native R-vine sampling requires exact registered built-in "
                "edge copulas"
            )

        module = _cpp_extension.load()
        context, initial_parameters = self._native_unconditional_context(
            module, native_traversal_plan, r_all, n)
        return _cpp_rvine.sample(
            module,
            self,
            n,
            rng,
            active_keys,
            native_traversal_plan,
            r_all,
            uniforms=uniforms,
            parameter_sources=context['parameter_sources'],
            native_plan=context['plan'],
            native_edges=context['edges'],
            parameter_pack=initial_parameters,
            request_state=native_request,
        )

    def _given_suffix_start_col(self, given, matrix=None):
        matrix = self._natural_order_matrix if matrix is None else matrix
        return given_suffix_start_col(self.d, given, matrix)

    def _suffix_sampling_state(self, given):
        cache = getattr(self, '_suffix_state_cache', None)
        if cache is None:
            cache = {}
            self._suffix_state_cache = cache
        cache_key = frozenset(int(var) for var in given)
        if cache_key in cache:
            return cache[cache_key]
        state = suffix_sampling_state(
            self.d,
            self._trees,
            self._natural_order_matrix,
            self._edge_map,
            self.pair_copulas,
            self._matrix_key,
            given,
        )
        cache[cache_key] = state
        return state

    def _sample_suffix_given_with_r(
            self, n, r_all, rng, given, start_col, matrix=None,
            pair_copulas=None, *, uniforms=None):
        """Dispatch exact suffix conditioning with supplied edge parameters."""
        M = self._natural_order_matrix if matrix is None else matrix
        pair_copulas = self.pair_copulas if pair_copulas is None else pair_copulas
        from pyscarcopula._native import vine as _cpp_rvine

        plan = self._native_suffix_conditional_plan(
            start_col, M, given)
        active_keys = _cpp_rvine.conditional_active_keys(plan)
        if not _cpp_rvine.native_edges_supported(pair_copulas, active_keys):
            raise NativeUnsupported(
                "native suffix-conditioned R-vine sampling requires exact "
                "registered built-in edge copulas"
            )
        module = _cpp_extension.load()
        cached_layout = self._native_cached_conditional_layout(
            module, plan, pair_copulas, r_all, given, active_keys)
        native_context = None
        parameter_pack = None
        if cached_layout is None:
            normalized_paths, parameter_sources = (
                _cpp_rvine.conditional_parameter_layout(
                pair_copulas, active_keys, r_all, int(n))
            )
        else:
            normalized_paths, parameter_sources, native_context = cached_layout
            parameter_pack = _cpp_rvine.RVineParameterPack(
                scalar_parameters=native_context['scalar_parameters'],
                row_parameters=np.empty((int(n), 0), dtype=np.float64),
                n_rows=int(n),
            )

        native_executor = self._native_conditional_executor(
            plan,
            pair_copulas,
            r_all,
            given,
            n,
            rng,
            uniforms=uniforms,
            active_keys=active_keys,
            normalized_paths=normalized_paths,
            parameter_sources=parameter_sources,
            native_context=native_context,
            parameter_pack=parameter_pack,
        )

        return native_executor(module)

    # Dynamic conditioning contract is documented in
    # docs/rvine-conditional-notes.md. Keep these helpers strategy-generic:
    # no dynamic-model formulas or result-type checks belong here.
    def _predictive_given_update_r(self, edge, u_train_pair, u_observed_pair,
                                   n, horizon, rng, predictive_r_mode,
                                   state_cache=None, cache_key=None,
                                   posterior_cache=None):
        return predictive_given_update_r(
            edge,
            u_train_pair,
            u_observed_pair,
            n,
            horizon=horizon,
            rng=rng,
            predictive_r_mode=predictive_r_mode,
            state_cache=state_cache,
            cache_key=cache_key,
            posterior_cache=posterior_cache,
            strategy_for_result=_strategy_for_result,
        )

    def _dynamic_edge_update_from_observation(
            self, key, edge, r_current, u_pair, edge_map, train_pseudo,
            horizon, rng, predictive_r_mode, state_cache=None,
            posterior_cache=None):
        u_train_pair = None
        if train_pseudo is not None:
            u_train_pair = self._edge_pair_from_pseudo_map(
                key, train_pseudo, edge_map)
        return dynamic_edge_update_from_observation(
            edge,
            r_current,
            u_pair,
            u_train_pair,
            horizon,
            rng,
            predictive_r_mode,
            state_cache=state_cache,
            cache_key=_predictive_state_cache_key(key, horizon),
            posterior_cache=posterior_cache,
            strategy_for_result=_strategy_for_result,
        )

    def _dynamic_edge_skip_reason(self, edge, train_pseudo, horizon):
        return dynamic_edge_skip_reason(edge, train_pseudo, horizon)

    def _dynamic_update_record(
            self, key, edge, edge_map, r_before, r_after, status, reason=None):
        return dynamic_update_record(
            self._trees,
            key,
            edge,
            edge_map,
            r_before,
            r_after,
            status,
            reason=reason,
        )

    def _dynamic_skip_records(
            self, pair_copulas, edge_map, r_all, reason):
        return dynamic_skip_records(
            self._trees, pair_copulas, edge_map, r_all, reason)

    def _apply_given_only_dynamic_updates_ordered(
            self, n, r_all, given, start_col, matrix, pair_copulas, edge_map,
            train_pseudo, horizon, rng, predictive_r_mode, state_cache=None,
            posterior_cache=None):
        from pyscarcopula._native import vine as native_vine

        return native_vine.apply_given_only_dynamic_updates(
            _cpp_extension.load(),
            dimension=self.d,
            trees=self._trees,
            n=n,
            r_all=r_all,
            given=given,
            start_col=start_col,
            matrix=matrix,
            pair_copulas=pair_copulas,
            edge_map=edge_map,
            train_pseudo=train_pseudo,
            horizon=horizon,
            rng=rng,
            predictive_r_mode=predictive_r_mode,
            dynamic_update=self._dynamic_edge_update_from_observation,
            update_record=self._dynamic_update_record,
            skip_reason=self._dynamic_edge_skip_reason,
            state_cache=state_cache,
            posterior_cache=posterior_cache,
        )

    def _edge_pair_from_pseudo(self, key, pseudo_obs):
        return self._edge_pair_from_pseudo_map(key, pseudo_obs, self._edge_map)

    def _edge_pair_from_pseudo_map(self, key, pseudo_obs, edge_map):
        return edge_pair_from_pseudo_map(
            self._trees, key, pseudo_obs, edge_map)

    def _compute_pseudo_obs(self, u):
        from pyscarcopula._native import vine as native_vine

        return native_vine.pseudo_observations(
            _cpp_extension.load(),
            self.pair_copulas,
            self.d,
            self._trees,
            self._edge_map,
            self.matrix,
            u,
        )

    def _history_prediction_cache(self, u, *, fitted_history):
        """Return a mutation-safe cache scoped to one prediction history."""
        root = getattr(self, '_predict_history_cache', None)
        if root is None:
            root = {}
            self._predict_history_cache = root
        cache = root.setdefault(
            'fitted' if fitted_history else 'explicit', {})

        edge_token = tuple(
            (
                key,
                id(edge),
                id(edge_copula(edge)),
                id(edge_result(edge)),
                edge_param(edge),
                getattr(edge_copula(edge), 'rotate', None),
                getattr(edge_copula(edge), '_transform_type', None),
            )
            for key, edge in sorted(self.pair_copulas.items())
        )
        if fitted_history:
            history_token = id(u)
        else:
            contiguous = np.ascontiguousarray(u, dtype=np.float64)
            history_token = (
                contiguous.shape,
                hashlib.blake2b(
                    memoryview(contiguous).cast('B'), digest_size=16
                ).digest(),
            )
        token = (history_token, edge_token)
        if cache.get('token') != token:
            cache.clear()
            cache['token'] = token
            cache['state_cache'] = {}
            cache['posterior_cache'] = {}
        return cache

    def _predict_r_for_edges(self, edge_keys, pair_copulas, edge_map, n,
                             train_pseudo, horizon, rng,
                             predictive_r_mode=None, state_cache=None,
                             posterior_cache=None):
        edge_horizon = _normalize_predict_horizon(horizon)
        r_all = {}
        for key in edge_keys:
            edge = pair_copulas[key]
            u_pair = None
            if (
                    train_pseudo is not None
                    and edge_result(edge) is not None
                    and edge_has_dynamic_params(edge)):
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
                cache_key=_predictive_state_cache_key(key, edge_horizon),
                posterior_cache=posterior_cache,
            )
        return r_all

    def _sample_dag_given_with_r(
            self, n, r_all, rng, given, plan, pair_copulas, *,
            uniforms=None):
        """Dispatch an arbitrary-DAG conditional initialization program."""
        from pyscarcopula._native import vine as _cpp_rvine

        missing = sorted(set(plan.edges_used) - set(r_all))
        if missing:
            raise KeyError(
                "RVineCopula._sample_dag_given_with_r: missing predicted "
                f"parameters for DAG edges {missing}"
            )
        active_keys = _cpp_rvine.conditional_active_keys(plan)
        if not _cpp_rvine.native_edges_supported(pair_copulas, active_keys):
            raise NativeUnsupported(
                "native DAG-conditioned R-vine sampling requires exact "
                "registered built-in edge copulas"
            )
        module = _cpp_extension.load()
        cached_layout = self._native_cached_conditional_layout(
            module, plan, pair_copulas, r_all, given, active_keys)
        native_context = None
        parameter_pack = None
        if cached_layout is None:
            normalized_paths, parameter_sources = (
                _cpp_rvine.conditional_parameter_layout(
                pair_copulas, active_keys, r_all, int(n))
            )
        else:
            normalized_paths, parameter_sources, native_context = cached_layout
            parameter_pack = _cpp_rvine.RVineParameterPack(
                scalar_parameters=native_context['scalar_parameters'],
                row_parameters=np.empty((int(n), 0), dtype=np.float64),
                n_rows=int(n),
            )

        native_executor = self._native_conditional_executor(
            plan,
            pair_copulas,
            r_all,
            given,
            n,
            rng,
            uniforms=uniforms,
            active_keys=active_keys,
            normalized_paths=normalized_paths,
            parameter_sources=parameter_sources,
            native_context=native_context,
            parameter_pack=parameter_pack,
        )

        return native_executor(module)

    def _log_pdf_rows_with_r(
            self, u, r_all, pair_copulas=None, edge_map=None):
        """Dispatch fused row log-density for supplied edge parameters."""
        from pyscarcopula._native import vine as _cpp_rvine

        pair_copulas = (
            self.pair_copulas if pair_copulas is None else pair_copulas)
        edge_map = self._edge_map if edge_map is None else edge_map
        active_keys = _cpp_rvine.density_active_keys(
            self._trees, edge_map)
        if not _cpp_rvine.native_edges_supported(pair_copulas, active_keys):
            raise NativeUnsupported(
                "native R-vine density requires exact registered built-in "
                "edge copulas"
            )
        observations = _cpp_rvine._rvine_observations(
            u, self.d, "density")
        module = _cpp_extension.load()
        cached_layout = self._native_cached_density_layout(
            module, pair_copulas, edge_map, r_all, active_keys)
        if cached_layout is None:
            normalized_paths, parameter_sources = (
                _cpp_rvine.density_parameter_layout(
                pair_copulas, active_keys, r_all, len(observations))
            )
            context, parameters = self._native_density_context(
                module,
                pair_copulas,
                edge_map,
                r_all,
                len(observations),
                active_keys=active_keys,
                normalized_paths=normalized_paths,
                parameter_sources=parameter_sources,
            )
        else:
            normalized_paths, parameter_sources, context = cached_layout
            parameters = _cpp_rvine.RVineParameterPack(
                scalar_parameters=context['scalar_parameters'],
                row_parameters=np.empty(
                    (len(observations), 0), dtype=np.float64),
                n_rows=len(observations),
            )
        return _cpp_rvine.log_pdf_rows(
            module,
            pair_copulas,
            self.d,
            self._trees,
            edge_map,
            r_all,
            observations,
            active_keys=context['active_keys'],
            normalized_parameter_paths=normalized_paths,
            parameter_sources=context['parameter_sources'],
            native_plan=context['plan'],
            native_edges=context['edges'],
            parameter_pack=parameters,
        )

    def _matrix_key_from_map(self, tree_level, orig_idx, edge_map):
        for key, mapped_idx in edge_map.items():
            if key[0] == tree_level and mapped_idx == orig_idx:
                return key
        raise KeyError((tree_level, orig_idx))

    def _sample_arbitrary_given_mcmc(
            self, n, r_all, rng, given, initial=None, n_steps=None,
            burnin_steps=None, *, initial_uniforms=None, random_draws=None,
            step_offset=0, density_algorithm="auto", chunk_steps=256):
        """Dispatch bounded coordinate-update MCMC for arbitrary given sets."""
        from pyscarcopula._native import vine as _cpp_rvine

        active_keys = _cpp_rvine.density_active_keys(
            self._trees, self._edge_map)
        if not _cpp_rvine.native_edges_supported(
                self.pair_copulas, active_keys):
            raise NativeUnsupported(
                "native R-vine MCMC requires exact registered built-in "
                "edge copulas"
            )
        module = _cpp_extension.load()
        cached_layout = self._native_cached_density_layout(
            module,
            self.pair_copulas,
            self._edge_map,
            r_all,
            active_keys,
        )
        if cached_layout is None:
            normalized_paths, parameter_sources = (
                _cpp_rvine.density_parameter_layout(
                self.pair_copulas, active_keys, r_all, int(n))
            )
            context, parameters = self._native_density_context(
                module,
                self.pair_copulas,
                self._edge_map,
                r_all,
                n,
                active_keys=active_keys,
                normalized_paths=normalized_paths,
                parameter_sources=parameter_sources,
            )
        else:
            normalized_paths, parameter_sources, context = cached_layout
            parameters = _cpp_rvine.RVineParameterPack(
                scalar_parameters=context['scalar_parameters'],
                row_parameters=np.empty((int(n), 0), dtype=np.float64),
                n_rows=int(n),
            )
        return _cpp_rvine.mcmc(
            module,
            self.pair_copulas,
            self.d,
            self._trees,
            self._edge_map,
            r_all,
            n,
            rng,
            given,
            initial=initial,
            n_steps=n_steps,
            burnin_steps=burnin_steps,
            initial_uniforms=initial_uniforms,
            random_draws=random_draws,
            step_offset=step_offset,
            active_keys=context['active_keys'],
            normalized_parameter_paths=normalized_paths,
            parameter_sources=context['parameter_sources'],
            native_plan=context['plan'],
            native_edges=context['edges'],
            parameter_pack=parameters,
            density_algorithm=density_algorithm,
            chunk_steps=chunk_steps,
        )

    def predict(
            self,
            n: int,
            u: Any = None,
            rng: np.random.Generator | None = None,
            given: Mapping[int, float] | None = None,
            horizon: str = 'next',
            predictive_r_mode: str | None = None,
            predict_config: PredictConfig | None = None,
            dynamic_conditioning: str = 'ignore',
            return_diagnostics: bool = False,
            mcmc_steps: int | None = None,
            mcmc_burnin: int | None = None,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
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

        For dynamic edges, ``horizon`` selects whether prediction starts from
        the current or next strategy-owned predictive state.

        ``dynamic_conditioning='given_only'`` additionally lets fixed suffix
        observations update strategy-owned dynamic edge states when an edge
        pair is fully determined before any free variable is sampled. Static
        edges have no dynamic state, so ``'given_only'`` is a no-op.

        For arbitrary non-suffix ``given`` patterns, prediction uses a DAG
        initializer followed by MCMC. In that mode ``'given_only'`` is
        reported as skipped rather than partially applied.

        Parameters
        ----------
        n : int
            Number of predictive samples to draw.
        u : (T, d) array-like or None, default None
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
        predictive_r_mode : {'grid', 'histogram'} or None, default None
            Predictive parameter sampling mode for strategies with non-point
            predictive state. ``None`` uses the strategy default.
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
            Number of Metropolis-within-Gibbs single-coordinate updates used
            after the DAG initializer for arbitrary non-suffix ``given``
            patterns. One full sweep contains one update per free variable.
            If ``None``, a dimension-based default is used.
        mcmc_burnin : int or None, default None
            Number of burn-in single-coordinate updates for the arbitrary-
            ``given`` MCMC fallback. If ``None``, a dimension-based default is
            used.

        Returns
        -------
        samples : (n, d) ndarray
            Predictive pseudo-observations.
        samples, diagnostics : tuple
            Returned when ``return_diagnostics=True``. Diagnostics include the
            conditioning method, suffix position, dynamic edge updates and
            MCMC acceptance, completed-sweep and convergence-warning
            information when applicable. ``convergence_warning`` is a
            conservative heuristic: it is set when a free coordinate accepts
            fewer than 2 percent of proposals or fewer than five moves per
            parallel chain. It is not a proof of convergence.
        """
        self._require_fit()
        if not isinstance(n, (int, np.integer)) or n <= 0:
            raise ValueError(f"VineCopula.predict: n must be positive int, got {n!r}")
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
        if pcfg.return_diagnostics:
            predict_start = time.perf_counter()
            timings = {}

            def timed(name, call):
                start = time.perf_counter()
                try:
                    return call()
                finally:
                    timings[name] = (
                        timings.get(name, 0.0)
                        + time.perf_counter() - start)

            def attach_timings(diagnostics):
                timings["total"] = time.perf_counter() - predict_start
                diagnostics["timings_ms"] = {
                    name: 1e3 * value
                    for name, value in sorted(timings.items())
                }
                return diagnostics
        else:
            def timed(_name, call):
                return call()

            def attach_timings(diagnostics):
                return diagnostics

        if rng is None:
            rng = np.random.default_rng()

        n = int(n)
        given = timed(
            "validate_given",
            lambda: validate_rvine_given(pcfg.given, self.d),
        )
        target_given = (
            bool(given)
            and self._target_given_vars
            and tuple(sorted(given)) == self._target_given_vars
        )
        if target_given and not self._conditional_fit_supported:
            raise ValueError(
                "VineCopula.predict: model was fitted with "
                f"given_vars={list(self._target_given_vars)}, "
                "but the fitted structure does not support exact conditional "
                "sampling for that target set"
            )
        if len(given) == self.d:
            def fill_given():
                out_given = np.empty((n, self.d), dtype=np.float64)
                for i in range(self.d):
                    out_given[:, i] = given[i]
                return out_given
            out = timed("fill_all_given", fill_given)
            if pcfg.return_diagnostics:
                diagnostics = {
                    'given': dict(given),
                    'dynamic_conditioning': dynamic_conditioning,
                    'suffix_start_col': 0,
                    'matrix_rebuilt': False,
                    'conditional_method': 'suffix',
                    'updated_edges': [],
                    'skipped_edges': [],
                    'all_variables_given': True,
                }
                return out, attach_timings(diagnostics)
            return out
        suffix_state = timed(
            "suffix_state",
            lambda: self._suffix_sampling_state(given) if given else None,
        )

        use_fitted_history_cache = u is None and self._last_u is not None
        u_ref = (
            self._last_u
            if u is None
            else _as_rvine_observations(
                u,
                operation="predict",
                expected_dimension=self.d,
                to_pobs=False,
            )
        )

        history_cache = (
            self._history_prediction_cache(
                u_ref, fitted_history=use_fitted_history_cache)
            if u_ref is not None else None
        )
        if history_cache is not None and 'train_pseudo' in history_cache:
            train_pseudo = history_cache['train_pseudo']
        else:
            from pyscarcopula._native import vine as native_vine

            train_pseudo = timed(
                "compute_pseudo_obs",
                lambda: (
                    self._compute_pseudo_obs(u_ref)
                    if (
                        u_ref is not None
                        and native_vine.pseudo_observation_trace_supported(
                            self.pair_copulas, self._trees, self._edge_map)
                    ) else None
                ),
            )
            if history_cache is not None:
                history_cache['train_pseudo'] = train_pseudo
        if suffix_state is None:
            suffix_start_col = None
            matrix = self._natural_order_matrix
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
                and not np.array_equal(matrix, self._natural_order_matrix)
            ),
            'conditional_method': 'unconditional' if not given else 'suffix',
            'updated_edges': [],
            'skipped_edges': [],
        }
        if history_cache is None:
            state_cache = {}
            posterior_cache = {}
        else:
            state_cache = history_cache['state_cache']
            posterior_cache = history_cache['posterior_cache']

        if given:
            if suffix_start_col is None:
                dag = timed(
                    "dag_build",
                    lambda: build_runtime_rvine_dag(
                        self._natural_order_matrix, self._edge_map),
                )
                plan = timed(
                    "dag_plan",
                    lambda: plan_conditional_sample(dag, given, self.d),
                )
                r_all = timed(
                    "predict_r_for_edges",
                    lambda: self._predict_r_for_edges(
                        self.pair_copulas.keys(),
                        self.pair_copulas,
                        self._edge_map,
                        n,
                        train_pseudo,
                        horizon,
                        rng,
                        predictive_r_mode=predictive_r_mode,
                        state_cache=state_cache,
                        posterior_cache=posterior_cache,
                    ),
                )
                initial = timed(
                    "dag_initial_sample",
                    lambda: self._sample_dag_given_with_r(
                        n,
                        r_all,
                        rng,
                        given,
                        plan,
                        self.pair_copulas,
                    ),
                )
                samples, mcmc_diag = timed(
                    "dag_mcmc_sample",
                    lambda: self._sample_arbitrary_given_mcmc(
                        n,
                        r_all,
                        rng,
                        given,
                        initial=initial,
                        n_steps=pcfg.mcmc_steps,
                        burnin_steps=pcfg.mcmc_burnin,
                    ),
                )
                if pcfg.return_diagnostics:
                    diagnostics['conditional_method'] = 'dag_mcmc'
                    diagnostics['dag_steps'] = tuple(dict(step) for step in plan)
                    diagnostics['dag_edges_used'] = tuple(plan.edges_used)
                    diagnostics['mcmc'] = mcmc_diag
                    if dynamic_conditioning == 'given_only':
                        diagnostics['dynamic_conditioning_reason'] = (
                            'dag_mcmc_not_suffix_supported')
                        diagnostics['skipped_edges'] = self._dynamic_skip_records(
                            self.pair_copulas,
                            self._edge_map,
                            r_all,
                            'dag_mcmc_not_suffix_supported',
                        )
                    return samples, attach_timings(diagnostics)
                return samples
            r_all = timed(
                "predict_r_for_edges",
                lambda: self._predict_r_for_edges(
                    pair_copulas.keys(),
                    pair_copulas,
                    edge_map,
                    n,
                    train_pseudo,
                    horizon,
                    rng,
                    predictive_r_mode=predictive_r_mode,
                    state_cache=state_cache,
                    posterior_cache=posterior_cache,
                ),
            )
            if dynamic_conditioning == 'given_only':
                r_all, dynamic_diag = timed(
                    "dynamic_update",
                    lambda: self._apply_given_only_dynamic_updates_ordered(
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
                        posterior_cache=posterior_cache,
                    ),
                )
                diagnostics['updated_edges'] = dynamic_diag['updated_edges']
                diagnostics['skipped_edges'] = dynamic_diag['skipped_edges']
            samples = timed(
                "suffix_sample",
                lambda: self._sample_suffix_given_with_r(
                    n,
                    r_all,
                    rng,
                    given,
                    suffix_start_col,
                    matrix=matrix,
                    pair_copulas=pair_copulas,
                ),
            )
            if pcfg.return_diagnostics:
                return samples, attach_timings(diagnostics)
            return samples
        r_all = timed(
            "predict_r_for_edges",
            lambda: self._predict_r_for_edges(
                pair_copulas.keys(),
                pair_copulas,
                edge_map,
                n,
                train_pseudo,
                horizon,
                rng,
                predictive_r_mode=predictive_r_mode,
                state_cache=state_cache,
                posterior_cache=posterior_cache,
            ),
        )
        samples = timed(
            "unconditional_sample",
            lambda: self._sample_with_r(n, r_all, rng),
        )
        if pcfg.return_diagnostics:
            return samples, attach_timings(diagnostics)
        return samples


# Canonical runtime and compatibility name are intentionally the same type.
RVineCopula = VineCopula
