"""
vine._rvine_dissmann — MLE-only Dissmann sequential selection for R-vines.

Given pseudo-observations ``u`` of shape ``(T, d)``, build an R-vine
level by level:

    Tree 0: maximum spanning tree on ``|Kendall's tau|``.
    Tree t > 0: MST over edge candidates satisfying the proximity
        condition, weighted by ``|tau|`` on h-transformed pseudo-obs.

At each tree, every edge has a pair copula selected by
``select_best_copula`` (AIC/BIC/logL with rotation-aware itau screening +
L-BFGS-B refinement). Pseudo-observations for the next tree are obtained
via the edge's h-function.

Output is two parallel, level-indexed lists suitable for
``build_rvine_matrix_with_edge_map``:

    trees_repr[t][i] = (conditioned_fz, conditioning_fz)
    fitted[t][i]     = PairCopula

This module is MLE-only by design (Block 1). SCAR / GAS variants live
elsewhere.
"""

from dataclasses import dataclass

import numpy as np

from pyscarcopula.copula.independent import IndependentCopula
from pyscarcopula.vine._conditional_rvine import (
    find_rvine_peel_order_for_given_suffix,
)
from pyscarcopula.vine._rvine_matrix_builder import build_rvine_matrix_with_edge_map
from pyscarcopula.vine._reachability import (
    analyze_conditional_reachability,
    build_rvine_dag,
)
from pyscarcopula.vine._selection import select_best_copula, _default_candidates
from pyscarcopula.vine._structure import (
    _build_next_tree,
    _build_next_tree_conditional,
    _build_tree_0,
    _build_tree_0_conditional,
    _kendall_tau_value,
)


_EPS = 1e-10


def _clip(u):
    return np.clip(u, _EPS, 1.0 - _EPS)


@dataclass(frozen=True)
class PairCopula:
    """One fitted edge of an R-vine.

    Attributes
    ----------
    copula : BivariateCopula instance
        Family + rotation. Parameter is stored separately in ``param``.
    param : float
        MLE copula parameter (0.0 for IndependentCopula).
    log_likelihood : float
        Edge log-likelihood at the fitted parameter.
    nfev : int
        Optimizer function evaluations (0 for closed-form / Independent).
    tau : float
        Empirical Kendall's tau on the pseudo-observations used to fit
        this edge (for diagnostics only).
    fit_result : FitResult
        Strategy result for this edge (MLEResult, GASResult,
        LatentResult, or IndependentResult).
    """
    copula: object
    param: float
    log_likelihood: float
    nfev: int
    tau: float
    fit_result: object = None

    @property
    def n_params(self) -> int:
        if self.fit_result is not None:
            return int(getattr(self.fit_result, 'n_params', 0))
        return 0 if isinstance(self.copula, IndependentCopula) else 1

    def h(self, u_conditioned, u_given):
        """h(u_conditioned | u_given) using this edge's fit result."""
        from pyscarcopula.vine._rvine_edges import _edge_h
        return _edge_h(self, u_conditioned, u_given)


def select_rvine(
    u,
    *,
    candidates=None,
    allow_rotations=True,
    criterion='aic',
    method='mle',
    copulas=None,
    config=None,
    truncation_level=None,
    truncation_fill='independent',
    threshold=0.0,
    min_edge_logL=None,
    transform_type='xtanh',
    given_vars=None,
    return_diagnostics=False,
    structure_search='beam',
    beam_width=4,
    **fit_kwargs,
):
    """Run MLE-only Dissmann selection on pseudo-observations.

    Parameters
    ----------
    u : (T, d) ndarray
        Pseudo-observations in (0, 1).
    candidates : list of copula classes or None
        Family pool. ``None`` uses the package default.
    allow_rotations : bool
        Whether to search over rotations for Archimedean families.
    criterion : {'aic', 'bic', 'loglik'}
        Model selection criterion within and between families.
    method : str
        Edge fitting method: ``'mle'``, ``'gas'``, or
        ``'scar-tm-ou'``.
    copulas : list-of-lists or None
        Optional fixed edge families as ``(copula_class, rotation)`` in
        the Dissmann edge order for each tree.
    config : NumericalConfig or None
        Optional numerical configuration passed to strategies.
    truncation_level : int or None
        If set, tree levels ``>= truncation_level`` use
        ``truncation_fill``.
    truncation_fill : {'mle', 'independent'}
        For truncated trees, either fit edges with MLE only or force
        ``IndependentCopula``. The default is ``'independent'``.
    threshold : float or None
        Pre-fit Kendall's tau threshold. If ``abs(tau) < threshold``,
        the edge is set to ``IndependentCopula`` without fitting. The
        default ``0.0`` matches pyvinecopulib and disables filtering.
    min_edge_logL : float or None
        If set, any fitted edge with ``log_likelihood < min_edge_logL`` is
        replaced by ``IndependentCopula``.
    transform_type : str
        Parameter transform passed to candidate copulas.
    given_vars : iterable[int] or None
        Optional target set of variables that tree construction should keep
        compatible with the current exact sampler.
    return_diagnostics : bool, default False
        If True, also return candidate-structure diagnostics for fit-time
        conditioning support analysis.
    structure_search : {'beam', 'multi-start'}, default 'beam'
        Candidate-structure search strategy used when ``given_vars`` is set.
    beam_width : int, default 4
        Number of partial candidates retained per tree level under beam
        search. Ignored by ``multi-start``.

    Returns
    -------
    trees_repr : list of ``(d - 1)`` lists
        ``trees_repr[t][i]`` = ``(conditioned_fz, conditioning_fz)``.
    fitted : list of ``(d - 1)`` lists
        ``fitted[t][i]`` = ``PairCopula``.
    diagnostics : dict, optional
        Returned only when ``return_diagnostics=True``.
    """
    u = np.asarray(u, dtype=np.float64)
    if u.ndim != 2:
        raise ValueError(f"select_rvine: u must be 2D, got shape {u.shape}")
    _, d = u.shape
    if d < 2:
        raise ValueError(f"select_rvine: need d >= 2, got d={d}")
    if truncation_fill not in ('mle', 'independent'):
        raise ValueError(
            "truncation_fill must be 'mle' or 'independent', "
            f"got {truncation_fill!r}"
        )
    if threshold is not None and threshold < 0:
        raise ValueError(f"threshold must be >= 0 or None, got {threshold}")
    structure_search = str(structure_search).lower()
    if structure_search not in ('beam', 'multi-start'):
        raise ValueError(
            "structure_search must be 'beam' or 'multi-start', "
            f"got {structure_search!r}"
        )
    if not isinstance(beam_width, (int, np.integer)) or int(beam_width) <= 0:
        raise ValueError(f"beam_width must be positive int, got {beam_width!r}")
    beam_width = int(beam_width)

    candidates = candidates if candidates is not None else _default_candidates()
    pseudo_obs = {(i, frozenset()): u[:, i].copy() for i in range(d)}
    given_vars = tuple(given_vars or ())

    if not given_vars:
        trees_repr, fitted = _build_and_fit_candidate(
            u,
            d,
            pseudo_obs,
            candidates,
            allow_rotations,
            criterion,
            method,
            copulas,
            config,
            truncation_level,
            truncation_fill,
            threshold,
            min_edge_logL,
            transform_type,
            fit_kwargs,
            mode='fit_first',
        )
        if not return_diagnostics:
            return trees_repr, fitted
        score = _score_candidate_structure(
            trees_repr, fitted, d, given_vars, 'fit_first')
        diagnostics = _selection_diagnostics(
            given_vars,
            score,
            [score],
        )
        return trees_repr, fitted, diagnostics

    if structure_search == 'beam':
        candidate_results = _beam_search_candidates(
            u,
            d,
            pseudo_obs,
            candidates,
            allow_rotations,
            criterion,
            method,
            copulas,
            config,
            truncation_level,
            truncation_fill,
            threshold,
            min_edge_logL,
            transform_type,
            fit_kwargs,
            given_vars,
            beam_width,
        )
    else:
        candidate_results = []
        for mode in _search_modes(given_vars):
            trees_repr, fitted = _build_and_fit_candidate(
                u,
                d,
                pseudo_obs,
                candidates,
                allow_rotations,
                criterion,
                method,
                copulas,
                config,
                truncation_level,
                truncation_fill,
                threshold,
                min_edge_logL,
                transform_type,
                fit_kwargs,
                mode=mode,
                given_vars=given_vars,
            )
            candidate_results.append(_score_candidate_structure(
                trees_repr,
                fitted,
                d,
                given_vars,
                mode,
            ))

    best = max(
        candidate_results,
        key=lambda item: (
            int(item['exact_supported']),
            int(item['dag_complete']),
            -item['missing_count'],
            item['fit_score'],
            -item['mode_rank'],
            tuple(-rank for rank in item['mode_path_ranks']),
        ),
    )
    if not return_diagnostics:
        return best['trees_repr'], best['fitted']
    diagnostics = _selection_diagnostics(
        given_vars,
        best,
        candidate_results,
    )
    return best['trees_repr'], best['fitted'], diagnostics


def _search_modes(given_vars):
    if not given_vars:
        return ('fit_first',)
    return ('given_first', 'balanced', 'fit_first')


def _build_and_fit_candidate(
        u, d, pseudo_obs_seed, candidates, allow_rotations, criterion, method,
        copulas, config, truncation_level, truncation_fill, threshold,
        min_edge_logL, transform_type, fit_kwargs, *, mode, given_vars=()):
    pseudo_obs = {
        key: value.copy()
        for key, value in pseudo_obs_seed.items()
    }
    trees_repr = []
    fitted = []

    if mode == 'fit_first':
        _, repr_0 = _build_tree_0(u)
    else:
        tree0_limit = None
        if mode == 'balanced':
            tree0_limit = max(len(given_vars) // 2, 0)
        _, repr_0 = _build_tree_0_conditional(
            u,
            given_vars,
            priority_limit_override=tree0_limit,
        )
    trees_repr.append(repr_0)
    fitted_0 = _fit_tree_level(
        0, repr_0, pseudo_obs, d,
        candidates=candidates,
        allow_rotations=allow_rotations,
        criterion=criterion,
        method=method,
        copulas=copulas[0] if copulas is not None else None,
        config=config,
        truncation_level=truncation_level,
        truncation_fill=truncation_fill,
        threshold=threshold,
        min_edge_logL=min_edge_logL,
        transform_type=transform_type,
        fit_kwargs=fit_kwargs,
    )
    fitted.append(fitted_0)

    for t in range(1, d - 1):
        if mode == 'fit_first':
            _, new_repr = _build_next_tree(
                t,
                trees_repr[-1],
                pseudo_obs,
                truncation_level=truncation_level,
            )
        else:
            priority_limit = None
            if mode == 'balanced':
                base_limit = max(len(given_vars) - t - 1, 0)
                priority_limit = max((base_limit + 1) // 2, 0)
            _, new_repr = _build_next_tree_conditional(
                t,
                trees_repr[-1],
                pseudo_obs,
                given_vars,
                truncation_level=truncation_level,
                priority_limit_override=priority_limit,
            )
        if new_repr is None or len(new_repr) != d - 1 - t:
            raise RuntimeError(
                f"select_rvine: could not build tree {t} "
                f"(got {None if new_repr is None else len(new_repr)} edges, "
                f"expected {d - 1 - t})"
            )
        trees_repr.append(new_repr)
        fitted_t = _fit_tree_level(
            t, new_repr, pseudo_obs, d,
            candidates=candidates,
            allow_rotations=allow_rotations,
            criterion=criterion,
            method=method,
            copulas=copulas[t] if copulas is not None else None,
            config=config,
            truncation_level=truncation_level,
            truncation_fill=truncation_fill,
            threshold=threshold,
            min_edge_logL=min_edge_logL,
            transform_type=transform_type,
            fit_kwargs=fit_kwargs,
        )
        fitted.append(fitted_t)

    return trees_repr, fitted


def _beam_search_candidates(
        u, d, pseudo_obs_seed, candidates, allow_rotations, criterion, method,
        copulas, config, truncation_level, truncation_fill, threshold,
        min_edge_logL, transform_type, fit_kwargs, given_vars, beam_width):
    beam = []
    # Tree 0 seeds the beam. Higher trees extend each partial structure with
    # all modes, so mode_path is an intentional per-level builder trace.
    for mode in _search_modes(given_vars):
        pseudo_obs = {key: value.copy() for key, value in pseudo_obs_seed.items()}
        repr_0 = _build_tree_level_repr(
            0, u, None, pseudo_obs, mode, given_vars, truncation_level)
        fitted_0 = _fit_tree_level(
            0, repr_0, pseudo_obs, d,
            candidates=candidates,
            allow_rotations=allow_rotations,
            criterion=criterion,
            method=method,
            copulas=copulas[0] if copulas is not None else None,
            config=config,
            truncation_level=truncation_level,
            truncation_fill=truncation_fill,
            threshold=threshold,
            min_edge_logL=min_edge_logL,
            transform_type=transform_type,
            fit_kwargs=fit_kwargs,
        )
        beam.append({
            'trees_repr': [repr_0],
            'fitted': [fitted_0],
            'pseudo_obs': pseudo_obs,
            'fit_score_partial': _fit_score_levels([fitted_0]),
            'mode_path': (mode,),
        })

    beam = _prune_beam(beam, beam_width)
    for t in range(1, d - 1):
        expanded = []
        for state in beam:
            for mode in _search_modes(given_vars):
                pseudo_obs = {
                    key: value.copy()
                    for key, value in state['pseudo_obs'].items()
                }
                new_repr = _build_tree_level_repr(
                    t, u, state['trees_repr'][-1], pseudo_obs, mode,
                    given_vars, truncation_level)
                if new_repr is None or len(new_repr) != d - 1 - t:
                    continue
                fitted_t = _fit_tree_level(
                    t, new_repr, pseudo_obs, d,
                    candidates=candidates,
                    allow_rotations=allow_rotations,
                    criterion=criterion,
                    method=method,
                    copulas=copulas[t] if copulas is not None else None,
                    config=config,
                    truncation_level=truncation_level,
                    truncation_fill=truncation_fill,
                    threshold=threshold,
                    min_edge_logL=min_edge_logL,
                    transform_type=transform_type,
                    fit_kwargs=fit_kwargs,
                )
                expanded.append({
                    'trees_repr': state['trees_repr'] + [new_repr],
                    'fitted': state['fitted'] + [fitted_t],
                    'pseudo_obs': pseudo_obs,
                    'fit_score_partial': (
                        state['fit_score_partial']
                        + _fit_score_levels([fitted_t])
                    ),
                    'mode_path': state['mode_path'] + (mode,),
                })
        if not expanded:
            modes_tried = tuple(_search_modes(given_vars))
            partial_paths = tuple(state['mode_path'] for state in beam)
            raise RuntimeError(
                "select_rvine: beam search could not build a valid "
                f"tree_level={t}; expected_edges={d - 1 - t}; "
                f"beam_size={len(beam)}; beam_width={beam_width}; "
                f"given_vars={tuple(given_vars)}; "
                f"modes_tried={modes_tried}; "
                f"partial_mode_paths={partial_paths}"
            )
        beam = _prune_beam(expanded, beam_width)

    return [
        _score_candidate_structure(
            state['trees_repr'],
            state['fitted'],
            d,
            given_vars,
            state['mode_path'][0],
            mode_path=state['mode_path'],
        )
        for state in beam
    ]


def _build_tree_level_repr(
        tree_level, u, prev_repr, pseudo_obs, mode, given_vars,
        truncation_level):
    if tree_level == 0:
        if mode == 'fit_first':
            _, repr_0 = _build_tree_0(u)
            return repr_0
        tree0_limit = None
        if mode == 'balanced':
            tree0_limit = max(len(given_vars) // 2, 0)
        _, repr_0 = _build_tree_0_conditional(
            u,
            given_vars,
            priority_limit_override=tree0_limit,
        )
        return repr_0

    if mode == 'fit_first':
        _, new_repr = _build_next_tree(
            tree_level,
            prev_repr,
            pseudo_obs,
            truncation_level=truncation_level,
        )
        return new_repr

    priority_limit = None
    if mode == 'balanced':
        base_limit = max(len(given_vars) - tree_level - 1, 0)
        priority_limit = max((base_limit + 1) // 2, 0)
    _, new_repr = _build_next_tree_conditional(
        tree_level,
        prev_repr,
        pseudo_obs,
        given_vars,
        truncation_level=truncation_level,
        priority_limit_override=priority_limit,
    )
    return new_repr


def _fit_score_levels(fitted_levels):
    return float(sum(
        abs(float(pc.tau))
        for level in fitted_levels
        for pc in level
    ))


def _prune_beam(states, beam_width):
    seen = set()
    unique = []
    for state in sorted(
            states,
            key=lambda item: (
                item['fit_score_partial'],
                tuple(-_mode_rank(mode) for mode in item['mode_path']),
            ),
            reverse=True):
        signature = tuple(
            tuple((tuple(sorted(cond)), tuple(sorted(cs))) for cond, cs in level)
            for level in state['trees_repr']
        )
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(state)
        if len(unique) >= beam_width:
            break
    return unique


def _score_candidate_structure(
        trees_repr, fitted, d, given_vars, mode, *, mode_path=None):
    fit_score = _fit_score_levels(fitted)
    if given_vars:
        matrix, edge_map = build_rvine_matrix_with_edge_map(d, trees_repr)
        dag = build_rvine_dag(matrix, edge_map)
        dag_info = analyze_conditional_reachability(dag, given_vars, d)
        exact_supported = (
            find_rvine_peel_order_for_given_suffix(trees_repr, d, given_vars)
            is not None
        )
    else:
        dag_info = {
            'complete': True,
            'missing_base_vars': (),
            'reachable_base_vars': (),
            'n_known_nodes': 0,
            'n_steps': 0,
            'n_inverse_steps': 0,
        }
        exact_supported = True
    if mode_path is None:
        mode_path = (mode,)
    mode_path = tuple(mode_path)
    mode_rank = _mode_rank(mode)
    return {
        'trees_repr': trees_repr,
        'fitted': fitted,
        'mode': mode,
        'mode_path': mode_path,
        'mode_path_ranks': tuple(_mode_rank(item) for item in mode_path),
        'fit_score': fit_score,
        'dag_complete': bool(dag_info['complete']),
        'missing_count': len(dag_info['missing_base_vars']),
        'exact_supported': bool(exact_supported),
        'mode_rank': mode_rank,
        'dag_info': dag_info,
    }


def _mode_rank(mode):
    return {
        'given_first': 0,
        'balanced': 1,
        'fit_first': 2,
    }[mode]


def _selection_diagnostics(given_vars, selected, candidates):
    return {
        'target_given_vars': tuple(given_vars),
        'selected_mode': selected['mode'],
        'selected_index': int(selected['mode_rank']),
        'selected_candidate': _candidate_diagnostics(selected),
        'candidates': tuple(
            _candidate_diagnostics(candidate)
            for candidate in sorted(candidates, key=lambda item: item['mode_rank'])
        ),
    }


def _candidate_diagnostics(candidate):
    dag_info = candidate['dag_info']
    return {
        'mode': candidate['mode'],
        'mode_path': tuple(candidate.get('mode_path', (candidate['mode'],))),
        'exact_supported': bool(candidate['exact_supported']),
        'dag_complete': bool(candidate['dag_complete']),
        'fit_score': float(candidate['fit_score']),
        'missing_base_vars': tuple(dag_info['missing_base_vars']),
        'reachable_base_vars': tuple(dag_info['reachable_base_vars']),
        'n_known_nodes': int(dag_info['n_known_nodes']),
        'n_steps': int(dag_info['n_steps']),
        'n_inverse_steps': int(dag_info['n_inverse_steps']),
    }


def _fit_tree_level(
    t, tree_repr, pseudo_obs, d,
    *,
    candidates,
    allow_rotations,
    criterion,
    method,
    copulas,
    config,
    truncation_level,
    truncation_fill,
    threshold,
    min_edge_logL,
    transform_type,
    fit_kwargs,
):
    """Fit all edges at one tree level; populate pseudo_obs for next tree."""
    is_truncated = (truncation_level is not None and t >= truncation_level)

    fitted_level = []
    for edge_idx, (conditioned, conditioning) in enumerate(tree_repr):
        v1, v2 = sorted(conditioned)

        u1 = _clip(pseudo_obs[(v1, conditioning)])
        u2 = _clip(pseudo_obs[(v2, conditioning)])
        u_pair = np.column_stack((u1, u2))

        force_independent = is_truncated and truncation_fill == 'independent'
        if force_independent:
            tau_val = 0.0
        else:
            tau_val = _kendall_tau_value(u1, u2)
            if np.isnan(tau_val):
                tau_val = 0.0

        if (
            force_independent
            or threshold is not None and abs(tau_val) < threshold
        ):
            cop = IndependentCopula()
            result = cop.fit(u_pair)
            pc = _pair_from_result(cop, result, tau_val)
        else:
            if copulas is not None:
                cop = _make_fixed_copula(copulas[edge_idx], transform_type)
                if isinstance(cop, IndependentCopula):
                    selection_result = cop.fit(u_pair)
                else:
                    selection_result = _fit_with_strategy(
                        cop, u_pair, 'mle', config, fit_kwargs)
            else:
                cop, selection_result = select_best_copula(
                    u1, u2, candidates, allow_rotations, criterion,
                    transform_type=transform_type,
                )

            if (
                min_edge_logL is not None
                and selection_result.log_likelihood < min_edge_logL
                and not isinstance(cop, IndependentCopula)
            ):
                cop = IndependentCopula()
                result = cop.fit(u_pair)
            elif is_truncated or method.lower() == 'mle' or isinstance(cop, IndependentCopula):
                result = selection_result
            else:
                result = _fit_with_strategy(
                    cop, u_pair, method, config, fit_kwargs)
                if not bool(getattr(result, 'success', True)):
                    result = selection_result

            pc = _pair_from_result(cop, result, tau_val)

        fitted_level.append(pc)

        # Propagate pseudo-observations for higher trees.
        if t < d - 2:
            pseudo_obs[(v2, conditioning | {v1})] = _clip(pc.h(u2, u1))
            pseudo_obs[(v1, conditioning | {v2})] = _clip(pc.h(u1, u2))

    return fitted_level


def _make_fixed_copula(spec, transform_type):
    cop_class, rotation = spec
    try:
        return cop_class(rotate=rotation, transform_type=transform_type)
    except TypeError:
        return cop_class(rotate=rotation)


def _fit_with_strategy(copula, u_pair, method, config, fit_kwargs):
    from pyscarcopula.strategy._base import get_strategy
    strategy_kwargs = {
        key: value for key, value in fit_kwargs.items()
        if key not in ('alpha0', 'tol', 'verbose')
    }
    strategy = get_strategy(method, config=config, **strategy_kwargs)
    return strategy.fit(copula, u_pair, **fit_kwargs)


def _pair_from_result(copula, result, tau_val):
    param = float(getattr(result, 'copula_param', 0.0) or 0.0)
    return PairCopula(
        copula=copula,
        param=param,
        log_likelihood=float(result.log_likelihood),
        nfev=int(getattr(result, 'nfev', 0) or 0),
        tau=float(tau_val),
        fit_result=result,
    )
