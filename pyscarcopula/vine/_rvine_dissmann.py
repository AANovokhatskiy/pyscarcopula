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
from pyscarcopula.vine._selection import select_best_copula, _default_candidates
from pyscarcopula.vine._structure import _build_tree_0, _build_next_tree


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

    Returns
    -------
    trees_repr : list of ``(d - 1)`` lists
        ``trees_repr[t][i]`` = ``(conditioned_fz, conditioning_fz)``.
    fitted : list of ``(d - 1)`` lists
        ``fitted[t][i]`` = ``PairCopula``.
    """
    u = np.asarray(u, dtype=np.float64)
    if u.ndim != 2:
        raise ValueError(f"select_rvine: u must be 2D, got shape {u.shape}")
    T, d = u.shape
    if d < 2:
        raise ValueError(f"select_rvine: need d >= 2, got d={d}")
    if truncation_fill not in ('mle', 'independent'):
        raise ValueError(
            "truncation_fill must be 'mle' or 'independent', "
            f"got {truncation_fill!r}"
        )
    if threshold is not None and threshold < 0:
        raise ValueError(f"threshold must be >= 0 or None, got {threshold}")

    candidates = candidates if candidates is not None else _default_candidates()

    pseudo_obs = {(i, frozenset()): u[:, i].copy() for i in range(d)}

    trees_repr = []
    fitted = []

    # ── Tree 0 ─────────────────────────────────────────────────
    _, repr_0 = _build_tree_0(u)
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

    # ── Trees 1 .. d-2 ─────────────────────────────────────────
    for t in range(1, d - 1):
        _, new_repr = _build_next_tree(
            t, trees_repr[-1], pseudo_obs,
            truncation_level=truncation_level,
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
    from scipy.stats import kendalltau

    is_truncated = (truncation_level is not None and t >= truncation_level)

    fitted_level = []
    for edge_idx, (conditioned, conditioning) in enumerate(tree_repr):
        v1, v2 = sorted(conditioned)

        u1 = _clip(pseudo_obs[(v1, conditioning)])
        u2 = _clip(pseudo_obs[(v2, conditioning)])
        u_pair = np.column_stack((u1, u2))

        tau_val, _ = kendalltau(u1, u2)
        if np.isnan(tau_val):
            tau_val = 0.0

        if (
            is_truncated and truncation_fill == 'independent'
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
