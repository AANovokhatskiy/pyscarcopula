"""Shared pair-edge fitting for an already specified regular vine."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from pyscarcopula.copula.independent import IndependentCopula
from pyscarcopula.numerical._arrays import as_pseudo_observation_array
from pyscarcopula.vine._helpers import _clip_unit
from pyscarcopula.vine._pair_copula import PairCopula
from pyscarcopula.vine._selection import (
    _default_candidates,
    select_best_copula,
    validate_fixed_copula_specs,
    validate_pair_candidates,
)
from pyscarcopula.vine._structure import RVineMatrix, _kendall_tau_value


@dataclass(frozen=True)
class VineEdgeFit:
    """Fitted edge state indexed by canonical ``(tree, edge_index)``."""

    pair_copulas: dict
    fit_diagnostics: dict
    log_likelihood: float
    parameter_count: int
    actual_methods: dict
    fallback_count: int
    fallback_edges: tuple

    def as_levels(self):
        """Return pair copulas as the historical list-of-levels layout."""
        n_trees = max((tree for tree, _ in self.pair_copulas), default=-1) + 1
        return [
            [
                self.pair_copulas[(tree, edge)]
                for edge in range(n_trees - tree)
            ]
            for tree in range(n_trees)
        ]


def _materialize_trees(trees):
    try:
        return [
            [
                (
                    frozenset(conditioned),
                    frozenset(conditioning),
                )
                for conditioned, conditioning in level
            ]
            for level in trees
        ]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "fit_vine_edges: trees must contain "
            "(conditioned, conditioning) edge pairs") from exc


def _validate_fit_policy(
        truncation_level, truncation_fill, threshold):
    if (
            truncation_level is not None
            and (
                isinstance(truncation_level, (bool, np.bool_))
                or not isinstance(truncation_level, (int, np.integer))
            )):
        raise TypeError("truncation_level must be an integer or None")
    if truncation_level is not None and int(truncation_level) < 0:
        raise ValueError("truncation_level must be >= 0 or None")
    if truncation_fill not in ("mle", "independent"):
        raise ValueError(
            "truncation_fill must be 'mle' or 'independent', "
            f"got {truncation_fill!r}")
    if threshold is not None and threshold < 0:
        raise ValueError(f"threshold must be >= 0 or None, got {threshold}")


def fit_vine_edges(
    u,
    trees,
    *,
    candidates=None,
    copulas=None,
    method="mle",
    criterion="aic",
    allow_rotations=True,
    truncation_level=None,
    truncation_fill="independent",
    threshold=0.0,
    min_edge_logL=None,
    transform_type="xtanh",
    config=None,
    fit_kwargs=None,
    _pre_fitted=None,
):
    """Fit pair copulas for a validated, fixed regular-vine structure.

    Structure validation completes before family selection, optimizers, or
    native numerical strategies are invoked.
    """
    u = as_pseudo_observation_array(
        u, name="u", allow_boundary=False)
    if u.ndim != 2:
        raise ValueError(
            f"fit_vine_edges: u must be 2D, got shape {u.shape}")
    if u.shape[0] == 0:
        raise ValueError("fit_vine_edges: u must contain at least one row")
    _, d = u.shape
    if d < 2:
        raise ValueError(f"fit_vine_edges: need d >= 2, got d={d}")

    tree_levels = _materialize_trees(trees)
    RVineMatrix.from_trees(d, tree_levels)
    _validate_fit_policy(
        truncation_level, truncation_fill, threshold)

    candidates = (
        candidates if candidates is not None else _default_candidates())
    validate_pair_candidates(candidates)
    validate_fixed_copula_specs(copulas, d)

    if _pre_fitted is not None:
        fitted_levels = [list(level) for level in _pre_fitted]
        if (
                len(fitted_levels) != len(tree_levels)
                or any(
                    len(fitted) != len(edges)
                    for fitted, edges in zip(fitted_levels, tree_levels)
                )
                or any(
                    not isinstance(pair, PairCopula)
                    for level in fitted_levels
                    for pair in level
                )):
            raise ValueError(
                "fit_vine_edges: pre-fitted working edges do not match trees")
        return _build_vine_edge_fit(
            fitted_levels, requested_method=method)

    pseudo_obs = {
        (variable, frozenset()): u[:, variable].copy()
        for variable in range(d)
    }
    fitted_levels = []
    fit_kwargs = dict(fit_kwargs or {})
    for tree, level in enumerate(tree_levels):
        fitted_levels.append(_fit_tree_level(
            tree,
            level,
            pseudo_obs,
            d,
            candidates=candidates,
            allow_rotations=allow_rotations,
            criterion=criterion,
            method=method,
            copulas=copulas[tree] if copulas is not None else None,
            config=config,
            truncation_level=truncation_level,
            truncation_fill=truncation_fill,
            threshold=threshold,
            min_edge_logL=min_edge_logL,
            transform_type=transform_type,
            fit_kwargs=fit_kwargs,
        ))

    return _build_vine_edge_fit(fitted_levels, requested_method=method)


def _fit_tree_level(
    t,
    tree_repr,
    pseudo_obs,
    d,
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
    fit_with_strategy=None,
):
    """Fit one level and populate pseudo-observations for the next level."""
    if fit_with_strategy is None:
        fit_with_strategy = _fit_with_strategy
    is_truncated = (
        truncation_level is not None and t >= truncation_level)

    fitted_level = []
    for edge_idx, (conditioned, conditioning) in enumerate(tree_repr):
        fit_started = perf_counter()
        edge_fit_diagnostics = {
            "requested_method": str(method).upper(),
            "dynamic_attempted": False,
            "fallback_used": False,
            "fallback_reason": None,
            "attempted_method": None,
            "attempted_success": None,
            "attempted_nfev": 0,
            "attempted_message": None,
            "selection_nfev": 0,
            "selection_ms": 0.0,
            "dynamic_fit_ms": 0.0,
        }
        selection_result = None
        v1, v2 = sorted(conditioned)

        u1 = _clip_unit(pseudo_obs[(v1, conditioning)])
        u2 = _clip_unit(pseudo_obs[(v2, conditioning)])
        u_pair = np.column_stack((u1, u2))

        force_independent = (
            is_truncated and truncation_fill == "independent")
        if force_independent:
            tau_val = 0.0
        else:
            tau_val = _kendall_tau_value(u1, u2)
            if np.isnan(tau_val):
                tau_val = 0.0

        if (
                force_independent
                or threshold is not None and abs(tau_val) < threshold):
            selection_started = perf_counter()
            copula = IndependentCopula()
            result = copula._fit_validated(u_pair)
            edge_fit_diagnostics["selection_ms"] = (
                1e3 * (perf_counter() - selection_started))
            edge_fit_diagnostics["selection_nfev"] = int(
                getattr(result, "nfev", 0) or 0)
        else:
            selection_started = perf_counter()
            if copulas is not None:
                copula = _make_fixed_copula(
                    copulas[edge_idx], transform_type)
                if isinstance(copula, IndependentCopula):
                    selection_result = copula._fit_validated(u_pair)
                else:
                    selection_result = fit_with_strategy(
                        copula, u_pair, "mle", config, fit_kwargs)
            else:
                copula, selection_result = select_best_copula(
                    u1,
                    u2,
                    candidates,
                    allow_rotations,
                    criterion,
                    transform_type=transform_type,
                    u_pair=u_pair,
                    tau_value=tau_val,
                )
            edge_fit_diagnostics["selection_ms"] = (
                1e3 * (perf_counter() - selection_started))
            edge_fit_diagnostics["selection_nfev"] = int(
                getattr(selection_result, "nfev", 0) or 0)

            if (
                    min_edge_logL is not None
                    and selection_result.log_likelihood < min_edge_logL
                    and not isinstance(copula, IndependentCopula)):
                copula = IndependentCopula()
                result = copula._fit_validated(u_pair)
            elif (
                    is_truncated
                    or str(method).lower() == "mle"
                    or isinstance(copula, IndependentCopula)):
                result = selection_result
            else:
                dynamic_fit_kwargs = dict(fit_kwargs)
                dynamic_fit_kwargs["initial_mle_result"] = selection_result
                dynamic_started = perf_counter()
                dynamic_result = fit_with_strategy(
                    copula,
                    u_pair,
                    method,
                    config,
                    dynamic_fit_kwargs,
                )
                edge_fit_diagnostics["dynamic_fit_ms"] = (
                    1e3 * (perf_counter() - dynamic_started))
                edge_fit_diagnostics.update({
                    "dynamic_attempted": True,
                    "attempted_method": str(
                        getattr(dynamic_result, "method", method)).upper(),
                    "attempted_success": bool(
                        getattr(dynamic_result, "success", True)),
                    "attempted_nfev": int(
                        getattr(dynamic_result, "nfev", 0) or 0),
                    "attempted_message": str(
                        getattr(dynamic_result, "message", "") or ""),
                })
                result = dynamic_result
                if not edge_fit_diagnostics["attempted_success"]:
                    edge_fit_diagnostics["fallback_used"] = True
                    edge_fit_diagnostics["fallback_reason"] = (
                        "dynamic_fit_unsuccessful")
                    result = selection_result

        edge_fit_diagnostics["actual_method"] = str(
            getattr(result, "method", None) or "STATIC").upper()
        edge_fit_diagnostics["actual_success"] = bool(
            getattr(result, "success", True))
        edge_fit_diagnostics["fit_ms"] = (
            1e3 * (perf_counter() - fit_started))
        pair = _pair_from_result(
            copula,
            result,
            tau_val,
            fit_diagnostics=edge_fit_diagnostics,
        )
        fitted_level.append(pair)

        if t < d - 2:
            u1_next, u2_next = pair.h_pair(u1, u2)
            pseudo_obs[(v2, conditioning | {v1})] = _clip_unit(u2_next)
            pseudo_obs[(v1, conditioning | {v2})] = _clip_unit(u1_next)

    return fitted_level


def _make_fixed_copula(spec, transform_type):
    copula_class, rotation = spec
    try:
        return copula_class(
            rotate=rotation, transform_type=transform_type)
    except TypeError:
        return copula_class(rotate=rotation)


def _fit_with_strategy(copula, u_pair, method, config, fit_kwargs):
    from pyscarcopula.strategy._base import get_strategy

    fit_call_kwargs = dict(fit_kwargs)
    if str(method).lower() == "mle" and "alpha0" in fit_call_kwargs:
        alpha0 = np.asarray(fit_call_kwargs["alpha0"])
        if alpha0.size != 1:
            fit_call_kwargs.pop("alpha0")
    strategy_kwargs = {
        key: value
        for key, value in fit_call_kwargs.items()
        if key not in (
            "alpha0",
            "gamma0",
            "gtol",
            "ftol",
            "maxfun",
            "maxiter",
            "maxls",
            "eps",
            "maxcor",
            "finite_diff_rel_step",
            "score_eps",
            "gamma_bound",
            "beta_bound",
            "seed",
            "dwt",
            "verbose",
        )
    }
    strategy = get_strategy(method, config=config, **strategy_kwargs)
    return strategy.fit(copula, u_pair, **fit_call_kwargs)


def _pair_from_result(
        copula, result, tau_val, *, fit_diagnostics=None):
    result_param = getattr(result, "copula_param", None)
    if result_param is None:
        param = 0.0 if isinstance(copula, IndependentCopula) else None
    else:
        param = float(result_param)
    return PairCopula(
        copula=copula,
        param=param,
        log_likelihood=float(result.log_likelihood),
        nfev=int(getattr(result, "nfev", 0) or 0),
        tau=float(tau_val),
        fit_result=result,
        fit_diagnostics=dict(fit_diagnostics or {}),
    )


def _build_vine_edge_fit(fitted_levels, *, requested_method):
    pair_copulas = {
        (tree, edge): pair
        for tree, level in enumerate(fitted_levels)
        for edge, pair in enumerate(level)
    }
    records = []
    for key, pair in pair_copulas.items():
        record = {
            "key": key,
            "family": type(pair.copula).__name__,
            "rotation": int(getattr(pair.copula, "rotate", 0)),
            **dict(pair.fit_diagnostics),
        }
        result = getattr(pair, "fit_result", None)
        record.setdefault(
            "actual_method",
            str(getattr(result, "method", requested_method)).upper(),
        )
        record.setdefault(
            "actual_success",
            bool(getattr(result, "success", True)),
        )
        record.setdefault("actual_nfev", int(
            getattr(result, "nfev", 0) or 0))
        record.setdefault("selection_nfev", record["actual_nfev"])
        record.setdefault("dynamic_attempted", False)
        record.setdefault("fallback_used", False)
        record.setdefault("fallback_reason", None)
        records.append(record)
    edge_records = tuple(records)
    actual_methods = {}
    for record in edge_records:
        method = record["actual_method"]
        actual_methods[method] = actual_methods.get(method, 0) + 1
    fallback_edges = tuple(
        record for record in edge_records if record["fallback_used"])
    diagnostics = {
        "requested_method": str(requested_method).upper(),
        "edge_count": len(pair_copulas),
        "actual_methods": dict(actual_methods),
        "fallback_count": len(fallback_edges),
        "fallback_edges": fallback_edges,
        "edges": edge_records,
    }
    return VineEdgeFit(
        pair_copulas=pair_copulas,
        fit_diagnostics=diagnostics,
        log_likelihood=float(sum(
            pair.log_likelihood for pair in pair_copulas.values())),
        parameter_count=int(sum(
            pair.n_params for pair in pair_copulas.values())),
        actual_methods=actual_methods,
        fallback_count=len(fallback_edges),
        fallback_edges=fallback_edges,
    )
