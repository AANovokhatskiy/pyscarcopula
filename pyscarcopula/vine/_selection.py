"""
vine._selection — copula family selection for vine edges.

Two-phase approach (mirroring pyvinecopulib):
  Phase 1 — Itau screening: compute r = itau(tau) for each
    (family, rotation), evaluate logL analytically (no optimizer).
  Phase 2 — Refinement: run L-BFGS-B on the top-N candidates.
"""

from functools import lru_cache
from typing import NamedTuple
import warnings

import numpy as np

from pyscarcopula._native import model_policy, statistics
from pyscarcopula._native.errors import NativeUnsupported
from pyscarcopula._native.registry import is_registered_type
from pyscarcopula.copula.base import BivariateCopula


class SelectedCopula(NamedTuple):
    """Result returned by :func:`select_best_copula`.

    The tuple shape is intentionally ``(copula, result)`` for backward
    compatibility with existing ``copula, result = ...`` callers.
    """

    copula: object
    result: object


def validate_pair_candidates(candidates):
    """Require exact registered built-in pair classes in vine family pools."""
    if candidates is None:
        return
    for candidate in candidates:
        if not isinstance(candidate, type):
            raise TypeError(
                "vine candidates must be exact registered copula classes")
        if not is_registered_type(candidate):
            name = getattr(candidate, "__name__", type(candidate).__name__)
            raise NativeUnsupported(
                f"{name} is not an exact registered native model")
        if not issubclass(candidate, BivariateCopula):
            name = getattr(candidate, "__name__", type(candidate).__name__)
            hint = ""
            if name == "GaussianCopula":
                hint = (
                    "; use BivariateGaussianCopula for Gaussian vine pair "
                    "edges"
                )
            raise TypeError(
                f"{name} is multivariate and cannot be used as a vine pair "
                f"copula{hint}")


def validate_fixed_copula_specs(copulas, d=None, *, arg_name="copulas"):
    """Validate fixed vine edge family specs passed via ``copulas=``."""
    if copulas is None:
        return
    if not isinstance(copulas, (list, tuple)):
        raise TypeError(
            f"{arg_name}= expects a list of tree-level lists containing "
            "(CopulaClass, rotation) pairs; use candidates= to provide a "
            "family pool")

    def _looks_like_edge_spec(value):
        return (
            isinstance(value, (list, tuple))
            and len(value) == 2
            and isinstance(value[0], type)
        )

    if copulas and (
            not isinstance(copulas[0], (list, tuple))
            or _looks_like_edge_spec(copulas[0])):
        raise TypeError(
            f"{arg_name}= expects a list-of-lists of "
            "(CopulaClass, rotation) pairs, not copula instances or a "
            "flat edge-spec list. Use candidates= to provide candidate "
            "family classes.")

    if d is not None and len(copulas) != max(int(d) - 1, 0):
        raise ValueError(
            f"{arg_name}= must contain {max(int(d) - 1, 0)} tree levels "
            f"for d={int(d)}, got {len(copulas)}")

    for tree_index, tree in enumerate(copulas):
        if not isinstance(tree, (list, tuple)):
            raise TypeError(
                f"{arg_name}= expects a list-of-lists of "
                "(CopulaClass, rotation) pairs; tree {tree_index} is "
                f"{type(tree).__name__}")
        if d is not None:
            expected_edges = int(d) - tree_index - 1
            if len(tree) != expected_edges:
                raise ValueError(
                    f"{arg_name}[{tree_index}] must contain "
                    f"{expected_edges} edge specs for d={int(d)}, "
                    f"got {len(tree)}")
        for edge_index, spec in enumerate(tree):
            if not isinstance(spec, (list, tuple)) or len(spec) != 2:
                raise TypeError(
                    f"{arg_name}= expects fixed edge specs "
                    "(CopulaClass, rotation); got "
                    f"{type(spec).__name__} at tree {tree_index}, "
                    f"edge {edge_index}. Use candidates= to provide "
                    "candidate family classes.")
            copula_class, _rotation = spec
            if not isinstance(copula_class, type):
                raise TypeError(
                    f"{arg_name}[{tree_index}][{edge_index}] must start "
                    "with a copula class, not "
                    f"{type(copula_class).__name__}. Use candidates= for "
                    "family candidates or pass (CopulaClass, rotation).")
            validate_pair_candidates([copula_class])

def _default_candidates():
    """Default set of bivariate copula classes to try."""
    from pyscarcopula import (GumbelCopula, ClaytonCopula, FrankCopula,
                                JoeCopula, BivariateGaussianCopula)
    return [GumbelCopula, ClaytonCopula, FrankCopula, JoeCopula,
            BivariateGaussianCopula]


@lru_cache(maxsize=None)
def _all_rotations(copula_class):
    """Get valid rotations for a copula class."""
    cop = copula_class()
    if hasattr(cop, 'rotatable') and not cop.rotatable:
        return (0,)
    try:
        copula_class(rotate=180)
        return (0, 90, 180, 270)
    except (ValueError, TypeError):
        return (0,)


def _kendall_tau(u1, u2):
    """Kendall's tau value with the same fast path used by vine structure."""
    from pyscarcopula.vine._structure import _kendall_tau_value
    return _kendall_tau_value(u1, u2)


def _itau_initial_param(copula, tau_value):
    """Delegate itau starts and exact dependence limits to native policy."""
    return model_policy.pair_mle_initial_parameter(copula, tau_value)


def _tau_for_itau(cop_class, tau_value):
    """Return the family-scale tau without altering interior observations."""
    from pyscarcopula.copula.elliptical import BivariateGaussianCopula

    return statistics.tau_for_itau(
        tau_value,
        preserve_sign=cop_class is BivariateGaussianCopula,
    )


def _rotation_compatible(tau, rotate):
    """Check if rotation is compatible with sign of Kendall's tau."""
    return statistics.rotation_compatible(tau, rotate)


def select_best_copula(u1, u2, candidates, allow_rotations=True,
                       criterion='aic', transform_type='softplus', *,
                       u_pair=None, tau_value=None, config=None,
                       fit_kwargs=None):
    """
    Select best bivariate copula for (u1, u2) by AIC/BIC/logL.

    Two-phase approach:
      Phase 1 — Itau screening: rank by AIC/BIC, keep top-N.
      Phase 2 — Refinement: L-BFGS-B on top-N, pick winner.

    Always includes IndependentCopula as a baseline competitor.

    Parameters
    ----------
    u1, u2 : (T,) arrays
    candidates : list of copula classes
    allow_rotations : bool
    criterion : 'aic', 'bic', or 'loglik'
    transform_type : str
        Parameter transform forwarded to compatible candidate constructors.
    u_pair : (T, 2) array, optional
        Precomputed ``column_stack((u1, u2))``. When supplied, it must contain
        the same observations as ``u1`` and ``u2`` in the same order.
    tau_value : float, optional
        Precomputed Kendall's tau for ``u1`` and ``u2``. When omitted, the
        statistic is computed internally.
    config : NumericalConfig, optional
        Native thread policy and MLE optimizer defaults.
    fit_kwargs : dict, optional
        MLE optimizer overrides. A natural-space ``alpha0`` is accepted only
        with one non-independent candidate family; otherwise use itau starts.

    Returns
    -------
    SelectedCopula
        Named tuple with ``.copula`` and ``.result`` fields. It can still be
        unpacked as ``best_copula, best_result`` for backward compatibility.
    """
    validate_pair_candidates(candidates)

    from pyscarcopula.copula.independent import IndependentCopula
    from pyscarcopula.copula.elliptical import BivariateGaussianCopula
    from pyscarcopula._types import DEFAULT_CONFIG, IndependentResult
    from pyscarcopula.strategy._base import partition_strategy_fit_kwargs

    config = DEFAULT_CONFIG if config is None else config
    _, fit_kwargs = partition_strategy_fit_kwargs("mle", dict(fit_kwargs or {}))
    if fit_kwargs.get("alpha0") is not None and len({
            candidate for candidate in candidates
            if candidate is not IndependentCopula}) != 1:
        raise ValueError(
            "alpha0 with automatic family selection requires exactly one "
            "non-independent candidate family; omit alpha0 or fix copulas")

    T = len(u1)
    if u_pair is None:
        u_pair = np.column_stack((u1, u2))

    tau = _kendall_tau(u1, u2) if tau_value is None else float(tau_value)

    indep = IndependentCopula()
    indep_result = IndependentResult(
        log_likelihood=0.0, method='MLE',
        copula_name=indep.name, success=True)

    # ── Phase 1: itau screening ──────────────────────────────
    itau_candidates = []
    numerical_failures = []

    for cop_class in candidates:
        if cop_class is IndependentCopula:
            continue

        rotations = _all_rotations(cop_class) if allow_rotations else (0,)
        tau_for_family = _tau_for_itau(cop_class, tau)
        if tau_for_family is None:
            continue

        for angle in rotations:
            if (cop_class is not BivariateGaussianCopula
                    and not _rotation_compatible(tau, angle)):
                continue

            try:
                try:
                    cop = cop_class(rotate=angle, transform_type=transform_type)
                except TypeError:
                    cop = cop_class(rotate=angle)

                r0 = _itau_initial_param(cop, tau_for_family)
                if r0 is None:
                    continue

                logL, evaluator = _screen_log_likelihood(
                    cop, u_pair, float(r0), n_threads=config.n_threads)

                if not statistics.is_finite(logL):
                    raise FloatingPointError("non-finite screening likelihood")

                n_params = 1
                score = statistics.candidate_score(
                    logL, n_params, T, criterion)

                itau_candidates.append([score, cop, r0, evaluator])
                _retain_top_prepared_evaluators(itau_candidates, 3)
            except FloatingPointError as exc:
                message = (
                    f"{cop_class.__name__}(rotate={angle}) screening failed: "
                    f"{exc}")
                numerical_failures.append(message)
                warnings.warn(message, RuntimeWarning, stacklevel=2)

    # ── Phase 2: refine top-3 ────────────────────────────────
    itau_candidates.sort(key=lambda x: x[0])
    n_refine = min(3, len(itau_candidates))

    best_score = 0.0  # independence baseline
    best_copula = indep
    best_result = indep_result
    refined_count = 0

    for idx in range(n_refine):
        _, cop, r0, evaluator = itau_candidates[idx]
        try:
            refine_kwargs = dict(fit_kwargs)
            if refine_kwargs.get("alpha0") is None:
                refine_kwargs["alpha0"] = np.array([r0], dtype=np.float64)
            result = _fit_mle_direct(
                cop, u_pair, evaluator=evaluator, config=config,
                **refine_kwargs)
            logL = result.log_likelihood
            if not statistics.is_finite(logL):
                raise FloatingPointError("non-finite fitted likelihood")
            refined_count += 1

            n_params = 1
            score = statistics.candidate_score(
                logL, n_params, T, criterion)

            if score < best_score:
                best_score = score
                best_copula = cop
                best_result = result
        except FloatingPointError as exc:
            message = (
                f"{type(cop).__name__}(rotate={cop.rotate}) refinement failed: "
                f"{exc}")
            numerical_failures.append(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)

    if numerical_failures and refined_count == 0:
        raise FloatingPointError(
            "No candidate family could be evaluated: "
            + "; ".join(numerical_failures))

    return SelectedCopula(best_copula, best_result)


def _screen_log_likelihood(copula, u_pair, parameter, *, n_threads=1):
    """Evaluate screening logL and retain reusable native state when safe."""
    from pyscarcopula.copula.base import BivariateCopula
    from pyscarcopula._native import static as static_likelihood
    from pyscarcopula._native.registry import registry_entry_for

    registry_entry_for(copula)

    if static_likelihood.supported(copula):
        evaluator = static_likelihood.prepare(
            copula, u_pair, n_threads=n_threads)
        return float(evaluator.log_likelihood(parameter)), evaluator
    raise NativeUnsupported(
        f"{type(copula).__name__} has no registered native static likelihood")


def _retain_top_prepared_evaluators(candidates, limit):
    """Keep native observation copies only for the current stable top-N."""
    if len(candidates) <= limit:
        return
    retained = set(sorted(
        range(len(candidates)),
        key=lambda index: candidates[index][0],
    )[:limit])
    for index, candidate in enumerate(candidates):
        if index not in retained:
            candidate[3] = None


def _fit_mle_direct(
        copula, u_pair, alpha0=None, evaluator=None, *, config=None, **kwargs):
    """Fit MLE without the public API dispatch overhead."""
    from pyscarcopula.strategy.mle import MLEStrategy

    return MLEStrategy(config=config).fit(
        copula,
        u_pair,
        alpha0=alpha0,
        _prepared_evaluator=evaluator,
        **kwargs,
    )
