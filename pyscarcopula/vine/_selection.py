"""
vine._selection — copula family selection for vine edges.

Two-phase approach (mirroring pyvinecopulib):
  Phase 1 — Itau screening: compute r = itau(tau) for each
    (family, rotation), evaluate logL analytically (no optimizer).
  Phase 2 — Refinement: run L-BFGS-B on the top-N candidates.
"""

from typing import NamedTuple

import numpy as np

from pyscarcopula.copula.base import CopulaCapabilities


class SelectedCopula(NamedTuple):
    """Result returned by :func:`select_best_copula`.

    The tuple shape is intentionally ``(copula, result)`` for backward
    compatibility with existing ``copula, result = ...`` callers.
    """

    copula: object
    result: object


def validate_pair_candidates(candidates):
    """Reject explicitly multivariate classes from vine family pools."""
    if candidates is None:
        return
    for candidate in candidates:
        capabilities = getattr(candidate, "_capabilities", None)
        if (
                isinstance(capabilities, CopulaCapabilities)
                and not capabilities.supports_pair_ops):
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


def _all_rotations(copula_class):
    """Get valid rotations for a copula class."""
    cop = copula_class()
    if hasattr(cop, 'rotatable') and not cop.rotatable:
        return [0]
    try:
        copula_class(rotate=180)
        return [0, 90, 180, 270]
    except (ValueError, TypeError):
        return [0]


def _kendall_tau(u1, u2):
    """Kendall's tau value with the same fast path used by vine structure."""
    from pyscarcopula.vine._structure import _kendall_tau_value
    return _kendall_tau_value(u1, u2)


def _itau_initial_param(copula, tau_value):
    """Compute initial copula parameter from Kendall's tau.

    Returns parameter in the copula's natural domain (before inv_transform).
    Returns None when the candidate does not implement the public mapping.
    """
    try:
        parameter = copula.tau_to_param(
            np.array([tau_value], dtype=np.float64))
    except NotImplementedError:
        return None
    parameter = np.asarray(parameter, dtype=np.float64).reshape(-1)
    if parameter.size != 1 or not np.isfinite(parameter[0]):
        raise ValueError("tau_to_param must return one finite parameter")
    return float(parameter[0])


def _tau_for_itau(cop_class, tau_value):
    """Return the family-scale tau without altering interior observations."""
    from pyscarcopula.copula.elliptical import BivariateGaussianCopula

    tau = (
        float(tau_value)
        if cop_class is BivariateGaussianCopula
        else abs(float(tau_value))
    )
    if tau == 0.0:
        return None
    if tau >= 1.0 or tau <= -1.0:
        # Perfect concordance is attained only at a parameter boundary and
        # has no finite itau start for the screening likelihood.
        return None
    return tau


def _rotation_compatible(tau, rotate):
    """Check if rotation is compatible with sign of Kendall's tau."""
    if abs(tau) < 0.15:
        return True
    if rotate == 0 or rotate == 180:
        return tau > 0
    else:
        return tau < 0


def select_best_copula(u1, u2, candidates, allow_rotations=True,
                       criterion='aic', transform_type='softplus'):
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

    Returns
    -------
    SelectedCopula
        Named tuple with ``.copula`` and ``.result`` fields. It can still be
        unpacked as ``best_copula, best_result`` for backward compatibility.
    """
    validate_pair_candidates(candidates)

    from pyscarcopula.copula.independent import IndependentCopula
    from pyscarcopula.copula.elliptical import BivariateGaussianCopula
    from pyscarcopula._types import IndependentResult

    T = len(u1)
    u_pair = np.column_stack((u1, u2))

    tau = _kendall_tau(u1, u2)

    indep = IndependentCopula()
    indep_result = IndependentResult(
        log_likelihood=0.0, method='MLE',
        copula_name=indep.name, success=True)

    # ── Phase 1: itau screening ──────────────────────────────
    itau_candidates = []

    for cop_class in candidates:
        if cop_class is IndependentCopula:
            continue

        rotations = _all_rotations(cop_class) if allow_rotations else [0]

        for angle in rotations:
            if (cop_class is not BivariateGaussianCopula
                    and not _rotation_compatible(tau, angle)):
                continue

            try:
                try:
                    cop = cop_class(rotate=angle, transform_type=transform_type)
                except TypeError:
                    cop = cop_class(rotate=angle)

                tau_for_family = _tau_for_itau(cop_class, tau)
                if tau_for_family is None:
                    continue
                r0 = _itau_initial_param(cop, tau_for_family)
                if r0 is None:
                    continue

                logL = float(cop.log_likelihood(u_pair, float(r0)))

                if not np.isfinite(logL):
                    continue

                n_params = 1
                if criterion == 'aic':
                    score = -2 * logL + 2 * n_params
                elif criterion == 'bic':
                    score = -2 * logL + n_params * np.log(T)
                else:
                    score = -logL

                itau_candidates.append((score, cop, r0))
            except Exception:
                continue

    # ── Phase 2: refine top-3 ────────────────────────────────
    itau_candidates.sort(key=lambda x: x[0])
    n_refine = min(3, len(itau_candidates))

    best_score = 0.0  # independence baseline
    best_copula = indep
    best_result = indep_result

    for idx in range(n_refine):
        _, cop, r0 = itau_candidates[idx]
        try:
            x0 = cop.inv_transform(
                np.atleast_1d(np.array([r0], dtype=np.float64)))
            alpha0 = np.atleast_1d(x0)[0:1]
            result = _fit_mle_direct(cop, u_pair, alpha0=alpha0)
            logL = result.log_likelihood

            n_params = 1
            if criterion == 'aic':
                score = -2 * logL + 2 * n_params
            elif criterion == 'bic':
                score = -2 * logL + n_params * np.log(T)
            else:
                score = -logL

            if score < best_score:
                best_score = score
                best_copula = cop
                best_result = result
        except Exception:
            continue

    return SelectedCopula(best_copula, best_result)


def _fit_mle_direct(copula, u_pair, alpha0=None):
    """Fit MLE without the public API dispatch overhead."""
    from pyscarcopula.strategy.mle import MLEStrategy

    return MLEStrategy().fit(copula, u_pair, alpha0=alpha0)
