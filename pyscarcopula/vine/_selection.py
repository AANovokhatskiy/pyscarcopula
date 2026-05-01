"""
vine._selection — copula family selection for vine edges.

Two-phase approach (mirroring pyvinecopulib):
  Phase 1 — Itau screening: compute r = itau(tau) for each
    (family, rotation), evaluate logL analytically (no optimizer).
  Phase 2 — Refinement: run L-BFGS-B on the top-N candidates.
"""

import numpy as np


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


def _itau_initial_param(cop_class, tau_value, rotate):
    """Compute initial copula parameter from Kendall's tau.

    Returns parameter in the copula's natural domain (before inv_transform).
    Returns None if no closed-form available.
    """
    from pyscarcopula.copula.gumbel import GumbelCopula
    from pyscarcopula.copula.clayton import ClaytonCopula
    from pyscarcopula.copula.frank import FrankCopula
    from pyscarcopula.copula.joe import JoeCopula
    from pyscarcopula.copula.elliptical import BivariateGaussianCopula

    if cop_class is BivariateGaussianCopula:
        rho = np.sin(np.pi * tau_value / 2.0)
        return np.clip(rho, -0.99, 0.99)

    tau = max(abs(tau_value), 0.01)

    if cop_class is GumbelCopula:
        theta = 1.0 / (1.0 - min(tau, 0.95))
        return max(theta, 1.001)

    if cop_class is ClaytonCopula:
        theta = 2.0 * tau / (1.0 - min(tau, 0.95))
        return max(theta, 0.01)

    if cop_class is FrankCopula:
        return max(9.0 * tau, 0.1)

    if cop_class is JoeCopula:
        if tau < 0.2:
            theta = 1.0 + 2.0 * tau
        else:
            theta = (1.0 + np.sqrt(1.0 + 8.0 / (1.0 - min(tau, 0.95)))) / 2.0
        return max(theta, 1.001)

    return None


def _rotation_compatible(tau, rotate):
    """Check if rotation is compatible with sign of Kendall's tau."""
    if abs(tau) < 0.15:
        return True
    if rotate == 0 or rotate == 180:
        return tau > 0
    else:
        return tau < 0


def select_best_copula(u1, u2, candidates, allow_rotations=True,
                       criterion='aic', transform_type='xtanh'):
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
    best_copula : fitted BivariateCopula instance
    best_result : fit result
    """
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
                tau_for_family = (
                    tau if cop_class is BivariateGaussianCopula
                    else abs(tau)
                )
                r0 = _itau_initial_param(cop_class, tau_for_family, angle)
                if r0 is None:
                    continue

                try:
                    cop = cop_class(rotate=angle, transform_type=transform_type)
                except TypeError:
                    cop = cop_class(rotate=angle)

                r0_arr = np.atleast_1d(np.asarray(r0, dtype=np.float64))
                logL = float(np.sum(cop.log_pdf(u1, u2, r0_arr)))

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

    return best_copula, best_result


def _fit_mle_direct(copula, u_pair, alpha0=None):
    """Fit MLE without the public API dispatch overhead."""
    from pyscarcopula.strategy.mle import MLEStrategy

    return MLEStrategy().fit(copula, u_pair, alpha0=alpha0)
