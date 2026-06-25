"""
Goodness-of-fit tests for copula models (bivariate and vine).

Bivariate:
    MLE:  e2 = h(u2, u1, r)
    SCAR: e2 = E[h(u2, u1, Psi(x_k)) | u_{1:k-1}]  (mixture)

C-Vine (d dimensions):
    Rosenblatt transform through the tree:
    e_1 = u_1
    e_i = h_{i-1}( ... h_1(h_0(u_i | u_1) | ...) ... )
    where each h uses the copula parameter from the corresponding edge.

    MLE:  constant r per edge
    SCAR: predictive E[Psi(x_k) | u_{1:k-1}] per edge

Under correct model: e ~ iid U[0,1]^d.
Test: Phi^{-1}(e) -> chi2(d) CDF -> CvM against U[0,1].

Usage:
    from pyscarcopula.stattests import gof_test, vine_gof_test
"""

import numpy as np
import time
from dataclasses import dataclass
from scipy.special import stdtr
from scipy.stats import chi2, norm, cramervonmises

from pyscarcopula._utils import (
    clip_pseudo_observations,
    clip_pseudo_observations_no_copy,
    clip_rosenblatt_output,
)


@dataclass(frozen=True)
class BootstrapGoFResult:
    """Goodness-of-fit result with bootstrap-calibrated p-value."""

    statistic: float
    pvalue: float
    bootstrap_statistics: np.ndarray
    n_bootstrap: int
    calibration: str = 'parametric_bootstrap'
    bootstrap_diagnostics: tuple[dict, ...] = ()


# ══════════════════════════════════════════════════════════════════
# CvM test (shared)
# ══════════════════════════════════════════════════════════════════

def cvm_test(e):
    """
    One-sample Cramér-von Mises test.

    Parameters
    ----------
    e : array-like, shape (T, d)
        Under H0:
            y_t = chi2.cdf(sum_j Phi^{-1}(e_tj)^2, df=d) ~ U[0,1]

    Returns
    -------
    CramerVonMisesResult
        Has .statistic and .pvalue
    """
    e = np.asarray(e, dtype=np.float64)
    if e.ndim != 2:
        raise ValueError(f"e must have shape (T, d), got {e.shape}")

    T, d = e.shape
    if T == 0:
        raise ValueError("e must contain at least one observation")

    # Avoid inf in norm.ppf at exactly 0 or 1
    e = clip_pseudo_observations(e)

    z = norm.ppf(e)                       # (T, d)
    q = np.sum(z * z, axis=1)             # (T,)
    y = chi2.cdf(q, df=d)                 # should be U[0,1] under H0

    return cramervonmises(y, "uniform")


def _as_float64_array_no_copy(value):
    """Return a float64 array while preserving an already compatible input."""
    if type(value) is np.ndarray and value.dtype == np.float64:
        return value
    return np.asarray(value, dtype=np.float64)


def _grid_transition_method(transition_method):
    if str(transition_method).lower() == 'spectral':
        return 'auto'
    return transition_method


# ══════════════════════════════════════════════════════════════════
# Bivariate Rosenblatt
# ══════════════════════════════════════════════════════════════════

def rosenblatt_transform_mle(copula, u, r):
    """Rosenblatt for constant copula parameter (MLE). Returns (T, 2)."""
    T = len(u)
    e = np.empty((T, 2))
    e[:, 0] = u[:, 0]
    e[:, 1] = copula.h(u[:, 1], u[:, 0], np.full(T, float(r)))
    return clip_rosenblatt_output(e)


def rosenblatt_transform_scar(copula, u, alpha, K=300, grid_range=5.0,
                              grid_method='auto', adaptive=True,
                              pts_per_sigma=4, transition_method='matrix',
                              max_K=None, r_gh=3.0, gh_order=5):
    """Mixture Rosenblatt for SCAR (bivariate). Returns (T, 2)."""
    from pyscarcopula.numerical import _cpp_scar_ou
    from pyscarcopula.numerical._scar_ou_config import AutoTMConfig

    kappa, mu, nu = alpha
    config = AutoTMConfig(
        transition_method=transition_method,
        K=K,
        grid_range=grid_range,
        grid_method=grid_method,
        adaptive=adaptive,
        pts_per_sigma=pts_per_sigma,
        max_K=max_K,
        r_gh=r_gh,
        gh_order=gh_order,
    )
    e = np.empty((len(u), 2), dtype=np.float64)
    e[:, 0] = u[:, 0]
    e[:, 1] = _cpp_scar_ou.mixture_h(
        kappa, mu, nu, u, copula, config)
    return e


def rosenblatt_transform_gas(copula, u, gas_params, scaling='unit'):
    """Rosenblatt for GAS (bivariate). Returns (T, 2)."""
    from pyscarcopula.numerical.gas_filter import gas_rosenblatt
    omega, gamma, beta = gas_params
    return gas_rosenblatt(omega, gamma, beta, u, copula, scaling)


# ══════════════════════════════════════════════════════════════════
# Unified tests
# ══════════════════════════════════════════════════════════════════

def gof_test(model, data, to_pobs=True, K=300, grid_range=5.0,
             fit_result=None, bootstrap=False, n_bootstrap=199,
             bootstrap_refit=True, bootstrap_fit_kwargs=None, rng=None):
    """
    Unified goodness-of-fit test for any copula model.

    Dispatches based on model type:
      - BivariateCopula  -> bivariate Rosenblatt (MLE or SCAR mixture)
      - CVineCopula      -> vine Rosenblatt (per-edge bivariate approach)
      - GaussianCopula   -> Cholesky-based Rosenblatt
      - StudentCopula    -> conditional t-distribution Rosenblatt

    Parameters
    ----------
    model : BivariateCopula, CVineCopula, GaussianCopula, or StudentCopula
    data : (T, d) array
    to_pobs : bool
    K : int — grid size (SCAR only)
    grid_range : float (SCAR only)
    fit_result : FitResult or None
        If provided, use this instead of model.fit_result.
        Enables the stateless API: gof_test(copula, u, fit_result=result)
    bootstrap : bool
        If True, calibrate the bivariate CvM statistic by parametric
        bootstrap instead of using the one-sample asymptotic p-value.
    n_bootstrap : int
        Number of bootstrap replications.
    bootstrap_refit : bool
        If True, re-estimate the model on each bootstrap sample.
    bootstrap_fit_kwargs : dict or None
        Extra keyword arguments for each bootstrap fit.
    rng : int, Generator, or None
        Random seed/source for bootstrap simulation.

    Returns
    -------
    CramérVonMisesResult with .statistic and .pvalue
    """
    from pyscarcopula.copula.base import BivariateCopula
    from pyscarcopula.copula.multivariate import GaussianCopula, StudentCopula
    from pyscarcopula.vine.cvine import CVineCopula
    from pyscarcopula.vine.rvine import RVineCopula
    from pyscarcopula.copula.multivariate import (
        EquicorrGaussianCopula,
        StochasticStudentCopula,
    )

    if isinstance(model, StochasticStudentCopula):
        if bootstrap:
            raise NotImplementedError(
                "Bootstrap GoF is currently implemented for bivariate "
                "copulas only.")
        return stochastic_student_gof_test(model, data, to_pobs, K,
                                           grid_range, fit_result=fit_result)
    elif isinstance(model, EquicorrGaussianCopula):
        if bootstrap:
            raise NotImplementedError(
                "Bootstrap GoF is currently implemented for bivariate "
                "copulas only.")
        return equicorr_gof_test(model, data, to_pobs, K, grid_range,
                                 fit_result=fit_result)
    elif isinstance(model, BivariateCopula):
        return _gof_bivariate(model, data, to_pobs, K, grid_range,
                              fit_result=fit_result, bootstrap=bootstrap,
                              n_bootstrap=n_bootstrap,
                              bootstrap_refit=bootstrap_refit,
                              bootstrap_fit_kwargs=bootstrap_fit_kwargs,
                              rng=rng)
    elif isinstance(model, CVineCopula):
        if bootstrap:
            raise NotImplementedError(
                "Bootstrap GoF is currently implemented for bivariate "
                "copulas only.")
        return vine_gof_test(model, data, to_pobs, K, grid_range)
    elif isinstance(model, RVineCopula):
        if bootstrap:
            raise NotImplementedError(
                "Bootstrap GoF is currently implemented for bivariate "
                "copulas only.")
        return rvine_gof_test(model, data, to_pobs, K, grid_range)
    elif isinstance(model, GaussianCopula):
        if bootstrap:
            raise NotImplementedError(
                "Bootstrap GoF is currently implemented for bivariate "
                "copulas only.")
        return gaussian_gof_test(model, data, to_pobs)
    elif isinstance(model, StudentCopula):
        if bootstrap:
            raise NotImplementedError(
                "Bootstrap GoF is currently implemented for bivariate "
                "copulas only.")
        return student_gof_test(model, data, to_pobs)
    else:
        raise TypeError(f"Unsupported model type: {type(model).__name__}")

# ══════════════════════════════════════════════════════════════════
# Bivariate gof_test
# ══════════════════════════════════════════════════════════════════

def _gof_bivariate(copula, data, to_pobs=True, K=300, grid_range=5.0,
                   fit_result=None, bootstrap=False, n_bootstrap=199,
                   bootstrap_refit=True, bootstrap_fit_kwargs=None,
                   rng=None):
    """
    Goodness-of-fit for a fitted BivariateCopula.

    MLE: constant parameter Rosenblatt.
    SCAR: mixture Rosenblatt (integrates h over predictive distribution).
    GAS: deterministic Rosenblatt (h evaluated at filtered r_t).

    Parameters
    ----------
    copula : BivariateCopula
    data : (T, 2) array
    to_pobs : bool
    K : int
    grid_range : float
    fit_result : FitResult or None
        If None, uses copula.fit_result (set by copula.fit()).
    """
    from pyscarcopula._utils import pobs as compute_pobs

    u = _as_float64_array_no_copy(data)
    if to_pobs:
        u = compute_pobs(u)

    fr = fit_result if fit_result is not None else getattr(copula, 'fit_result', None)
    if fr is None:
        raise ValueError("No fit_result provided and copula has no fit_result. "
                         "Call copula.fit() first or pass fit_result=.")

    e = _bivariate_rosenblatt_from_result(copula, u, fr, K, grid_range)
    result = cvm_test(e)

    if not bootstrap:
        return result

    return _bootstrap_gof_bivariate(
        copula, u, fr, float(result.statistic), K, grid_range,
        n_bootstrap=n_bootstrap, bootstrap_refit=bootstrap_refit,
        bootstrap_fit_kwargs=bootstrap_fit_kwargs, rng=rng)


def _bivariate_rosenblatt_from_result(copula, u, fit_result,
                                      K=300, grid_range=5.0):
    method = fit_result.method.upper()

    if method == 'MLE':
        r = getattr(fit_result, 'copula_param', 0.0)
        return rosenblatt_transform_mle(copula, u, r)
    if method == 'GAS':
        scaling = getattr(fit_result, 'scaling', 'unit')
        return rosenblatt_transform_gas(
            copula, u, fit_result.params.values, scaling)

    if getattr(fit_result, 'params', None) is None:
        raise ValueError(
            f"Cannot compute bivariate Rosenblatt transform for {method}")

    from pyscarcopula.strategy._base import get_strategy_for_result

    strategy = get_strategy_for_result(
        fit_result, K=K, grid_range=grid_range)
    e = np.empty((len(u), 2), dtype=np.float64)
    e[:, 0] = u[:, 0]
    e[:, 1] = strategy.rosenblatt_e2(copula, u, fit_result)
    return clip_rosenblatt_output(e)


def _as_rng(rng):
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _bootstrap_fit_kwargs(fit_result, fit_kwargs):
    """Warm-start bootstrap refits from the original fitted parameters."""
    out = dict(fit_kwargs)
    if 'alpha0' in out or 'gamma0' in out:
        return out

    method = fit_result.method.upper()
    if method == 'MLE' and hasattr(fit_result, 'copula_param'):
        out['alpha0'] = np.array([fit_result.copula_param], dtype=np.float64)
    else:
        params = getattr(fit_result, 'params', None)
        if params is not None:
            key = 'gamma0' if method == 'GAS' else 'alpha0'
            out[key] = np.asarray(params.values, dtype=np.float64)
    return out


def _fit_result_diagnostics(result):
    row = {
        'bootstrap_fit_method': getattr(result, 'method', ''),
        'bootstrap_fit_log_likelihood': float(
            getattr(result, 'log_likelihood', np.nan)),
        'bootstrap_fit_success': bool(getattr(result, 'success', False)),
        'bootstrap_fit_nfev': int(getattr(result, 'nfev', 0)),
        'bootstrap_fit_message': str(getattr(result, 'message', '')),
    }
    if hasattr(result, 'copula_param'):
        row['bootstrap_param_theta'] = float(result.copula_param)

    params = getattr(result, 'params', None)
    if params is not None:
        values = np.asarray(params.values, dtype=np.float64)
        row['bootstrap_params_json'] = {
            name: float(value)
            for name, value in zip(params.names, values)
        }
        for name, value in zip(params.names, values):
            row[f'bootstrap_param_{name}'] = float(value)
    return row


def _bootstrap_gof_bivariate(copula, u, fit_result, observed_statistic,
                             K=300, grid_range=5.0, n_bootstrap=199,
                             bootstrap_refit=True,
                             bootstrap_fit_kwargs=None, rng=None):
    """Parametric bootstrap calibration for bivariate GoF."""
    from pyscarcopula.strategy._base import get_strategy_for_result

    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")

    rng = _as_rng(rng)
    fit_kwargs = {} if bootstrap_fit_kwargs is None else dict(bootstrap_fit_kwargs)
    strategy = None
    if not (fit_result.method.upper() == 'MLE'
            and not hasattr(fit_result, 'copula_param')):
        strategy = get_strategy_for_result(
            fit_result, K=K, grid_range=grid_range)

    boot_stats = np.empty(int(n_bootstrap), dtype=np.float64)
    diagnostics = []
    for b in range(int(n_bootstrap)):
        iter_start = time.perf_counter()
        if strategy is None:
            u_boot = copula.sample(len(u), rng=rng)
        else:
            u_boot = strategy.sample(copula, u, fit_result, len(u), rng=rng)

        fit_start = time.perf_counter()
        if bootstrap_refit:
            if strategy is None:
                boot_result = fit_result
            else:
                boot_strategy = get_strategy_for_result(
                    fit_result, K=K, grid_range=grid_range)
                boot_result = boot_strategy.fit(
                    copula, u_boot,
                    **_bootstrap_fit_kwargs(fit_result, fit_kwargs))
        else:
            boot_result = fit_result
        fit_elapsed = time.perf_counter() - fit_start

        stat_start = time.perf_counter()
        e_boot = _bivariate_rosenblatt_from_result(
            copula, u_boot, boot_result, K, grid_range)
        boot_stats[b] = float(cvm_test(e_boot).statistic)
        stat_elapsed = time.perf_counter() - stat_start

        row = {
            'bootstrap_iteration': int(b + 1),
            'bootstrap_statistic': float(boot_stats[b]),
            'bootstrap_exceeds_observed': bool(
                boot_stats[b] >= float(observed_statistic)),
            'bootstrap_fit_time_sec': float(fit_elapsed),
            'bootstrap_stat_time_sec': float(stat_elapsed),
            'bootstrap_total_time_sec': float(
                time.perf_counter() - iter_start),
        }
        row.update(_fit_result_diagnostics(boot_result))
        diagnostics.append(row)

    pvalue = (
        1.0 + np.sum(boot_stats >= float(observed_statistic))
    ) / (len(boot_stats) + 1.0)
    return BootstrapGoFResult(
        statistic=float(observed_statistic),
        pvalue=float(pvalue),
        bootstrap_statistics=boot_stats,
        n_bootstrap=len(boot_stats),
        bootstrap_diagnostics=tuple(diagnostics),
    )


# ══════════════════════════════════════════════════════════════════
# Vine Rosenblatt transform
# ══════════════════════════════════════════════════════════════════

def _vine_edge_h(edge, u2, u1, u_pair, K=300, grid_range=5.0):
    """Delegate to the shared pair-edge runtime."""
    from pyscarcopula.vine._rvine_edges import _edge_h
    return _edge_h(
        edge,
        u2,
        u1,
        u_pair=u_pair,
        K=K,
        grid_range=grid_range,
    )


def vine_rosenblatt_transform(vine, u, K=300, grid_range=5.0):
    """
    Rosenblatt transform for a fitted C-vine copula.

    Each edge in the vine is an independent bivariate copula
    (possibly with its own latent OU process). The vine Rosenblatt
    simply applies h-functions level by level, reusing the bivariate
    approach on every edge — no vine-specific modifications needed.

    ```
    v[0][i] = u_i
    v[j+1][i] = h(v[j][i+1] | v[j][0]; edge_{j,i})
    e_0 = u_0
    e_{j+1} = h(v[j][1] | v[j][0]; edge_{j,0})
    ```

    Parameters
    ----------
    vine : CVineCopula (fitted)
    u : (T, d) pseudo-observations
    K : int — grid size for SCAR mixture
    grid_range : float

    Returns
    -------
    e : (T, d) — should be iid U[0,1]^d under correct model
    """
    T, d = u.shape
    v = [[None] * d for _ in range(d)]
    for i in range(d):
        v[0][i] = clip_pseudo_observations(u[:, i].copy())

    e = np.empty((T, d))
    e[:, 0] = v[0][0]

    for j in range(d - 1):
        n_edges = d - j - 1

        # e_{j+1}: first edge of tree j
        u1 = clip_pseudo_observations(v[j][0])
        u2 = clip_pseudo_observations(v[j][1])
        u_pair = np.column_stack((u1, u2))
        edge = vine.edges[j][0]
        e[:, j + 1] = clip_pseudo_observations(
            _vine_edge_h(edge, u2, u1, u_pair, K, grid_range))

        # Propagate v to next level (all edges, same approach)
        if j < d - 2:
            for i in range(n_edges):
                u1 = clip_pseudo_observations(v[j][0])
                u2 = clip_pseudo_observations(v[j][i + 1])
                u_pair = np.column_stack((u1, u2))
                edge_i = vine.edges[j][i]
                v[j + 1][i] = clip_pseudo_observations(
                    _vine_edge_h(edge_i, u2, u1, u_pair, K, grid_range))

    return clip_rosenblatt_output(e)


# ══════════════════════════════════════════════════════════════════
# Vine gof_test
# ══════════════════════════════════════════════════════════════════

def vine_gof_test(vine, data, to_pobs=True, K=500, grid_range=7.0):
    """
    Goodness-of-fit test for a fitted C-vine copula.

    Applies the d-dimensional Rosenblatt transform through the vine
    tree structure, then tests e ~ iid U[0,1]^d via CvM.

    For SCAR edges: uses mixture h-function (avoids Jensen bias).
    For MLE edges: uses constant parameter h-function.

    Parameters
    ----------
    vine : CVineCopula (fitted)
    data : (T, d)
    to_pobs : bool
    K : int — grid size for SCAR mixture Rosenblatt
    grid_range : float

    Returns
    -------
    CramérVonMisesResult with .statistic and .pvalue
    """
    from pyscarcopula._utils import pobs as compute_pobs

    u = np.asarray(data, dtype=np.float64)
    if to_pobs:
        u = compute_pobs(u)

    if vine.edges is None:
        raise ValueError("Fit the vine first")

    e = vine_rosenblatt_transform(vine, u, K=K, grid_range=grid_range)
    return cvm_test(e)


def rvine_rosenblatt_transform(vine, u, K=300, grid_range=5.0):
    """
    Rosenblatt transform for a fitted R-vine copula.

    Mirrors ``RVineCopula.sample`` for the natural-order matrix:
    columns are traversed right-to-left, and each anti-diagonal leaf is
    transformed by h-functions from tree 0 up to the column's top tree.
    """
    from pyscarcopula.vine._rvine_edges import _edge_h

    if getattr(vine, 'matrix', None) is None:
        raise ValueError("Fit the vine first")

    u = np.asarray(u, dtype=np.float64)
    T, d = u.shape
    if d != vine.d:
        raise ValueError(f"u has d={d}, but fitted vine has d={vine.d}")

    M = vine.matrix

    if d == 2:
        edge = vine.pair_copulas[(0, 0)]
        u1 = clip_pseudo_observations(u[:, 0])
        u2 = clip_pseudo_observations(u[:, 1])
        u_pair = np.column_stack((u1, u2))
        e = np.empty((T, d), dtype=np.float64)
        e[:, 0] = u1
        e[:, 1] = clip_pseudo_observations(
            _edge_h(edge, u2, u1, u_pair=u_pair, K=K,
                    grid_range=grid_range))
        return clip_rosenblatt_output(e)

    pseudo = {
        (var, frozenset()): clip_pseudo_observations(u[:, var].copy())
        for var in range(d)
    }

    e = np.empty((T, d), dtype=np.float64)

    last_var = int(M[0, d - 1])
    e[:, d - 1] = pseudo[(last_var, frozenset())]

    for col in range(d - 2, -1, -1):
        leaf = int(M[d - 1 - col, col])
        top_tree = d - 2 - col
        cur = pseudo[(leaf, frozenset())]

        for t in range(top_tree + 1):
            row = d - 2 - col - t
            partner = int(M[row, col])
            conditioning = frozenset(
                int(M[r, col])
                for r in range(row + 1, d - 1 - col)
            )
            next_leaf_cond = conditioning | {partner}
            next_partner_cond = conditioning | {leaf}

            edge = vine.pair_copulas[(t, col)]
            leaf_val = pseudo.get((leaf, conditioning))
            partner_val = pseudo.get((partner, conditioning))
            if leaf_val is None:
                raise RuntimeError(
                    "Missing leaf pseudo-observation during Rosenblatt: "
                    f"var={leaf}, cond_set={sorted(conditioning)}, "
                    f"column={col}, tree={t}"
                )
            if partner_val is None:
                raise RuntimeError(
                    "Missing partner pseudo-observation during Rosenblatt: "
                    f"var={partner}, cond_set={sorted(conditioning)}, "
                    f"column={col}, tree={t}"
                )

            cur = clip_pseudo_observations(
                _edge_h(edge, leaf_val, partner_val, K=K,
                        grid_range=grid_range))
            pseudo[(leaf, next_leaf_cond)] = cur
            pseudo[(partner, next_partner_cond)] = clip_pseudo_observations(
                _edge_h(edge, partner_val, leaf_val, K=K,
                        grid_range=grid_range))

        e[:, col] = cur

    return clip_rosenblatt_output(e)

def rvine_gof_test(vine, data, to_pobs=True,
                    K=500, grid_range=7.0):
    """
    Goodness-of-fit test for a fitted R-vine copula.

    Parameters
    ----------
    vine : RVineCopula (fitted)
    data : (T, d)
    to_pobs : bool
    K : int
    grid_range : float

    Returns
    -------
    CramérVonMisesResult
    """
    from pyscarcopula._utils import pobs as compute_pobs

    u = np.asarray(data, dtype=np.float64)
    if to_pobs:
        u = compute_pobs(u)

    if getattr(vine, 'matrix', None) is None:
        raise ValueError("Fit the vine first")

    e = rvine_rosenblatt_transform(vine, u, K=K, grid_range=grid_range)
    return cvm_test(e)


# ══════════════════════════════════════════════════════════════════
# Gaussian copula Rosenblatt
# ══════════════════════════════════════════════════════════════════

def gaussian_rosenblatt_transform(R, u):
    """
    Rosenblatt transform for d-dimensional Gaussian copula.

    x = Phi^{-1}(u), x ~ N(0, R).
    Conditional: x_i | x_{1:i-1} ~ N(mu_{i|1..i-1}, sigma^2_{i|1..i-1})
    e_i = Phi((x_i - mu_{i|1..i-1}) / sigma_{i|1..i-1})

    Uses Cholesky: R = L L^T, then z = L^{-1} x has independent
    components, and e_i = Phi(z_i).

    Parameters
    ----------
    R : (d, d) correlation matrix
    u : (T, d) pseudo-observations

    Returns
    -------
    e : (T, d)
    """
    u_c = clip_pseudo_observations_no_copy(u)
    x = norm.ppf(u_c)

    L = np.linalg.cholesky(R)
    # z = L^{-1} x, so z_i are independent N(0,1)
    # e_i = Phi(z_i)
    z = np.linalg.solve(L, x.T).T  # (T, d)
    e = norm.cdf(z)

    return clip_pseudo_observations(e)


def gaussian_gof_test(copula, data, to_pobs=True):
    """
    Goodness-of-fit test for a fitted GaussianCopula.

    Parameters
    ----------
    copula : GaussianCopula (fitted, has .corr)
    data : (T, d)
    to_pobs : bool

    Returns
    -------
    CramérVonMisesResult
    """
    from pyscarcopula._utils import pobs as compute_pobs

    u = np.asarray(data, dtype=np.float64)
    if to_pobs:
        u = compute_pobs(u)

    if copula.corr is None:
        raise ValueError("Fit the copula first")

    e = gaussian_rosenblatt_transform(copula.corr, u)
    return cvm_test(e)


# ══════════════════════════════════════════════════════════════════
# Student-t copula Rosenblatt
# ══════════════════════════════════════════════════════════════════

def student_rosenblatt_transform(R, df, u):
    """
    Rosenblatt transform for d-dimensional Student-t copula.

    x = t_df^{-1}(u), x ~ t_d(0, R, df).

    Sequential conditioning using the property that for
    multivariate t with shape R and df degrees of freedom:

        x_i | x_0,...,x_{i-1} ~ t_{df+i}(mu_i, sigma^2_i * scale)

    where:
        mu_i = R_{i,0:i} R_{0:i,0:i}^{-1} x_{0:i}
        sigma^2_i = R_{ii} - R_{i,0:i} R_{0:i,0:i}^{-1} R_{0:i,i}
        scale = (df + x_{0:i}^T R_{0:i,0:i}^{-1} x_{0:i}) / (df + i)

    Here i is the zero-based coordinate index, so the conditioning set has
    size i.

    Parameters
    ----------
    R : (d, d) shape matrix (correlation)
    df : float — degrees of freedom
    u : (T, d) pseudo-observations

    Returns
    -------
    e : (T, d)
    """
    from scipy.stats import t as t_dist

    u_c = clip_pseudo_observations(u)
    x = t_dist.ppf(u_c, df=df)

    T, d = x.shape
    e = np.empty((T, d))

    # First variable: e_0 = t_df.cdf(x_0)
    e[:, 0] = t_dist.cdf(x[:, 0], df=df)

    for i in range(1, d):
        # Conditional distribution of x_i | x_{0:i-1}
        R_11 = R[:i, :i]          # (i, i)
        R_21 = R[i, :i]           # (i,)
        R_22 = R[i, i]            # scalar

        R_11_inv = np.linalg.inv(R_11)
        beta = R_21 @ R_11_inv    # (i,) — regression coefficients

        # Conditional variance (without scale)
        sigma2_cond = R_22 - R_21 @ R_11_inv @ R_21  # scalar
        sigma_cond = np.sqrt(max(sigma2_cond, 1e-12))

        # For each observation
        x_prev = x[:, :i]                          # (T, i)
        mu_cond = x_prev @ beta                     # (T,)

        # Quadratic form: x_{1:i-1}^T R_{1:i-1}^{-1} x_{1:i-1}
        quad = np.sum(x_prev @ R_11_inv * x_prev, axis=1)  # (T,)

        # Scale factor and conditional df
        df_cond = df + i
        scale = (df + quad) / df_cond

        # Standardized residual
        z = (x[:, i] - mu_cond) / (sigma_cond * np.sqrt(scale))

        e[:, i] = t_dist.cdf(z, df=df_cond)

    return clip_pseudo_observations(e)


def student_gof_test(copula, data, to_pobs=True):
    """
    Goodness-of-fit test for a fitted StudentCopula.

    Parameters
    ----------
    copula : StudentCopula (fitted, has .shape and .df)
    data : (T, d)
    to_pobs : bool

    Returns
    -------
    CramérVonMisesResult
    """
    from pyscarcopula._utils import pobs as compute_pobs

    u = np.asarray(data, dtype=np.float64)
    if to_pobs:
        u = compute_pobs(u)

    if copula.shape is None:
        raise ValueError("Fit the copula first")

    e = student_rosenblatt_transform(copula.shape, copula.df, u)
    return cvm_test(e)


# ══════════════════════════════════════════════════════════════════
# Equicorrelation Gaussian copula GoF
# ══════════════════════════════════════════════════════════════════

def _gas_parameter_path(copula, u, fit_result):
    """Return deterministic GAS parameter path r_t for a fitted model."""
    from pyscarcopula.numerical.gas_filter import gas_filter

    omega, gamma, beta = fit_result.params.values
    scaling = getattr(fit_result, 'scaling', 'unit')
    score_eps = getattr(fit_result, 'score_eps', 1e-4)
    _, r_path, _ = gas_filter(
        omega, gamma, beta, u, copula, scaling=scaling,
        score_eps=score_eps)
    return np.asarray(r_path, dtype=np.float64)


def _tm_grid_kwargs_from_result(fit_result):
    """SCAR-TM numerical options stored on a fitted result."""
    out = {}
    for name in (
            'pts_per_sigma', 'transition_method', 'max_K',
            'r_gh', 'gh_order'):
        value = getattr(fit_result, name, None)
        if value is not None:
            out[name] = value
    if 'transition_method' in out:
        out['transition_method'] = _grid_transition_method(
            out['transition_method'])
    return out


def equicorr_rosenblatt_transform(copula, u, fit_result, K=300, grid_range=5.0):
    """
    Rosenblatt transform for EquicorrGaussianCopula.

    MLE: constant rho, Cholesky-based sequential conditioning.
    SCAR: mixture over predictive rho(t) distribution from TM forward pass.

    For equicorrelation R = (1-rho)*I + rho*11':
        E[x_i | x_0,...,x_{i-1}] = rho * sum(x_0,...,x_{i-1}) / (1 + (i-1)*rho)
        Var(x_i | x_0,...,x_{i-1}) = 1 - i*rho^2 / (1 + (i-1)*rho)

    Parameters
    ----------
    copula : EquicorrGaussianCopula
    u : (T, d)
    fit_result : FitResult
        Estimation result. Required.
    K, grid_range : TM grid params (SCAR only)

    Returns
    -------
    e : (T, d) — should be iid U[0,1]^d under correct model
    """
    u_c = clip_pseudo_observations(u)
    x_norm = norm.ppf(u_c)
    T, d = u.shape

    method = fit_result.method.upper()

    if method == 'MLE':
        rho = fit_result.copula_param
        e = np.empty((T, d))
        e[:, 0] = u[:, 0]
        for i in range(1, d):
            sx = np.sum(x_norm[:, :i], axis=1)
            cond_mean = rho * sx / (1.0 + (i - 1) * rho)
            cond_var = 1.0 - i * rho ** 2 / (1.0 + (i - 1) * rho)
            cond_var = max(cond_var, 1e-10)
            z_i = (x_norm[:, i] - cond_mean) / np.sqrt(cond_var)
            e[:, i] = norm.cdf(z_i)
        return clip_pseudo_observations(e)

    if method == 'GAS':
        rho_path = _gas_parameter_path(copula, u, fit_result)
        e = np.empty((T, d))
        e[:, 0] = u[:, 0]
        for i in range(1, d):
            rho = rho_path
            sx = np.sum(x_norm[:, :i], axis=1)
            cond_mean = rho * sx / (1.0 + (i - 1) * rho)
            cond_var = 1.0 - i * rho ** 2 / (1.0 + (i - 1) * rho)
            cond_var = np.maximum(cond_var, 1e-10)
            z_i = (x_norm[:, i] - cond_mean) / np.sqrt(cond_var)
            e[:, i] = norm.cdf(z_i)
        return clip_pseudo_observations(e)

    # SCAR: mixture Rosenblatt via TM forward pass
    from pyscarcopula.numerical.tm_grid import TMGrid as _TMGrid
    from pyscarcopula.numerical.gof_blocks import iter_forward_weight_blocks

    kappa, mu, nu = fit_result.params.values
    grid = _TMGrid(
        kappa, mu, nu, T, K, grid_range,
        **_tm_grid_kwargs_from_result(fit_result))
    x_grid = grid.z + grid.mu
    rho_grid = copula.transform(x_grid)

    e = np.empty((T, d))
    e[:, 0] = u[:, 0]

    def leading_equicorr_density(x_prefix, prefix_dim):
        if prefix_dim <= 1:
            return np.ones((len(x_prefix), grid.K), dtype=np.float64)
        rho = rho_grid[np.newaxis, :]
        a = 1.0 - rho
        b = 1.0 + (prefix_dim - 1) * rho
        s2 = np.sum(x_prefix * x_prefix, axis=1)[:, np.newaxis]
        s1 = np.sum(x_prefix, axis=1)[:, np.newaxis] ** 2
        log_det = (prefix_dim - 1) * np.log(a) + np.log(b)
        log_density = (
            -0.5 * log_det
            -0.5 * ((rho / a) * s2 - (rho / (a * b)) * s1)
        )
        return np.exp(log_density)

    cdf_blocks = None
    prefix_density_blocks = None
    for k, local, weights, fi_block in iter_forward_weight_blocks(
            grid, u, copula, x_grid=x_grid, element_width=max(1, d)):
        if local == 0:
            start = k
            stop = start + fi_block.shape[0]
            x_block = x_norm[start:stop]
            cdf_blocks = []
            prefix_density_blocks = []
            for i in range(1, d):
                sx = np.sum(x_block[:, :i], axis=1)[:, np.newaxis]
                denom = 1.0 + (i - 1) * rho_grid[np.newaxis, :]
                cond_mean = rho_grid[np.newaxis, :] * sx / denom
                cond_var = (
                    1.0
                    - i * rho_grid[np.newaxis, :] ** 2 / denom
                )
                cond_var = np.maximum(cond_var, 1e-10)
                z_i = (
                    x_block[:, i, np.newaxis] - cond_mean
                ) / np.sqrt(cond_var)
                cdf_blocks.append(norm.cdf(z_i))
                prefix_density_blocks.append(
                    leading_equicorr_density(x_block[:, :i], i))

        for i in range(1, d):
            prefix_density = prefix_density_blocks[i - 1][local]
            reweighted = weights * prefix_density
            total = np.sum(reweighted)
            if total > 0.0 and np.isfinite(total):
                reweighted = reweighted / total
            else:
                reweighted = weights
            e[k, i] = np.sum(reweighted * cdf_blocks[i - 1][local])

    e = clip_pseudo_observations(e)
    return e


def equicorr_gof_test(copula, data, to_pobs=True,
                      K=300, grid_range=5.0, fit_result=None):
    """
    Goodness-of-fit test for EquicorrGaussianCopula.

    Parameters
    ----------
    copula : EquicorrGaussianCopula
    data : (T, d)
    to_pobs : bool
    K : int
    grid_range : float
    fit_result : FitResult or None
        If None, uses copula.fit_result (set by copula.fit()).

    Returns
    -------
    CramérVonMisesResult
    """
    from pyscarcopula._utils import pobs as compute_pobs

    u = np.asarray(data, dtype=np.float64)
    if to_pobs:
        u = compute_pobs(u)

    fr = fit_result if fit_result is not None else getattr(copula, 'fit_result', None)
    if fr is None:
        raise ValueError("No fit_result provided and copula has no fit_result. "
                         "Call copula.fit() first or pass fit_result=.")

    e = equicorr_rosenblatt_transform(copula, u, fr, K, grid_range)
    return cvm_test(e)


# ══════════════════════════════════════════════════════════════════
# Stochastic Student-t copula GoF
# ══════════════════════════════════════════════════════════════════

def stochastic_student_rosenblatt_transform(copula, u, fit_result,
                                             K=300, grid_range=5.0):
    """
    Rosenblatt transform for StochasticStudentCopula.

    MLE: constant df, standard sequential conditioning.
    SCAR: mixture over predictive df(t) distribution from TM forward pass.

    For Student-t copula with shape matrix R and df:
        x = t_df^{-1}(u)
        x_i | x_0,...,x_{i-1} ~ t_{df+i}(mu_i, sigma^2_i * scale)
    where:
        mu_i = R_{i,0:i} R_{0:i,0:i}^{-1} x_{0:i}
        sigma^2_i = R_{ii} - R_{i,0:i} R_{0:i,0:i}^{-1} R_{0:i,i}
        scale = (df + x_{0:i}^T R_{0:i,0:i}^{-1} x_{0:i}) / (df + i)

    Parameters
    ----------
    copula : StochasticStudentCopula
    u : (T, d) pseudo-observations
    fit_result : FitResult
    K, grid_range : TM grid params (SCAR only)

    Returns
    -------
    e : (T, d) — should be iid U[0,1]^d under correct model
    """
    T, d = u.shape
    R = copula.R
    method = fit_result.method.upper()

    if method == 'MLE':
        df = fit_result.copula_param
        e = student_rosenblatt_transform(R, df, u)
        return clip_pseudo_observations(e)

    if method == 'GAS':
        df_path = _gas_parameter_path(copula, u, fit_result)
        e = np.empty((T, d))
        for t_idx, df_t in enumerate(df_path):
            e[t_idx] = student_rosenblatt_transform(
                R, float(df_t), u[t_idx:t_idx + 1])[0]
        return clip_pseudo_observations(e)

    # SCAR: mixture Rosenblatt via TM forward pass
    from pyscarcopula.numerical.tm_grid import TMGrid as _TMGrid
    from pyscarcopula.numerical.gof_blocks import iter_forward_weight_block_arrays
    from pyscarcopula.numerical.student_gof import (
        student_conditional_z_block,
        student_weighted_cdf_block,
    )

    kappa, mu, nu_ou = fit_result.params.values
    grid = _TMGrid(
        kappa, mu, nu_ou, T, K, grid_range,
        **_tm_grid_kwargs_from_result(fit_result))
    x_grid = grid.z + grid.mu
    df_grid = copula.transform(x_grid)  # (K_eff,)

    # Precompute padded fixed-R conditional terms for the Numba block kernel.
    beta_padded = np.zeros((d - 1, d), dtype=np.float64)
    sigma_cond = np.empty(d - 1, dtype=np.float64)
    r_inv_padded = np.zeros((d - 1, d, d), dtype=np.float64)
    log_det_prefix = np.zeros(d - 1, dtype=np.float64)
    for i in range(1, d):
        R_11 = R[:i, :i]
        R_21 = R[i, :i]
        R_22 = R[i, i]
        R_11_inv = np.linalg.inv(R_11)
        beta_padded[i - 1, :i] = R_21 @ R_11_inv
        sigma2 = R_22 - R_21 @ R_11_inv @ R_21
        sigma_cond[i - 1] = np.sqrt(max(sigma2, 1e-12))
        r_inv_padded[i - 1, :i, :i] = R_11_inv
        if i > 1:
            sign, log_det = np.linalg.slogdet(R_11)
            if sign <= 0:
                raise ValueError("leading Student correlation block is not SPD")
            log_det_prefix[i - 1] = log_det

    u_c = clip_pseudo_observations_no_copy(u)
    emission_cache = copula.prepare_emission_cache(u_c)

    e = np.empty((T, d))
    e[:, 0] = u[:, 0]

    def emission_block(u_block, x_grid, start, stop):
        return copula.copula_grid_batch(
            u_block, x_grid, t_index=start, cache=emission_cache)

    for start, stop, weights_block, _fi_block, _u_block in (
            iter_forward_weight_block_arrays(
                grid,
                u_c,
                copula,
                x_grid=x_grid,
                emission_block=emission_block,
                element_width=max(2, 2 * d),
            )):
        n_block = stop - start
        x_all = np.empty((n_block, grid.K, d), dtype=np.float64)
        for j, df_j in enumerate(df_grid):
            x_all[:, j, :] = emission_cache.ppf_rows(df_j, start, stop)

        z_blocks = student_conditional_z_block(
            x_all, df_grid, beta_padded, r_inv_padded, sigma_cond)
        cdf_blocks = np.empty_like(z_blocks)
        for cond_idx in range(d - 1):
            dim = cond_idx + 1
            cdf_blocks[cond_idx] = stdtr(
                df_grid[np.newaxis, :] + dim,
                z_blocks[cond_idx],
            )
        e[start:stop, 1:] = student_weighted_cdf_block(
            cdf_blocks, weights_block, x_all, df_grid, r_inv_padded,
            log_det_prefix)

    e = clip_pseudo_observations(e)
    return e


def stochastic_student_gof_test(copula, data, to_pobs=True,
                                 K=300, grid_range=5.0, fit_result=None):
    """
    Goodness-of-fit test for StochasticStudentCopula.

    Parameters
    ----------
    copula : StochasticStudentCopula
    data : (T, d)
    to_pobs : bool
    K : int
    grid_range : float
    fit_result : FitResult or None

    Returns
    -------
    CramérVonMisesResult
    """
    from pyscarcopula._utils import pobs as compute_pobs

    u = _as_float64_array_no_copy(data)
    if to_pobs:
        u = compute_pobs(u)

    fr = fit_result if fit_result is not None else getattr(copula, 'fit_result', None)
    if fr is None:
        raise ValueError("No fit_result provided and copula has no fit_result. "
                         "Call copula.fit() first or pass fit_result=.")

    e = stochastic_student_rosenblatt_transform(copula, u, fr, K, grid_range)
    return cvm_test(e)


# ══════════════════════════════════════════════════════════════════
