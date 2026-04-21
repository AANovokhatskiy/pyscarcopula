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
from scipy.stats import chi2, norm, cramervonmises


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
    eps = 1e-10
    e = np.clip(e, eps, 1.0 - eps)

    z = norm.ppf(e)                       # (T, d)
    q = np.sum(z * z, axis=1)             # (T,)
    y = chi2.cdf(q, df=d)                 # should be U[0,1] under H0

    return cramervonmises(y, "uniform")


def _clip(x):
    eps = 1e-6
    return np.clip(x, eps, 1.0 - eps)


# ══════════════════════════════════════════════════════════════════
# Bivariate Rosenblatt
# ══════════════════════════════════════════════════════════════════

def rosenblatt_transform_mle(copula, u, r):
    """Rosenblatt for constant copula parameter (MLE). Returns (T, 2)."""
    T = len(u)
    e = np.empty((T, 2))
    e[:, 0] = u[:, 0]
    e[:, 1] = copula.h(u[:, 1], u[:, 0], np.full(T, float(r)))
    return _clip(e)


def rosenblatt_transform_scar(copula, u, alpha, K=300, grid_range=5.0):
    """Mixture Rosenblatt for SCAR (bivariate). Returns (T, 2)."""
    from pyscarcopula.numerical.tm_functions import tm_forward_rosenblatt as _tm_forward_rosenblatt
    theta, mu, nu = alpha
    return _tm_forward_rosenblatt(theta, mu, nu, u, copula, K, grid_range)


def rosenblatt_transform_gas(copula, u, gas_params, scaling='unit'):
    """Rosenblatt for GAS (bivariate). Returns (T, 2)."""
    from pyscarcopula.numerical.gas_filter import gas_rosenblatt
    omega, alpha_g, beta = gas_params
    return gas_rosenblatt(omega, alpha_g, beta, u, copula, scaling)


# ══════════════════════════════════════════════════════════════════
# Unified tests
# ══════════════════════════════════════════════════════════════════

def gof_test(model, data, to_pobs=True, K=300, grid_range=5.0,
             fit_result=None):
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

    Returns
    -------
    CramérVonMisesResult with .statistic and .pvalue
    """
    from pyscarcopula.copula.base import BivariateCopula
    from pyscarcopula.copula.elliptical import GaussianCopula, StudentCopula
    from pyscarcopula.vine.cvine import CVineCopula
    from pyscarcopula.vine.rvine import RVineCopula
    from pyscarcopula.copula.experimental.equicorr import EquicorrGaussianCopula
    from pyscarcopula.copula.experimental.stochastic_student import StochasticStudentCopula
    from pyscarcopula.copula.experimental.stochastic_student_dcc import StochasticStudentDCCCopula

    if isinstance(model, StochasticStudentDCCCopula):
        return stochastic_student_dcc_gof_test(model, data, to_pobs, K,
                                                grid_range, fit_result=fit_result)
    elif isinstance(model, StochasticStudentCopula):
        return stochastic_student_gof_test(model, data, to_pobs, K,
                                           grid_range, fit_result=fit_result)
    elif isinstance(model, EquicorrGaussianCopula):
        return equicorr_gof_test(model, data, to_pobs, K, grid_range,
                                 fit_result=fit_result)
    elif isinstance(model, BivariateCopula):
        return _gof_bivariate(model, data, to_pobs, K, grid_range,
                              fit_result=fit_result)
    elif isinstance(model, CVineCopula):
        return vine_gof_test(model, data, to_pobs, K, grid_range)
    elif isinstance(model, RVineCopula):
        return rvine_gof_test(model, data, to_pobs, K, grid_range)
    elif isinstance(model, GaussianCopula):
        return gaussian_gof_test(model, data, to_pobs)
    elif isinstance(model, StudentCopula):
        return student_gof_test(model, data, to_pobs)
    else:
        raise TypeError(f"Unsupported model type: {type(model).__name__}")

# ══════════════════════════════════════════════════════════════════
# Bivariate gof_test
# ══════════════════════════════════════════════════════════════════

def _gof_bivariate(copula, data, to_pobs=True, K=300, grid_range=5.0,
                   fit_result=None):
    """
    Goodness-of-fit for a fitted BivariateCopula.

    MLE: constant parameter Rosenblatt.
    SCAR: mixture Rosenblatt (integrates h over predictive distribution).
    GAS: deterministic Rosenblatt (h evaluated at filtered theta_t).

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

    u = np.asarray(data, dtype=np.float64)
    if to_pobs:
        u = compute_pobs(u)

    fr = fit_result if fit_result is not None else getattr(copula, 'fit_result', None)
    if fr is None:
        raise ValueError("No fit_result provided and copula has no fit_result. "
                         "Call copula.fit() first or pass fit_result=.")

    method = fr.method.upper()

    if method == 'MLE':
        r = fr.copula_param
        e = rosenblatt_transform_mle(copula, u, r)
    elif method == 'GAS':
        scaling = getattr(fr, 'scaling', 'unit')
        gp = fr.params.values
        e = rosenblatt_transform_gas(copula, u, gp, scaling)
    else:
        alpha = fr.params.values
        e = rosenblatt_transform_scar(copula, u, alpha, K, grid_range)

    return cvm_test(e)


# ══════════════════════════════════════════════════════════════════
# Vine Rosenblatt transform
# ══════════════════════════════════════════════════════════════════

def _vine_edge_h(edge, u2, u1, u_pair, K=300, grid_range=5.0):
    """Delegate to vine._edge._edge_h (single source of truth)."""
    from pyscarcopula.vine._edge import _edge_h
    return _edge_h(edge, u2, u1, u_pair, K, grid_range)


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
    eps = 1e-10

    v = [[None] * d for _ in range(d)]
    for i in range(d):
        v[0][i] = np.clip(u[:, i].copy(), eps, 1.0 - eps)

    e = np.empty((T, d))
    e[:, 0] = v[0][0]

    for j in range(d - 1):
        n_edges = d - j - 1

        # e_{j+1}: first edge of tree j
        u1 = np.clip(v[j][0], eps, 1.0 - eps)
        u2 = np.clip(v[j][1], eps, 1.0 - eps)
        u_pair = np.column_stack((u1, u2))
        edge = vine.edges[j][0]
        e[:, j + 1] = np.clip(
            _vine_edge_h(edge, u2, u1, u_pair, K, grid_range),
            eps, 1.0 - eps)

        # Propagate v to next level (all edges, same approach)
        if j < d - 2:
            for i in range(n_edges):
                u1 = np.clip(v[j][0], eps, 1.0 - eps)
                u2 = np.clip(v[j][i + 1], eps, 1.0 - eps)
                u_pair = np.column_stack((u1, u2))
                edge_i = vine.edges[j][i]
                v[j + 1][i] = np.clip(
                    _vine_edge_h(edge_i, u2, u1, u_pair, K, grid_range),
                    eps, 1.0 - eps)

    return _clip(e)


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

    eps = 1e-10
    M = vine.matrix

    pseudo = {
        (var, frozenset()): np.clip(u[:, var].copy(), eps, 1.0 - eps)
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

            cur = np.clip(
                _edge_h(edge, leaf_val, partner_val),
                eps,
                1.0 - eps,
            )
            pseudo[(leaf, next_leaf_cond)] = cur
            pseudo[(partner, next_partner_cond)] = np.clip(
                _edge_h(edge, partner_val, leaf_val),
                eps,
                1.0 - eps,
            )

        e[:, col] = cur

    return _clip(e)

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
    eps = 1e-10
    u_c = np.clip(u, eps, 1.0 - eps)
    x = norm.ppf(u_c)

    L = np.linalg.cholesky(R)
    # z = L^{-1} x, so z_i are independent N(0,1)
    # e_i = Phi(z_i)
    z = np.linalg.solve(L, x.T).T  # (T, d)
    e = norm.cdf(z)

    return np.clip(e, eps, 1.0 - eps)


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

        x_i | x_{1:i-1} ~ t_{df+i-1}(mu_i, sigma^2_i * scale)

    where:
        mu_i = R_{i,1:i-1} R_{1:i-1,1:i-1}^{-1} x_{1:i-1}
        sigma^2_i = R_{ii} - R_{i,1:i-1} R_{1:i-1,1:i-1}^{-1} R_{1:i-1,i}
        scale = (df + x_{1:i-1}^T R_{1:i-1,1:i-1}^{-1} x_{1:i-1}) / (df + i - 1)

    e_i = t_{df+i-1}(x_i; mu_i, sigma^2_i * scale)

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

    eps = 1e-10
    u_c = np.clip(u, eps, 1.0 - eps)
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

    return np.clip(e, eps, 1.0 - eps)


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

def equicorr_rosenblatt_transform(copula, u, fit_result, K=300, grid_range=5.0):
    """
    Rosenblatt transform for EquicorrGaussianCopula.

    MLE: constant rho, Cholesky-based sequential conditioning.
    SCAR: mixture over predictive rho(t) distribution from TM forward pass.

    For equicorrelation R = (1-rho)*I + rho*11':
        E[x_i | x_{1:i-1}] = rho * sum(x_{1:i-1}) / (1 + (i-2)*rho)
        Var(x_i | x_{1:i-1}) = 1 - i*rho^2 / (1 + (i-1)*rho)

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
    eps = 1e-10
    u_c = np.clip(u, eps, 1.0 - eps)
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
        return np.clip(e, eps, 1.0 - eps)

    # SCAR: mixture Rosenblatt via TM forward pass
    from pyscarcopula.numerical.tm_grid import TMGrid as _TMGrid

    theta, mu, nu = fit_result.params.values
    grid = _TMGrid(theta, mu, nu, T, K, grid_range)
    x_grid = grid.z + grid.mu
    rho_grid = copula.transform(x_grid)
    fi_grid = copula.copula_grid_batch(u, x_grid)
    K_eff = grid.K

    weights = grid.forward_weights(fi_grid)

    e = np.empty((T, d))
    e[:, 0] = u[:, 0]

    # Vectorized over T: loop over grid points and dimensions only
    for j in range(K_eff):
        rho = rho_grid[j]
        for i in range(1, d):
            sx = np.sum(x_norm[:, :i], axis=1)              # (T,)
            cond_mean = rho * sx / (1.0 + (i - 1) * rho)
            cond_var = max(1.0 - i * rho ** 2 / (1.0 + (i - 1) * rho), 1e-10)
            z_i = (x_norm[:, i] - cond_mean) / np.sqrt(cond_var)
            cdf_val = norm.cdf(z_i)                          # (T,)

            if j == 0:
                e[:, i] = weights[:, j] * cdf_val
            else:
                e[:, i] += weights[:, j] * cdf_val

    e = np.clip(e, eps, 1.0 - eps)
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
        x_i | x_{1:i-1} ~ t_{df+i-1}(mu_i, sigma^2_i * scale)
    where:
        mu_i = R_{i,1:i-1} R_{1:i-1,1:i-1}^{-1} x_{1:i-1}
        sigma^2_i = R_{ii} - R_{i,1:i-1} R_{1:i-1,1:i-1}^{-1} R_{1:i-1,i}
        scale = (df + x_{1:i-1}^T R_{1:i-1,1:i-1}^{-1} x_{1:i-1}) / (df + i - 1)

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
    from scipy.stats import t as t_dist_fn

    eps = 1e-10
    T, d = u.shape
    R = copula.R
    method = fit_result.method.upper()

    if method == 'MLE':
        df = fit_result.copula_param
        e = student_rosenblatt_transform(R, df, u)
        return np.clip(e, eps, 1.0 - eps)

    # SCAR: mixture Rosenblatt via TM forward pass
    from pyscarcopula.numerical.tm_grid import TMGrid as _TMGrid

    theta, mu, nu_ou = fit_result.params.values
    grid = _TMGrid(theta, mu, nu_ou, T, K, grid_range)
    x_grid = grid.z + grid.mu
    df_grid = copula.transform(x_grid)  # (K_eff,)
    fi_grid = copula.copula_grid_batch(u, x_grid)
    K_eff = grid.K

    weights = grid.forward_weights(fi_grid)  # (T, K_eff)

    # Precompute R sub-matrices
    R_inv_sub = []
    beta_sub = []
    sigma_cond_sub = []
    for i in range(1, d):
        R_11 = R[:i, :i]
        R_21 = R[i, :i]
        R_22 = R[i, i]
        R_11_inv = np.linalg.inv(R_11)
        beta_sub.append(R_21 @ R_11_inv)
        sigma2 = R_22 - R_21 @ R_11_inv @ R_21
        sigma_cond_sub.append(np.sqrt(max(sigma2, 1e-12)))
        R_inv_sub.append(R_11_inv)

    u_c = np.clip(u, eps, 1.0 - eps)

    e = np.empty((T, d))
    e[:, 0] = u[:, 0]

    # Vectorized over T: loop over grid points and dimensions only
    for j in range(K_eff):
        df_j = df_grid[j]
        x_all = t_dist_fn.ppf(u_c, df=df_j)  # (T, d)

        for i in range(1, d):
            x_prev = x_all[:, :i]                          # (T, i)
            mu_cond = x_prev @ beta_sub[i - 1]             # (T,)
            quad = np.sum(x_prev @ R_inv_sub[i - 1] * x_prev, axis=1)

            df_cond = df_j + i
            scale = (df_j + quad) / df_cond
            z_i = (x_all[:, i] - mu_cond) / (
                sigma_cond_sub[i - 1] * np.sqrt(np.maximum(scale, 1e-12)))
            cdf_val = t_dist_fn.cdf(z_i, df=df_cond)       # (T,)

            if j == 0:
                e[:, i] = weights[:, j] * cdf_val
            else:
                e[:, i] += weights[:, j] * cdf_val

    e = np.clip(e, eps, 1.0 - eps)
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

    u = np.asarray(data, dtype=np.float64)
    if to_pobs:
        u = compute_pobs(u)

    fr = fit_result if fit_result is not None else getattr(copula, 'fit_result', None)
    if fr is None:
        raise ValueError("No fit_result provided and copula has no fit_result. "
                         "Call copula.fit() first or pass fit_result=.")

    e = stochastic_student_rosenblatt_transform(copula, u, fr, K, grid_range)
    return cvm_test(e)


# ══════════════════════════════════════════════════════════════════
# Stochastic Student-t DCC copula GoF
# ══════════════════════════════════════════════════════════════════

def stochastic_student_dcc_rosenblatt_transform(copula, u, fit_result,
                                                 K=300, grid_range=5.0):
    """
    Rosenblatt transform for StochasticStudentDCCCopula.

    Like the fixed-R stochastic Student-t, but uses time-varying R_t
    from the DCC(1,1) path at each observation.

    MLE: constant df, time-varying R_t.
    SCAR: mixture over predictive df(t) from TM forward pass, time-varying R_t.

    Parameters
    ----------
    copula : StochasticStudentDCCCopula
    u : (T, d) pseudo-observations
    fit_result : FitResult
    K, grid_range : TM grid params (SCAR only)

    Returns
    -------
    e : (T, d) — should be iid U[0,1]^d under correct model
    """
    from scipy.stats import t as t_dist_fn

    eps = 1e-10
    T, d = u.shape
    R_path = copula.R_path
    if R_path is None:
        raise ValueError("DCC R_t path not available. Call fit_R_t() first.")
    if len(R_path) != T:
        raise ValueError(f"u has length {T}, but R_path has length {len(R_path)}")

    method = fit_result.method.upper()

    # Precompute R sub-matrices for each time step
    def _sequential_rosenblatt_one(x_row, R, df):
        """Rosenblatt for one observation with given R and df."""
        e_row = np.empty(d)
        e_row[0] = t_dist_fn.cdf(x_row[0], df=df)
        for i in range(1, d):
            R_11 = R[:i, :i]
            R_21 = R[i, :i]
            R_22 = R[i, i]
            R_11_inv = np.linalg.inv(R_11)
            beta = R_21 @ R_11_inv
            sigma2 = R_22 - R_21 @ R_11_inv @ R_21
            sigma_c = np.sqrt(max(sigma2, 1e-12))

            mu_c = beta @ x_row[:i]
            quad = x_row[:i] @ R_11_inv @ x_row[:i]
            df_cond = df + i
            scale = (df + quad) / df_cond
            z_i = (x_row[i] - mu_c) / (sigma_c * np.sqrt(max(scale, 1e-12)))
            e_row[i] = t_dist_fn.cdf(z_i, df=df_cond)
        return e_row

    if method == 'MLE':
        df = fit_result.copula_param
        e = np.empty((T, d))
        for t_idx in range(T):
            u_c = np.clip(u[t_idx], eps, 1.0 - eps)
            x = t_dist_fn.ppf(u_c, df=df)
            e[t_idx] = _sequential_rosenblatt_one(x, R_path[t_idx], df)
        return np.clip(e, eps, 1.0 - eps)

    # SCAR: mixture Rosenblatt via TM forward pass
    from pyscarcopula.numerical.tm_grid import TMGrid as _TMGrid

    theta, mu, nu_ou = fit_result.params.values
    grid = _TMGrid(theta, mu, nu_ou, T, K, grid_range)
    x_grid = grid.z + grid.mu
    df_grid = copula.transform(x_grid)
    fi_grid = copula.copula_grid_batch(u, x_grid)
    K_eff = grid.K

    weights = grid.forward_weights(fi_grid)

    # Precompute R sub-matrices for each time step
    beta_path = []    # beta_path[i-1] shape (T, i)
    sigma_path = []   # sigma_path[i-1] shape (T,)
    R_inv_path = []   # R_inv_path[i-1] shape (T, i, i)
    for i in range(1, d):
        betas = np.empty((T, i))
        sigmas = np.empty(T)
        R_invs = np.empty((T, i, i))
        for t_idx in range(T):
            R_t = R_path[t_idx]
            R_11 = R_t[:i, :i]
            R_21 = R_t[i, :i]
            R_22 = R_t[i, i]
            R_11_inv = np.linalg.inv(R_11)
            betas[t_idx] = R_21 @ R_11_inv
            sigma2 = R_22 - R_21 @ R_11_inv @ R_21
            sigmas[t_idx] = np.sqrt(max(sigma2, 1e-12))
            R_invs[t_idx] = R_11_inv
        beta_path.append(betas)
        sigma_path.append(sigmas)
        R_inv_path.append(R_invs)

    u_c = np.clip(u, eps, 1.0 - eps)

    e = np.empty((T, d))
    e[:, 0] = u[:, 0]

    # Vectorized over T: loop over grid points and dimensions only
    for j in range(K_eff):
        df_j = df_grid[j]
        x_all = t_dist_fn.ppf(u_c, df=df_j)  # (T, d)

        for i in range(1, d):
            x_prev = x_all[:, :i]                                    # (T, i)
            mu_c = np.sum(beta_path[i - 1] * x_prev, axis=1)        # (T,)
            quad = np.sum(
                np.einsum('ti,tij->tj', x_prev, R_inv_path[i - 1]) * x_prev,
                axis=1)                                               # (T,)

            df_cond = df_j + i
            scale = (df_j + quad) / df_cond
            z_i = (x_all[:, i] - mu_c) / (
                sigma_path[i - 1] * np.sqrt(np.maximum(scale, 1e-12)))
            cdf_val = t_dist_fn.cdf(z_i, df=df_cond)                 # (T,)

            if j == 0:
                e[:, i] = weights[:, j] * cdf_val
            else:
                e[:, i] += weights[:, j] * cdf_val

    e = np.clip(e, eps, 1.0 - eps)
    return e


def stochastic_student_dcc_gof_test(copula, data, to_pobs=True,
                                     K=300, grid_range=5.0, fit_result=None):
    """
    Goodness-of-fit test for StochasticStudentDCCCopula.

    Parameters
    ----------
    copula : StochasticStudentDCCCopula
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

    u = np.asarray(data, dtype=np.float64)
    if to_pobs:
        u = compute_pobs(u)

    fr = fit_result if fit_result is not None else getattr(copula, 'fit_result', None)
    if fr is None:
        raise ValueError("No fit_result provided and copula has no fit_result. "
                         "Call copula.fit() first or pass fit_result=.")

    e = stochastic_student_dcc_rosenblatt_transform(copula, u, fr, K, grid_range)
    return cvm_test(e)
