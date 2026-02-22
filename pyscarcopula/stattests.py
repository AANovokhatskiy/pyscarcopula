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
from scipy.stats import chi2, norm, cramervonmises_2samp


# ══════════════════════════════════════════════════════════════════
# CvM test (shared)
# ══════════════════════════════════════════════════════════════════

def cvm_test(e, seed=None):
    """
    Cramér-von Mises test.
    e: (T, d). Under H0: y = chi2.cdf(sum(Phi^{-1}(e_j)^2), df=d) ~ U[0,1].
    """
    T, d = e.shape

    y = np.empty(T)
    for k in range(T):
        val = 0.0
        for j in range(d):
            val += norm.ppf(e[k, j]) ** 2
        y[k] = chi2.cdf(val, df=d)

    size = 1_000_000
    if seed is None:
        ref = np.random.uniform(0, 1, size=size)
    else:
        rng = np.random.default_rng(seed)
        ref = rng.random(size=size)

    return cramervonmises_2samp(ref, y, method='auto')


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
    from pyscarcopula.latent.ou_process import _tm_forward_rosenblatt
    theta, mu, nu = alpha
    return _tm_forward_rosenblatt(theta, mu, nu, u, copula, K, grid_range)


# ══════════════════════════════════════════════════════════════════
# Unified tests
# ══════════════════════════════════════════════════════════════════

def gof_test(model, data, to_pobs=True, seed=None, K=300, grid_range=5.0):
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
    seed : int or None
    K : int — grid size (SCAR only)
    grid_range : float (SCAR only)

    Returns
    -------
    CramérVonMisesResult with .statistic and .pvalue
    """
    from pyscarcopula.copula.base import BivariateCopula
    from pyscarcopula.copula.elliptical import GaussianCopula, StudentCopula
    from pyscarcopula.copula.vine import CVineCopula

    if isinstance(model, BivariateCopula):
        return _gof_bivariate(model, data, to_pobs, seed, K, grid_range)
    elif isinstance(model, CVineCopula):
        return vine_gof_test(model, data, to_pobs, seed, K, grid_range)
    elif isinstance(model, GaussianCopula):
        return gaussian_gof_test(model, data, to_pobs, seed)
    elif isinstance(model, StudentCopula):
        return student_gof_test(model, data, to_pobs, seed)
    else:
        raise TypeError(f"Unsupported model type: {type(model).__name__}")

# ══════════════════════════════════════════════════════════════════
# Bivariate gof_test
# ══════════════════════════════════════════════════════════════════

def _gof_bivariate(copula, data, to_pobs=True, seed=None, K=300, grid_range=5.0):
    """
    Goodness-of-fit for a fitted BivariateCopula.

    MLE: constant parameter Rosenblatt.
    SCAR: mixture Rosenblatt (integrates h over predictive distribution).
    """
    from pyscarcopula.utils import pobs as compute_pobs

    u = np.asarray(data, dtype=np.float64)
    if to_pobs:
        u = compute_pobs(u)

    if copula.fit_result is None:
        raise ValueError("Fit the copula first")

    method = copula.fit_result.method.upper()

    if method == 'MLE':
        e = rosenblatt_transform_mle(copula, u, copula.fit_result.copula_param)
    else:
        e = rosenblatt_transform_scar(copula, u, copula.fit_result.alpha,
                                       K, grid_range)

    return cvm_test(e, seed=seed)


# ══════════════════════════════════════════════════════════════════
# Vine Rosenblatt transform
# ══════════════════════════════════════════════════════════════════

def _vine_edge_h(edge, u2, u1, u_pair, K=300, grid_range=5.0):
    """
    Compute h(u2 | u1; r) for a vine edge.

    Each vine edge is an independent bivariate copula, so this is
    identical to the bivariate Rosenblatt approach:
      MLE:  h(u2, u1; r) with constant r
      SCAR: E[h(u2, u1; Psi(z)) | u_{1:k-1}] via TM forward pass
    """
    method = edge.method.upper() if edge.method else 'MLE'

    if method == 'MLE':
        r = edge.get_r(u_pair)
        return edge.copula.h(u2, u1, r)
    else:
        from pyscarcopula.latent.ou_process import _tm_forward_mixture_h
        alpha = edge.fit_result.alpha
        theta, mu, nu = alpha
        return _tm_forward_mixture_h(theta, mu, nu, u_pair,
                                      edge.copula, K, grid_range)


def vine_rosenblatt_transform(vine, u, K=300, grid_range=5.0):
    """
    Rosenblatt transform for a fitted C-vine copula.

    Each edge in the vine is an independent bivariate copula
    (possibly with its own latent OU process). The vine Rosenblatt
    simply applies h-functions level by level, reusing the bivariate
    approach on every edge — no vine-specific modifications needed.

    v[0][i] = u_i
    v[j+1][i] = h(v[j][i+1] | v[j][0]; edge_{j,i})
    e_0 = u_0
    e_{j+1} = h(v[j][1] | v[j][0]; edge_{j,0})

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

def vine_gof_test(vine, data, to_pobs=True, seed=None, K=500, grid_range=7.0):
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
    seed : int or None
    K : int — grid size for SCAR mixture Rosenblatt
    grid_range : float

    Returns
    -------
    CramérVonMisesResult with .statistic and .pvalue
    """
    from pyscarcopula.utils import pobs as compute_pobs

    u = np.asarray(data, dtype=np.float64)
    if to_pobs:
        u = compute_pobs(u)

    if vine.edges is None:
        raise ValueError("Fit the vine first")

    e = vine_rosenblatt_transform(vine, u, K=K, grid_range=grid_range)
    return cvm_test(e, seed=seed)

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


def gaussian_gof_test(copula, data, to_pobs=True, seed=None):
    """
    Goodness-of-fit test for a fitted GaussianCopula.

    Parameters
    ----------
    copula : GaussianCopula (fitted, has .corr)
    data : (T, d)
    to_pobs : bool
    seed : int or None

    Returns
    -------
    CramérVonMisesResult
    """
    from pyscarcopula.utils import pobs as compute_pobs

    u = np.asarray(data, dtype=np.float64)
    if to_pobs:
        u = compute_pobs(u)

    if copula.corr is None:
        raise ValueError("Fit the copula first")

    e = gaussian_rosenblatt_transform(copula.corr, u)
    return cvm_test(e, seed=seed)


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


def student_gof_test(copula, data, to_pobs=True, seed=None):
    """
    Goodness-of-fit test for a fitted StudentCopula.

    Parameters
    ----------
    copula : StudentCopula (fitted, has .shape and .df)
    data : (T, d)
    to_pobs : bool
    seed : int or None

    Returns
    -------
    CramérVonMisesResult
    """
    from pyscarcopula.utils import pobs as compute_pobs

    u = np.asarray(data, dtype=np.float64)
    if to_pobs:
        u = compute_pobs(u)

    if copula.shape is None:
        raise ValueError("Fit the copula first")

    e = student_rosenblatt_transform(copula.shape, copula.df, u)
    return cvm_test(e, seed=seed)
