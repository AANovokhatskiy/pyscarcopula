"""
Risk metrics: VaR, CVaR, portfolio optimization.

Two pipelines:
  1. MC-based (SCAR-P-OU, SCAR-M-OU, MLE): sample copula -> ppf -> loss -> CVaR
  2. Transfer-matrix (SCAR-TM-OU): xT_distribution -> precompute_scenarios -> F_sc_gamma

Usage:
    from pyscarcopula.risk import risk_metrics

    result = risk_metrics(
        copula, data, window_len=250,
        gamma=0.95, N_mc=100000,
        marginals_method='johnsonsu',
        method='scar-tm-ou',
        optimize_portfolio=True,
    )
"""

import numpy as np
from numba import njit, prange
from scipy.optimize import minimize, Bounds
from tqdm import tqdm
from typing import Literal

from pyscarcopula.utils import pobs


# ══════════════════════════════════════════════════════════════════
# Numba kernels
# ══════════════════════════════════════════════════════════════════

@njit(cache=True)
def loss_func(r, w):
    """
    Portfolio loss = 1 - sum_j exp(r_j) * w_j.
    r: (N, d), w: (d,). Returns: (N,).
    """
    N, d = r.shape
    loss = np.empty(N)
    for i in range(N):
        port = 0.0
        for j in range(d):
            port += np.exp(r[i, j]) * w[j]
        loss[i] = 1.0 - port
    return loss


@njit(cache=True)
def F_cvar_q(q, gamma, loss):
    """
    CVaR objective as function of q (VaR).
    F(q) = q + 1/(1-gamma) * mean(max(loss - q, 0)).
    q: (1,), loss: (N,).
    """
    N = len(loss)
    acc = 0.0
    for k in range(N):
        excess = loss[k] - q[0]
        if excess > 0.0:
            acc += excess
    return q[0] + acc / (N * (1.0 - gamma))


@njit(cache=True)
def F_cvar_wq(x, gamma, r):
    """
    Joint CVaR objective: x[0]=q, x[1:]=w.
    F(q,w) = q + 1/(1-gamma) * mean(max(1 - exp(r)@w - q, 0)).
    """
    q = x[0]
    d = r.shape[1]
    N = r.shape[0]
    acc = 0.0
    for i in range(N):
        port = 0.0
        for j in range(d):
            port += np.exp(r[i, j]) * x[1 + j]
        excess = 1.0 - port - q
        if excess > 0.0:
            acc += excess
    return q + acc / (N * (1.0 - gamma))


@njit(cache=True)
def F_sc_gamma(w, q, gamma, r, sw):
    """
    Weighted CVaR (formula 37) for transfer-matrix scenarios.
    F = q + 1/(1-gamma) * sum_i sw_i * max(1 - exp(r_i)@w - q, 0).
    r: (N, d), sw: (N,), sum(sw)=1.
    """
    N, d = r.shape
    acc = 0.0
    for i in range(N):
        port = 0.0
        for j in range(d):
            port += np.exp(r[i, j]) * w[j]
        excess = 1.0 - port - q
        if excess > 0.0:
            acc += sw[i] * excess
    return q + acc / (1.0 - gamma)


@njit(cache=True)
def F_sc_gamma_wq(x, gamma, r, sw):
    """Joint optimization wrapper: x[0]=q, x[1:]=w."""
    return F_sc_gamma(x[1:], x[0], gamma, r, sw)


# ══════════════════════════════════════════════════════════════════
# Weighted VaR / CVaR from scenarios
# ══════════════════════════════════════════════════════════════════

def cvar_from_weighted_losses(losses, weights, gamma):
    """
    Compute VaR and CVaR from weighted loss scenarios.
    losses: (N,), weights: (N,) summing to 1.
    Returns (var, cvar).
    """
    order = np.argsort(losses)
    sl = losses[order]
    sw = weights[order]
    cum_w = np.cumsum(sw)

    idx = np.searchsorted(cum_w, gamma)
    idx = min(idx, len(sl) - 1)
    var = sl[idx]

    tail_mask = sl >= var
    tail_w = sw[tail_mask]
    tail_l = sl[tail_mask]
    total_tail_w = tail_w.sum()
    cvar = (tail_w * tail_l).sum() / total_tail_w if total_tail_w > 0 else var

    return var, cvar


# ══════════════════════════════════════════════════════════════════
# Scenario generation for transfer-matrix pipeline
# ══════════════════════════════════════════════════════════════════

def precompute_scenarios(z_grid, prob,
                         copula_sample_func, marginal_ppf_func, transform,
                         N_mc, dim, rng=None):
    """
    Generate return scenarios from x_T distribution.

    Parameters
    ----------
    z_grid : (K,) — latent state grid
    prob : (K,) — probability weights summing to 1
    copula_sample_func : callable(n, r_param) -> (n, d)
    marginal_ppf_func : callable(u) -> (n, d)
    transform : callable(z) -> copula parameter
    N_mc : int
    dim : int
    rng : np.random.Generator or None

    Returns
    -------
    r : (N_total, dim) contiguous float64
    sw : (N_total,) contiguous float64, sum=1
    """
    if rng is None:
        rng = np.random.default_rng()

    mask = prob > 1e-8
    active_z = z_grid[mask]
    active_p = prob[mask]
    active_p /= active_p.sum()
    n_nodes = len(active_z)

    n_per_node = np.maximum(np.round(active_p * N_mc).astype(np.int64), 1)
    n_per_node[np.argmax(active_p)] += N_mc - n_per_node.sum()

    r = np.empty((N_mc, dim), dtype=np.float64)
    sw = np.empty(N_mc, dtype=np.float64)

    offset = 0
    for j in range(n_nodes):
        nj = int(n_per_node[j])
        if nj <= 0:
            continue
        end = offset + nj
        u_j = copula_sample_func(nj, transform(active_z[j]))
        r[offset:end] = marginal_ppf_func(u_j)
        sw[offset:end] = active_p[j] / nj
        offset = end

    r = np.ascontiguousarray(r[:offset])
    sw = np.ascontiguousarray(sw[:offset])
    sw /= sw.sum()
    return r, sw


# ══════════════════════════════════════════════════════════════════
# Empirical VaR / CVaR (rolling window)
# ══════════════════════════════════════════════════════════════════

def var_empirical(arr, gamma, window_len):
    """Rolling empirical VaR. arr: (T,), Returns: (T,)."""
    T = len(arr)
    res = np.zeros(T)
    for k in range(T - window_len + 1):
        idx = k + window_len - 1
        res[idx] = np.quantile(arr[k:k + window_len], gamma)
    return res


def cvar_empirical(arr, gamma, window_len):
    """Rolling empirical CVaR. arr: (T,). Returns: (T,)."""
    T = len(arr)
    res = np.zeros(T)
    for k in range(T - window_len + 1):
        idx = k + window_len - 1
        data = arr[k:k + window_len]
        q = np.quantile(data, gamma)
        tail = data[data <= q]
        res[idx] = np.mean(tail) if len(tail) > 0 else q
    return res


# ══════════════════════════════════════════════════════════════════
# Rolling CVaR with fixed weights (MC and TM pipelines)
# ══════════════════════════════════════════════════════════════════

def _calculate_cvar_fixed(copula, data, method, marginal_model,
                          marg_params, gamma, window_len, N_mc,
                          portfolio_weight, **kwargs):
    """CVaR with fixed portfolio weights."""
    T = len(data)
    dim = data.shape[1]
    var = np.zeros(T)
    cvar = np.zeros(T)

    for k in tqdm(range(T - window_len + 1)):
        idx = k + window_len - 1
        uk = pobs(data[k:k + window_len])

        if method.upper() == 'SCAR-TM-OU':
            # Transfer-matrix pipeline
            copula.fit(uk, method=method, **kwargs)
            z, prob = copula.xT_distribution(uk)
            ppf_func = lambda u, _idx=idx: marginal_model.ppf(u, marg_params[_idx])
            r, sw = precompute_scenarios(
                z, prob, copula.sample, ppf_func,
                copula.transform, N_mc, dim)
            loss = loss_func(r, portfolio_weight)
            v, c = cvar_from_weighted_losses(loss, sw, gamma)
            var[idx] = v
            cvar[idx] = c
        else:
            # MC pipeline
            copula.fit(uk, method=method, **kwargs)
            u_pred = copula.predict(N_mc)
            r = marginal_model.ppf(u_pred, marg_params[idx])
            loss = loss_func(r, portfolio_weight)
            x0 = np.array([0.0])
            res = minimize(F_cvar_q, x0, args=(gamma, loss),
                           method='SLSQP', tol=1e-7)
            var[idx] = res.x[0]
            cvar[idx] = res.fun

    return var, cvar, portfolio_weight


def _calculate_cvar_optimal(copula, data, method, marginal_model,
                            marg_params, gamma, window_len, N_mc, **kwargs):
    """CVaR with portfolio weight optimization."""
    T = len(data)
    dim = data.shape[1]
    eq_w = np.ones(dim) / dim

    var = np.zeros(T)
    cvar = np.zeros(T)
    weight_data = np.zeros((T, dim))

    constr = {'type': 'eq', 'fun': lambda x: np.sum(x[1:]) - 1.0}
    lb = np.zeros(dim + 1)
    lb[0] = -1.0
    rb = np.ones(dim + 1)
    bounds = Bounds(lb, rb)
    x0 = np.array([0.0, *eq_w])

    for k in tqdm(range(T - window_len + 1)):
        idx = k + window_len - 1
        uk = pobs(data[k:k + window_len])

        if method.upper() == 'SCAR-TM-OU':
            copula.fit(uk, method=method, **kwargs)
            z, prob = copula.xT_distribution(uk)
            ppf_func = lambda u, _idx=idx: marginal_model.ppf(u, marg_params[_idx])
            r, sw = precompute_scenarios(
                z, prob, copula.sample, ppf_func,
                copula.transform, N_mc, dim)
            res = minimize(F_sc_gamma_wq, x0, args=(gamma, r, sw),
                           method='SLSQP', bounds=bounds,
                           constraints=constr, tol=1e-7)
        else:
            copula.fit(uk, method=method, **kwargs)
            u_pred = copula.predict(N_mc)
            r = marginal_model.ppf(u_pred, marg_params[idx])
            res = minimize(F_cvar_wq, x0, args=(gamma, r),
                           method='SLSQP', bounds=bounds,
                           constraints=constr, tol=1e-7)

        var[idx] = res.x[0]
        cvar[idx] = res.fun
        weight_data[idx] = res.x[1:dim + 1]
        x0 = res.x.copy()

    return var, cvar, weight_data


# ══════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════

def risk_metrics(copula, data, window_len,
                 gamma=0.95, N_mc=100000,
                 marginals_method='johnsonsu',
                 method: Literal['mle', 'scar-p-ou', 'scar-m-ou', 
                                        'scar-tm-ou'] = 'scar-tm-ou',
                 optimize_portfolio=True,
                 portfolio_weight=None,
                 **kwargs):
    """
    Rolling VaR/CVaR estimation with copula models.

    Parameters
    ----------
    copula : BivariateCopula
    data : (T, dim) log-returns
    window_len : int
    gamma : float or list — confidence level(s)
    N_mc : int or list — MC sample sizes
    marginals_method : str — 'normal', 'johnsonsu', etc.
    method : str — 'mle', 'scar-p-ou', 'scar-m-ou', 'scar-tm-ou'
    optimize_portfolio : bool
    portfolio_weight : (dim,) or None (equal weights)
    **kwargs : forwarded to copula.fit()

    Returns
    -------
    dict: res[gamma][N_mc] = {'var': ..., 'cvar': ..., 'weight': ...}
    """
    from pyscarcopula.marginal import MarginalModel

    data = np.asarray(data, dtype=np.float64)
    T, dim = data.shape

    if window_len > T:
        raise ValueError(f"window_len={window_len} > T={T}")

    if portfolio_weight is None:
        portfolio_weight = np.ones(dim) / dim

    # Fit marginals
    marginal_model = MarginalModel.create(marginals_method)
    print(f"Fitting marginals ({marginals_method})...")
    marg_params = marginal_model.fit_rolling(data, window_len)

    # Normalize gamma / N_mc to lists
    gammas = [gamma] if not hasattr(gamma, '__iter__') else list(gamma)
    N_mcs = [N_mc] if not hasattr(N_mc, '__iter__') else list(N_mc)

    res = {}
    for g in gammas:
        res[g] = {}
        for n in N_mcs:
            print(f"gamma={g}, N_mc={n}, method={method}")
            if optimize_portfolio:
                var, cvar, w = _calculate_cvar_optimal(
                    copula, data, method, marginal_model,
                    marg_params, g, window_len, n, **kwargs)
            else:
                var, cvar, w = _calculate_cvar_fixed(
                    copula, data, method, marginal_model,
                    marg_params, g, window_len, n,
                    portfolio_weight, **kwargs)
            res[g][n] = {'var': var, 'cvar': cvar, 'weight': w}

    return res