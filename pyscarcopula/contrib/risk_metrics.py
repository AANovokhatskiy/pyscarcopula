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
    # Ensure total doesn't exceed N_mc
    total = n_per_node.sum()
    if total > N_mc:
        # Scale down proportionally, keeping at least 1 per node
        excess = total - N_mc
        # Remove from largest nodes first
        order = np.argsort(-n_per_node)
        for idx in order:
            if excess <= 0:
                break
            can_remove = n_per_node[idx] - 1
            remove = min(can_remove, excess)
            n_per_node[idx] -= remove
            excess -= remove
    elif total < N_mc:
        n_per_node[np.argmax(active_p)] += N_mc - total

    r = np.empty((N_mc, dim), dtype=np.float64)
    sw = np.empty(N_mc, dtype=np.float64)

    offset = 0
    for j in range(n_nodes):
        nj = int(n_per_node[j])
        if nj <= 0:
            continue
        end = offset + nj
        u_j = copula_sample_func(nj, float(np.atleast_1d(transform(active_z[j]))[0]))
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

def _process_chunk_fixed(args):
    """
    Process a chunk of rolling windows (fixed weights).

    Runs in a separate process. Numba compiles once per process,
    then all windows in the chunk run without overhead.
    """
    (chunk_start, chunk_end, data, method, copula_class, copula_kwargs,
     marginal_model, marg_params, gamma, window_len, N_mc,
     portfolio_weight, fit_kwargs) = args

    dim = data.shape[1]
    results = []
    use_tm_pipeline = (method.upper() == 'SCAR-TM-OU'
                       and hasattr(copula_class, 'xT_distribution'))

    for k in range(chunk_start, chunk_end):
        idx = k + window_len - 1
        uk = pobs(data[k:k + window_len])
        cop = copula_class(**copula_kwargs)
        cop.fit(uk, method=method, **fit_kwargs)

        if use_tm_pipeline and hasattr(cop, 'xT_distribution'):
            z, prob = cop.xT_distribution(uk)
            ppf_func = lambda u, _idx=idx: marginal_model.ppf(u, marg_params[_idx])
            r, sw = precompute_scenarios(
                z, prob, cop.sample, ppf_func,
                cop.transform, N_mc, dim)
            loss = loss_func(r, portfolio_weight)
            v, c = cvar_from_weighted_losses(loss, sw, gamma)
        else:
            u_pred = cop.predict(N_mc) if hasattr(cop, 'predict') else cop.sample(N_mc)
            r = marginal_model.ppf(u_pred, marg_params[idx])
            loss = loss_func(r, portfolio_weight)
            x0 = np.array([0.0])
            res = minimize(F_cvar_q, x0, args=(gamma, loss),
                           method='SLSQP', tol=1e-7)
            v, c = res.x[0], res.fun

        results.append((idx, v, c))

    return results


def _process_chunk_optimal(args):
    """
    Process a chunk of rolling windows (portfolio optimization).

    Same as _process_chunk_fixed but optimizes weights jointly.
    """
    (chunk_start, chunk_end, data, method, copula_class, copula_kwargs,
     marginal_model, marg_params, gamma, window_len, N_mc,
     fit_kwargs) = args

    dim = data.shape[1]
    eq_w = np.ones(dim) / dim
    constr = {'type': 'eq', 'fun': lambda x: np.sum(x[1:]) - 1.0}
    lb = np.zeros(dim + 1)
    lb[0] = -1.0
    rb = np.ones(dim + 1)
    bounds = Bounds(lb, rb)
    x0 = np.array([0.0, *eq_w])

    use_tm_pipeline = (method.upper() == 'SCAR-TM-OU'
                       and hasattr(copula_class, 'xT_distribution'))

    results = []

    for k in range(chunk_start, chunk_end):
        idx = k + window_len - 1
        uk = pobs(data[k:k + window_len])
        cop = copula_class(**copula_kwargs)
        cop.fit(uk, method=method, **fit_kwargs)

        if use_tm_pipeline and hasattr(cop, 'xT_distribution'):
            z, prob = cop.xT_distribution(uk)
            ppf_func = lambda u, _idx=idx: marginal_model.ppf(u, marg_params[_idx])
            r, sw = precompute_scenarios(
                z, prob, cop.sample, ppf_func,
                cop.transform, N_mc, dim)
            res = minimize(F_sc_gamma_wq, x0, args=(gamma, r, sw),
                           method='SLSQP', bounds=bounds,
                           constraints=constr, tol=1e-7)
        else:
            u_pred = cop.predict(N_mc) if hasattr(cop, 'predict') else cop.sample(N_mc)
            r = marginal_model.ppf(u_pred, marg_params[idx])
            res = minimize(F_cvar_wq, x0, args=(gamma, r),
                           method='SLSQP', bounds=bounds,
                           constraints=constr, tol=1e-7)

        results.append((idx, res.x[0], res.fun, res.x[1:dim + 1].copy()))
        x0 = res.x.copy()

    return results


def _make_chunks(n_windows, n_jobs):
    """Split n_windows into n_jobs roughly equal chunks."""
    if n_jobs <= 0:
        import os
        n_jobs = max(os.cpu_count() or 1, 1)
    n_jobs = min(n_jobs, n_windows)
    chunk_size = (n_windows + n_jobs - 1) // n_jobs
    chunks = []
    for i in range(n_jobs):
        start = i * chunk_size
        end = min(start + chunk_size, n_windows)
        if start < end:
            chunks.append((start, end))
    return chunks


def _get_copula_constructor(copula):
    """Extract copula class and kwargs for reconstruction in workers."""
    from pyscarcopula.copula.vine import CVineCopula
    if isinstance(copula, CVineCopula):
        return (CVineCopula,
                dict(candidates=copula.candidates,
                     allow_rotations=copula.allow_rotations,
                     criterion=copula.criterion))
    else:
        return (type(copula), dict(rotate=copula._rotate))


def _calculate_cvar_fixed(copula, data, method, marginal_model,
                          marg_params, gamma, window_len, N_mc,
                          portfolio_weight, n_jobs=1, **kwargs):
    """
    CVaR with fixed portfolio weights.

    Parameters
    ----------
    n_jobs : int
        Number of parallel workers. Default 1 (sequential).
        Use -1 for all CPU cores. Parallelization is over
        rolling windows via multiprocessing with chunk-based
        work distribution to amortize numba compilation.
    """
    T = len(data)
    dim = data.shape[1]
    var = np.zeros(T)
    cvar = np.zeros(T)
    n_windows = T - window_len + 1
    copula_class, copula_kwargs = _get_copula_constructor(copula)

    if n_jobs == 1:
        # Sequential (original behavior with tqdm)
        _has_xT = hasattr(copula, 'xT_distribution')
        for k in tqdm(range(n_windows)):
            idx = k + window_len - 1
            uk = pobs(data[k:k + window_len])

            if method.upper() == 'SCAR-TM-OU' and _has_xT:
                copula.fit(uk, method=method, **kwargs)
                z, prob = copula.xT_distribution(uk)
                ppf_func = lambda u, _idx=idx: marginal_model.ppf(
                    u, marg_params[_idx])
                r, sw = precompute_scenarios(
                    z, prob, copula.sample, ppf_func,
                    copula.transform, N_mc, dim)
                loss = loss_func(r, portfolio_weight)
                v, c = cvar_from_weighted_losses(loss, sw, gamma)
                var[idx] = v
                cvar[idx] = c
            else:
                copula.fit(uk, method=method, **kwargs)
                u_pred = (copula.predict(N_mc) if hasattr(copula, 'predict')
                          else copula.sample(N_mc))
                r = marginal_model.ppf(u_pred, marg_params[idx])
                loss = loss_func(r, portfolio_weight)
                x0 = np.array([0.0])
                res = minimize(F_cvar_q, x0, args=(gamma, loss),
                               method='SLSQP', tol=1e-7)
                var[idx] = res.x[0]
                cvar[idx] = res.fun
    else:
        import multiprocessing as mp

        chunks = _make_chunks(n_windows, n_jobs)
        pool_args = [
            (start, end, data, method, copula_class, copula_kwargs,
             marginal_model, marg_params, gamma, window_len, N_mc,
             portfolio_weight, kwargs)
            for start, end in chunks
        ]

        with mp.Pool(len(chunks)) as pool:
            chunk_results = pool.map(_process_chunk_fixed, pool_args)

        for chunk in chunk_results:
            for idx, v, c in chunk:
                var[idx] = v
                cvar[idx] = c

    return var, cvar, portfolio_weight


def _calculate_cvar_optimal(copula, data, method, marginal_model,
                            marg_params, gamma, window_len, N_mc,
                            n_jobs=1, **kwargs):
    """
    CVaR with portfolio weight optimization.

    Parameters
    ----------
    n_jobs : int
        Number of parallel workers. Default 1 (sequential).
        Note: with n_jobs > 1, x0 warm-starting between adjacent
        windows is lost within each chunk boundary.
    """
    T = len(data)
    dim = data.shape[1]
    eq_w = np.ones(dim) / dim
    copula_class, copula_kwargs = _get_copula_constructor(copula)

    var = np.zeros(T)
    cvar = np.zeros(T)
    weight_data = np.zeros((T, dim))

    if n_jobs == 1:
        constr = {'type': 'eq', 'fun': lambda x: np.sum(x[1:]) - 1.0}
        lb = np.zeros(dim + 1)
        lb[0] = -1.0
        rb = np.ones(dim + 1)
        bounds = Bounds(lb, rb)
        x0 = np.array([0.0, *eq_w])
        _has_xT = hasattr(copula, 'xT_distribution')

        for k in tqdm(range(T - window_len + 1)):
            idx = k + window_len - 1
            uk = pobs(data[k:k + window_len])

            if method.upper() == 'SCAR-TM-OU' and _has_xT:
                copula.fit(uk, method=method, **kwargs)
                z, prob = copula.xT_distribution(uk)
                ppf_func = lambda u, _idx=idx: marginal_model.ppf(
                    u, marg_params[_idx])
                r, sw = precompute_scenarios(
                    z, prob, copula.sample, ppf_func,
                    copula.transform, N_mc, dim)
                res = minimize(F_sc_gamma_wq, x0, args=(gamma, r, sw),
                               method='SLSQP', bounds=bounds,
                               constraints=constr, tol=1e-7)
            else:
                copula.fit(uk, method=method, **kwargs)
                u_pred = (copula.predict(N_mc) if hasattr(copula, 'predict')
                          else copula.sample(N_mc))
                r = marginal_model.ppf(u_pred, marg_params[idx])
                res = minimize(F_cvar_wq, x0, args=(gamma, r),
                               method='SLSQP', bounds=bounds,
                               constraints=constr, tol=1e-7)

            var[idx] = res.x[0]
            cvar[idx] = res.fun
            weight_data[idx] = res.x[1:dim + 1]
            x0 = res.x.copy()
    else:
        import multiprocessing as mp

        chunks = _make_chunks(T - window_len + 1, n_jobs)
        pool_args = [
            (start, end, data, method, copula_class, copula_kwargs,
             marginal_model, marg_params, gamma, window_len, N_mc,
             kwargs)
            for start, end in chunks
        ]

        with mp.Pool(len(chunks)) as pool:
            chunk_results = pool.map(_process_chunk_optimal, pool_args)

        for chunk in chunk_results:
            for idx, v, c, w in chunk:
                var[idx] = v
                cvar[idx] = c
                weight_data[idx] = w

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
                 n_jobs=1,
                 **kwargs):
    """
    Rolling VaR/CVaR estimation with copula models.

    Parameters
    ----------
    copula : BivariateCopula or CVineCopula
    data : (T, dim) log-returns
    window_len : int
    gamma : float or list — confidence level(s)
    N_mc : int or list — MC sample sizes
    marginals_method : str — 'normal', 'johnsonsu', etc.
    method : str — 'mle', 'scar-p-ou', 'scar-m-ou', 'scar-tm-ou'
    optimize_portfolio : bool
    portfolio_weight : (dim,) or None (equal weights)
    n_jobs : int
        Number of parallel workers for rolling window computation.
        Default 1 (sequential). Use -1 for all CPU cores.
        Each worker processes a contiguous chunk of windows,
        so numba compilation overhead is paid once per worker.
    **kwargs : forwarded to copula.fit()

    Returns
    -------
    dict: res[gamma][N_mc] = {'var': ..., 'cvar': ..., 'weight': ...}
    """
    from pyscarcopula.contrib.marginal import MarginalModel

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
            print(f"gamma={g}, N_mc={n}, method={method}, n_jobs={n_jobs}")
            if optimize_portfolio:
                var, cvar, w = _calculate_cvar_optimal(
                    copula, data, method, marginal_model,
                    marg_params, g, window_len, n,
                    n_jobs=n_jobs, **kwargs)
            else:
                var, cvar, w = _calculate_cvar_fixed(
                    copula, data, method, marginal_model,
                    marg_params, g, window_len, n,
                    portfolio_weight, n_jobs=n_jobs, **kwargs)
            res[g][n] = {'var': var, 'cvar': cvar, 'weight': w}

    return res