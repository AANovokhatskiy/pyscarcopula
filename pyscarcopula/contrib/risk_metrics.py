"""
Risk metrics: VaR, CVaR, portfolio optimization.

Pipeline (all methods: MLE, SCAR-P-OU, SCAR-M-OU, SCAR-TM-OU, GAS):
  copula.fit() -> copula.predict(N_mc) / copula.sample(N_mc)
  -> marginal ppf -> loss -> minimize F_gamma -> VaR, CVaR

Usage:
    from pyscarcopula.contrib.risk_metrics import risk_metrics

    result = risk_metrics(
        copula, data, window_len=250,
        gamma=0.95, N_mc=100000,
        marginals_method='johnsonsu',
        method='scar-tm-ou',
        optimize_portfolio=True,
    )
"""

import numpy as np
from numba import njit
from scipy.optimize import minimize, Bounds
from tqdm import tqdm
from typing import Literal

from pyscarcopula._utils import pobs


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
# Copula type detection helpers
# ══════════════════════════════════════════════════════════════════

def _is_elliptical_multivariate(copula):
    """Check if copula is a multivariate elliptical (Gaussian/Student).

    These have a different interface:
      - fit(data) without method parameter
      - sample(n) without r parameter
      - no _rotate attribute
    """
    from pyscarcopula.copula.elliptical import GaussianCopula, StudentCopula
    return isinstance(copula, (GaussianCopula, StudentCopula))


def _fit_copula(copula, uk, method, **kwargs):
    """Fit copula, handling interface differences."""
    if _is_elliptical_multivariate(copula):
        copula.fit(uk)
    else:
        copula.fit(uk, method=method, **kwargs)


def _risk_predict_kwargs(copula, kwargs):
    out = {}
    for name in ('horizon', 'predictive_r_mode'):
        if name in kwargs:
            out[name] = kwargs[name]

    from pyscarcopula.vine.cvine import CVineCopula
    if isinstance(copula, CVineCopula):
        for name in ('K', 'grid_range'):
            if name in kwargs:
                out[name] = kwargs[name]
    return out


def _predict_copula(copula, uk, N_mc, rng=None, **kwargs):
    """Sample from fitted copula for next-step prediction."""
    if _is_elliptical_multivariate(copula):
        # GaussianCopula/StudentCopula: simple predict (no method param)
        return copula.predict(N_mc, rng=rng)
    else:
        # BivariateCopula and CVineCopula: pass u explicitly
        return copula.predict(
            N_mc, u=uk, rng=rng, **_risk_predict_kwargs(copula, kwargs))


def _coerce_seed_sequence(rng=None):
    """Create a SeedSequence root for independent per-window RNGs."""
    if isinstance(rng, np.random.SeedSequence):
        return rng
    if isinstance(rng, np.random.Generator):
        return np.random.SeedSequence(rng.bit_generator.random_raw(4))
    return np.random.SeedSequence(rng)


# ══════════════════════════════════════════════════════════════════
# Copula constructor for multiprocessing
# ══════════════════════════════════════════════════════════════════

def _get_copula_constructor(copula):
    """Extract copula class and kwargs for reconstruction in workers."""
    from pyscarcopula.vine.cvine import CVineCopula
    from pyscarcopula.vine.rvine import RVineCopula
    from pyscarcopula.copula.elliptical import GaussianCopula, StudentCopula
    from pyscarcopula.copula.experimental.stochastic_student import StochasticStudentCopula
    from pyscarcopula.copula.experimental.equicorr import EquicorrGaussianCopula

    if isinstance(copula, CVineCopula):
        return (CVineCopula,
                dict(candidates=copula.candidates,
                     allow_rotations=copula.allow_rotations,
                     criterion=copula.criterion))
    elif isinstance(copula, RVineCopula):
        return (RVineCopula,
                dict(candidates=copula.candidates,
                     allow_rotations=copula.allow_rotations,
                     criterion=copula.criterion))
    elif isinstance(copula, StochasticStudentCopula):
        return (StochasticStudentCopula, dict(d=copula.d))
    elif isinstance(copula, EquicorrGaussianCopula):
        return (EquicorrGaussianCopula, dict(d=copula.d))
    elif isinstance(copula, (GaussianCopula, StudentCopula)):
        return (type(copula), {})
    else:
        return (type(copula), dict(rotate=copula._rotate))


# ══════════════════════════════════════════════════════════════════
# Rolling CVaR with fixed weights (MC pipeline)
# ══════════════════════════════════════════════════════════════════

def _process_chunk_fixed(args):
    """
    Process a chunk of rolling windows (fixed weights).

    Runs in a separate process. Numba compiles once per process,
    then all windows in the chunk run without overhead.
    """
    (chunk_start, chunk_end, data, method, copula_class, copula_kwargs,
     marginal_model, marg_params, gamma, window_len, N_mc,
     portfolio_weight, fit_kwargs, window_seeds) = args

    results = []

    for k in range(chunk_start, chunk_end):
        idx = k + window_len - 1
        uk = pobs(data[k:k + window_len])
        cop = copula_class(**copula_kwargs)
        _fit_copula(cop, uk, method, **fit_kwargs)

        rng = np.random.default_rng(window_seeds[k - chunk_start])
        u_pred = _predict_copula(cop, uk, N_mc, rng=rng, **fit_kwargs)
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
    """
    (chunk_start, chunk_end, data, method, copula_class, copula_kwargs,
     marginal_model, marg_params, gamma, window_len, N_mc,
     fit_kwargs, window_seeds) = args

    dim = data.shape[1]
    eq_w = np.ones(dim) / dim
    constr = {'type': 'eq', 'fun': lambda x: np.sum(x[1:]) - 1.0}
    lb = np.zeros(dim + 1)
    lb[0] = -1.0
    rb = np.ones(dim + 1)
    bounds = Bounds(lb, rb)
    x0 = np.array([0.0, *eq_w])

    results = []

    for k in range(chunk_start, chunk_end):
        idx = k + window_len - 1
        uk = pobs(data[k:k + window_len])
        cop = copula_class(**copula_kwargs)
        _fit_copula(cop, uk, method, **fit_kwargs)

        rng = np.random.default_rng(window_seeds[k - chunk_start])
        u_pred = _predict_copula(cop, uk, N_mc, rng=rng, **fit_kwargs)
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


def _calculate_cvar_fixed(copula, data, method, marginal_model,
                          marg_params, gamma, window_len, N_mc,
                          portfolio_weight, n_jobs=1,
                          window_seed_sequences=None, **kwargs):
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
    var = np.zeros(T)
    cvar = np.zeros(T)
    n_windows = T - window_len + 1
    copula_class, copula_kwargs = _get_copula_constructor(copula)
    if window_seed_sequences is None:
        window_seed_sequences = np.random.SeedSequence().spawn(n_windows)

    if n_jobs == 1:
        for k in tqdm(range(n_windows)):
            idx = k + window_len - 1
            uk = pobs(data[k:k + window_len])

            _fit_copula(copula, uk, method, **kwargs)
            rng = np.random.default_rng(window_seed_sequences[k])
            u_pred = _predict_copula(copula, uk, N_mc, rng=rng, **kwargs)
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
             portfolio_weight, kwargs, window_seed_sequences[start:end])
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
                            n_jobs=1, window_seed_sequences=None, **kwargs):
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
    n_windows = T - window_len + 1
    if window_seed_sequences is None:
        window_seed_sequences = np.random.SeedSequence().spawn(n_windows)

    if n_jobs == 1:
        constr = {'type': 'eq', 'fun': lambda x: np.sum(x[1:]) - 1.0}
        lb = np.zeros(dim + 1)
        lb[0] = -1.0
        rb = np.ones(dim + 1)
        bounds = Bounds(lb, rb)
        x0 = np.array([0.0, *eq_w])

        for k in tqdm(range(n_windows)):
            idx = k + window_len - 1
            uk = pobs(data[k:k + window_len])

            _fit_copula(copula, uk, method, **kwargs)
            rng = np.random.default_rng(window_seed_sequences[k])
            u_pred = _predict_copula(copula, uk, N_mc, rng=rng, **kwargs)
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

        chunks = _make_chunks(n_windows, n_jobs)
        pool_args = [
            (start, end, data, method, copula_class, copula_kwargs,
             marginal_model, marg_params, gamma, window_len, N_mc,
             kwargs, window_seed_sequences[start:end])
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
                                        'scar-tm-ou', 'gas'] = 'mle',
                 optimize_portfolio=True,
                 portfolio_weight=None,
                 n_jobs=1,
                 rng=None,
                 **kwargs):
    """
    Rolling VaR/CVaR estimation with copula models.

    Parameters
    ----------
    copula : BivariateCopula, GaussianCopula, StudentCopula, or CVineCopula
    data : (T, dim) log-returns
    window_len : int
    gamma : float or list — confidence level(s)
    N_mc : int or list — MC sample sizes
    marginals_method : str — 'normal', 'johnsonsu', etc.
    method : str — 'mle', 'scar-p-ou', 'scar-m-ou', 'scar-tm-ou', 'gas'
        Ignored for multivariate elliptical copulas (GaussianCopula,
        StudentCopula), which always use their own MLE fit.
    optimize_portfolio : bool
    portfolio_weight : (dim,) or None (equal weights)
    n_jobs : int
        Number of parallel workers for rolling window computation.
        Default 1 (sequential). Use -1 for all CPU cores.
        Each worker processes a contiguous chunk of windows,
        so numba compilation overhead is paid once per worker.
    rng : int, np.random.Generator, np.random.SeedSequence, or None
        Root randomness source. Independent child SeedSequences are spawned
        per rolling window, so parallel workers never share one Generator.
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
    root_seed_seq = _coerce_seed_sequence(rng)
    n_windows = T - window_len + 1

    res = {}
    for g in gammas:
        res[g] = {}
        for n in N_mcs:
            print(f"gamma={g}, N_mc={n}, method={method}, n_jobs={n_jobs}")
            window_seed_sequences = root_seed_seq.spawn(n_windows)
            if optimize_portfolio:
                var, cvar, w = _calculate_cvar_optimal(
                    copula, data, method, marginal_model,
                    marg_params, g, window_len, n,
                    n_jobs=n_jobs,
                    window_seed_sequences=window_seed_sequences,
                    **kwargs)
            else:
                var, cvar, w = _calculate_cvar_fixed(
                    copula, data, method, marginal_model,
                    marg_params, g, window_len, n,
                    portfolio_weight, n_jobs=n_jobs,
                    window_seed_sequences=window_seed_sequences,
                    **kwargs)
            res[g][n] = {'var': var, 'cvar': cvar, 'weight': w}

    return res
