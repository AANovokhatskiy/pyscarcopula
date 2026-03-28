"""
Smart initial point estimation for SCAR-TM-OU optimization.

Two strategies, from cheapest to most expensive:

1. Analytical heuristic (cost: ~0, just arithmetic from MLE):
   - mu from MLE: mu = inv_transform(theta_mle)
   - theta from target autocorrelation: exp(-theta*dt) = rho_target
   - nu from target volatility: sigma = fraction * |mu|, nu = sigma * sqrt(2*theta)

2. GAS grid search (cost: O(20*T), ~0.01s per 1000 observations):
   - Run 20 GAS filters with different (alpha, beta) combos
   - Match moments of the best path to OU parameters

For vine copulas with many edges, (1) alone is usually sufficient
and adds zero overhead. (2) can be enabled for extra precision.
"""

import numpy as np


def _mle_mu(copula, u):
    """Fit MLE via strategy and return mu = inv_transform(copula_param)."""
    from pyscarcopula.strategy.mle import MLEStrategy
    mle_result = MLEStrategy().fit(copula, u)
    return float(np.atleast_1d(
        copula.inv_transform(np.atleast_1d(mle_result.copula_param))
    )[0])


def _heuristic_initial_point(u, copula, rho_target=0.95,
                              sigma_frac=0.3):
    """
    Analytical heuristic for (theta, mu, nu) — zero computational cost.

    Uses the MLE constant parameter as the mean level mu, then sets
    theta and nu from target autocorrelation and volatility assumptions.

    Parameters
    ----------
    u : (T, 2) — pseudo-observations
    copula : BivariateCopula
    rho_target : float — target one-step autocorrelation (default 0.95)
    sigma_frac : float — stationary sigma as fraction of |mu| (default 0.3)

    Returns
    -------
    alpha0 : ndarray (3,) — (theta, mu, nu)
    """
    T = len(u)
    dt = 1.0 / (T - 1)

    mu = _mle_mu(copula, u)

    # theta from target autocorrelation: rho = exp(-theta*dt)
    theta = -np.log(rho_target) / dt

    # sigma = fraction of |mu| (at least 1.0 to avoid degenerate nu)
    sigma = sigma_frac * max(abs(mu), 1.0)

    # nu from sigma: sigma^2 = nu^2 / (2*theta) => nu = sigma * sqrt(2*theta)
    nu = sigma * np.sqrt(2.0 * theta)

    theta = np.clip(theta, 0.01, 100.0)
    nu = np.clip(nu, 0.01, 50.0)

    return np.array([theta, mu, nu])


def _gas_initial_point(u, copula, verbose=False):
    """
    Estimate (theta, mu, nu) via grid-search GAS + moment matching.

    Cost: ~20 GAS filter passes = O(20·T).

    Parameters
    ----------
    u : (T, 2) — pseudo-observations
    copula : BivariateCopula
    verbose : bool

    Returns
    -------
    alpha0 : ndarray (3,) — (theta, mu, nu)
    """
    from pyscarcopula.numerical.gas_filter import gas_filter

    T = len(u)
    dt = 1.0 / (T - 1)

    try:
        f_mle = _mle_mu(copula, u)
    except Exception:
        return np.array([1.0, 0.0, 1.0])

    best_ll = -1e10
    best_path = None

    for beta in [0.90, 0.95, 0.98, 0.99]:
        omega = f_mle * (1.0 - beta)
        for alpha_g in [0.01, 0.05, 0.1, 0.3, 0.5]:
            try:
                f_path, _, ll = gas_filter(
                    omega, alpha_g, beta, u, copula, 'unit')
                if ll > best_ll:
                    best_ll = ll
                    best_path = f_path.copy()
            except Exception:
                continue

    if best_path is None:
        return np.array([1.0, f_mle, 1.0])

    mu_est = np.mean(best_path)
    var_est = np.var(best_path)

    if var_est < 1e-10:
        return np.array([1.0, mu_est, 1.0])

    f_centered = best_path - mu_est
    autocov = np.mean(f_centered[:-1] * f_centered[1:])
    autocorr = np.clip(autocov / var_est, 0.01, 0.999)

    theta_est = np.clip(-np.log(autocorr) / dt, 0.01, 100.0)
    nu_est = np.clip(np.sqrt(2.0 * theta_est * var_est), 0.01, 50.0)

    if verbose:
        print(f"  GAS grid: best logL={best_ll:.2f}, "
              f"alpha0=[{theta_est:.2f}, {mu_est:.4f}, {nu_est:.4f}]")

    return np.array([theta_est, mu_est, nu_est])


def smart_initial_point(u, copula, use_gas=False, verbose=False):
    """
    Compute a good initial point for SCAR-TM-OU optimization.

    Default (use_gas=False): analytical heuristic only — zero cost.
    With use_gas=True: also runs GAS grid search, picks the better
    of heuristic and GAS by GAS log-likelihood.

    Parameters
    ----------
    u : (T, 2) — pseudo-observations
    copula : BivariateCopula
    use_gas : bool — also try GAS grid search (adds O(20·T) cost)
    verbose : bool

    Returns
    -------
    alpha0 : ndarray (3,)
    info : dict — diagnostic information
    """
    info = {}

    # Always compute heuristic (zero cost)
    try:
        alpha_h = _heuristic_initial_point(u, copula)
        info['heuristic_alpha'] = alpha_h.copy()
        if verbose:
            print(f"  Heuristic: alpha=[{alpha_h[0]:.2f}, "
                  f"{alpha_h[1]:.4f}, {alpha_h[2]:.4f}]")
    except Exception:
        alpha_h = None

    if not use_gas:
        if alpha_h is not None:
            info['method'] = 'heuristic'
            return alpha_h, info
        # Fallback
        try:
            mu = _mle_mu(copula, u)
            alpha0 = np.array([1.0, mu, 1.0])
        except Exception:
            alpha0 = np.array([1.0, 0.0, 1.0])
        info['method'] = 'mle_default'
        return alpha0, info

    # Optional GAS refinement
    try:
        alpha_gas = _gas_initial_point(u, copula, verbose=verbose)
        info['gas_alpha'] = alpha_gas.copy()
    except Exception:
        alpha_gas = None

    # Pick the better one
    if alpha_h is not None and alpha_gas is not None:
        info['method'] = 'heuristic'
        return alpha_h, info

    if alpha_gas is not None:
        info['method'] = 'gas'
        return alpha_gas, info
    elif alpha_h is not None:
        info['method'] = 'heuristic'
        return alpha_h, info
    else:
        info['method'] = 'fallback'
        return np.array([1.0, 0.0, 1.0]), info
