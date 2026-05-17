"""
Smart initial point estimation for SCAR-TM-OU optimization.

The current default uses a dependence-aware analytical heuristic:

1. Fit a static MLE and use mu = inv_transform(theta_mle).
2. Estimate dependence strength from |Kendall tau| and static logL / T.
3. Initialize the OU stationary amplitude
   sigma_x = nu / sqrt(2*kappa) close to zero for weak pairs, and smoothly
   increase it toward the legacy broad heuristic for stronger pairs.

The previous heuristic is retained as ``legacy_smart_initial_point`` for
side-by-side experiments.  ``use_gas=True`` still requests the legacy GAS
moment-matching warm start.
"""

import numpy as np


def _mle_info(copula, u):
    """Fit MLE and return (theta_mle, mu, log_likelihood)."""
    from pyscarcopula.strategy.mle import MLEStrategy
    mle_result = MLEStrategy().fit(copula, u)
    theta = float(np.atleast_1d(mle_result.copula_param)[0])
    mu = float(np.atleast_1d(
        copula.inv_transform(np.atleast_1d(theta))
    )[0])
    return theta, mu, float(mle_result.log_likelihood)


def _mle_mu(copula, u):
    """Fit MLE via strategy and return mu = inv_transform(copula_param)."""
    return _mle_info(copula, u)[1]


def _heuristic_initial_point(u, copula, rho_target=0.95,
                             sigma_frac=0.3):
    """
    Legacy analytical heuristic for (kappa, mu, nu).

    Uses the MLE constant parameter as the mean level mu, then sets kappa and
    nu from target autocorrelation and volatility assumptions.
    """
    T = len(u)
    dt = 1.0 / (T - 1)

    mu = _mle_mu(copula, u)

    # kappa from target autocorrelation: rho = exp(-kappa*dt)
    kappa = -np.log(rho_target) / dt

    # sigma = fraction of |mu| (at least 1.0 to avoid degenerate nu)
    sigma = sigma_frac * max(abs(mu), 1.0)

    # nu from sigma: sigma^2 = nu^2 / (2*kappa)
    nu = sigma * np.sqrt(2.0 * kappa)

    kappa = np.clip(kappa, 0.01, 100.0)
    nu = np.clip(nu, 0.01, 50.0)

    return np.array([kappa, mu, nu])


def _kendall_tau_abs(u):
    """Absolute Kendall tau with a robust fallback."""
    try:
        from scipy.stats import kendalltau
        tau = float(kendalltau(u[:, 0], u[:, 1]).statistic)
    except Exception:
        tau = 0.0
    if not np.isfinite(tau):
        tau = 0.0
    return abs(tau)


def _kappa_from_target_autocorr(T, rho_target=0.96,
                                kappa_min=0.01, kappa_max=100.0):
    """Persistent start from a target one-step OU autocorrelation."""
    if T < 2:
        return float(kappa_min)
    dt = 1.0 / (T - 1)
    kappa = -np.log(rho_target) / dt
    return float(np.clip(kappa, kappa_min, kappa_max))


def _strength_aware_initial_point(
        u, copula, rho_target=0.96, sigma_frac=0.3,
        weak_tau=0.06, strong_tau=0.25,
        weak_loglik_per_obs=0.003, strong_loglik_per_obs=0.04,
        weak_sigma_x=0.01, sigma_x_max=2.0):
    """
    Dependence-aware analytical heuristic for (kappa, mu, nu).

    The optimizer works in (kappa, mu, nu), but the identifiable amplitude
    near weak dependence is the stationary OU scale
    sigma_x = nu / sqrt(2*kappa).  Weak pairs start near the static MLE;
    stronger pairs smoothly approach the legacy broad sigma heuristic.
    """
    u = np.asarray(u, dtype=np.float64)
    T = len(u)
    theta, mu, static_loglik = _mle_info(copula, u)

    tau_abs = _kendall_tau_abs(u)
    static_per_obs = static_loglik / max(T, 1)

    tau_strength = np.clip(
        (tau_abs - weak_tau) / max(strong_tau - weak_tau, 1e-12),
        0.0, 1.0)
    ll_strength = np.clip(
        (static_per_obs - weak_loglik_per_obs)
        / max(strong_loglik_per_obs - weak_loglik_per_obs, 1e-12),
        0.0, 1.0)
    strength = float(max(tau_strength, ll_strength))

    sigma_x_legacy = sigma_frac * max(abs(mu), 1.0)
    sigma_x_legacy = float(np.clip(
        sigma_x_legacy, weak_sigma_x, sigma_x_max))
    sigma_x = float(
        weak_sigma_x * (sigma_x_legacy / weak_sigma_x) ** strength)

    kappa = _kappa_from_target_autocorr(T, rho_target=rho_target)
    nu = sigma_x * np.sqrt(2.0 * kappa)
    nu = float(np.clip(nu, 0.01, 50.0))

    if tau_abs < weak_tau and static_per_obs < weak_loglik_per_obs:
        regime = 'weak'
    elif strength > 0.75:
        regime = 'strong'
    else:
        regime = 'medium'

    alpha0 = np.array([kappa, mu, nu], dtype=np.float64)
    info = {
        'method': 'strength_aware',
        'chosen_method': 'strength_aware',
        'theta_mle': theta,
        'mu_mle': mu,
        'static_loglik': static_loglik,
        'static_loglik_per_obs': static_per_obs,
        'tau_abs': tau_abs,
        'strength': strength,
        'regime': regime,
        'sigma_x': sigma_x,
        'sigma_x_legacy': sigma_x_legacy,
        'weak_sigma_x': weak_sigma_x,
    }
    return alpha0, info


def _gas_initial_point(u, copula, verbose=False):
    """
    Estimate (kappa, mu, nu) via grid-search GAS + moment matching.

    Cost: O(20*T).  This path is retained for explicit ``use_gas=True``.
    """
    from pyscarcopula.numerical.gas_filter import gas_filter

    T = len(u)
    dt = 1.0 / (T - 1)

    try:
        g_mle = _mle_mu(copula, u)
    except Exception:
        return np.array([1.0, 0.0, 1.0])

    best_ll = -1e10
    best_path = None

    for beta in [0.90, 0.95, 0.98, 0.99]:
        omega = g_mle * (1.0 - beta)
        for gamma_g in [0.01, 0.05, 0.1, 0.3, 0.5]:
            try:
                g_path, _, ll = gas_filter(
                    omega, gamma_g, beta, u, copula, 'unit')
                if ll > best_ll:
                    best_ll = ll
                    best_path = g_path.copy()
            except Exception:
                continue

    if best_path is None:
        return np.array([1.0, g_mle, 1.0])

    mu_est = np.mean(best_path)
    var_est = np.var(best_path)

    if var_est < 1e-10:
        return np.array([1.0, mu_est, 1.0])

    g_centered = best_path - mu_est
    autocov = np.mean(g_centered[:-1] * g_centered[1:])
    autocorr = np.clip(autocov / var_est, 0.01, 0.999)

    kappa_est = np.clip(-np.log(autocorr) / dt, 0.01, 100.0)
    nu_est = np.clip(np.sqrt(2.0 * kappa_est * var_est), 0.01, 50.0)

    if verbose:
        print(f"  GAS grid: best logL={best_ll:.2f}, "
              f"alpha0=[{kappa_est:.2f}, {mu_est:.4f}, {nu_est:.4f}]")

    return np.array([kappa_est, mu_est, nu_est])


def legacy_smart_initial_point(u, copula, use_gas=False, verbose=False):
    """
    Legacy smart initial point for SCAR-TM-OU optimization.

    Default (use_gas=False): broad analytical heuristic.
    With use_gas=True: use GAS moment matching when available, otherwise
    fall back to the broad analytical heuristic.
    """
    info = {}

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
            info['chosen_method'] = 'heuristic'
            return alpha_h, info
        try:
            mu = _mle_mu(copula, u)
            alpha0 = np.array([1.0, mu, 1.0])
        except Exception:
            alpha0 = np.array([1.0, 0.0, 1.0])
        info['method'] = 'mle_default'
        info['chosen_method'] = 'mle_default'
        return alpha0, info

    try:
        alpha_from_gas = _gas_initial_point(u, copula, verbose=verbose)
        info['gas_initial'] = alpha_from_gas.copy()
    except Exception:
        alpha_from_gas = None

    if alpha_from_gas is not None:
        info['method'] = 'gas'
        info['chosen_method'] = 'gas'
        return alpha_from_gas, info
    if alpha_h is not None:
        info['method'] = 'heuristic'
        info['chosen_method'] = 'heuristic'
        return alpha_h, info

    info['method'] = 'fallback'
    info['chosen_method'] = 'fallback'
    return np.array([1.0, 0.0, 1.0]), info


def legacy_smart_init(u, copula, use_gas=False, verbose=False):
    """Backward-compatible short alias for the legacy heuristic."""
    return legacy_smart_initial_point(
        u, copula, use_gas=use_gas, verbose=verbose)


def smart_initial_point(u, copula, use_gas=False, verbose=False):
    """
    Compute a dependence-aware initial point for SCAR-TM-OU optimization.

    ``use_gas=True`` preserves the old explicit GAS warm-start behavior.
    Otherwise, use a static-MLE-based start whose OU stationary amplitude is
    small for weak dependence and smoothly broadens for stronger pairs.
    """
    if use_gas:
        return legacy_smart_initial_point(
            u, copula, use_gas=True, verbose=verbose)

    try:
        alpha0, info = _strength_aware_initial_point(u, copula)
        if verbose:
            print(
                "  Strength-aware: "
                f"regime={info['regime']}, "
                f"tau_abs={info['tau_abs']:.4f}, "
                f"static_ll/T={info['static_loglik_per_obs']:.6f}, "
                f"strength={info['strength']:.3f}, "
                f"sigma_x={info['sigma_x']:.4f}, "
                f"alpha=[{alpha0[0]:.2f}, {alpha0[1]:.4f}, {alpha0[2]:.4f}]"
            )
        return alpha0, info
    except Exception:
        alpha0, info = legacy_smart_initial_point(
            u, copula, use_gas=False, verbose=verbose)
        info = dict(info)
        info['fallback_from'] = 'strength_aware'
        return alpha0, info
