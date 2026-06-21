"""
Smart initial point estimation for SCAR-TM-OU optimization.

The default uses one of two static-MLE-based analytical heuristics:

1. ``StochasticStudentCopula`` starts close to its constant-df MLE with a
   small diffusion coefficient.
2. Other copulas use the dependence-aware heuristic based on |Kendall tau|
   and static logL / T.

``use_gas=True`` requests the GAS moment-matching warm start.
"""

import numpy as np


def _initialization_attempt(method, *, success, error=None):
    """Build one JSON-serializable initialization attempt record."""
    attempt = {
        'method': str(method),
        'success': bool(success),
    }
    if error is not None:
        attempt['error_type'] = type(error).__name__
        attempt['error_message'] = str(error)
    return attempt


def _initialization_diagnostics(
        requested_method, selected_method, alpha0, attempts):
    """Build the common serializable initialization diagnostics payload."""
    alpha = np.asarray(alpha0, dtype=np.float64).reshape(-1)
    return {
        'requested_method': str(requested_method),
        'selected_method': str(selected_method),
        'alpha0': [float(value) for value in alpha],
        'attempts': [dict(attempt) for attempt in attempts],
        'success': True,
    }


def _explicit_initialization_diagnostics(alpha0):
    """Describe a user-provided initial point."""
    attempt = _initialization_attempt('user_provided', success=True)
    return _initialization_diagnostics(
        'user_provided', 'user_provided', alpha0, [attempt])


def _fallback_initialization_diagnostics(
        previous, method, alpha0, *, error=None):
    """Append a successful or failed fallback attempt."""
    attempts = list((previous or {}).get('attempts', ()))
    attempts.append(_initialization_attempt(
        method, success=error is None, error=error))
    selected = (
        str(method) if error is None
        else str((previous or {}).get('selected_method', 'failed')))
    requested = str((previous or {}).get(
        'requested_method', 'automatic'))
    diagnostics = _initialization_diagnostics(
        requested, selected, alpha0, attempts)
    diagnostics['success'] = error is None
    return diagnostics


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


def _stochastic_student_initial_point(
        u, copula, rho_target=0.96, nu0=0.1):
    """Initialize stochastic Student df dynamics near the static MLE."""
    u = np.asarray(u, dtype=np.float64)
    df0, inverse_mu0, static_loglik = _mle_info(copula, u)
    mu0 = float(inverse_mu0)
    kappa0 = _kappa_from_target_autocorr(
        len(u), rho_target=rho_target)
    nu0 = float(np.clip(nu0, 0.001, 50.0))

    alpha0 = np.array([kappa0, mu0, nu0], dtype=np.float64)
    info = {
        'method': 'stochastic_student_mle',
        'chosen_method': 'stochastic_student_mle',
        'theta_mle': df0,
        'df_mle': df0,
        'mu_mle': inverse_mu0,
        'mu0': mu0,
        'df_minus_two': df0 - 2.0,
        'static_loglik': static_loglik,
        'rho_target': float(rho_target),
        'nu0': nu0,
    }
    return alpha0, info


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
    from pyscarcopula.numerical.gas_filter import gas_filter, gas_loglik

    T = len(u)
    dt = 1.0 / (T - 1)

    try:
        g_mle = _mle_mu(copula, u)
    except Exception:
        return np.array([1.0, 0.0, 1.0])

    best_ll = -1e10
    best_params = None

    for beta in [0.90, 0.95, 0.98, 0.99]:
        omega = g_mle * (1.0 - beta)
        for gamma_g in [0.01, 0.05, 0.1, 0.3, 0.5]:
            try:
                ll = gas_loglik(
                    omega, gamma_g, beta, u, copula, 'unit')
                if ll > best_ll:
                    best_ll = ll
                    best_params = (omega, gamma_g, beta)
            except Exception:
                continue

    if best_params is None:
        return np.array([1.0, g_mle, 1.0])
    best_path, _, _ = gas_filter(
        *best_params, u, copula, 'unit')

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


def _fallback_initial_point(u, copula, use_gas=False, verbose=False):
    """
    Legacy smart initial point for SCAR-TM-OU optimization.

    Default (use_gas=False): broad analytical heuristic.
    With use_gas=True: use GAS moment matching when available, otherwise
    fall back to the broad analytical heuristic.
    """
    info = {}
    attempts = []

    try:
        alpha_h = _heuristic_initial_point(u, copula)
        attempts.append(_initialization_attempt(
            'heuristic', success=True))
        info['heuristic_alpha'] = alpha_h.copy()
        if verbose:
            print(f"  Heuristic: alpha=[{alpha_h[0]:.2f}, "
                  f"{alpha_h[1]:.4f}, {alpha_h[2]:.4f}]")
    except Exception as exc:
        attempts.append(_initialization_attempt(
            'heuristic', success=False, error=exc))
        alpha_h = None

    if not use_gas:
        if alpha_h is not None:
            info['method'] = 'heuristic'
            info['chosen_method'] = 'heuristic'
            info['initialization'] = _initialization_diagnostics(
                'legacy_heuristic', 'heuristic', alpha_h, attempts)
            return alpha_h, info
        try:
            mu = _mle_mu(copula, u)
            alpha0 = np.array([1.0, mu, 1.0])
            attempts.append(_initialization_attempt(
                'mle_default', success=True))
        except Exception as exc:
            attempts.append(_initialization_attempt(
                'mle_default', success=False, error=exc))
            alpha0 = np.array([1.0, 0.0, 1.0])
            attempts.append(_initialization_attempt(
                'constant_default', success=True))
        info['method'] = 'mle_default'
        info['chosen_method'] = 'mle_default'
        selected_method = (
            'mle_default'
            if attempts[-1]['method'] == 'mle_default'
            else 'constant_default')
        info['initialization'] = _initialization_diagnostics(
            'legacy_heuristic',
            selected_method,
            alpha0,
            attempts,
        )
        return alpha0, info

    try:
        alpha_from_gas = _gas_initial_point(u, copula, verbose=verbose)
        attempts.append(_initialization_attempt('gas', success=True))
        info['gas_initial'] = alpha_from_gas.copy()
    except Exception as exc:
        attempts.append(_initialization_attempt(
            'gas', success=False, error=exc))
        alpha_from_gas = None

    if alpha_from_gas is not None:
        info['method'] = 'gas'
        info['chosen_method'] = 'gas'
        info['initialization'] = _initialization_diagnostics(
            'gas', 'gas', alpha_from_gas, attempts)
        return alpha_from_gas, info
    if alpha_h is not None:
        info['method'] = 'heuristic'
        info['chosen_method'] = 'heuristic'
        info['initialization'] = _initialization_diagnostics(
            'gas', 'heuristic', alpha_h, attempts)
        return alpha_h, info

    info['method'] = 'fallback'
    info['chosen_method'] = 'fallback'
    alpha0 = np.array([1.0, 0.0, 1.0])
    attempts.append(_initialization_attempt(
        'constant_default', success=True))
    info['initialization'] = _initialization_diagnostics(
        'gas', 'constant_default', alpha0, attempts)
    return alpha0, info


def smart_initial_point(u, copula, use_gas=False, verbose=False):
    """
    Compute a static-MLE-based initial point for SCAR-TM-OU optimization.

    ``use_gas=True`` preserves the old explicit GAS warm-start behavior.
    Stochastic Student models start near the constant-df MLE. Other copulas
    use a stationary amplitude that broadens with dependence strength.
    """
    if use_gas:
        return _fallback_initial_point(
            u, copula, use_gas=True, verbose=verbose)

    static_df_mle = bool(getattr(
        copula, '_scar_static_df_mle_initialization', False))
    requested_method = (
        'stochastic_student_mle' if static_df_mle else 'strength_aware')
    try:
        if static_df_mle:
            alpha0, info = _stochastic_student_initial_point(u, copula)
        else:
            alpha0, info = _strength_aware_initial_point(u, copula)
        if verbose:
            if info['method'] == 'stochastic_student_mle':
                print(
                    "  Stochastic Student MLE init: "
                    f"df0={info['df_mle']:.4f}, "
                    f"alpha=[{alpha0[0]:.2f}, "
                    f"{alpha0[1]:.4f}, {alpha0[2]:.4f}]"
                )
            else:
                print(
                    "  Strength-aware: "
                    f"regime={info['regime']}, "
                    f"tau_abs={info['tau_abs']:.4f}, "
                    f"static_ll/T={info['static_loglik_per_obs']:.6f}, "
                    f"strength={info['strength']:.3f}, "
                    f"sigma_x={info['sigma_x']:.4f}, "
                    f"alpha=[{alpha0[0]:.2f}, "
                    f"{alpha0[1]:.4f}, {alpha0[2]:.4f}]"
                )
        info['initialization'] = _initialization_diagnostics(
            requested_method,
            info['chosen_method'],
            alpha0,
            [_initialization_attempt(
                requested_method, success=True)],
        )
        return alpha0, info
    except Exception as exc:
        alpha0, info = _fallback_initial_point(
            u, copula, use_gas=False, verbose=verbose)
        info = dict(info)
        info['fallback_from'] = requested_method
        legacy_diagnostics = info.get('initialization', {})
        attempts = [
            _initialization_attempt(
                requested_method, success=False, error=exc),
            *legacy_diagnostics.get('attempts', ()),
        ]
        info['initialization'] = _initialization_diagnostics(
            requested_method,
            legacy_diagnostics.get(
                'selected_method', info.get('chosen_method', 'fallback')),
            alpha0,
            attempts,
        )
        if verbose:
            selected = info['initialization']['selected_method']
            print(
                f"  {requested_method} init failed "
                f"({type(exc).__name__}: {exc}); using {selected}")
        return alpha0, info
