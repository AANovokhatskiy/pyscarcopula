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

from pyscarcopula._native import gas as native_gas
from pyscarcopula._native import scar_ou as native_ou


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


def _mle_info(copula, u, initial_mle_result=None):
    """Fit MLE and return (theta_mle, mu, log_likelihood)."""
    mle_result = initial_mle_result
    if mle_result is None:
        from pyscarcopula.strategy.mle import MLEStrategy
        mle_result = MLEStrategy().fit(copula, u)
    theta = float(np.atleast_1d(mle_result.copula_param)[0])
    mu = float(np.atleast_1d(
        copula.inv_transform(np.atleast_1d(theta))
    )[0])
    return theta, mu, float(mle_result.log_likelihood)


def _mle_mu(copula, u, initial_mle_result=None):
    """Fit MLE via strategy and return mu = inv_transform(copula_param)."""
    if initial_mle_result is None:
        return _mle_info(copula, u)[1]
    return _mle_info(copula, u, initial_mle_result)[1]


def resolve_ou_initial_point(
        copula, u, config, smart_init, verbose, alpha0,
        smart_initial_point_func=None, initial_mle_result=None):
    """Return an OU initial point and common initialization diagnostics."""
    if alpha0 is not None:
        alpha = np.asarray(alpha0, dtype=np.float64)
        return alpha, _explicit_initialization_diagnostics(alpha)

    smart_initial = (
        smart_initial_point
        if smart_initial_point_func is None
        else smart_initial_point_func
    )
    smart_diagnostics = None
    if smart_init:
        try:
            smart_kwargs = {'verbose': verbose}
            if initial_mle_result is not None:
                smart_kwargs['initial_mle_result'] = initial_mle_result
            alpha, info = smart_initial(u, copula, **smart_kwargs)
            if verbose:
                print(f"Smart init: {info.get('chosen_method')}, "
                      f"alpha0={alpha}")
            diagnostics = dict(info['initialization'])
            diagnostics['mle_source'] = (
                'selection_result' if initial_mle_result is not None
                else 'strategy_fit')
            return alpha, diagnostics
        except Exception as exc:
            smart_diagnostics = _initialization_diagnostics(
                'automatic',
                'failed',
                native_ou.default_initial_point(0.0)[0],
                [_initialization_attempt(
                    'smart_initial_point', success=False, error=exc)],
            )
            smart_diagnostics['success'] = False
            if verbose:
                print(
                    "Smart init failed "
                    f"({type(exc).__name__}: {exc}); trying mle_default")

    try:
        mle_result = initial_mle_result
        if mle_result is None:
            from pyscarcopula.strategy.mle import MLEStrategy
            mle_result = MLEStrategy(config=config).fit(copula, u)
        mu0 = float(np.atleast_1d(
            copula.inv_transform(
                np.atleast_1d(mle_result.copula_param))
        )[0])
        alpha = native_ou.default_initial_point(mu0)[0]
    except Exception as exc:
        if smart_diagnostics is not None:
            _fallback_initialization_diagnostics(
                smart_diagnostics,
                'mle_default',
                native_ou.default_initial_point(0.0)[0],
                error=exc,
            )
        raise

    if smart_diagnostics is None:
        diagnostics = _initialization_diagnostics(
            'mle_default',
            'mle_default',
            alpha,
            [_initialization_attempt('mle_default', success=True)],
        )
    else:
        diagnostics = _fallback_initialization_diagnostics(
            smart_diagnostics, 'mle_default', alpha)
    diagnostics['mle_source'] = (
        'selection_result' if initial_mle_result is not None
        else 'strategy_fit')
    return alpha, diagnostics


def _heuristic_initial_point(u, copula, rho_target=0.95,
                             sigma_frac=0.3, initial_mle_result=None):
    """
    Legacy analytical heuristic for (kappa, mu, nu).

    Uses the MLE constant parameter as the mean level mu, then sets kappa and
    nu from target autocorrelation and volatility assumptions.
    """
    mu = _mle_mu(copula, u, initial_mle_result)
    return native_ou.heuristic_initial_point(
        len(u), mu, rho_target, sigma_frac)[0]


def _kappa_from_target_autocorr(T, rho_target=0.96,
                                kappa_min=0.01, kappa_max=100.0):
    """Persistent start from a target one-step OU autocorrelation."""
    return native_ou.initial_kappa(
        T, rho_target, kappa_min, kappa_max)


def _stochastic_student_initial_point(
        u, copula, rho_target=0.96, nu0=0.1,
        initial_mle_result=None):
    """Initialize stochastic Student df dynamics near the static MLE."""
    u = np.asarray(u, dtype=np.float64)
    if initial_mle_result is None:
        df0, inverse_mu0, static_loglik = _mle_info(copula, u)
    else:
        df0, inverse_mu0, static_loglik = _mle_info(
            copula, u, initial_mle_result)
    alpha0, native_info = native_ou.stochastic_student_initial_point(
        len(u), df0, inverse_mu0, static_loglik, rho_target, nu0)
    info = {
        'method': 'stochastic_student_mle',
        'chosen_method': 'stochastic_student_mle',
        'theta_mle': df0,
        'df_mle': df0,
        'mu_mle': inverse_mu0,
        'mu0': float(alpha0[1]),
        'df_minus_two': float(native_info['df_minus_two']),
        'static_loglik': static_loglik,
        'rho_target': float(rho_target),
        'nu0': float(alpha0[2]),
    }
    return alpha0, info


def _strength_aware_initial_point(
        u, copula, rho_target=0.96, sigma_frac=0.3,
        weak_tau=0.06, strong_tau=0.25,
        weak_loglik_per_obs=0.003, strong_loglik_per_obs=0.04,
        weak_sigma_x=0.01, sigma_x_max=2.0,
        initial_mle_result=None):
    """
    Dependence-aware analytical heuristic for (kappa, mu, nu).

    The optimizer works in (kappa, mu, nu), but the identifiable amplitude
    near weak dependence is the stationary OU scale
    sigma_x = nu / sqrt(2*kappa).  Weak pairs start near the static MLE;
    stronger pairs smoothly approach the legacy broad sigma heuristic.
    """
    u = np.asarray(u, dtype=np.float64)
    if initial_mle_result is None:
        theta, mu, static_loglik = _mle_info(copula, u)
    else:
        theta, mu, static_loglik = _mle_info(
            copula, u, initial_mle_result)

    alpha0, native_info = native_ou.strength_aware_initial_point(
        u,
        theta,
        mu,
        static_loglik,
        rho_target=rho_target,
        sigma_fraction=sigma_frac,
        weak_tau=weak_tau,
        strong_tau=strong_tau,
        weak_log_likelihood_per_observation=weak_loglik_per_obs,
        strong_log_likelihood_per_observation=strong_loglik_per_obs,
        weak_stationary_scale=weak_sigma_x,
        maximum_stationary_scale=sigma_x_max,
    )
    info = {
        'method': 'strength_aware',
        'chosen_method': 'strength_aware',
        'theta_mle': theta,
        'mu_mle': mu,
        'static_loglik': static_loglik,
        'static_loglik_per_obs': float(
            native_info['static_log_likelihood_per_observation']),
        'tau_abs': float(native_info['tau_abs']),
        'strength': float(native_info['strength']),
        'regime': native_info['regime'],
        'sigma_x': float(native_info['sigma_x']),
        'sigma_x_legacy': float(native_info['sigma_x_legacy']),
        'weak_sigma_x': float(weak_sigma_x),
    }
    return alpha0, info


def _gas_initial_point(
        u, copula, verbose=False, initial_mle_result=None):
    """
    Estimate (kappa, mu, nu) via grid-search GAS + moment matching.

    Cost: O(20*T).  This path is retained for explicit ``use_gas=True``.
    """
    try:
        g_mle = _mle_mu(copula, u, initial_mle_result)
    except Exception:
        return native_ou.default_initial_point(0.0)[0]

    alpha0, native_info = native_gas.ou_initial_point(g_mle, u, copula)

    if verbose:
        print(
            f"  GAS grid: best logL={native_info['best_log_likelihood']:.2f}, "
            f"alpha0=[{alpha0[0]:.2f}, {alpha0[1]:.4f}, {alpha0[2]:.4f}]")

    return alpha0


def _fallback_initial_point(
        u, copula, use_gas=False, verbose=False,
        initial_mle_result=None):
    """
    Legacy smart initial point for SCAR-TM-OU optimization.

    Default (use_gas=False): broad analytical heuristic.
    With use_gas=True: use GAS moment matching when available, otherwise
    fall back to the broad analytical heuristic.
    """
    info = {}
    attempts = []

    try:
        if initial_mle_result is None:
            alpha_h = _heuristic_initial_point(u, copula)
        else:
            alpha_h = _heuristic_initial_point(
                u, copula, initial_mle_result=initial_mle_result)
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
            mu = _mle_mu(copula, u, initial_mle_result)
            alpha0 = native_ou.default_initial_point(mu)[0]
            attempts.append(_initialization_attempt(
                'mle_default', success=True))
        except Exception as exc:
            attempts.append(_initialization_attempt(
                'mle_default', success=False, error=exc))
            alpha0 = native_ou.default_initial_point(0.0)[0]
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
        gas_kwargs = {'verbose': verbose}
        if initial_mle_result is not None:
            gas_kwargs['initial_mle_result'] = initial_mle_result
        alpha_from_gas = _gas_initial_point(u, copula, **gas_kwargs)
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
    alpha0 = native_ou.default_initial_point(0.0)[0]
    attempts.append(_initialization_attempt(
        'constant_default', success=True))
    info['initialization'] = _initialization_diagnostics(
        'gas', 'constant_default', alpha0, attempts)
    return alpha0, info


def smart_initial_point(
        u, copula, use_gas=False, verbose=False,
        initial_mle_result=None):
    """
    Compute a static-MLE-based initial point for SCAR-TM-OU optimization.

    ``use_gas=True`` preserves the old explicit GAS warm-start behavior.
    Stochastic Student models start near the constant-df MLE. Other copulas
    use a stationary amplitude that broadens with dependence strength.
    """
    if use_gas:
        return _fallback_initial_point(
            u, copula, use_gas=True, verbose=verbose,
            initial_mle_result=initial_mle_result)

    static_df_mle = bool(getattr(
        copula, '_scar_static_df_mle_initialization', False))
    requested_method = (
        'stochastic_student_mle' if static_df_mle else 'strength_aware')
    try:
        if static_df_mle:
            if initial_mle_result is None:
                alpha0, info = _stochastic_student_initial_point(u, copula)
            else:
                alpha0, info = _stochastic_student_initial_point(
                    u, copula, initial_mle_result=initial_mle_result)
        else:
            if initial_mle_result is None:
                alpha0, info = _strength_aware_initial_point(u, copula)
            else:
                alpha0, info = _strength_aware_initial_point(
                    u, copula, initial_mle_result=initial_mle_result)
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
            u, copula, use_gas=False, verbose=verbose,
            initial_mle_result=initial_mle_result)
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
