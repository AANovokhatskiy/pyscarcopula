"""
pyscarcopula.numerical.gas_filter — core GAS filter computation.

Single source of truth for the GAS (Generalized Autoregressive Score) filter.

Implements the observation-driven model of Creal, Koopman and Lucas (2013):

    f_t = omega + beta * f_{t-1} + alpha * s_{t-1}

where s_t = S_t * nabla_t is the scaled score.

Scaling options:
    'unit'   : S_t = 1  (identity — simple and robust)
    'fisher' : S_t = I_t^{-1}  (inverse Fisher information — optimal but less stable)

Used by:
    - strategy/gas.py (GASStrategy) for fitting and evaluation
    - latent/initial_point.py for GAS-based initial point estimation
    - stattests.py for GAS Rosenblatt transform
    - copula/vine.py for GAS h-functions in vine edges
"""

import numpy as np


# ══════════════════════════════════════════════════════════════════
# Core GAS filter
# ══════════════════════════════════════════════════════════════════

def gas_filter(omega, alpha, beta, u, copula, scaling='unit',
               score_eps=1e-4):
    """Run GAS filter, return full path and log-likelihood.

    At each step t:
      1. r_t = Psi(f_t) — copula parameter
      2. log c(u_t; r_t) — contributes to log L
      3. nabla_t = d log c / d f_t — numerical score (central diff in f-space)
      4. (Optional) Fisher scaling: I_t^{-1}
      5. f_{t+1} = omega + beta*f_t + alpha*s_t

    Parameters
    ----------
    omega, alpha, beta : float
        GAS recursion parameters.
    u : (T, 2) pseudo-observations
    copula : CopulaProtocol
    scaling : 'unit' or 'fisher'
        Score scaling. 'unit' uses S_t=1, 'fisher' uses inverse Fisher info.
    score_eps : float
        Step size for central difference score approximation.

    Returns
    -------
    f_path : (T,) — untransformed parameter path
    r_path : (T,) — copula parameter path Psi(f_t)
    total_logL : float
    """
    T = len(u)
    F_CLIP = 50.0        # prevent divergence of f
    S_CLIP = 100.0       # clip score to prevent explosions

    f_path = np.empty(T)
    r_path = np.empty(T)
    total_logL = 0.0

    # Initial value: unconditional mean f_bar = omega / (1 - beta)
    if abs(beta) < 1.0 - 1e-8:
        f_bar = omega / (1.0 - beta)
    else:
        f_bar = omega

    f_t = f_bar

    for t in range(T):
        f_path[t] = f_t
        r_t = float(copula.transform(np.array([f_t]))[0])
        r_path[t] = r_t

        # Log-likelihood contribution
        u1 = u[t:t+1, 0]
        u2 = u[t:t+1, 1]
        ll_t = float(copula.log_pdf(u1, u2, np.array([r_t]))[0])

        if not np.isfinite(ll_t):
            return f_path, r_path, -1e10  # signal failure

        total_logL += ll_t

        # Score for next step
        if t < T - 1:
            f_plus = f_t + score_eps
            f_minus = f_t - score_eps
            r_plus = float(copula.transform(np.array([f_plus]))[0])
            r_minus = float(copula.transform(np.array([f_minus]))[0])

            ll_plus = float(copula.log_pdf(u1, u2, np.array([r_plus]))[0])
            ll_minus = float(copula.log_pdf(u1, u2, np.array([r_minus]))[0])

            nabla_t = (ll_plus - ll_minus) / (2.0 * score_eps)

            # Scaling
            if scaling == 'fisher':
                d2 = (ll_plus - 2.0 * ll_t + ll_minus) / (score_eps ** 2)
                fisher = max(-d2, 1e-6)
                s_t = nabla_t / fisher
            else:
                s_t = nabla_t

            s_t = np.clip(s_t, -S_CLIP, S_CLIP)
            f_t = omega + beta * f_t + alpha * s_t
            f_t = np.clip(f_t, -F_CLIP, F_CLIP)

    return f_path, r_path, total_logL


def gas_negloglik(omega, alpha, beta, u, copula, scaling='unit',
                  score_eps=1e-4):
    """Minus log-likelihood for optimizer.

    Returns
    -------
    float : -logL (for minimization). Returns 1e10 on failure.
    """
    try:
        _, _, ll = gas_filter(omega, alpha, beta, u, copula,
                              scaling, score_eps)
        if np.isnan(ll) or np.isinf(ll):
            return 1e10
        return -ll
    except Exception:
        return 1e10


# ══════════════════════════════════════════════════════════════════
# GAS Rosenblatt transform (for GoF tests)
# ══════════════════════════════════════════════════════════════════

def gas_rosenblatt(omega, alpha, beta, u, copula, scaling='unit',
                   score_eps=1e-4):
    """Rosenblatt transform for the bivariate GAS copula model.

    Because GAS is observation-driven, theta_t is deterministic (no
    latent uncertainty to marginalize over). So the Rosenblatt is:
        e_1 = u_1,    e_2 = h(u_2, u_1; theta_t).

    Returns
    -------
    e : ndarray (T, 2)
    """
    _, r_path, _ = gas_filter(omega, alpha, beta, u, copula,
                              scaling, score_eps)
    T = len(u)
    e = np.empty((T, 2))
    e[:, 0] = u[:, 0]
    e[:, 1] = copula.h(u[:, 1], u[:, 0], r_path)
    return np.clip(e, 1e-6, 1.0 - 1e-6)


# ══════════════════════════════════════════════════════════════════
# GAS h-function (for vine pseudo-observations)
# ══════════════════════════════════════════════════════════════════

def gas_mixture_h(omega, alpha, beta, u, copula, scaling='unit',
                  score_eps=1e-4):
    """h(u2 | u1; Psi(f_t)) along GAS-filtered path.

    This is the GAS counterpart of SCAR's mixture h-function.
    Unlike SCAR (which integrates over the predictive distribution),
    GAS uses the point estimate Psi(f_t).

    Returns
    -------
    h_vals : ndarray (T,)
    """
    _, r_path, _ = gas_filter(omega, alpha, beta, u, copula,
                              scaling, score_eps)
    return np.clip(
        copula.h(u[:, 1], u[:, 0], r_path),
        1e-6, 1.0 - 1e-6)
