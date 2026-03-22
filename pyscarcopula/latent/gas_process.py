"""
GAS (Generalized Autoregressive Score) process for dynamic copula models.

Implements the observation-driven model of Creal, Koopman and Lucas (2013):

    f_t = omega + beta * f_{t-1} + alpha * s_{t-1}

where f_t is the (untransformed) copula parameter at time t, and s_t is the
scaled score of the copula log-density with respect to f_t:

    s_t = S_t * nabla_t,
    nabla_t = d log c(u_{1t}, u_{2t}; Psi(f_t)) / d f_t.

Scaling options:
    'unit'   :  S_t = 1  (identity scaling — the default; simple and robust)
    'fisher' :  S_t = I_t^{-1}, where I_t = -d^2 log c / df_t^2
                (inverse Fisher information scaling — optimal but less stable)

Key difference from SCAR-OU:
    SCAR is parameter-driven — the latent x_t follows an independent OU
    process, and the likelihood requires T-dimensional integration (via
    transfer matrix or Monte Carlo).

    GAS is observation-driven — f_t is a deterministic function of past
    observations, so the likelihood is available in closed form.  No MC
    or transfer-matrix integration is needed.

Parameters: omega, alpha, beta  (3 parameters, same count as SCAR-OU).
Stationarity: |beta| < 1.  Unconditional mean: f_bar = omega / (1 - beta).

References
----------
Creal, D., Koopman, S.J. and Lucas, A. (2013).
    Generalized Autoregressive Score Models with Applications.
    Journal of Applied Econometrics, 28(5), 777--795.
"""

import numpy as np
from scipy.optimize import minimize, Bounds


# ══════════════════════════════════════════════════════════════════
# Core GAS filter (interleaved with copula evaluation)
# ══════════════════════════════════════════════════════════════════

def _gas_filter_full(omega, alpha, beta, u, copula, scaling='unit'):
    """
    Run the GAS filter, returning the full path and log-likelihood.

    At each step t:
      1. Evaluate log c(u_t; Psi(f_t))     — contributes to log L
      2. Compute numerical score nabla_t    — central differences in f-space
      3. (Optional) compute Fisher scaling
      4. Update: f_{t+1} = omega + beta*f_t + alpha*s_t

    Parameters
    ----------
    omega, alpha, beta : float
        GAS recursion parameters.
    u : ndarray (T, 2)
        Pseudo-observations.
    copula : BivariateCopula
    scaling : 'unit' or 'fisher'

    Returns
    -------
    f : ndarray (T,)
        Latent path f_t.
    theta : ndarray (T,)
        Copula parameter path Psi(f_t).
    log_lik : float
        Total log-likelihood.
    """
    T = len(u)
    eps = 1e-4           # finite-difference step
    F_CLIP = 50.0        # prevent divergence of f
    S_CLIP = 100.0       # clip score to prevent explosions

    # Stationary unconditional mean
    f = np.empty(T)
    f[0] = omega / (1.0 - beta) if abs(beta) < 1.0 else omega

    theta = np.empty(T)
    log_lik = 0.0

    for t in range(T):
        # --- copula parameter at this step ---
        r_c = np.atleast_1d(np.asarray(copula.transform(f[t]), dtype=np.float64))
        theta[t] = r_c[0]

        # --- log-density ---
        u1 = np.array([u[t, 0]])
        u2 = np.array([u[t, 1]])
        ll_c = copula.log_pdf(u1, u2, r_c)[0]

        if not np.isfinite(ll_c):
            return f, theta, -1e10  # signal failure
        log_lik += ll_c

        # --- score for next step ---
        if t < T - 1:
            r_p = np.atleast_1d(np.asarray(copula.transform(f[t] + eps), dtype=np.float64))
            r_m = np.atleast_1d(np.asarray(copula.transform(f[t] - eps), dtype=np.float64))

            ll_p = copula.log_pdf(u1, u2, r_p)[0]
            ll_m = copula.log_pdf(u1, u2, r_m)[0]

            nabla = (ll_p - ll_m) / (2.0 * eps)

            if scaling == 'fisher':
                d2 = (ll_p - 2.0 * ll_c + ll_m) / (eps * eps)
                fisher = max(-d2, 1e-6)
                s_t = nabla / fisher
            else:
                s_t = nabla

            s_t = np.clip(s_t, -S_CLIP, S_CLIP)
            f[t + 1] = omega + beta * f[t] + alpha * s_t
            f[t + 1] = np.clip(f[t + 1], -F_CLIP, F_CLIP)

    return f, theta, log_lik


def _gas_loglik(omega, alpha, beta, u, copula, scaling='unit'):
    """
    Compute minus log-likelihood for the GAS copula model.

    Returns
    -------
    float : minus log-likelihood  (for minimization)
    """
    _, _, log_lik = _gas_filter_full(omega, alpha, beta, u, copula, scaling)
    return -log_lik


# ══════════════════════════════════════════════════════════════════
# Rosenblatt transform (for GoF tests)
# ══════════════════════════════════════════════════════════════════

def _gas_rosenblatt(omega, alpha, beta, u, copula, scaling='unit'):
    """
    Rosenblatt transform for the bivariate GAS copula model.

    Because GAS is observation-driven, theta_t is deterministic (no
    latent uncertainty to marginalize over).  So the Rosenblatt is
    simply:
        e_1 = u_1,    e_2 = h(u_2, u_1; theta_t).

    Returns
    -------
    e : ndarray (T, 2)
    """
    f, theta, _ = _gas_filter_full(omega, alpha, beta, u, copula, scaling)
    T = len(u)
    e = np.empty((T, 2))
    e[:, 0] = u[:, 0]

    for t in range(T):
        r_arr = np.atleast_1d(np.asarray(theta[t], dtype=np.float64))
        e[t, 1] = copula.h(
            np.array([u[t, 1]]),
            np.array([u[t, 0]]),
            r_arr
        )[0]

    return np.clip(e, 1e-6, 1.0 - 1e-6)


def _gas_mixture_h(omega, alpha, beta, u, copula, scaling='unit'):
    """
    h-function values along the GAS-filtered path.

    Used in vine Rosenblatt: for each t, compute h(u2_t, u1_t; theta_t).
    Analogous to _tm_forward_mixture_h but deterministic.

    Returns
    -------
    h_vals : ndarray (T,)
    """
    f, theta, _ = _gas_filter_full(omega, alpha, beta, u, copula, scaling)
    T = len(u)
    h_vals = np.empty(T)
    for t in range(T):
        r_arr = np.atleast_1d(np.asarray(theta[t], dtype=np.float64))
        h_vals[t] = copula.h(
            np.array([u[t, 1]]),
            np.array([u[t, 0]]),
            r_arr
        )[0]
    return np.clip(h_vals, 1e-6, 1.0 - 1e-6)


# ══════════════════════════════════════════════════════════════════
# Main class
# ══════════════════════════════════════════════════════════════════

class GASProcess:
    """
    GAS (Generalized Autoregressive Score) process for dynamic copulas.

    Usage
    -----
    >>> cop = GumbelCopula(rotate=180)
    >>> gas = GASProcess(cop)
    >>> result = gas.fit(u)
    >>> print(result.log_likelihood)

    Or via the unified interface:
    >>> cop.fit(data, method='gas')

    Parameters
    ----------
    copula : BivariateCopula
    scaling : str
        'unit' (default) or 'fisher'.
    """

    def __init__(self, copula, scaling='unit'):
        self.copula = copula
        self.scaling = scaling
        self.fit_result = None

    def fit(self, u, alpha0=None, tol=1e-3, verbose=False):
        """
        Fit the GAS copula model via maximum likelihood.

        Parameters
        ----------
        u : ndarray (T, 2)
            Pseudo-observations.
        alpha0 : array-like (3,) or None
            Initial [omega, alpha, beta].  If None, initialized from MLE.
        tol : float
            Gradient tolerance for L-BFGS-B.
        verbose : bool

        Returns
        -------
        OptimizeResult with extra fields:
            .method = 'GAS'
            .gas_params = ndarray(3,) — [omega, alpha, beta]
            .alpha = gas_params   (compatibility with SCAR interface)
            .log_likelihood : float
            .name : str
            .scaling : str
        """
        u = np.asarray(u, dtype=np.float64)
        copula = self.copula

        # Initial guess: start near the MLE level with mild dynamics
        if alpha0 is None:
            mle = copula._fit_mle(u)
            f_mle = float(np.atleast_1d(copula.inv_transform(np.atleast_1d(mle.copula_param)))[0])
            # omega ≈ f_bar * (1 - beta),  beta = 0.95
            alpha0 = np.array([f_mle * 0.05, 0.05, 0.95])

        # Bounds: omega free, alpha in [-5, 5], beta in (-1, 1)
        bounds = Bounds(
            [-np.inf, -5.0, -0.999],
            [np.inf,   5.0,  0.999]
        )

        def objective(params):
            if np.any(np.isnan(params)):
                return 1e10
            try:
                return _gas_loglik(params[0], params[1], params[2],
                                   u, copula, self.scaling)
            except Exception:
                return 1e10

        if verbose:
            print(f"GAS fit: alpha0={alpha0}, scaling={self.scaling}")

        result = minimize(
            objective, alpha0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'gtol': tol, 'eps': 1e-5, 'maxfun': 200},
        )

        result.gas_params = result.x.copy()
        result.alpha = result.x.copy()     # SCAR-compatible field
        result.log_likelihood = -result.fun
        result.method = 'GAS'
        result.scaling = self.scaling
        result.name = copula.name
        self.fit_result = result

        if verbose:
            om, al, be = result.x
            print(f"  => omega={om:.4f}, alpha={al:.4f}, "
                  f"beta={be:.4f}, logL={result.log_likelihood:.4f}")

        return result

    def filter(self, u, params=None):
        """
        Run the GAS filter and return (f, theta, log_lik).

        Parameters
        ----------
        u : ndarray (T, 2)
        params : array (3,) or None — uses fitted if None

        Returns
        -------
        f : ndarray (T,)     — latent f_t
        theta : ndarray (T,) — copula param Psi(f_t)
        log_lik : float
        """
        if params is None:
            if self.fit_result is None:
                raise ValueError("Fit the model first or provide params")
            params = self.fit_result.gas_params
        omega, alpha, beta = params
        u = np.asarray(u, dtype=np.float64)
        return _gas_filter_full(omega, alpha, beta, u, self.copula,
                                self.scaling)

    def rosenblatt(self, u, params=None):
        """
        Rosenblatt transform for GoF test.  Returns (T, 2).
        """
        if params is None:
            if self.fit_result is None:
                raise ValueError("Fit the model first or provide params")
            params = self.fit_result.gas_params
        omega, alpha, beta = params
        u = np.asarray(u, dtype=np.float64)
        return _gas_rosenblatt(omega, alpha, beta, u, self.copula,
                               self.scaling)

    def smoothed_params(self, u, params=None):
        """
        Filtered copula parameter path theta_t = Psi(f_t).
        (For GAS there is no smoothing uncertainty — this is the
        deterministic filtered path, analogous to SCAR's
        E[Psi(x_k) | u_{1:k}].)
        """
        _, theta, _ = self.filter(u, params)
        return theta

    def mixture_h(self, u, params=None):
        """
        h-function values along the GAS path (for vine Rosenblatt).
        """
        if params is None:
            if self.fit_result is None:
                raise ValueError("Fit the model first or provide params")
            params = self.fit_result.gas_params
        omega, alpha, beta = params
        u = np.asarray(u, dtype=np.float64)
        return _gas_mixture_h(omega, alpha, beta, u, self.copula,
                              self.scaling)
