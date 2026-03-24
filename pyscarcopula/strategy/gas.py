"""
pyscarcopula.strategy.gas — Generalized Autoregressive Score model.

Observation-driven model of Creal, Koopman and Lucas (2013):

    f_t = omega + beta * f_{t-1} + alpha * s_{t-1}

where s_t = S_t * nabla_t is the scaled score.

Scaling options (preserved from original):
    'unit'   : S_t = 1  (identity — simple and robust)
    'fisher' : S_t = I_t^{-1}  (inverse Fisher information — optimal but less stable)

Key difference from SCAR: f_t is deterministic given past observations,
so the likelihood is closed-form (no MC or TM integration needed).

Parameters: (omega, alpha, beta) — 3 params, same as OU.
But note: this could be extended to GAS with different score scalings
or higher-order GAS(p,q) — the LatentProcessParams abstraction handles this.
"""

import numpy as np
from scipy.optimize import minimize, Bounds

from pyscarcopula._types import (
    GASResult, NumericalConfig, DEFAULT_CONFIG,
    LatentProcessParams, gas_params,
)
from pyscarcopula.strategy._base import register_strategy


# ══════════════════════════════════════════════════════════════════
# Core GAS filter — the mathematical heart
# ══════════════════════════════════════════════════════════════════

def _gas_filter(omega, alpha, beta, u, copula, scaling='unit',
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
        ll_t = float(copula.log_pdf(
            u[t:t+1, 0], u[t:t+1, 1], np.array([r_t]))[0])
        total_logL += ll_t

        # Score: d log c / d f (central differences in f-space)
        f_plus = f_t + score_eps
        f_minus = f_t - score_eps
        r_plus = float(copula.transform(np.array([f_plus]))[0])
        r_minus = float(copula.transform(np.array([f_minus]))[0])

        ll_plus = float(copula.log_pdf(
            u[t:t+1, 0], u[t:t+1, 1], np.array([r_plus]))[0])
        ll_minus = float(copula.log_pdf(
            u[t:t+1, 0], u[t:t+1, 1], np.array([r_minus]))[0])

        nabla_t = (ll_plus - ll_minus) / (2.0 * score_eps)

        # Scaling
        if scaling == 'fisher':
            # Fisher info: -d^2 log c / df^2 (central diff)
            fisher = -(ll_plus - 2.0 * ll_t + ll_minus) / (score_eps ** 2)
            if fisher > 1e-10:
                s_t = nabla_t / fisher
            else:
                s_t = nabla_t
        else:
            s_t = nabla_t

        # GAS update
        f_t = omega + beta * f_t + alpha * s_t

    return f_path, r_path, total_logL


def _gas_loglik(omega, alpha, beta, u, copula, scaling='unit',
                score_eps=1e-4):
    """Minus log-likelihood for optimizer."""
    try:
        _, _, ll = _gas_filter(omega, alpha, beta, u, copula,
                               scaling, score_eps)
        if np.isnan(ll) or np.isinf(ll):
            return 1e10
        return -ll
    except Exception:
        return 1e10


# ══════════════════════════════════════════════════════════════════
# GAS mixture h-function (for vine pseudo-observations)
# ══════════════════════════════════════════════════════════════════

def _gas_mixture_h(omega, alpha, beta, u, copula, scaling='unit',
                   score_eps=1e-4):
    """h(u2 | u1; Psi(f_t)) along GAS-filtered path.

    This is the GAS counterpart of SCAR's mixture h-function.
    Unlike SCAR (which integrates over the predictive distribution),
    GAS uses the point estimate Psi(f_t).
    """
    _, r_path, _ = _gas_filter(omega, alpha, beta, u, copula,
                               scaling, score_eps)
    return copula.h(u[:, 1], u[:, 0], r_path)


# ══════════════════════════════════════════════════════════════════
# Strategy class
# ══════════════════════════════════════════════════════════════════

@register_strategy('GAS')
class GASStrategy:
    """GAS estimation strategy.

    Parameters
    ----------
    config : NumericalConfig
    scaling : 'unit' or 'fisher'
        Score scaling type. Preserved from original implementation.
        'unit' is default (simple, robust).
        'fisher' uses inverse Fisher information (optimal but less stable).
    """

    def __init__(self, config: NumericalConfig | None = None,
                 scaling: str = 'unit', **kwargs):
        self.config = config or DEFAULT_CONFIG
        self.scaling = scaling

    def fit(self, copula, u: np.ndarray,
            alpha0: np.ndarray | None = None,
            tol: float | None = None,
            verbose: bool = False,
            **kwargs) -> GASResult:
        """Fit GAS model.

        Parameters
        ----------
        copula : CopulaProtocol
        u : (T, 2) pseudo-observations
        alpha0 : (3,) initial guess [omega, alpha, beta], or None
        tol : optimizer tolerance
        verbose : print progress
        **kwargs : ignored

        Returns
        -------
        GASResult
        """
        tol = tol or self.config.default_tol_gas
        score_eps = self.config.gas_score_eps

        # Default initial point
        if alpha0 is None:
            # Heuristic: low sensitivity, high persistence, mean near MLE
            from pyscarcopula.strategy.mle import MLEStrategy
            mle = MLEStrategy(config=self.config)
            mle_result = mle.fit(copula, u)
            mu_mle = float(np.atleast_1d(
                copula.inv_transform(np.atleast_1d(mle_result.copula_param))
            )[0])
            alpha0 = np.array([
                mu_mle * 0.01,   # omega: small
                0.05,             # alpha: moderate sensitivity
                0.95,             # beta: high persistence
            ])

        if verbose:
            print(f"GAS fit: alpha0={alpha0}, scaling={self.scaling}")

        # Bounds: omega free, alpha >= 0, |beta| < 1
        bounds = Bounds(
            [-np.inf, 0.0, -0.999],
            [np.inf, 10.0, 0.999],
        )

        result = minimize(
            lambda x: _gas_loglik(x[0], x[1], x[2], u, copula,
                                  self.scaling, score_eps),
            alpha0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'gtol': tol, 'maxfun': self.config.default_maxfun},
        )

        params = gas_params(
            omega=result.x[0],
            alpha=result.x[1],
            beta=result.x[2],
        )

        return GASResult(
            log_likelihood=-result.fun,
            method='GAS',
            copula_name=copula.name,
            success=result.success,
            nfev=result.nfev,
            message=str(result.message),
            params=params,
            scaling=self.scaling,
        )

    def log_likelihood(self, copula, u: np.ndarray,
                       result: GASResult) -> float:
        """Evaluate GAS log-likelihood."""
        p = result.params
        _, _, ll = _gas_filter(
            p.omega, p.alpha, p.beta, u, copula,
            result.scaling, self.config.gas_score_eps)
        return ll

    def smoothed_params(self, copula, u: np.ndarray,
                        result: GASResult) -> np.ndarray:
        """Deterministic Psi(f_t) path."""
        p = result.params
        _, r_path, _ = _gas_filter(
            p.omega, p.alpha, p.beta, u, copula,
            result.scaling, self.config.gas_score_eps)
        return r_path

    def rosenblatt_e2(self, copula, u: np.ndarray,
                      result: GASResult) -> np.ndarray:
        """e2 = h(u2, u1; Psi(f_t))."""
        r_path = self.smoothed_params(copula, u, result)
        return copula.h(u[:, 1], u[:, 0], r_path)

    def mixture_h(self, copula, u: np.ndarray,
                  result: GASResult) -> np.ndarray:
        """h along GAS-filtered path (point estimate, not mixture)."""
        p = result.params
        return _gas_mixture_h(
            p.omega, p.alpha, p.beta, u, copula,
            result.scaling, self.config.gas_score_eps)
