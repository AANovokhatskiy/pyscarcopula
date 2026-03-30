"""
pyscarcopula.strategy.gas — GAS estimation strategy.

Observation-driven model of Creal, Koopman and Lucas (2013).
All numerical computation delegated to numerical/gas_filter.py.
"""

import numpy as np
from scipy.optimize import minimize, Bounds

from pyscarcopula._types import (
    GASResult, NumericalConfig, DEFAULT_CONFIG,
    gas_params,
)
from pyscarcopula.strategy._base import register_strategy
from pyscarcopula.numerical.gas_filter import (
    gas_filter, gas_negloglik, gas_rosenblatt, gas_mixture_h,
)


@register_strategy('GAS')
class GASStrategy:
    """GAS estimation strategy.

    Parameters
    ----------
    config : NumericalConfig
    scaling : 'unit' or 'fisher'
        Score scaling type. 'unit' is default (simple, robust).
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
            from pyscarcopula.strategy.mle import MLEStrategy
            mle = MLEStrategy(config=self.config)
            mle_result = mle.fit(copula, u)
            mu_mle = float(np.atleast_1d(
                copula.inv_transform(np.atleast_1d(mle_result.copula_param))
            )[0])
            alpha0 = np.array([
                mu_mle * 0.05,    # omega ≈ f_bar * (1 - beta)
                0.05,             # alpha: moderate sensitivity
                0.95,             # beta: high persistence
            ])

        if verbose:
            print(f"GAS fit: alpha0={alpha0}, scaling={self.scaling}")

        bounds = Bounds(
            [-np.inf, -5.0, -0.999],
            [np.inf, 5.0, 0.999],
        )

        result = minimize(
            lambda x: gas_negloglik(x[0], x[1], x[2], u, copula,
                                    self.scaling, score_eps),
            alpha0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'gtol': tol, 'eps': 1e-5, 'maxfun': 200},
        )

        params = gas_params(
            omega=result.x[0],
            alpha=result.x[1],
            beta=result.x[2],
        )

        # Compute last filtered value for predict (avoids re-running filter)
        _, r_path, _ = gas_filter(
            result.x[0], result.x[1], result.x[2], u, copula,
            self.scaling, score_eps)
        r_last = float(r_path[-1]) if len(r_path) > 0 else 0.0

        return GASResult(
            log_likelihood=-result.fun,
            method='GAS',
            copula_name=copula.name,
            success=result.success,
            nfev=result.nfev,
            message=str(result.message),
            params=params,
            scaling=self.scaling,
            r_last=r_last,
        )

    def log_likelihood(self, copula, u: np.ndarray,
                       result: GASResult) -> float:
        """Evaluate GAS log-likelihood."""
        p = result.params
        _, _, ll = gas_filter(
            p.omega, p.alpha, p.beta, u, copula,
            result.scaling, self.config.gas_score_eps)
        return ll

    def smoothed_params(self, copula, u: np.ndarray,
                        result: GASResult) -> np.ndarray:
        """Deterministic Psi(f_t) path."""
        p = result.params
        _, r_path, _ = gas_filter(
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
        return gas_mixture_h(
            p.omega, p.alpha, p.beta, u, copula,
            result.scaling, self.config.gas_score_eps)

    def objective(self, copula, u: np.ndarray,
                  alpha: np.ndarray, **kwargs) -> float:
        """Minus log-likelihood: -logL(omega, alpha, beta)."""
        return gas_negloglik(
            alpha[0], alpha[1], alpha[2], u, copula,
            self.scaling, self.config.gas_score_eps)

    def sample(self, copula, u, result, n, rng=None, **kwargs):
        """Recursive GAS simulation.

        At each step t:
          1. r_t = Psi(f_t)
          2. Sample (u1_t, u2_t) from copula with r_t
          3. Compute score s_t from the sampled observation
          4. f_{t+1} = omega + beta*f_t + alpha*s_t
        """
        if rng is None:
            rng = np.random.default_rng()

        p = result.params
        omega, alpha_gas, beta = p.omega, p.alpha, p.beta
        score_eps = self.config.gas_score_eps

        F_CLIP = 50.0
        S_CLIP = 100.0

        # Initial f
        if abs(beta) < 1.0 - 1e-8:
            f_t = omega / (1.0 - beta)
        else:
            f_t = omega

        samples = np.empty((n, 2))

        for t in range(n):
            r_t = float(copula.transform(np.array([f_t]))[0])

            # Sample one observation from copula with r_t
            obs = copula.sample(1, np.array([r_t]), rng=rng)
            samples[t] = obs[0]

            # Compute score for next step
            if t < n - 1:
                u1 = obs[0:1, 0]
                u2 = obs[0:1, 1]

                f_plus = f_t + score_eps
                f_minus = f_t - score_eps
                r_plus = float(copula.transform(np.array([f_plus]))[0])
                r_minus = float(copula.transform(np.array([f_minus]))[0])

                ll_plus = float(copula.log_pdf(u1, u2, np.array([r_plus]))[0])
                ll_minus = float(copula.log_pdf(u1, u2, np.array([r_minus]))[0])

                nabla_t = (ll_plus - ll_minus) / (2.0 * score_eps)

                if self.scaling == 'fisher':
                    ll_t = float(copula.log_pdf(u1, u2, np.array([r_t]))[0])
                    d2 = (ll_plus - 2.0 * ll_t + ll_minus) / (score_eps ** 2)
                    fisher = max(-d2, 1e-6)
                    s_t = nabla_t / fisher
                else:
                    s_t = nabla_t

                s_t = np.clip(s_t, -S_CLIP, S_CLIP)
                f_t = omega + beta * f_t + alpha_gas * s_t
                f_t = np.clip(f_t, -F_CLIP, F_CLIP)

        return samples

    def predict(self, copula, u, result, n, rng=None, **kwargs):
        """Predict using last GAS-filtered value f_T.

        GAS path is deterministic given data, so the predictive
        distribution is a point mass at r = Psi(f_T).
        """
        if rng is None:
            rng = np.random.default_rng()

        r_path = self.smoothed_params(copula, u, result)
        r_T = float(r_path[-1])
        return copula.sample(n, np.full(n, r_T), rng=rng)