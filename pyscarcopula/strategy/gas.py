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
    PredictiveState,
)
from pyscarcopula.strategy._base import register_strategy
from pyscarcopula.numerical.gas_filter import (
    gas_filter, gas_predict_param, gas_negloglik, gas_rosenblatt,
    gas_mixture_h, _gas_score,
)
from pyscarcopula.strategy.predict_helpers import conditional_sample_bivariate


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
            gamma0: np.ndarray | None = None,
            tol: float | None = None,
            maxfun: int | None = None,
            score_eps: float | None = None,
            gamma_bound: float | None = None,
            beta_bound: float | None = None,
            verbose: bool = False,
            **kwargs) -> GASResult:
        """Fit GAS model.

        Parameters
        ----------
        copula : CopulaProtocol
        u : (T, 2) pseudo-observations
        gamma0 : (3,) initial guess [omega, gamma, beta], or None
        tol : optimizer tolerance
        verbose : print progress
        **kwargs : ignored

        Returns
        -------
        GASResult
        """
        tol = tol or self.config.default_tol_gas
        maxfun = int(maxfun if maxfun is not None
                     else self.config.default_maxfun_gas)
        score_eps = float(score_eps if score_eps is not None
                          else self.config.gas_score_eps)
        gamma_bound = float(gamma_bound if gamma_bound is not None
                            else self.config.gas_gamma_bound)
        beta_bound = float(beta_bound if beta_bound is not None
                           else self.config.gas_beta_bound)
        if maxfun <= 0:
            raise ValueError("maxfun must be positive")
        if gamma_bound <= 0:
            raise ValueError("gamma_bound must be positive")
        if not (0 < beta_bound < 1):
            raise ValueError("beta_bound must be in (0, 1)")

        # Default initial point
        if gamma0 is None:
            from pyscarcopula.strategy.mle import MLEStrategy
            mle = MLEStrategy(config=self.config)
            mle_result = mle.fit(copula, u)
            mu_mle = float(np.atleast_1d(
                copula.inv_transform(np.atleast_1d(mle_result.copula_param))
            )[0])
            gamma0 = np.array([
                mu_mle * 0.05,     # omega ≈ f_bar * (1 - beta)
                0.05,              # gamma: moderate score sensitivity
                0.95,              # beta: high persistence
            ])

        if verbose:
            print(
                f"GAS fit: gamma0={gamma0}, scaling={self.scaling}, "
                f"score_eps={score_eps}, gamma_bound={gamma_bound}, "
                f"beta_bound={beta_bound}"
            )

        bounds = Bounds(
            [-np.inf, -gamma_bound, -beta_bound],
            [np.inf, gamma_bound, beta_bound],
        )

        objective = lambda x: gas_negloglik(
            x[0], x[1], x[2], u, copula, self.scaling, score_eps)

        result = minimize(
            objective,
            gamma0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'gtol': tol, 'eps': 1e-5,
                     'maxfun': maxfun, 'maxiter': maxfun},
        )

        params = gas_params(
            omega=result.x[0],
            gamma=result.x[1],
            beta=result.x[2],
            gamma_bound=gamma_bound,
            beta_bound=beta_bound,
        )

        # Compute one-step-ahead value for predict (uses the final score).
        r_last = gas_predict_param(
            result.x[0], result.x[1], result.x[2], u, copula,
            self.scaling, score_eps)

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
            p.omega, p.gamma, p.beta, u, copula,
            result.scaling, self.config.gas_score_eps)
        return ll

    def predictive_mean(self, copula, u: np.ndarray,
                        result: GASResult) -> np.ndarray:
        """Deterministic Psi(g_t) path."""
        p = result.params
        _, r_path, _ = gas_filter(
            p.omega, p.gamma, p.beta, u, copula,
            result.scaling, self.config.gas_score_eps)
        return r_path

    def smoothed_params(self, copula, u: np.ndarray,
                        result: GASResult) -> np.ndarray:
        """Backward-compatible alias for predictive_mean."""
        return self.predictive_mean(copula, u, result)

    def rosenblatt_e2(self, copula, u: np.ndarray,
                      result: GASResult) -> np.ndarray:
        """e2 = h(u2, u1; Psi(g_t))."""
        r_path = self.predictive_mean(copula, u, result)
        return copula.h(u[:, 1], u[:, 0], r_path)

    def mixture_h(self, copula, u: np.ndarray,
                  result: GASResult) -> np.ndarray:
        """h along GAS-filtered path (point estimate, not mixture)."""
        p = result.params
        return gas_mixture_h(
            p.omega, p.gamma, p.beta, u, copula,
            result.scaling, self.config.gas_score_eps)

    def objective(self, copula, u: np.ndarray,
                  gamma: np.ndarray, **kwargs) -> float:
        """Minus log-likelihood: -logL(omega, gamma, beta)."""
        return gas_negloglik(
            gamma[0], gamma[1], gamma[2], u, copula,
            self.scaling, self.config.gas_score_eps)

    def sample(self, copula, u, result, n, rng=None, **kwargs):
        """Recursive GAS simulation.

        At each step t:
          1. r_t = Psi(g_t)
          2. Sample (u1_t, u2_t) from copula with r_t
          3. Compute score s_t from the sampled observation
          4. g_{t+1} = omega + beta*g_t + gamma*s_t
        """
        if rng is None:
            rng = np.random.default_rng()

        p = result.params
        omega, gamma_gas, beta = p.omega, p.gamma, p.beta
        score_eps = self.config.gas_score_eps

        G_CLIP = 50.0
        S_CLIP = 100.0

        # Initial g
        if abs(beta) < 1.0 - 1e-8:
            g_t = omega / (1.0 - beta)
        else:
            g_t = omega

        samples = np.empty((n, 2))

        for t in range(n):
            r_t = float(copula.transform(np.array([g_t]))[0])

            # Sample one observation from copula with r_t
            obs = copula.sample(1, np.array([r_t]), rng=rng)
            samples[t] = obs[0]

            # Compute score for next step
            if t < n - 1:
                u1 = obs[0:1, 0]
                u2 = obs[0:1, 1]

                ll_t = float(copula.log_pdf(u1, u2, np.array([r_t]))[0])
                s_t = _gas_score(
                    u1, u2, g_t, r_t, ll_t, copula,
                    self.scaling, score_eps)

                s_t = np.clip(s_t, -S_CLIP, S_CLIP)
                g_t = omega + beta * g_t + gamma_gas * s_t
                g_t = np.clip(g_t, -G_CLIP, G_CLIP)

        return samples

    def predict(self, copula, u, result, n, rng=None, **kwargs):
        """Predict using last GAS-filtered value g_T.

        GAS path is deterministic given data, so the predictive
        distribution is a point mass at r = Psi(g_T).
        """
        if rng is None:
            rng = np.random.default_rng()

        r = self.predictive_params(copula, u, result, n, rng=rng, **kwargs)
        return conditional_sample_bivariate(
            copula, n, r,
            given=kwargs.get('given'), rng=rng)

    def predictive_params(self, copula, u, result, n, rng=None, **kwargs):
        """Deterministic GAS predictive parameter path."""
        state = self.predictive_state(copula, u, result, **kwargs)
        return self.sample_params(copula, state, n, rng=rng, **kwargs)

    def predictive_state(self, copula, u, result, **kwargs):
        """Point-mass GAS predictive state."""
        if u is None or len(u) == 0:
            r_T = float(result.r_last)
        else:
            horizon = kwargs.get('horizon', 'next')
            p = result.params
            r_T = gas_predict_param(
                p.omega, p.gamma, p.beta, u, copula,
                result.scaling, self.config.gas_score_eps, horizon=horizon)
        return PredictiveState(
            method='GAS',
            horizon=str(kwargs.get('horizon', 'next')).lower(),
            kind='point',
            r=np.array([r_T], dtype=np.float64),
        )

    def condition_state(self, copula, state, observation, result, **kwargs):
        """Apply one GAS score update using a fully observed pair."""
        if observation is None:
            return state
        u = np.asarray(observation, dtype=np.float64)
        if u.ndim != 2 or u.shape[1] != 2 or len(u) == 0:
            return state
        # Prediction-time conditioning contributes one partial observation.
        # RVine samplers pass n identical rows, one per Monte Carlo draw.
        u = u[:1]

        p = result.params
        g_t = float(copula.inv_transform(np.array([float(state.r[0])]))[0])
        score_eps = self.config.gas_score_eps
        g_plus = g_t + score_eps
        g_minus = g_t - score_eps
        r_t = float(copula.transform(np.array([g_t]))[0])
        r_plus = float(copula.transform(np.array([g_plus]))[0])
        r_minus = float(copula.transform(np.array([g_minus]))[0])

        u1 = u[:, 0]
        u2 = u[:, 1]
        ll_t = float(np.sum(copula.log_pdf(u1, u2, np.full(len(u), r_t))))
        ll_plus = float(np.sum(
            copula.log_pdf(u1, u2, np.full(len(u), r_plus))))
        ll_minus = float(np.sum(
            copula.log_pdf(u1, u2, np.full(len(u), r_minus))))

        nabla = (ll_plus - ll_minus) / (2.0 * score_eps)
        if result.scaling == 'fisher':
            d2 = (ll_plus - 2.0 * ll_t + ll_minus) / (score_eps ** 2)
            fisher = max(-d2, 1e-6)
            score = nabla / fisher
        else:
            score = nabla

        score = float(np.clip(score, -100.0, 100.0))
        g_new = p.omega + p.beta * g_t + p.gamma * score
        g_new = float(np.clip(g_new, -50.0, 50.0))
        r_new = float(copula.transform(np.array([g_new]))[0])
        return PredictiveState(
            method=state.method,
            horizon=state.horizon,
            kind=state.kind,
            r=np.array([r_new], dtype=np.float64),
            metadata=dict(state.metadata),
        )

    def sample_params(self, copula, state, n, rng=None, **kwargs):
        return np.full(n, float(np.asarray(state.r)[0]), dtype=np.float64)
