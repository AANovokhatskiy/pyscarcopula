"""
pyscarcopula.strategy.scar_tm — SCAR-TM-OU estimation strategy.

Transfer matrix method for deterministic likelihood evaluation of
the stochastic copula model with OU latent process.

This strategy wraps the numerical modules (tm_grid, tm_gradient,
tm_functions) into the FitStrategy interface, producing LatentResult.

All grid parameters (K, grid_range, pts_per_sigma, adaptive, grid_method)
and optimizer parameters (tol, analytical_grad, smart_init) are preserved
from the original OULatentProcess.fit() method.
"""

import numpy as np
from scipy.optimize import minimize, Bounds

from pyscarcopula._types import (
    LatentResult, NumericalConfig, DEFAULT_CONFIG,
    LatentProcessParams, ou_params,
)
from pyscarcopula.strategy._base import register_strategy
from pyscarcopula.numerical.tm_functions import (
    tm_loglik, tm_forward_smoothed,
    tm_forward_rosenblatt, tm_forward_mixture_h,
)
from pyscarcopula.numerical.tm_gradient import tm_loglik_with_grad
from pyscarcopula.strategy.predict_helpers import conditional_sample_bivariate


@register_strategy('SCAR-TM-OU')
class SCARTMStrategy:
    """Transfer matrix estimation for SCAR-OU model.

    Parameters
    ----------
    config : NumericalConfig
        Central numerical constants.
    K : int
        Minimum grid size. Auto-increased by adaptive rule (default 300).
    grid_range : float
        Grid spans [-grid_range*sigma, +grid_range*sigma] (default 5.0).
    grid_method : str
        'auto' (recommended), 'dense', or 'sparse'.
    adaptive : bool
        Adaptive grid refinement (default True).
    pts_per_sigma : int
        Points per conditional sigma for adaptive rule (default 2).
    analytical_grad : bool
        Use analytical gradient (default True).
        Reduces nfev by ~3-4x. Parameters are auto-rescaled.
    smart_init : bool
        Compute initial point via analytical heuristic (default True).
    """

    def __init__(self, config: NumericalConfig | None = None,
                 K: int | None = None,
                 grid_range: float | None = None,
                 grid_method: str | None = None,
                 adaptive: bool | None = None,
                 pts_per_sigma: int | None = None,
                 analytical_grad: bool = True,
                 smart_init: bool = True,
                 **kwargs):
        self.config = config or DEFAULT_CONFIG
        self.K = K if K is not None else self.config.default_K
        self.grid_range = grid_range if grid_range is not None else self.config.default_grid_range
        self.grid_method = grid_method if grid_method is not None else self.config.default_grid_method
        self.adaptive = adaptive if adaptive is not None else self.config.default_adaptive
        self.pts_per_sigma = pts_per_sigma if pts_per_sigma is not None else self.config.default_pts_per_sigma
        self.analytical_grad = analytical_grad
        self.smart_init = smart_init

    def fit(self, copula, u: np.ndarray,
            alpha0: np.ndarray | None = None,
            tol: float | None = None,
            verbose: bool = False,
            **kwargs) -> LatentResult:
        """Fit SCAR-TM-OU model.

        Parameters
        ----------
        copula : CopulaProtocol
        u : (T, 2) pseudo-observations
        alpha0 : (3,) initial [theta, mu, nu], or None for auto
        tol : gradient tolerance for L-BFGS-B
        verbose : print progress

        Returns
        -------
        LatentResult
        """
        tol = tol or self.config.default_tol_scar
        u = np.asarray(u, dtype=np.float64)

        # ── Initial point ─────────────────────────────────────────
        if alpha0 is None:
            if self.smart_init:
                try:
                    from pyscarcopula.strategy.initial_point import smart_initial_point
                    alpha0, init_info = smart_initial_point(
                        u, copula, verbose=verbose)
                    if verbose:
                        print(f"Smart init: {init_info.get('chosen_method')}, "
                              f"alpha0={alpha0}")
                except Exception:
                    alpha0 = None

            if alpha0 is None:
                # Fallback: MLE-based heuristic
                from pyscarcopula.strategy.mle import MLEStrategy
                mle = MLEStrategy(config=self.config)
                mle_result = mle.fit(copula, u)
                mu0 = float(np.atleast_1d(
                    copula.inv_transform(np.atleast_1d(mle_result.copula_param))
                )[0])
                alpha0 = np.array([1.0, mu0, 1.0])

        alpha0 = np.asarray(alpha0, dtype=np.float64)

        # ── Grid params for closures ──────────────────────────────
        K = self.K
        grid_range = self.grid_range
        grid_method = self.grid_method
        adaptive = self.adaptive
        pts_per_sigma = self.pts_per_sigma

        # ── Fit with analytical gradient ──────────────────────────
        if self.analytical_grad:
            # Rescale parameters so all three are O(1) at start.
            # This helps L-BFGS-B estimate the initial Hessian.
            scale = np.array([
                max(abs(alpha0[0]), 1.0),
                max(abs(alpha0[1]), 1.0),
                max(abs(alpha0[2]), 1.0),
            ])
            x0_scaled = alpha0 / scale
            bounds_scaled = Bounds(
                [0.001 / scale[0], -np.inf, 0.001 / scale[2]],
                [np.inf, np.inf, np.inf]
            )

            def objective_and_grad(x_scaled):
                alpha = x_scaled * scale
                if np.isnan(np.sum(alpha)):
                    return 1e10, np.zeros(3)
                th, mu_v, nu_v = alpha
                try:
                    val, grad = tm_loglik_with_grad(
                        th, mu_v, nu_v, u, copula, K, grid_range,
                        grid_method, adaptive, pts_per_sigma)
                    return val, grad * scale  # chain rule
                except Exception as e:
                    if verbose:
                        print(f"  error at alpha={alpha}: {e}")
                    return 1e10, np.zeros(3)

            if verbose:
                print(f"Fitting SCAR-TM-OU (analytical gradient), alpha0={alpha0}")

            result = minimize(
                objective_and_grad, x0_scaled,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds_scaled,
                options={'gtol': tol, 'maxfun': self.config.default_maxfun,
                         'maxiter': self.config.default_maxfun},
            )

            # Unscale
            result.x = result.x * scale

        # ── Fit with numerical gradient ───────────────────────────
        else:
            bounds = Bounds([0.001, -np.inf, 0.001], [np.inf, np.inf, np.inf])

            def objective(alpha):
                if np.isnan(np.sum(alpha)):
                    return 1e10
                th, mu_v, nu_v = alpha
                try:
                    return tm_loglik(
                        th, mu_v, nu_v, u, copula, K, grid_range,
                        grid_method, adaptive, pts_per_sigma)
                except Exception as e:
                    if verbose:
                        print(f"  error at alpha={alpha}: {e}")
                    return 1e10

            if verbose:
                print(f"Fitting SCAR-TM-OU (numerical gradient), alpha0={alpha0}")

            result = minimize(
                objective, alpha0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'gtol': tol, 'eps': 1e-4,
                         'maxfun': self.config.default_maxfun},
            )

        alpha = result.x
        if verbose:
            print(f"  => alpha={alpha}, logL={-result.fun:.4f}")

        params = ou_params(theta=alpha[0], mu=alpha[1], nu=alpha[2])

        return LatentResult(
            log_likelihood=-result.fun,
            method='SCAR-TM-OU',
            copula_name=copula.name,
            success=result.success,
            nfev=result.nfev,
            message=str(result.message),
            params=params,
            K=K,
            grid_range=grid_range,
            pts_per_sigma=pts_per_sigma,
        )

    def log_likelihood(self, copula, u: np.ndarray,
                       result: LatentResult) -> float:
        """Evaluate TM log-likelihood at fitted parameters."""
        p = result.params
        neg_ll = tm_loglik(
            p.theta, p.mu, p.nu, u, copula,
            self.K, self.grid_range, self.grid_method,
            self.adaptive, self.pts_per_sigma)
        return -neg_ll

    def smoothed_params(self, copula, u: np.ndarray,
                        result: LatentResult) -> np.ndarray:
        """E[Psi(x_k) | u_{1:k-1}] via TM forward pass."""
        p = result.params
        return tm_forward_smoothed(
            p.theta, p.mu, p.nu, u, copula,
            self.K, self.grid_range, self.grid_method,
            self.adaptive, self.pts_per_sigma)

    def rosenblatt_e2(self, copula, u: np.ndarray,
                      result: LatentResult) -> np.ndarray:
        """Mixture Rosenblatt: e2 = E[h(u2, u1; Psi(x_k)) | u_{1:k-1}]."""
        p = result.params
        e = tm_forward_rosenblatt(
            p.theta, p.mu, p.nu, u, copula,
            self.K, self.grid_range, self.grid_method,
            self.adaptive, self.pts_per_sigma)
        return e[:, 1]

    def mixture_h(self, copula, u: np.ndarray,
                  result: LatentResult) -> np.ndarray:
        """Mixture h-function for vine pseudo-obs propagation."""
        p = result.params
        return tm_forward_mixture_h(
            p.theta, p.mu, p.nu, u, copula,
            self.K, self.grid_range, self.grid_method,
            self.adaptive, self.pts_per_sigma)

    def objective(self, copula, u: np.ndarray,
                  alpha: np.ndarray, **kwargs) -> float:
        """Minus log-likelihood: TM integrated -logL(theta, mu, nu)."""
        try:
            return tm_loglik(
                alpha[0], alpha[1], alpha[2], u, copula,
                self.K, self.grid_range, self.grid_method,
                self.adaptive, self.pts_per_sigma)
        except Exception:
            return 1e10

    def sample(self, copula, u, result, n, rng=None, **kwargs):
        """Simulate n observations with OU-driven copula parameter.

        Generates an OU trajectory x(t), transforms to r(t) = Psi(x(t)),
        and samples from the copula with time-varying r(t).

        Uses the same time discretization as the model: dt = 1/(n-1),
        so the full trajectory covers [0, 1].
        """
        if rng is None:
            rng = np.random.default_rng()

        p = result.params
        theta, mu, nu = p.theta, p.mu, p.nu

        # Same dt convention as ou_sample_paths_exact
        dt = 1.0 / (n - 1) if n > 1 else 1.0
        rho_ou = np.exp(-theta * dt)
        sigma_cond = np.sqrt(nu ** 2 / (2.0 * theta) * (1.0 - rho_ou ** 2))

        x = np.empty(n)
        # Start from stationary distribution
        x[0] = rng.normal(mu, nu / np.sqrt(2.0 * theta))
        for t in range(1, n):
            x[t] = mu + rho_ou * (x[t - 1] - mu) + sigma_cond * rng.standard_normal()

        r = copula.transform(x)
        return copula.sample(n, r, rng=rng)

    def predict(self, copula, u, result, n, rng=None, **kwargs):
        """Mixture sampling from posterior p(x_T | data).

        Uses transfer matrix forward pass to compute the posterior
        distribution of x_T, then samples r from it.
        """
        if rng is None:
            rng = np.random.default_rng()

        r_samples = self.predictive_params(
            copula, u, result, n, rng=rng, **kwargs)
        return conditional_sample_bivariate(
            copula, n, r_samples, given=kwargs.get('given'), rng=rng)

    def predictive_params(self, copula, u, result, n, rng=None, **kwargs):
        """Sample predictive copula parameters for SCAR-TM."""
        if rng is None:
            rng = np.random.default_rng()

        p = result.params
        if u is None:
            sigma2 = p.nu ** 2 / (2.0 * p.theta)
            x_t = rng.normal(p.mu, np.sqrt(sigma2), n)
            return copula.transform(x_t)

        state_cache = kwargs.get('state_cache')
        cache_key = kwargs.get('cache_key')
        cached = None
        if state_cache is not None and cache_key is not None:
            cached = state_cache.get(cache_key)

        if cached is None:
            from pyscarcopula.numerical import predictive_tm
            cached = predictive_tm.tm_state_distribution(
                p.theta, p.mu, p.nu, u, copula,
                K=self.K,
                grid_range=self.grid_range,
                grid_method=self.grid_method,
                adaptive=self.adaptive,
                pts_per_sigma=self.pts_per_sigma,
                horizon=kwargs.get('horizon', 'next'))
            if state_cache is not None and cache_key is not None:
                state_cache[cache_key] = cached

        z_grid, prob = cached
        from pyscarcopula.numerical.predictive_tm import sample_grid_distribution
        mode = kwargs.get('predictive_r_mode')
        if mode is None:
            z_samples = sample_grid_distribution(z_grid, prob, n, rng)
        else:
            z_samples = sample_grid_distribution(
                z_grid, prob, n, rng, mode=mode)
        return copula.transform(z_samples)
