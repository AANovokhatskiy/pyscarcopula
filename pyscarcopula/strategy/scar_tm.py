"""
pyscarcopula.strategy.scar_tm — SCAR-TM-OU estimation strategy.

Transfer matrix method for deterministic likelihood evaluation of
the stochastic copula model with OU latent process.

This strategy wraps the numerical modules (tm_grid, tm_gradient,
tm_functions) into the FitStrategy interface, producing LatentResult.

All grid parameters (K, grid_range, pts_per_sigma, adaptive, grid_method)
and optimizer parameters (gtol, ftol, maxfun, maxiter, maxls, eps,
analytical_grad, smart_init) are preserved
from the original OULatentProcess.fit() method.
"""

import numpy as np
from scipy.optimize import minimize, Bounds

from pyscarcopula._types import (
    LatentResult, NumericalConfig, DEFAULT_CONFIG,
    ou_params,
    PredictiveState,
)
from pyscarcopula.strategy._base import register_strategy
from pyscarcopula.numerical.tm_functions import (
    tm_loglik, tm_forward_predictive_mean,
    tm_forward_rosenblatt, tm_forward_mixture_h,
)
from pyscarcopula.numerical.tm_gradient import tm_loglik_with_grad
from pyscarcopula.numerical.auto_tm import (
    AutoTMConfig,
    auto_loglik,
    auto_neg_loglik,
    auto_neg_loglik_with_grad,
)
from pyscarcopula.numerical._arrays import as_float64_array
from pyscarcopula.numerical._transition_methods import (
    normalize_ou_transition_method,
)
from pyscarcopula.strategy.predict_helpers import sample_predictive


_INVALID_OBJECTIVE_THRESHOLD = 1e9


def _objective_is_invalid(value):
    return (not np.isfinite(value)) or float(value) >= _INVALID_OBJECTIVE_THRESHOLD


def _projected_gradient_norm(x, grad, lower, upper):
    """Infinity norm of the L-BFGS-B projected gradient."""
    x = np.asarray(x, dtype=np.float64)
    grad = np.asarray(grad, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)

    projected = grad.copy()
    at_lower = np.isclose(x, lower, rtol=0.0, atol=1e-10)
    at_upper = np.isclose(x, upper, rtol=0.0, atol=1e-10)
    projected[at_lower & (grad > 0.0)] = 0.0
    projected[at_upper & (grad < 0.0)] = 0.0
    return float(np.max(np.abs(projected)))


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
        Points per conditional sigma for adaptive rule (default from config).
    transition_method : str
        'auto' is the default likelihood evaluator: spectral Hermite for
        ordinary transitions and local Gauss-Hermite for narrow kernels or
        spectral numerical fallback.  'matrix', 'local', and 'spectral' force
        the corresponding likelihood backend.
    max_K : int or None
        Optional cap for adaptive TM grid size.  Defaults to 1000 in the
        strategy to prevent pathological fit-time grid blowups on long series.
    r_gh : float
        Locality threshold for auto transition selection.
    gh_order : int
        Gauss-Hermite order for local GH transition.
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
                 transition_method: str = 'auto',
                 max_K: int | None = 1000,
                 r_gh: float = 3.0,
                 gh_order: int = 5,
                 auto_small_kdt: float = 1e-3,
                 auto_large_kdt: float = 5e-2,
                 spectral_basis_order: int = 32,
                 spectral_quad_order: int | None = None,
                 analytical_grad: bool = True,
                 smart_init: bool = True,
                 **kwargs):
        self.config = config or DEFAULT_CONFIG
        self.K = K if K is not None else self.config.default_K
        self.grid_range = grid_range if grid_range is not None else self.config.default_grid_range
        self.grid_method = grid_method if grid_method is not None else self.config.default_grid_method
        self.adaptive = adaptive if adaptive is not None else self.config.default_adaptive
        self.pts_per_sigma = pts_per_sigma if pts_per_sigma is not None else self.config.default_pts_per_sigma
        self.transition_method = normalize_ou_transition_method(
            transition_method)
        self.max_K = max_K
        self.r_gh = r_gh
        self.gh_order = gh_order
        self.auto_small_kdt = auto_small_kdt
        self.auto_large_kdt = auto_large_kdt
        self.spectral_basis_order = spectral_basis_order
        self.spectral_quad_order = spectral_quad_order
        self.analytical_grad = analytical_grad
        self.smart_init = smart_init

    def _uses_dispatcher(self):
        return self.transition_method in {'auto', 'spectral'}

    def _grid_transition_method(self):
        if self.transition_method == 'spectral':
            return 'auto'
        return self.transition_method

    def _uses_local_transition(self):
        return self._grid_transition_method() != 'matrix' or self.max_K is not None

    def _tm_kwargs(self):
        if self.transition_method == 'matrix' and self.max_K is None:
            return {}
        return {
            'transition_method': self.transition_method,
            'max_K': self.max_K,
            'r_gh': self.r_gh,
            'gh_order': self.gh_order,
        }

    def _grid_tm_kwargs(self):
        transition_method = self._grid_transition_method()
        if transition_method == 'matrix' and self.max_K is None:
            return {}
        return {
            'transition_method': transition_method,
            'max_K': self.max_K,
            'r_gh': self.r_gh,
            'gh_order': self.gh_order,
        }

    def _auto_config(self):
        return AutoTMConfig(
            transition_method=self.transition_method,
            small_kdt=self.auto_small_kdt,
            large_kdt=self.auto_large_kdt,
            basis_order=self.spectral_basis_order,
            quad_order=self.spectral_quad_order,
            K=self.K,
            grid_range=self.grid_range,
            grid_method=self.grid_method,
            adaptive=self.adaptive,
            pts_per_sigma=self.pts_per_sigma,
            max_K=self.max_K,
            gh_order=self.gh_order,
            r_gh=self.r_gh,
        )

    def fit(self, copula, u: np.ndarray,
            alpha0: np.ndarray | None = None,
            gtol: float | None = None,
            ftol: float | None = None,
            maxfun: int | None = None,
            maxiter: int | None = None,
            maxls: int | None = None,
            eps: float | None = None,
            maxcor: int | None = None,
            finite_diff_rel_step: float | None = None,
            verbose: bool = False,
            **kwargs) -> LatentResult:
        """Fit SCAR-TM-OU model.

        Parameters
        ----------
        copula : CopulaProtocol
        u : (T, 2) pseudo-observations
        alpha0 : (3,) initial [kappa, mu, nu], or None for auto
        gtol, ftol, maxfun, maxiter, maxls, eps, maxcor,
        finite_diff_rel_step : L-BFGS-B options
        verbose : print progress

        Returns
        -------
        LatentResult
        """
        if 'tol' in kwargs:
            raise TypeError("tol is not supported; use gtol")
        optimizer_options = self.config.scar_optimizer.options(
            gtol=gtol,
            ftol=ftol,
            maxfun=maxfun,
            maxiter=maxiter,
            maxls=maxls,
            eps=eps,
            maxcor=maxcor,
            finite_diff_rel_step=finite_diff_rel_step,
        )
        u = as_float64_array(u)

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
        tm_kwargs = self._tm_kwargs()
        auto_config = self._auto_config()

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
                kappa_v, mu_v, nu_v = alpha
                try:
                    if self._uses_dispatcher():
                        val, grad = auto_neg_loglik_with_grad(
                            kappa_v, mu_v, nu_v, u, copula, auto_config)
                    else:
                        val, grad = tm_loglik_with_grad(
                            kappa_v, mu_v, nu_v, u, copula, K, grid_range,
                            grid_method, adaptive, pts_per_sigma, **tm_kwargs)
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
                options=optimizer_options,
            )

            if not result.success and str(result.message).startswith('ABNORMAL'):
                final_val, final_grad = objective_and_grad(result.x)
                pg_norm = _projected_gradient_norm(
                    result.x,
                    final_grad,
                    bounds_scaled.lb,
                    bounds_scaled.ub,
                )
                acceptable_boundary = (
                    np.isfinite(final_val)
                    and not _objective_is_invalid(final_val)
                    and np.all(np.isfinite(result.x))
                    and pg_norm <= max(float(optimizer_options.get('gtol', 1e-5)), 1e-2)
                )
                if acceptable_boundary:
                    result.fun = final_val
                    result.jac = final_grad
                    result.success = True
                    result.message = (
                        f"{result.message} accepted as boundary convergence "
                        f"(projected_grad={pg_norm:.3g})"
                    )

            # Unscale
            result.x = result.x * scale

        # ── Fit with numerical gradient ───────────────────────────
        else:
            bounds = Bounds([0.001, -np.inf, 0.001], [np.inf, np.inf, np.inf])

            def objective(alpha):
                if np.isnan(np.sum(alpha)):
                    return 1e10
                kappa_v, mu_v, nu_v = alpha
                try:
                    if self._uses_dispatcher():
                        return auto_neg_loglik(
                            kappa_v, mu_v, nu_v, u, copula, auto_config)
                    return tm_loglik(
                        kappa_v, mu_v, nu_v, u, copula, K, grid_range,
                        grid_method, adaptive, pts_per_sigma, **tm_kwargs)
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
                options=optimizer_options,
            )

        if _objective_is_invalid(result.fun):
            result.success = False
            result.message = (
                f"{result.message}; invalid objective value {float(result.fun):.6g}"
            )

        alpha = result.x
        if verbose:
            print(f"  => alpha={alpha}, logL={-result.fun:.4f}")

        params = ou_params(kappa=alpha[0], mu=alpha[1], nu=alpha[2])

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
            transition_method=self.transition_method,
            max_K=self.max_K,
            r_gh=self.r_gh,
            gh_order=self.gh_order,
            auto_small_kdt=self.auto_small_kdt,
            auto_large_kdt=self.auto_large_kdt,
            spectral_basis_order=self.spectral_basis_order,
            spectral_quad_order=self.spectral_quad_order,
        )

    def log_likelihood(self, copula, u: np.ndarray,
                       result: LatentResult) -> float:
        """Evaluate TM log-likelihood at fitted parameters."""
        p = result.params
        if self._uses_dispatcher():
            return auto_loglik(
                p.kappa, p.mu, p.nu, u, copula, self._auto_config())
        neg_ll = tm_loglik(
            p.kappa, p.mu, p.nu, u, copula,
            self.K, self.grid_range, self.grid_method,
            self.adaptive, self.pts_per_sigma, **self._tm_kwargs())
        return -neg_ll

    def predictive_mean(self, copula, u: np.ndarray,
                        result: LatentResult) -> np.ndarray:
        """E[Psi(x_k) | u_{1:k-1}] via TM forward pass."""
        p = result.params
        return tm_forward_predictive_mean(
            p.kappa, p.mu, p.nu, u, copula,
            self.K, self.grid_range, self.grid_method,
            self.adaptive, self.pts_per_sigma, **self._grid_tm_kwargs())

    def rosenblatt_e2(self, copula, u: np.ndarray,
                      result: LatentResult) -> np.ndarray:
        """Mixture Rosenblatt: e2 = E[h(u2, u1; Psi(x_k)) | u_{1:k-1}]."""
        p = result.params
        e = tm_forward_rosenblatt(
            p.kappa, p.mu, p.nu, u, copula,
            self.K, self.grid_range, self.grid_method,
            self.adaptive, self.pts_per_sigma, **self._grid_tm_kwargs())
        return e[:, 1]

    def mixture_h(self, copula, u: np.ndarray,
                  result: LatentResult, state_cache=None,
                  current_cache_key=None, next_cache_key=None) -> np.ndarray:
        """Mixture h-function for vine pseudo-obs propagation."""
        p = result.params
        return tm_forward_mixture_h(
            p.kappa, p.mu, p.nu, u, copula,
            self.K, self.grid_range, self.grid_method,
            self.adaptive, self.pts_per_sigma,
            **self._grid_tm_kwargs(),
            state_cache=state_cache,
            current_cache_key=current_cache_key,
            next_cache_key=next_cache_key)

    def objective(self, copula, u: np.ndarray,
                  alpha: np.ndarray, **kwargs) -> float:
        """Minus log-likelihood: TM integrated -logL(kappa, mu, nu)."""
        try:
            if self._uses_dispatcher():
                return auto_neg_loglik(
                    alpha[0], alpha[1], alpha[2], u, copula,
                    self._auto_config())
            return tm_loglik(
                alpha[0], alpha[1], alpha[2], u, copula,
                self.K, self.grid_range, self.grid_method,
                self.adaptive, self.pts_per_sigma, **self._tm_kwargs())
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

        r = self.model_sample_params(copula, result, n, rng=rng)
        d = u.shape[1] if u is not None and np.ndim(u) == 2 else 2
        return sample_predictive(copula, n, r, rng=rng, d=d)

    def predict(self, copula, u, result, n, rng=None, **kwargs):
        """Mixture sampling from posterior p(x_T | data).

        Uses transfer matrix forward pass to compute the posterior
        distribution of x_T, then samples r from it.
        """
        if rng is None:
            rng = np.random.default_rng()

        r_samples = self.predictive_params(
            copula, u, result, n, rng=rng, **kwargs)
        d = u.shape[1] if u is not None and np.ndim(u) == 2 else 2
        return sample_predictive(
            copula, n, r_samples, given=kwargs.get('given'), rng=rng, d=d)

    def predictive_params(self, copula, u, result, n, rng=None, **kwargs):
        """Sample predictive copula parameters for SCAR-TM."""
        if rng is None:
            rng = np.random.default_rng()

        state = self.predictive_state(copula, u, result, **kwargs)
        return self.sample_params(copula, state, n, rng=rng, **kwargs)

    def predictive_state(self, copula, u, result, **kwargs):
        """Return SCAR-TM predictive state as a grid distribution."""
        p = result.params
        if u is None:
            return PredictiveState(
                method='SCAR-TM-OU',
                horizon=str(kwargs.get('horizon', 'next')).lower(),
                kind='stationary_normal',
                metadata={
                    'mu': p.mu,
                    'sigma': np.sqrt(p.nu ** 2 / (2.0 * p.kappa)),
                },
            )

        state_cache = kwargs.get('state_cache')
        cache_key = kwargs.get('cache_key')
        cached = None
        if state_cache is not None and cache_key is not None:
            cached = state_cache.get(cache_key)

        if cached is None:
            from pyscarcopula.numerical import predictive_tm
            cached = predictive_tm.tm_state_distribution(
                p.kappa, p.mu, p.nu, u, copula,
                K=self.K,
                grid_range=self.grid_range,
                grid_method=self.grid_method,
                adaptive=self.adaptive,
                pts_per_sigma=self.pts_per_sigma,
                **self._grid_tm_kwargs(),
                horizon=kwargs.get('horizon', 'next'))
            if state_cache is not None and cache_key is not None:
                state_cache[cache_key] = cached

        z_grid, prob = cached
        return PredictiveState(
            method='SCAR-TM-OU',
            horizon=str(kwargs.get('horizon', 'next')).lower(),
            kind='grid',
            z_grid=np.asarray(z_grid, dtype=np.float64),
            prob=np.asarray(prob, dtype=np.float64),
        )

    def condition_state(self, copula, state, observation, result, **kwargs):
        """Bayes-reweight a SCAR-TM grid state by one observed pair."""
        if observation is None or state.kind != 'grid':
            return state
        u = np.asarray(observation, dtype=np.float64)
        if u.ndim != 2 or u.shape[1] != 2 or len(u) == 0:
            return state
        u = u[:1]

        z_grid = np.asarray(state.z_grid, dtype=np.float64)
        prob = np.asarray(state.prob, dtype=np.float64)
        r_grid = copula.transform(z_grid)
        u1 = np.full(len(r_grid), float(u[0, 0]), dtype=np.float64)
        u2 = np.full(len(r_grid), float(u[0, 1]), dtype=np.float64)
        log_w = np.asarray(copula.log_pdf(u1, u2, r_grid), dtype=np.float64)
        finite = np.isfinite(log_w)
        if not np.any(finite):
            return state

        weights = np.zeros_like(prob, dtype=np.float64)
        weights[finite] = (
            prob[finite] * np.exp(log_w[finite] - np.max(log_w[finite]))
        )
        total = np.sum(weights)
        if total <= 0.0:
            return state
        weights /= total
        return PredictiveState(
            method=state.method,
            horizon=state.horizon,
            kind=state.kind,
            z_grid=z_grid,
            prob=weights,
            metadata=dict(state.metadata),
        )

    def sample_params(self, copula, state, n, rng=None, **kwargs):
        if rng is None:
            rng = np.random.default_rng()
        if state.kind == 'stationary_normal':
            x_t = rng.normal(
                state.metadata['mu'],
                state.metadata['sigma'],
                n,
            )
            return copula.transform(x_t)

        from pyscarcopula.numerical.predictive_tm import sample_grid_distribution
        mode = kwargs.get('predictive_r_mode')
        if mode is None:
            z_samples = sample_grid_distribution(
                state.z_grid, state.prob, n, rng)
        else:
            z_samples = sample_grid_distribution(
                state.z_grid, state.prob, n, rng, mode=mode)
        return copula.transform(z_samples)

    def model_sample_params(self, copula, result, n, rng=None, **kwargs):
        """OU trajectory parameters for unconditional model reproduction."""
        if rng is None:
            rng = np.random.default_rng()

        p = result.params
        kappa, mu, nu = p.kappa, p.mu, p.nu
        dt = 1.0 / (n - 1) if n > 1 else 1.0
        rho_ou = np.exp(-kappa * dt)
        sigma_cond = np.sqrt(nu ** 2 / (2.0 * kappa) * (1.0 - rho_ou ** 2))

        x = np.empty(n, dtype=np.float64)
        x[0] = rng.normal(mu, nu / np.sqrt(2.0 * kappa))
        for t in range(1, n):
            x[t] = (
                mu + rho_ou * (x[t - 1] - mu)
                + sigma_cond * rng.standard_normal()
            )
        return copula.transform(x)

    def model_sample_state(self, copula, result, **kwargs):
        return None
