"""SCAR strategy with Jacobi diffusion for Kendall's tau."""

from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, minimize
from scipy.special import expit

from pyscarcopula._types import (
    DEFAULT_CONFIG,
    LatentResult,
    NumericalConfig,
    PredictiveState,
    jacobi_params,
)
from pyscarcopula.numerical.jacobi_tm import (
    _jacobi_stationary_shape,
    jacobi_forward_mixture_h,
    jacobi_forward_predictive_mean,
    jacobi_loglik,
    jacobi_matrix_forward_mixture_h,
    jacobi_matrix_forward_predictive_mean,
    jacobi_matrix_loglik,
    jacobi_matrix_neg_loglik,
    jacobi_matrix_neg_loglik_with_grad,
    jacobi_matrix_state_distribution,
    jacobi_neg_loglik,
    jacobi_state_distribution,
    jacobi_transition_matrix,
)
from pyscarcopula.numerical._transition_methods import (
    normalize_jacobi_strategy_transition_method,
)
from pyscarcopula.numerical import copula_native
from pyscarcopula.strategy._base import register_strategy
from pyscarcopula.strategy.predict_helpers import predict_from_strategy
from pyscarcopula.strategy.initial_point import (
    _explicit_initialization_diagnostics,
    _initialization_attempt,
    _initialization_diagnostics,
)


_INVALID_OBJECTIVE_THRESHOLD = 1e9
_DEFAULT_KAPPA_BOUNDS = (1e-3, 100.0)
_DEFAULT_XI_BOUNDS = (1e-3, 5.0)


def _logit(x):
    x = np.asarray(x, dtype=np.float64)
    return np.log(x / (1.0 - x))


def _raw_to_physical(raw):
    raw = np.asarray(raw, dtype=np.float64)
    clipped = np.clip(raw, -50.0, 50.0)
    return np.array([
        np.exp(clipped[0]),
        expit(clipped[1]),
        np.exp(clipped[2]),
    ], dtype=np.float64)


def _physical_to_raw(alpha, tau_eps):
    alpha = np.asarray(alpha, dtype=np.float64)
    return np.array([
        np.log(max(alpha[0], 1e-300)),
        _logit(np.clip(alpha[1], tau_eps, 1.0 - tau_eps)),
        np.log(max(alpha[2], 1e-300)),
    ], dtype=np.float64)


def _validate_positive_bounds(bounds, name):
    if bounds is None:
        return 1e-300, np.inf
    if len(bounds) != 2:
        raise ValueError(f"{name} must be a (lower, upper) pair")
    lower, upper = bounds
    lower = 1e-300 if lower is None else float(lower)
    upper = np.inf if upper is None else float(upper)
    if lower <= 0.0 or upper <= 0.0 or lower >= upper:
        raise ValueError(f"{name} must satisfy 0 < lower < upper")
    return lower, upper


def _objective_is_invalid(value):
    return (not np.isfinite(value)) or float(value) >= _INVALID_OBJECTIVE_THRESHOLD


@register_strategy('SCAR-TM-JACOBI')
class SCARJacobiStrategy:
    """TM estimation for a Jacobi-diffusion Kendall tau model.

    Parameters
    ----------
    analytical_grad : bool, default False
        Pass a model-provided Jacobian to the optimizer.  ``local_fixed``
        supplies fully analytical setup and filtering derivatives.  ``local``
        and ``spectral_matrix`` (including either backend selected by
        ``auto``) use finite differences for setup arrays followed by
        analytical filtering derivatives.  ``spectral_coeff`` does not
        support this option.
    """

    def __init__(self, config: NumericalConfig | None = None,
                 basis_order: int = 32,
                 quad_order: int | None = None,
                 transition_method: str = "auto",
                 tau_eps: float = 1e-6,
                 theta_cap: float | None = None,
                 clip_negative: bool = False,
                 negative_mass_tol: float = 1e-5,
                 gh_order: int = 5,
                 kappa_bounds: tuple[float | None, float | None] | None = _DEFAULT_KAPPA_BOUNDS,
                 xi_bounds: tuple[float | None, float | None] | None = _DEFAULT_XI_BOUNDS,
                 stationary_shape_max: float | None = 500.0,
                 analytical_grad: bool = False,
                 smart_init: bool = True,
                 **kwargs):
        self.config = config or DEFAULT_CONFIG
        basis_order = kwargs.pop('spectral_basis_order', basis_order)
        quad_order = kwargs.pop('spectral_quad_order', quad_order)
        self.basis_order = int(basis_order)
        self.quad_order = None if quad_order is None else int(quad_order)
        self.transition_method = normalize_jacobi_strategy_transition_method(
            transition_method)
        self.tau_eps = float(tau_eps)
        self.theta_cap = theta_cap
        self.clip_negative = bool(clip_negative)
        self.negative_mass_tol = float(negative_mass_tol)
        self.gh_order = int(gh_order)
        self.kappa_bounds = _validate_positive_bounds(
            kappa_bounds, "kappa_bounds")
        self.xi_bounds = _validate_positive_bounds(xi_bounds, "xi_bounds")
        self.stationary_shape_max = (
            None if stationary_shape_max is None
            else float(stationary_shape_max)
        )
        self.analytical_grad = bool(analytical_grad)
        self.smart_init = bool(smart_init)
        if not (0.0 < self.tau_eps < 0.5):
            raise ValueError("tau_eps must be in (0, 0.5)")
        if self.negative_mass_tol < 0.0:
            raise ValueError("negative_mass_tol must be non-negative")
        if self.gh_order <= 0:
            raise ValueError("gh_order must be positive")
        if (self.stationary_shape_max is not None
                and self.stationary_shape_max <= 0.0):
            raise ValueError("stationary_shape_max must be positive or None")

    def _uses_matrix_backend(self):
        return self.transition_method != 'spectral_coeff'

    def _raw_bounds(self):
        return Bounds(
            [
                np.log(self.kappa_bounds[0]),
                _logit(self.tau_eps),
                np.log(self.xi_bounds[0]),
            ],
            [
                np.log(self.kappa_bounds[1]),
                _logit(1.0 - self.tau_eps),
                np.log(self.xi_bounds[1]),
            ],
        )

    def _shape_is_supported(self, kappa, m, xi):
        shapes = _jacobi_stationary_shape(kappa, m, xi)
        if shapes is None:
            return False
        if self.stationary_shape_max is None:
            return True
        alpha, beta = shapes
        return (
            np.isfinite(alpha)
            and np.isfinite(beta)
            and alpha <= self.stationary_shape_max
            and beta <= self.stationary_shape_max
        )

    def _backend_kwargs(self):
        return {
            'basis_order': self.basis_order,
            'quad_order': self.quad_order,
            'theta_cap': self.theta_cap,
        }

    def _matrix_backend_kwargs(self):
        return {
            'basis_order': self.basis_order,
            'quad_order': self.quad_order,
            'theta_cap': self.theta_cap,
            'transition_method': self.transition_method,
            'clip_negative': self.clip_negative,
            'negative_mass_tol': self.negative_mass_tol,
            'gh_order': self.gh_order,
        }

    def _likelihood_kwargs(self):
        if self._uses_matrix_backend():
            return self._matrix_backend_kwargs()
        return self._backend_kwargs()

    def _neg_loglik(self, kappa, m, xi, u, copula):
        if not self._shape_is_supported(kappa, m, xi):
            return 1e10
        if self._uses_matrix_backend():
            return jacobi_matrix_neg_loglik(
                kappa, m, xi, u, copula, **self._matrix_backend_kwargs())
        return jacobi_neg_loglik(
            kappa, m, xi, u, copula, **self._backend_kwargs())

    def _neg_loglik_with_grad(self, kappa, m, xi, u, copula):
        if not self._shape_is_supported(kappa, m, xi):
            return 1e10, np.zeros(3, dtype=np.float64)
        if not self._uses_matrix_backend():
            return 1e10, np.zeros(3, dtype=np.float64)
        return jacobi_matrix_neg_loglik_with_grad(
            kappa, m, xi, u, copula, **self._matrix_backend_kwargs())

    def _selected_transition_backend(self, kappa, m, xi, n_obs):
        if not self._uses_matrix_backend():
            return "spectral_coeff"
        try:
            _, _, _, diagnostics = jacobi_transition_matrix(
                kappa,
                m,
                xi,
                n_obs=n_obs,
                basis_order=self.basis_order,
                quad_order=self.quad_order,
                transition_method=self.transition_method,
                clip_negative=self.clip_negative,
                negative_mass_tol=self.negative_mass_tol,
                gh_order=self.gh_order,
                return_diagnostics=True,
            )
        except Exception:
            return self.transition_method
        return str(diagnostics.get(
            "transition_method", self.transition_method))

    def _gradient_diagnostics(self, selected_backend):
        requested = self.analytical_grad
        if not requested:
            return {
                "gradient_requested": False,
                "gradient_used": False,
                "analytical_grad_requested": False,
                "analytical_grad_used": False,
                "model_score": "not_applicable",
                "optimizer_gradient": "numerical",
                "gradient_kind": "numerical",
                "setup_derivative": "not_provided",
                "filter_derivative": "not_provided_to_optimizer",
                "transition_backend_requested": self.transition_method,
                "transition_backend": selected_backend,
            }

        fully_analytical = selected_backend == "local_fixed"
        return {
            "gradient_requested": True,
            "gradient_used": True,
            "analytical_grad_requested": True,
            "analytical_grad_used": True,
            "model_score": "not_applicable",
            "optimizer_gradient": "model_provided",
            "gradient_kind": (
                "analytical" if fully_analytical else "semi_analytical"),
            "setup_derivative": (
                "analytical" if fully_analytical
                else "numerical_finite_difference"),
            "filter_derivative": "analytical",
            "transition_backend_requested": self.transition_method,
            "transition_backend": selected_backend,
        }

    def _loglik(self, kappa, m, xi, u, copula):
        if not self._shape_is_supported(kappa, m, xi):
            return -np.inf
        if self._uses_matrix_backend():
            return jacobi_matrix_loglik(
                kappa, m, xi, u, copula, **self._matrix_backend_kwargs())
        return jacobi_loglik(
            kappa, m, xi, u, copula, **self._backend_kwargs())

    def _predictive_mean(self, kappa, m, xi, u, copula):
        if not self._shape_is_supported(kappa, m, xi):
            raise ValueError("Jacobi stationary shape is outside supported range")
        if self._uses_matrix_backend():
            return jacobi_matrix_forward_predictive_mean(
                kappa, m, xi, u, copula, **self._matrix_backend_kwargs())
        return jacobi_forward_predictive_mean(
            kappa, m, xi, u, copula, **self._backend_kwargs())

    def _mixture_h(self, kappa, m, xi, u, copula):
        if not self._shape_is_supported(kappa, m, xi):
            raise ValueError("Jacobi stationary shape is outside supported range")
        if self._uses_matrix_backend():
            return jacobi_matrix_forward_mixture_h(
                kappa, m, xi, u, copula, **self._matrix_backend_kwargs())
        return jacobi_forward_mixture_h(
            kappa, m, xi, u, copula, **self._backend_kwargs())

    def _state_distribution(self, kappa, m, xi, u, copula, horizon):
        if not self._shape_is_supported(kappa, m, xi):
            raise ValueError("Jacobi stationary shape is outside supported range")
        if self._uses_matrix_backend():
            return jacobi_matrix_state_distribution(
                kappa, m, xi, u, copula,
                **self._matrix_backend_kwargs(),
                horizon=horizon,
            )
        return jacobi_state_distribution(
            kappa, m, xi, u, copula,
            **self._backend_kwargs(),
            horizon=horizon,
        )

    @staticmethod
    def _check_kendall_mapping(copula):
        try:
            if copula_native.supported(copula):
                copula_native.tau_to_param(
                    copula, np.array([0.5], dtype=np.float64))
            else:
                copula.tau_to_param(np.array([0.5], dtype=np.float64))
            return
        except NotImplementedError:
            raise
        except Exception as exc:
            raise ValueError(
                f"{type(copula).__name__} does not provide a usable "
                "tau_to_param mapping"
            ) from exc

    def _initial_point(self, copula, u):
        if not self.smart_init:
            alpha0 = np.array([1.0, 0.5, 0.2], dtype=np.float64)
            diagnostics = _initialization_diagnostics(
                'constant_default',
                'constant_default',
                alpha0,
                [_initialization_attempt(
                    'constant_default', success=True)],
            )
            return alpha0, diagnostics

        try:
            from pyscarcopula.strategy.mle import MLEStrategy
            mle = MLEStrategy(config=self.config)
            mle_result = mle.fit(copula, u)
            parameter = np.array([mle_result.copula_param])
            if copula_native.supported(copula):
                tau_hat = float(
                    copula_native.param_to_tau(copula, parameter)[0])
            else:
                tau_hat = float(np.asarray(
                    copula.param_to_tau(parameter))[0])
            m0 = float(np.clip(tau_hat, self.tau_eps, 1.0 - self.tau_eps))
            mle_attempt = _initialization_attempt(
                'static_mle_tau', success=True)
            selected_method = 'static_mle_tau'
        except Exception as exc:
            m0 = 0.5
            mle_attempt = _initialization_attempt(
                'static_mle_tau', success=False, error=exc)
            selected_method = 'm0_default'
        alpha0 = np.array([1.0, m0, 0.2], dtype=np.float64)
        attempts = [mle_attempt]
        if selected_method == 'm0_default':
            attempts.append(_initialization_attempt(
                'm0_default', success=True))
        diagnostics = _initialization_diagnostics(
            'static_mle_tau',
            selected_method,
            alpha0,
            attempts,
        )
        return alpha0, diagnostics

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
        if 'tol' in kwargs:
            raise TypeError("tol is not supported; use gtol")
        if self.analytical_grad:
            if not self._uses_matrix_backend():
                raise NotImplementedError(
                    "analytical_grad is not implemented for the "
                    "spectral_coeff Jacobi backend")

        self._check_kendall_mapping(copula)
        u = np.asarray(u, dtype=np.float64)
        if alpha0 is None:
            alpha0, initialization = self._initial_point(copula, u)
        else:
            initialization = _explicit_initialization_diagnostics(alpha0)
        alpha0 = np.asarray(alpha0, dtype=np.float64)
        raw0 = _physical_to_raw(alpha0, self.tau_eps)
        bounds = self._raw_bounds()
        raw0 = np.clip(raw0, bounds.lb, bounds.ub)

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

        def objective_raw(raw):
            alpha = _raw_to_physical(raw)
            try:
                return self._neg_loglik(
                    alpha[0], alpha[1], alpha[2], u, copula)
            except Exception as exc:
                if verbose:
                    print(f"  error at alpha={alpha}: {exc}")
                return 1e10

        def objective_raw_with_grad(raw):
            alpha = _raw_to_physical(raw)
            try:
                val, grad = self._neg_loglik_with_grad(
                    alpha[0], alpha[1], alpha[2], u, copula)
                raw_grad = grad * np.array([
                    alpha[0],
                    alpha[1] * (1.0 - alpha[1]),
                    alpha[2],
                ], dtype=np.float64)
                return val, raw_grad
            except Exception as exc:
                if verbose:
                    print(f"  error at alpha={alpha}: {exc}")
                return 1e10, np.zeros(3, dtype=np.float64)

        if self.analytical_grad:
            result = minimize(
                objective_raw_with_grad,
                raw0,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options=optimizer_options,
            )
        else:
            result = minimize(
                objective_raw,
                raw0,
                method='L-BFGS-B',
                bounds=bounds,
                options=optimizer_options,
            )

        alpha = _raw_to_physical(result.x)
        if self.analytical_grad:
            final_fun, _ = objective_raw_with_grad(result.x)
        else:
            final_fun = objective_raw(result.x)
        if _objective_is_invalid(final_fun):
            result.success = False
            result.message = (
                f"{result.message}; invalid objective value {float(final_fun):.6g}"
            )

        if verbose:
            print(f"SCAR-TM-JACOBI alpha={alpha}, logL={-final_fun:.4f}")

        selected_backend = self._selected_transition_backend(
            alpha[0], alpha[1], alpha[2], len(u))
        diagnostics = self._gradient_diagnostics(selected_backend)
        diagnostics["initialization"] = initialization

        return LatentResult(
            log_likelihood=-float(final_fun),
            method='SCAR-TM-JACOBI',
            copula_name=copula.name,
            success=bool(result.success),
            nfev=int(result.nfev),
            message=str(result.message),
            params=jacobi_params(alpha[0], alpha[1], alpha[2]),
            transition_method=self.transition_method,
            gh_order=self.gh_order if self._uses_matrix_backend() else None,
            spectral_basis_order=self.basis_order,
            spectral_quad_order=self.quad_order,
            diagnostics=diagnostics,
        )

    def log_likelihood(self, copula, u: np.ndarray,
                       result: LatentResult) -> float:
        p = result.params
        return self._loglik(p.kappa, p.m, p.xi, u, copula)

    def predictive_mean(self, copula, u: np.ndarray,
                        result: LatentResult) -> np.ndarray:
        p = result.params
        return self._predictive_mean(p.kappa, p.m, p.xi, u, copula)

    def rosenblatt_e2(self, copula, u: np.ndarray,
                      result: LatentResult) -> np.ndarray:
        return self.mixture_h(copula, u, result)

    def mixture_h(self, copula, u: np.ndarray,
                  result: LatentResult, **kwargs) -> np.ndarray:
        p = result.params
        h_mix = self._mixture_h(p.kappa, p.m, p.xi, u, copula)

        state_cache = kwargs.get('state_cache')
        current_cache_key = kwargs.get('current_cache_key')
        next_cache_key = kwargs.get('next_cache_key')
        if state_cache is not None:
            if current_cache_key is not None:
                state_cache[current_cache_key] = self._state_distribution(
                    p.kappa, p.m, p.xi, u, copula, horizon='current')
            if next_cache_key is not None:
                state_cache[next_cache_key] = self._state_distribution(
                    p.kappa, p.m, p.xi, u, copula, horizon='next')

        return h_mix

    def objective(self, copula, u: np.ndarray,
                  alpha: np.ndarray, **kwargs) -> float:
        alpha = np.asarray(alpha, dtype=np.float64)
        try:
            return self._neg_loglik(alpha[0], alpha[1], alpha[2], u, copula)
        except Exception:
            return 1e10

    def predict(self, copula, u, result, n, rng=None, **kwargs):
        return predict_from_strategy(
            self, copula, u, result, n, rng=rng, **kwargs)

    def predictive_params(self, copula, u, result, n, rng=None, **kwargs):
        if rng is None:
            rng = np.random.default_rng()
        state = self.predictive_state(copula, u, result, **kwargs)
        return self.sample_params(copula, state, n, rng=rng, **kwargs)

    def predictive_state(self, copula, u, result, **kwargs):
        horizon = str(kwargs.get('horizon', 'next')).lower()
        p = result.params
        if u is None:
            shapes = _jacobi_stationary_shape(p.kappa, p.m, p.xi)
            if shapes is None:
                raise ValueError("invalid Jacobi parameters")
            alpha, beta = shapes
            return PredictiveState(
                method='SCAR-TM-JACOBI',
                horizon=horizon,
                kind='stationary_jacobi',
                metadata={'alpha': alpha, 'beta': beta},
            )

        state_cache = kwargs.get('state_cache')
        cache_key = kwargs.get('cache_key')
        cached = None
        if state_cache is not None and cache_key is not None:
            cached = state_cache.get(cache_key)

        if cached is None:
            cached = self._state_distribution(
                p.kappa, p.m, p.xi, u, copula, horizon=horizon)
            if state_cache is not None and cache_key is not None:
                state_cache[cache_key] = cached

        tau_grid, prob = cached
        return PredictiveState(
            method='SCAR-TM-JACOBI',
            horizon=horizon,
            kind='grid',
            z_grid=tau_grid,
            prob=prob,
        )

    def condition_state(self, copula, state, observation, result, **kwargs):
        if observation is None or state.kind != 'grid':
            return state
        u = np.asarray(observation, dtype=np.float64)
        if u.ndim != 2 or u.shape[1] != 2 or len(u) == 0:
            return state

        tau_grid = np.asarray(state.z_grid, dtype=np.float64)
        prob = np.asarray(state.prob, dtype=np.float64)
        native = copula_native.supported(copula)
        if native:
            theta = copula_native.tau_to_param(copula, tau_grid)
        else:
            theta = copula.tau_to_param(tau_grid)
        if self.theta_cap is not None:
            theta = np.minimum(theta, float(self.theta_cap))
        u1 = np.full(len(theta), float(u[0, 0]), dtype=np.float64)
        u2 = np.full(len(theta), float(u[0, 1]), dtype=np.float64)
        if native:
            log_w = copula_native.log_pdf(copula, u1, u2, theta)
        else:
            log_w = np.asarray(
                copula.log_pdf(u1, u2, theta), dtype=np.float64)
        finite = np.isfinite(log_w)
        if not np.any(finite):
            return state

        weights = np.zeros_like(prob)
        weights[finite] = prob[finite] * np.exp(
            log_w[finite] - np.max(log_w[finite]))
        total = np.sum(weights)
        if total <= 0.0:
            return state
        weights /= total
        return PredictiveState(
            method=state.method,
            horizon=state.horizon,
            kind=state.kind,
            z_grid=tau_grid,
            prob=weights,
            metadata=dict(state.metadata),
        )

    def sample_params(self, copula, state, n, rng=None, **kwargs):
        if rng is None:
            rng = np.random.default_rng()
        if state.kind == 'stationary_jacobi':
            tau = rng.beta(state.metadata['alpha'], state.metadata['beta'], n)
            if copula_native.supported(copula):
                theta = copula_native.tau_to_param(copula, tau)
            else:
                theta = copula.tau_to_param(tau)
            if self.theta_cap is not None:
                theta = np.minimum(theta, float(self.theta_cap))
            return theta

        from pyscarcopula.numerical.predictive_tm import sample_grid_distribution
        mode = kwargs.get('predictive_r_mode')
        tau = sample_grid_distribution(state.z_grid, state.prob, n, rng, mode=mode)
        if copula_native.supported(copula):
            theta = copula_native.tau_to_param(copula, tau)
        else:
            theta = copula.tau_to_param(tau)
        if self.theta_cap is not None:
            theta = np.minimum(theta, float(self.theta_cap))
        return theta

    def sample(self, copula, u, result, n, rng=None, **kwargs):
        raise NotImplementedError(
            "Unconditional Jacobi model simulation is not implemented yet")

    def model_sample_params(self, copula, result, n, rng=None, **kwargs):
        raise NotImplementedError(
            "Unconditional Jacobi parameter-path simulation is not implemented yet")

    def model_sample_state(self, copula, result, **kwargs):
        return None
