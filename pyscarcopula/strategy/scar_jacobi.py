"""SCAR strategy with Jacobi diffusion for Kendall's tau."""

from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, minimize

from pyscarcopula._native import jacobi as jacobi_native
from pyscarcopula._types import (
    DEFAULT_CONFIG,
    LatentResult,
    NumericalConfig,
    PredictiveState,
    jacobi_params,
)
from pyscarcopula.numerical.jacobi_tm import (
    DEFAULT_JACOBI_MEMORY_BUDGET_BYTES,
    MAX_JACOBI_ORDER,
    _validate_jacobi_workspace,
    default_quad_order,
    _jacobi_stationary_shape,
    jacobi_forward_mixture_h,
    jacobi_forward_mixture_h_pair,
    jacobi_forward_predictive_mean,
    jacobi_loglik,
    jacobi_matrix_forward_mixture_h,
    jacobi_matrix_forward_mixture_h_pair,
    jacobi_matrix_forward_predictive_mean,
    jacobi_matrix_loglik,
    jacobi_matrix_neg_loglik,
    jacobi_matrix_neg_loglik_with_grad,
    jacobi_matrix_state_distribution,
    jacobi_neg_loglik,
    jacobi_state_distribution,
    jacobi_transition_matrix,
    sample_jacobi_grid_trajectory,
)
from pyscarcopula.numerical.jacobi_sampling import (
    DEFAULT_LAMPERTI_CHUNK_OBSERVATIONS,
    normalize_jacobi_sampling_method,
    normalize_lamperti_boundary,
    normalize_lamperti_engine,
    sample_jacobi_lamperti_trajectory,
    validate_lamperti_eps,
)
from pyscarcopula.numerical.jacobi_sparse import (
    jacobi_sparse_matrix_forward_mixture_h,
    jacobi_sparse_matrix_forward_mixture_h_pair,
    jacobi_sparse_matrix_forward_predictive_mean,
    jacobi_sparse_matrix_loglik,
    jacobi_sparse_matrix_neg_loglik_with_grad,
    jacobi_sparse_matrix_state_distribution,
    select_sparse_jacobi_order,
)
from pyscarcopula.numerical._arrays import (
    validate_float64_allocation,
    validate_positive_int,
)
from pyscarcopula.numerical._transition_methods import (
    normalize_jacobi_stationarity_correction,
    normalize_jacobi_strategy_transition_method,
    normalize_jacobi_transition_storage,
)
from pyscarcopula.numerical import copula_native
from pyscarcopula.strategy._base import (
    copula_dimension,
    lbfgsb_options,
    lbfgsb_overrides,
    register_strategy,
    reject_legacy_tol,
    validate_copula_data,
)
from pyscarcopula.strategy.predict_helpers import (
    predict_from_strategy,
    sample_predictive,
)
from pyscarcopula.strategy.initial_point import (
    _explicit_initialization_diagnostics,
    _initialization_attempt,
    _initialization_diagnostics,
)


_INVALID_OBJECTIVE_THRESHOLD = 1e9
_DEFAULT_KAPPA_BOUNDS = (1e-3, 100.0)
_DEFAULT_XI_BOUNDS = (1e-3, 5.0)


def _raw_to_physical(raw):
    return jacobi_native.raw_to_physical(raw)


def _physical_to_raw(alpha, tau_eps):
    return jacobi_native.physical_to_raw(alpha, tau_eps)


def _validate_positive_bounds(bounds, name):
    if bounds is None:
        return 1e-300, np.inf
    if len(bounds) != 2:
        raise ValueError(f"{name} must be a (lower, upper) pair")
    lower, upper = bounds
    lower = 1e-300 if lower is None else _finite_float(
        lower, f"{name} lower")
    upper = np.inf if upper is None else _finite_float(
        upper, f"{name} upper")
    if lower <= 0.0 or upper <= 0.0 or lower >= upper:
        raise ValueError(f"{name} must satisfy 0 < lower < upper")
    return lower, upper


def _finite_float(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a finite real number") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_optional_memory_budget(value):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)):
        raise TypeError("memory_budget_bytes must be an integer or None")
    value = int(value)
    if value < 0:
        raise ValueError("memory_budget_bytes must be non-negative")
    return value


def _validate_alpha0(alpha0):
    try:
        alpha = np.asarray(alpha0, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("alpha0 must contain three real values") from exc
    if alpha.shape != (3,):
        raise ValueError(f"alpha0 must have shape (3,), got {alpha.shape}")
    if np.any(~np.isfinite(alpha)):
        raise ValueError("alpha0 must contain only finite values")
    if alpha[0] <= 0.0 or not (0.0 < alpha[1] < 1.0) or alpha[2] <= 0.0:
        raise ValueError(
            "alpha0 must satisfy kappa > 0, 0 < m < 1, and xi > 0")
    return alpha


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
                 transition_storage: str = "dense",
                 stationarity_correction: str = "none",
                 adaptive_quad_order: bool = False,
                 adaptive_quad_orders=(48, 80, 128, 192, 384, 768),
                 adaptive_max_full_horizon_tv: float = 0.02,
                 adaptive_max_relative_variance_error: float = 0.10,
                 adaptive_max_conditional_mean_rmse: float = 1e-3,
                 adaptive_max_lag_one_correlation_error: float = 1e-2,
                 adaptive_require_pass: bool = False,
                 tau_eps: float = 1e-6,
                 theta_cap: float | None = None,
                 clip_negative: bool = False,
                 negative_mass_tol: float = 1e-5,
                 gh_order: int = 5,
                 kappa_bounds: tuple[float | None, float | None] | None = _DEFAULT_KAPPA_BOUNDS,
                 xi_bounds: tuple[float | None, float | None] | None = _DEFAULT_XI_BOUNDS,
                 stationary_shape_max: float | None = 500.0,
                 memory_budget_bytes: int | None = (
                     DEFAULT_JACOBI_MEMORY_BUDGET_BYTES),
                 sampling_method: str = "tm_grid",
                 lamperti_substeps: int = 8,
                 lamperti_boundary: str = "reflect",
                 lamperti_eps: float = 1e-10,
                 lamperti_engine: str = "numba",
                 lamperti_chunk_observations: int = (
                     DEFAULT_LAMPERTI_CHUNK_OBSERVATIONS),
                 analytical_grad: bool = False,
                 smart_init: bool = True,
                 **kwargs):
        self.config = config or DEFAULT_CONFIG
        basis_order = kwargs.pop('spectral_basis_order', basis_order)
        quad_order = kwargs.pop('spectral_quad_order', quad_order)
        self.basis_order = validate_positive_int(basis_order, "basis_order")
        self.quad_order = (
            None if quad_order is None
            else validate_positive_int(quad_order, "quad_order"))
        if self.basis_order > MAX_JACOBI_ORDER:
            raise ValueError(f"basis_order must be <= {MAX_JACOBI_ORDER}")
        if self.quad_order is not None and self.quad_order > MAX_JACOBI_ORDER:
            raise ValueError(f"quad_order must be <= {MAX_JACOBI_ORDER}")
        self.transition_method = normalize_jacobi_strategy_transition_method(
            transition_method)
        self.transition_storage = normalize_jacobi_transition_storage(
            transition_storage)
        self.stationarity_correction = (
            normalize_jacobi_stationarity_correction(
                stationarity_correction))
        self.adaptive_quad_order = bool(adaptive_quad_order)
        if isinstance(adaptive_quad_orders, (str, bytes)):
            raise TypeError(
                "adaptive_quad_orders must be an iterable of integers")
        self.adaptive_quad_orders = tuple(
            validate_positive_int(order, "adaptive_quad_orders")
            for order in adaptive_quad_orders)
        if not self.adaptive_quad_orders:
            raise ValueError("adaptive_quad_orders must not be empty")
        if any(
                right <= left
                for left, right in zip(
                    self.adaptive_quad_orders,
                    self.adaptive_quad_orders[1:])):
            raise ValueError(
                "adaptive_quad_orders must be strictly increasing")
        if self.adaptive_quad_orders[-1] > MAX_JACOBI_ORDER:
            raise ValueError(
                f"adaptive_quad_orders must be <= {MAX_JACOBI_ORDER}")
        self.adaptive_max_full_horizon_tv = _finite_float(
            adaptive_max_full_horizon_tv,
            "adaptive_max_full_horizon_tv")
        self.adaptive_max_relative_variance_error = _finite_float(
            adaptive_max_relative_variance_error,
            "adaptive_max_relative_variance_error")
        self.adaptive_max_conditional_mean_rmse = _finite_float(
            adaptive_max_conditional_mean_rmse,
            "adaptive_max_conditional_mean_rmse")
        self.adaptive_max_lag_one_correlation_error = _finite_float(
            adaptive_max_lag_one_correlation_error,
            "adaptive_max_lag_one_correlation_error")
        self.adaptive_require_pass = bool(adaptive_require_pass)
        self.tau_eps = _finite_float(tau_eps, "tau_eps")
        self.theta_cap = (
            None if theta_cap is None
            else _finite_float(theta_cap, "theta_cap"))
        self.clip_negative = bool(clip_negative)
        self.negative_mass_tol = _finite_float(
            negative_mass_tol, "negative_mass_tol")
        self.gh_order = validate_positive_int(gh_order, "gh_order")
        if self.gh_order > MAX_JACOBI_ORDER:
            raise ValueError(f"gh_order must be <= {MAX_JACOBI_ORDER}")
        self.kappa_bounds = _validate_positive_bounds(
            kappa_bounds, "kappa_bounds")
        self.xi_bounds = _validate_positive_bounds(xi_bounds, "xi_bounds")
        self.stationary_shape_max = (
            None if stationary_shape_max is None
            else _finite_float(stationary_shape_max, "stationary_shape_max")
        )
        self.memory_budget_bytes = _validate_optional_memory_budget(
            memory_budget_bytes)
        self.sampling_method = normalize_jacobi_sampling_method(
            sampling_method)
        self.lamperti_substeps = validate_positive_int(
            lamperti_substeps, "lamperti_substeps")
        self.lamperti_boundary = normalize_lamperti_boundary(
            lamperti_boundary)
        self.lamperti_eps = validate_lamperti_eps(lamperti_eps)
        self.lamperti_engine = normalize_lamperti_engine(lamperti_engine)
        self.lamperti_chunk_observations = validate_positive_int(
            lamperti_chunk_observations,
            "lamperti_chunk_observations",
        )
        self.analytical_grad = bool(analytical_grad)
        self.smart_init = bool(smart_init)
        if (
                self.transition_storage == "sparse"
                and self.transition_method not in {"local", "local_fixed"}):
            raise ValueError(
                "transition_storage='sparse' currently requires "
                "transition_method='local' or 'local_fixed'")
        if (
                self.stationarity_correction != "none"
                and (
                    self.transition_storage != "sparse"
                    or self.transition_method != "local")):
            raise ValueError(
                "stationarity_correction requires sparse "
                "transition_method='local'")
        if (
                self.transition_storage == "sparse"
                and self.analytical_grad
                and self.transition_method != "local_fixed"):
            raise ValueError(
                "sparse analytical_grad requires "
                "transition_method='local_fixed'")
        if self.adaptive_quad_order and (
                self.transition_storage != "sparse"
                or self.transition_method != "local"
                or self.stationarity_correction != "none"):
            raise ValueError(
                "adaptive_quad_order requires an uncorrected sparse "
                "transition_method='local'")
        if self.adaptive_quad_order and self.quad_order is not None:
            raise ValueError(
                "adaptive_quad_order cannot be combined with quad_order")
        for name in (
                "adaptive_max_full_horizon_tv",
                "adaptive_max_relative_variance_error",
                "adaptive_max_conditional_mean_rmse",
                "adaptive_max_lag_one_correlation_error"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if not (0.0 < self.tau_eps < 0.5):
            raise ValueError("tau_eps must be in (0, 0.5)")
        if self.negative_mass_tol < 0.0:
            raise ValueError("negative_mass_tol must be non-negative")
        if self.theta_cap is not None and self.theta_cap <= 0.0:
            raise ValueError("theta_cap must be positive or None")
        if (self.stationary_shape_max is not None
                and self.stationary_shape_max <= 0.0):
            raise ValueError("stationary_shape_max must be positive or None")

    def _uses_matrix_backend(self):
        return self.transition_method != 'spectral_coeff'

    def _uses_sparse_backend(self):
        return self.transition_storage == "sparse"

    def _raw_bounds(self):
        lower, upper = jacobi_native.raw_bounds(
            self.kappa_bounds, self.xi_bounds, self.tau_eps)
        return Bounds(lower, upper)

    def _shape_is_supported(self, kappa, m, xi):
        return jacobi_native.shape_is_supported(
            kappa, m, xi, self.stationary_shape_max)

    def _backend_kwargs(self):
        return {
            'basis_order': self.basis_order,
            'quad_order': self.quad_order,
            'theta_cap': self.theta_cap,
            'memory_budget_bytes': self.memory_budget_bytes,
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
            'memory_budget_bytes': self.memory_budget_bytes,
        }

    def _sparse_backend_kwargs(self):
        return {
            'basis_order': self.basis_order,
            'quad_order': self.quad_order,
            'theta_cap': self.theta_cap,
            'transition_method': self.transition_method,
            'gh_order': self.gh_order,
            'correction': self.stationarity_correction,
            'memory_budget_bytes': self.memory_budget_bytes,
        }

    def _neg_loglik(self, kappa, m, xi, u, copula):
        if not self._shape_is_supported(kappa, m, xi):
            return 1e10
        if self._uses_sparse_backend():
            value = jacobi_sparse_matrix_loglik(
                kappa, m, xi, u, copula, **self._sparse_backend_kwargs())
            return -value if np.isfinite(value) else 1e10
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
        if self._uses_sparse_backend():
            return jacobi_sparse_matrix_neg_loglik_with_grad(
                kappa, m, xi, u, copula, **self._sparse_backend_kwargs())
        return jacobi_matrix_neg_loglik_with_grad(
            kappa, m, xi, u, copula, **self._matrix_backend_kwargs())

    def _selected_transition_backend(self, kappa, m, xi, n_obs):
        if not self._uses_matrix_backend():
            return "spectral_coeff"
        if self._uses_sparse_backend():
            suffix = (
                f"_{self.stationarity_correction}"
                if self.stationarity_correction != "none" else "")
            return f"{self.transition_method}_sparse{suffix}"
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
                memory_budget_bytes=self.memory_budget_bytes,
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

        fully_analytical = selected_backend in {
            "local_fixed", "local_fixed_sparse"}
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
        if self._uses_sparse_backend():
            return jacobi_sparse_matrix_loglik(
                kappa, m, xi, u, copula, **self._sparse_backend_kwargs())
        if self._uses_matrix_backend():
            return jacobi_matrix_loglik(
                kappa, m, xi, u, copula, **self._matrix_backend_kwargs())
        return jacobi_loglik(
            kappa, m, xi, u, copula, **self._backend_kwargs())

    def _predictive_mean(self, kappa, m, xi, u, copula):
        if not self._shape_is_supported(kappa, m, xi):
            raise ValueError("Jacobi stationary shape is outside supported range")
        if self._uses_sparse_backend():
            return jacobi_sparse_matrix_forward_predictive_mean(
                kappa, m, xi, u, copula, **self._sparse_backend_kwargs())
        if self._uses_matrix_backend():
            return jacobi_matrix_forward_predictive_mean(
                kappa, m, xi, u, copula, **self._matrix_backend_kwargs())
        return jacobi_forward_predictive_mean(
            kappa, m, xi, u, copula, **self._backend_kwargs())

    def _mixture_h(self, kappa, m, xi, u, copula):
        if not self._shape_is_supported(kappa, m, xi):
            raise ValueError("Jacobi stationary shape is outside supported range")
        if self._uses_sparse_backend():
            return jacobi_sparse_matrix_forward_mixture_h(
                kappa, m, xi, u, copula, **self._sparse_backend_kwargs())
        if self._uses_matrix_backend():
            return jacobi_matrix_forward_mixture_h(
                kappa, m, xi, u, copula, **self._matrix_backend_kwargs())
        return jacobi_forward_mixture_h(
            kappa, m, xi, u, copula, **self._backend_kwargs())

    def _mixture_h_pair(self, kappa, m, xi, u, copula):
        if not self._shape_is_supported(kappa, m, xi):
            raise ValueError("Jacobi stationary shape is outside supported range")
        if self._uses_sparse_backend():
            return jacobi_sparse_matrix_forward_mixture_h_pair(
                kappa, m, xi, u, copula, **self._sparse_backend_kwargs())
        if self._uses_matrix_backend():
            return jacobi_matrix_forward_mixture_h_pair(
                kappa, m, xi, u, copula, **self._matrix_backend_kwargs())
        return jacobi_forward_mixture_h_pair(
            kappa, m, xi, u, copula, **self._backend_kwargs())

    def _state_distribution(self, kappa, m, xi, u, copula, horizon):
        if not self._shape_is_supported(kappa, m, xi):
            raise ValueError("Jacobi stationary shape is outside supported range")
        if self._uses_sparse_backend():
            return jacobi_sparse_matrix_state_distribution(
                kappa, m, xi, u, copula,
                **self._sparse_backend_kwargs(),
                horizon=horizon,
            )
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

    def _initial_point(self, copula, u, initial_mle_result=None):
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
            mle_result = initial_mle_result
            if mle_result is None:
                from pyscarcopula.strategy.mle import MLEStrategy
                mle_result = MLEStrategy(config=self.config).fit(copula, u)
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
        diagnostics['mle_source'] = (
            'selection_result' if initial_mle_result is not None
            else 'strategy_fit')
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
            initial_mle_result=None,
            **kwargs) -> LatentResult:
        reject_legacy_tol(kwargs)
        if self.analytical_grad:
            if not self._uses_matrix_backend():
                raise NotImplementedError(
                    "analytical_grad is not implemented for the "
                    "spectral_coeff Jacobi backend")

        self._check_kendall_mapping(copula)
        u = validate_copula_data(copula, u)
        if alpha0 is None:
            alpha0, initialization = self._initial_point(
                copula, u, initial_mle_result)
        else:
            initialization = _explicit_initialization_diagnostics(alpha0)
        alpha0 = _validate_alpha0(alpha0)
        adaptive_initial = None
        if self.adaptive_quad_order:
            _, _, _, adaptive_initial = select_sparse_jacobi_order(
                alpha0[0],
                alpha0[1],
                alpha0[2],
                n_obs=len(u),
                quad_orders=self.adaptive_quad_orders,
                basis_order=self.basis_order,
                gh_order=self.gh_order,
                max_full_horizon_tv=(
                    self.adaptive_max_full_horizon_tv),
                max_relative_variance_error=(
                    self.adaptive_max_relative_variance_error),
                max_conditional_mean_rmse=(
                    self.adaptive_max_conditional_mean_rmse),
                max_lag_one_correlation_error=(
                    self.adaptive_max_lag_one_correlation_error),
                memory_budget_bytes=self.memory_budget_bytes,
                require_pass=self.adaptive_require_pass,
            )
            self.quad_order = int(
                adaptive_initial["selected_quad_order"])
        resolved_quad_order = (
            default_quad_order(self.basis_order)
            if self.quad_order is None else self.quad_order)
        _validate_jacobi_workspace(
            quad_order=resolved_quad_order,
            basis_order=self.basis_order,
            n_obs=len(u),
            matrix=self._uses_matrix_backend(),
            gradient=self.analytical_grad,
            memory_budget_bytes=self.memory_budget_bytes,
        )
        raw0 = _physical_to_raw(alpha0, self.tau_eps)
        bounds = self._raw_bounds()
        raw0 = np.clip(raw0, bounds.lb, bounds.ub)

        optimizer_options = lbfgsb_options(
            self.config.scar_optimizer,
            **lbfgsb_overrides(
                gtol=gtol,
                ftol=ftol,
                maxfun=maxfun,
                maxiter=maxiter,
                maxls=maxls,
                eps=eps,
                maxcor=maxcor,
                finite_diff_rel_step=finite_diff_rel_step,
            ),
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
        gradient_final_fun = None
        if self.analytical_grad:
            gradient_final_fun, _ = objective_raw_with_grad(result.x)
        final_fun = objective_raw(result.x)

        final_objective_consistent = True
        if self.analytical_grad:
            final_objective_consistent = (
                not _objective_is_invalid(gradient_final_fun)
                and not _objective_is_invalid(final_fun)
                and np.isclose(
                    gradient_final_fun,
                    final_fun,
                    rtol=1e-8,
                    atol=1e-10,
                )
            )
            if not final_objective_consistent:
                result.success = False
                result.message = (
                    f"{result.message}; inconsistent gradient objective: "
                    f"gradient={float(gradient_final_fun):.6g}, "
                    f"plain={float(final_fun):.6g}"
                )

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
        diagnostics["final_objective_value"] = float(final_fun)
        diagnostics["final_gradient_objective_value"] = (
            None
            if gradient_final_fun is None
            else float(gradient_final_fun)
        )
        diagnostics["final_objective_consistent"] = bool(
            final_objective_consistent)
        diagnostics["transition_storage"] = self.transition_storage
        diagnostics["stationarity_correction"] = (
            self.stationarity_correction)
        if adaptive_initial is not None:
            diagnostics["adaptive_quad_order_initial"] = adaptive_initial
            _, _, _, adaptive_final = select_sparse_jacobi_order(
                alpha[0],
                alpha[1],
                alpha[2],
                n_obs=len(u),
                quad_orders=(self.quad_order,),
                basis_order=self.basis_order,
                gh_order=self.gh_order,
                max_full_horizon_tv=(
                    self.adaptive_max_full_horizon_tv),
                max_relative_variance_error=(
                    self.adaptive_max_relative_variance_error),
                max_conditional_mean_rmse=(
                    self.adaptive_max_conditional_mean_rmse),
                max_lag_one_correlation_error=(
                    self.adaptive_max_lag_one_correlation_error),
                memory_budget_bytes=self.memory_budget_bytes,
                require_pass=False,
            )
            diagnostics["adaptive_quad_order_final"] = adaptive_final

        return LatentResult(
            log_likelihood=-float(final_fun),
            method='SCAR-TM-JACOBI',
            copula_name=copula.name,
            success=bool(result.success),
            nfev=int(result.nfev),
            message=str(result.message),
            params=jacobi_params(alpha[0], alpha[1], alpha[2]),
            transition_method=self.transition_method,
            transition_storage=self.transition_storage,
            stationarity_correction=self.stationarity_correction,
            gh_order=self.gh_order if self._uses_matrix_backend() else None,
            spectral_basis_order=self.basis_order,
            spectral_quad_order=self.quad_order,
            tau_eps=self.tau_eps,
            theta_cap=self.theta_cap,
            clip_negative=self.clip_negative,
            negative_mass_tol=self.negative_mass_tol,
            stationary_shape_max=self.stationary_shape_max,
            sampling_method=self.sampling_method,
            lamperti_substeps=self.lamperti_substeps,
            lamperti_boundary=self.lamperti_boundary,
            lamperti_eps=self.lamperti_eps,
            lamperti_engine=self.lamperti_engine,
            lamperti_chunk_observations=(
                self.lamperti_chunk_observations),
            memory_budget_bytes=self.memory_budget_bytes,
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

    def mixture_h_pair(self, copula, u: np.ndarray,
                       result: LatentResult, **kwargs):
        """Both h-directions from one Jacobi posterior pass."""
        p = result.params
        h_pair = self._mixture_h_pair(p.kappa, p.m, p.xi, u, copula)

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

        return h_pair

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
        """Sample the fitted discrete Jacobi Markov model unconditionally."""
        if isinstance(n, (bool, np.bool_)) or not isinstance(
                n, (int, np.integer)):
            raise TypeError("n must be a non-negative integer")
        n = int(n)
        if n < 0:
            raise ValueError("n must be non-negative")
        d = copula_dimension(copula, u)
        if d is None:
            raise ValueError("copula dimension is unknown")
        validate_float64_allocation(
            (n, int(d) + 1),
            name="Jacobi sample output and parameter path",
            memory_budget_bytes=self.memory_budget_bytes,
        )
        if n == 0:
            return np.empty((0, int(d)), dtype=np.float64)
        if rng is None:
            rng = np.random.default_rng()
        theta = self.model_sample_params(
            copula, result, n, rng=rng, **kwargs)
        return sample_predictive(
            copula,
            n,
            theta,
            given=kwargs.get("given"),
            rng=rng,
            d=d,
        )

    def model_sample_params(self, copula, result, n, rng=None, **kwargs):
        """Return an unconditional Jacobi copula-parameter trajectory."""
        diagnostics_out = kwargs.get("sampling_diagnostics")
        if (
                diagnostics_out is not None
                and not hasattr(diagnostics_out, "update")):
            raise TypeError(
                "sampling_diagnostics must be a mutable mapping")
        p = result.params
        if self.sampling_method == "lamperti_euler":
            tau, sampling_diagnostics = sample_jacobi_lamperti_trajectory(
                p.kappa,
                p.m,
                p.xi,
                n,
                rng=rng,
                substeps=self.lamperti_substeps,
                boundary=self.lamperti_boundary,
                eps=self.lamperti_eps,
                engine=self.lamperti_engine,
                chunk_observations=self.lamperti_chunk_observations,
                memory_budget_bytes=self.memory_budget_bytes,
                return_diagnostics=True,
            )
        else:
            tau, sampling_diagnostics = sample_jacobi_grid_trajectory(
                p.kappa,
                p.m,
                p.xi,
                n,
                rng=rng,
                basis_order=self.basis_order,
                quad_order=self.quad_order,
                transition_method=self.transition_method,
                clip_negative=self.clip_negative,
                negative_mass_tol=self.negative_mass_tol,
                gh_order=self.gh_order,
                transition_storage=self.transition_storage,
                stationarity_correction=self.stationarity_correction,
                memory_budget_bytes=self.memory_budget_bytes,
                return_diagnostics=True,
            )
            sampling_diagnostics = dict(sampling_diagnostics)
            sampling_diagnostics["sampling_method"] = "tm_grid"
        if diagnostics_out is not None:
            diagnostics_out.update(sampling_diagnostics)
        if tau.size == 0:
            return np.empty(0, dtype=np.float64)
        if copula_native.supported(copula):
            theta = copula_native.tau_to_param(copula, tau)
        else:
            theta = np.asarray(copula.tau_to_param(tau), dtype=np.float64)
        if self.theta_cap is not None:
            theta = np.minimum(theta, float(self.theta_cap))
        if np.any(~np.isfinite(theta)):
            raise FloatingPointError(
                "tau_to_param produced non-finite sampling parameters")
        return np.asarray(theta, dtype=np.float64)

    def model_sample_state(self, copula, result, **kwargs):
        return None
