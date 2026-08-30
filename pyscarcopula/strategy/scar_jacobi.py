"""SCAR strategy with Jacobi diffusion for Kendall's tau."""

from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, minimize

from pyscarcopula._native import jacobi as jacobi_native
from pyscarcopula._native import model_policy
from pyscarcopula._native import validation as native_validation
from pyscarcopula._types import (
    DEFAULT_CONFIG,
    LatentResult,
    NumericalConfig,
    PredictiveState,
    jacobi_params,
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
from pyscarcopula.strategy._base import (
    copula_dimension,
    lbfgsb_options,
    lbfgsb_overrides,
    register_strategy,
    reject_unknown_strategy_kwargs,
    validate_copula_data,
)
from pyscarcopula.strategy.predict_helpers import (
    predictive_params_from_state_with_rng,
    sample_predictive,
    strategy_predict,
)
from pyscarcopula.strategy.initial_point import (
    _explicit_initialization_diagnostics,
    _initialization_attempt,
    _initialization_diagnostics,
)


_DEFAULT_KAPPA_BOUNDS, _DEFAULT_XI_BOUNDS = (
    jacobi_native.default_parameter_bounds())
DEFAULT_JACOBI_MEMORY_BUDGET_BYTES = (
    jacobi_native.DEFAULT_JACOBI_MEMORY_BUDGET_BYTES)
DEFAULT_LAMPERTI_CHUNK_OBSERVATIONS = (
    jacobi_native.DEFAULT_LAMPERTI_CHUNK_OBSERVATIONS)
MAX_JACOBI_ORDER = jacobi_native.MAX_JACOBI_ORDER


def _raw_to_physical(raw):
    return jacobi_native.raw_to_physical(raw)


def _physical_to_raw(alpha, tau_eps):
    return jacobi_native.physical_to_raw(alpha, tau_eps)


def _validate_positive_bounds(bounds, name):
    if bounds is not None:
        if len(bounds) != 2:
            raise ValueError(f"{name} must be a (lower, upper) pair")
        bounds = tuple(
            None if value is None
            else _finite_float(value, f"{name} {label}")
            for value, label in zip(bounds, ("lower", "upper"))
        )
    return model_policy.normalize_positive_bounds(bounds, name)


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


@register_strategy('SCAR-TM-JACOBI')
class SCARJacobiStrategy:
    """TM estimation for a Jacobi-diffusion Kendall tau model.

    Parameters
    ----------
    analytical_grad : bool, default False
        Pass a model-provided Jacobian to the optimizer.  ``local_fixed``
        supplies fully analytical setup and filtering derivatives.  ``local``
        and ``spectral_matrix`` (including either backend selected by
        ``auto``) use native finite differences for setup arrays followed by
        analytical filtering derivatives.  ``spectral_coeff`` uses a complete
        physical central-difference objective inside the native evaluator.
    """

    _strict_keyword_contract = True
    _constructor_keyword_aliases = frozenset({
        "spectral_basis_order",
        "spectral_quad_order",
    })

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
                 lamperti_engine: str = "native",
                 lamperti_chunk_observations: int = (
                     DEFAULT_LAMPERTI_CHUNK_OBSERVATIONS),
                 analytical_grad: bool = False,
                 smart_init: bool = True,
                 **kwargs):
        basis_order = kwargs.pop('spectral_basis_order', basis_order)
        quad_order = kwargs.pop('spectral_quad_order', quad_order)
        reject_unknown_strategy_kwargs("SCAR-TM-JACOBI", kwargs)
        self.config = config or DEFAULT_CONFIG
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
        self.sampling_method = jacobi_native.normalize_sampling_method(
            sampling_method)
        self.lamperti_substeps = validate_positive_int(
            lamperti_substeps, "lamperti_substeps")
        self.lamperti_boundary = jacobi_native.normalize_lamperti_boundary(
            lamperti_boundary)
        self.lamperti_eps = jacobi_native.validate_lamperti_eps(lamperti_eps)
        self.lamperti_engine = jacobi_native.normalize_lamperti_engine(
            lamperti_engine)
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

    def _resolved_quad_order(self):
        return (
            jacobi_native.default_quad_order(self.basis_order)
            if self.quad_order is None else self.quad_order)

    def _resolved_memory_budget(self):
        return (
            DEFAULT_JACOBI_MEMORY_BUDGET_BYTES
            if self.memory_budget_bytes is None
            else self.memory_budget_bytes)

    def _evaluator_kwargs(self):
        return {
            'basis_order': self.basis_order,
            'quad_order': self.quad_order,
            'theta_cap': self.theta_cap,
            'transition_method': self.transition_method,
            'storage': self.transition_storage,
            'correction': self.stationarity_correction,
            'clip_negative': self.clip_negative,
            'negative_mass_tol': self.negative_mass_tol,
            'gh_order': self.gh_order,
            'memory_budget_bytes': self._resolved_memory_budget(),
            'stationary_shape_max': self.stationary_shape_max,
        }

    def _prepared_evaluator(self, u, copula):
        return jacobi_native.PreparedScarJacobiEvaluator(
            u, copula, **self._evaluator_kwargs())

    def _neg_loglik(self, kappa, m, xi, u, copula, *, evaluator=None):
        domain_result = jacobi_native.optimizer_domain_evaluation(
            kappa, m, xi, self.stationary_shape_max,
            self.config.fail_value)
        if domain_result is not None:
            return domain_result[0]
        evaluator = evaluator or self._prepared_evaluator(u, copula)
        try:
            value = float(evaluator.neg_loglik(kappa, m, xi))
        except FloatingPointError as error:
            return model_policy.optimizer_failure_objective(error)
        if not np.isfinite(value):
            raise FloatingPointError(
                "native Jacobi objective returned a non-finite value")
        return value

    def _neg_loglik_with_grad(
            self, kappa, m, xi, u, copula, *, evaluator=None):
        domain_result = jacobi_native.optimizer_domain_evaluation(
            kappa, m, xi, self.stationary_shape_max,
            self.config.fail_value)
        if domain_result is not None:
            return domain_result
        evaluator = evaluator or self._prepared_evaluator(u, copula)
        try:
            return evaluator.neg_loglik_with_grad(kappa, m, xi)
        except FloatingPointError as error:
            return model_policy.optimizer_numerical_failure_evaluation(
                error,
                [kappa, m, xi],
                [kappa, m, xi],
                self.config.fail_value,
                directional_gradient=False,
            )

    def _selected_transition_backend(self, kappa, m, xi, n_obs):
        if not self._uses_matrix_backend():
            return "spectral_coeff"
        if self._uses_sparse_backend():
            suffix = (
                f"_{self.stationarity_correction}"
                if self.stationarity_correction != "none" else "")
            return f"{self.transition_method}_sparse{suffix}"
        try:
            _, _, _, _, _, diagnostics = jacobi_native.dense_transition(
                kappa,
                m,
                xi,
                n_obs=n_obs,
                basis_order=self.basis_order,
                quad_order=self._resolved_quad_order(),
                method=self.transition_method,
                clip_negative=self.clip_negative,
                negative_mass_tol=self.negative_mass_tol,
                gh_order=self.gh_order,
                memory_budget_bytes=self._resolved_memory_budget(),
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
        coefficient_fd = selected_backend == "spectral_coeff"
        return {
            "gradient_requested": True,
            "gradient_used": True,
            "analytical_grad_requested": True,
            "analytical_grad_used": True,
            "model_score": "not_applicable",
            "optimizer_gradient": "model_provided",
            "gradient_kind": (
                "analytical" if fully_analytical
                else (
                    "native_finite_difference"
                    if coefficient_fd else "semi_analytical")),
            "setup_derivative": (
                "analytical" if fully_analytical
                else "numerical_finite_difference"),
            "filter_derivative": (
                "numerical_finite_difference"
                if coefficient_fd else "analytical"),
            "transition_backend_requested": self.transition_method,
            "transition_backend": selected_backend,
        }

    def _loglik(self, kappa, m, xi, u, copula, *, evaluator=None):
        evaluator = evaluator or self._prepared_evaluator(u, copula)
        return evaluator.loglik(kappa, m, xi)

    def _predictive_mean(
            self, kappa, m, xi, u, copula, *, evaluator=None):
        evaluator = evaluator or self._prepared_evaluator(u, copula)
        return evaluator.predictive_mean(kappa, m, xi)

    def _mixture_h(
            self, kappa, m, xi, u, copula, *, evaluator=None):
        evaluator = evaluator or self._prepared_evaluator(u, copula)
        return evaluator.mixture_h(kappa, m, xi)

    def _mixture_h_pair(
            self, kappa, m, xi, u, copula, *, evaluator=None):
        evaluator = evaluator or self._prepared_evaluator(u, copula)
        return evaluator.mixture_h_pair(kappa, m, xi)

    def _state_distribution(
            self, kappa, m, xi, u, copula, horizon, *, evaluator=None):
        evaluator = evaluator or self._prepared_evaluator(u, copula)
        return evaluator.state_distribution(kappa, m, xi, horizon=horizon)

    @staticmethod
    def _check_kendall_mapping(copula):
        jacobi_native.validate_copula_mapping(copula)

    def _initial_point(self, copula, u, initial_mle_result=None):
        if not self.smart_init:
            alpha0 = jacobi_native.initial_point(None, self.tau_eps)
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
            tau_hat = float(
                jacobi_native.parameter_to_tau(copula, parameter)[0])
            alpha0 = jacobi_native.initial_point(tau_hat, self.tau_eps)
            mle_attempt = _initialization_attempt(
                'static_mle_tau', success=True)
            selected_method = 'static_mle_tau'
        except Exception as exc:
            alpha0 = jacobi_native.initial_point(None, self.tau_eps)
            mle_attempt = _initialization_attempt(
                'static_mle_tau', success=False, error=exc)
            selected_method = 'm0_default'
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

    def _adaptive_order_report(self, kappa, m, xi, n_obs, quad_orders, *,
                               require_pass):
        *_, report = jacobi_native.select_sparse_order(
            kappa,
            m,
            xi,
            n_obs=n_obs,
            quad_orders=quad_orders,
            basis_order=self.basis_order,
            gh_order=self.gh_order,
            max_full_horizon_tv=self.adaptive_max_full_horizon_tv,
            max_relative_variance_error=(
                self.adaptive_max_relative_variance_error),
            max_conditional_mean_rmse=(
                self.adaptive_max_conditional_mean_rmse),
            max_lag_one_correlation_error=(
                self.adaptive_max_lag_one_correlation_error),
            memory_budget_bytes=self._resolved_memory_budget(),
            require_pass=require_pass,
        )
        report["thresholds"] = {
            "full_horizon_stationary_tv": (
                self.adaptive_max_full_horizon_tv),
            "relative_variance_error": (
                self.adaptive_max_relative_variance_error),
            "conditional_mean_rmse": (
                self.adaptive_max_conditional_mean_rmse),
            "absolute_lag_one_correlation_error": (
                self.adaptive_max_lag_one_correlation_error),
        }
        return report

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
        reject_unknown_strategy_kwargs("SCAR-TM-JACOBI", kwargs)
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
            adaptive_initial = self._adaptive_order_report(
                alpha0[0],
                alpha0[1],
                alpha0[2],
                len(u),
                self.adaptive_quad_orders,
                require_pass=self.adaptive_require_pass,
            )
            self.quad_order = int(
                adaptive_initial["selected_quad_order"])
        jacobi_native.estimate_workspace(
            quad_order=self._resolved_quad_order(),
            basis_order=self.basis_order,
            n_obs=len(u),
            matrix=self._uses_matrix_backend(),
            gradient=self.analytical_grad,
            gh_order=self.gh_order,
            memory_budget_bytes=self._resolved_memory_budget(),
        )
        evaluator = self._prepared_evaluator(u, copula)
        raw0 = _physical_to_raw(alpha0, self.tau_eps)
        bounds = self._raw_bounds()
        raw0 = model_policy.project_optimizer_point(
            raw0, bounds.lb, bounds.ub)

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
            return self._neg_loglik(
                alpha[0], alpha[1], alpha[2], u, copula,
                evaluator=evaluator)

        def objective_raw_with_grad(raw):
            alpha = _raw_to_physical(raw)
            val, grad = self._neg_loglik_with_grad(
                alpha[0], alpha[1], alpha[2], u, copula,
                evaluator=evaluator)
            return val, jacobi_native.raw_gradient(alpha, grad)

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
                not native_validation.objective_is_invalid(gradient_final_fun)
                and not native_validation.objective_is_invalid(final_fun)
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

        if native_validation.objective_is_invalid(final_fun):
            result.success = False
            result.message = (
                f"{result.message}; invalid objective value {float(final_fun):.6g}"
            )

        if verbose:
            print(f"SCAR-TM-JACOBI alpha={alpha}, logL={-final_fun:.4f}")

        try:
            native_diagnostics = evaluator.filter(
                alpha[0], alpha[1], alpha[2])["diagnostics"]
            selected_backend = str(native_diagnostics["transition_method"])
            if self._uses_sparse_backend():
                selected_backend += "_sparse"
                if self.stationarity_correction != "none":
                    selected_backend += f"_{self.stationarity_correction}"
        except Exception:
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
            adaptive_final = self._adaptive_order_report(
                alpha[0],
                alpha[1],
                alpha[2],
                len(u),
                (self.quad_order,),
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
        evaluator = self._prepared_evaluator(u, copula)
        h_mix = self._mixture_h(
            p.kappa, p.m, p.xi, u, copula, evaluator=evaluator)

        state_cache = kwargs.get('state_cache')
        current_cache_key = kwargs.get('current_cache_key')
        next_cache_key = kwargs.get('next_cache_key')
        if state_cache is not None:
            if current_cache_key is not None:
                state_cache[current_cache_key] = self._state_distribution(
                    p.kappa, p.m, p.xi, u, copula, horizon='current',
                    evaluator=evaluator)
            if next_cache_key is not None:
                state_cache[next_cache_key] = self._state_distribution(
                    p.kappa, p.m, p.xi, u, copula, horizon='next',
                    evaluator=evaluator)

        return h_mix

    def mixture_h_pair(self, copula, u: np.ndarray,
                       result: LatentResult, **kwargs):
        """Both h-directions from one Jacobi posterior pass."""
        p = result.params
        evaluator = self._prepared_evaluator(u, copula)
        h_pair = self._mixture_h_pair(
            p.kappa, p.m, p.xi, u, copula, evaluator=evaluator)

        state_cache = kwargs.get('state_cache')
        current_cache_key = kwargs.get('current_cache_key')
        next_cache_key = kwargs.get('next_cache_key')
        if state_cache is not None:
            if current_cache_key is not None:
                state_cache[current_cache_key] = self._state_distribution(
                    p.kappa, p.m, p.xi, u, copula, horizon='current',
                    evaluator=evaluator)
            if next_cache_key is not None:
                state_cache[next_cache_key] = self._state_distribution(
                    p.kappa, p.m, p.xi, u, copula, horizon='next',
                    evaluator=evaluator)

        return h_pair

    def objective(self, copula, u: np.ndarray,
                  alpha: np.ndarray, **kwargs) -> float:
        alpha = np.asarray(alpha, dtype=np.float64)
        return self._neg_loglik(alpha[0], alpha[1], alpha[2], u, copula)

    predict = strategy_predict
    predictive_params = predictive_params_from_state_with_rng

    def predictive_state(self, copula, u, result, **kwargs):
        horizon = str(kwargs.get('horizon', 'next')).lower()
        p = result.params
        if u is None:
            shapes = jacobi_native.stationary_shape(p.kappa, p.m, p.xi)
            if shapes is None:
                raise ValueError("invalid Jacobi parameters")
            alpha, beta = shapes
            return PredictiveState(
                method='SCAR-TM-JACOBI',
                horizon=horizon,
                kind='stationary_jacobi',
                metadata={
                    'alpha': alpha,
                    'beta': beta,
                    'kappa': p.kappa,
                    'm': p.m,
                    'xi': p.xi,
                },
            )

        state_cache = kwargs.get('state_cache')
        cache_key = kwargs.get('cache_key')
        cached = None
        if state_cache is not None and cache_key is not None:
            cached = state_cache.get(cache_key)

        if cached is None:
            evaluator = self._prepared_evaluator(u, copula)
            cached = self._state_distribution(
                p.kappa, p.m, p.xi, u, copula, horizon=horizon,
                evaluator=evaluator)
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
        evaluator = self._prepared_evaluator(u[:1], copula)
        conditioned_tau, weights = evaluator.condition_state(
            tau_grid,
            prob,
            u[0],
            horizon=state.horizon,
        )
        return PredictiveState(
            method=state.method,
            horizon=state.horizon,
            kind=state.kind,
            z_grid=conditioned_tau,
            prob=weights,
            metadata=dict(state.metadata),
        )

    def sample_params(self, copula, state, n, rng=None, **kwargs):
        if rng is None:
            rng = np.random.default_rng()
        if state.kind == 'stationary_jacobi':
            tau = jacobi_native.sample_stationary_fixed_draws(
                state.metadata['kappa'],
                state.metadata['m'],
                state.metadata['xi'],
                rng.uniform(0.0, 1.0, size=n),
            )
            return jacobi_native.tau_to_parameter(
                copula, tau, theta_cap=self.theta_cap)

        mode = kwargs.get('predictive_r_mode')
        mode = 'grid' if mode is None else str(mode).lower()
        if mode not in {'grid', 'histogram'}:
            raise ValueError(
                "predictive_r_mode must be 'grid' or 'histogram'")
        selection_draws = rng.uniform(0.0, 1.0, size=n)
        jitter_draws = (
            rng.uniform(0.0, 1.0, size=n)
            if mode == 'histogram' and len(state.z_grid) > 1
            else np.empty(0, dtype=np.float64)
        )
        _, parameters, _ = jacobi_native.sample_state_distribution_fixed_draws(
            copula,
            state.z_grid,
            state.prob,
            selection_draws,
            jitter_draws,
            mode=mode,
            theta_cap=self.theta_cap,
        )
        return parameters

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
            tau, sampling_diagnostics = jacobi_native.sample_lamperti_trajectory(
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
            tau, sampling_diagnostics = jacobi_native.sample_grid_trajectory(
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
        return jacobi_native.tau_to_parameter(
            copula, tau, theta_cap=self.theta_cap)

    def model_sample_state(self, copula, result, **kwargs):
        return None
