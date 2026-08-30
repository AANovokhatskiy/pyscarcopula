"""Native SCAR-TM-OU estimation strategy.

Python owns optimizer orchestration and result construction. The compiled
evaluator owns likelihood, gradients, forward filtering, and state outputs.
"""

import numpy as np
from scipy.optimize import minimize, Bounds

from pyscarcopula._types import (
    LatentResult, NumericalConfig, DEFAULT_CONFIG,
    ou_params,
    PredictiveState,
)
from pyscarcopula.strategy._base import (
    copula_dimension,
    is_multivariate_copula,
    lbfgsb_options,
    lbfgsb_overrides,
    register_strategy,
    reject_unknown_strategy_kwargs,
)
from pyscarcopula.numerical._scar_ou_config import (
    AutoTMConfig,
    validate_cpp_config,
)
from pyscarcopula.numerical._arrays import as_pseudo_observation_array
from pyscarcopula.numerical._transition_methods import (
    normalize_ou_transition_method,
)
from pyscarcopula.strategy.predict_helpers import (
    predictive_params_from_state_with_rng,
    sample_predictive,
    strategy_predict,
)
from pyscarcopula.strategy.initial_point import (
    resolve_ou_initial_point,
    smart_initial_point,
)
from pyscarcopula._native import (
    model_policy,
    scar_ou as _cpp_scar_ou,
    validation as native_validation,
)
from pyscarcopula.numerical.ou_kernels import sample_ou_trajectory
from pyscarcopula.copula.multivariate.corr_param import (
    _corr_gradient_to_raw_params,
    _shrinkage_raw_corr_direction,
)


_OU_PHYSICAL_LOWER_VALUES, _OU_PHYSICAL_UPPER_VALUES = (
    model_policy.latent_bounds("ou"))
_OU_PHYSICAL_LOWER = np.asarray(
    _OU_PHYSICAL_LOWER_VALUES, dtype=np.float64)
_OU_PHYSICAL_UPPER = np.asarray(
    _OU_PHYSICAL_UPPER_VALUES, dtype=np.float64)
(
    _,
    (
        _OU_BOUNDED_LOG_STATIONARY_RESULT_LOWER_VALUES,
        _OU_BOUNDED_LOG_STATIONARY_RESULT_UPPER_VALUES,
    ),
) = model_policy.ou_log_stationary_optimizer_bounds()
_OU_BOUNDED_LOG_STATIONARY_RESULT_LOWER = np.asarray(
    _OU_BOUNDED_LOG_STATIONARY_RESULT_LOWER_VALUES, dtype=np.float64)

_DIAGNOSTIC_COUNTERS = (
    "objective_evaluations",
    "cpp_evaluations",
    "spectral_evaluations",
    "spectral_failures",
    "matrix_evaluations",
    "matrix_failures",
    "matrix_capped",
    "matrix_fallback_unknown",
    "local_evaluations",
    "fallback_spectral_to_matrix",
    "fallback_matrix_to_local",
)

def _normalize_spectral_basis_order(value):
    if isinstance(value, str):
        order = value.lower()
        if order == "auto":
            return "auto"
        try:
            value = int(order)
        except ValueError as exc:
            raise ValueError(
                "spectral_basis_order must be a positive integer or 'auto'"
            ) from exc
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)):
        raise ValueError(
            "spectral_basis_order must be a positive integer or 'auto'")
    order = int(value)
    if order <= 0:
        raise ValueError("spectral_basis_order must be positive")
    return order


def _uses_log_stationary_scale(copula, override=None):
    """Whether SCAR optimization uses (log kappa, mu, log sigma_x)."""
    if override is not None:
        return bool(override)
    return bool(getattr(
        copula, "_scar_log_stationary_scale_optimization", False))


def _uses_bounded_log_stationary_scale(copula, override=None):
    """Whether the explicit bivariate log-scale bounds are active."""
    model_default = bool(getattr(
        copula, "_scar_log_stationary_scale_optimization", False))
    return bool(override) and not model_default


def _normalize_stationary_scale_bounds(value):
    """Validate optional positive bounds for stationary OU scale."""
    if value is None:
        return None
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise TypeError(
            "stationary_scale_bounds must be a (lower, upper) pair or None")
    try:
        return model_policy.normalize_optional_positive_bounds(
            value, "stationary_scale_bounds")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "stationary_scale_bounds values must be positive, finite, and "
            "ordered") from exc


def _resolved_stationary_scale_bounds(copula, override, explicit_bounds):
    """Resolve explicit, model-specific, or bivariate opt-in scale bounds."""
    if explicit_bounds is not None:
        return explicit_bounds
    model_bounds = getattr(copula, "_scar_stationary_scale_bounds", None)
    if model_bounds is not None:
        return _normalize_stationary_scale_bounds(model_bounds)
    if _uses_bounded_log_stationary_scale(copula, override):
        return model_policy.stationary_scale_bounds()[0], None
    return None, None


def _log_stationary_optimizer_bounds(copula, override, explicit_bounds):
    scale_lower, scale_upper = _resolved_stationary_scale_bounds(
        copula, override, explicit_bounds)
    (lower_values, upper_values), _ = (
        model_policy.ou_log_stationary_optimizer_bounds(
            (scale_lower, scale_upper)))
    lower = np.asarray(lower_values, dtype=np.float64)
    upper = np.asarray(upper_values, dtype=np.float64)
    return lower, upper, (scale_lower, scale_upper)


def _project_log_stationary_initial_point(values, lower, upper):
    """Project the OU block onto the log-kappa/log-sigma box bounds."""
    return _cpp_scar_ou.project_optimizer_block(values, lower, upper)


def _ou_to_log_stationary(alpha):
    """Map physical (kappa, mu, nu) to optimizer coordinates."""
    return _cpp_scar_ou.to_log_stationary(alpha)


def _ou_from_log_stationary(values):
    """Map optimizer coordinates to physical (kappa, mu, nu)."""
    return _cpp_scar_ou.from_log_stationary(values)


def _ou_grad_to_log_stationary(physical, gradient):
    """Apply the chain rule for (log kappa, mu, log sigma_x)."""
    return _cpp_scar_ou.gradient_to_log_stationary(physical, gradient)


def _ou_grad_from_log_stationary(physical, gradient):
    """Map an optimizer-coordinate gradient back to physical OU units."""
    return _cpp_scar_ou.gradient_from_log_stationary(physical, gradient)


def _resolve_initial_point(
        copula, u, config, smart_init, verbose, alpha0,
        initial_mle_result=None):
    return resolve_ou_initial_point(
        copula,
        u,
        config,
        smart_init,
        verbose,
        alpha0,
        smart_initial_point_func=smart_initial_point,
        initial_mle_result=initial_mle_result,
    )


def _new_backend_diagnostics() -> dict:
    diagnostics = {name: 0 for name in _DIAGNOSTIC_COUNTERS}
    diagnostics["selected_engine"] = "cpp"
    return diagnostics


def _record_backend_diagnostics(diagnostics: dict, info: dict,
                                engine: str) -> None:
    diagnostics["objective_evaluations"] += 1
    diagnostics[f"{engine}_evaluations"] += 1

    backend = info.get("backend")
    chain = list(info.get("fallback_chain") or [])
    attempts = []
    attempts.extend(item for item in chain if item in {"spectral", "matrix"})
    if backend in {"spectral", "matrix", "local"}:
        attempts.append(backend)
    if not attempts and info.get("transition_method") in {"matrix", "local"}:
        attempts.append(info["transition_method"])

    for item in attempts:
        key = f"{item}_evaluations"
        if key in diagnostics:
            diagnostics[key] += 1

    if "spectral" in chain:
        diagnostics["spectral_failures"] += 1
        diagnostics["fallback_spectral_to_matrix"] += 1
    if "matrix" in chain:
        diagnostics["fallback_matrix_to_local"] += 1
        reason = info.get("matrix_fallback_reason")
        if reason == "capped":
            diagnostics["matrix_capped"] += 1
        elif reason == "failed":
            diagnostics["matrix_failures"] += 1
        else:
            diagnostics["matrix_fallback_unknown"] += 1

    diagnostics["last_engine"] = engine
    diagnostics["last_backend"] = backend
    diagnostics["last_transition_method"] = info.get("transition_method")
    diagnostics["last_kappa_dt"] = info.get("kappa_dt")
    diagnostics["last_n_obs"] = info.get("n_obs")
    basis_order = info.get("basis_order")
    if basis_order is not None:
        try:
            basis_order_int = int(basis_order)
        except (TypeError, ValueError):
            basis_order_int = None
        if basis_order_int is not None:
            diagnostics[f"basis_order_{basis_order_int}_evaluations"] = (
                diagnostics.get(f"basis_order_{basis_order_int}_evaluations", 0)
                + 1
            )
            diagnostics["last_spectral_basis_order"] = basis_order_int
    if chain:
        diagnostics["last_fallback_chain"] = tuple(chain)


class _PreparedScarOuFitCache:
    """Prepared native SCAR-OU objectives scoped to one optimizer loop."""

    def __init__(self, u, copula, diagnostics):
        self.u = u
        self.copula = copula
        self.diagnostics = diagnostics
        self.cache = {}
        self.disabled = False
        diagnostics.setdefault("prepared_native_evaluator", False)
        diagnostics.setdefault("prepared_native_evaluator_count", 0)
        diagnostics.setdefault("prepared_native_fallback", None)

    def disable(self, reason):
        self.disabled = True
        self.cache.clear()
        self.diagnostics["prepared_native_evaluator"] = False
        self.diagnostics["prepared_native_fallback"] = reason

    def prepared_for(self, auto_config):
        if self.disabled:
            return None
        try:
            prepared = self.cache.get(auto_config)
            if prepared is None:
                prepared = _cpp_scar_ou.prepare_objective(
                    self.u, self.copula, auto_config)
                self.cache[auto_config] = prepared
                self.diagnostics["prepared_native_evaluator"] = True
                self.diagnostics["prepared_native_evaluator_count"] = (
                    len(self.cache))
            return prepared
        except AttributeError:
            self.disable("missing_api")
            return None
        except _cpp_scar_ou.NativeUnsupported:
            self.disable("unsupported")
            return None

    def _call(self, auto_config, prepared_call, fallback_call):
        prepared = self.prepared_for(auto_config)
        if prepared is not None:
            try:
                prepared.update_copula(self.copula)
                return prepared_call(prepared)
            except AttributeError:
                self.disable("missing_method")
            except _cpp_scar_ou.NativeUnsupported:
                self.disable("unsupported_method")
        return fallback_call()

    def neg_loglik_info(self, kappa, mu, nu, auto_config):
        return self._call(
            auto_config,
            lambda prepared: prepared.neg_loglik_info(kappa, mu, nu),
            lambda: _cpp_scar_ou.neg_loglik_info(
                kappa, mu, nu, self.u, self.copula, auto_config),
        )

    def neg_loglik_with_grad_info(self, kappa, mu, nu, auto_config):
        return self._call(
            auto_config,
            lambda prepared: prepared.neg_loglik_with_grad_info(
                kappa, mu, nu),
            lambda: _cpp_scar_ou.neg_loglik_with_grad_info(
                kappa, mu, nu, self.u, self.copula, auto_config),
        )

    def neg_loglik_with_grad_and_corr_info(
            self, kappa, mu, nu, auto_config):
        return self._call(
            auto_config,
            lambda prepared: prepared.neg_loglik_with_grad_and_corr_info(
                kappa, mu, nu),
            lambda: _cpp_scar_ou.neg_loglik_with_grad_and_corr_info(
                kappa, mu, nu, self.u, self.copula, auto_config),
        )

    def neg_loglik_with_grad_and_corr_directional_info(
            self, kappa, mu, nu, direction, auto_config):
        return self._call(
            auto_config,
            lambda prepared: (
                prepared.neg_loglik_with_grad_and_corr_directional_info(
                    kappa, mu, nu, direction)),
            lambda: (
                _cpp_scar_ou.neg_loglik_with_grad_and_corr_directional_info(
                    kappa, mu, nu, self.u, self.copula, direction,
                    auto_config)),
        )


_POSTERIOR_WORKSPACE_KEY = object()
_POSTERIOR_WORKSPACE_MISSING = object()
_POSTERIOR_WORKSPACE_UNSUPPORTED = object()


class _PreparedScarOuPosteriorCache:
    """Prepared native posterior evaluators scoped to one caller workflow."""

    def __init__(self):
        self.cache = {}
        self.disabled = False

    def disable(self):
        self.disabled = True
        self.cache.clear()

    def prepared_for(self, u, copula, auto_config):
        if self.disabled:
            return None
        key = (id(u), id(copula), auto_config)
        prepared = self.cache.get(key, _POSTERIOR_WORKSPACE_MISSING)
        if prepared is _POSTERIOR_WORKSPACE_UNSUPPORTED:
            return None
        if prepared is _POSTERIOR_WORKSPACE_MISSING:
            try:
                prepared = _cpp_scar_ou.prepare_objective(
                    u, copula, auto_config)
            except AttributeError:
                self.disable()
                return None
            except _cpp_scar_ou.NativeUnsupported:
                self.cache[key] = _POSTERIOR_WORKSPACE_UNSUPPORTED
                return None
            self.cache[key] = prepared
        try:
            prepared.update_copula(copula)
            return prepared
        except AttributeError:
            self.disable()
            return None
        except _cpp_scar_ou.NativeUnsupported:
            self.cache[key] = _POSTERIOR_WORKSPACE_UNSUPPORTED
            return None

    def _call(self, u, copula, auto_config, prepared_call, fallback_call):
        prepared = self.prepared_for(u, copula, auto_config)
        if prepared is not None:
            try:
                return prepared_call(prepared)
            except AttributeError:
                self.disable()
            except _cpp_scar_ou.NativeUnsupported:
                self.cache[(id(u), id(copula), auto_config)] = (
                    _POSTERIOR_WORKSPACE_UNSUPPORTED)
        return fallback_call()

    def predictive_mean(self, kappa, mu, nu, u, copula, auto_config):
        return self._call(
            u,
            copula,
            auto_config,
            lambda prepared: prepared.predictive_mean(kappa, mu, nu),
            lambda: _cpp_scar_ou.predictive_mean(
                kappa, mu, nu, u, copula, auto_config),
        )

    def mixture_h(self, kappa, mu, nu, u, copula, auto_config):
        return self._call(
            u,
            copula,
            auto_config,
            lambda prepared: prepared.mixture_h(kappa, mu, nu),
            lambda: _cpp_scar_ou.mixture_h(
                kappa, mu, nu, u, copula, auto_config),
        )

    def mixture_h_pair(self, kappa, mu, nu, u, copula, auto_config):
        return self._call(
            u,
            copula,
            auto_config,
            lambda prepared: prepared.mixture_h_pair(kappa, mu, nu),
            lambda: _cpp_scar_ou.mixture_h_pair(
                kappa, mu, nu, u, copula, auto_config),
        )

    def state_distribution(
            self, kappa, mu, nu, u, copula, auto_config,
            horizon: str = "current"):
        return self._call(
            u,
            copula,
            auto_config,
            lambda prepared: prepared.state_distribution(
                kappa, mu, nu, horizon=horizon),
            lambda: _cpp_scar_ou.state_distribution(
                kappa, mu, nu, u, copula, auto_config, horizon=horizon),
        )


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


def _append_failure_message(message, reasons):
    if not reasons:
        return str(message)
    return f"{message}; final validation failed: {'; '.join(reasons)}"


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
        spectral numerical fallback through matrix/local grid paths.  'matrix',
        'local', and 'spectral' force the corresponding likelihood backend.
    max_K : int or None
        Optional cap for adaptive TM grid size.  Defaults to 1000 in the
        strategy to prevent pathological fit-time grid blowups on long series.
    r_gh : float
        Locality threshold for auto transition selection.
    gh_order : int
        Gauss-Hermite order for local GH transition.
    auto_small_kdt : float
        Threshold for selecting the local transition in auto mode.
    spectral_basis_order : int or {'auto'}
        Number of Hermite basis functions in the spectral likelihood.  The
        default ``'auto'`` policy selects 128, 96, 64, or 32 from the current
        optimizer evaluation's ``kappa / (T - 1)``.
    spectral_quad_order : int or None
        Gauss-Hermite quadrature order for spectral multiplication.
    analytical_grad : bool
        Use analytical gradient (default True).
        Reduces nfev by ~3-4x. Parameters are auto-rescaled.
    smart_init : bool
        Compute initial point via analytical heuristic (default True).
        StochasticStudentCopula keeps the public ``(kappa, mu, nu)``
        parameters but optimizes internally in
        ``(log(kappa), mu, log(nu / sqrt(2*kappa)))`` coordinates.
    log_stationary_scale_optimization : bool or None
        Select the optimizer coordinates for the OU block. ``True`` uses
        ``(log(kappa), mu, log(nu / sqrt(2*kappa)))`` with lower bounds of
        ``0.001`` on ``kappa`` and the stationary scale for bivariate
        models. ``False`` uses the scaled ``(kappa, mu, nu)`` coordinates.
        ``None`` (default) uses the copula model preference; currently only
        StochasticStudentCopula opts in.
    stationary_scale_bounds : (float or None, float or None) or None
        Optional lower and upper bounds for ``sigma_x`` in log-stationary
        coordinates. ``None`` uses the model policy. StochasticStudentCopula
        and the independent bivariate log-coordinate policy each default to
        ``(0.001, 10000.0)``.
    final_validation_abs_per_obs : float
        Absolute cross-backend objective tolerance per observation.
    final_validation_rel_tol : float
        Relative cross-backend objective tolerance.
    final_gradient_tolerance : float or None
        Maximum projected-gradient norm for a successful final fit. When
        omitted, the tolerance is derived from the optimizer ``gtol``.
    final_growth_limit : float
        Maximum allowed OU parameter growth relative to initialization.
    final_rho_tolerance : float
        Distance from zero or one below which the one-step OU correlation is
        treated as numerically degenerate.
    strict_gradient_policy : bool
        Enforce the final projected-gradient tolerance (default False).
    """

    _strict_keyword_contract = True
    _constructor_keyword_aliases = frozenset({"backend"})

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
                 auto_small_kdt: float = 1e-2,
                 spectral_basis_order: int | str = "auto",
                 spectral_quad_order: int | None = None,
                 analytical_grad: bool = True,
                 smart_init: bool = True,
                 log_stationary_scale_optimization: bool | None = None,
                 stationary_scale_bounds: tuple[
                     float | None, float | None] | None = None,
                 final_validation_abs_per_obs: float = 5e-5,
                 final_validation_rel_tol: float = 1e-5,
                 final_gradient_tolerance: float | None = None,
                 final_growth_limit: float = 1e8,
                 final_rho_tolerance: float = 1e-15,
                 strict_gradient_policy: bool = False,
                 **kwargs):
        if "backend" in kwargs:
            raise TypeError(
                "SCAR-TM-OU backend selection was removed; native execution "
                "is always used")
        reject_unknown_strategy_kwargs("SCAR-TM-OU", kwargs)
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
        self.spectral_basis_order = _normalize_spectral_basis_order(
            spectral_basis_order)
        self.spectral_quad_order = spectral_quad_order
        self.analytical_grad = analytical_grad
        self.smart_init = smart_init
        if (
                log_stationary_scale_optimization is not None
                and not isinstance(
                    log_stationary_scale_optimization, (bool, np.bool_))):
            raise TypeError(
                "log_stationary_scale_optimization must be bool or None")
        self.log_stationary_scale_optimization = (
            None if log_stationary_scale_optimization is None
            else bool(log_stationary_scale_optimization)
        )
        self.stationary_scale_bounds = _normalize_stationary_scale_bounds(
            stationary_scale_bounds)
        self.final_validation_abs_per_obs = float(
            final_validation_abs_per_obs)
        self.final_validation_rel_tol = float(final_validation_rel_tol)
        self.strict_gradient_policy = bool(strict_gradient_policy)
        self.final_gradient_tolerance = (
            None if final_gradient_tolerance is None
            else float(final_gradient_tolerance))
        self.final_growth_limit = float(final_growth_limit)
        self.final_rho_tolerance = float(final_rho_tolerance)
        validation_options = {
            "final_validation_abs_per_obs":
                self.final_validation_abs_per_obs,
            "final_validation_rel_tol": self.final_validation_rel_tol,
            "final_growth_limit": self.final_growth_limit,
            "final_rho_tolerance": self.final_rho_tolerance,
        }
        if self.final_gradient_tolerance is not None:
            validation_options[
                "final_gradient_tolerance"] = self.final_gradient_tolerance
        for name, value in validation_options.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a positive finite number")
        validate_cpp_config(
            AutoTMConfig(
                    transition_method=self.transition_method,
                    small_kdt=self.auto_small_kdt,
                    basis_order=(
                        32 if self.spectral_basis_order == "auto"
                        else self.spectral_basis_order
                    ),
                    quad_order=self.spectral_quad_order,
                    K=self.K,
                    grid_range=self.grid_range,
                    grid_method=self.grid_method,
                    adaptive=self.adaptive,
                    pts_per_sigma=self.pts_per_sigma,
                    max_K=self.max_K,
                    gh_order=self.gh_order,
                    r_gh=self.r_gh,
                    n_threads=self.config.n_threads,
            )
        )

    def _optimizer_config(self, copula, *, log_stationary):
        if log_stationary:
            config_name = getattr(
                copula, "_scar_log_optimizer_config", None)
            if config_name is not None:
                return getattr(self.config, config_name)
        config_name = getattr(copula, "_scar_optimizer_config", None)
        if config_name is None:
            return self.config.scar_optimizer
        return getattr(self.config, config_name)

    def _grid_transition_method(self):
        if self.transition_method == 'spectral':
            return 'auto'
        return self.transition_method

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

    def _kappa_dt(self, kappa: float, n_obs: int) -> float:
        return model_policy.ou_kappa_dt(kappa, n_obs)

    def _adaptive_spectral_basis_order(self, kappa: float, n_obs: int) -> int:
        return model_policy.ou_adaptive_spectral_basis_order(kappa, n_obs)

    def _spectral_basis_order_for(self, kappa: float | None = None,
                                  n_obs: int | None = None) -> int:
        if self.spectral_basis_order != "auto":
            return int(self.spectral_basis_order)
        if kappa is None or n_obs is None:
            raise ValueError(
                "auto spectral_basis_order requires kappa and n_obs")
        return self._adaptive_spectral_basis_order(kappa, n_obs)

    def _auto_config(self, transition_method: str | None = None,
                     *, kappa: float | None = None,
                     n_obs: int | None = None):
        return AutoTMConfig(
            transition_method=transition_method or self.transition_method,
            small_kdt=self.auto_small_kdt,
            basis_order=self._spectral_basis_order_for(kappa, n_obs),
            quad_order=self.spectral_quad_order,
            K=self.K,
            grid_range=self.grid_range,
            grid_method=self.grid_method,
            adaptive=self.adaptive,
            pts_per_sigma=self.pts_per_sigma,
            max_K=self.max_K,
            gh_order=self.gh_order,
            r_gh=self.r_gh,
            n_threads=self.config.n_threads,
        )

    def _uses_cpp(self, copula):
        _cpp_scar_ou.ensure_supported(copula)
        _cpp_scar_ou.require_available()
        return True

    def _posterior_workspace_or_none(self, posterior_cache):
        if posterior_cache is None:
            return None
        if isinstance(posterior_cache, _PreparedScarOuPosteriorCache):
            return posterior_cache
        workspace = posterior_cache.get(_POSTERIOR_WORKSPACE_KEY)
        if workspace is None:
            workspace = _PreparedScarOuPosteriorCache()
            posterior_cache[_POSTERIOR_WORKSPACE_KEY] = workspace
        return workspace

    def _prepared_or_stateless_posterior(
            self, copula, u: np.ndarray, cfg,
            prepared_call, stateless_call, posterior_cache=None):
        workspace = self._posterior_workspace_or_none(posterior_cache)
        if workspace is not None:
            return workspace._call(
                u, copula, cfg, prepared_call, stateless_call)
        try:
            prepared = _cpp_scar_ou.prepare_objective(u, copula, cfg)
            prepared.update_copula(copula)
            return prepared_call(prepared)
        except AttributeError:
            pass
        except _cpp_scar_ou.NativeUnsupported:
            pass
        return stateless_call()

    def _validate_final_fit(
            self, result, final_params, initial_params, lower, upper,
            selected_evaluator, validation_evaluator, selected_engine,
            validation_engine, n_obs, optimizer_options,
            correlation_validator=None):
        """Re-evaluate and validate a candidate optimizer solution."""
        final_params = np.asarray(final_params, dtype=np.float64).reshape(-1)
        initial_params = np.asarray(initial_params, dtype=np.float64).reshape(-1)
        lower = np.asarray(lower, dtype=np.float64).reshape(-1)
        upper = np.asarray(upper, dtype=np.float64).reshape(-1)
        diagnostics = {
            "final_validation_passed": False,
            "final_validation_reasons": (),
            "final_selected_engine": selected_engine,
            "final_validation_engine": validation_engine,
        }

        selected_value = float("nan")
        selected_grad = np.asarray((), dtype=np.float64)
        selected_evaluation_succeeded = True
        selected_error = ""
        if native_validation.valid_ou_final_parameters(final_params):
            try:
                selected_value, selected_grad = selected_evaluator(final_params)
                selected_value = float(selected_value)
                selected_grad = np.asarray(
                    selected_grad, dtype=np.float64).reshape(-1)
            except Exception as exc:
                selected_evaluation_succeeded = False
                selected_error = str(exc)

        validation = native_validation.validate_final_fit(
            final_params,
            initial_params,
            lower,
            upper,
            optimizer_value=result.fun,
            selected_value=selected_value,
            selected_gradient=selected_grad,
            selected_evaluation_succeeded=selected_evaluation_succeeded,
            selected_engine=selected_engine,
            selected_error=selected_error,
            n_obs=n_obs,
            strict_gradient_policy=self.strict_gradient_policy,
            explicit_gradient_tolerance=self.final_gradient_tolerance,
            optimizer_gtol=optimizer_options.get("gtol", 1e-3),
            rho_tolerance=self.final_rho_tolerance,
            growth_limit=self.final_growth_limit,
        )
        reasons = list(validation["reasons"])

        diagnostics["final_selected_backend_value"] = selected_value
        diagnostics["final_optimizer_value"] = float(result.fun)
        diagnostics["final_optimizer_abs_tolerance"] = validation[
            "optimizer_abs_tolerance"]
        diagnostics["final_optimizer_rel_tolerance"] = validation[
            "optimizer_rel_tolerance"]
        if validation["has_projected_gradient"]:
            diagnostics["final_projected_gradient_norm"] = validation[
                "projected_gradient_norm"]
            diagnostics[
                "final_projected_gradient_tolerance"] = validation[
                    "projected_gradient_tolerance"]
        diagnostics["final_boundary_flags"] = tuple(
            validation["boundary_flags"])

        if validation["has_ou_diagnostics"]:
            diagnostics.update({
                "final_kappa_dt": validation["kappa_dt"],
                "final_rho": validation["rho"],
                "final_stationary_std": validation["stationary_std"],
                "final_conditional_std": validation["conditional_std"],
            })
        if validation["has_parameter_growth"]:
            diagnostics["final_parameter_growth"] = tuple(
                validation["parameter_growth"])
            diagnostics["final_parameter_growth_limit"] = (
                self.final_growth_limit)

        backend_enabled = validation_evaluator is not None and not reasons
        backend_succeeded = True
        backend_error = ""
        backend_value = float("nan")
        if backend_enabled:
            try:
                backend_value = float(validation_evaluator(final_params))
            except Exception as exc:
                backend_succeeded = False
                backend_error = str(exc)
        backend = native_validation.validate_backend_agreement(
            enabled=backend_enabled,
            evaluation_succeeded=backend_succeeded,
            engine=validation_engine,
            error=backend_error,
            validation_value=backend_value,
            selected_value=selected_value,
            n_obs=n_obs,
            abs_per_observation=self.final_validation_abs_per_obs,
            relative_tolerance=self.final_validation_rel_tol,
        )
        reasons.extend(backend["reasons"])
        diagnostics.update({
            "final_validation_backend_value": backend["value"],
            "final_backend_value_difference": backend["difference"],
            "final_backend_value_tolerance": backend["tolerance"],
            "final_validation_abs_per_obs":
                self.final_validation_abs_per_obs,
            "final_validation_rel_tolerance":
                self.final_validation_rel_tol,
            "final_rho_tolerance": self.final_rho_tolerance,
        })

        if correlation_validator is not None:
            try:
                correlation_reasons = list(correlation_validator())
            except Exception as exc:
                correlation_reasons = [
                    f"final correlation validation failed: {exc}"]
            reasons.extend(correlation_reasons)

        diagnostics["final_validation_reasons"] = tuple(reasons)
        diagnostics["final_validation_passed"] = not reasons
        result.fun = selected_value
        result.jac = selected_grad
        if reasons:
            result.success = False
            result.message = _append_failure_message(result.message, reasons)
        return diagnostics

    def _fit_joint_static(self, copula, u, alpha0, optimizer_options,
                          verbose, initial_mle_result=None):
        """Fit OU and Python-parameterized static correlation parameters."""

        n_corr = int(copula._corr_num_params())
        copula._ensure_corr_initialized(u)
        corr0 = np.asarray(
            copula._initial_corr_params(u), dtype=np.float64).reshape(-1)

        alpha0, initialization = _resolve_initial_point(
            copula,
            u,
            self.config,
            self.smart_init,
            verbose,
            alpha0,
            initial_mle_result,
        )
        if initialization["selected_method"] != "user_provided":
            fitted_corr = np.asarray(
                copula._pack_corr_params(), dtype=np.float64).reshape(-1)
            if fitted_corr.size == n_corr:
                corr0 = fitted_corr

        self._uses_cpp(copula)
        selected_engine = "cpp"

        alpha0 = np.asarray(alpha0, dtype=np.float64).reshape(-1)
        if alpha0.size == 3:
            joint0 = np.concatenate([alpha0, corr0])
        elif alpha0.size == 3 + n_corr:
            joint0 = alpha0.copy()
        else:
            raise ValueError(
                f"alpha0 must contain 3 OU parameters or {3 + n_corr} "
                f"joint parameters, got {alpha0.size}")
        if not np.all(np.isfinite(joint0)):
            raise ValueError("alpha0 must contain only finite values")
        initialization = dict(initialization)
        initialization["alpha0"] = [
            float(value) for value in joint0]

        log_stationary = _uses_log_stationary_scale(
            copula, self.log_stationary_scale_optimization)
        bounded_log_stationary = _uses_bounded_log_stationary_scale(
            copula, self.log_stationary_scale_optimization)
        if log_stationary:
            scale = np.ones_like(joint0)
            log_lower, log_upper, resolved_scale_bounds = (
                _log_stationary_optimizer_bounds(
                    copula,
                    self.log_stationary_scale_optimization,
                    self.stationary_scale_bounds,
                ))
            x0_scaled = _ou_to_log_stationary(joint0)
            x0_scaled[:3] = _project_log_stationary_initial_point(
                x0_scaled[:3], log_lower, log_upper)
        else:
            scale = model_policy.optimizer_unit_scale(joint0)
            x0_scaled = _cpp_scar_ou.physical_to_scaled(joint0, scale)
            resolved_scale_bounds = (None, None)
        lower = np.full(3 + n_corr, -np.inf, dtype=np.float64)
        upper = np.full(3 + n_corr, np.inf, dtype=np.float64)
        if log_stationary:
            lower[:3] = log_lower
            upper[:3] = log_upper
        else:
            scaled_lower, scaled_upper = (
                model_policy.ou_scaled_optimizer_bounds(scale[:3]))
            lower[:3] = scaled_lower
            upper[:3] = scaled_upper
        optimizer_box = Bounds(lower, upper)

        def optimizer_to_joint(values):
            return (
                _ou_from_log_stationary(values)
                if log_stationary
                else _cpp_scar_ou.scaled_to_physical(values, scale))

        def joint_to_optimizer(values):
            return (
                _ou_to_log_stationary(values)
                if log_stationary
                else _cpp_scar_ou.physical_to_scaled(values, scale))

        diagnostics = _new_backend_diagnostics()
        diagnostics["n_threads"] = self.config.n_threads
        diagnostics.update({
            "initialization": initialization,
            "optimizer_parameterization": (
                "log_kappa_mu_log_stationary_sigma"
                if log_stationary else "scaled_kappa_mu_nu"),
            "optimizer_stationary_scale_bounds": resolved_scale_bounds,
            "joint_static": True,
            "joint_optimizer": "python-lbfgsb",
            "correlation_parameterization_engine": "cpp",
            "correlation_gradient": (
                "analytical" if self.analytical_grad else "optimizer_numerical"),
            "cpp_correlation_derivatives": False,
            "analytical_grad_requested": bool(self.analytical_grad),
            "analytical_grad_used": bool(self.analytical_grad),
            "joint_gradient": (
                "analytical" if self.analytical_grad else "optimizer_numerical"),
            "ou_gradient": (
                "analytical" if self.analytical_grad else "numerical"),
            "correlation_fd_scheme": (
                "none" if self.analytical_grad else "optimizer"),
            "hybrid_gradient_evaluations": 0,
            "correlation_fd_evaluations": 0,
            "native_correlation_gradient_evaluations": 0,
            "shrinkage_directional_gradient": False,
            "prepared_native_evaluator": False,
            "prepared_native_evaluator_count": 0,
            "prepared_native_fallback": None,
            "adaptive_spectral_basis_order": (
                self.spectral_basis_order == "auto"),
            "auto_spectral_basis_order": (
                self.spectral_basis_order == "auto"),
            "model_score": "not_applicable",
            "optimizer_gradient": (
                "analytical" if self.analytical_grad else "numerical"),
            "gradient_kind": (
                "analytical" if self.analytical_grad else "numerical"),
            "setup_derivative": (
                "analytical" if self.analytical_grad else "not_used"),
            "filter_derivative": (
                "analytical" if self.analytical_grad else "not_used"),
        })
        fail_value = float(getattr(self.config, "fail_value", 1e10))
        prepared_cache = _PreparedScarOuFitCache(u, copula, diagnostics)

        def evaluate_value(joint):
            kappa_v, mu_v, nu_v = joint[:3]
            copula._set_corr_from_params(joint[3:])
            auto_config = self._auto_config(
                kappa=kappa_v, n_obs=len(u))
            value, info = prepared_cache.neg_loglik_info(
                kappa_v, mu_v, nu_v, auto_config)
            _record_backend_diagnostics(diagnostics, info, "cpp")
            return value

        def evaluate_value_and_ou_grad(joint):
            kappa_v, mu_v, nu_v = joint[:3]
            copula._set_corr_from_params(joint[3:])
            auto_config = self._auto_config(
                kappa=kappa_v, n_obs=len(u))
            try:
                if getattr(copula, "_corr_mode", None) == "shrinkage":
                    direction = _shrinkage_raw_corr_direction(
                        joint[3:], copula._corr_base)
                    value, grad, corr_grad, info = (
                        prepared_cache
                        .neg_loglik_with_grad_and_corr_directional_info(
                            kappa_v, mu_v, nu_v, direction, auto_config))
                    corr_kind = "directional"
                else:
                    value, grad, corr_grad, info = (
                        prepared_cache.neg_loglik_with_grad_and_corr_info(
                            kappa_v, mu_v, nu_v, auto_config))
                    corr_kind = "full"
            except AttributeError:
                value, grad, corr_grad, info = (
                    _cpp_scar_ou.neg_loglik_with_grad_and_corr_info(
                        kappa_v, mu_v, nu_v, u, copula, auto_config))
                corr_kind = "full"
            _record_backend_diagnostics(diagnostics, info, "cpp")
            return (
                float(value),
                np.asarray(grad, dtype=np.float64),
                np.asarray(corr_grad, dtype=np.float64),
                corr_kind,
            )

        def objective_scaled(x_scaled):
            joint = optimizer_to_joint(x_scaled)
            if not np.all(np.isfinite(joint)):
                return model_policy.optimizer_failure_evaluation(
                    x_scaled,
                    x0_scaled,
                    fail_value,
                    directional_gradient=False,
                )[0]
            try:
                return evaluate_value(joint)
            except FloatingPointError as exc:
                if verbose:
                    print(f"  error at joint alpha={joint}: {exc}")
                return model_policy.optimizer_failure_objective(
                    exc, fail_value)

        def objective_and_grad_scaled(x_scaled):
            joint = optimizer_to_joint(x_scaled)
            if not np.all(np.isfinite(joint)):
                return model_policy.optimizer_failure_evaluation(
                    x_scaled,
                    x0_scaled,
                    fail_value,
                    directional_gradient=False,
                )
            diagnostics["hybrid_gradient_evaluations"] += 1
            try:
                value, ou_grad, corr_grad, corr_kind = (
                    evaluate_value_and_ou_grad(joint))
                if (
                        native_validation.objective_is_invalid(value)
                        or ou_grad.shape != (3,)
                        or not np.all(np.isfinite(ou_grad))):
                    return model_policy.optimizer_failure_evaluation(
                        x_scaled,
                        x0_scaled,
                        fail_value,
                        directional_gradient=False,
                    )

                grad = np.empty_like(joint)
                grad[:3] = ou_grad
                if corr_kind == "directional":
                    if corr_grad.shape != (n_corr,):
                        return model_policy.optimizer_failure_evaluation(
                            x_scaled,
                            x0_scaled,
                            fail_value,
                            directional_gradient=False,
                        )
                    grad[3:] = corr_grad
                    diagnostics[
                        "correlation_gradient"] = "analytical_directional"
                    diagnostics[
                        "shrinkage_directional_gradient"] = True
                else:
                    grad[3:] = _corr_gradient_to_raw_params(
                        copula._corr_mode,
                        joint[3:],
                        copula.R,
                        corr_grad,
                        copula._corr_base,
                    )
                    diagnostics["correlation_gradient"] = "analytical"
                diagnostics["cpp_correlation_derivatives"] = True
                diagnostics["joint_gradient"] = "analytical"
                diagnostics["correlation_fd_scheme"] = "none"
                diagnostics[
                    "native_correlation_gradient_evaluations"] += 1
                if log_stationary:
                    grad[:3] = _ou_grad_to_log_stationary(
                        joint[:3], grad[:3])
                return value, _cpp_scar_ou.gradient_to_scaled(grad, scale)
            except FloatingPointError as exc:
                if verbose:
                    print(f"  error at joint alpha={joint}: {exc}")
                return model_policy.optimizer_numerical_failure_evaluation(
                    exc,
                    x_scaled,
                    x0_scaled,
                    fail_value,
                    directional_gradient=False,
                )

        if verbose:
            gradient = "hybrid gradient" if self.analytical_grad else (
                "numerical gradient")
            print(
                f"Fitting SCAR-TM-OU (C++, joint static correlation, "
                f"{gradient}), alpha0={joint0}")

        if self.analytical_grad:
            scaled_options = dict(optimizer_options)
            scaled_options.pop("eps", None)
            scaled_options.pop("finite_diff_rel_step", None)
            result = minimize(
                objective_and_grad_scaled,
                x0_scaled,
                method='L-BFGS-B',
                jac=True,
                bounds=optimizer_box,
                options=scaled_options,
            )
        else:
            scaled_options = dict(optimizer_options)
            if 'eps' in scaled_options and not log_stationary:
                scaled_options['eps'] = (
                    float(scaled_options['eps']) / scale)
            result = minimize(
                objective_scaled,
                x0_scaled,
                method='L-BFGS-B',
                bounds=optimizer_box,
                options=scaled_options,
            )
        result.x = optimizer_to_joint(result.x)

        joint = result.x
        try:
            copula._set_corr_from_params(joint[3:])
        except Exception as exc:
            result.success = False
            result.message = (
                f"{result.message}; failed to set final correlation: {exc}")
            copula._set_corr_from_params(corr0)

        def selected_final_evaluator(values):
            value, gradient_scaled = objective_and_grad_scaled(
                joint_to_optimizer(values))
            copula._set_corr_from_params(values[3:])
            if log_stationary:
                # The validation contract expects the physical OU gradient.
                gradient = np.asarray(gradient_scaled, dtype=np.float64)
                gradient[:3] = _ou_grad_from_log_stationary(
                    values[:3], gradient[:3])
                return value, gradient
            return value, _cpp_scar_ou.gradient_from_scaled(
                gradient_scaled, scale)

        def validate_correlation():
            return native_validation.validate_correlation_fit_state(
                joint[3:],
                n_corr,
                copula.R,
                copula.d,
                copula._L_inv,
                copula._log_det,
                tolerance=1e-10,
            )

        validation_diagnostics = self._validate_final_fit(
            result=result,
            final_params=joint,
            initial_params=joint0,
            lower=np.concatenate((
                (_OU_BOUNDED_LOG_STATIONARY_RESULT_LOWER
                 if bounded_log_stationary else _OU_PHYSICAL_LOWER),
                np.full(n_corr, -np.inf),
            )),
            upper=np.full(3 + n_corr, np.inf),
            selected_evaluator=selected_final_evaluator,
            validation_evaluator=None,
            selected_engine=selected_engine,
            validation_engine=None,
            n_obs=len(u),
            optimizer_options=optimizer_options,
            correlation_validator=validate_correlation,
        )
        diagnostics.update(validation_diagnostics)
        try:
            copula._set_corr_from_params(joint[3:])
        except Exception:
            pass

        diagnostics.update({
            "corr_mode": copula._corr_mode,
            "corr_n_params": n_corr,
            "corr_params_raw": copula.corr_params(),
            "corr_alpha": copula.corr_alpha(),
            "corr_matrix": copula.R.copy(),
        })

        if verbose:
            print(f"  => joint alpha={joint}, logL={-result.fun:.4f}")

        params = ou_params(
            kappa=joint[0], mu=joint[1], nu=joint[2])
        return LatentResult(
            log_likelihood=-result.fun,
            method='SCAR-TM-OU',
            copula_name=copula.name,
            success=result.success,
            nfev=result.nfev,
            message=str(result.message),
            params=params,
            parameter_count=3 + n_corr,
            K=self.K,
            grid_range=self.grid_range,
            grid_method=self.grid_method,
            adaptive=self.adaptive,
            pts_per_sigma=self.pts_per_sigma,
            transition_method=self.transition_method,
            max_K=self.max_K,
            r_gh=self.r_gh,
            gh_order=self.gh_order,
            auto_small_kdt=self.auto_small_kdt,
            spectral_basis_order=self.spectral_basis_order,
            spectral_quad_order=self.spectral_quad_order,
            diagnostics=diagnostics,
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
            initial_mle_result=None,
            **kwargs) -> LatentResult:
        """Fit SCAR-TM-OU model.

        Parameters
        ----------
        copula : exact registered built-in copula
        u : (T, 2) pseudo-observations
        alpha0 : (3,) or (3 + n_corr,)
            Initial ``[kappa, mu, nu]`` with model correlation defaults, a
            full joint vector, or None for automatic initialization.
        gtol, ftol, maxfun, maxiter, maxls, eps, maxcor,
        finite_diff_rel_step : L-BFGS-B options
        verbose : print progress
        initial_mle_result : MLEResult, optional
            Existing static fit used only for automatic initialization.

        Returns
        -------
        LatentResult
        """
        reject_unknown_strategy_kwargs("SCAR-TM-OU", kwargs)
        log_stationary = _uses_log_stationary_scale(
            copula, self.log_stationary_scale_optimization)
        optimizer_options = lbfgsb_options(
            self._optimizer_config(
                copula, log_stationary=log_stationary),
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
        from pyscarcopula.copula.multivariate.equicorr_prepared import (
            EquicorrPreparedData,
        )
        if not isinstance(u, EquicorrPreparedData):
            u = as_pseudo_observation_array(u)
        if len(u) < 2:
            raise ValueError(
                "SCAR-TM-OU requires at least two observations")
        corr_num_params = getattr(copula, "_corr_num_params", None)
        n_corr = int(corr_num_params()) if callable(corr_num_params) else 0
        if n_corr:
            return self._fit_joint_static(
                copula, u, alpha0, optimizer_options, verbose,
                initial_mle_result)
        # ── Initial point ─────────────────────────────────────────
        alpha0, initialization = _resolve_initial_point(
            copula,
            u,
            self.config,
            self.smart_init,
            verbose,
            alpha0,
            initial_mle_result,
        )
        alpha0 = np.asarray(alpha0, dtype=np.float64)
        bounded_log_stationary = _uses_bounded_log_stationary_scale(
            copula, self.log_stationary_scale_optimization)
        if log_stationary:
            log_lower, log_upper, resolved_scale_bounds = (
                _log_stationary_optimizer_bounds(
                    copula,
                    self.log_stationary_scale_optimization,
                    self.stationary_scale_bounds,
                ))
        else:
            log_lower = log_upper = None
            resolved_scale_bounds = (None, None)

        self._uses_cpp(copula)
        selected_engine = "cpp"
        diagnostics = _new_backend_diagnostics()
        diagnostics["n_threads"] = self.config.n_threads
        diagnostics["adaptive_spectral_basis_order"] = (
            self.spectral_basis_order == "auto")
        diagnostics["auto_spectral_basis_order"] = (
            self.spectral_basis_order == "auto")
        diagnostics.update({
            "initialization": initialization,
            "optimizer_parameterization": (
                "log_kappa_mu_log_stationary_sigma"
                if log_stationary
                else "scaled_kappa_mu_nu"),
            "optimizer_stationary_scale_bounds": resolved_scale_bounds,
            "model_score": "not_applicable",
            "optimizer_gradient": (
                "analytical" if self.analytical_grad else "numerical"),
            "gradient_kind": (
                "analytical" if self.analytical_grad else "numerical"),
            "setup_derivative": (
                "analytical" if self.analytical_grad else "not_used"),
            "filter_derivative": (
                "analytical" if self.analytical_grad else "not_used"),
            "analytical_grad_requested": bool(self.analytical_grad),
            "analytical_grad_used": bool(self.analytical_grad),
        })
        fail_value = float(getattr(self.config, "fail_value", 1e10))
        prepared_cache = _PreparedScarOuFitCache(u, copula, diagnostics)

        def _auto_config_for(kappa_v):
            return self._auto_config(kappa=kappa_v, n_obs=len(u))

        # ── Fit with analytical gradient ──────────────────────────
        if self.analytical_grad:
            if log_stationary:
                scale = np.ones(3, dtype=np.float64)
                x0_scaled = _ou_to_log_stationary(alpha0)
                x0_scaled = _project_log_stationary_initial_point(
                    x0_scaled, log_lower, log_upper)
                optimizer_box = Bounds(log_lower, log_upper)
            else:
                # Rescale parameters so all three are O(1) at start.
                # This helps L-BFGS-B estimate the initial Hessian.
                scale = model_policy.optimizer_unit_scale(alpha0)
                x0_scaled = _cpp_scar_ou.physical_to_scaled(alpha0, scale)
                optimizer_box = Bounds(
                    *model_policy.ou_scaled_optimizer_bounds(scale))

            def objective_and_grad(x_scaled):
                alpha = (
                    _ou_from_log_stationary(x_scaled)
                    if log_stationary
                    else _cpp_scar_ou.scaled_to_physical(
                        x_scaled, scale))
                if not np.all(np.isfinite(alpha)):
                    return model_policy.optimizer_failure_evaluation(
                        x_scaled,
                        x0_scaled,
                        fail_value,
                        directional_gradient=False,
                    )
                kappa_v, mu_v, nu_v = alpha
                try:
                    auto_config = _auto_config_for(kappa_v)
                    val, grad, info = (
                        prepared_cache.neg_loglik_with_grad_info(
                            kappa_v, mu_v, nu_v, auto_config))
                    _record_backend_diagnostics(diagnostics, info, "cpp")
                    if log_stationary:
                        return val, _ou_grad_to_log_stationary(alpha, grad)
                    return val, _cpp_scar_ou.gradient_to_scaled(
                        grad, scale)
                except FloatingPointError as e:
                    if verbose:
                        print(f"  error at alpha={alpha}: {e}")
                    return model_policy.optimizer_numerical_failure_evaluation(
                        e,
                        x_scaled,
                        x0_scaled,
                        fail_value,
                        directional_gradient=False,
                    )

            if verbose:
                print(f"Fitting SCAR-TM-OU (analytical gradient), alpha0={alpha0}")

            result = minimize(
                objective_and_grad, x0_scaled,
                method='L-BFGS-B',
                jac=True,
                bounds=optimizer_box,
                options=optimizer_options,
            )

            if not result.success and str(result.message).startswith('ABNORMAL'):
                final_val, final_grad = objective_and_grad(result.x)
                if log_stationary:
                    physical_x = _ou_from_log_stationary(result.x)
                    physical_grad = _ou_grad_from_log_stationary(
                        physical_x, final_grad)
                else:
                    physical_x = _cpp_scar_ou.scaled_to_physical(
                        result.x, scale)
                    physical_grad = _cpp_scar_ou.gradient_from_scaled(
                        final_grad, scale)
                pg_norm = _projected_gradient_norm(
                    physical_x,
                    physical_grad,
                    (_OU_BOUNDED_LOG_STATIONARY_RESULT_LOWER
                     if bounded_log_stationary else _OU_PHYSICAL_LOWER),
                    _OU_PHYSICAL_UPPER,
                )
                acceptable_boundary = (
                    np.isfinite(final_val)
                    and not native_validation.objective_is_invalid(final_val)
                    and np.all(np.isfinite(result.x))
                    and pg_norm <= max(float(optimizer_options.get('gtol', 1e-5)), 1e-2)
                )
                if acceptable_boundary:
                    result.fun = final_val
                    result.jac = final_grad
                    result.success = True
                    result.message = (
                        f"{result.message} accepted as boundary convergence "
                        f"(physical_projected_grad={pg_norm:.3g})"
                    )

            result.x = (
                _ou_from_log_stationary(result.x)
                if log_stationary
                else _cpp_scar_ou.scaled_to_physical(result.x, scale))

        # ── Fit with numerical gradient ───────────────────────────
        else:
            if log_stationary:
                scale = np.ones(3, dtype=np.float64)
                x0_scaled = _ou_to_log_stationary(alpha0)
                x0_scaled = _project_log_stationary_initial_point(
                    x0_scaled, log_lower, log_upper)
                optimizer_box = Bounds(log_lower, log_upper)
            else:
                scale = model_policy.optimizer_unit_scale(alpha0)
                x0_scaled = _cpp_scar_ou.physical_to_scaled(alpha0, scale)
                optimizer_box = Bounds(
                    *model_policy.ou_scaled_optimizer_bounds(scale))

            def objective_scaled(x_scaled):
                alpha = (
                    _ou_from_log_stationary(x_scaled)
                    if log_stationary
                    else _cpp_scar_ou.scaled_to_physical(
                        x_scaled, scale))
                if not np.all(np.isfinite(alpha)):
                    return model_policy.optimizer_failure_evaluation(
                        x_scaled,
                        x0_scaled,
                        fail_value,
                        directional_gradient=False,
                    )[0]
                kappa_v, mu_v, nu_v = alpha
                try:
                    auto_config = _auto_config_for(kappa_v)
                    val, info = prepared_cache.neg_loglik_info(
                        kappa_v, mu_v, nu_v, auto_config)
                    _record_backend_diagnostics(diagnostics, info, "cpp")
                    return val
                except FloatingPointError as e:
                    if verbose:
                        print(f"  error at alpha={alpha}: {e}")
                    return model_policy.optimizer_failure_objective(
                        e, fail_value)

            if verbose:
                print(
                    f"Fitting SCAR-TM-OU (C++, numerical gradient), "
                    f"alpha0={alpha0}")

            scaled_options = dict(optimizer_options)
            if 'eps' in scaled_options and not log_stationary:
                scaled_options['eps'] = float(scaled_options['eps']) / scale

            result = minimize(
                objective_scaled, x0_scaled,
                method='L-BFGS-B',
                bounds=optimizer_box,
                options=scaled_options,
            )
            result.x = (
                _ou_from_log_stationary(result.x)
                if log_stationary
                else _cpp_scar_ou.scaled_to_physical(result.x, scale))

        alpha = result.x
        def evaluate_final(values, with_grad, record=False):
            kappa_v, mu_v, nu_v = values[:3]
            auto_config = _auto_config_for(kappa_v)
            if with_grad:
                value, grad, info = prepared_cache.neg_loglik_with_grad_info(
                    kappa_v, mu_v, nu_v, auto_config)
                if record:
                    _record_backend_diagnostics(diagnostics, info, "cpp")
                return value, grad
            value, info = prepared_cache.neg_loglik_info(
                kappa_v, mu_v, nu_v, auto_config)
            if record:
                _record_backend_diagnostics(diagnostics, info, "cpp")
            return value

        validation_diagnostics = self._validate_final_fit(
            result=result,
            final_params=alpha,
            initial_params=alpha0,
            lower=(
                _OU_BOUNDED_LOG_STATIONARY_RESULT_LOWER
                if bounded_log_stationary else _OU_PHYSICAL_LOWER),
            upper=_OU_PHYSICAL_UPPER,
            selected_evaluator=lambda values: evaluate_final(
                values, True, record=True),
            validation_evaluator=None,
            selected_engine=selected_engine,
            validation_engine=None,
            n_obs=len(u),
            optimizer_options=optimizer_options,
        )
        diagnostics.update(validation_diagnostics)

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
            K=self.K,
            grid_range=self.grid_range,
            grid_method=self.grid_method,
            adaptive=self.adaptive,
            pts_per_sigma=self.pts_per_sigma,
            transition_method=self.transition_method,
            max_K=self.max_K,
            r_gh=self.r_gh,
            gh_order=self.gh_order,
            auto_small_kdt=self.auto_small_kdt,
            spectral_basis_order=self.spectral_basis_order,
            spectral_quad_order=self.spectral_quad_order,
            diagnostics=diagnostics,
        )

    def log_likelihood(self, copula, u: np.ndarray,
                       result: LatentResult) -> float:
        """Evaluate TM log-likelihood at fitted parameters."""
        p = result.params
        cfg = self._auto_config(kappa=p.kappa, n_obs=len(u))
        self._uses_cpp(copula)
        value, _ = _cpp_scar_ou.loglik(
            p.kappa, p.mu, p.nu, u, copula, cfg)
        return value

    def predictive_mean(self, copula, u: np.ndarray,
                        result: LatentResult,
                        posterior_cache=None) -> np.ndarray:
        """E[Psi(x_k) | u_{1:k-1}] via TM forward pass."""
        p = result.params
        self._uses_cpp(copula)
        cfg = self._auto_config(
            self._grid_transition_method(),
            kappa=p.kappa,
            n_obs=len(u),
        )
        return self._prepared_or_stateless_posterior(
            copula,
            u,
            cfg,
            lambda prepared: prepared.predictive_mean(
                p.kappa, p.mu, p.nu),
            lambda: _cpp_scar_ou.predictive_mean(
                p.kappa, p.mu, p.nu, u, copula, cfg),
            posterior_cache=posterior_cache,
        )

    def rosenblatt_e2(self, copula, u: np.ndarray,
                      result: LatentResult,
                      posterior_cache=None) -> np.ndarray:
        """Mixture Rosenblatt: e2 = E[h(u2, u1; Psi(x_k)) | u_{1:k-1}]."""
        p = result.params
        self._uses_cpp(copula)
        cfg = self._auto_config(
            self._grid_transition_method(),
            kappa=p.kappa,
            n_obs=len(u),
        )
        return self._prepared_or_stateless_posterior(
            copula,
            u,
            cfg,
            lambda prepared: prepared.mixture_h(p.kappa, p.mu, p.nu),
            lambda: _cpp_scar_ou.mixture_h(
                p.kappa, p.mu, p.nu, u, copula, cfg),
            posterior_cache=posterior_cache,
        )

    def mixture_h(self, copula, u: np.ndarray,
                  result: LatentResult, state_cache=None,
                  current_cache_key=None, next_cache_key=None,
                  posterior_cache=None) -> np.ndarray:
        """Mixture h-function for vine pseudo-obs propagation."""
        if is_multivariate_copula(copula):
            raise NotImplementedError(
                "mixture_h is not defined for multivariate "
                "StochasticStudent-compatible copulas")
        p = result.params
        self._uses_cpp(copula)
        cfg = self._auto_config(
            self._grid_transition_method(),
            kappa=p.kappa,
            n_obs=len(u),
        )
        current_state = None
        next_state = None
        workspace = self._posterior_workspace_or_none(posterior_cache)
        if workspace is not None:
            h_mix = workspace.mixture_h(
                p.kappa, p.mu, p.nu, u, copula, cfg)
            if state_cache is not None:
                if current_cache_key is not None:
                    current_state = workspace.state_distribution(
                        p.kappa, p.mu, p.nu, u, copula, cfg,
                        horizon='current')
                if next_cache_key is not None:
                    next_state = workspace.state_distribution(
                        p.kappa, p.mu, p.nu, u, copula, cfg, horizon='next')
        else:
            prepared = None
            try:
                prepared = _cpp_scar_ou.prepare_objective(u, copula, cfg)
                prepared.update_copula(copula)
                h_mix = prepared.mixture_h(p.kappa, p.mu, p.nu)
                if state_cache is not None:
                    if current_cache_key is not None:
                        current_state = prepared.state_distribution(
                            p.kappa, p.mu, p.nu, horizon='current')
                    if next_cache_key is not None:
                        next_state = prepared.state_distribution(
                            p.kappa, p.mu, p.nu, horizon='next')
            except AttributeError:
                prepared = None
            except _cpp_scar_ou.NativeUnsupported:
                prepared = None
            if prepared is None:
                h_mix = _cpp_scar_ou.mixture_h(
                    p.kappa, p.mu, p.nu, u, copula, cfg)
                if state_cache is not None:
                    if current_cache_key is not None:
                        current_state = _cpp_scar_ou.state_distribution(
                            p.kappa, p.mu, p.nu, u, copula, cfg,
                            horizon='current')
                    if next_cache_key is not None:
                        next_state = _cpp_scar_ou.state_distribution(
                            p.kappa, p.mu, p.nu, u, copula, cfg,
                            horizon='next')
        if state_cache is not None:
            if current_cache_key is not None:
                state_cache[current_cache_key] = current_state
            if next_cache_key is not None:
                state_cache[next_cache_key] = next_state
        return h_mix

    def mixture_h_pair(self, copula, u: np.ndarray,
                       result: LatentResult, state_cache=None,
                       current_cache_key=None, next_cache_key=None,
                       posterior_cache=None):
        """Both vine h-directions from one SCAR posterior pass."""
        if is_multivariate_copula(copula):
            raise NotImplementedError(
                "mixture_h_pair is not defined for multivariate "
                "StochasticStudent-compatible copulas")
        p = result.params
        self._uses_cpp(copula)
        cfg = self._auto_config(
            self._grid_transition_method(),
            kappa=p.kappa,
            n_obs=len(u),
        )
        current_state = None
        next_state = None
        workspace = self._posterior_workspace_or_none(posterior_cache)
        if workspace is not None:
            h_pair = workspace.mixture_h_pair(
                p.kappa, p.mu, p.nu, u, copula, cfg)
            if state_cache is not None:
                if current_cache_key is not None:
                    current_state = workspace.state_distribution(
                        p.kappa, p.mu, p.nu, u, copula, cfg,
                        horizon='current')
                if next_cache_key is not None:
                    next_state = workspace.state_distribution(
                        p.kappa, p.mu, p.nu, u, copula, cfg, horizon='next')
        else:
            prepared = None
            try:
                prepared = _cpp_scar_ou.prepare_objective(u, copula, cfg)
                prepared.update_copula(copula)
                h_pair = prepared.mixture_h_pair(p.kappa, p.mu, p.nu)
                if state_cache is not None:
                    if current_cache_key is not None:
                        current_state = prepared.state_distribution(
                            p.kappa, p.mu, p.nu, horizon='current')
                    if next_cache_key is not None:
                        next_state = prepared.state_distribution(
                            p.kappa, p.mu, p.nu, horizon='next')
            except AttributeError:
                prepared = None
            except _cpp_scar_ou.NativeUnsupported:
                prepared = None
            if prepared is None:
                h_pair = _cpp_scar_ou.mixture_h_pair(
                    p.kappa, p.mu, p.nu, u, copula, cfg)
                if state_cache is not None:
                    if current_cache_key is not None:
                        current_state = _cpp_scar_ou.state_distribution(
                            p.kappa, p.mu, p.nu, u, copula, cfg,
                            horizon='current')
                    if next_cache_key is not None:
                        next_state = _cpp_scar_ou.state_distribution(
                            p.kappa, p.mu, p.nu, u, copula, cfg,
                            horizon='next')
        if state_cache is not None:
            if current_cache_key is not None:
                state_cache[current_cache_key] = current_state
            if next_cache_key is not None:
                state_cache[next_cache_key] = next_state
        return h_pair

    def objective(self, copula, u: np.ndarray,
                  alpha: np.ndarray, **kwargs) -> float:
        """Minus log-likelihood: TM integrated -logL(kappa, mu, nu)."""
        cfg = self._auto_config(kappa=alpha[0], n_obs=len(u))
        self._uses_cpp(copula)
        return _cpp_scar_ou.neg_loglik(
            alpha[0], alpha[1], alpha[2], u, copula,
            cfg)

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
        d = copula_dimension(copula, u)
        return sample_predictive(
            copula, n, r, given=kwargs.get('given'), rng=rng, d=d)

    predict = strategy_predict
    predictive_params = predictive_params_from_state_with_rng

    def predictive_state(self, copula, u, result, **kwargs):
        """Return SCAR-TM predictive state as a grid distribution."""
        p = result.params
        if u is None:
            return PredictiveState(
                method='SCAR-TM-OU',
                horizon=str(kwargs.get('horizon', 'next')).lower(),
                kind='stationary_normal',
                metadata={
                    'kappa': p.kappa,
                    'mu': p.mu,
                    'nu': p.nu,
                },
            )

        state_cache = kwargs.get('state_cache')
        cache_key = kwargs.get('cache_key')
        cached = None
        if state_cache is not None and cache_key is not None:
            cached = state_cache.get(cache_key)

        if cached is None:
            horizon = kwargs.get('horizon', 'next')
            self._uses_cpp(copula)
            cfg = self._auto_config(
                self._grid_transition_method(),
                kappa=p.kappa,
                n_obs=len(u),
            )
            cached = self._prepared_or_stateless_posterior(
                copula,
                u,
                cfg,
                lambda prepared: prepared.state_distribution(
                    p.kappa, p.mu, p.nu, horizon=horizon),
                lambda: _cpp_scar_ou.state_distribution(
                    p.kappa, p.mu, p.nu, u, copula, cfg,
                    horizon=horizon),
                posterior_cache=kwargs.get('posterior_cache'),
            )
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
        if u.ndim != 2 or len(u) == 0:
            return state
        u = u[:1]

        z_grid, weights = _cpp_scar_ou.condition_state(
            copula, state.z_grid, state.prob, u)
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
            x_t = _cpp_scar_ou.sample_stationary_fixed_draws(
                state.metadata['kappa'],
                state.metadata['mu'],
                state.metadata['nu'],
                rng.standard_normal(n),
            )
            return copula.transform(x_t)

        mode = kwargs.get('predictive_r_mode')
        mode = 'histogram' if mode is None else str(mode).lower()
        selection_draws = rng.uniform(0.0, 1.0, size=n)
        jitter_draws = (
            rng.uniform(0.0, 1.0, size=n)
            if mode == 'histogram' and len(state.z_grid) > 1
            else np.empty(0, dtype=np.float64)
        )
        z_samples, _ = _cpp_scar_ou.sample_state_distribution_fixed_draws(
            state.z_grid,
            state.prob,
            selection_draws,
            jitter_draws,
            mode=mode,
        )
        return copula.transform(z_samples)

    def model_sample_params(self, copula, result, n, rng=None, **kwargs):
        """OU trajectory parameters for unconditional model reproduction."""
        if rng is None:
            rng = np.random.default_rng()

        p = result.params
        kappa, mu, nu = p.kappa, p.mu, p.nu
        x = sample_ou_trajectory(kappa, mu, nu, n, rng)
        return copula.transform(x)

    def model_sample_state(self, copula, result, **kwargs):
        return None
