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
    get_copula_capabilities,
    register_strategy,
)
from pyscarcopula.numerical._scar_ou_config import (
    AutoTMConfig,
    validate_cpp_config,
)
from pyscarcopula.numerical._arrays import as_float64_array
from pyscarcopula.numerical._transition_methods import (
    normalize_ou_transition_method,
)
from pyscarcopula.strategy.predict_helpers import (
    predict_from_strategy,
    sample_predictive,
)
from pyscarcopula.strategy.initial_point import (
    _explicit_initialization_diagnostics,
    _fallback_initialization_diagnostics,
    _initialization_attempt,
    _initialization_diagnostics,
    smart_initial_point,
)
from pyscarcopula.numerical import _cpp_scar_ou, copula_native
from pyscarcopula.copula.multivariate.corr_param import (
    _corr_gradient_to_raw_params,
    _shrinkage_raw_corr_direction,
)


_INVALID_OBJECTIVE_THRESHOLD = 1e9

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

_ADAPTIVE_SPECTRAL_BASIS_ORDER = (
    (0.015, 128),
    (0.025, 96),
    (0.06, 64),
    (np.inf, 32),
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
    order = int(value)
    if order <= 0:
        raise ValueError("spectral_basis_order must be positive")
    return order


def _objective_is_invalid(value):
    return (not np.isfinite(value)) or float(value) >= _INVALID_OBJECTIVE_THRESHOLD


def _resolve_initial_point(
        copula, u, config, smart_init, verbose, alpha0):
    """Return an OU initial point and common initialization diagnostics."""
    if alpha0 is not None:
        alpha = np.asarray(alpha0, dtype=np.float64)
        return alpha, _explicit_initialization_diagnostics(alpha)

    smart_diagnostics = None
    if smart_init:
        try:
            alpha, info = smart_initial_point(
                u, copula, verbose=verbose)
            if verbose:
                print(f"Smart init: {info.get('chosen_method')}, "
                      f"alpha0={alpha}")
            return alpha, info['initialization']
        except Exception as exc:
            smart_diagnostics = _initialization_diagnostics(
                'automatic',
                'failed',
                np.array([1.0, 0.0, 1.0]),
                [_initialization_attempt(
                    'smart_initial_point', success=False, error=exc)],
            )
            smart_diagnostics['success'] = False
            if verbose:
                print(
                    "Smart init failed "
                    f"({type(exc).__name__}: {exc}); trying mle_default")

    from pyscarcopula.strategy.mle import MLEStrategy
    mle = MLEStrategy(config=config)
    try:
        mle_result = mle.fit(copula, u)
        mu0 = float(np.atleast_1d(
            copula.inv_transform(
                np.atleast_1d(mle_result.copula_param))
        )[0])
        alpha = np.array([1.0, mu0, 1.0])
    except Exception as exc:
        if smart_diagnostics is not None:
            _fallback_initialization_diagnostics(
                smart_diagnostics,
                'mle_default',
                np.array([1.0, 0.0, 1.0]),
                error=exc,
            )
        raise

    if smart_diagnostics is None:
        diagnostics = _initialization_diagnostics(
            'mle_default',
            'mle_default',
            alpha,
            [_initialization_attempt('mle_default', success=True)],
        )
    else:
        diagnostics = _fallback_initialization_diagnostics(
            smart_diagnostics, 'mle_default', alpha)
    return alpha, diagnostics


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
        except _cpp_scar_ou.CppUnsupported:
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
            except _cpp_scar_ou.CppUnsupported:
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
            except _cpp_scar_ou.CppUnsupported:
                self.cache[key] = _POSTERIOR_WORKSPACE_UNSUPPORTED
                return None
            self.cache[key] = prepared
        try:
            prepared.update_copula(copula)
            return prepared
        except AttributeError:
            self.disable()
            return None
        except _cpp_scar_ou.CppUnsupported:
            self.cache[key] = _POSTERIOR_WORKSPACE_UNSUPPORTED
            return None

    def _call(self, u, copula, auto_config, prepared_call, fallback_call):
        prepared = self.prepared_for(u, copula, auto_config)
        if prepared is not None:
            try:
                return prepared_call(prepared)
            except AttributeError:
                self.disable()
            except _cpp_scar_ou.CppUnsupported:
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
                 auto_small_kdt: float = 1e-2,
                 spectral_basis_order: int | str = "auto",
                 spectral_quad_order: int | None = None,
                 analytical_grad: bool = True,
                 smart_init: bool = True,
                 final_validation_abs_per_obs: float = 5e-5,
                 final_validation_rel_tol: float = 1e-5,
                 final_gradient_tolerance: float | None = None,
                 final_growth_limit: float = 1e8,
                 final_rho_tolerance: float = 1e-15,
                 **kwargs):
        if "backend" in kwargs:
            raise TypeError(
                "SCAR-TM-OU backend selection was removed; native execution "
                "is always used")
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
        self.final_validation_abs_per_obs = float(
            final_validation_abs_per_obs)
        self.final_validation_rel_tol = float(final_validation_rel_tol)
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
            )
        )

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

    def _kappa_dt(self, kappa: float, n_obs: int) -> float:
        if n_obs <= 1:
            return float(kappa)
        return float(kappa) / float(n_obs - 1)

    def _adaptive_spectral_basis_order(self, kappa: float, n_obs: int) -> int:
        kdt = self._kappa_dt(kappa, n_obs)
        for threshold, basis_order in _ADAPTIVE_SPECTRAL_BASIS_ORDER:
            if kdt < threshold:
                return basis_order
        return 32

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
        except _cpp_scar_ou.CppUnsupported:
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
        reasons = []
        diagnostics = {
            "final_validation_passed": False,
            "final_validation_reasons": (),
            "final_selected_engine": selected_engine,
            "final_validation_engine": validation_engine,
        }

        if final_params.size < 3 or not np.all(np.isfinite(final_params)):
            reasons.append("final parameters are not finite")
        elif final_params[0] <= 0.0 or final_params[2] <= 0.0:
            reasons.append("final kappa and nu must be positive")

        selected_value = np.nan
        selected_grad = np.full(final_params.shape, np.nan)
        if not reasons:
            try:
                selected_value, selected_grad = selected_evaluator(final_params)
                selected_value = float(selected_value)
                selected_grad = np.asarray(
                    selected_grad, dtype=np.float64).reshape(-1)
            except Exception as exc:
                reasons.append(
                    f"{selected_engine} final evaluation failed: {exc}")

        diagnostics["final_selected_backend_value"] = selected_value
        diagnostics["final_optimizer_value"] = float(result.fun)
        if _objective_is_invalid(result.fun):
            reasons.append("optimizer returned an invalid objective value")
        if _objective_is_invalid(selected_value):
            reasons.append("invalid objective value from selected backend")
        if (
                selected_grad.shape != final_params.shape
                or not np.all(np.isfinite(selected_grad))):
            reasons.append("final gradient is not finite")

        objective_abs_tol = max(1e-7, max(int(n_obs), 1) * 1e-8)
        objective_rel_tol = 1e-8
        diagnostics["final_optimizer_abs_tolerance"] = objective_abs_tol
        diagnostics["final_optimizer_rel_tolerance"] = objective_rel_tol
        if (
                np.isfinite(selected_value)
                and not _objective_is_invalid(result.fun)
                and not np.isclose(
                    selected_value,
                    float(result.fun),
                    rtol=objective_rel_tol,
                    atol=objective_abs_tol)):
            reasons.append("optimizer and selected-backend objectives disagree")

        projected_norm = np.inf
        gradient_tolerance = (
            self.final_gradient_tolerance
            if self.final_gradient_tolerance is not None
            else max(
                1e-2, 10.0 * float(optimizer_options.get("gtol", 1e-3)))
        )
        if (
                selected_grad.shape == final_params.shape
                and np.all(np.isfinite(selected_grad))
                and lower.shape == final_params.shape
                and upper.shape == final_params.shape):
            projected_norm = _projected_gradient_norm(
                final_params, selected_grad, lower, upper)
            if projected_norm > gradient_tolerance:
                reasons.append(
                    "projected gradient exceeds validation tolerance")
        diagnostics["final_projected_gradient_norm"] = projected_norm
        diagnostics["final_projected_gradient_tolerance"] = gradient_tolerance

        boundary_atol = 1e-10
        at_lower = np.isfinite(lower) & np.isclose(
            final_params, lower, rtol=0.0, atol=boundary_atol)
        at_upper = np.isfinite(upper) & np.isclose(
            final_params, upper, rtol=0.0, atol=boundary_atol)
        diagnostics["final_boundary_flags"] = tuple(
            bool(value) for value in (at_lower | at_upper))

        if final_params.size >= 3 and np.all(np.isfinite(final_params[:3])):
            kappa, _, nu = final_params[:3]
            dt = 1.0 / max(int(n_obs) - 1, 1)
            kappa_dt = kappa * dt
            rho = np.exp(-kappa_dt) if kappa_dt < 746.0 else 0.0
            stationary_std = (
                nu / np.sqrt(2.0 * kappa)
                if kappa > 0.0 and nu > 0.0 else np.nan)
            conditional_variance = (
                -np.expm1(-2.0 * kappa_dt)
                if kappa_dt >= 0.0 else np.nan)
            conditional_std = (
                stationary_std * np.sqrt(conditional_variance)
                if np.isfinite(stationary_std)
                and np.isfinite(conditional_variance)
                and conditional_variance >= 0.0 else np.nan)
            diagnostics.update({
                "final_kappa_dt": float(kappa_dt),
                "final_rho": float(rho),
                "final_stationary_std": float(stationary_std),
                "final_conditional_std": float(conditional_std),
            })
            if (
                    not np.isfinite(stationary_std)
                    or stationary_std <= 0.0
                    or not np.isfinite(conditional_std)
                    or conditional_std <= 0.0):
                reasons.append("final OU variance is degenerate")
            if (
                    rho <= self.final_rho_tolerance
                    or 1.0 - rho <= self.final_rho_tolerance):
                reasons.append("final one-step autocorrelation is degenerate")

            if initial_params.size >= 3 and np.all(
                    np.isfinite(initial_params[:3])):
                baseline = np.maximum(np.abs(initial_params[:3]), 1.0)
                growth = np.abs(final_params[:3]) / baseline
                diagnostics["final_parameter_growth"] = tuple(
                    float(value) for value in growth)
                diagnostics["final_parameter_growth_limit"] = (
                    self.final_growth_limit)
                if np.any(growth > self.final_growth_limit):
                    reasons.append(
                        "final OU parameters exceed initialization scale "
                        f"by more than {self.final_growth_limit:g}")

        validation_value = np.nan
        difference = np.nan
        tolerance = np.nan
        if validation_evaluator is not None and not reasons:
            try:
                validation_value = float(validation_evaluator(final_params))
                if _objective_is_invalid(validation_value):
                    reasons.append(
                        f"{validation_engine} validation returned an "
                        "invalid objective")
                else:
                    difference = abs(validation_value - selected_value)
                    tolerance = max(
                        1e-5,
                        max(int(n_obs), 1)
                        * self.final_validation_abs_per_obs,
                        self.final_validation_rel_tol * max(
                            abs(validation_value), abs(selected_value), 1.0),
                    )
                    if difference > tolerance:
                        reasons.append(
                            "selected and validation backends disagree")
            except Exception as exc:
                reasons.append(
                    f"{validation_engine} validation failed: {exc}")
        diagnostics.update({
            "final_validation_backend_value": validation_value,
            "final_backend_value_difference": difference,
            "final_backend_value_tolerance": tolerance,
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
                          verbose):
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

        scale = np.maximum(np.abs(joint0), 1.0)
        x0_scaled = joint0 / scale
        lower = np.full(3 + n_corr, -np.inf, dtype=np.float64)
        upper = np.full(3 + n_corr, np.inf, dtype=np.float64)
        lower[0] = 0.001 / scale[0]
        lower[2] = 0.001 / scale[2]
        bounds_scaled = Bounds(lower, upper)

        diagnostics = _new_backend_diagnostics()
        diagnostics.update({
            "initialization": initialization,
            "joint_static": True,
            "joint_optimizer": "python-lbfgsb",
            "correlation_parameterization_engine": "python",
            "correlation_gradient": "numerical",
            "cpp_correlation_derivatives": False,
            "analytical_grad_requested": bool(self.analytical_grad),
            "analytical_grad_used": bool(self.analytical_grad),
            "joint_gradient": (
                "hybrid" if self.analytical_grad else "numerical"),
            "ou_gradient": (
                "analytical" if self.analytical_grad else "numerical"),
            "correlation_fd_scheme": (
                "forward" if self.analytical_grad else "optimizer"),
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
            return value if np.isfinite(value) else fail_value

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
            except _cpp_scar_ou.CppUnsupported:
                value, grad, info = prepared_cache.neg_loglik_with_grad_info(
                    kappa_v, mu_v, nu_v, auto_config)
                corr_grad = None
                corr_kind = None
            except AttributeError:
                value, grad, corr_grad, info = (
                    _cpp_scar_ou.neg_loglik_with_grad_and_corr_info(
                        kappa_v, mu_v, nu_v, u, copula, auto_config))
                corr_kind = "full"
            _record_backend_diagnostics(diagnostics, info, "cpp")
            return (
                float(value),
                np.asarray(grad, dtype=np.float64),
                None if corr_grad is None else np.asarray(
                    corr_grad, dtype=np.float64),
                corr_kind,
            )

        def objective_scaled(x_scaled):
            joint = x_scaled * scale
            if not np.all(np.isfinite(joint)):
                return fail_value
            try:
                return evaluate_value(joint)
            except Exception as exc:
                if verbose:
                    print(f"  error at joint alpha={joint}: {exc}")
                return fail_value

        def correlation_fd_steps(joint):
            if "eps" in optimizer_options:
                return np.full(
                    n_corr, abs(float(optimizer_options["eps"])),
                    dtype=np.float64)
            rel_step = optimizer_options.get("finite_diff_rel_step")
            if rel_step is None:
                rel_step = np.sqrt(np.finfo(np.float64).eps)
            return (
                abs(float(rel_step))
                * np.maximum(1.0, np.abs(joint[3:]))
            )

        def objective_and_grad_scaled(x_scaled):
            joint = x_scaled * scale
            if not np.all(np.isfinite(joint)):
                return fail_value, np.zeros_like(x_scaled)
            diagnostics["hybrid_gradient_evaluations"] += 1
            try:
                value, ou_grad, corr_grad, corr_kind = (
                    evaluate_value_and_ou_grad(joint))
                if (
                        _objective_is_invalid(value)
                        or ou_grad.shape != (3,)
                        or not np.all(np.isfinite(ou_grad))):
                    return fail_value, np.zeros_like(x_scaled)

                grad = np.empty_like(joint)
                grad[:3] = ou_grad
                if corr_grad is not None:
                    if corr_kind == "directional":
                        if corr_grad.shape != (n_corr,):
                            return fail_value, np.zeros_like(x_scaled)
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
                    return value, grad * scale

                steps = correlation_fd_steps(joint)
                try:
                    for index, step in enumerate(steps):
                        trial = joint.copy()
                        trial[3 + index] += step
                        diagnostics["correlation_fd_evaluations"] += 1
                        trial_value = evaluate_value(trial)
                        if _objective_is_invalid(trial_value):
                            return fail_value, np.zeros_like(x_scaled)
                        grad[3 + index] = (trial_value - value) / step
                finally:
                    copula._set_corr_from_params(joint[3:])
                return value, grad * scale
            except Exception as exc:
                try:
                    copula._set_corr_from_params(joint[3:])
                except Exception:
                    pass
                if verbose:
                    print(f"  error at joint alpha={joint}: {exc}")
                return fail_value, np.zeros_like(x_scaled)

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
                bounds=bounds_scaled,
                options=scaled_options,
            )
        else:
            scaled_options = dict(optimizer_options)
            if 'eps' in scaled_options:
                scaled_options['eps'] = (
                    float(scaled_options['eps']) / scale)
            result = minimize(
                objective_scaled,
                x0_scaled,
                method='L-BFGS-B',
                bounds=bounds_scaled,
                options=scaled_options,
            )
        result.x = result.x * scale

        joint = result.x
        try:
            copula._set_corr_from_params(joint[3:])
        except Exception as exc:
            result.success = False
            result.message = (
                f"{result.message}; failed to set final correlation: {exc}")
            copula._set_corr_from_params(corr0)

        def selected_final_evaluator(values):
            value, gradient_scaled = objective_and_grad_scaled(values / scale)
            copula._set_corr_from_params(values[3:])
            return value, np.asarray(gradient_scaled) / scale

        def validate_correlation():
            reasons = []
            raw = np.asarray(joint[3:], dtype=np.float64)
            if raw.size != n_corr or not np.all(np.isfinite(raw)):
                reasons.append("final correlation parameters are not finite")
                return reasons
            matrix = np.asarray(copula.R, dtype=np.float64)
            if (
                    matrix.shape != (copula.d, copula.d)
                    or not np.all(np.isfinite(matrix))):
                reasons.append("final correlation matrix is invalid")
                return reasons
            if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-10):
                reasons.append("final correlation matrix is not symmetric")
            if not np.allclose(
                    np.diag(matrix), 1.0, rtol=0.0, atol=1e-10):
                reasons.append("final correlation diagonal is not one")
            try:
                if np.min(np.linalg.eigvalsh(matrix)) <= 0.0:
                    reasons.append(
                        "final correlation matrix is not positive definite")
            except np.linalg.LinAlgError:
                reasons.append(
                    "final correlation eigenvalue check failed")
            if (
                    not np.all(np.isfinite(copula._L_inv))
                    or not np.isfinite(copula._log_det)):
                reasons.append(
                    "final correlation factorization is not finite")
            return reasons

        validation_diagnostics = self._validate_final_fit(
            result=result,
            final_params=joint,
            initial_params=joint0,
            lower=np.concatenate((
                np.array([0.001, -np.inf, 0.001]),
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
            **kwargs) -> LatentResult:
        """Fit SCAR-TM-OU model.

        Parameters
        ----------
        copula : CopulaProtocol
        u : (T, 2) pseudo-observations
        alpha0 : (3,) or (3 + n_corr,)
            Initial ``[kappa, mu, nu]`` with model correlation defaults, a
            full joint vector, or None for automatic initialization.
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
        corr_num_params = getattr(copula, "_corr_num_params", None)
        n_corr = int(corr_num_params()) if callable(corr_num_params) else 0
        if n_corr:
            return self._fit_joint_static(
                copula, u, alpha0, optimizer_options, verbose)
        # ── Initial point ─────────────────────────────────────────
        alpha0, initialization = _resolve_initial_point(
            copula,
            u,
            self.config,
            self.smart_init,
            verbose,
            alpha0,
        )
        alpha0 = np.asarray(alpha0, dtype=np.float64)

        self._uses_cpp(copula)
        selected_engine = "cpp"
        diagnostics = _new_backend_diagnostics()
        diagnostics["adaptive_spectral_basis_order"] = (
            self.spectral_basis_order == "auto")
        diagnostics["auto_spectral_basis_order"] = (
            self.spectral_basis_order == "auto")
        diagnostics.update({
            "initialization": initialization,
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
                    return fail_value, np.zeros(3)
                kappa_v, mu_v, nu_v = alpha
                try:
                    auto_config = _auto_config_for(kappa_v)
                    val, grad, info = (
                        prepared_cache.neg_loglik_with_grad_info(
                            kappa_v, mu_v, nu_v, auto_config))
                    _record_backend_diagnostics(diagnostics, info, "cpp")
                    return val, grad * scale  # chain rule
                except Exception as e:
                    if verbose:
                        print(f"  error at alpha={alpha}: {e}")
                    return fail_value, np.zeros(3)

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

            def objective_scaled(x_scaled):
                alpha = x_scaled * scale
                if np.isnan(np.sum(alpha)):
                    return fail_value
                kappa_v, mu_v, nu_v = alpha
                try:
                    auto_config = _auto_config_for(kappa_v)
                    val, info = prepared_cache.neg_loglik_info(
                        kappa_v, mu_v, nu_v, auto_config)
                    _record_backend_diagnostics(diagnostics, info, "cpp")
                    return val
                except Exception as e:
                    if verbose:
                        print(f"  error at alpha={alpha}: {e}")
                    return fail_value

            if verbose:
                print(
                    f"Fitting SCAR-TM-OU (C++, numerical gradient), "
                    f"alpha0={alpha0}")

            scaled_options = dict(optimizer_options)
            if 'eps' in scaled_options:
                scaled_options['eps'] = float(scaled_options['eps']) / scale

            result = minimize(
                objective_scaled, x0_scaled,
                method='L-BFGS-B',
                bounds=bounds_scaled,
                options=scaled_options,
            )
            result.x = result.x * scale

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
            lower=np.array([0.001, -np.inf, 0.001]),
            upper=np.array([np.inf, np.inf, np.inf]),
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
        capabilities = get_copula_capabilities(copula)
        if capabilities is not None and not capabilities.supports_pair_ops:
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
            except _cpp_scar_ou.CppUnsupported:
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

    def objective(self, copula, u: np.ndarray,
                  alpha: np.ndarray, **kwargs) -> float:
        """Minus log-likelihood: TM integrated -logL(kappa, mu, nu)."""
        try:
            cfg = self._auto_config(kappa=alpha[0], n_obs=len(u))
            self._uses_cpp(copula)
            return _cpp_scar_ou.neg_loglik(
                alpha[0], alpha[1], alpha[2], u, copula,
                cfg)
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
        d = copula_dimension(copula, u)
        return sample_predictive(copula, n, r, rng=rng, d=d)

    def predict(self, copula, u, result, n, rng=None, **kwargs):
        """Mixture sampling from posterior p(x_T | data).

        Uses transfer matrix forward pass to compute the posterior
        distribution of x_T, then samples r from it.
        """
        return predict_from_strategy(
            self, copula, u, result, n, rng=rng, **kwargs)

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

        z_grid = np.asarray(state.z_grid, dtype=np.float64)
        prob = np.asarray(state.prob, dtype=np.float64)
        r_grid = copula.transform(z_grid)
        density = copula_native.pdf_parameter_grid(copula, u, r_grid)[0]
        log_w = np.log(np.maximum(density, np.finfo(np.float64).tiny))
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
