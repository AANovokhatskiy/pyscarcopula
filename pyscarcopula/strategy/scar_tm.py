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

from __future__ import annotations

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
    auto_neg_loglik_info,
    auto_neg_loglik_with_grad_info,
)
from pyscarcopula.numerical._arrays import as_float64_array
from pyscarcopula.numerical._transition_methods import (
    normalize_ou_transition_method,
)
from pyscarcopula.strategy.predict_helpers import sample_predictive
from pyscarcopula.numerical import _cpp_scar_ou


_INVALID_OBJECTIVE_THRESHOLD = 1e9

_DIAGNOSTIC_COUNTERS = (
    "objective_evaluations",
    "python_evaluations",
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


def _normalize_backend(value):
    backend = str(value).lower()
    if backend not in {"python", "cpp", "auto"}:
        raise ValueError("backend must be 'python', 'cpp', or 'auto'")
    return backend


def _normalize_spectral_basis_order(value):
    if isinstance(value, str):
        order = value.lower()
        if order in {"adaptive", "auto"}:
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


def _new_backend_diagnostics(requested_backend: str, selected_engine: str) -> dict:
    diagnostics = {name: 0 for name in _DIAGNOSTIC_COUNTERS}
    diagnostics["requested_backend"] = requested_backend
    diagnostics["selected_engine"] = selected_engine
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
    spectral_basis_order : int or {'auto', 'adaptive'}
        Number of Hermite basis functions in the spectral likelihood.  The
        default ``'auto'`` policy selects 128, 96, 64, or 32 from the current
        optimizer evaluation's ``kappa / (T - 1)``. ``'adaptive'`` is accepted
        as a backwards-compatible alias.
    spectral_quad_order : int or None
        Gauss-Hermite quadrature order for spectral multiplication.
    backend : {'auto', 'python', 'cpp'}
        Numerical implementation.  'auto' uses C++ for supported SCAR-TM-OU
        copula/transform combinations and otherwise uses Python.  'cpp'
        requires the compiled extension and raises for unsupported models.
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
                 auto_small_kdt: float = 1e-2,
                 spectral_basis_order: int | str = "auto",
                 spectral_quad_order: int | None = None,
                 backend: str = 'auto',
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
        self.spectral_basis_order = _normalize_spectral_basis_order(
            spectral_basis_order)
        self.spectral_quad_order = spectral_quad_order
        self.backend = _normalize_backend(backend)
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

    def _uses_cpp(self, copula, require=False):
        if self.backend == 'python':
            return False
        if _cpp_scar_ou.supported(copula):
            return True
        if require or self.backend == 'cpp':
            _cpp_scar_ou.ensure_supported(copula)
            if not _cpp_scar_ou.available():
                raise _cpp_scar_ou.CppUnavailable(
                    "pyscarcopula C++ extension is not available")
        return False

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
        use_cpp = self._uses_cpp(copula, require=self.backend == 'cpp')
        selected_engine = "cpp" if use_cpp else "python"
        diagnostics = _new_backend_diagnostics(self.backend, selected_engine)
        diagnostics["adaptive_spectral_basis_order"] = (
            self.spectral_basis_order == "auto")
        diagnostics["auto_spectral_basis_order"] = (
            self.spectral_basis_order == "auto")

        def _auto_config_for(kappa_v):
            return self._auto_config(kappa=kappa_v, n_obs=len(u))

        def _manual_info(kappa_v):
            n_obs = len(u)
            kdt = float(kappa_v) if n_obs <= 1 else float(kappa_v) / (n_obs - 1)
            return {
                "backend": self._grid_transition_method(),
                "transition_method": self.transition_method,
                "kappa_dt": kdt,
                "n_obs": int(n_obs),
                "basis_order": self._spectral_basis_order_for(kappa_v, n_obs),
            }

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
                    auto_config = _auto_config_for(kappa_v)
                    if use_cpp:
                        val, grad, info = _cpp_scar_ou.neg_loglik_with_grad_info(
                            kappa_v, mu_v, nu_v, u, copula, auto_config)
                        _record_backend_diagnostics(diagnostics, info, "cpp")
                    elif self._uses_dispatcher():
                        val, grad, info = auto_neg_loglik_with_grad_info(
                            kappa_v, mu_v, nu_v, u, copula, auto_config)
                        _record_backend_diagnostics(diagnostics, info, "python")
                    else:
                        val, grad = tm_loglik_with_grad(
                            kappa_v, mu_v, nu_v, u, copula, K, grid_range,
                            grid_method, adaptive, pts_per_sigma, **tm_kwargs)
                        _record_backend_diagnostics(
                            diagnostics, _manual_info(kappa_v), "python")
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
                    return 1e10
                kappa_v, mu_v, nu_v = alpha
                try:
                    auto_config = _auto_config_for(kappa_v)
                    if use_cpp:
                        val, info = _cpp_scar_ou.neg_loglik_info(
                            kappa_v, mu_v, nu_v, u, copula, auto_config)
                        _record_backend_diagnostics(diagnostics, info, "cpp")
                        return val
                    if self._uses_dispatcher():
                        val, info = auto_neg_loglik_info(
                            kappa_v, mu_v, nu_v, u, copula, auto_config)
                        _record_backend_diagnostics(diagnostics, info, "python")
                        return val
                    val = tm_loglik(
                        kappa_v, mu_v, nu_v, u, copula, K, grid_range,
                        grid_method, adaptive, pts_per_sigma, **tm_kwargs)
                    _record_backend_diagnostics(
                        diagnostics, _manual_info(kappa_v), "python")
                    return val
                except Exception as e:
                    if verbose:
                        print(f"  error at alpha={alpha}: {e}")
                    return 1e10

            if verbose:
                engine = "C++" if use_cpp else "Python"
                print(
                    f"Fitting SCAR-TM-OU ({engine}, numerical gradient), "
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
            spectral_basis_order=self.spectral_basis_order,
            spectral_quad_order=self.spectral_quad_order,
            backend=self.backend,
            diagnostics=diagnostics,
        )

    def log_likelihood(self, copula, u: np.ndarray,
                       result: LatentResult) -> float:
        """Evaluate TM log-likelihood at fitted parameters."""
        p = result.params
        cfg = self._auto_config(kappa=p.kappa, n_obs=len(u))
        if self._uses_cpp(copula, require=self.backend == 'cpp'):
            value, _ = _cpp_scar_ou.loglik(
                p.kappa, p.mu, p.nu, u, copula, cfg)
            return value
        if self._uses_dispatcher():
            return auto_loglik(
                p.kappa, p.mu, p.nu, u, copula, cfg)
        neg_ll = tm_loglik(
            p.kappa, p.mu, p.nu, u, copula,
            self.K, self.grid_range, self.grid_method,
            self.adaptive, self.pts_per_sigma, **self._tm_kwargs())
        return -neg_ll

    def predictive_mean(self, copula, u: np.ndarray,
                        result: LatentResult) -> np.ndarray:
        """E[Psi(x_k) | u_{1:k-1}] via TM forward pass."""
        p = result.params
        if self._uses_cpp(copula, require=self.backend == 'cpp'):
            return _cpp_scar_ou.predictive_mean(
                p.kappa, p.mu, p.nu, u, copula,
                self._auto_config(
                    self._grid_transition_method(),
                    kappa=p.kappa,
                    n_obs=len(u),
                ))
        return tm_forward_predictive_mean(
            p.kappa, p.mu, p.nu, u, copula,
            self.K, self.grid_range, self.grid_method,
            self.adaptive, self.pts_per_sigma, **self._grid_tm_kwargs())

    def rosenblatt_e2(self, copula, u: np.ndarray,
                      result: LatentResult) -> np.ndarray:
        """Mixture Rosenblatt: e2 = E[h(u2, u1; Psi(x_k)) | u_{1:k-1}]."""
        p = result.params
        if self._uses_cpp(copula, require=self.backend == 'cpp'):
            return _cpp_scar_ou.mixture_h(
                p.kappa, p.mu, p.nu, u, copula,
                self._auto_config(
                    self._grid_transition_method(),
                    kappa=p.kappa,
                    n_obs=len(u),
                ))
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
        if self._uses_cpp(copula, require=self.backend == 'cpp'):
            cfg = self._auto_config(
                self._grid_transition_method(),
                kappa=p.kappa,
                n_obs=len(u),
            )
            h_mix = _cpp_scar_ou.mixture_h(
                p.kappa, p.mu, p.nu, u, copula,
                cfg)
            if state_cache is not None:
                if current_cache_key is not None:
                    state_cache[current_cache_key] = _cpp_scar_ou.state_distribution(
                        p.kappa, p.mu, p.nu, u, copula, cfg, horizon='current')
                if next_cache_key is not None:
                    state_cache[next_cache_key] = _cpp_scar_ou.state_distribution(
                        p.kappa, p.mu, p.nu, u, copula, cfg, horizon='next')
            return h_mix
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
            cfg = self._auto_config(kappa=alpha[0], n_obs=len(u))
            if self._uses_cpp(copula, require=self.backend == 'cpp'):
                return _cpp_scar_ou.neg_loglik(
                    alpha[0], alpha[1], alpha[2], u, copula,
                    cfg)
            if self._uses_dispatcher():
                return auto_neg_loglik(
                    alpha[0], alpha[1], alpha[2], u, copula,
                    cfg)
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
            horizon = kwargs.get('horizon', 'next')
            if self._uses_cpp(copula, require=self.backend == 'cpp'):
                cached = _cpp_scar_ou.state_distribution(
                    p.kappa, p.mu, p.nu, u, copula,
                    self._auto_config(
                        self._grid_transition_method(),
                        kappa=p.kappa,
                        n_obs=len(u),
                    ),
                    horizon=horizon)
            else:
                from pyscarcopula.numerical import predictive_tm
                cached = predictive_tm.tm_state_distribution(
                    p.kappa, p.mu, p.nu, u, copula,
                    K=self.K,
                    grid_range=self.grid_range,
                    grid_method=self.grid_method,
                    adaptive=self.adaptive,
                    pts_per_sigma=self.pts_per_sigma,
                    **self._grid_tm_kwargs(),
                    horizon=horizon)
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
