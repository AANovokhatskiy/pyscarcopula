"""GAS estimation strategy backed by the native numerical evaluator."""

from copy import copy
from functools import partial

import numpy as np
from scipy.optimize import Bounds, minimize

from pyscarcopula._types import (
    DEFAULT_CONFIG,
    GASResult,
    NumericalConfig,
    PredictiveState,
    gas_params,
)
from pyscarcopula._native import gas as _cpp_gas
from pyscarcopula._native import model_policy
from pyscarcopula._native.threads import validate_n_threads
from pyscarcopula.numerical._arrays import (
    as_float64_array,
    as_float64_scalar,
    validate_float64_allocation,
    validate_positive_int,
)
from pyscarcopula.numerical.gas_filter import (
    gas_filter,
    gas_loglik,
    gas_mixture_h,
    gas_mixture_h_pair,
    gas_negloglik,
    gas_predict_param,
)
from pyscarcopula.strategy._base import (
    copula_dimension,
    is_multivariate_copula,
    lbfgsb_options,
    lbfgsb_overrides,
    register_strategy,
    reject_unknown_strategy_kwargs,
    reject_unknown_operation_kwargs,
)
from pyscarcopula.strategy.predict_helpers import (
    predictive_params_from_state,
    predict_from_strategy,
    sample_predictive,
)


_DEFAULT_REFINEMENT_FTOL = 1e-12
_DEFAULT_REFINEMENT_MIN_LOGL_GAIN = 0.0
_DEFAULT_OPTIMIZER_GRADIENT_EPS = 1e-8
_MATERIAL_LOGL_GAIN = 1e-3


def _native_optimizer_gradient_config(options):
    """Split native finite-difference controls from SciPy options."""
    scipy_options = dict(options)
    relative_step = scipy_options.pop("finite_diff_rel_step", None)
    absolute_step = scipy_options.pop("eps", None)
    if relative_step is not None:
        return scipy_options, float(relative_step), True
    return (
        scipy_options,
        float(
            _DEFAULT_OPTIMIZER_GRADIENT_EPS
            if absolute_step is None else absolute_step),
        False,
    )


def _minimize_gas_objective(objective, initial, *, bounds, options):
    """Keep maxfun and nfev in scalar-objective units across native FD."""
    evaluations_per_point = int(getattr(
        objective, "evaluations_per_point", int(np.size(initial)) + 1))
    native_options = dict(options)
    if "maxfun" in native_options:
        native_options["maxfun"] = (
            int(native_options["maxfun"]) // evaluations_per_point)
    mean_parameterization = bool(getattr(objective, "gas_mean_parameterization", False))
    if mean_parameterization:
        # Recovery uses the stationary state mu=omega/(1-beta). This avoids
        # the near-collinearity of intercept and persistence close to beta=1.
        # Objective scaling only conditions line search; rescale gtol too so
        # its units remain those of the summed objective in these coordinates.
        scale = float(getattr(objective, "objective_scale", 1.0))
        start = _cpp_gas.optimizer_coordinates(
            initial, objective=float(native_options.get("gtol", 1e-3)),
            objective_scale=scale)
        native_options["gtol"] = start["objective"]

        def transformed(values):
            physical = _cpp_gas.optimizer_coordinates(values, to_optimizer=False)
            value, gradient = objective(physical["parameters"])
            mapped = _cpp_gas.optimizer_coordinates(
                physical["parameters"], objective=value, gradient=gradient,
                objective_scale=scale)
            return mapped["objective"], mapped["gradient"]

        if getattr(objective, "derivative_free", False):
            derivative_free_options = {
                "maxfev": native_options.get("maxfun", 15000),
                "maxiter": native_options.get("maxiter", 15000),
                "ftol": native_options.get("ftol", _DEFAULT_REFINEMENT_FTOL),
            }
            result = minimize(lambda values: transformed(values)[0],
                              start["parameters"], method="Powell", bounds=bounds,
                              options=derivative_free_options)
            result.fun, result.jac = transformed(result.x)
            result.nfev += 1
        else:
            result = minimize(transformed, start["parameters"], method="L-BFGS-B", jac=True,
                              bounds=bounds, options=native_options)
        physical = _cpp_gas.optimizer_coordinates(
            result.x, objective=result.fun, gradient=result.jac,
            objective_scale=scale, to_optimizer=False)
        result.x = physical["parameters"]
        result.fun = physical["objective"]
        result.jac = physical["gradient"]
    else:
        result = minimize(
            objective, initial, method="L-BFGS-B", jac=True,
            bounds=bounds, options=native_options,
        )
    result.nfev = int(result.nfev) * evaluations_per_point
    return result


def _automatic_gas_start(copula, u, config, initial_mle_result=None):
    """Build the standard GAS start, reusing a static fit when available."""
    mle_result = initial_mle_result
    if mle_result is None:
        from pyscarcopula.strategy.mle import MLEStrategy
        mle_result = MLEStrategy(config=config).fit(copula, u)
    mu_mle = float(np.atleast_1d(
        copula.inv_transform(np.atleast_1d(mle_result.copula_param))
    )[0])
    return model_policy.gas_default_initial_point(mu_mle)


def _fit_gas_starts(objective, initial, *, bounds, options, automatic, refine,
                    recovery_objectives=(), validation_objective=None):
    """Try the nested static model and retain the best finite evaluation.

    A successful relative-function stopping test does not imply a good GAS
    fit: the score recursion can amplify a tiny parameter change. Starting
    from gamma=0 supplies a well-behaved, exactly constant parameter path.
    Explicit user starts retain their single-start semantics.
    """
    starts = [np.asarray(initial, dtype=np.float64).copy()]
    if automatic:
        static = starts[0].copy()
        static[1] = 0.0
        starts.append(static)
    candidates = []
    traces = []
    total_nfev = 0
    best_evaluation = None

    def tracked(values, provider=objective):
        nonlocal best_evaluation
        value, gradient = provider(values)
        if (np.isfinite(value) and np.all(np.isfinite(gradient))
                and (best_evaluation is None or value < best_evaluation[0])):
            best_evaluation = (
                float(value), np.asarray(values).copy(),
                np.asarray(gradient).copy())
        return value, gradient

    def run(start, run_options, label, provider=objective):
        nonlocal total_nfev
        initial_objective = None

        def stage_objective(values):
            nonlocal initial_objective
            evaluated = tracked(values, provider)
            if initial_objective is None:
                initial_objective = float(evaluated[0])
            return evaluated

        stage_objective.evaluations_per_point = getattr(
            provider, "evaluations_per_point", np.size(start) + 1)
        stage_objective.gas_mean_parameterization = getattr(
            provider, "gas_mean_parameterization", False)
        stage_objective.objective_scale = getattr(provider, "objective_scale", 1.0)
        stage_objective.derivative_free = getattr(provider, "derivative_free", False)
        result = _minimize_gas_objective(
            stage_objective, start, bounds=bounds, options=run_options)
        total_nfev += int(result.nfev)
        traces.append({
            "stage": label, "initial_params": start.copy(),
            "final_params": np.asarray(result.x).copy(),
            "objective": float(result.fun),
            "initial_objective": initial_objective,
            "optimizer_success": bool(result.success),
            "optimizer_message": str(result.message),
            "nfev": int(result.nfev), "nit": int(getattr(result, "nit", 0)),
        })
        if np.isfinite(result.fun):
            candidates.append(result)
        return result

    for index, start in enumerate(starts):
        run(start, options, "standard" if index == 0 else "nested_static")
    if not candidates:
        raise FloatingPointError("no finite GAS optimization result")
    selected = min(candidates, key=lambda item: (float(item.fun), not bool(item.success)))
    if refine and float(options["ftol"]) > _DEFAULT_REFINEMENT_FTOL:
        refinement_options = dict(options, ftol=_DEFAULT_REFINEMENT_FTOL)
        run(np.asarray(selected.x).copy(), refinement_options, "refinement")
        selected = min(candidates, key=lambda item: (float(item.fun), not bool(item.success)))
    verification_nfev = 0
    stationarity = None

    def verify(point):
        nonlocal verification_nfev
        providers = [validation_objective]
        finer = getattr(validation_objective, "finer_provider", None)
        if finer is not None:
            providers.append(finer)
        providers.extend(getattr(validation_objective, "refinement_providers", ()))
        used_providers = []
        gradients = []
        raw_norms = []
        values = []
        gradient_scale = float(getattr(validation_objective, "objective_scale", 1.0))
        coordinates = "omega_gamma_beta"
        def failed_validation():
            return {"passed": False, "projected_gradient_inf_norm": float("inf"),
                    "gradient_step_discrepancy": float("inf"),
                    "gtol": float(options.get("gtol", 1e-3)),
                    "reason": "native_gradient_unresolved"}

        tolerance = float(options.get("gtol", 1e-3))
        for provider in providers:
            used_providers.append(provider)
            verification_nfev += getattr(
                provider, "evaluations_per_point", 2 * np.size(point) + 1)
            try:
                value, gradient = provider(point)
                if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
                    return failed_validation()
                gradient = np.asarray(gradient, dtype=float).copy()
                raw_norms.append(float(np.max(np.abs(gradient))))
                if getattr(validation_objective, "gas_mean_parameterization", False):
                    mapped = _cpp_gas.optimizer_coordinates(
                        point, objective=value, gradient=gradient,
                        objective_scale=gradient_scale)
                    gradient = mapped["gradient"]
                    coordinates = "stationary_mean_gamma_beta"
            except FloatingPointError:
                return failed_validation()
            gradient[(point <= bounds.lb) & (gradient > 0)] = 0.0
            gradient[(point >= bounds.ub) & (gradient < 0)] = 0.0
            gradients.append(gradient)
            values.append(value)
            if len(gradients) >= 2:
                pair_norm = max(float(np.max(np.abs(item))) for item in gradients[-2:])
                pair_discrepancy = float(np.max(np.abs(gradients[-2] - gradients[-1])))
                if pair_norm <= tolerance and pair_discrepancy <= tolerance:
                    break
                # Refine only while FD uncertainty straddles the acceptance
                # threshold. A resolved nonstationary gradient needs recovery,
                # not a search for one convenient small difference step.
                if pair_norm - pair_discrepancy > tolerance:
                    break
        all_norms = [float(np.max(np.abs(gradient))) for gradient in gradients]
        norms = all_norms[-2:]
        norm = max(norms)
        uncertainty = (float(np.max(np.abs(gradients[-2] - gradients[-1])))
                       if len(gradients) >= 2 else 0.0)
        passed = bool(np.all(np.isfinite(values)) and np.isfinite(norm)
                      and norm <= tolerance and uncertainty <= tolerance)
        return {"passed": passed, "projected_gradient_inf_norm": norm,
                "projected_gradient_norms": norms,
                "gradient_step_discrepancy": uncertainty,
                "gradient_steps": [getattr(provider, "difference_step", None)
                                   for provider in used_providers[-2:]],
                "all_gradient_steps": [getattr(provider, "difference_step", None)
                                       for provider in used_providers],
                "all_projected_gradient_norms": all_norms,
                "refinement_count": max(0, len(used_providers) - 2),
                "gtol": tolerance,
                "gradient_kind": "native_three_point",
                "coordinates": coordinates,
                "raw_gradient_inf_norm": max(raw_norms),
                "objective_scale": gradient_scale,
                "objective_units": "mean_negative_log_likelihood"}

    if validation_objective is not None:
        point = (best_evaluation[1] if best_evaluation is not None
                 and best_evaluation[0] < selected.fun - _MATERIAL_LOGL_GAIN else selected.x)
        stationarity = verify(point)
        for index, provider in enumerate(recovery_objectives):
            if stationarity["passed"]:
                break
            recovery_options = dict(options, ftol=_DEFAULT_REFINEMENT_FTOL)
            start = np.asarray(getattr(provider, "restart_point", point)).copy()
            recovered = run(start, recovery_options, f"recovery_{index + 1}", provider)
            selected = min(candidates, key=lambda item: (float(item.fun), not bool(item.success)))
            if (bool(recovered.success)
                    and float(recovered.fun) <= float(selected.fun) + _MATERIAL_LOGL_GAIN):
                recovered_check = verify(np.asarray(recovered.x))
                if recovered_check["passed"]:
                    selected = recovered
            point = (best_evaluation[1] if best_evaluation is not None
                     and best_evaluation[0] < selected.fun - _MATERIAL_LOGL_GAIN else selected.x)
            stationarity = verify(point)
    result = copy(selected)
    result.raw_optimizer_success = bool(selected.success)
    result.raw_optimizer_message = str(selected.message)
    retained_trial = bool(
        best_evaluation is not None
        and best_evaluation[0] < float(result.fun) - _MATERIAL_LOGL_GAIN)
    if retained_trial:
        result.fun, result.x, result.jac = best_evaluation
        result.success = False
        result.message = (
            f"{result.message}; retained a better finite evaluation; "
            "convergence at this point was not established")
    if validation_objective is not None:
        stationarity = verify(np.asarray(result.x))
        if not stationarity["passed"]:
            result.success = False
            result.message = (
                f"{result.message}; GAS stationarity was not established "
                f"(projected gradient {stationarity['projected_gradient_inf_norm']:.6g})")
    result.nfev = total_nfev + verification_nfev
    diagnostics = {
        "optimizer_stages": traces,
        "retained_best_trial": retained_trial,
        "automatic_multistart": automatic,
        "stationarity_validation": stationarity,
        "verification_nfev": verification_nfev,
        "stationary_selection_loglik_loss": (
            max(0.0, float(result.fun) - best_evaluation[0])
            if best_evaluation is not None else 0.0),
    }
    if automatic:
        diagnostics["initial_static_log_likelihood"] = -traces[1]["initial_objective"]
    refinement_index = next((i for i, stage in enumerate(traces)
                             if stage["stage"] == "refinement"), None)
    if refinement_index is not None:
        previous = min(traces[:refinement_index], key=lambda item: item["objective"])
        last = traces[refinement_index]
        diagnostics["optimizer_refinement"] = {
            "enabled": True,
            "first_ftol": float(options["ftol"]),
            "refinement_ftol": _DEFAULT_REFINEMENT_FTOL,
            "minimum_loglik_gain": _DEFAULT_REFINEMENT_MIN_LOGL_GAIN,
            "first_objective": previous["objective"],
            "refined_objective": last["objective"],
            "loglik_gain": previous["objective"] - last["objective"],
            "first_success": previous["optimizer_success"],
            "refined_success": last["optimizer_success"],
            "first_nfev": sum(stage["nfev"] for stage in traces[:refinement_index]),
            "refined_nfev": last["nfev"],
            "selected_stage": (
                "refined" if last["objective"] < previous["objective"]
                else "first"),
        }
    return result, diagnostics


@register_strategy("GAS")
class GASStrategy:
    """GAS estimation strategy.

    Parameters
    ----------
    config : NumericalConfig
    scaling : {'unit', 'fisher'} or None
        Explicit score scaling override. None uses ``unit`` when fitting and
        inherits the fitted result's scaling for subsequent operations.

    Notes
    -----
    GAS numerical operations require the compiled extension. There is no
    Python numerical backend or silent fallback. The copula score driving the
    recursion is computed natively. L-BFGS-B receives the objective and its
    optimizer gradient from one C++ entry point; any required numerical
    differentiation remains inside the native evaluator.

    Automatic initialization tries both the standard score-driven start and
    a nested static start (gamma=0), then refines the best result when ``ftol``
    is omitted. Each run has its own optimizer budget. Passing ``gamma0``
    selects a single start. The default optimizer difference step is 1e-8;
    ``eps`` and ``finite_diff_rel_step`` explicitly override it.

    ``success`` requires optimizer convergence and consistent finite
    likelihoods at least as high as the nested static path. Fixed-correlation
    fits additionally validate a native three-point gradient of the mean
    negative log-likelihood in (stationary mean, gamma, beta) coordinates
    against ``gtol``; diagnostics state its scale and coordinate system.
    Default validation balances three-point truncation and roundoff with
    h=cbrt(machine epsilon) and also requires the half-step estimate to agree.
    The optimizer itself retains the summed-objective tolerance. Automatic
    fits that fail this check retry the existing starts at a second native
    difference step, then polish in stationary-mean coordinates. It is not a
    certificate of global optimality. Diagnostics retain each optimizer
    stage and its raw convergence status, including the projected gradient.
    """

    _strict_keyword_contract = True
    _constructor_keyword_aliases = frozenset({"backend"})
    # Shared prediction/vine adapters pass the context to both state steps.
    _prediction_context_keywords = frozenset({
        "given", "horizon", "predictive_r_mode", "n_threads",
        "memory_budget_bytes", "state_cache", "cache_key", "posterior_cache",
    })
    _operation_keyword_aliases = {
        "objective": frozenset({"score_eps"}),
        "sample": frozenset({"given", "n_threads", "memory_budget_bytes"}),
        "predict": frozenset({
            "given", "horizon", "predictive_r_mode", "n_threads",
            "memory_budget_bytes",
        }),
        "predictive_params": _prediction_context_keywords,
        "predictive_state": _prediction_context_keywords,
        "sample_params": _prediction_context_keywords,
    }

    def __init__(
        self,
        config: NumericalConfig | None = None,
        scaling: str | None = None,
        **kwargs,
    ):
        if "backend" in kwargs:
            raise TypeError(
                "GAS backend selection was removed; native execution is "
                "always used")
        reject_unknown_strategy_kwargs("GAS", kwargs)
        self.config = config or DEFAULT_CONFIG
        self._explicit_scaling = scaling is not None
        self.scaling = _cpp_gas._scaling_name(
            'unit' if scaling is None else scaling)

    def __setstate__(self, state):
        """Restore JSON/pickle state without changing legacy scaling semantics."""
        self.__dict__.update(state)
        # Before explicit overrides, post-fit operations used result.scaling.
        self._explicit_scaling = state.get("_explicit_scaling", False)

    def _result_scaling(self, result: GASResult) -> str:
        """Use an explicit constructor override, otherwise inherit the fit."""
        return (
            self.scaling if self._explicit_scaling
            else _cpp_gas._scaling_name(result.scaling)
        )

    def _score_eps(self, result: GASResult | None = None) -> float:
        if result is None:
            return float(self.config.gas_score_eps)
        return float(getattr(result, "score_eps", self.config.gas_score_eps))

    def _optimizer_config(self, copula):
        config_name = getattr(copula, "_gas_optimizer_config", None)
        if config_name is not None:
            return getattr(self.config, config_name)
        return self.config.gas_optimizer

    def _ensure_correlation_initialized(self, copula, u):
        ensure = getattr(copula, "_ensure_corr_initialized", None)
        if callable(ensure):
            ensure(u)

    def _correlation_diagnostics(self, copula) -> dict:
        diagnostics = {}
        count_diagnostics = getattr(copula, "_corr_count_diagnostics", None)
        if callable(count_diagnostics):
            diagnostics.update(count_diagnostics())
        preprocessing_diagnostics = getattr(
            copula, "correlation_preprocessing_diagnostics", None)
        if callable(preprocessing_diagnostics):
            diagnostics.update(preprocessing_diagnostics())
        corr_params = getattr(copula, "corr_params", None)
        if callable(corr_params):
            diagnostics["corr_params_raw"] = corr_params()
        corr_alpha = getattr(copula, "corr_alpha", None)
        if callable(corr_alpha):
            diagnostics["corr_alpha"] = corr_alpha()
        if getattr(copula, "_corr_mode", None) != "factor":
            R = getattr(copula, "R", None)
            if R is not None:
                diagnostics["corr_matrix"] = R
        return diagnostics

    def _build_result(
        self,
        copula,
        u,
        result,
        gas_values,
        score_eps,
        gamma_bound,
        beta_bound,
        *,
        parameter_count=None,
        diagnostics=None,
    ):
        gas_values = np.asarray(gas_values, dtype=np.float64).reshape(-1)
        params = gas_params(
            omega=gas_values[0],
            gamma=gas_values[1],
            beta=gas_values[2],
            gamma_bound=gamma_bound,
            beta_bound=beta_bound,
        )

        success = bool(result.success)
        message = str(result.message)
        final_log_likelihood = gas_loglik(
            gas_values[0],
            gas_values[1],
            gas_values[2],
            u,
            copula,
            self.scaling,
            score_eps,
        )
        if not np.isfinite(final_log_likelihood):
            raise FloatingPointError(
                "final GAS log-likelihood is not finite")
        # A constant path with the same intercept and persistence is nested
        # in GAS. Reject false optimizer success below this feasible model.
        static_log_likelihood = gas_loglik(
            gas_values[0], 0.0, gas_values[2], u, copula,
            self.scaling, score_eps)
        static_baseline = max(
            static_log_likelihood,
            (diagnostics or {}).get("initial_static_log_likelihood", -np.inf))
        objective = float(getattr(result, "fun", -final_log_likelihood))
        objective_discrepancy = final_log_likelihood + objective
        consistent = abs(objective_discrepancy) <= 1e-6
        above_static = final_log_likelihood >= static_baseline - _MATERIAL_LOGL_GAIN
        if not consistent or not above_static:
            success = False
            reason = (
                "optimizer/report likelihood mismatch" if not consistent
                else "likelihood below the nested static model")
            message = f"{message}; GAS validation failed: {reason}"
        r_last = gas_predict_param(
            gas_values[0],
            gas_values[1],
            gas_values[2],
            u,
            copula,
            self.scaling,
            score_eps,
        )

        result_diagnostics = {
            "n_threads": self.config.n_threads,
            "model_score": "native",
            "optimizer_gradient": "native",
            "gradient_kind": "native_finite_difference",
            "setup_derivative": "native_objective_gradient",
            "filter_derivative": "native_objective_gradient",
            "analytical_grad_requested": False,
            "analytical_grad_used": False,
            "optimizer_success": bool(getattr(
                result, "raw_optimizer_success", result.success)),
            "optimizer_message": str(getattr(
                result, "raw_optimizer_message", result.message)),
            "optimizer_objective": objective,
            "objective_discrepancy": objective_discrepancy,
            "nested_static_log_likelihood": static_log_likelihood,
            "static_baseline_log_likelihood": static_baseline,
            "likelihood_validation_passed": consistent and above_static,
        }
        gradient = getattr(result, "jac", None)
        if gradient is not None:
            gradient = np.asarray(gradient, dtype=np.float64).copy()
            lower, upper = model_policy.latent_bounds(
                "gas", gamma_bound=gamma_bound, beta_bound=beta_bound)
            point = np.asarray(getattr(result, "x", gas_values))
            for coordinate in range(3):
                if ((point[coordinate] <= lower[coordinate] and gradient[coordinate] > 0)
                        or (point[coordinate] >= upper[coordinate] and gradient[coordinate] < 0)):
                    gradient[coordinate] = 0.0
            result_diagnostics["projected_gradient_inf_norm"] = float(
                np.max(np.abs(gradient)))
        result_diagnostics.update(self._correlation_diagnostics(copula))
        if diagnostics:
            result_diagnostics.update(diagnostics)

        return GASResult(
            log_likelihood=final_log_likelihood,
            method="GAS",
            copula_name=copula.name,
            success=success,
            nfev=result.nfev,
            message=message,
            params=params,
            scaling=self.scaling,
            score_eps=score_eps,
            r_last=r_last,
            diagnostics=result_diagnostics,
            parameter_count=parameter_count,
        )

    def _fit_joint_static_shrinkage(
        self,
        copula,
        u,
        gamma0,
        optimizer_options,
        optimizer_gradient_eps,
        optimizer_gradient_relative,
        score_eps,
        gamma_bound,
        beta_bound,
        verbose,
        initial_mle_result=None,
        refine=True,
    ):
        n_corr = int(copula._corr_num_params())
        self._ensure_correlation_initialized(copula, u)
        corr0 = np.asarray(
            copula._initial_corr_params(u), dtype=np.float64).reshape(-1)
        if n_corr != 1 or corr0.size != 1:
            raise NotImplementedError(
                "GAS joint static correlation currently supports only "
                "corr_mode='shrinkage'")

        if gamma0 is None:
            gas0 = _automatic_gas_start(
                copula, u, self.config, initial_mle_result)
            fitted_corr = np.asarray(
                copula._pack_corr_params(), dtype=np.float64).reshape(-1)
            if fitted_corr.size == n_corr:
                corr0 = fitted_corr
        else:
            gamma0 = np.asarray(gamma0, dtype=np.float64).reshape(-1)
            if gamma0.size == 3:
                gas0 = gamma0.copy()
            elif gamma0.size == 3 + n_corr:
                gas0 = gamma0[:3].copy()
                corr0 = gamma0[3:].copy()
            else:
                raise ValueError(
                    f"gamma0 must contain 3 GAS parameters or "
                    f"{3 + n_corr} joint parameters, got {gamma0.size}")

        joint0 = np.concatenate([gas0, corr0])
        if not np.all(np.isfinite(joint0)):
            raise ValueError("gamma0 must contain only finite values")

        gas_lower, gas_upper = model_policy.latent_bounds(
            "gas", gamma_bound=gamma_bound, beta_bound=beta_bound)
        bounds = Bounds(
            np.concatenate([gas_lower, [float("-inf")]]),
            np.concatenate([gas_upper, [float("inf")]]),
        )
        base_correlation = np.ascontiguousarray(
            copula._corr_base, dtype=np.float64)

        def objective(joint):
            joint = np.asarray(joint, dtype=np.float64).reshape(-1)
            if joint.size != 3 + n_corr:
                raise ValueError(
                    f"joint GAS point must contain {3 + n_corr} values")
            if not np.all(np.isfinite(joint)):
                raise FloatingPointError(
                    "joint GAS point must contain only finite values")
            try:
                return (
                    _cpp_gas
                    .negative_log_likelihood_and_gradient_shrinkage(
                        joint[0],
                        joint[1],
                        joint[2],
                        joint[3],
                        base_correlation,
                        u,
                        copula,
                        self.scaling,
                        score_eps,
                        optimizer_gradient_eps=optimizer_gradient_eps,
                        optimizer_gradient_relative=(
                            optimizer_gradient_relative),
                        optimizer_bounds=(bounds.lb, bounds.ub),
                    )
                )
            except FloatingPointError:
                return model_policy.optimizer_failure_evaluation(
                    joint,
                    joint0,
                    self.config.fail_value,
                    directional_gradient=True,
                )

        if verbose:
            print(
                f"GAS fit: joint shrinkage gamma0={joint0}, "
                f"scaling={self.scaling}, score_eps={score_eps}, "
                f"options={optimizer_options}, gamma_bound={gamma_bound}, "
                f"beta_bound={beta_bound}"
            )

        result, optimizer_diagnostics = _fit_gas_starts(
            objective, joint0, bounds=bounds, options=optimizer_options,
            automatic=gamma0 is None, refine=refine,
        )
        try:
            copula._set_corr_from_params(result.x[3:])
        except Exception as exc:
            result.success = False
            result.message = (
                f"{result.message}; failed to set final correlation: {exc}")
            copula._set_corr_from_params(corr0)

        diagnostics = {
            "joint_static": True,
            "joint_optimizer": "python-lbfgsb",
            "joint_correlation": "shrinkage",
            "optimizer_gradient_eps": optimizer_gradient_eps,
            "optimizer_gradient_relative": optimizer_gradient_relative,
            "initial_params": joint0.copy(),
            "final_params": np.asarray(result.x, dtype=np.float64).copy(),
        }
        diagnostics.update(optimizer_diagnostics)
        if gamma0 is None:
            diagnostics["initialization"] = {
                "mle_source": (
                    "selection_result" if initial_mle_result is not None
                    else "strategy_fit")
            }
        return self._build_result(
            copula,
            u,
            result,
            result.x[:3],
            score_eps,
            gamma_bound,
            beta_bound,
            parameter_count=3 + n_corr,
            diagnostics=diagnostics,
        )

    def fit(
        self,
        copula,
        u: np.ndarray,
        gamma0: np.ndarray | None = None,
        gtol: float | None = None,
        ftol: float | None = None,
        maxfun: int | None = None,
        maxiter: int | None = None,
        maxls: int | None = None,
        eps: float | None = None,
        maxcor: int | None = None,
        finite_diff_rel_step: float | None = None,
        score_eps: float | None = None,
        gamma_bound: float | None = None,
        beta_bound: float | None = None,
        verbose: bool = False,
        initial_mle_result=None,
        **kwargs,
    ) -> GASResult:
        """Fit the native GAS model."""
        if "backend" in kwargs:
            raise TypeError(
                "GAS backend selection was removed; native execution is "
                "always used")
        reject_unknown_strategy_kwargs("GAS", kwargs)
        corr_num_params = int(
            getattr(copula, "_corr_num_params", lambda: 0)())
        if (
                corr_num_params
                and getattr(copula, "_corr_mode", None) != "shrinkage"):
            raise NotImplementedError(
                "GAS joint static correlation currently supports only "
                "corr_mode='shrinkage'")

        self._ensure_correlation_initialized(copula, u)
        _cpp_gas.ensure_supported(copula)
        _cpp_gas.require_available()

        optimizer_options = lbfgsb_options(
            self._optimizer_config(copula),
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
        (
            optimizer_options,
            optimizer_gradient_eps,
            optimizer_gradient_relative,
        ) = _native_optimizer_gradient_config(optimizer_options)
        score_eps = float(
            score_eps
            if score_eps is not None
            else self.config.gas_score_eps
        )
        gamma_bound = float(
            gamma_bound
            if gamma_bound is not None
            else self.config.gas_gamma_bound
        )
        beta_bound = float(
            beta_bound
            if beta_bound is not None
            else self.config.gas_beta_bound
        )
        if gamma_bound <= 0:
            raise ValueError("gamma_bound must be positive")
        if not 0 < beta_bound < 1:
            raise ValueError("beta_bound must be in (0, 1)")

        automatic_initialization = gamma0 is None
        if corr_num_params:
            return self._fit_joint_static_shrinkage(
                copula,
                u,
                gamma0,
                optimizer_options,
                optimizer_gradient_eps,
                optimizer_gradient_relative,
                score_eps,
                gamma_bound,
                beta_bound,
                verbose,
                initial_mle_result,
                refine=ftol is None,
            )

        if gamma0 is None:
            gamma0 = _automatic_gas_start(
                copula, u, self.config, initial_mle_result)

        if verbose:
            print(
                f"GAS fit: gamma0={gamma0}, scaling={self.scaling}, "
                f"score_eps={score_eps}, options={optimizer_options}, "
                f"gamma_bound={gamma_bound}, beta_bound={beta_bound}"
            )

        bounds = Bounds(*model_policy.latent_bounds(
            "gas", gamma_bound=gamma_bound, beta_bound=beta_bound))

        def objective(x, *, gradient_eps=optimizer_gradient_eps,
                      gradient_central=False, gradient_mean_coordinates=False,
                      allow_failure_penalty=True):
            try:
                return _cpp_gas.negative_log_likelihood_and_gradient(
                    x[0],
                    x[1],
                    x[2],
                    u,
                    copula,
                    self.scaling,
                    score_eps,
                    optimizer_gradient_eps=gradient_eps,
                    optimizer_gradient_central=gradient_central,
                    optimizer_gradient_mean_coordinates=gradient_mean_coordinates,
                    optimizer_gradient_relative=optimizer_gradient_relative,
                    optimizer_bounds=(bounds.lb, bounds.ub),
                )
            except FloatingPointError:
                if not allow_failure_penalty:
                    raise
                return model_policy.optimizer_failure_evaluation(
                    x,
                    gamma0,
                    self.config.fail_value,
                    directional_gradient=True,
                )

        validation_objective = None
        recovery_objectives = []
        if is_multivariate_copula(copula):
            explicit_difference_step = eps is not None or finite_diff_rel_step is not None
            validation_steps = _cpp_gas.optimizer_validation_steps()
            validation_step = (optimizer_gradient_eps if explicit_difference_step
                               else validation_steps[0])

            def validation_provider(step):
                provider = partial(
                    objective, gradient_eps=step, gradient_central=True,
                    gradient_mean_coordinates=True, allow_failure_penalty=False)
                provider.evaluations_per_point = 7
                provider.difference_step = step
                return provider

            validation_objective = validation_provider(validation_step)
            if not explicit_difference_step:
                validation_objective.finer_provider = validation_provider(validation_steps[1])
                validation_objective.refinement_providers = [
                    validation_provider(step) for step in validation_steps[2:]]
            validation_objective.gas_mean_parameterization = True
            validation_objective.objective_scale = float(len(u))
            if (automatic_initialization and ftol is None and not explicit_difference_step
                    and maxfun is None and maxiter is None and maxls is None):
                # Preserve both deterministic starts at a second derivative
                # scale, then condition the local refinement in mean space.
                for static in (False, True):
                    provider = partial(objective, gradient_eps=1e-7)
                    provider.evaluations_per_point = 4
                    provider.restart_point = np.asarray(gamma0).copy()
                    if static:
                        provider.restart_point[1] = 0.0
                    recovery_objectives.append(provider)
                for derivative_free in (False, True):
                    provider = partial(
                        objective, gradient_eps=validation_steps[0],
                        gradient_central=True, gradient_mean_coordinates=True)
                    provider.evaluations_per_point = 7
                    provider.gas_mean_parameterization = True
                    provider.objective_scale = float(len(u))
                    provider.derivative_free = derivative_free
                    recovery_objectives.append(provider)
        result, optimizer_diagnostics = _fit_gas_starts(
            objective, gamma0, bounds=bounds, options=optimizer_options,
            automatic=automatic_initialization, refine=ftol is None,
            recovery_objectives=recovery_objectives,
            validation_objective=validation_objective,
        )
        parameter_count = None
        corr_effective_num_params = getattr(
            copula, "_corr_effective_num_params", None)
        if callable(corr_effective_num_params):
            parameter_count = 3 + int(corr_effective_num_params())
        diagnostics = (
            {"initialization": {
                "mle_source": (
                    "selection_result"
                    if initial_mle_result is not None
                    else "strategy_fit")
            }}
            if automatic_initialization else {}
        )
        diagnostics.update({
            "optimizer_gradient_eps": optimizer_gradient_eps,
            "optimizer_gradient_relative": optimizer_gradient_relative,
        })
        diagnostics.update(optimizer_diagnostics)
        return self._build_result(
            copula,
            u,
            result,
            result.x,
            score_eps,
            gamma_bound,
            beta_bound,
            parameter_count=parameter_count,
            diagnostics=diagnostics or None,
        )

    def log_likelihood(self, copula, u: np.ndarray, result: GASResult) -> float:
        p = result.params
        return gas_loglik(
            p.omega,
            p.gamma,
            p.beta,
            u,
            copula,
            self._result_scaling(result),
            self._score_eps(result),
        )

    def predictive_mean(
        self,
        copula,
        u: np.ndarray,
        result: GASResult,
        **kwargs,
    ) -> np.ndarray:
        reject_unknown_operation_kwargs(self, 'predictive_mean', kwargs)
        p = result.params
        _, r_path, _ = gas_filter(
            p.omega,
            p.gamma,
            p.beta,
            u,
            copula,
            self._result_scaling(result),
            self._score_eps(result),
        )
        return r_path

    def rosenblatt_e2(
        self,
        copula,
        u: np.ndarray,
        result: GASResult,
    ) -> np.ndarray:
        return self.mixture_h(copula, u, result)

    def mixture_h(
        self,
        copula,
        u: np.ndarray,
        result: GASResult,
    ) -> np.ndarray:
        if is_multivariate_copula(copula):
            raise NotImplementedError(
                "pair h-functions are not defined for multivariate GAS")
        p = result.params
        return gas_mixture_h(
            p.omega,
            p.gamma,
            p.beta,
            u,
            copula,
            self._result_scaling(result),
            self._score_eps(result),
        )

    def mixture_h_pair(
        self,
        copula,
        u: np.ndarray,
        result: GASResult,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Both h-directions from one GAS-filtered parameter path."""
        if is_multivariate_copula(copula):
            raise NotImplementedError(
                "pair h-functions are not defined for multivariate GAS")
        p = result.params
        return gas_mixture_h_pair(
            p.omega,
            p.gamma,
            p.beta,
            u,
            copula,
            self._result_scaling(result),
            self._score_eps(result),
        )

    def objective(
        self,
        copula,
        u: np.ndarray,
        gamma: np.ndarray,
        **kwargs,
    ) -> float:
        if "backend" in kwargs:
            raise TypeError(
                "GAS backend selection was removed; native execution is "
                "always used")
        score_eps = as_float64_scalar(
            kwargs.pop("score_eps", self._score_eps()), name="score_eps")
        reject_unknown_strategy_kwargs("GAS", kwargs)
        gamma = as_float64_array(gamma, name="gamma")
        return gas_negloglik(
            gamma[0],
            gamma[1],
            gamma[2],
            u,
            copula,
            self.scaling,
            score_eps,
            fail_value=self.config.fail_value,
        )

    def sample(self, copula, u, result, n, rng=None, **kwargs):
        """Recursively sample using native GAS state updates."""
        reject_unknown_operation_kwargs(self, 'sample', kwargs)
        if "n_threads" in kwargs:
            validate_n_threads(kwargs["n_threads"])
        n = validate_positive_int(n, "n")
        if rng is None:
            rng = np.random.default_rng()
        given = kwargs.get("given")
        p = result.params
        score_eps = self._score_eps(result)
        d = copula_dimension(copula, u)
        if d is None:
            raise ValueError("copula dimension is unknown")
        validate_float64_allocation(
            (n, d),
            name="GAS sample output",
            memory_budget_bytes=kwargs.get("memory_budget_bytes"),
        )

        family = getattr(copula, "_native_pair_family", None)
        if d == 2 and family is not None and given is None:
            validate_float64_allocation(
                (n, 3 * d),
                name=(
                    "GAS fused sample output, native staging, "
                    "and RNG draws"),
                memory_budget_bytes=kwargs.get("memory_budget_bytes"),
            )
            draws = (
                rng.standard_normal((n, 2))
                if family == "Gaussian"
                else rng.uniform(0.0, 1.0, size=(n, 2))
            )
            return _cpp_gas.sample_bivariate(
                p.omega,
                p.gamma,
                p.beta,
                draws,
                copula,
                self._result_scaling(result),
                score_eps,
            )

        state = _cpp_gas.initial_state(
            p.omega,
            p.gamma,
            p.beta,
            copula,
            self._result_scaling(result),
            score_eps,
        )
        g_t = state.g
        r_t = state.parameter
        samples = np.empty((n, d), dtype=np.float64)

        for t in range(n):
            obs = sample_predictive(
                copula,
                1,
                np.array([r_t]),
                given=given,
                rng=rng,
                d=d,
                n_threads=kwargs.get("n_threads", 1),
                memory_budget_bytes=kwargs.get("memory_budget_bytes"),
                config=self.config,
            )
            samples[t] = obs[0]
            if t < n - 1:
                update = _cpp_gas.update_one(
                    p.omega,
                    p.gamma,
                    p.beta,
                    g_t,
                    obs,
                    copula,
                    self._result_scaling(result),
                    score_eps,
                )
                g_t = update.g_next
                r_t = update.r_next
        return samples

    def predict(self, copula, u, result, n, rng=None, **kwargs):
        reject_unknown_operation_kwargs(self, 'predict', kwargs)
        n = validate_positive_int(n, "n")
        d = copula_dimension(copula, u)
        if d is None:
            raise ValueError("copula dimension is unknown")
        validate_float64_allocation(
            (n, d + 1),
            name="GAS prediction output and parameter path",
            memory_budget_bytes=kwargs.get("memory_budget_bytes"),
        )
        return predict_from_strategy(
            self, copula, u, result, n, rng=rng, **kwargs)

    predictive_params = predictive_params_from_state

    def predictive_state(self, copula, u, result, **kwargs):
        reject_unknown_operation_kwargs(self, 'predictive_state', kwargs)
        horizon = kwargs.get("horizon", "next")
        if horizon in (0, "0"):
            horizon = "current"
        elif horizon in (1, "1"):
            horizon = "next"
        else:
            horizon = str(horizon).lower()
        if horizon not in {"current", "next"}:
            raise ValueError("horizon must be 'current' or 'next'")
        if u is None or len(u) == 0:
            if horizon == "current":
                raise ValueError(
                    "prediction history is required for GAS horizon='current'; "
                    "the fitted result stores only the next parameter")
            if self._result_scaling(result) != result.scaling:
                raise ValueError(
                    "prediction history is required to override GAS scaling")
            r_t = float(result.r_last)
        else:
            p = result.params
            r_t = gas_predict_param(
                p.omega,
                p.gamma,
                p.beta,
                u,
                copula,
                self._result_scaling(result),
                self._score_eps(result),
                horizon=horizon,
            )
        return PredictiveState(
            method="GAS",
            horizon=horizon,
            kind="point",
            r=np.array([r_t], dtype=np.float64),
            metadata={
                "g": float(copula.inv_transform(np.array([r_t]))[0])
            }
            if hasattr(copula, "inv_transform")
            else {},
        )

    def condition_state(self, copula, state, observation, result, **kwargs):
        reject_unknown_operation_kwargs(self, 'condition_state', kwargs)
        if observation is None:
            return state
        u = as_float64_array(observation, name="observation")
        d = copula_dimension(copula, u)
        if u.ndim != 2 or d is None or u.shape[1] != d or len(u) == 0:
            return state
        u = u[:1]

        p = result.params
        if "g" in state.metadata:
            g_t = float(state.metadata["g"])
        else:
            g_t = float(
                copula.inv_transform(np.array([float(state.r[0])]))[0])
        update = _cpp_gas.update_one(
            p.omega,
            p.gamma,
            p.beta,
            g_t,
            u,
            copula,
            self._result_scaling(result),
            self._score_eps(result),
        )
        return PredictiveState(
            method=state.method,
            horizon=state.horizon,
            kind=state.kind,
            r=np.array([update.r_next], dtype=np.float64),
            metadata={**dict(state.metadata), "g": update.g_next},
        )

    def sample_params(self, copula, state, n, rng=None, **kwargs):
        reject_unknown_operation_kwargs(self, 'sample_params', kwargs)
        n = validate_positive_int(n, "n")
        validate_float64_allocation(
            (n,),
            name="GAS parameter path",
            memory_budget_bytes=kwargs.get("memory_budget_bytes"),
        )
        return np.full(n, float(np.asarray(state.r)[0]), dtype=np.float64)

    def model_sample_params(self, copula, result, n, rng=None, **kwargs):
        reject_unknown_operation_kwargs(self, 'model_sample_params', kwargs)
        raise ValueError(
            "GAS sample paths require stepwise score updates and cannot be "
            "precomputed"
        )

    def model_sample_state(self, copula, result, **kwargs):
        reject_unknown_operation_kwargs(self, 'model_sample_state', kwargs)
        p = result.params
        initial = _cpp_gas.initial_state(
            p.omega,
            p.gamma,
            p.beta,
            copula,
            self._result_scaling(result),
            self._score_eps(result),
        )
        return PredictiveState(
            method="GAS",
            horizon="model",
            kind="point",
            r=np.array([initial.parameter], dtype=np.float64),
            metadata={"g": initial.g},
        )
