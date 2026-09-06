"""Shared orchestration for static multivariate maximum likelihood fits."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

from pyscarcopula._native import model_policy
from pyscarcopula._types import LBFGSBConfig
from pyscarcopula.copula.multivariate.corr_param import validate_corr_matrix
from pyscarcopula.copula.multivariate.correlation_policy import FloatArray
from pyscarcopula.numerical._arrays import as_float64_array, as_float64_scalar


def sampling_model_from_result(copula, result):
    """Build independent sampling state from a static result's physical values."""
    from pyscarcopula._native.registry import native_id_for

    family = native_id_for(copula)
    if family == "EquicorrGaussian":
        # Its sampler takes rho explicitly and has no fitted correlation state.
        return copula
    if family not in {"Gaussian", "Student", "StochasticStudent"}:
        raise TypeError("result requires a static multivariate model")
    correlation = result.correlation_matrix
    if correlation is None:
        loadings = result.model_parameters.get("factor_loadings")
        if loadings is None:
            raise ValueError("static result is missing its correlation state")
        dimension, rank = loadings.shape
        options = dict(
            corr_mode="factor", factor_rank=rank, factor_loadings=loadings,
            factor_uniqueness_min=np.finfo(np.float64).tiny)
    else:
        dimension = correlation.shape[0]
        # A static result contains physical correlation state. Student's
        # shape setter validates that state without projecting it again.
        options = {} if family == "Student" else dict(R=correlation)
    if copula.dimension is not None and copula.dimension != dimension:
        raise ValueError("static result dimension does not match the model")
    snapshot = type(copula)(d=dimension, **options)
    if family == "Gaussian" and correlation is not None:
        snapshot.corr = correlation.copy()
    if family == "Student":
        snapshot.shape = correlation
        snapshot.df = as_float64_scalar(
            result.copula_param, name="copula_param")
    return snapshot


def log_likelihood_from_result(copula, u, result, *, n_threads=1):
    """Evaluate an owned static result without reading fitted model state."""
    from pyscarcopula._native import static as static_likelihood
    from pyscarcopula._native.registry import native_id_for
    from pyscarcopula.copula.multivariate.factor_correlation import FactorCorrelation
    from pyscarcopula.copula.multivariate.factor_student import FactorStudentEvaluator

    family = native_id_for(copula)
    if family == "EquicorrGaussian":
        return static_likelihood.prepare(
            copula, u, n_threads=n_threads).log_likelihood(result.copula_param)
    if family not in {"Gaussian", "Student", "StochasticStudent"}:
        raise TypeError("result requires a static multivariate model")
    correlation = result.correlation_matrix
    loadings = result.model_parameters.get("factor_loadings")
    if correlation is None:
        if loadings is None:
            raise ValueError("static result is missing its correlation state")
        # The result already contains physical loadings. Do not impose the
        # prototype's possibly different optimization uniqueness constraint.
        operator = FactorCorrelation(
            loadings, uniqueness_min=np.finfo(np.float64).tiny).prepare()
        if family != "Gaussian":
            return FactorStudentEvaluator(operator, u).evaluate(
                result.copula_param, n_threads=n_threads).log_likelihood
        evaluator = static_likelihood.prepare_factor_gaussian(
            operator, u, n_threads=n_threads)
    else:
        prepare = (static_likelihood.prepare_gaussian if family == "Gaussian"
                   else static_likelihood.prepare_student)
        evaluator = prepare(correlation, u, n_threads=n_threads)
    return evaluator.log_likelihood(
        0.0 if family == "Gaussian" else result.copula_param)


@dataclass(frozen=True)
class StaticMLEEvaluation:
    """One valid objective evaluation and its unpublished candidate state."""

    objective: float
    gradient: FloatArray
    correlation: FloatArray | None = None
    state: Mapping[str, object] = field(default_factory=dict)
    diagnostics: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class StaticMLEProblem:
    """Model-specific adapter consumed by the shared optimizer workflow."""

    family: str
    initial_parameters: FloatArray
    bounds: Sequence[tuple[float | None, float | None]]
    evaluate: Callable[[FloatArray], StaticMLEEvaluation]
    require_not_worse: bool = True
    gradient_gate_multiplier: float = 200.0
    gradient_gate_floor: float = 1e-4
    objective_match_rtol: float = 1e-9
    objective_match_atol: float = 1e-8
    initial_worse_tolerance: float = 1e-8


@dataclass(frozen=True)
class StaticMLEOutcome:
    """Optimizer output after independent final-point validation."""

    parameters: FloatArray
    evaluation: StaticMLEEvaluation | None
    accepted: bool
    optimizer_success: bool
    nfev: int
    message: str
    initial_objective: float
    optimizer_objective: float
    final_objective: float
    final_gradient_inf_norm: float
    gradient_gate: float
    objective_match: bool
    not_worse_than_initial: bool
    evaluations: int

    def diagnostics(self) -> dict[str, object]:
        """Return common acceptance metadata for public fit diagnostics."""
        return {
            "static_mle_strategy": "shared_multivariate",
            "optimizer_success": self.optimizer_success,
            "final_validation_passed": self.accepted,
            "initial_objective": self.initial_objective,
            "optimizer_objective": self.optimizer_objective,
            "final_objective": self.final_objective,
            "final_gradient_inf_norm": self.final_gradient_inf_norm,
            "gradient_gate": self.gradient_gate,
            "objective_match": self.objective_match,
            "not_worse_than_initial": self.not_worse_than_initial,
            "strategy_evaluations": self.evaluations,
        }


def make_student_static_mle_evaluator(
        initial_correlation, policy, observations, *, n_threads, fail_value):
    """Create the shared dense Student df/correlation native evaluator."""
    from pyscarcopula._native import static as static_likelihood

    n_corr = policy.optimized_n_params
    fixed_evaluator = (
        static_likelihood.prepare_student(
            initial_correlation, observations, n_threads=n_threads)
        if n_corr == 0 else None)

    def evaluate(parameters):
        correlation = (
            initial_correlation.copy()
            if n_corr == 0
            else policy.trial_correlation(parameters[1:]))
        evaluator = fixed_evaluator
        if evaluator is None:
            evaluator = static_likelihood.prepare_student(
                correlation, observations, n_threads=n_threads)
            value, df_gradient, corr_gradient = (
                evaluator.objective_and_joint_gradient(
                    float(parameters[0]), fail_value=fail_value))
        else:
            value, df_gradient = evaluator.objective_and_gradient(
                float(parameters[0]), fail_value=fail_value)
        gradient = np.empty_like(parameters)
        gradient[0] = df_gradient[0]
        if n_corr:
            gradient[1:] = policy.raw_gradient(
                parameters[1:], correlation, corr_gradient)
        return StaticMLEEvaluation(
            objective=value,
            gradient=gradient,
            correlation=correlation,
            state={"df": float(parameters[0])},
        )

    return evaluate


class _RejectedEvaluation(FloatingPointError):
    """A status-ok evaluator result that violates the optimizer contract."""


def _validate_evaluation(
        evaluation: StaticMLEEvaluation,
        parameter_count: int,
        fail_value: float) -> StaticMLEEvaluation:
    value = float(evaluation.objective)
    gradient = np.asarray(evaluation.gradient, dtype=np.float64).reshape(-1)
    if gradient.size != parameter_count:
        raise ValueError(
            f"objective gradient has size {gradient.size}, "
            f"expected {parameter_count}")
    if (
            not np.isfinite(value)
            or value >= fail_value
            or np.any(~np.isfinite(gradient))):
        raise _RejectedEvaluation(
            "static MLE objective returned a non-finite/failure result")
    correlation = evaluation.correlation
    if correlation is not None:
        correlation = np.asarray(correlation, dtype=np.float64)
        validate_corr_matrix(correlation)
        correlation = correlation.copy()
        correlation.setflags(write=False)
    gradient = gradient.copy()
    gradient.setflags(write=False)
    return StaticMLEEvaluation(
        objective=value,
        gradient=gradient,
        correlation=correlation,
        state=dict(evaluation.state),
        diagnostics=dict(evaluation.diagnostics),
    )


def _projected_gradient(
        parameters: FloatArray,
        gradient: FloatArray,
        bounds: Sequence[tuple[float | None, float | None]],
        ) -> FloatArray:
    """Return the L-BFGS-B projected gradient at a bounded point."""
    projected = np.asarray(gradient, dtype=np.float64).copy()
    for index, (lower, upper) in enumerate(bounds):
        value = float(parameters[index])
        tolerance = 1e-10 * (1.0 + abs(value))
        if (
                lower is not None
                and value <= float(lower) + tolerance
                and projected[index] > 0.0):
            projected[index] = 0.0
        if (
                upper is not None
                and value >= float(upper) - tolerance
                and projected[index] < 0.0):
            projected[index] = 0.0
    return projected


def run_static_multivariate_mle(
        problem: StaticMLEProblem,
        *,
        optimizer_options: Mapping[str, float | int],
        fail_value: float) -> StaticMLEOutcome:
    """Optimize and independently validate one static multivariate problem.

    Typed native failures propagate through the shared status-to-exception
    policy. Structured numerical failures and rejected numeric results use the
    C++-owned optimizer penalty. No model object is accepted by this function,
    so objective calls cannot publish fitted state accidentally.
    """
    optimizer_options = LBFGSBConfig().options(**optimizer_options)
    x0 = as_float64_array(
        problem.initial_parameters, name="initial_parameters").reshape(-1).copy()
    if np.any(~np.isfinite(x0)):
        raise ValueError("initial_parameters must contain only finite values")
    bounds = tuple(problem.bounds)
    if len(bounds) != x0.size:
        raise ValueError("bounds must match initial_parameters")
    fail_value = as_float64_scalar(fail_value, name="fail_value")
    if not np.isfinite(fail_value) or fail_value <= 0.0:
        raise ValueError("fail_value must be positive and finite")

    evaluations = 0

    def strict_evaluate(parameters: FloatArray) -> StaticMLEEvaluation:
        nonlocal evaluations
        parameters = np.asarray(parameters, dtype=np.float64).reshape(-1)
        if parameters.size != x0.size or np.any(~np.isfinite(parameters)):
            raise ValueError("invalid static MLE optimizer point")
        evaluations += 1
        return _validate_evaluation(
            problem.evaluate(parameters.copy()), x0.size, fail_value)

    def objective_and_gradient(
            parameters: FloatArray) -> tuple[float, FloatArray]:
        parameters = np.asarray(parameters, dtype=np.float64).reshape(-1)
        try:
            evaluation = strict_evaluate(parameters)
            return evaluation.objective, evaluation.gradient.copy()
        except _RejectedEvaluation:
            return model_policy.optimizer_failure_evaluation(
                parameters, x0, fail_value, directional_gradient=True)
        except FloatingPointError as error:
            return model_policy.optimizer_numerical_failure_evaluation(
                error, parameters, x0, fail_value,
                directional_gradient=True)

    initial_evaluation = None
    try:
        initial_evaluation = strict_evaluate(x0)
        initial_objective = initial_evaluation.objective
    except _RejectedEvaluation:
        initial_objective, _ = model_policy.optimizer_failure_evaluation(
            x0, x0, fail_value, directional_gradient=True)
    except FloatingPointError as error:
        initial_objective = model_policy.optimizer_failure_objective(
            error, fail_value)

    if x0.size:
        optimizer_result = minimize(
            objective_and_gradient,
            x0,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options=dict(optimizer_options),
        )
        final_parameters = np.asarray(
            optimizer_result.x, dtype=np.float64).reshape(-1)
        optimizer_objective = float(getattr(
            optimizer_result, "fun", np.nan))
        optimizer_success = bool(optimizer_result.success)
        nfev = int(getattr(optimizer_result, "nfev", evaluations))
        message = str(getattr(optimizer_result, "message", ""))
    else:
        final_parameters = x0.copy()
        optimizer_objective = initial_objective
        optimizer_success = initial_evaluation is not None
        nfev = 0
        message = "closed-form/static plug-in evaluation"

    final_evaluation = None
    try:
        final_evaluation = strict_evaluate(final_parameters)
        final_objective = final_evaluation.objective
    except _RejectedEvaluation:
        final_objective, _ = model_policy.optimizer_failure_evaluation(
            final_parameters, x0, fail_value, directional_gradient=True)
    except FloatingPointError as error:
        final_objective = model_policy.optimizer_failure_objective(
            error, fail_value)
    projected_gradient = (
        None
        if final_evaluation is None
        else _projected_gradient(
            final_parameters, final_evaluation.gradient, bounds))
    gradient_inf_norm = (
        float("inf")
        if projected_gradient is None
        else float(np.max(np.abs(projected_gradient), initial=0.0)))
    gtol = float(optimizer_options.get("gtol", 1e-5))
    gradient_gate = max(
        float(problem.gradient_gate_floor),
        float(problem.gradient_gate_multiplier) * gtol,
    )
    objective_match = bool(
        final_evaluation is not None
        and np.isfinite(optimizer_objective)
        and np.isclose(
            optimizer_objective,
            final_objective,
            rtol=float(problem.objective_match_rtol),
            atol=float(problem.objective_match_atol),
        ))
    not_worse = bool(
        not problem.require_not_worse
        or final_objective <= (
            initial_objective + float(problem.initial_worse_tolerance)))
    gradient_ok = bool(
        final_evaluation is not None
        and (
            final_parameters.size == 0
            or gradient_inf_norm <= gradient_gate))
    improved_after_optimizer_failure = bool(
        not optimizer_success
        and final_evaluation is not None
        and final_objective < (
            initial_objective - float(problem.initial_worse_tolerance)))
    optimizer_status_ok = bool(
        optimizer_success or improved_after_optimizer_failure)
    accepted = bool(
        optimizer_status_ok
        and final_evaluation is not None
        and objective_match
        and not_worse
        and gradient_ok)

    rejection_reasons = []
    if not optimizer_status_ok:
        rejection_reasons.append("optimizer status")
    if final_evaluation is None:
        rejection_reasons.append("invalid final evaluation")
    if not objective_match:
        rejection_reasons.append("objective mismatch")
    if not not_worse:
        rejection_reasons.append("worse than initial point")
    if not gradient_ok:
        rejection_reasons.append(
            f"gradient gate ({gradient_inf_norm:.6g} > "
            f"{gradient_gate:.6g})")
    if rejection_reasons:
        suffix = "; rejected by final validation: " + ", ".join(
            rejection_reasons)
        message = f"{message}{suffix}"
    elif not optimizer_success:
        message = f"{message}; accepted by independent final validation"

    final_parameters = final_parameters.copy()
    final_parameters.setflags(write=False)
    return StaticMLEOutcome(
        parameters=final_parameters,
        evaluation=final_evaluation,
        accepted=accepted,
        optimizer_success=optimizer_success,
        nfev=nfev,
        message=message,
        initial_objective=float(initial_objective),
        optimizer_objective=optimizer_objective,
        final_objective=float(final_objective),
        final_gradient_inf_norm=gradient_inf_norm,
        gradient_gate=gradient_gate,
        objective_match=objective_match,
        not_worse_than_initial=not_worse,
        evaluations=evaluations,
    )
