"""Shared orchestration for static multivariate maximum likelihood fits."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

from pyscarcopula.copula.multivariate.corr_param import validate_corr_matrix
from pyscarcopula.copula.multivariate.correlation_policy import FloatArray
from pyscarcopula._native.errors import NativeError


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


_EXPECTED_NUMERICAL_ERRORS = (
    FloatingPointError,
    OverflowError,
    ValueError,
    np.linalg.LinAlgError,
    NativeError,
)


def _failure_value_and_gradient(
        parameters: FloatArray,
        initial_parameters: FloatArray,
        fail_value: float) -> tuple[float, FloatArray]:
    direction = parameters - initial_parameters
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm == 0.0:
        direction = np.ones_like(initial_parameters)
    else:
        direction = direction / norm
    return fail_value, direction * np.sqrt(fail_value)


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
        raise FloatingPointError(
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

    Expected numerical failures are translated to a large objective with a
    non-zero gradient. Unexpected exceptions propagate to expose programming
    errors. No model object is accepted by this function, so objective calls
    cannot publish fitted state accidentally.
    """
    x0 = np.asarray(
        problem.initial_parameters, dtype=np.float64).reshape(-1).copy()
    if np.any(~np.isfinite(x0)):
        raise ValueError("initial_parameters must contain only finite values")
    bounds = tuple(problem.bounds)
    if len(bounds) != x0.size:
        raise ValueError("bounds must match initial_parameters")
    fail_value = float(fail_value)
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
        except _EXPECTED_NUMERICAL_ERRORS:
            return _failure_value_and_gradient(parameters, x0, fail_value)

    try:
        initial_evaluation = strict_evaluate(x0)
        initial_objective = initial_evaluation.objective
    except _EXPECTED_NUMERICAL_ERRORS:
        initial_evaluation = None
        initial_objective = fail_value

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
    except _EXPECTED_NUMERICAL_ERRORS:
        pass

    final_objective = (
        fail_value
        if final_evaluation is None else final_evaluation.objective)
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
