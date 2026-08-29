"""Thin adapters for C++-owned numerical validation and preprocessing."""

from __future__ import annotations

import numpy as np

from pyscarcopula._native import _extension


def _array(values):
    return np.ascontiguousarray(values, dtype=np.float64)


def clip_open_unit(values, epsilon):
    result = dict(
        _extension.load().validation_clip_open_unit(
            _array(values), float(epsilon)))
    if int(result["status"]) != 0:
        raise ValueError("clipping epsilon must be finite and in (0, 0.5)")
    return np.asarray(
        result["values"],
        dtype=np.float64,
    )


def open_unit_clip_required(values, epsilon):
    result = dict(
        _extension.load().validation_open_unit_clip_required(
            _array(values), float(epsilon)))
    if int(result["status"]) != 0:
        raise ValueError("clipping epsilon must be finite and in (0, 0.5)")
    return bool(result["value"])


def objective_is_invalid(value):
    """Return the C++-owned optimizer objective rejection decision."""
    return bool(
        _extension.load().validation_objective_is_invalid(float(value)))


def validate_fit_data(values, model_name):
    result = dict(
        _extension.load().validation_validate_fit_data(_array(values)))
    code = int(result["code"])
    if int(result["status"]) == 0:
        return
    if code == 2:
        raise ValueError("data must contain only finite values")
    if code == 3:
        raise ValueError(
            "MLE expects pseudo-observations in [0, 1]; use to_pobs=True")
    if code == 4:
        raise ValueError(
            f"{model_name} copula correlation is not identifiable for "
            "constant data columns")
    if code == 5:
        raise ValueError(
            f"{model_name} copula correlation is not identifiable for "
            "duplicate data columns")
    raise ValueError("invalid fit data")


def validate_equicorr_prepared(
        sum_z, sum_z2, dimension, clipping_epsilon):
    result = dict(_extension.load().validation_validate_equicorr_prepared(
        _array(sum_z),
        _array(sum_z2),
        int(dimension),
        float(clipping_epsilon),
    ))
    code = int(result["code"])
    if int(result["status"]) == 0:
        return
    if code == 1:
        raise ValueError(
            "clipping_epsilon must be finite and in (0, 0.5)")
    if code == 2:
        raise ValueError("prepared statistics must contain finite values")
    if code == 6:
        raise ValueError("sum_z2 must be non-negative")
    if code == 7:
        raise ValueError(
            "prepared statistics violate sum_z**2 <= dimension*sum_z2")
    raise ValueError("invalid prepared statistics")


def valid_ou_final_parameters(values):
    return bool(
        _extension.load().validation_valid_ou_final_parameters(
            _array(values)))


def validate_final_fit(
        final_parameters, initial_parameters, lower, upper, *,
        optimizer_value, selected_value, selected_gradient,
        selected_evaluation_succeeded, selected_engine, selected_error,
        n_obs, strict_gradient_policy, explicit_gradient_tolerance,
        optimizer_gtol, rho_tolerance, growth_limit):
    return dict(_extension.load().validation_validate_ou_final_fit(
        _array(final_parameters),
        _array(initial_parameters),
        _array(lower),
        _array(upper),
        float(optimizer_value),
        float(selected_value),
        _array(selected_gradient),
        bool(selected_evaluation_succeeded),
        str(selected_engine),
        str(selected_error),
        int(n_obs),
        bool(strict_gradient_policy),
        (None if explicit_gradient_tolerance is None
         else float(explicit_gradient_tolerance)),
        float(optimizer_gtol),
        float(rho_tolerance),
        float(growth_limit),
    ))


def validate_backend_agreement(
        *, enabled, evaluation_succeeded, engine, error,
        validation_value, selected_value, n_obs,
        abs_per_observation, relative_tolerance):
    return dict(_extension.load().validation_validate_backend_agreement(
        bool(enabled),
        bool(evaluation_succeeded),
        str(engine),
        str(error),
        float(validation_value),
        float(selected_value),
        int(n_obs),
        float(abs_per_observation),
        float(relative_tolerance),
    ))


def validate_correlation_fit_state(
        raw_parameters, expected_parameter_count, correlation,
        dimension, inverse_factor, log_determinant, *, tolerance):
    return list(
        _extension.load().validation_validate_correlation_fit_state(
            _array(raw_parameters),
            int(expected_parameter_count),
            _array(correlation),
            int(dimension),
            _array(inverse_factor),
            float(log_determinant),
            float(tolerance),
        ))
