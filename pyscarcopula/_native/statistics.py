"""Thin adapters for C++-owned reductions and model statistics."""

from __future__ import annotations

import numpy as np

from pyscarcopula._native import _extension
from pyscarcopula._native.errors import raise_for_status


def _checked(result, operation):
    raise_for_status(result, operation, prefix="C++ model statistics")
    return result


def _double(result, operation):
    return float(_checked(result, operation)["value"])


def _boolean(result, operation):
    return bool(_checked(result, operation)["value"])


def sum_values(values):
    source = values if isinstance(values, np.ndarray) else tuple(values)
    array = np.asarray(source, dtype=np.float64)
    return _double(
        _extension.load().statistics_sum_values(array),
        "sum values")


def sum_int64(values):
    array = np.asarray(tuple(values), dtype=np.int64)
    return int(_checked(
        _extension.load().statistics_sum_int64(array),
        "sum integer values")["value"])


def sum_absolute(values):
    array = np.asarray(tuple(values), dtype=np.float64)
    return _double(
        _extension.load().statistics_sum_absolute(array),
        "sum absolute values")


def add_scores(left, right):
    return _double(
        _extension.load().statistics_add_scores(float(left), float(right)),
        "add scores")


def information_criterion(
        log_likelihood, parameter_count, observation_count, criterion):
    module = _extension.load()
    criteria = {
        "aic": module.InformationCriterion.AIC,
        "bic": module.InformationCriterion.BIC,
        "loglik": module.InformationCriterion.NEGATIVE_LOG_LIKELIHOOD,
    }
    try:
        native_criterion = criteria[criterion]
    except KeyError as exc:
        raise ValueError(f"unknown information criterion: {criterion!r}") from exc
    return _double(
        module.statistics_information_criterion(
            float(log_likelihood),
            int(parameter_count),
            int(observation_count),
            native_criterion),
        f"{criterion} criterion")


def candidate_score(
        log_likelihood, parameter_count, observation_count, criterion):
    return information_criterion(
        log_likelihood, parameter_count, observation_count, criterion)


def dense_ranks_no_ties(values):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2:
        return None
    result = _extension.load().statistics_dense_ranks_no_ties(array)
    if int(result["status"]) in (2, 6):
        return None
    _checked(result, "dense ranks")
    return np.asarray(result["value"], dtype=np.intp)


def dense_rank_matrix_no_ties(values):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 2:
        return None
    result = _extension.load().statistics_dense_rank_matrix_no_ties(array)
    if int(result["status"]) in (2, 6):
        return None
    _checked(result, "dense rank matrix")
    return np.asarray(result["value"], dtype=np.intp)


def kendall_tau_from_dense_ranks(first, second):
    first_array = np.asarray(first, dtype=np.int64)
    second_array = np.asarray(second, dtype=np.int64)
    return _double(
        _extension.load().statistics_kendall_tau_from_dense_ranks(
            first_array, second_array),
        "Kendall tau from dense ranks")


def kendall_tau(first, second):
    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    return _double(
        _extension.load().statistics_kendall_tau(first_array, second_array),
        "Kendall tau")


def tau_for_itau(tau, *, preserve_sign):
    result = _extension.load().statistics_tau_for_itau(
        float(tau), bool(preserve_sign))
    if int(result["status"]) == 6:
        return None
    return float(_checked(result, "itau family statistic")["value"])


def rotation_compatible(tau, rotation):
    return _boolean(
        _extension.load().statistics_rotation_compatible(
            float(tau), int(rotation)),
        "rotation compatibility")


def absolute_below(value, threshold):
    return _boolean(
        _extension.load().statistics_absolute_below(
            float(value), float(threshold)),
        "absolute threshold")


def absolute_value(value):
    return _double(
        _extension.load().statistics_absolute_value(float(value)),
        "absolute value")


def is_finite(value):
    return _boolean(
        _extension.load().statistics_is_finite(float(value)),
        "finite predicate")


def is_nan(value):
    return _boolean(
        _extension.load().statistics_is_nan(float(value)),
        "NaN predicate")
