"""Native GAS numerical operations.

Python owns optimizer orchestration, RNG, and sampling. The compiled
``GasEvaluator`` is the single implementation of GAS filtering, likelihood,
state updates, prediction, and bivariate h-path evaluation.
"""

import numpy as np

from pyscarcopula._utils import clip_rosenblatt_output
from pyscarcopula._native import gas as _cpp_gas
from pyscarcopula._native import model_policy
from pyscarcopula.numerical._arrays import as_float64_array


def _gas_input(u):
    from pyscarcopula.copula.multivariate.equicorr_prepared import (
        EquicorrPreparedData,
    )
    return u if isinstance(u, EquicorrPreparedData) else as_float64_array(u)


def gas_filter(
    omega,
    gamma,
    beta,
    u,
    copula,
    scaling="unit",
    score_eps=1e-4,
):
    """Run the native GAS filter and return paths and log-likelihood."""
    return _cpp_gas.filter(
        omega,
        gamma,
        beta,
        _gas_input(u),
        copula,
        scaling,
        score_eps,
    )


def gas_predict_param(
    omega,
    gamma,
    beta,
    u,
    copula,
    scaling="unit",
    score_eps=1e-4,
    horizon="next",
):
    """Return the native current or next GAS copula parameter."""
    return _cpp_gas.predict_parameter(
        omega,
        gamma,
        beta,
        _gas_input(u),
        copula,
        scaling,
        score_eps,
        horizon,
    )


def gas_loglik(
    omega,
    gamma,
    beta,
    u,
    copula,
    scaling="unit",
    score_eps=1e-4,
):
    """Evaluate the native GAS log-likelihood."""
    return _cpp_gas.log_likelihood(
        omega,
        gamma,
        beta,
        _gas_input(u),
        copula,
        scaling,
        score_eps,
    )


def gas_negloglik(
    omega,
    gamma,
    beta,
    u,
    copula,
    scaling="unit",
    score_eps=1e-4,
):
    """Return native ``-logL`` and propagate its structured failures."""
    try:
        return _cpp_gas.negative_log_likelihood(
            omega,
            gamma,
            beta,
            _gas_input(u),
            copula,
            scaling,
            score_eps,
        )
    except FloatingPointError as error:
        return model_policy.optimizer_failure_objective(error)


def gas_rosenblatt(
    omega,
    gamma,
    beta,
    u,
    copula,
    scaling="unit",
    score_eps=1e-4,
):
    """Return the bivariate GAS Rosenblatt transform."""
    observations = as_float64_array(u)
    if observations.ndim != 2 or observations.shape[1] != 2:
        raise ValueError(
            f"u must have shape (T, 2), got {observations.shape}")
    e = np.empty((len(observations), 2), dtype=np.float64)
    e[:, 0] = observations[:, 0]
    e[:, 1] = _cpp_gas.h_path(
        omega,
        gamma,
        beta,
        observations,
        copula,
        scaling,
        score_eps,
    )
    return clip_rosenblatt_output(e)


def gas_mixture_h(
    omega,
    gamma,
    beta,
    u,
    copula,
    scaling="unit",
    score_eps=1e-4,
):
    """Evaluate ``h(u2 | u1; Psi(g_t))`` along the native GAS path."""
    return _cpp_gas.h_path(
        omega,
        gamma,
        beta,
        as_float64_array(u),
        copula,
        scaling,
        score_eps,
    )


def gas_mixture_h_pair(
    omega,
    gamma,
    beta,
    u,
    copula,
    scaling="unit",
    score_eps=1e-4,
):
    """Evaluate both h-directions along one native GAS filter path."""
    observations = as_float64_array(u)
    _, r_path, _ = gas_filter(
        omega, gamma, beta, observations, copula, scaling, score_eps)
    first_given_second, second_given_first = copula.h_pair(
        observations[:, 0], observations[:, 1], r_path)
    return second_given_first, first_given_second


__all__ = [
    "gas_filter",
    "gas_loglik",
    "gas_mixture_h",
    "gas_mixture_h_pair",
    "gas_negloglik",
    "gas_predict_param",
    "gas_rosenblatt",
]
