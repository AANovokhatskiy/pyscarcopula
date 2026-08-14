"""Full analytical Stage-3 bivariate family/rotation sweep."""

from __future__ import annotations

import numpy as np
import pytest

from pyscarcopula import (
    BivariateGaussianCopula,
    ClaytonCopula,
    FrankCopula,
    GumbelCopula,
    IndependentCopula,
    JoeCopula,
)
from pyscarcopula._types import (
    GASResult,
    IndependentResult,
    MLEResult,
    gas_params,
)
from pyscarcopula.api import predict as api_predict
from pyscarcopula.strategy._base import get_strategy_for_result
from pyscarcopula.strategy.predict_helpers import conditional_sample_bivariate

from ._bivariate_cases import (
    CASES,
    CONFIGURATIONS,
    MEDIUM_CASES,
    PARAMETER_REGIMES,
    BivariateCase,
    transposed_rotation,
)
from ._bivariate_oracles import conditional_cdf, rotated_cdf
from ._statistical_assertions import assert_uniform_pit


FAMILY_CLASSES = {
    "independent": IndependentCopula,
    "gaussian": BivariateGaussianCopula,
    "clayton": ClaytonCopula,
    "gumbel": GumbelCopula,
    "frank": FrankCopula,
    "joe": JoeCopula,
}

GIVEN_LEVELS = (0.015, 0.5, 0.985)

DYNAMIC_GAS_CASES = tuple(
    next(
        case
        for case in MEDIUM_CASES
        if case.family == family and case.rotation == rotation
    )
    for family, rotation in (
        ("gaussian", 0),
        ("clayton", 90),
        ("gumbel", 270),
        ("frank", 0),
        ("joe", 180),
    )
)

ASYMMETRIC_ROTATION_CASES = tuple(
    case for case in MEDIUM_CASES if case.rotation in (90, 270)
)


def _copula(case: BivariateCase):
    return FAMILY_CLASSES[case.family](rotate=case.rotation)


def _transposed_copula(case: BivariateCase):
    return FAMILY_CLASSES[case.family](
        rotate=transposed_rotation(case.rotation)
    )


def _native_directional_h(copula, transposed, free, given, parameter, free_index):
    directional = copula if free_index == 0 else transposed
    return directional.h(free, given, parameter)


def _native_directional_inverse(
        copula, transposed, q, given, parameter, free_index):
    directional = copula if free_index == 0 else transposed
    return directional.h_inverse(q, given, parameter)


def _mle_result(case: BivariateCase, copula):
    if case.family == "independent":
        return IndependentResult(
            log_likelihood=0.0,
            method="MLE",
            copula_name=copula.name,
            success=True,
        )
    return MLEResult(
        log_likelihood=0.0,
        method="MLE",
        copula_name=copula.name,
        success=True,
        copula_param=case.parameter,
    )


def _history(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0.12, 0.88, size=(19, 2))


def test_stage3_matrix_has_exactly_fifteen_supported_rotation_cells():
    assert len(CONFIGURATIONS) == 15
    assert len(CASES) == 44
    assert len(MEDIUM_CASES) == 15
    assert len(set(CONFIGURATIONS)) == len(CONFIGURATIONS)


@pytest.mark.parametrize("case", MEDIUM_CASES, ids=lambda case: case.id)
def test_reference_directional_h_is_derivative_of_rotated_cdf(case):
    u = np.array([0.14, 0.36, 0.63, 0.84])
    v = np.array([0.21, 0.47, 0.72, 0.79])
    step = 2e-4 if case.family == "gaussian" else 2e-5

    derivative_v = (
        rotated_cdf(
            u, v - 2.0 * step, case.parameter, case.family, case.rotation
        )
        - 8.0 * rotated_cdf(
            u, v - step, case.parameter, case.family, case.rotation
        )
        + 8.0 * rotated_cdf(
            u, v + step, case.parameter, case.family, case.rotation
        )
        - rotated_cdf(
            u, v + 2.0 * step, case.parameter, case.family, case.rotation
        )
    ) / (12.0 * step)
    derivative_u = (
        rotated_cdf(
            u - 2.0 * step, v, case.parameter, case.family, case.rotation
        )
        - 8.0 * rotated_cdf(
            u - step, v, case.parameter, case.family, case.rotation
        )
        + 8.0 * rotated_cdf(
            u + step, v, case.parameter, case.family, case.rotation
        )
        - rotated_cdf(
            u + 2.0 * step, v, case.parameter, case.family, case.rotation
        )
    ) / (12.0 * step)

    expected_first = conditional_cdf(
        u, v, case.parameter, case.family, case.rotation, free_index=0
    )
    expected_second = conditional_cdf(
        v, u, case.parameter, case.family, case.rotation, free_index=1
    )
    tolerance = 3e-5 if case.family == "gaussian" else 2e-8
    np.testing.assert_allclose(
        derivative_v, expected_first, rtol=0.0, atol=tolerance
    )
    np.testing.assert_allclose(
        derivative_u, expected_second, rtol=0.0, atol=tolerance
    )


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
@pytest.mark.parametrize("free_index", [0, 1], ids=["free-u1", "free-u2"])
def test_native_directional_h_and_inverse_match_analytical_oracle(
        case, free_index):
    copula = _copula(case)
    transposed = _transposed_copula(case)
    free = np.array([
        1e-6, 0.008, 0.04, 0.17, 0.43, 0.68, 0.91, 0.992, 1.0 - 1e-6
    ])
    given = np.array([
        0.37, 0.992, 0.08, 0.73, 0.51, 0.03, 0.88, 0.19, 0.64
    ])
    parameter = np.full(len(free), case.parameter)
    observed_h = _native_directional_h(
        copula, transposed, free, given, parameter, free_index
    )
    expected_h = conditional_cdf(
        free,
        given,
        parameter,
        case.family,
        case.rotation,
        free_index,
    )
    h_tolerance = 6e-9 if case.family == "gaussian" else 2e-10
    np.testing.assert_allclose(
        observed_h, expected_h, rtol=0.0, atol=h_tolerance
    )

    q = np.array([1e-4, 0.008, 0.08, 0.29, 0.61, 0.94, 1.0 - 1e-4])
    inverse_given = np.array([0.02, 0.91, 0.15, 0.48, 0.79, 0.98, 0.33])
    inverse_parameter = np.full(len(q), case.parameter)
    free_from_q = _native_directional_inverse(
        copula,
        transposed,
        q,
        inverse_given,
        inverse_parameter,
        free_index,
    )
    recovered_q = conditional_cdf(
        free_from_q,
        inverse_given,
        inverse_parameter,
        case.family,
        case.rotation,
        free_index,
    )
    np.testing.assert_allclose(recovered_q, q, rtol=0.0, atol=8e-8)


@pytest.mark.parametrize(
    "case", ASYMMETRIC_ROTATION_CASES, ids=lambda case: case.id
)
def test_conditional_sampler_transposes_asymmetric_rotation_for_given_u1(case):
    n = 513
    given = 0.83
    seed = 20261080 + ASYMMETRIC_ROTATION_CASES.index(case)
    expected_q = np.random.default_rng(seed).uniform(0.0, 1.0, size=n)
    samples = conditional_sample_bivariate(
        _copula(case),
        n,
        case.parameter,
        given={0: given},
        rng=np.random.default_rng(seed),
    )
    recovered_q = conditional_cdf(
        samples[:, 1],
        given,
        case.parameter,
        case.family,
        case.rotation,
        free_index=1,
    )
    np.testing.assert_array_equal(samples[:, 0], given)
    np.testing.assert_allclose(
        recovered_q, expected_q, rtol=0.0, atol=8e-8
    )


@pytest.mark.validation
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
@pytest.mark.parametrize("free_index", [0, 1], ids=["free-u1", "free-u2"])
@pytest.mark.parametrize(
    "given_level", GIVEN_LEVELS, ids=["given-lower-tail", "given-center", "given-upper-tail"]
)
def test_mle_public_conditional_prediction_passes_analytical_pit(
        case, free_index, given_level):
    copula = _copula(case)
    result = _mle_result(case, copula)
    given_index = 1 - free_index
    case_index = CASES.index(case)
    level_index = GIVEN_LEVELS.index(given_level)
    samples = api_predict(
        copula,
        _history(20261100 + case_index),
        result,
        5_000,
        given={given_index: given_level},
        rng=np.random.default_rng(
            20261200 + 20 * case_index + 5 * free_index + level_index
        ),
    )
    np.testing.assert_array_equal(samples[:, given_index], given_level)
    pit = conditional_cdf(
        samples[:, free_index],
        given_level,
        case.parameter,
        case.family,
        case.rotation,
        free_index,
    )
    assert_uniform_pit(pit)


@pytest.mark.validation
@pytest.mark.parametrize("case", MEDIUM_CASES, ids=lambda case: case.id)
@pytest.mark.parametrize("free_index", [0, 1], ids=["free-u1", "free-u2"])
def test_rowwise_dynamic_parameter_path_passes_directional_pit(case, free_index):
    n = 7_000
    if case.family == "independent":
        parameter = np.zeros(n)
    elif case.family == "gaussian":
        parameter = np.linspace(-0.72, 0.86, n)
    else:
        values = [value for _, value in PARAMETER_REGIMES[case.family]]
        phase = np.linspace(0.0, 3.0 * np.pi, n)
        low, high = min(values), max(values)
        parameter = low + (high - low) * (0.5 + 0.5 * np.sin(phase))

    given_index = 1 - free_index
    given_level = 0.87
    samples = conditional_sample_bivariate(
        _copula(case),
        n,
        parameter,
        given={given_index: given_level},
        rng=np.random.default_rng(20261300 + MEDIUM_CASES.index(case) * 2 + free_index),
    )
    pit = conditional_cdf(
        samples[:, free_index],
        given_level,
        parameter,
        case.family,
        case.rotation,
        free_index,
    )
    assert_uniform_pit(pit)


@pytest.mark.validation
@pytest.mark.parametrize("case", DYNAMIC_GAS_CASES, ids=lambda case: case.id)
@pytest.mark.parametrize("free_index", [0, 1], ids=["free-u1", "free-u2"])
def test_gas_public_conditional_prediction_uses_directional_oracle(
        case, free_index):
    copula = _copula(case)
    beta = 0.35
    baseline = float(copula.inv_transform(np.array([case.parameter]))[0])
    result = GASResult(
        log_likelihood=0.0,
        method="GAS",
        copula_name=copula.name,
        success=True,
        params=gas_params(baseline * (1.0 - beta), 0.08, beta),
        scaling="unit",
        r_last=case.parameter,
    )
    history = _history(20261400 + DYNAMIC_GAS_CASES.index(case))
    strategy = get_strategy_for_result(result)
    target_parameter = float(
        strategy.predictive_state(
            copula, history, result, horizon="next"
        ).r[0]
    )
    given_index = 1 - free_index
    given_level = 0.82
    samples = api_predict(
        copula,
        history,
        result,
        6_000,
        given={given_index: given_level},
        horizon="next",
        rng=np.random.default_rng(
            20261500 + DYNAMIC_GAS_CASES.index(case) * 2 + free_index
        ),
    )
    pit = conditional_cdf(
        samples[:, free_index],
        given_level,
        target_parameter,
        case.family,
        case.rotation,
        free_index,
    )
    assert_uniform_pit(pit)
