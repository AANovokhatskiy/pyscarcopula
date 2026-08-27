"""Executable contracts for the conditional-sampling support registry."""

from __future__ import annotations

import importlib
import inspect

import numpy as np
import pytest
from scipy.stats import norm

from pyscarcopula import (
    BivariateGaussianCopula,
    ClaytonCopula,
    EquicorrGaussianCopula,
    FrankCopula,
    GaussianCopula,
    GumbelCopula,
    IndependentCopula,
    JoeCopula,
    RVineCopula,
    StochasticStudentCopula,
    StudentCopula,
    VineCopula,
)
from pyscarcopula._types import IndependentResult, MLEResult
from pyscarcopula._native.registry import query_capability, strategy_support
from pyscarcopula.strategy._base import (
    ensure_strategy_supported,
    has_dynamic_scalar_parameter,
    is_pair_copula,
    supports_conditional_sampling,
)

from ._registry import (
    REGISTRY,
    ConditionalCase,
    ModelCase,
    UnsupportedCase,
)


_PAIR_PARAMETERS = {
    "bivariate-gaussian": 0.45,
    "bivariate-clayton": 2.0,
    "bivariate-gumbel": 2.0,
    "bivariate-frank": 5.0,
    "bivariate-joe": 2.0,
}


def _correlation(d: int = 3) -> np.ndarray:
    result = np.full((d, d), 0.2, dtype=np.float64)
    np.fill_diagonal(result, 1.0)
    return result


def _observations(d: int = 3, n: int = 100) -> np.ndarray:
    rng = np.random.default_rng(20260813)
    latent = rng.multivariate_normal(np.zeros(d), _correlation(d), size=n)
    return np.clip(norm.cdf(latent), 1e-8, 1.0 - 1e-8)


def _public_class(case: ModelCase):
    module = importlib.import_module(case.class_module)
    return getattr(module, case.class_name)


def _construct_for_capabilities(case: ModelCase):
    cls = _public_class(case)
    if case.category == "bivariate":
        return cls()
    if case.id in {"multivariate-gaussian", "multivariate-student"}:
        return cls(d=3, R=_correlation())
    if case.id == "multivariate-equicorr-gaussian":
        return cls(d=3)
    if case.id == "multivariate-stochastic-student":
        return cls(d=3, R=_correlation())
    if case.id == "vine-generic":
        return cls.cvine(d=3, order=[0, 1, 2])
    raise AssertionError(f"missing constructor for {case.id}")


def _attach_pair_mle(case: ModelCase, model, observations) -> None:
    if case.id == "bivariate-independent":
        model.fit_result = IndependentResult(
            log_likelihood=0.0,
            method="MLE",
            copula_name=model.name,
            success=True,
        )
    else:
        model.fit_result = MLEResult(
            log_likelihood=0.0,
            method="MLE",
            copula_name=model.name,
            success=True,
            copula_param=_PAIR_PARAMETERS[case.id],
        )
    model._last_u = observations


def _independent_vine_specs(d: int):
    return [
        [(IndependentCopula, 0) for _ in range(d - tree - 1)]
        for tree in range(d - 1)
    ]


def _minimal_conditional_draw(case: ModelCase) -> np.ndarray:
    rng = np.random.default_rng(20260814)
    if case.category == "bivariate":
        model = _public_class(case)()
        observations = _observations(d=2)
        _attach_pair_mle(case, model, observations)
        return model.predict(4, given={0: 0.3}, rng=rng)

    if case.id == "multivariate-gaussian":
        model = GaussianCopula(d=3, R=_correlation())
        model.fit(_observations())
        return model.sample_conditional(4, {0: 0.3}, rng=rng)

    if case.id == "multivariate-student":
        model = StudentCopula(d=3, R=_correlation())
        model.fit(_observations())
        return model.sample_conditional(4, {0: 0.3}, rng=rng)

    if case.id == "multivariate-equicorr-gaussian":
        return EquicorrGaussianCopula(d=3).sample_conditional(
            4, r=0.2, given={0: 0.3}, rng=rng
        )

    if case.id == "multivariate-stochastic-student":
        return StochasticStudentCopula(
            d=3, R=_correlation()
        ).sample_conditional(4, r=6.0, given={0: 0.3}, rng=rng)

    observations = _observations()
    specs = _independent_vine_specs(3)
    if case.id == "vine-generic":
        model = VineCopula.cvine(d=3, order=[0, 1, 2]).fit(
            observations, method="mle", copulas=specs
        )
        return model.predict(4, given={2: 0.3}, rng=rng)
    raise AssertionError(f"missing minimal draw for {case.id}")


def _unsupported_probe(case: UnsupportedCase) -> None:
    if case.probe == "bivariate_gaussian_rotation_90":
        BivariateGaussianCopula(rotate=90)
    elif case.probe == "bivariate_frank_rotation_90":
        FrankCopula(rotate=90)
    elif case.probe == "independent_gas":
        ensure_strategy_supported(IndependentCopula(), "GAS")
    elif case.probe == "static_gaussian_gas":
        ensure_strategy_supported(
            GaussianCopula(d=3, R=_correlation()), "GAS"
        )
    elif case.probe == "static_gaussian_joint_factor":
        GaussianCopula(
            d=4,
            corr_mode="factor",
            factor_rank=2,
            factor_estimation="joint",
        )
    elif case.probe == "static_student_scar_tm_ou":
        ensure_strategy_supported(
            StudentCopula(d=3, R=_correlation()), "SCAR-TM-OU"
        )
    elif case.probe == "equicorr_scar_tm_jacobi":
        ensure_strategy_supported(
            EquicorrGaussianCopula(3), "SCAR-TM-JACOBI"
        )
    elif case.probe == "stochastic_student_scar_tm_jacobi":
        ensure_strategy_supported(
            StochasticStudentCopula(3, R=_correlation()),
            "SCAR-TM-JACOBI",
        )
    elif case.probe == "stochastic_student_joint_factor_gas":
        model = StochasticStudentCopula(
            4,
            corr_mode="factor",
            factor_rank=2,
            factor_estimation="joint",
        )
        ensure_strategy_supported(model, "GAS")
    elif case.probe == "gaussian_cholesky_d50_default":
        GaussianCopula(d=50, corr_mode="cholesky")
    elif case.probe == "student_cholesky_d50_default":
        StudentCopula(d=50, corr_mode="cholesky")
    elif case.probe == "stochastic_student_cholesky_d50_default":
        StochasticStudentCopula(d=50, corr_mode="cholesky")
    elif case.probe == "stochastic_student_estimated_cholesky_gas":
        model = StochasticStudentCopula(d=4, corr_mode="cholesky")
        ensure_strategy_supported(model, "GAS")
    else:
        raise AssertionError(f"missing unsupported probe {case.probe}")


def test_registry_has_all_canonical_runtimes_and_unique_ids():
    assert REGISTRY.schema_version == 1
    assert len(REGISTRY.models) == 11
    assert len(REGISTRY.by_id) == len(REGISTRY.models)
    assert {case.category for case in REGISTRY.models} == {
        "bivariate",
        "multivariate_static",
        "multivariate_dynamic",
        "vine",
    }


def test_registry_has_fifteen_distinct_bivariate_rotation_cells():
    bivariate = [
        case for case in REGISTRY.models if case.category == "bivariate"
    ]
    assert len(bivariate) == 6
    assert sum(len(case.rotations) for case in bivariate) == 15


def test_rvine_name_is_an_alias_not_a_thirteenth_runtime():
    assert RVineCopula is VineCopula
    assert all(case.class_name != "RVineCopula" for case in REGISTRY.models)


def test_conditional_cases_cover_every_registered_model_method_pair():
    cells = REGISTRY.conditional_cases
    assert all(isinstance(cell, ConditionalCase) for cell in cells)
    assert len({cell.id for cell in cells}) == len(cells)
    assert {(cell.model_id, cell.method) for cell in cells} == {
        (model.id, method)
        for model in REGISTRY.models
        for method in model.methods
    }
    for cell in cells:
        model = REGISTRY.by_id[cell.model_id]
        assert cell.entrypoints == model.conditional_entrypoints
        assert cell.exactness == model.exactness
        assert cell.oracles


@pytest.mark.parametrize("case", REGISTRY.models, ids=lambda case: case.id)
def test_registry_matches_public_signatures(case: ModelCase):
    cls = _public_class(case)
    for expected in case.signatures:
        method = getattr(cls, expected.name)
        parameters = tuple(
            name for name in inspect.signature(method).parameters
            if name != "self"
        )
        assert parameters == expected.parameters


@pytest.mark.parametrize("case", REGISTRY.models, ids=lambda case: case.id)
def test_registry_matches_declared_capabilities(case: ModelCase):
    model = _construct_for_capabilities(case)
    if not case.capability_flags:
        assert case.category == "vine"
        return
    pair = is_pair_copula(model)
    gas = strategy_support(model, "GAS")
    scar_ou = strategy_support(model, "SCAR-TM-OU")
    capabilities = {
        "supports_pair_ops": pair,
        "supports_native_point_ops": pair,
        "supports_gas": bool(gas and gas.supported),
        "supports_scar_ou": bool(scar_ou and scar_ou.supported),
        "supports_latent_grid": bool(query_capability(
            model, "row_grid_density_gradient", "SCAR-TM-OU").supported),
        "supports_conditional_sampling": supports_conditional_sampling(model),
        "has_dynamic_scalar_parameter": has_dynamic_scalar_parameter(model),
    }
    for name, expected in case.capability_flags.items():
        assert capabilities[name] is expected, (
            f"{case.id}: capability {name} drifted from the support registry"
        )


@pytest.mark.parametrize("case", REGISTRY.models, ids=lambda case: case.id)
def test_registered_positive_strategies_pass_capability_gate(case: ModelCase):
    if not case.capability_flags:
        pytest.skip("vine strategies are resolved per fitted pair edge")
    model = _construct_for_capabilities(case)
    for method in case.methods:
        ensure_strategy_supported(model, method)


@pytest.mark.parametrize("case", REGISTRY.models, ids=lambda case: case.id)
def test_each_registered_runtime_executes_a_minimal_conditional_draw(
        case: ModelCase):
    samples = _minimal_conditional_draw(case)
    expected_dimension = 2 if case.category == "bivariate" else 3
    assert samples.shape == (4, expected_dimension)
    assert np.all(np.isfinite(samples))
    assert np.all((samples > 0.0) & (samples < 1.0))


_EXCEPTIONS = {
    "TypeError": TypeError,
    "ValueError": ValueError,
    "NotImplementedError": NotImplementedError,
}


@pytest.mark.parametrize(
    "case", REGISTRY.unsupported, ids=lambda case: case.id
)
def test_registered_unsupported_combinations_fail_early(
        case: UnsupportedCase):
    assert case.model_id in REGISTRY.by_id
    with pytest.raises(_EXCEPTIONS[case.expected_exception]):
        _unsupported_probe(case)
