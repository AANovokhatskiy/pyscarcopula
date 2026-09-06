"""Validation matrix for ``d=50`` conditional sampling."""

from __future__ import annotations

from functools import lru_cache
import os

import numpy as np
import pytest
from scipy.stats import norm

from pyscarcopula import EquicorrGaussianCopula, StochasticStudentCopula
from pyscarcopula._types import GASResult, LatentResult, MLEResult, gas_params, ou_params
from pyscarcopula.api import predict as api_predict

from ._analytical_oracles import (
    gaussian_conditional_parameters,
    student_conditional_parameters,
)
from ._high_dimensional import (
    DIMENSION,
    FREE_COUNTS,
    configured_gaussian,
    configured_student,
    dense_correlation,
    factor_correlation,
    factor_loadings,
    high_dimensional_gaussian_vine,
    high_dimensional_mixed_truncated_vine,
    regular_vine_structure,
    scattered_given,
    suffix_given,
)
from ._statistical_assertions import (
    assert_covariance_with_whitening,
    assert_mean_with_mc_error,
)
from ._vine_oracles import (
    gaussian_vine_correlation,
    pyvine_exact_suffix_sample,
)


DENSE_KINDS = ("ar1", "block", "near-singular-moderate")
FACTOR_RANKS = (1, 3, 8)
STUDENT_DF = (3.0, 8.0, 30.0)
STUDENT_FACTOR_CASES = tuple(
    (rank, df) for rank in FACTOR_RANKS for df in (5.0, 10.0)
)
EQUICORR_REGIMES = (
    ("negative", -0.018),
    ("zero", 0.0),
    ("positive", 0.65),
)


def _case_id(prefix: str, *parts) -> str:
    return "-".join((prefix, *(str(part) for part in parts), "d=50"))


def _assert_fixed_bit_exact(samples, given) -> None:
    assert samples.shape[1] == DIMENSION
    assert np.all(np.isfinite(samples))
    assert np.all((samples > 0.0) & (samples < 1.0))
    for index, value in given.items():
        np.testing.assert_array_equal(
            samples[:, index],
            np.full(len(samples), value, dtype=np.float64),
        )


def _assert_gaussian_oracle(samples, correlation, given) -> None:
    oracle = gaussian_conditional_parameters(correlation, given)
    latent = norm.ppf(np.clip(
        samples[:, oracle.free_indices], 1e-12, 1.0 - 1e-12
    ))
    assert_mean_with_mc_error(
        latent,
        oracle.mean,
        oracle.covariance,
        sigma=7.0,
        numerical_floor=0.004,
    )
    assert_covariance_with_whitening(
        latent,
        oracle.mean,
        oracle.covariance,
        sigma=8.0,
        numerical_floor=0.025,
    )
    _assert_fixed_bit_exact(samples, given)


def _assert_student_oracle(samples, correlation, df, given) -> None:
    oracle = student_conditional_parameters(correlation, df, given)
    latent = oracle.latent_from_copula(samples)
    assert_mean_with_mc_error(
        latent,
        oracle.location,
        oracle.covariance,
        sigma=8.0,
        numerical_floor=0.006,
    )
    assert_covariance_with_whitening(
        latent,
        oracle.location,
        oracle.covariance,
        sigma=10.0,
        numerical_floor=0.035,
    )
    _assert_fixed_bit_exact(samples, given)


@pytest.mark.validation
@pytest.mark.high_dimensional
@pytest.mark.parametrize(
    "kind", DENSE_KINDS, ids=lambda kind: _case_id("gaussian-dense", kind)
)
@pytest.mark.parametrize("k_free", FREE_COUNTS, ids=lambda value: f"k_free={value}")
def test_d50_gaussian_dense_matrix_matches_conditional_mvn(kind, k_free):
    model = configured_gaussian(kind)
    given = scattered_given(k_free)
    samples = model.sample_conditional(
        5_000,
        given,
        rng=np.random.default_rng(20267010 + k_free),
        n_threads=4,
    )
    _assert_gaussian_oracle(samples, dense_correlation(kind), given)


@pytest.mark.validation
@pytest.mark.high_dimensional
@pytest.mark.parametrize(
    "rank", FACTOR_RANKS, ids=lambda rank: _case_id("gaussian-factor", f"rank={rank}")
)
@pytest.mark.parametrize("k_free", FREE_COUNTS, ids=lambda value: f"k_free={value}")
def test_d50_gaussian_factor_matrix_matches_dense_mvn_oracle(rank, k_free):
    model = configured_gaussian("factor", rank)
    given = scattered_given(k_free)
    samples = model.sample_conditional(
        5_000,
        given,
        rng=np.random.default_rng(20267020 + 10 * rank + k_free),
        n_threads=4,
    )
    _assert_gaussian_oracle(samples, factor_correlation(rank), given)


@pytest.mark.validation
@pytest.mark.high_dimensional
@pytest.mark.parametrize(
    "df", STUDENT_DF, ids=lambda df: _case_id("student-dense", f"df={df:g}")
)
@pytest.mark.parametrize("k_free", FREE_COUNTS, ids=lambda value: f"k_free={value}")
def test_d50_student_dense_df_matrix_matches_conditional_t(df, k_free):
    model = configured_student(df=df, kind="ar1")
    given = scattered_given(k_free)
    samples = model.sample_conditional(
        7_000,
        given,
        rng=np.random.default_rng(20267030 + int(df) + k_free),
        n_threads=4,
    )
    _assert_student_oracle(samples, dense_correlation("ar1"), df, given)


@pytest.mark.validation
@pytest.mark.high_dimensional
@pytest.mark.parametrize(
    ("rank", "df"),
    STUDENT_FACTOR_CASES,
    ids=lambda value: str(value),
)
@pytest.mark.parametrize("k_free", FREE_COUNTS, ids=lambda value: f"k_free={value}")
def test_d50_student_factor_rank_df_matrix_matches_dense_t_oracle(
    rank, df, k_free
):
    model = configured_student(df=df, rank=rank)
    given = scattered_given(k_free)
    samples = model.sample_conditional(
        7_000,
        given,
        rng=np.random.default_rng(
            20267040 + 100 * rank + 10 * int(df) + k_free
        ),
        n_threads=4,
    )
    _assert_student_oracle(samples, factor_correlation(rank), df, given)


@pytest.mark.validation
@pytest.mark.high_dimensional
@pytest.mark.parametrize(
    ("regime", "rho"),
    EQUICORR_REGIMES,
    ids=[_case_id("equicorr", name) for name, _rho in EQUICORR_REGIMES],
)
@pytest.mark.parametrize("k_free", FREE_COUNTS, ids=lambda value: f"k_free={value}")
def test_d50_equicorr_parameter_regimes_match_conditional_mvn(
    regime, rho, k_free
):
    del regime
    model = EquicorrGaussianCopula(DIMENSION)
    given = scattered_given(k_free)
    samples = model.sample_conditional(
        5_000,
        r=rho,
        given=given,
        rng=np.random.default_rng(20267050 + k_free),
        n_threads=4,
    )
    correlation = np.full((DIMENSION, DIMENSION), rho, dtype=np.float64)
    np.fill_diagonal(correlation, 1.0)
    _assert_gaussian_oracle(samples, correlation, given)


def _dynamic_result(model, method: str, family: str):
    parameter = 0.18 if family == "equicorr" else 7.0
    if method == "MLE":
        return MLEResult(
            log_likelihood=0.0,
            method="MLE",
            copula_name=model.name,
            success=True,
            copula_param=parameter,
        )
    if method == "GAS":
        return GASResult(
            log_likelihood=0.0,
            method="GAS",
            copula_name=model.name,
            success=True,
            params=gas_params(0.04, 0.03, 0.70),
            scaling="unit",
            r_last=parameter,
        )
    if method == "SCAR-TM-OU":
        return LatentResult(
            log_likelihood=0.0,
            method="SCAR-TM-OU",
            copula_name=model.name,
            success=True,
            params=ou_params(1.7, 0.35, 0.45),
            K=21,
            grid_range=3.5,
            adaptive=False,
            grid_method="dense",
        )
    raise ValueError(method)


def _dynamic_model(family: str, corr_mode: str):
    if family == "equicorr":
        if corr_mode != "equicorr":
            raise ValueError("Equicorr has no fixed/factor axis")
        return EquicorrGaussianCopula(DIMENSION)
    if corr_mode == "fixed":
        return StochasticStudentCopula(DIMENSION, R=dense_correlation("ar1"))
    if corr_mode == "factor":
        return StochasticStudentCopula(
            DIMENSION,
            corr_mode="factor",
            factor_rank=3,
            factor_loadings=factor_loadings(3),
            factor_tile_size=13,
        )
    raise ValueError(corr_mode)


@pytest.mark.validation
@pytest.mark.high_dimensional
@pytest.mark.parametrize("method", ["MLE", "GAS", "SCAR-TM-OU"])
@pytest.mark.parametrize("k_free", FREE_COUNTS, ids=lambda value: f"k_free={value}")
def test_d50_equicorr_mle_gas_tm_public_prediction_contract(method, k_free):
    model = _dynamic_model("equicorr", "equicorr")
    history = np.random.default_rng(20267060).uniform(
        0.08, 0.92, size=(7, DIMENSION)
    )
    given = scattered_given(k_free)
    result = _dynamic_result(model, method, "equicorr")
    samples = api_predict(
        model,
        history,
        result,
        97,
        given=given,
        horizon="next",
        predictive_r_mode="grid" if method == "SCAR-TM-OU" else None,
        rng=np.random.default_rng(20267061 + k_free),
        n_threads=3,
    )
    _assert_fixed_bit_exact(samples, given)


@pytest.mark.validation
@pytest.mark.high_dimensional
@pytest.mark.parametrize("corr_mode", ["fixed", "factor"])
@pytest.mark.parametrize("method", ["MLE", "GAS", "SCAR-TM-OU"])
@pytest.mark.parametrize("k_free", FREE_COUNTS, ids=lambda value: f"k_free={value}")
def test_d50_stochastic_student_modes_and_methods_public_contract(
    corr_mode, method, k_free
):
    model = _dynamic_model("student", corr_mode)
    history = np.random.default_rng(20267070).uniform(
        0.08, 0.92, size=(7, DIMENSION)
    )
    given = scattered_given(k_free)
    result = _dynamic_result(model, method, "student")
    samples = api_predict(
        model,
        history,
        result,
        97,
        given=given,
        horizon="next",
        predictive_r_mode="grid" if method == "SCAR-TM-OU" else None,
        rng=np.random.default_rng(20267071 + k_free),
        n_threads=3,
    )
    _assert_fixed_bit_exact(samples, given)


@pytest.mark.high_dimensional
@pytest.mark.parametrize("model_kind", ["gaussian", "student", "stochastic-student"])
def test_d50_factor_paths_do_not_materialize_dense_correlation(
    model_kind, monkeypatch
):
    if model_kind == "gaussian":
        model = configured_gaussian("factor", 3)
        draw = lambda: model.sample_conditional(
            31, scattered_given(5), rng=np.random.default_rng(20267080)
        )
    elif model_kind == "student":
        model = configured_student(df=7.0, rank=3)
        draw = lambda: model.sample_conditional(
            31, scattered_given(5), rng=np.random.default_rng(20267080)
        )
    else:
        model = _dynamic_model("student", "factor")
        draw = lambda: model.sample_conditional(
            31,
            r=7.0,
            given=scattered_given(5),
            rng=np.random.default_rng(20267080),
        )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("factor conditional path materialized dense R")

    monkeypatch.setattr(model, "to_correlation_matrix", forbidden)
    sample = draw()
    assert sample.shape == (31, DIMENSION)
    assert getattr(model, "_R", None) is None


@pytest.mark.high_dimensional
@pytest.mark.parametrize("family", ["gaussian", "equicorr", "student"])
def test_d50_memory_budget_contract_precedes_sampling(family):
    n = 11
    required = n * DIMENSION * 8
    given = scattered_given(3)
    if family == "gaussian":
        model = configured_gaussian("ar1")
        kwargs = {}
    elif family == "equicorr":
        model = EquicorrGaussianCopula(DIMENSION)
        kwargs = {"r": 0.2}
    else:
        model = StochasticStudentCopula(DIMENSION, R=dense_correlation("ar1"))
        kwargs = {"r": 7.0}
    sample = model.sample_conditional(
        n,
        given=given,
        memory_budget_bytes=required,
        rng=np.random.default_rng(20267090),
        **kwargs,
    )
    assert sample.shape == (n, DIMENSION)
    with pytest.raises(MemoryError, match="requires"):
        model.sample_conditional(
            n,
            given=given,
            memory_budget_bytes=required - 1,
            rng=np.random.default_rng(20267090),
            **kwargs,
        )


@lru_cache(maxsize=None)
def _vine_correlation(kind: str) -> np.ndarray:
    return gaussian_vine_correlation(high_dimensional_gaussian_vine(kind))


@pytest.mark.validation
@pytest.mark.high_dimensional
@pytest.mark.parametrize("kind", ["c-vine", "d-vine", "r-vine"])
@pytest.mark.parametrize("k_free", FREE_COUNTS, ids=lambda value: f"k_free={value}")
def test_d50_generic_gaussian_c_d_r_exact_suffix_matches_mvn(
    kind, k_free
):
    vine = high_dimensional_gaussian_vine(kind)
    given = suffix_given(vine, k_free)
    samples, diagnostics = vine.predict(
        3_000,
        given=given,
        return_diagnostics=True,
        rng=np.random.default_rng(20267100 + k_free),
    )
    assert diagnostics["conditional_method"] == "suffix"
    assert diagnostics["matrix_rebuilt"] is False
    assert "mcmc" not in diagnostics
    _assert_gaussian_oracle(samples, _vine_correlation(kind), given)


@pytest.mark.external
@pytest.mark.validation
@pytest.mark.high_dimensional
@pytest.mark.parametrize("kind", ["c-vine", "d-vine", "r-vine"])
@pytest.mark.parametrize("k_free", FREE_COUNTS, ids=lambda value: f"k_free={value}")
def test_d50_generic_gaussian_exact_suffix_matches_pyvine_pointwise(
    kind, k_free
):
    pv = pytest.importorskip("pyvinecopulib")
    vine = high_dimensional_gaussian_vine(kind)
    given = suffix_given(vine, k_free)
    seed = 20267105 + k_free
    production = vine.predict(
        41,
        given=given,
        rng=np.random.default_rng(seed),
    )
    reference, _model = pyvine_exact_suffix_sample(
        vine, 41, given, seed, pv
    )
    _assert_fixed_bit_exact(production, given)
    np.testing.assert_allclose(production, reference, rtol=0.0, atol=3e-8)


@pytest.mark.validation
@pytest.mark.high_dimensional
@pytest.mark.parametrize("k_free", FREE_COUNTS, ids=lambda value: f"k_free={value}")
def test_d50_mixed_rotated_truncated_rvine_exact_suffix_contract(k_free):
    vine = high_dimensional_mixed_truncated_vine()
    given = suffix_given(vine, k_free)
    samples, diagnostics = vine.predict(
        511,
        given=given,
        return_diagnostics=True,
        rng=np.random.default_rng(20267110 + k_free),
    )
    assert diagnostics["conditional_method"] == "suffix"
    assert diagnostics["matrix_rebuilt"] is False
    assert "mcmc" not in diagnostics
    _assert_fixed_bit_exact(samples, given)


@pytest.mark.benchmark
@pytest.mark.high_dimensional
@pytest.mark.parametrize("k_free", FREE_COUNTS, ids=lambda value: f"k_free={value}")
def test_d50_gaussian_dag_mcmc_stress_is_measurement_only(
    k_free, monkeypatch
):
    if os.environ.get("PYSCA_RUN_BENCHMARKS") != "1":
        pytest.skip("set PYSCA_RUN_BENCHMARKS=1 to run benchmark checks")
    vine = high_dimensional_gaussian_vine("d-vine")
    given = suffix_given(vine, k_free)
    monkeypatch.setattr(vine, "_suffix_sampling_state", lambda _given: None)
    samples, diagnostics = vine.predict(
        8,
        given=given,
        mcmc_steps=max(20, 4 * k_free),
        mcmc_burnin=max(10, 2 * k_free),
        return_diagnostics=True,
        rng=np.random.default_rng(20267140 + k_free),
    )
    assert diagnostics["conditional_method"] == "dag_mcmc"
    assert diagnostics["mcmc"]["n_free"] == k_free
    assert diagnostics["mcmc"]["step_unit"] == "single_coordinate_update"
    _assert_fixed_bit_exact(samples, given)


def test_d50_rvine_fixture_is_neither_c_vine_nor_d_vine():
    # This assertion guards the C/D/R structure axis against accidentally
    # replacing the R-vine fixture with a relabelled special case.
    from pyscarcopula.vine.vine import _structure_kinds

    assert _structure_kinds(regular_vine_structure()) == frozenset({"rvine"})
