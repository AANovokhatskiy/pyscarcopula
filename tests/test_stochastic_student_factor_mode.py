"""Phase 9.4 model adapter contracts for factor Student correlation."""

import inspect

import numpy as np
import pytest

from pyscarcopula import FactorStudentEvaluator, StochasticStudentCopula
from pyscarcopula._types import MLEResult
from pyscarcopula.contrib.risk_metrics import _get_copula_constructor
from pyscarcopula.numerical import _cpp_scar_ou
from pyscarcopula.numerical._scar_ou_config import AutoTMConfig
from pyscarcopula.stattests import (
    factor_student_rosenblatt_transform,
    stochastic_student_rosenblatt_transform,
    student_rosenblatt_transform,
)


def _loadings(d=6, k=2):
    rng = np.random.default_rng(9401)
    return rng.normal(scale=0.12, size=(d, k))


def _observations(rows=5, d=6):
    return np.random.default_rng(9402).uniform(0.08, 0.92, size=(rows, d))


def test_factor_constructor_exposes_compact_read_only_contract():
    loadings = _loadings()
    model = StochasticStudentCopula(
        6,
        corr_mode="factor",
        factor_rank=2,
        factor_loadings=loadings,
    )

    np.testing.assert_allclose(model.factor_loadings_, loadings)
    np.testing.assert_allclose(
        model.factor_uniqueness_,
        1.0 - np.sum(loadings * loadings, axis=1),
    )
    assert model.factor_rank == 2
    assert model.correlation_operator_.rank == 2
    assert model._R is None
    assert model._L is None
    with pytest.raises(RuntimeError, match="not materialized"):
        _ = model.R
    np.testing.assert_allclose(
        model.to_correlation_matrix(),
        model.correlation_operator_.to_dense(),
    )


@pytest.mark.parametrize(
    "kwargs, error, message",
    [
        (
            {"corr_mode": "factor", "factor_rank": None},
            TypeError,
            "factor_rank",
        ),
        (
            {"corr_mode": "factor", "factor_rank": 6},
            ValueError,
            "1 <= k < d",
        ),
        (
            {
                "corr_mode": "factor",
                "factor_rank": 2,
                "R": np.eye(6),
            },
            ValueError,
            "forbidden",
        ),
        (
            {
                "corr_mode": "factor",
                "factor_rank": 2,
                "factor_loadings": np.ones((6, 3)),
            },
            ValueError,
            "shape",
        ),
        (
            {
                "corr_mode": "factor",
                "factor_rank": 2,
                "factor_estimation": "joint",
                "factor_joint_max_params": 5,
            },
            ValueError,
            "factor_joint_max_params",
        ),
    ],
)
def test_factor_constructor_rejects_ambiguous_or_unsafe_inputs(
        kwargs, error, message):
    with pytest.raises(error, match=message):
        StochasticStudentCopula(6, **kwargs)


def test_factor_model_row_and_grid_paths_match_dense_reference():
    loadings = _loadings()
    observations = _observations()
    factor = StochasticStudentCopula(
        6,
        corr_mode="factor",
        factor_rank=2,
        factor_loadings=loadings,
        factor_tile_size=3,
    )
    dense = StochasticStudentCopula(
        6, R=factor.to_correlation_matrix())

    factor_rows = factor.log_pdf_and_dlog_dr_rows(
        observations, 5.5, n_threads=2)
    dense_rows = dense.log_pdf_and_dlog_dr_rows(
        observations, 5.5, n_threads=2)
    np.testing.assert_allclose(
        factor_rows[0], dense_rows[0], rtol=4e-10, atol=4e-10)
    np.testing.assert_allclose(
        factor_rows[1], dense_rows[1], rtol=2e-8, atol=2e-8)

    grid = np.array([-1.0, 0.25, 1.5])
    factor_grid = factor.pdf_and_grad_on_grid_batch(
        observations, grid, n_threads=2)
    direct_density, direct_df_gradient = FactorStudentEvaluator(
        factor.correlation_operator_, observations
    ).pdf_and_grad_on_grid(
        factor.transform(grid),
        dimension_tile=3,
        n_threads=2,
    )
    np.testing.assert_allclose(
        factor_grid[0], direct_density, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(
        factor_grid[1],
        direct_df_gradient * factor.dtransform(grid)[None, :],
        rtol=2e-12,
        atol=2e-12,
    )
    dense_grid = dense.pdf_and_grad_on_grid_batch(
        observations, grid, n_threads=2)
    np.testing.assert_allclose(
        factor_grid[0], dense_grid[0], rtol=2e-6, atol=2e-8)
    np.testing.assert_allclose(
        factor_grid[1], dense_grid[1], rtol=1e-3, atol=5e-6)


def test_factor_model_native_smoothed_weights_match_dense_reference():
    observations = _observations(rows=17)
    factor = StochasticStudentCopula(
        6,
        corr_mode="factor",
        factor_rank=2,
        factor_loadings=_loadings(),
        factor_tile_size=3,
    )
    dense = StochasticStudentCopula(
        6, R=factor.to_correlation_matrix())
    config = AutoTMConfig(
        K=29,
        grid_range=3.0,
        adaptive=False,
        transition_method="matrix",
        grid_method="dense",
        max_K=None,
    )

    factor_grid, factor_weights = (
        _cpp_scar_ou.smoothed_state_distribution(
            1.1, 0.4, 0.7, observations, factor, config)
    )
    dense_grid, dense_weights = (
        _cpp_scar_ou.smoothed_state_distribution(
            1.1, 0.4, 0.7, observations, dense, config)
    )

    np.testing.assert_array_equal(factor_grid, dense_grid)
    np.testing.assert_allclose(
        factor_weights, dense_weights, rtol=2e-6, atol=2e-8)


@pytest.mark.parametrize(
    "df",
    [
        5.5,
        np.linspace(4.5, 7.5, 5),
    ],
)
def test_factor_student_rosenblatt_matches_dense_reference(df):
    observations = _observations()
    factor = StochasticStudentCopula(
        6,
        corr_mode="factor",
        factor_rank=2,
        factor_loadings=_loadings(),
    )

    actual = factor_student_rosenblatt_transform(
        factor.correlation_operator_, df, observations)
    if np.ndim(df) == 0:
        expected = student_rosenblatt_transform(
            factor.to_correlation_matrix(), df, observations)
    else:
        expected = np.vstack([
            student_rosenblatt_transform(
                factor.to_correlation_matrix(),
                float(row_df),
                observations[index:index + 1],
            )
            for index, row_df in enumerate(df)
        ])

    # The compact Woodbury path and the dense inverse use different valid
    # floating-point operation orders.  Keep an eleven-digit parity gate that
    # is stable across BLAS implementations and sanitizer builds.
    np.testing.assert_allclose(actual, expected, rtol=3e-11, atol=3e-12)
    assert factor._R is None


def test_factor_student_native_scar_rosenblatt_matches_dense_reference():
    observations = _observations(rows=8)
    factor = StochasticStudentCopula(
        6,
        corr_mode="factor",
        factor_rank=2,
        factor_loadings=_loadings(),
    )
    dense = StochasticStudentCopula(
        6, R=factor.to_correlation_matrix())
    config = AutoTMConfig(
        K=9,
        grid_range=3.0,
        adaptive=False,
        transition_method="matrix",
        max_K=None,
    )

    actual = _cpp_scar_ou.student_rosenblatt(
        0.8, 0.1, 0.7, observations, factor, config)
    expected = _cpp_scar_ou.student_rosenblatt(
        0.8, 0.1, 0.7, observations, dense, config)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=3e-10)
    assert factor._R is None


def test_factor_student_gof_supports_dimension_above_dense_limit():
    dimension = 2049
    factor = StochasticStudentCopula(
        dimension,
        corr_mode="factor",
        factor_rank=1,
        factor_loadings=np.zeros((dimension, 1)),
    )
    observations = np.full((2, dimension), 0.5)
    result = MLEResult(
        log_likelihood=0.0,
        method="MLE",
        copula_name=factor.name,
        success=True,
        copula_param=5.5,
    )

    transformed = stochastic_student_rosenblatt_transform(
        factor, observations, result)

    assert transformed.shape == observations.shape
    assert np.all(np.isfinite(transformed))
    assert factor._R is None


def test_two_stage_initialization_is_deterministic_and_matrix_free():
    observations = _observations(rows=32, d=8)
    first = StochasticStudentCopula(
        8,
        corr_mode="factor",
        factor_rank=2,
        factor_tile_size=3,
        factor_seed=17,
        factor_oversampling=3,
    )
    second = StochasticStudentCopula(
        8,
        corr_mode="factor",
        factor_rank=2,
        factor_tile_size=3,
        factor_seed=17,
        factor_oversampling=3,
    )

    first.initialize_factor(observations)
    second.initialize_factor(observations)

    np.testing.assert_allclose(
        first.factor_loadings_, second.factor_loadings_, atol=0.0, rtol=0.0)
    assert first._R is None
    assert first._L is None
    diagnostics = first.factor_diagnostics()
    assert diagnostics["initialization_source"] == (
        "two_stage_randomized_svd")
    assert diagnostics["factor_n_params"] == 15
    assert diagnostics["factor_initialized"] is True


def test_two_stage_raw_data_requires_explicit_pobs_conversion():
    model = StochasticStudentCopula(
        6, corr_mode="factor", factor_rank=2)
    raw = np.random.default_rng(9403).normal(size=(24, 6))

    with pytest.raises(ValueError, match="to_pobs=True"):
        model.initialize_factor(raw)

    model.initialize_factor(raw, to_pobs=True)
    assert model.factor_diagnostics()["factor_initialized"] is True


def test_factor_model_persistence_rebuilds_compact_operator(tmp_path):
    model = StochasticStudentCopula(
        6,
        corr_mode="factor",
        factor_rank=2,
        factor_loadings=_loadings(),
        factor_seed=19,
        factor_oversampling=4,
    )
    path = tmp_path / "factor-student.json"
    model.save(path)

    restored = StochasticStudentCopula.load(path)

    assert restored._R is None
    assert restored._L is None
    np.testing.assert_allclose(
        restored.factor_loadings_, model.factor_loadings_)
    np.testing.assert_allclose(
        restored.log_pdf_rows(_observations(), 6.0),
        model.log_pdf_rows(_observations(), 6.0),
    )
    assert restored.factor_diagnostics()[
        "initialization_source"] == "supplied"


def test_factor_sampling_is_available_without_dense_correlation():
    model = StochasticStudentCopula(
        6,
        corr_mode="factor",
        factor_rank=2,
        factor_loadings=_loadings(),
    )
    samples = model.sample_at_parameter(
        2, 5.0, rng=np.random.default_rng(901))
    assert samples.shape == (2, 6)
    assert np.all((samples > 0.0) & (samples < 1.0))
    assert model._R is None
    assert model.fit_result is None


def test_rolling_worker_preserves_factor_constructor_policy():
    model = StochasticStudentCopula(
        6,
        corr_mode="factor",
        factor_rank=2,
        factor_loadings=_loadings(),
        factor_tile_size=5,
        factor_seed=21,
        factor_oversampling=2,
    )

    model_type, kwargs = _get_copula_constructor(model)
    worker = model_type(**kwargs)

    assert worker.corr_mode == "factor"
    assert worker.factor_tile_size == 5
    assert worker._factor_seed == 21
    assert worker._factor_oversampling == 2
    np.testing.assert_allclose(
        worker.factor_loadings_, model.factor_loadings_)


@pytest.mark.parametrize(
    "method",
    [
        "log_likelihood",
        "log_pdf_rows",
        "dlog_pdf_dr_rows",
        "log_pdf_and_dlog_dr_rows",
        "pdf_on_grid",
        "pdf_and_grad_on_grid",
        "pdf_and_grad_on_grid_batch",
        "copula_grid_batch",
    ],
)
def test_factor_capable_methods_keep_absolute_single_thread_default(method):
    parameter = inspect.signature(
        getattr(StochasticStudentCopula, method)).parameters["n_threads"]
    assert parameter.default == 1
