"""Phase-8 contracts for compact equicorrelation preparation."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest
from scipy.stats import norm

from pyscarcopula import (
    EquicorrGaussianCopula,
    EquicorrPreparedData,
    StochasticStudentCopula,
)
from pyscarcopula import api
from pyscarcopula._constants import PSEUDO_OBS_EPS
from pyscarcopula.numerical import multivariate_native
from pyscarcopula.numerical import static_likelihood
from pyscarcopula.numerical import _cpp_gas
from pyscarcopula.numerical import _cpp_scar_ou
from pyscarcopula.numerical._scar_ou_config import AutoTMConfig


def test_native_statistics_match_dense_reference_and_report_clipping():
    u = np.array([
        [0.0, 0.2, 0.5, 0.8, 1.0],
        [0.1, 0.3, 0.6, 0.7, 0.9],
    ])
    sum_z, sum_z2, diagnostics = (
        multivariate_native.prepare_equicorr_statistics(
            u, dimension_tile=2))
    z = norm.ppf(np.clip(u, PSEUDO_OBS_EPS, 1.0 - PSEUDO_OBS_EPS))

    # The native Acklam quantile plus Newton refinement differs from SciPy by
    # a few nanounits at the deliberately clipped 1e-10 boundaries.
    np.testing.assert_allclose(sum_z, z.sum(axis=1), rtol=1e-10, atol=5e-9)
    np.testing.assert_allclose(
        sum_z2, np.square(z).sum(axis=1), rtol=1e-9, atol=5e-9)
    assert diagnostics["clipping_events"] == 2
    assert diagnostics["nonfinite_values"] == 0
    assert diagnostics["temporary_values"] == 12


@pytest.mark.parametrize(
    ("shape", "expected_axis"),
    [((2, 8192), "dimension_tiles"), ((32, 512), "rows")],
)
def test_thread_count_does_not_change_reduction(shape, expected_axis):
    rng = np.random.default_rng(90210)
    u = rng.uniform(0.001, 0.999, size=shape)
    sequential = multivariate_native.prepare_equicorr_statistics(
        u, dimension_tile=256, n_threads=1)
    parallel = multivariate_native.prepare_equicorr_statistics(
        u, dimension_tile=256, n_threads=4)

    np.testing.assert_array_equal(parallel[0], sequential[0])
    np.testing.assert_array_equal(parallel[1], sequential[1])
    assert parallel[2]["parallel_axis"] == expected_axis
    assert parallel[2]["parallel_blocks"] > 1


def test_model_streams_blocks_and_retains_only_o_t_statistics():
    rng = np.random.default_rng(17)
    u = rng.uniform(0.01, 0.99, size=(11, 257))
    model = EquicorrGaussianCopula(d=257)

    dense = model.prepare_sufficient_statistics(
        u, batch_rows=4, dimension_tile=32, n_threads=1)
    streamed = model.prepare_sufficient_statistics(
        (u[:3], u[3:8], u[8:]),
        batch_rows=2,
        dimension_tile=32,
        n_threads=4,
    )

    assert isinstance(dense, EquicorrPreparedData)
    np.testing.assert_array_equal(streamed.sum_z, dense.sum_z)
    np.testing.assert_array_equal(streamed.sum_z2, dense.sum_z2)
    assert dense.sum_z.shape == (len(u),)
    assert dense.sum_z2.shape == (len(u),)
    assert not hasattr(dense, "u")
    assert (
        dense.diagnostics["peak_temporary_values"]
        <= 2 * 4 * int(np.ceil(257 / 32))
    )
    with pytest.raises(ValueError):
        dense.sum_z[0] = 0.0


def test_static_likelihood_consumes_prepared_statistics_directly():
    rng = np.random.default_rng(23)
    u = rng.uniform(0.01, 0.99, size=(37, 11))
    model = EquicorrGaussianCopula(d=11)
    prepared = model.prepare_sufficient_statistics(
        u, batch_rows=7, dimension_tile=4, n_threads=4)
    dense_evaluator = static_likelihood.prepare(model, u, n_threads=1)
    prepared_evaluator = static_likelihood.prepare(
        model, prepared, n_threads=4)

    for rho in (-0.05, 0.0, 0.65):
        dense_result = dense_evaluator.result(rho)
        prepared_result = prepared_evaluator.result(rho)
        np.testing.assert_allclose(
            prepared_result["negative_log_likelihood"],
            dense_result["negative_log_likelihood"],
            rtol=2e-14,
            atol=2e-12,
        )
        np.testing.assert_allclose(
            prepared_result["negative_gradient"],
            dense_result["negative_gradient"],
            rtol=2e-14,
            atol=2e-12,
        )
        np.testing.assert_allclose(
            prepared_evaluator.log_pdf_rows(rho),
            dense_evaluator.log_pdf_rows(rho),
            rtol=2e-14,
            atol=2e-12,
        )


def test_row_and_grid_apis_consume_prepared_statistics_directly():
    rng = np.random.default_rng(27)
    u = rng.uniform(0.01, 0.99, size=(1300, 7))
    grid = np.linspace(-2.0, 2.0, 220)
    model = EquicorrGaussianCopula(d=7)
    prepared = model.prepare_sufficient_statistics(
        u, batch_rows=128, dimension_tile=3, n_threads=4)

    dense_rows = model.log_pdf_and_dlog_dr_rows(
        u, r=0.3, n_threads=1)
    prepared_rows = model.log_pdf_and_dlog_dr_rows(
        prepared, r=0.3, n_threads=4)
    for actual, expected in zip(prepared_rows, dense_rows, strict=True):
        np.testing.assert_allclose(
            actual, expected, rtol=2e-14, atol=2e-12)

    dense_grid = model.pdf_and_grad_on_grid_batch(
        u, grid, n_threads=1)
    sequential = model.pdf_and_grad_on_grid_batch(
        prepared, grid, n_threads=1)
    parallel = model.pdf_and_grad_on_grid_batch(
        prepared, grid, n_threads=4)
    for dense_values, sequential_values, parallel_values in zip(
            dense_grid, sequential, parallel, strict=True):
        np.testing.assert_allclose(
            sequential_values, dense_values, rtol=5e-12, atol=1e-12)
        np.testing.assert_array_equal(parallel_values, sequential_values)


def test_grid_output_budget_and_row_batches():
    rng = np.random.default_rng(28)
    u = rng.uniform(0.01, 0.99, size=(13, 5))
    grid = np.linspace(-1.0, 1.0, 9)
    model = EquicorrGaussianCopula(d=5)
    prepared = model.prepare_sufficient_statistics(u)
    expected = model.pdf_and_grad_on_grid_batch(prepared, grid)

    blocks = list(model.pdf_and_grad_on_grid_batches(
        prepared,
        grid,
        batch_rows=4,
        memory_budget_bytes=2 * 4 * len(grid) * 8,
    ))
    for index in range(2):
        np.testing.assert_array_equal(
            np.concatenate([block[index] for block in blocks]),
            expected[index],
        )

    required = 2 * len(prepared) * len(grid) * 8
    with pytest.raises(
            MemoryError, match="pdf_and_grad_on_grid_batches"):
        model.pdf_and_grad_on_grid_batch(
            prepared,
            grid,
            memory_budget_bytes=required - 1,
        )
    with pytest.raises(MemoryError, match="reduce batch_rows"):
        list(model.pdf_and_grad_on_grid_batches(
            prepared,
            grid,
            batch_rows=4,
            memory_budget_bytes=2 * 4 * len(grid) * 8 - 1,
        ))


def test_mle_accepts_prepared_data_without_dense_correlation_result():
    rng = np.random.default_rng(29)
    u = rng.uniform(0.05, 0.95, size=(80, 8))
    model = EquicorrGaussianCopula(d=8)
    prepared = model.prepare_sufficient_statistics(u)

    result = model.fit(prepared, method="MLE")

    assert np.isfinite(result.log_likelihood)
    assert result.correlation_matrix is None
    assert result.diagnostics["corr_matrix"] is None
    assert (
        result.diagnostics["correlation_representation"]
        == "equicorrelation_scalar"
    )
    assert np.isfinite(result.diagnostics["equicorrelation_rho"])


def test_fitted_mle_sampling_and_prediction_batches_are_bounded():
    rng = np.random.default_rng(291)
    u = rng.uniform(0.05, 0.95, size=(40, 6))
    model = EquicorrGaussianCopula(d=6)
    prepared = model.prepare_sufficient_statistics(u)
    model.fit(prepared, method="MLE")

    sample_blocks = list(model.sample_batches(
        11,
        batch_rows=4,
        given={0: 0.25},
        memory_budget_bytes=4 * 6 * 8,
        rng=np.random.default_rng(292),
    ))
    predict_blocks = list(model.predict_batches(
        11,
        batch_rows=4,
        memory_budget_bytes=4 * 6 * 8,
        rng=np.random.default_rng(293),
    ))
    assert [block.shape for block in sample_blocks] == [
        (4, 6), (4, 6), (3, 6)]
    assert [block.shape for block in predict_blocks] == [
        (4, 6), (4, 6), (3, 6)]
    np.testing.assert_array_equal(
        np.concatenate(sample_blocks)[:, 0],
        np.full(11, 0.25),
    )

    with pytest.raises(MemoryError, match="sample_batches"):
        model.sample(11, memory_budget_bytes=11 * 6 * 8 - 1)
    with pytest.raises(MemoryError, match="predict_batches"):
        model.predict(11, memory_budget_bytes=11 * 6 * 8 - 1)
    with pytest.raises(MemoryError, match="reduce batch_rows"):
        list(model.sample_batches(
            11, batch_rows=4, memory_budget_bytes=4 * 6 * 8 - 1))


def test_top_level_api_consumes_prepared_and_rejects_other_models():
    rng = np.random.default_rng(30)
    u = rng.uniform(0.05, 0.95, size=(35, 5))
    model = EquicorrGaussianCopula(d=5)
    prepared = model.prepare_sufficient_statistics(u)
    result = api.fit(model, prepared, method="MLE")

    assert api.log_likelihood(model, prepared, result) == pytest.approx(
        model.log_likelihood(prepared, result.copula_param))
    np.testing.assert_array_equal(
        api.predictive_mean(model, prepared, result),
        np.full(len(prepared), result.copula_param),
    )
    assert api.sample(
        model, prepared, result, 3, rng=np.random.default_rng(1)
    ).shape == (3, 5)
    assert api.predict(
        model, prepared, result, 3, rng=np.random.default_rng(1)
    ).shape == (3, 5)

    other = StochasticStudentCopula(d=5)
    with pytest.raises(TypeError, match="EquicorrGaussianCopula"):
        api.fit(other, prepared, method="MLE")


def test_refit_clears_stale_dense_or_prepared_training_state():
    rng = np.random.default_rng(301)
    u = rng.uniform(0.05, 0.95, size=(24, 4))
    model = EquicorrGaussianCopula(d=4)
    prepared = model.prepare_sufficient_statistics(u)

    model.fit(u, method="MLE")
    assert model._last_u is u
    assert model._last_prepared is None
    model.fit(prepared, method="MLE")
    assert model._last_u is None
    assert model._last_prepared is prepared
    model.fit(u.copy(), method="MLE")
    assert model._last_u is not None
    assert model._last_prepared is None


@pytest.mark.parametrize(
    "scaling",
    [
        "unit",
        pytest.param(
            "fisher",
            marks=pytest.mark.sanitizer_numerical,
        ),
    ],
)
def test_gas_recursion_consumes_prepared_statistics_directly(scaling):
    rng = np.random.default_rng(31)
    u = rng.uniform(0.05, 0.95, size=(40, 6))
    model = EquicorrGaussianCopula(d=6)
    prepared = model.prepare_sufficient_statistics(
        u, batch_rows=9, dimension_tile=4, n_threads=4)
    params = (0.02, 0.01, 0.90)

    dense = _cpp_gas.filter_result(
        *params, u, model, scaling=scaling)
    compact = _cpp_gas.filter_result(
        *params, prepared, model, scaling=scaling)
    path_tolerance = 2e-11 if scaling == "unit" else 5e-4
    score_tolerance = 2e-10 if scaling == "unit" else 6e-3
    likelihood_rtol = 2e-11 if scaling == "unit" else 1e-4
    np.testing.assert_allclose(
        compact.g_path, dense.g_path,
        rtol=path_tolerance, atol=path_tolerance)
    np.testing.assert_allclose(
        compact.r_path, dense.r_path,
        rtol=path_tolerance, atol=path_tolerance)
    np.testing.assert_allclose(
        compact.score_path, dense.score_path,
        rtol=score_tolerance, atol=score_tolerance)
    np.testing.assert_allclose(
        compact.log_likelihood,
        dense.log_likelihood,
        rtol=likelihood_rtol,
        atol=2e-11,
    )
    assert _cpp_gas.log_likelihood(
        *params, prepared, model, scaling=scaling
    ) == pytest.approx(compact.log_likelihood, rel=2e-14, abs=2e-14)
    assert _cpp_gas.negative_log_likelihood(
        *params, prepared, model, scaling=scaling
    ) == pytest.approx(-compact.log_likelihood, rel=2e-14, abs=2e-14)
    assert np.isfinite(_cpp_gas.predict_parameter(
        *params, prepared, model, scaling=scaling))


def test_gas_fit_accepts_prepared_data():
    rng = np.random.default_rng(33)
    u = rng.uniform(0.05, 0.95, size=(50, 5))
    model = EquicorrGaussianCopula(d=5)
    prepared = model.prepare_sufficient_statistics(u)

    result = model.fit(
        prepared,
        method="GAS",
        gamma0=np.array([0.02, 0.03, 0.9]),
        maxiter=3,
    )

    assert result.method == "GAS"
    assert np.isfinite(result.log_likelihood)
    assert result.log_likelihood != -1e10
    assert model._last_prepared is prepared

    expected = model.sample(9, rng=np.random.default_rng(330))
    actual = np.concatenate(list(model.sample_batches(
        9, batch_rows=4, rng=np.random.default_rng(330))))
    np.testing.assert_array_equal(actual, expected)

    prediction = np.concatenate(list(model.predict_batches(
        9, batch_rows=4, rng=np.random.default_rng(331))))
    assert prediction.shape == (9, 5)
    assert np.all(np.isfinite(prediction))
    assert model.predict(
        9, rng=np.random.default_rng(331)).shape == (9, 5)


def test_scar_evaluator_consumes_prepared_statistics_directly():
    rng = np.random.default_rng(44)
    u = rng.uniform(0.05, 0.95, size=(20, 6))
    model = EquicorrGaussianCopula(d=6)
    prepared = model.prepare_sufficient_statistics(
        u, dimension_tile=4, n_threads=4)
    config = AutoTMConfig(
        transition_method="matrix",
        K=20,
        adaptive=False,
        max_K=20,
    )
    dense = _cpp_scar_ou.prepare_objective(u, model, config)
    compact = _cpp_scar_ou.prepare_objective(prepared, model, config)
    params = (1.2, 0.1, 0.7)

    dense_value, dense_gradient, _ = (
        dense.neg_loglik_with_grad_info(*params))
    compact_value, compact_gradient, _ = (
        compact.neg_loglik_with_grad_info(*params))
    np.testing.assert_allclose(
        compact_value, dense_value, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(
        compact_gradient, dense_gradient, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(
        compact.predictive_mean(*params),
        dense.predictive_mean(*params),
        rtol=2e-13,
        atol=2e-13,
    )
    for horizon in ("current", "next"):
        compact_grid, compact_probability = compact.state_distribution(
            *params, horizon=horizon)
        dense_grid, dense_probability = dense.state_distribution(
            *params, horizon=horizon)
        np.testing.assert_array_equal(compact_grid, dense_grid)
        np.testing.assert_allclose(
            compact_probability,
            dense_probability,
            rtol=2e-13,
            atol=2e-13,
        )


def test_scar_fit_accepts_prepared_data():
    rng = np.random.default_rng(45)
    u = rng.uniform(0.05, 0.95, size=(30, 5))
    model = EquicorrGaussianCopula(d=5)
    prepared = model.prepare_sufficient_statistics(u)

    result = model.fit(
        prepared,
        method="scar-tm-ou",
        alpha0=np.array([1.0, 0.1, 0.5]),
        K=20,
        grid_range=4.0,
        maxiter=2,
    )

    assert result.method == "SCAR-TM-OU"
    assert np.isfinite(result.log_likelihood)
    assert result.diagnostics["prepared_native_evaluator"] is True
    assert model._last_prepared is prepared

    first = np.concatenate(list(model.sample_batches(
        9, batch_rows=4, rng=np.random.default_rng(450))))
    second = np.concatenate(list(model.sample_batches(
        9, batch_rows=4, rng=np.random.default_rng(450))))
    np.testing.assert_array_equal(first, second)

    prediction = np.concatenate(list(model.predict_batches(
        9,
        batch_rows=4,
        given={1: 0.6},
        rng=np.random.default_rng(451),
    )))
    assert prediction.shape == (9, 5)
    np.testing.assert_array_equal(prediction[:, 1], np.full(9, 0.6))


def test_public_constructor_copies_caller_owned_arrays():
    source = np.array([1.0, 2.0])
    prepared = EquicorrPreparedData(
        sum_z=source,
        sum_z2=np.array([3.0, 4.0]),
        n_obs=2,
        dimension=3,
    )
    source[0] = 99.0
    assert prepared.sum_z[0] == 1.0


def test_npz_and_mmap_round_trips(tmp_path):
    prepared = EquicorrGaussianCopula(d=4).prepare_sufficient_statistics(
        np.array([[0.2, 0.4, 0.6, 0.8]]))

    npz_path = prepared.save_npz(tmp_path / "prepared.npz")
    portable = EquicorrPreparedData.load_npz(npz_path)
    mmap_dir = prepared.save_mmap(tmp_path / "prepared-mmap")
    mapped = EquicorrPreparedData.load_mmap(mmap_dir)

    for restored in (portable, mapped):
        np.testing.assert_array_equal(restored.sum_z, prepared.sum_z)
        np.testing.assert_array_equal(restored.sum_z2, prepared.sum_z2)
        assert restored.dimension == prepared.dimension
        assert dict(restored.diagnostics) == dict(prepared.diagnostics)
        assert restored.sum_z.flags.writeable is False
    assert isinstance(mapped.sum_z, np.memmap)
    assert isinstance(mapped.sum_z2, np.memmap)


@pytest.mark.parametrize(
    "bad",
    [
        np.empty((0, 3)),
        np.ones((2, 2)),
        np.array([[0.2, np.nan, 0.8]]),
    ],
)
def test_invalid_input_is_rejected(bad):
    model = EquicorrGaussianCopula(d=3)
    with pytest.raises(ValueError):
        model.prepare_sufficient_statistics(bad)


def test_default_preparation_never_initializes_parallel_runtime():
    code = (
        "import json, numpy as np\n"
        "from pyscarcopula import EquicorrGaussianCopula\n"
        "from pyscarcopula.numerical import _cpp_extension\n"
        "m = _cpp_extension.load()\n"
        "before = dict(m._parallel_runtime_info())\n"
        "EquicorrGaussianCopula(8192).prepare_sufficient_statistics("
        "np.full((2, 8192), 0.5), dimension_tile=256)\n"
        "after = dict(m._parallel_runtime_info())\n"
        "print(json.dumps({'before': before, 'after': after}))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    payload = json.loads(completed.stdout)
    assert payload["before"]["initialized"] is False
    assert payload["after"]["initialized"] is False


def test_large_dimension_preparation_has_t_bounded_output_and_tile_workspace():
    dimension = 100_000
    model = EquicorrGaussianCopula(d=dimension)
    u = np.full((2, dimension), 0.5, dtype=np.float64)

    prepared = model.prepare_sufficient_statistics(
        (u[:1], u[1:]),
        batch_rows=1,
        dimension_tile=4096,
        n_threads=1,
    )

    assert prepared.sum_z.nbytes + prepared.sum_z2.nbytes == 32
    assert prepared.diagnostics["peak_temporary_values"] <= (
        2 * int(np.ceil(dimension / 4096))
    )
    assert not hasattr(prepared, "u")


def test_negative_equicorrelation_sampling_is_structural_and_batched():
    model = EquicorrGaussianCopula(d=4)
    rho = -0.2
    samples = model.sample_at_parameter(
        80_000, rho, rng=np.random.default_rng(120))
    normal_scores = norm.ppf(samples)
    empirical = np.corrcoef(normal_scores, rowvar=False)
    np.testing.assert_allclose(
        empirical[np.triu_indices(4, k=1)],
        rho,
        atol=0.012,
    )

    blocks = list(model.sample_at_parameter_batches(
        11,
        np.linspace(-0.2, 0.7, 11),
        batch_rows=4,
        rng=np.random.default_rng(121),
    ))
    assert [block.shape for block in blocks] == [(4, 4), (4, 4), (3, 4)]
    assert np.all((np.vstack(blocks) > 0.0) & (np.vstack(blocks) < 1.0))

    with pytest.raises(ValueError, match="r must be finite"):
        next(model.sample_at_parameter_batches(
            1, -1.0 / 3.0, batch_rows=1))


