"""Factor Student batch and conditional sampling contracts."""

from concurrent.futures import ThreadPoolExecutor
import inspect
import json
import subprocess
import sys

import numpy as np
import pytest
from scipy.stats import t as t_dist

from pyscarcopula import NumericalConfig, StochasticStudentCopula
from pyscarcopula.copula.multivariate.factor_correlation import (
    PreparedFactorCorrelation,
)


def _model(d=7, k=3, seed=9601):
    rng = np.random.default_rng(seed)
    loadings = rng.normal(scale=0.11, size=(d, k))
    return StochasticStudentCopula(
        d,
        corr_mode="factor",
        factor_rank=k,
        factor_loadings=loadings,
        factor_tile_size=max(2, d // 2),
    )


def test_fixed_factor_normal_draws_are_exact_across_threads():
    operator = _model().correlation_operator_
    rng = np.random.default_rng(9602)
    factors = rng.standard_normal((128, operator.rank))
    residuals = rng.standard_normal((128, operator.dimension))

    sequential = operator.transform_normal_draws(
        factors, residuals, n_threads=1)
    parallel = operator.transform_normal_draws(
        factors, residuals, n_threads=4)

    np.testing.assert_array_equal(parallel, sequential)
    assert not np.shares_memory(sequential, residuals)


def test_unconditional_factor_student_sampling_matches_correlation():
    model = _model()
    df = 5.5
    samples = model.sample_at_parameter(
        30_000,
        df,
        rng=np.random.default_rng(9603),
        n_threads=2,
    )
    latent = t_dist.ppf(samples, df=df)
    empirical = np.corrcoef(latent, rowvar=False)

    np.testing.assert_allclose(
        empirical,
        model.to_correlation_matrix(),
        rtol=0.0,
        atol=0.025,
    )
    assert model._R is None


def test_variable_df_and_seed_are_exact_across_threads():
    model = _model(d=9, k=2)
    df = np.resize(np.array([3.5, 6.0, 12.0]), 257)

    sequential = model.sample_at_parameter(
        len(df), df, rng=np.random.default_rng(9604), n_threads=1)
    parallel = model.sample_at_parameter(
        len(df), df, rng=np.random.default_rng(9604), n_threads=4)

    np.testing.assert_array_equal(parallel, sequential)
    assert np.all(np.isfinite(parallel))
    assert np.all((parallel > 0.0) & (parallel < 1.0))


def test_conditional_factor_student_matches_analytical_moments():
    model = _model(d=7, k=3)
    df = 6.0
    given = {0: 0.72, 4: 0.31}
    count = 40_000
    samples = model.sample_conditional(
        count,
        r=df,
        given=given,
        rng=np.random.default_rng(9605),
        n_threads=2,
    )

    given_idx = np.array(sorted(given))
    free_idx = np.array(
        [index for index in range(model.d) if index not in given])
    dense = model.to_correlation_matrix()
    x_given = t_dist.ppf(
        np.array([given[index] for index in given_idx]), df=df)
    r_gg = dense[np.ix_(given_idx, given_idx)]
    r_fg = dense[np.ix_(free_idx, given_idx)]
    r_ff = dense[np.ix_(free_idx, free_idx)]
    solved = np.linalg.solve(r_gg, x_given)
    expected_mean = r_fg @ solved
    schur = r_ff - r_fg @ np.linalg.solve(r_gg, r_fg.T)
    delta = float(x_given @ solved)
    expected_covariance = (
        (df + delta) / (df + len(given_idx) - 2.0) * schur)

    latent = t_dist.ppf(samples[:, free_idx], df=df)
    np.testing.assert_allclose(
        latent.mean(axis=0), expected_mean, rtol=0.0, atol=0.025)
    np.testing.assert_allclose(
        np.cov(latent, rowvar=False),
        expected_covariance,
        rtol=0.04,
        atol=0.025,
    )
    for index, value in given.items():
        np.testing.assert_array_equal(
            samples[:, index], np.full(count, value))
    assert model._R is None


def test_conditional_variable_df_is_exact_across_threads():
    model = _model(d=10, k=3)
    df = np.resize(np.array([4.0, 7.5]), 300)
    given = {1: 0.2, 7: 0.8}

    sequential = model.sample_conditional(
        len(df),
        r=df,
        given=given,
        rng=np.random.default_rng(9606),
        n_threads=1,
    )
    parallel = model.sample_conditional(
        len(df),
        r=df,
        given=given,
        rng=np.random.default_rng(9606),
        n_threads=4,
    )

    np.testing.assert_array_equal(parallel, sequential)
    np.testing.assert_array_equal(
        parallel[:, 1], np.full(len(df), 0.2))
    np.testing.assert_array_equal(
        parallel[:, 7], np.full(len(df), 0.8))


def test_all_coordinates_given_needs_no_dense_materialization():
    model = _model(d=5, k=2)
    given = {index: (index + 1.0) / 7.0 for index in range(5)}

    samples = model.sample_conditional(
        4, r=5.0, given=given, rng=np.random.default_rng(9607))

    np.testing.assert_array_equal(
        samples,
        np.tile(np.array(list(given.values())), (4, 1)),
    )
    assert model._R is None


def test_factor_sampling_rejects_invalid_sizes_df_threads_and_budget():
    model = _model(d=6, k=2)

    with pytest.raises(TypeError, match="n must be an integer"):
        model.sample_at_parameter(True, 5.0)
    with pytest.raises(ValueError, match="greater than 2"):
        model.sample_at_parameter(2, 2.0)
    with pytest.raises(ValueError, match="length 2"):
        model.sample_at_parameter(2, np.array([4.0, 5.0, 6.0]))
    with pytest.raises(ValueError, match="n_threads"):
        model.sample_at_parameter(2, 5.0, n_threads=0)
    with pytest.raises(MemoryError, match="sampling requires"):
        model.sample_at_parameter(
            2,
            5.0,
            memory_budget_bytes=model._factor_sampling_peak_bytes(2) - 1,
        )


def test_large_dimension_batches_are_bounded_and_compact():
    dimension = 10_000
    rank = 4
    model = StochasticStudentCopula(
        dimension,
        corr_mode="factor",
        factor_rank=rank,
        factor_loadings=np.full((dimension, rank), 0.002),
        factor_tile_size=512,
    )
    required = model._factor_sampling_peak_bytes(2)

    blocks = list(model.sample_at_parameter_batches(
        5,
        5.0,
        batch_rows=2,
        rng=np.random.default_rng(9608),
        n_threads=2,
        memory_budget_bytes=required,
    ))

    assert [block.shape for block in blocks] == [
        (2, dimension), (2, dimension), (1, dimension)]
    assert model._R is None

    conditional_required = model._factor_sampling_peak_bytes(
        2, conditional=True)
    conditional = model.sample_conditional(
        2,
        r=np.array([4.5, 7.0]),
        given={0: 0.25, dimension - 1: 0.75},
        rng=np.random.default_rng(9615),
        n_threads=2,
        memory_budget_bytes=conditional_required,
    )
    assert conditional.shape == (2, dimension)
    np.testing.assert_array_equal(
        conditional[:, 0], np.full(2, 0.25))
    np.testing.assert_array_equal(
        conditional[:, -1], np.full(2, 0.75))
    assert model._R is None

    with pytest.raises(MemoryError, match="reduce batch_rows"):
        list(model.sample_at_parameter_batches(
            2,
            5.0,
            batch_rows=2,
            memory_budget_bytes=required - 1,
        ))


def test_fitted_mle_batch_sampling_and_prediction_remain_compact():
    model = _model(d=6, k=2)
    observations = np.random.default_rng(9609).uniform(
        0.08, 0.92, size=(32, model.d))
    model.fit(
        observations,
        method="mle",
        maxiter=20,
        config=NumericalConfig(n_threads=1),
    )

    sample_blocks = list(model.sample_batches(
        9,
        batch_rows=4,
        rng=np.random.default_rng(9610),
        n_threads=2,
    ))
    predict_blocks = list(model.predict_batches(
        9,
        batch_rows=4,
        given={2: 0.45},
        rng=np.random.default_rng(9611),
        n_threads=2,
    ))

    assert [block.shape for block in sample_blocks] == [
        (4, 6), (4, 6), (1, 6)]
    assert [block.shape for block in predict_blocks] == [
        (4, 6), (4, 6), (1, 6)]
    np.testing.assert_array_equal(
        np.concatenate(predict_blocks)[:, 2],
        np.full(9, 0.45),
    )
    assert model._R is None


@pytest.mark.parametrize(
    ("method", "fit_kwargs"),
    [
        (
            "gas",
            {
                "gamma0": np.array([0.08, 0.04, 0.82]),
                "maxiter": 1,
                "maxfun": 15,
            },
        ),
        (
            "scar-tm-ou",
            {
                "alpha0": np.array([1.0, 0.2, 0.7]),
                "K": 8,
                "max_K": 8,
                "adaptive": False,
                "transition_method": "matrix",
                "analytical_grad": True,
                "smart_init": False,
                "maxiter": 1,
                "maxfun": 15,
            },
        ),
    ],
)
def test_dynamic_factor_sampling_and_prediction_batches(method, fit_kwargs):
    model = _model(d=5, k=2)
    observations = np.random.default_rng(9612).uniform(
        0.08, 0.92, size=(20, model.d))
    model.fit(
        observations,
        method=method,
        config=NumericalConfig(n_threads=1),
        **fit_kwargs,
    )

    sampled = np.concatenate(list(model.sample_batches(
        7,
        batch_rows=3,
        rng=np.random.default_rng(9613),
        n_threads=2,
    )))
    predicted = np.concatenate(list(model.predict_batches(
        7,
        batch_rows=3,
        given={3: 0.55},
        rng=np.random.default_rng(9614),
        n_threads=2,
    )))

    assert sampled.shape == (7, model.d)
    assert predicted.shape == (7, model.d)
    assert np.all(np.isfinite(sampled))
    assert np.all(np.isfinite(predicted))
    np.testing.assert_array_equal(
        predicted[:, 3], np.full(7, 0.55))
    assert model._R is None


def test_concurrent_seeded_sampling_is_model_state_safe():
    model = _model(d=128, k=4)

    def draw(seed):
        return model.sample_at_parameter(
            64,
            5.0,
            rng=np.random.default_rng(seed),
            n_threads=1,
        )

    expected = [draw(seed) for seed in range(4)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(draw, range(4)))

    for sequential, concurrent in zip(expected, actual):
        np.testing.assert_array_equal(concurrent, sequential)
    assert model.fit_result is None
    assert model._R is None


@pytest.mark.parametrize(
    ("owner", "method"),
    [
        (PreparedFactorCorrelation, "transform_normal_draws"),
        (StochasticStudentCopula, "sample_at_parameter"),
        (StochasticStudentCopula, "sample_at_parameter_batches"),
        (StochasticStudentCopula, "sample"),
        (StochasticStudentCopula, "sample_batches"),
        (StochasticStudentCopula, "sample_conditional"),
        (StochasticStudentCopula, "predict"),
        (StochasticStudentCopula, "predict_batches"),
    ],
)
def test_factor_sampling_methods_default_to_one_thread(owner, method):
    parameter = inspect.signature(
        getattr(owner, method)).parameters["n_threads"]
    assert parameter.default == 1


def test_default_factor_sampling_does_not_initialize_native_pool():
    code = (
        "import json, numpy as np\n"
        "from pyscarcopula import StochasticStudentCopula\n"
        "from pyscarcopula._native import _extension as _cpp_extension\n"
        "m = _cpp_extension.load()\n"
        "before = dict(m._parallel_runtime_info())\n"
        "cop = StochasticStudentCopula("
        "1024, corr_mode='factor', factor_rank=4, "
        "factor_loadings=np.full((1024, 4), 0.01))\n"
        "cop.sample_at_parameter(16, 5.0, rng=np.random.default_rng(1))\n"
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
