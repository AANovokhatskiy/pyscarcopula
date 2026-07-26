"""Phase 9.7 contracts for factor correlation in GaussianCopula."""

from concurrent.futures import ThreadPoolExecutor
import inspect
import json
import pickle
import subprocess
import sys

import numpy as np
import pytest
from scipy.stats import norm

from pyscarcopula import (
    GaussianCopula,
    MultivariateMLEResult,
    NumericalConfig,
)
from pyscarcopula._utils import pobs
from pyscarcopula.api import fit, log_likelihood, predict
from pyscarcopula.contrib.risk_metrics import _get_copula_constructor
from pyscarcopula.numerical import _cpp_copula, _cpp_extension
from pyscarcopula.stattests import (
    factor_gaussian_rosenblatt_transform,
    gaussian_rosenblatt_transform,
    gof_test,
)


def _loadings(d=8, rank=3, seed=9701):
    return np.random.default_rng(seed).normal(
        scale=0.08, size=(d, rank))


def _factor_model(d=8, rank=3):
    return GaussianCopula(
        d,
        corr_mode="factor",
        factor_rank=rank,
        factor_loadings=_loadings(d, rank),
        factor_tile_size=4,
    )


def _dense_model(correlation):
    model = GaussianCopula(correlation.shape[0])
    model.corr = np.array(correlation, copy=True)
    return model


@pytest.mark.parametrize(
    "kwargs, error, message",
    [
        ({"d": 5, "corr_mode": "unknown"}, ValueError, "corr_mode"),
        (
            {"d": 5, "corr_mode": "factor", "factor_rank": None},
            TypeError,
            "factor_rank",
        ),
        (
            {"d": 5, "corr_mode": "factor", "factor_rank": 5},
            ValueError,
            "1 <= k < d",
        ),
        (
            {"d": 5, "factor_rank": 2},
            ValueError,
            "corr_mode='factor'",
        ),
    ],
)
def test_factor_constructor_validation(kwargs, error, message):
    with pytest.raises(error, match=message):
        GaussianCopula(**kwargs)


def test_factor_static_spec_and_likelihood_match_dense_reference():
    factor = _factor_model()
    dense = _dense_model(factor.to_correlation_matrix())
    observations = np.random.default_rng(9702).uniform(
        0.03, 0.97, size=(31, factor.dimension))

    module = _cpp_extension.load()
    spec = _cpp_copula.make_static_likelihood_spec(
        module, factor, observations)

    assert spec.correlation_kind == module.CorrelationKind.Factor
    assert len(spec.l_inv) == 0
    np.testing.assert_allclose(
        factor.log_pdf_rows(observations, n_threads=4),
        dense.log_pdf_rows(observations, n_threads=1),
        rtol=2e-12,
        atol=2e-12,
    )
    assert factor.log_likelihood(
        observations, n_threads=1) == pytest.approx(
            factor.log_likelihood(observations, n_threads=4),
            rel=0.0,
            abs=0.0,
        )


def test_two_stage_fit_is_compact_and_top_level_config_is_forwarded():
    observations = pobs(
        np.random.default_rng(9703).standard_normal((70, 12)))
    model = GaussianCopula(
        12,
        corr_mode="factor",
        factor_rank=3,
        factor_tile_size=5,
        factor_seed=19,
    )

    result = fit(
        model,
        observations,
        method="mle",
        config=NumericalConfig(n_threads=2),
    )

    assert isinstance(result, MultivariateMLEResult)
    assert result is model.fit_result
    assert result.correlation_matrix is None
    assert result.parameter_count == 12 * 3 - 3
    assert result.diagnostics["n_threads"] == 2
    assert result.diagnostics["corr_mode"] == "factor"
    assert result.model_parameters["factor_loadings"].shape == (12, 3)
    assert model.corr is None
    assert model._factor_operator is not None
    assert np.isfinite(log_likelihood(
        model,
        observations,
        result,
        config=NumericalConfig(n_threads=2),
    ))


def test_compact_pickle_round_trip_restores_native_operator():
    model = _factor_model(d=11, rank=2)
    restored = pickle.loads(pickle.dumps(model))

    assert restored.corr is None
    assert restored._factor_operator is not None
    np.testing.assert_array_equal(
        restored.factor_loadings_, model.factor_loadings_)
    expected = model.sample(
        9, rng=np.random.default_rng(9704), n_threads=1)
    actual = restored.sample(
        9, rng=np.random.default_rng(9704), n_threads=4)
    np.testing.assert_array_equal(actual, expected)


def test_factor_rosenblatt_and_gof_match_dense_path():
    factor = _factor_model(d=9, rank=3)
    dense_correlation = factor.to_correlation_matrix()
    observations = np.random.default_rng(9711).uniform(
        0.02, 0.98, size=(47, 9))

    actual = factor_gaussian_rosenblatt_transform(
        factor.correlation_operator_, observations)
    expected = gaussian_rosenblatt_transform(
        dense_correlation, observations)

    np.testing.assert_allclose(
        actual, expected, rtol=2e-13, atol=2e-13)
    result = gof_test(factor, observations, to_pobs=False)
    assert np.isfinite(result.statistic)


def test_factor_conditional_sampling_matches_dense_gaussian_moments():
    model = _factor_model(d=7, rank=2)
    correlation = model.to_correlation_matrix()
    given = {1: 0.31, 5: 0.77}
    sample = model.sample_conditional(
        30000,
        given,
        rng=np.random.default_rng(9705),
        n_threads=4,
    )

    fixed = np.array(sorted(given))
    free = np.array([idx for idx in range(7) if idx not in given])
    z_fixed = norm.ppf([given[idx] for idx in fixed])
    r_ff = correlation[np.ix_(free, free)]
    r_fg = correlation[np.ix_(free, fixed)]
    r_gg = correlation[np.ix_(fixed, fixed)]
    expected_mean = r_fg @ np.linalg.solve(r_gg, z_fixed)
    expected_cov = (
        r_ff - r_fg @ np.linalg.solve(r_gg, r_fg.T))
    latent = norm.ppf(sample[:, free])

    np.testing.assert_array_equal(
        sample[:, fixed],
        np.broadcast_to(
            np.array([given[idx] for idx in fixed]),
            (len(sample), len(fixed)),
        ),
    )
    np.testing.assert_allclose(
        latent.mean(axis=0), expected_mean, atol=0.025)
    np.testing.assert_allclose(
        np.cov(latent, rowvar=False), expected_cov, atol=0.035)


def test_seeded_sampling_is_exact_across_thread_counts_and_reentrant():
    model = _factor_model(d=128, rank=4)

    for given in (None, {0: 0.42, 91: 0.63}):
        draw = model.sample if given is None else model.sample_conditional
        args = (64,) if given is None else (64, given)
        one = draw(
            *args, rng=np.random.default_rng(9706), n_threads=1)
        four = draw(
            *args, rng=np.random.default_rng(9706), n_threads=4)
        np.testing.assert_array_equal(four, one)

    def concurrent(seed):
        return model.sample(
            32, rng=np.random.default_rng(seed), n_threads=1)

    expected = [concurrent(seed) for seed in range(4)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(concurrent, range(4)))
    for sequential, parallel in zip(expected, actual):
        np.testing.assert_array_equal(parallel, sequential)
    assert model.fit_result is None
    assert model.corr is None


def test_batches_budget_predict_and_rolling_constructor_contracts():
    loadings = _loadings(d=9, rank=2)
    model = GaussianCopula(
        9,
        corr_mode="factor",
        factor_rank=2,
        factor_loadings=loadings,
        factor_tile_size=7,
        factor_seed=23,
        factor_oversampling=5,
    )
    batches = list(model.sample_batches(
        11,
        batch_rows=4,
        given={2: 0.4},
        rng=np.random.default_rng(9707),
    ))
    assert [len(block) for block in batches] == [4, 4, 3]
    assert all(np.all(block[:, 2] == 0.4) for block in batches)
    with pytest.raises(MemoryError):
        model.sample(10, memory_budget_bytes=1)

    history = np.random.default_rng(9710).uniform(
        0.1, 0.9, size=(12, 9))
    result = model.fit(history)
    predicted = predict(
        model,
        history,
        result,
        6,
        rng=np.random.default_rng(9708),
    )
    assert predicted.shape == (6, 9)

    model_type, kwargs = _get_copula_constructor(model)
    rebuilt = model_type(**kwargs)
    assert rebuilt.corr_mode == "factor"
    assert rebuilt.factor_tile_size == 7
    np.testing.assert_array_equal(rebuilt.factor_loadings_, loadings)


@pytest.mark.parametrize(
    "method",
    [
        "log_likelihood",
        "log_pdf_rows",
        "sample",
        "sample_batches",
        "sample_conditional",
        "predict",
        "predict_batches",
    ],
)
def test_factor_gaussian_public_methods_default_to_one_thread(method):
    parameter = inspect.signature(
        getattr(GaussianCopula, method)).parameters["n_threads"]
    assert parameter.default == 1


def test_default_factor_gaussian_calls_do_not_initialize_native_pool():
    code = (
        "import json, numpy as np\n"
        "from pyscarcopula import GaussianCopula\n"
        "from pyscarcopula.numerical import _cpp_extension\n"
        "m = _cpp_extension.load()\n"
        "before = dict(m._parallel_runtime_info())\n"
        "cop = GaussianCopula("
        "1024, corr_mode='factor', factor_rank=4, "
        "factor_loadings=np.full((1024, 4), 0.01))\n"
        "cop.log_pdf_rows(np.full((4, 1024), 0.5))\n"
        "cop.sample(4, rng=np.random.default_rng(1))\n"
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


def test_large_dimension_path_never_materializes_dense_correlation():
    d = 10000
    rank = 4
    model = GaussianCopula(
        d,
        corr_mode="factor",
        factor_rank=rank,
        factor_loadings=np.full((d, rank), 0.005),
        factor_tile_size=512,
    )
    observations = np.full((2, d), 0.5)

    rows = model.log_pdf_rows(observations, n_threads=2)
    sample = model.sample(
        2, rng=np.random.default_rng(9709), n_threads=2)

    assert rows.shape == (2,)
    assert sample.shape == (2, d)
    assert np.all(np.isfinite(rows))
    assert np.all(np.isfinite(sample))
    assert model.corr is None
    assert model.fit_result is None
