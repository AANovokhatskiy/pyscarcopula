"""Parallelization correctness and resource regressions."""

import numpy as np
import pytest

from pyscarcopula import EquicorrGaussianCopula, StochasticStudentCopula
from pyscarcopula.copula.multivariate import conditional
from pyscarcopula.copula.multivariate.equicorr_prepared import (
    EquicorrPreparedData,
)


def _student_models():
    return (
        StochasticStudentCopula(3, R=np.eye(3)),
        StochasticStudentCopula(
            3,
            corr_mode="factor",
            factor_rank=1,
            factor_loadings=np.array([[0.2], [0.3], [-0.1]]),
        ),
    )


def test_equicorr_conditional_never_requests_a_dense_matrix():
    assert not hasattr(conditional, "equicorr_matrix")
    model = EquicorrGaussianCopula(10_000)
    samples = model.sample_conditional(
        2,
        r=np.array([-5e-5, 0.2]),
        given={0: 0.5},
        rng=np.random.default_rng(8101),
        memory_budget_bytes=2 * 10_000 * 8,
        n_threads=4,
    )
    assert samples.shape == (2, 10_000)
    assert np.all(np.isfinite(samples))
    np.testing.assert_array_equal(samples[:, 0], np.full(2, 0.5))


def test_equicorr_dense_mle_result_stays_scalar_and_compact():
    u = np.random.default_rng(8102).uniform(0.05, 0.95, size=(30, 7))
    result = EquicorrGaussianCopula(7).fit(
        u, method="mle", maxiter=3)
    assert result.correlation_matrix is None
    assert result.diagnostics["corr_matrix"] is None
    assert (
        result.diagnostics["correlation_representation"]
        == "equicorrelation_scalar"
    )


def test_equicorr_failed_refit_preserves_previous_training_state():
    model = EquicorrGaussianCopula(3)
    u = np.random.default_rng(8103).uniform(0.05, 0.95, size=(30, 3))
    result = model.fit(u, method="mle", maxiter=3)
    previous_u = model._last_u

    with pytest.raises(ValueError, match="finite"):
        model.fit(
            np.array([[0.2, np.nan, 0.4]]),
            method="mle",
        )

    assert model.fit_result is result
    assert model._last_u is previous_u
    assert model._last_prepared is None


@pytest.mark.parametrize("df", [1.0, 2.0, np.nan, np.inf, -np.inf])
def test_all_student_sampling_modes_reject_invalid_df(df):
    for model in _student_models():
        with pytest.raises(ValueError, match="finite and greater than 2"):
            model.sample_at_parameter(
                3, df, rng=np.random.default_rng(8104))
        with pytest.raises(ValueError, match="finite and greater than 2"):
            model.sample_conditional(
                3,
                r=df,
                given={0: 0.4},
                rng=np.random.default_rng(8105),
            )


def test_factor_sampling_budget_covers_all_live_dimension_buffers():
    model = _student_models()[1]
    rows = 5
    unconditional = model._factor_sampling_peak_bytes(rows)
    conditional = model._factor_sampling_peak_bytes(
        rows, conditional=True)
    assert unconditional >= rows * 6 * model.d * 8
    assert conditional >= rows * 8 * model.d * 8


def test_prepared_equicorr_rejects_impossible_sufficient_statistics():
    with pytest.raises(ValueError, match=r"sum_z\*\*2"):
        EquicorrPreparedData(
            sum_z=np.array([10.0]),
            sum_z2=np.array([1.0]),
            n_obs=1,
            dimension=3,
        )


@pytest.mark.parametrize(
    "model",
    [
        EquicorrGaussianCopula(3),
        StochasticStudentCopula(3),
    ],
)
def test_mle_rejects_unknown_options_before_state_mutation(model):
    u = np.random.default_rng(8106).uniform(0.05, 0.95, size=(20, 3))
    with pytest.raises(TypeError, match="typo_option"):
        model.fit(u, method="mle", typo_option=123)
    assert model.fit_result is None
    assert getattr(model, "_last_u", None) is None
    if isinstance(model, StochasticStudentCopula):
        assert model.R is None


@pytest.mark.parametrize("key", [True, 1.5, "1"])
def test_conditional_given_requires_actual_integer_keys(key):
    with pytest.raises(TypeError, match="keys must be integers"):
        conditional.validate_multivariate_given({key: 0.4}, 3)
