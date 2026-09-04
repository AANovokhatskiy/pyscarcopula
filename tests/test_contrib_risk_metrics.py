import multiprocessing as mp
import sys

import numpy as np


def test_predict_copula_handles_standalone_independent():
    from pyscarcopula import IndependentCopula
    from pyscarcopula.contrib.risk_metrics import _predict_copula

    uk = np.full((5, 2), 0.5)
    copula = IndependentCopula()
    copula.fit(uk)

    samples = _predict_copula(
        copula, uk, 7, rng=np.random.default_rng(123))

    assert samples.shape == (7, 2)
    assert np.all(np.isfinite(samples))
    assert np.all((0.0 <= samples) & (samples <= 1.0))


def test_worker_constructor_preserves_bivariate_transform_type():
    from pyscarcopula import BivariateGaussianCopula
    from pyscarcopula.contrib.risk_metrics import _get_copula_constructor

    cls, kwargs = _get_copula_constructor(
        BivariateGaussianCopula(transform_type="xtanh"))
    rebuilt = cls(**kwargs)

    assert kwargs == {"rotate": 0, "transform_type": "xtanh"}
    assert rebuilt._transform_type == "xtanh"


def test_worker_constructor_preserves_stochastic_student_corr_config():
    from pyscarcopula import StochasticStudentCopula
    from pyscarcopula.contrib.risk_metrics import _get_copula_constructor

    R = np.array([
        [1.0, 0.20, 0.10],
        [0.20, 1.0, 0.15],
        [0.10, 0.15, 1.0],
    ])
    corr_base = np.array([
        [1.0, 0.12, 0.08],
        [0.12, 1.0, 0.18],
        [0.08, 0.18, 1.0],
    ])
    source = StochasticStudentCopula(
        d=3,
        R=R,
        corr_mode="shrinkage",
        corr_base=corr_base,
        corr_shrinkage_init=0.65,
    )

    cls, kwargs = _get_copula_constructor(source)
    rebuilt = cls(**kwargs)

    assert rebuilt.d == 3
    assert rebuilt.corr_mode == "shrinkage"
    assert rebuilt._corr_shrinkage_init == 0.65
    np.testing.assert_allclose(rebuilt.R, R)
    np.testing.assert_allclose(rebuilt._corr_base, corr_base)


def test_worker_constructor_excludes_fitted_student_correlation_state():
    from pyscarcopula import StochasticStudentCopula
    from pyscarcopula.contrib.risk_metrics import _get_copula_constructor

    source = StochasticStudentCopula(d=3, corr_mode="shrinkage")
    fitted_correlation = np.array([
        [1.0, 0.20, 0.10],
        [0.20, 1.0, 0.15],
        [0.10, 0.15, 1.0],
    ])
    source._set_R(fitted_correlation, source="generated")
    source._corr_base = fitted_correlation.copy()
    source.fit_result = object()
    source._ppf_cache = object()

    cls, kwargs = _get_copula_constructor(source)
    rebuilt = cls(**kwargs)

    assert kwargs["R"] is None
    assert kwargs["corr_base"] is None
    assert rebuilt.R is None
    assert rebuilt._ppf_cache is None


def test_worker_constructor_preserves_rvine_options():
    from pyscarcopula import BivariateGaussianCopula, RVineCopula
    from pyscarcopula.contrib.risk_metrics import _get_copula_constructor

    source = RVineCopula(
        candidates=[BivariateGaussianCopula],
        allow_rotations=False,
        criterion="bic",
        truncation_level=1,
        truncation_fill="mle",
        threshold=0.05,
        min_edge_logL=-1.0,
        transform_type="xtanh",
    )

    cls, kwargs = _get_copula_constructor(source)
    rebuilt = cls(**kwargs)

    assert rebuilt.candidates == [BivariateGaussianCopula]
    assert rebuilt.allow_rotations is False
    assert rebuilt.criterion == "bic"
    assert rebuilt.truncation_level == 1
    assert rebuilt.truncation_fill == "mle"
    assert rebuilt.threshold == 0.05
    assert rebuilt.min_edge_logL == -1.0
    assert rebuilt.transform_type == "xtanh"


def test_risk_metrics_forwards_n_jobs_to_marginal_fit(monkeypatch):
    from pyscarcopula import IndependentCopula
    from pyscarcopula.contrib import risk_metrics as risk_module
    from pyscarcopula.contrib.marginal import MarginalModel

    calls = {}

    class RecordingMarginal:
        def fit_rolling(self, data, window_len, n_jobs=-1):
            calls["marginal_n_jobs"] = n_jobs
            return np.zeros((data.shape[0], data.shape[1], 1))

    data = np.arange(12, dtype=np.float64).reshape(6, 2)
    expected_weight = np.array([0.5, 0.5])

    monkeypatch.setattr(
        MarginalModel,
        "create",
        staticmethod(lambda name: RecordingMarginal()),
    )

    def fake_calculate_fixed(
            copula, data, method, marginal_model, marg_params, gamma,
            window_len, N_mc, portfolio_weight, n_jobs=1,
            window_seed_sequences=None, **kwargs):
        calls["cvar_n_jobs"] = n_jobs
        np.testing.assert_allclose(portfolio_weight, expected_weight)
        return (
            np.zeros(data.shape[0]),
            np.zeros(data.shape[0]),
            portfolio_weight,
        )

    monkeypatch.setattr(
        risk_module,
        "_calculate_cvar_fixed",
        fake_calculate_fixed,
    )

    risk_module.risk_metrics(
        IndependentCopula(),
        data,
        window_len=3,
        gamma=0.9,
        N_mc=10,
        marginals_method="johnsonsu",
        optimize_portfolio=False,
        n_jobs=4,
        rng=123,
    )

    assert calls == {"marginal_n_jobs": 4, "cvar_n_jobs": 4}


def test_rolling_multiprocessing_matches_sequential_with_n_threads_one():
    from pyscarcopula import EquicorrGaussianCopula, NumericalConfig
    from pyscarcopula.contrib.marginal import MarginalModel
    from pyscarcopula.contrib.risk_metrics import _calculate_cvar_fixed

    data = np.random.default_rng(701).normal(0.0, 0.01, (6, 2))
    window_len = 3
    n_windows = len(data) - window_len + 1
    marginal = MarginalModel.create("normal")
    marginal_params = marginal.fit_rolling(
        data,
        window_len,
        n_jobs=1,
    )
    weights = np.array([0.5, 0.5])
    seeds = np.random.SeedSequence(702).spawn(n_windows)
    kwargs = {
        "config": NumericalConfig(n_threads=1),
        "maxiter": 3,
        "maxfun": 10,
    }

    sequential = _calculate_cvar_fixed(
        EquicorrGaussianCopula(d=2),
        data,
        "mle",
        marginal,
        marginal_params,
        0.9,
        window_len,
        20,
        weights,
        n_jobs=1,
        window_seed_sequences=seeds,
        **kwargs,
    )
    parallel = _calculate_cvar_fixed(
        EquicorrGaussianCopula(d=2),
        data,
        "mle",
        marginal,
        marginal_params,
        0.9,
        window_len,
        20,
        weights,
        n_jobs=2,
        window_seed_sequences=seeds,
        mp_start_method=(
            "forkserver"
            if sys.platform.startswith("linux")
            and "forkserver" in mp.get_all_start_methods()
            else "spawn"
        ),
        **kwargs,
    )

    for actual, expected in zip(parallel, sequential):
        np.testing.assert_array_equal(actual, expected)
