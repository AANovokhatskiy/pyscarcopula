import numpy as np
import pytest


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("bootstrap", "false"),
        ("bootstrap", 0),
        ("bootstrap", None),
        ("bootstrap_refit", "false"),
        ("bootstrap_refit", 1),
        ("bootstrap_refit", None),
    ],
)
def test_gof_rejects_non_boolean_bootstrap_flags(argument, value):
    from pyscarcopula import BivariateGaussianCopula
    from pyscarcopula.stattests import gof_test

    kwargs = {argument: value}
    with pytest.raises(TypeError, match=rf"{argument} must be a boolean"):
        gof_test(
            BivariateGaussianCopula(),
            np.full((4, 2), 0.5),
            to_pobs=False,
            **kwargs,
        )


def test_bootstrap_strategy_preserves_fitted_scar_grid_settings():
    from pyscarcopula._types import LatentResult, ou_params
    from pyscarcopula.stattests import _bootstrap_strategy

    result = LatentResult(
        log_likelihood=0.0,
        method="SCAR-TM-OU",
        copula_name="Gaussian",
        success=True,
        params=ou_params(0.8, 0.0, 0.5),
        K=17,
        grid_range=2.75,
        grid_method="sparse",
        adaptive=False,
        transition_method="matrix",
        max_K=None,
    )

    strategy = _bootstrap_strategy(result, config=None)

    assert strategy.K == 17
    assert strategy.grid_range == 2.75
    assert strategy.grid_method == "sparse"
    assert strategy.adaptive is False


def _sample_bivariate_gaussian(n=32, seed=910):
    from pyscarcopula import BivariateGaussianCopula

    source = BivariateGaussianCopula()
    return source.sample_at_parameter(
        n,
        np.full(n, 0.4),
        rng=np.random.default_rng(seed),
    )


def _fit_bivariate(method, fit_kwargs=None):
    from pyscarcopula import BivariateGaussianCopula
    from pyscarcopula.api import fit

    u = _sample_bivariate_gaussian()
    model = BivariateGaussianCopula()
    result = fit(model, u, method=method, **(fit_kwargs or {}))
    return model, u, result


def _assert_static_bootstrap_parallel_parity(
        model, u, fit_result, *, bootstrap_refit):
    from pyscarcopula.stattests import gof_test

    kwargs = dict(
        fit_result=fit_result,
        to_pobs=False,
        bootstrap=True,
        n_bootstrap=2,
        bootstrap_refit=bootstrap_refit,
        rng=918,
    )
    sequential = gof_test(model, u, n_jobs=1, **kwargs)
    parallel = gof_test(model, u, n_jobs=2, **kwargs)

    np.testing.assert_array_equal(
        parallel.bootstrap_statistics,
        sequential.bootstrap_statistics,
    )
    assert parallel.pvalue == sequential.pvalue
    assert sequential.backend == "sequential"
    assert parallel.backend == "loky"
    assert parallel.n_jobs == 2
    assert all(
        row["bootstrap_refit"] is bootstrap_refit
        for row in parallel.bootstrap_diagnostics
    )


def _dynamic_fit_kwargs(method):
    if method == "mle":
        return {"gtol": 1e-2, "maxiter": 100, "maxfun": 500}
    if method == "gas":
        return {"gtol": 1e-2, "maxiter": 150, "maxfun": 2000}
    return {
        "K": 12,
        "grid_range": 3.0,
        "gtol": 1e-2,
        "maxiter": 150,
        "maxfun": 2000,
        "transition_method": "matrix",
        "adaptive": False,
    }


def _assert_dynamic_bootstrap_parallel_parity(
        model, u, fit_result, *, bootstrap_refit):
    from pyscarcopula.stattests import gof_test

    kwargs = dict(
        fit_result=fit_result,
        to_pobs=False,
        K=12,
        grid_range=3.0,
        bootstrap=True,
        n_bootstrap=2,
        bootstrap_refit=bootstrap_refit,
        bootstrap_fit_kwargs={
            "gtol": 1e-2, "maxiter": 150, "maxfun": 2000},
        rng=930,
    )
    sequential = gof_test(model, u, n_jobs=1, **kwargs)
    parallel = gof_test(model, u, n_jobs=2, **kwargs)

    np.testing.assert_array_equal(
        parallel.bootstrap_statistics,
        sequential.bootstrap_statistics,
    )
    assert parallel.pvalue == sequential.pvalue
    assert sequential.backend == "sequential"
    assert parallel.backend == "loky"
    assert parallel.n_jobs == 2


def test_mle_bootstrap_refit_is_partition_invariant():
    from pyscarcopula.stattests import gof_test

    model, u, fit_result = _fit_bivariate("mle")
    kwargs = dict(
        fit_result=fit_result,
        to_pobs=False,
        bootstrap=True,
        n_bootstrap=4,
        bootstrap_refit=True,
        bootstrap_fit_kwargs={"maxiter": 10},
        rng=911,
    )

    sequential = gof_test(model, u, n_jobs=1, **kwargs)
    parallel = gof_test(model, u, n_jobs=2, **kwargs)

    np.testing.assert_array_equal(
        parallel.bootstrap_statistics,
        sequential.bootstrap_statistics,
    )
    assert parallel.pvalue == sequential.pvalue
    assert sequential.backend == "sequential"
    assert sequential.n_jobs_requested == 1
    assert sequential.n_jobs == 1
    assert parallel.backend == "loky"
    assert parallel.n_jobs_requested == 2
    assert parallel.n_jobs == 2
    assert parallel.n_threads == 1
    assert parallel.rng_policy == "seed_sequence_per_replication"
    assert parallel.worker_model_ownership == "per_task"
    assert [
        row["bootstrap_iteration"]
        for row in parallel.bootstrap_diagnostics
    ] == [1, 2, 3, 4]
    np.testing.assert_array_equal(
        parallel.bootstrap_statistics,
        [
            row["bootstrap_statistic"]
            for row in parallel.bootstrap_diagnostics
        ],
    )
    assert all(
        row["bootstrap_refit"]
        for row in parallel.bootstrap_diagnostics
    )


@pytest.mark.parametrize(
    ("method", "fit_kwargs"),
    [
        ("gas", {"maxiter": 2, "maxfun": 10}),
        (
            "scar-tm-ou",
            {
                "K": 20,
                "grid_range": 3.0,
                "maxiter": 1,
                "maxfun": 8,
                "transition_method": "matrix",
                "adaptive": False,
            },
        ),
    ],
)
def test_dynamic_bootstrap_no_refit_is_partition_invariant(
        method, fit_kwargs):
    from pyscarcopula.stattests import gof_test

    model, u, fit_result = _fit_bivariate(method, fit_kwargs)
    kwargs = dict(
        fit_result=fit_result,
        to_pobs=False,
        K=20,
        grid_range=3.0,
        bootstrap=True,
        n_bootstrap=3,
        bootstrap_refit=False,
        rng=912,
    )

    sequential = gof_test(model, u, n_jobs=1, **kwargs)
    parallel = gof_test(model, u, n_jobs=2, **kwargs)

    np.testing.assert_array_equal(
        parallel.bootstrap_statistics,
        sequential.bootstrap_statistics,
    )
    assert parallel.pvalue == sequential.pvalue
    assert all(
        not row["bootstrap_refit"]
        for row in parallel.bootstrap_diagnostics
    )


def test_bootstrap_caps_workers_and_does_not_mutate_source_model():
    from pyscarcopula.stattests import gof_test

    model, u, fit_result = _fit_bivariate("mle")
    sentinel = object()
    history = np.full((2, 2), 0.25)
    model.fit_result = sentinel
    model._last_u = history

    result = gof_test(
        model,
        u,
        fit_result=fit_result,
        to_pobs=False,
        bootstrap=True,
        n_bootstrap=2,
        bootstrap_refit=True,
        rng=913,
        n_jobs=8,
    )

    assert result.n_jobs_requested == 8
    assert result.n_jobs == 2
    assert model.fit_result is sentinel
    assert model._last_u is history


def test_independence_bootstrap_refits_owned_worker_models():
    from pyscarcopula import IndependentCopula
    from pyscarcopula.stattests import gof_test

    u = np.random.default_rng(916).uniform(0.01, 0.99, (24, 2))
    model = IndependentCopula()
    fit_result = model.fit(u)

    result = gof_test(
        model,
        u,
        fit_result=fit_result,
        to_pobs=False,
        bootstrap=True,
        n_bootstrap=2,
        bootstrap_refit=True,
        rng=917,
        n_jobs=2,
    )

    assert result.backend == "loky"
    assert np.all(np.isfinite(result.bootstrap_statistics))
    assert all(
        row["bootstrap_fit_method"] == "MLE"
        for row in result.bootstrap_diagnostics
    )


@pytest.mark.parametrize("bootstrap_refit", [False, True])
def test_dense_gaussian_bootstrap_is_partition_invariant(bootstrap_refit):
    from scipy.stats import norm

    from pyscarcopula import GaussianCopula

    correlation = np.array([
        [1.0, 0.30, 0.10],
        [0.30, 1.0, 0.20],
        [0.10, 0.20, 1.0],
    ])
    latent = np.random.default_rng(919).multivariate_normal(
        np.zeros(3), correlation, size=36)
    u = norm.cdf(latent)
    fitted = GaussianCopula()
    fit_result = fitted.fit(u)
    original_correlation = fitted.corr.copy()

    # Exercise the stateless fit_result API with an unfitted prototype.
    prototype = GaussianCopula()
    _assert_static_bootstrap_parallel_parity(
        prototype,
        u,
        fit_result,
        bootstrap_refit=bootstrap_refit,
    )

    np.testing.assert_array_equal(fitted.corr, original_correlation)
    assert prototype.fit_result is None
    assert prototype.corr is None


@pytest.mark.parametrize("bootstrap_refit", [False, True])
def test_factor_gaussian_bootstrap_is_partition_invariant(bootstrap_refit):
    from pyscarcopula import GaussianCopula

    loadings = np.array([[0.50], [0.30], [-0.20], [0.40]])
    source = GaussianCopula(
        4,
        corr_mode="factor",
        factor_rank=1,
        factor_loadings=loadings,
    )
    u = source.sample(36, rng=np.random.default_rng(920))
    fitted = GaussianCopula(
        4,
        corr_mode="factor",
        factor_rank=1,
        factor_seed=12,
    )
    fit_result = fitted.fit(u)
    original_loadings = fitted.factor_loadings_

    prototype = GaussianCopula(
        4,
        corr_mode="factor",
        factor_rank=1,
        factor_seed=12,
    )
    _assert_static_bootstrap_parallel_parity(
        prototype,
        u,
        fit_result,
        bootstrap_refit=bootstrap_refit,
    )

    np.testing.assert_array_equal(
        fitted.factor_loadings_, original_loadings)
    assert prototype.fit_result is None
    assert prototype.factor_loadings_ is None


@pytest.mark.parametrize("bootstrap_refit", [False, True])
def test_student_bootstrap_is_partition_invariant(bootstrap_refit):
    from pyscarcopula import StudentCopula

    correlation = np.array([
        [1.0, 0.25, 0.10],
        [0.25, 1.0, 0.15],
        [0.10, 0.15, 1.0],
    ])
    source = StudentCopula()
    source.shape = correlation
    source.df = 6.0
    u = source.sample(36, rng=np.random.default_rng(921))
    fitted = StudentCopula()
    fit_result = fitted.fit(u)
    original_shape = fitted.shape.copy()
    original_df = fitted.df

    prototype = StudentCopula()
    _assert_static_bootstrap_parallel_parity(
        prototype,
        u,
        fit_result,
        bootstrap_refit=bootstrap_refit,
    )

    np.testing.assert_array_equal(fitted.shape, original_shape)
    assert fitted.df == original_df
    assert prototype.fit_result is None
    assert prototype.shape is None
    assert prototype.df is None


@pytest.mark.parametrize("bootstrap_refit", [False, True])
def test_factor_student_bootstrap_is_partition_invariant(bootstrap_refit):
    from pyscarcopula import StudentCopula

    loadings = np.array([[0.48], [0.30], [-0.22], [0.37]])
    source = StudentCopula(
        4, corr_mode="factor", factor_rank=1,
        factor_loadings=loadings)
    source.df = 6.0
    u = source.sample(36, rng=np.random.default_rng(923))
    fitted = StudentCopula(
        4, corr_mode="factor", factor_rank=1, factor_seed=14)
    fit_result = fitted.fit(u, maxiter=250)
    original_loadings = fitted.factor_loadings_

    prototype = StudentCopula(
        4, corr_mode="factor", factor_rank=1, factor_seed=14)
    _assert_static_bootstrap_parallel_parity(
        prototype,
        u,
        fit_result,
        bootstrap_refit=bootstrap_refit,
    )

    np.testing.assert_array_equal(
        fitted.factor_loadings_, original_loadings)
    assert prototype.fit_result is None
    assert prototype.factor_loadings_ is None


@pytest.mark.parametrize("method", ["mle", "gas", "scar-tm-ou"])
@pytest.mark.parametrize("bootstrap_refit", [False, True])
def test_equicorr_bootstrap_is_partition_invariant(
        method, bootstrap_refit):
    from pyscarcopula import EquicorrGaussianCopula

    source = EquicorrGaussianCopula(3)
    u = source.sample_at_parameter(
        24,
        np.full(24, 0.25),
        rng=np.random.default_rng(924),
    )
    fitted = EquicorrGaussianCopula(3)
    fit_result = fitted.fit(
        u, method=method, **_dynamic_fit_kwargs(method))
    original_fit_result = fitted.fit_result
    original_history = fitted._last_u

    _assert_dynamic_bootstrap_parallel_parity(
        fitted, u, fit_result, bootstrap_refit=bootstrap_refit)

    assert fitted.fit_result is original_fit_result
    assert fitted._last_u is original_history


@pytest.mark.parametrize("method", ["mle", "gas", "scar-tm-ou"])
@pytest.mark.parametrize("bootstrap_refit", [False, True])
def test_stochastic_student_bootstrap_is_partition_invariant(
        method, bootstrap_refit):
    from pyscarcopula import StochasticStudentCopula

    correlation = np.array([
        [1.0, 0.25, 0.10],
        [0.25, 1.0, 0.15],
        [0.10, 0.15, 1.0],
    ])
    source = StochasticStudentCopula(3, R=correlation)
    u = source.sample_at_parameter(
        24,
        np.full(24, 6.0),
        rng=np.random.default_rng(925),
    )
    fitted = StochasticStudentCopula(3, R=correlation)
    fit_result = fitted.fit(
        u, method=method, **_dynamic_fit_kwargs(method))
    original_correlation = fitted.R
    original_fit_result = fitted.fit_result

    _assert_dynamic_bootstrap_parallel_parity(
        fitted, u, fit_result, bootstrap_refit=bootstrap_refit)

    np.testing.assert_array_equal(fitted.R, original_correlation)
    assert fitted.fit_result is original_fit_result


@pytest.mark.parametrize(
    ("method", "constructor_kwargs"),
    [
        ("mle", {"corr_mode": "shrinkage"}),
        ("gas", {"corr_mode": "shrinkage"}),
        ("mle", {"corr_mode": "cholesky"}),
        (
            "mle",
            {
                "corr_mode": "factor",
                "factor_rank": 1,
                "factor_loadings": np.array(
                    [[0.40], [0.20], [-0.15]]),
            },
        ),
        (
            "gas",
            {
                "corr_mode": "factor",
                "factor_rank": 1,
                "factor_loadings": np.array(
                    [[0.40], [0.20], [-0.15]]),
            },
        ),
        (
            "scar-tm-ou",
            {
                "corr_mode": "factor",
                "factor_rank": 1,
                "factor_loadings": np.array(
                    [[0.40], [0.20], [-0.15]]),
            },
        ),
    ],
)
def test_stochastic_student_estimated_correlation_modes(
        method, constructor_kwargs):
    from pyscarcopula import StochasticStudentCopula

    loadings = np.array([[0.40], [0.20], [-0.15]])
    source = StochasticStudentCopula(
        3,
        corr_mode="factor",
        factor_rank=1,
        factor_loadings=loadings,
    )
    u = source.sample_at_parameter(
        24,
        np.full(24, 6.0),
        rng=np.random.default_rng(926),
    )
    fitted = StochasticStudentCopula(3, **constructor_kwargs)
    fit_result = fitted.fit(
        u, method=method, **_dynamic_fit_kwargs(method))
    original_state = (
        fitted.factor_loadings_
        if fitted.corr_mode == "factor"
        else fitted.R
    )

    _assert_dynamic_bootstrap_parallel_parity(
        fitted, u, fit_result, bootstrap_refit=True)

    current_state = (
        fitted.factor_loadings_
        if fitted.corr_mode == "factor"
        else fitted.R
    )
    np.testing.assert_array_equal(current_state, original_state)


def test_stochastic_student_dynamic_stateless_bootstrap_requires_corr_state():
    from pyscarcopula import StochasticStudentCopula
    from pyscarcopula.stattests import gof_test

    source = StochasticStudentCopula(3, R=np.eye(3))
    u = source.sample_at_parameter(
        20,
        np.full(20, 6.0),
        rng=np.random.default_rng(927),
    )
    fitted = StochasticStudentCopula(3, corr_mode="shrinkage")
    fit_result = fitted.fit(
        u, method="gas", **_dynamic_fit_kwargs("gas"))
    prototype = StochasticStudentCopula(3, corr_mode="shrinkage")

    with pytest.raises(ValueError, match="fitted correlation state"):
        gof_test(
            prototype,
            u,
            fit_result=fit_result,
            to_pobs=False,
            bootstrap=True,
            n_bootstrap=1,
            bootstrap_refit=False,
            rng=928,
        )


def test_parallel_bootstrap_forces_one_native_thread_per_worker():
    from scipy.stats import norm

    from pyscarcopula import GaussianCopula, NumericalConfig
    from pyscarcopula.stattests import gof_test

    correlation = np.array([[1.0, 0.3], [0.3, 1.0]])
    latent = np.random.default_rng(922).multivariate_normal(
        np.zeros(2), correlation, size=30)
    u = norm.cdf(latent)
    model = GaussianCopula()
    fit_result = model.fit(u)
    kwargs = dict(
        fit_result=fit_result,
        to_pobs=False,
        bootstrap=True,
        n_bootstrap=2,
        bootstrap_refit=False,
        bootstrap_fit_kwargs={
            "config": NumericalConfig(n_threads=2),
        },
        rng=923,
    )

    sequential = gof_test(model, u, n_jobs=1, **kwargs)
    parallel = gof_test(model, u, n_jobs=2, **kwargs)

    assert sequential.n_threads == 2
    assert parallel.n_threads == 1
    np.testing.assert_array_equal(
        parallel.bootstrap_statistics,
        sequential.bootstrap_statistics,
    )


@pytest.mark.parametrize("n_jobs", [0, -2, True, 1.5, "2"])
def test_bootstrap_rejects_invalid_n_jobs(n_jobs):
    from pyscarcopula.stattests import gof_test

    model, u, fit_result = _fit_bivariate("mle")

    with pytest.raises(ValueError, match="n_jobs"):
        gof_test(
            model,
            u,
            fit_result=fit_result,
            to_pobs=False,
            bootstrap=True,
            n_bootstrap=2,
            rng=914,
            n_jobs=n_jobs,
        )


def test_nonbootstrap_gof_ignores_n_jobs():
    from pyscarcopula.stattests import gof_test

    model, u, fit_result = _fit_bivariate("mle")

    result = gof_test(
        model,
        u,
        fit_result=fit_result,
        to_pobs=False,
        bootstrap=False,
        n_jobs=0,
    )

    assert np.isfinite(result.statistic)


def test_bootstrap_worker_error_identifies_replication(monkeypatch):
    import pyscarcopula.stattests as st
    from dataclasses import replace
    from pyscarcopula.stattests import gof_test

    model, u, fit_result = _fit_bivariate("mle")

    def fail_simulation(*args, **kwargs):
        raise ValueError("simulation failed")

    monkeypatch.setitem(
        st._BOOTSTRAP_ADAPTERS, "bivariate",
        replace(st._BOOTSTRAP_ADAPTERS["bivariate"], simulate=fail_simulation))
    with pytest.raises(RuntimeError, match="bootstrap iteration 1 failed"):
        gof_test(
            model,
            u,
            fit_result=fit_result,
            to_pobs=False,
            bootstrap=True,
            n_bootstrap=2,
            bootstrap_refit=True,
            rng=915,
            n_jobs=1,
        )
