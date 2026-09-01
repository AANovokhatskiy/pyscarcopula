import numpy as np
import pytest


def _datasets(n_tasks=2, n=24, d=2):
    from pyscarcopula import GaussianCopula

    rng = np.random.default_rng(810)
    corr = np.full((d, d), 0.2)
    np.fill_diagonal(corr, 1.0)
    source = GaussianCopula()
    source.corr = corr
    source._set_dimension(d, allow_change=True)
    return [source.sample(n, rng=rng) for _ in range(n_tasks)]


def test_fit_independent_sequential_owns_one_model_per_task():
    import multiprocessing as mp

    from pyscarcopula import GaussianCopula
    from pyscarcopula.contrib.parallel_fit import fit_independent

    prototype = GaussianCopula()
    batch = fit_independent(prototype, _datasets(3), n_jobs=1)

    assert len(batch.fits) == 3
    assert len({id(model) for model in batch.models}) == 3
    assert prototype.fit_result is None
    assert all(model.fit_result is result for model, result in zip(
        batch.models, batch.results))
    assert batch.diagnostics == {
        "n_tasks": 3,
        "n_jobs_requested": 1,
        "n_jobs": 1,
        "n_threads_requested": None,
        "n_threads": 1,
        "multiprocessing_start_method": mp.get_context().get_start_method(),
        "nested_parallelism": False,
        "worker_model_ownership": "per_task",
        "prepared_evaluator_sharing": False,
    }


def test_fit_independent_process_default_disables_inner_threads():
    from pyscarcopula import EquicorrGaussianCopula
    from pyscarcopula.contrib.parallel_fit import fit_independent

    batch = fit_independent(
        EquicorrGaussianCopula(d=2),
        _datasets(2),
        n_jobs=2,
        mp_start_method="spawn",
    )

    assert batch.diagnostics["n_jobs"] == 2
    assert batch.diagnostics["n_threads"] == 1
    assert batch.diagnostics["nested_parallelism"] is False
    assert all(item.result.diagnostics["n_threads"] == 1
               for item in batch.fits)
    assert all(item.result.success for item in batch.fits)
    assert all(item.model.fit_result is item.result for item in batch.fits)


def test_fit_independent_explicit_threads_records_nested_opt_in():
    from pyscarcopula import EquicorrGaussianCopula
    from pyscarcopula.contrib.parallel_fit import fit_independent

    batch = fit_independent(
        EquicorrGaussianCopula(d=2),
        _datasets(2),
        n_jobs=2,
        n_threads=2,
        mp_start_method="spawn",
        fit_kwargs={"maxiter": 2, "maxfun": 8},
    )

    assert batch.diagnostics["n_threads_requested"] == 2
    assert batch.diagnostics["n_threads"] == 2
    assert batch.diagnostics["nested_parallelism"] is True
    assert all(item.result.diagnostics["n_threads"] == 2
               for item in batch.fits)


@pytest.mark.parametrize("n_jobs", [0, -2, True, 1.5])
def test_fit_independent_rejects_invalid_n_jobs(n_jobs):
    from pyscarcopula import GaussianCopula
    from pyscarcopula.contrib.parallel_fit import fit_independent

    with pytest.raises(ValueError, match="n_jobs"):
        fit_independent(GaussianCopula(), _datasets(1), n_jobs=n_jobs)


def test_fit_independent_supports_task_specific_starting_points():
    from pyscarcopula import BivariateGaussianCopula
    from pyscarcopula.contrib.parallel_fit import fit_independent

    batch = fit_independent(
        BivariateGaussianCopula(),
        _datasets(2),
        fit_kwargs=[
            {"alpha0": np.array([-0.25]), "maxiter": 2},
            {"alpha0": np.array([0.25]), "maxiter": 2},
        ],
        n_jobs=1,
    )

    assert len(batch.results) == 2
    assert all(np.isfinite(result.log_likelihood) for result in batch.results)


@pytest.mark.parametrize("n_jobs", [1, 2])
@pytest.mark.parametrize("key", ["unknown_option", "K", "tol"])
def test_fit_independent_rejects_invalid_options_before_any_fit(monkeypatch, n_jobs, key):
    from pyscarcopula import BivariateGaussianCopula
    from pyscarcopula.contrib import parallel_fit

    def unexpected_work(*args, **kwargs):
        pytest.fail("validation must precede fitting and process creation")

    monkeypatch.setattr(parallel_fit, "_fit_worker", unexpected_work)
    mp_context = parallel_fit.mp.get_context()
    monkeypatch.setattr(mp_context, "Pool", unexpected_work)
    monkeypatch.setattr(parallel_fit.mp, "get_context", lambda *args: mp_context)
    with pytest.raises(TypeError, match=key):
        parallel_fit.fit_independent(
            BivariateGaussianCopula(), _datasets(2), n_jobs=n_jobs,
            fit_kwargs=[{}, {key: 17}])


@pytest.mark.parametrize("n_jobs", [1, 2])
@pytest.mark.parametrize("object_array", [False, True])
def test_fit_independent_rejects_complex_data_before_fitting(monkeypatch, n_jobs, object_array):
    from pyscarcopula import BivariateGaussianCopula
    from pyscarcopula.contrib import parallel_fit

    data = _datasets(1)[0].astype(complex) + .2j
    if object_array:
        data = data.astype(object)
    monkeypatch.setattr(parallel_fit, "_fit_worker",
                        lambda *args: pytest.fail("invalid data reached fit"))
    with pytest.raises(TypeError, match="real|complex"):
        parallel_fit.fit_independent(BivariateGaussianCopula(), [data], n_jobs=n_jobs)


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_fit_independent_preserves_unsuccessful_result_without_publishing(n_jobs):
    from pyscarcopula import EquicorrGaussianCopula
    from pyscarcopula.contrib.parallel_fit import fit_independent

    batch = fit_independent(
        EquicorrGaussianCopula(d=2), _datasets(2), n_jobs=n_jobs,
        mp_start_method="spawn", fit_kwargs={"maxiter": 1, "maxfun": 2})
    assert all(not item.result.success for item in batch.fits)
    assert all(item.model.fit_result is None for item in batch.fits)


@pytest.mark.parametrize("method", ["gas", "scar-tm-ou"])
def test_fit_independent_rejects_retired_backend_before_worker(monkeypatch, method):
    from pyscarcopula import BivariateGaussianCopula
    from pyscarcopula.contrib import parallel_fit

    monkeypatch.setattr(parallel_fit, "_fit_worker",
                        lambda *args: pytest.fail("retired key reached a fit"))
    with pytest.raises(TypeError, match="backend"):
        parallel_fit.fit_independent(
            BivariateGaussianCopula(), _datasets(1), method=method,
            fit_kwargs={"backend": "auto"})


@pytest.mark.parametrize("model_name", [
    "EquicorrGaussianCopula", "StochasticStudentCopula",
])
@pytest.mark.parametrize("key", ["alpha0", "_prepared_evaluator"])
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_multivariate_mle_options_are_checked_before_worker(
        monkeypatch, model_name, key, n_jobs):
    import pyscarcopula as p
    from pyscarcopula.contrib import parallel_fit

    def unexpected(*args, **kwargs):
        pytest.fail("model-specific MLE option reached worker creation")

    monkeypatch.setattr(parallel_fit, "_fit_worker", unexpected)
    context = parallel_fit.mp.get_context()
    monkeypatch.setattr(context, "Pool", unexpected)
    monkeypatch.setattr(parallel_fit.mp, "get_context", lambda *args: context)
    with pytest.raises(TypeError, match=key):
        parallel_fit.fit_independent(
            getattr(p, model_name)(d=2), _datasets(2), n_jobs=n_jobs,
            fit_kwargs=[{}, {key: [.2]}])


def test_risk_metrics_records_resolved_parallel_policy(monkeypatch):
    from pyscarcopula.contrib import risk_metrics as risk_module
    from pyscarcopula.contrib.marginal import MarginalModel

    calls = {}

    class Marginal:
        def fit_rolling(self, data, window_len, n_jobs=-1):
            calls["marginal_n_jobs"] = n_jobs
            return np.zeros((len(data), data.shape[1], 1))

    class DummyCopula:
        _rotate = 0

    monkeypatch.setattr(
        MarginalModel, "create", staticmethod(lambda name: Marginal()))

    def calculate(*args, n_jobs=1, mp_start_method=None, **kwargs):
        calls["cvar_n_jobs"] = n_jobs
        calls["fit_n_threads"] = kwargs["config"].n_threads
        data = args[1]
        return np.zeros(len(data)), np.zeros(len(data)), args[8]

    monkeypatch.setattr(risk_module, "_calculate_cvar_fixed", calculate)
    result = risk_module.risk_metrics(
        DummyCopula(),
        np.arange(12, dtype=np.float64).reshape(6, 2),
        window_len=3,
        gamma=0.9,
        N_mc=10,
        optimize_portfolio=False,
        n_jobs=-1,
        rng=811,
    )[0.9][10]

    resolved_n_jobs = calls["marginal_n_jobs"]

    assert resolved_n_jobs > 1
    assert calls == {
        "marginal_n_jobs": resolved_n_jobs,
        "cvar_n_jobs": resolved_n_jobs,
        "fit_n_threads": 1,
    }

    diagnostics = result["diagnostics"]
    assert diagnostics["n_jobs_requested"] == -1
    assert diagnostics["n_jobs"] == resolved_n_jobs
    assert diagnostics["n_threads"] == 1
    assert diagnostics["nested_parallelism"] is False
