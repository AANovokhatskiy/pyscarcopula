import multiprocessing as mp

import numpy as np
import pytest

from pyscarcopula._parallel import (
    create_worker_model,
    resolve_parallelism,
    spawn_seed_sequences,
    validate_n_jobs,
)
from pyscarcopula._types import NumericalConfig


@pytest.mark.parametrize(
    ("n_jobs", "n_tasks", "expected"),
    [
        (1, 3, (1, 1)),
        (8, 3, (8, 3)),
        (2, 0, (2, 1)),
    ],
)
def test_validate_n_jobs_caps_workers_to_tasks(n_jobs, n_tasks, expected):
    assert validate_n_jobs(n_jobs, n_tasks) == expected


@pytest.mark.parametrize("n_jobs", [0, -2, True, 1.5, "2"])
def test_validate_n_jobs_rejects_invalid_values(n_jobs):
    with pytest.raises(ValueError, match="n_jobs"):
        validate_n_jobs(n_jobs, 2)


@pytest.mark.parametrize("n_tasks", [-1, True, 1.5])
def test_validate_n_jobs_rejects_invalid_task_counts(n_tasks):
    error = ValueError if n_tasks == -1 else TypeError
    with pytest.raises(error, match="n_tasks"):
        validate_n_jobs(1, n_tasks)


def test_resolve_parallelism_preserves_serial_configured_threads():
    config = NumericalConfig(n_threads=2)
    resolved_threads, diagnostics = resolve_parallelism(
        1,
        3,
        None,
        ({"config": config},),
    )

    assert resolved_threads == 2
    assert diagnostics == {
        "n_tasks": 3,
        "n_jobs_requested": 1,
        "n_jobs": 1,
        "n_threads_requested": None,
        "n_threads": 2,
        "multiprocessing_start_method":
            mp.get_context().get_start_method(),
        "nested_parallelism": False,
        "worker_model_ownership": "per_task",
        "prepared_evaluator_sharing": False,
    }


def test_spawn_seed_sequences_is_reproducible_and_independent():
    first = spawn_seed_sequences(20260730, 4)
    second = spawn_seed_sequences(20260730, 4)

    first_draws = [
        np.random.default_rng(seed).integers(0, 2**63, size=8)
        for seed in first
    ]
    second_draws = [
        np.random.default_rng(seed).integers(0, 2**63, size=8)
        for seed in second
    ]

    for actual, expected in zip(first_draws, second_draws):
        np.testing.assert_array_equal(actual, expected)
    assert len({tuple(draws) for draws in first_draws}) == len(first_draws)


def test_spawn_seed_sequences_accepts_generator_without_sharing_it():
    source = np.random.default_rng(91)
    seeds = spawn_seed_sequences(source, 2)

    assert len(seeds) == 2
    assert all(isinstance(seed, np.random.SeedSequence) for seed in seeds)
    assert np.random.default_rng(seeds[0]) is not source


def test_create_worker_model_preserves_constructor_policy_not_fit_state():
    from pyscarcopula import BivariateGaussianCopula

    source = BivariateGaussianCopula(transform_type="xtanh")
    source.fit_result = object()
    source._last_u = np.full((3, 2), 0.5)

    rebuilt = create_worker_model(source)

    assert rebuilt is not source
    assert rebuilt._transform_type == "xtanh"
    assert rebuilt.fit_result is None
    assert not hasattr(rebuilt, "_last_u")


def test_legacy_private_imports_alias_shared_runtime():
    from pyscarcopula import _parallel
    from pyscarcopula.contrib import parallel_fit, risk_metrics

    assert parallel_fit._validate_n_jobs is _parallel.validate_n_jobs
    assert parallel_fit._resolve_parallelism is _parallel.resolve_parallelism
    assert risk_metrics._resolve_parallelism is _parallel.resolve_parallelism
    assert (
        risk_metrics._get_copula_constructor
        is _parallel.get_copula_constructor
    )


@pytest.mark.parametrize("model_name", [
    "IndependentCopula", "BivariateGaussianCopula", "ClaytonCopula",
    "FrankCopula", "GumbelCopula", "JoeCopula", "GaussianCopula",
    "StudentCopula", "EquicorrGaussianCopula", "StochasticStudentCopula",
    "VineCopula",
])
@pytest.mark.parametrize("key", ["unknown_option", "K", "tol"])
def test_fit_preflight_rejects_unknown_or_wrong_method_keys(model_name, key):
    import pyscarcopula as p
    from pyscarcopula._parallel import validate_model_fit_kwargs

    constructor_options = {"d": 2} if model_name in {
        "EquicorrGaussianCopula", "StochasticStudentCopula"} else {}
    model = getattr(p, model_name)(**constructor_options)
    with pytest.raises(TypeError, match=key):
        validate_model_fit_kwargs(model, "mle", {key: 13})
    assert model.fit_result is None


@pytest.mark.parametrize("method, options", [
    ("mle", {"alpha0": [.2], "maxiter": 5}),
    ("gas", {"scaling": "fisher", "gamma0": [.1, .1, .7], "maxiter": 5}),
    ("scar-tm-ou", {"K": 17, "grid_range": 3., "alpha0": [2., .3, .4]}),
    ("scar-tm-jacobi", {"basis_order": 8, "alpha0": [2., .4, .3]}),
])
def test_fit_preflight_preserves_constructor_and_optimizer_options(method, options):
    from pyscarcopula import BivariateGaussianCopula
    from pyscarcopula._parallel import validate_model_fit_kwargs

    config = NumericalConfig(n_threads=2)
    kwargs = {**options, "config": config, "to_pobs": False}
    before = dict(kwargs)
    model = BivariateGaussianCopula()
    validate_model_fit_kwargs(model, method, kwargs)
    assert kwargs == before
    assert kwargs["config"] is config
    assert model.fit_result is None


def test_fit_preflight_respects_static_and_vine_specific_owners():
    from pyscarcopula import GaussianCopula, StudentCopula, VineCopula
    from pyscarcopula._parallel import validate_model_fit_kwargs

    with pytest.raises(TypeError, match="corr_mode"):
        validate_model_fit_kwargs(GaussianCopula(), "mle", {"maxiter": 9})
    validate_model_fit_kwargs(GaussianCopula(corr_mode="shrinkage"), "mle", {"maxiter": 9})
    validate_model_fit_kwargs(StudentCopula(), "mle", {"maxiter": 9})
    vine = VineCopula()
    validate_model_fit_kwargs(vine, "gas", {
        "scaling": "fisher", "beam_width": 2, "dynamic_failure_policy": "raise",
        "conditional_mode": "suffix", "maxiter": 9})
    for key in ("initial_mle_result", "_prepared_evaluator"):
        with pytest.raises(TypeError, match=key):
            validate_model_fit_kwargs(vine, "mle", {key: object()})


@pytest.mark.parametrize("options", [{"method": "gas"}, {"data": []}, {"config": {}}])
def test_fit_preflight_rejects_duplicate_arguments_and_invalid_config(options):
    from pyscarcopula import BivariateGaussianCopula
    from pyscarcopula._parallel import validate_model_fit_kwargs

    with pytest.raises(TypeError):
        validate_model_fit_kwargs(BivariateGaussianCopula(), "mle", options)


@pytest.mark.parametrize("model_name", [
    "EquicorrGaussianCopula", "StochasticStudentCopula",
])
def test_multivariate_mle_preflight_preserves_public_optimizer_options(model_name):
    import pyscarcopula as p
    from pyscarcopula._parallel import validate_model_fit_kwargs

    model = getattr(p, model_name)(d=2)
    options = {"config": NumericalConfig(n_threads=2), "to_pobs": False,
               "gtol": 1e-5, "ftol": 1e-8, "maxiter": 9, "maxfun": 80,
               "maxls": 10, "eps": 1e-6, "maxcor": 5,
               "finite_diff_rel_step": .01}
    before = dict(options)
    validate_model_fit_kwargs(model, "mle", options)
    assert options == before and model.fit_result is None
