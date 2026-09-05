"""Lamperti--Euler Jacobi sampling contracts and validation checks."""

import os

import numpy as np
import pytest
from scipy.special import betainc
from scipy.stats import norm

from tools.benchmark_timing import interleaved_timings
from pyscarcopula import GumbelCopula, VineCopula
from pyscarcopula.api import sample
from pyscarcopula._types import LatentResult, jacobi_params
from pyscarcopula.numerical.jacobi_sampling import (
    sample_jacobi_lamperti_trajectory,
)
from pyscarcopula.strategy._base import get_strategy_for_result
from pyscarcopula.strategy._base import get_strategy
from pyscarcopula.vine._edge_adapter import sample_r_path


class _DeterministicRNG:
    def __init__(self, initial_uniform, innovations):
        self.initial_uniform = float(initial_uniform)
        self.innovations = iter(innovations)

    def uniform(self, low=0.0, high=1.0, size=None):
        assert low == 0.0 and high == 1.0
        if size is None:
            return self.initial_uniform
        return np.full(size, self.initial_uniform, dtype=np.float64)

    def standard_normal(self, size=None):
        if size is None:
            return next(self.innovations)
        values = np.fromiter(
            (next(self.innovations) for _ in range(int(np.prod(size)))),
            dtype=np.float64,
            count=int(np.prod(size)),
        )
        return values.reshape(size)


def _jacobi_result(**kwargs):
    options = {
        "sampling_method": "lamperti_euler",
        "lamperti_substeps": 8,
        "lamperti_boundary": "reflect",
        "lamperti_eps": 1e-10,
    }
    options.update(kwargs)
    return LatentResult(
        log_likelihood=0.0,
        method="SCAR-TM-JACOBI",
        copula_name="Gumbel copula",
        success=True,
        params=jacobi_params(1.2, 0.4, 0.25),
        transition_method="local_fixed",
        spectral_basis_order=4,
        spectral_quad_order=20,
        **options,
    )


def test_lamperti_one_step_matches_independent_ito_formula():
    kappa, m, xi = 1.2, 0.4, 0.25
    tau0 = 0.37
    innovation = 0.2
    alpha = 2.0 * kappa * m / xi**2
    beta = 2.0 * kappa * (1.0 - m) / xi**2
    rng = _DeterministicRNG(betainc(alpha, beta, tau0), [innovation])

    actual = sample_jacobi_lamperti_trajectory(
        kappa, m, xi, 2, rng=rng, substeps=1)

    root = np.sqrt(tau0 * (1.0 - tau0))
    drift = (
        kappa * (m - tau0) / (xi * root)
        - xi * (1.0 - 2.0 * tau0) / (4.0 * root)
    )
    y0 = 2.0 / xi * np.arcsin(np.sqrt(tau0))
    expected = np.sin(0.5 * xi * (y0 + drift + innovation)) ** 2

    assert actual[0] == pytest.approx(tau0)
    assert actual[1] == pytest.approx(expected, rel=1e-14, abs=1e-14)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (-0.25, 0.25),
        (1.25, 0.75),
        (2.25, 0.25),
        (1234.25, 0.25),
        (-1234.25, 0.25),
    ],
)
def test_lamperti_reflection_maps_to_closed_interval_in_native_cpp(
        value, expected):
    from pyscarcopula._native._extension import load

    module = load()
    result = module.jacobi_apply_boundary(
        value, 1.0, module.JacobiBoundaryPolicy.Reflect)
    assert int(result["status"]) == 0
    assert result["value"] == pytest.approx(expected)


def test_lamperti_boundary_diagnostics_are_explicit():
    path, diagnostics = sample_jacobi_lamperti_trajectory(
        0.1,
        0.5,
        1.0,
        30,
        rng=np.random.default_rng(2),
        substeps=1,
        boundary="reflect",
        return_diagnostics=True,
    )

    assert np.all(np.isfinite(path))
    assert np.all((path >= 0.0) & (path <= 1.0))
    assert diagnostics["boundary_interventions"] > 0
    assert diagnostics["boundary_intervention_rate"] == pytest.approx(
        diagnostics["boundary_interventions"]
        / diagnostics["euler_steps"])
    assert diagnostics["sampling_engine"] == "native"
    assert diagnostics["stationary_boundary_singular"]


def test_lamperti_zero_length_does_not_advance_rng():
    rng = np.random.default_rng(12)
    path = sample_jacobi_lamperti_trajectory(
        1.2, 0.4, 0.25, 0, rng=rng)

    assert path.shape == (0,)
    np.testing.assert_array_equal(
        rng.random(8), np.random.default_rng(12).random(8))


def test_lamperti_memory_guard_precedes_rng_draw():
    rng = np.random.default_rng(13)
    with pytest.raises(MemoryError, match="memory_budget_bytes"):
        sample_jacobi_lamperti_trajectory(
            1.2,
            0.4,
            0.25,
            10,
            rng=rng,
            memory_budget_bytes=79,
        )
    np.testing.assert_array_equal(
        rng.random(8), np.random.default_rng(13).random(8))


@pytest.mark.parametrize(
    ("option", "value", "error"),
    [
        ("substeps", True, TypeError),
        ("substeps", 0, ValueError),
        ("boundary", 1, TypeError),
        ("boundary", "unknown", ValueError),
        ("eps", True, TypeError),
        ("eps", np.nan, ValueError),
        ("eps", 0.5, ValueError),
        ("engine", 1, TypeError),
        ("engine", "parallel", ValueError),
        ("chunk_observations", False, TypeError),
        ("chunk_observations", 0, ValueError),
    ],
)
def test_lamperti_rejects_invalid_options(option, value, error):
    kwargs = {option: value}
    with pytest.raises(error):
        sample_jacobi_lamperti_trajectory(
            1.2, 0.4, 0.25, 2, **kwargs)


def test_fitted_result_restores_lamperti_sampler_and_diagnostics():
    result = _jacobi_result(
        lamperti_substeps=4,
        lamperti_boundary="clip",
        lamperti_eps=2e-9,
        lamperti_engine="python",
        lamperti_chunk_observations=3,
        memory_budget_bytes=100_000,
    )
    strategy = get_strategy_for_result(result)
    diagnostics = {}
    copula = GumbelCopula()

    first = strategy.model_sample_params(
        copula,
        result,
        12,
        rng=np.random.default_rng(14),
        sampling_diagnostics=diagnostics,
    )
    second = strategy.model_sample_params(
        copula,
        result,
        12,
        rng=np.random.default_rng(14),
    )

    np.testing.assert_array_equal(first, second)
    assert strategy.sampling_method == "lamperti_euler"
    assert strategy.lamperti_substeps == 4
    assert strategy.lamperti_boundary == "clip"
    assert strategy.lamperti_eps == pytest.approx(2e-9)
    assert strategy.lamperti_engine == "native"
    assert strategy.lamperti_chunk_observations == 3
    assert strategy.memory_budget_bytes == 100_000
    assert diagnostics["sampling_method"] == "lamperti_euler"
    assert diagnostics["sampling_engine"] == "native"
    assert diagnostics["substeps"] == 4
    assert not diagnostics["stationary_boundary_singular"]


def test_public_sample_uses_lamperti_backend_without_changing_default():
    result = _jacobi_result()
    copula = GumbelCopula()
    u = np.random.default_rng(15).uniform(size=(8, 2))
    diagnostics = {}

    first = sample(
        copula,
        u,
        result,
        20,
        rng=np.random.default_rng(16),
        sampling_diagnostics=diagnostics,
    )
    second = sample(
        copula,
        u,
        result,
        20,
        rng=np.random.default_rng(16),
    )

    np.testing.assert_array_equal(first, second)
    assert first.shape == (20, 2)
    assert np.all((first >= 0.0) & (first <= 1.0))
    assert diagnostics["sampling_method"] == "lamperti_euler"
    assert get_strategy("scar-tm-jacobi").sampling_method == "tm_grid"


def test_vine_edge_adapter_uses_persisted_lamperti_backend():
    result = _jacobi_result(lamperti_substeps=2)
    diagnostics_path = sample_r_path(
        GumbelCopula(),
        result,
        10,
        rng=np.random.default_rng(17),
    )
    direct_path = get_strategy_for_result(result).model_sample_params(
        GumbelCopula(),
        result,
        10,
        rng=np.random.default_rng(17),
    )

    np.testing.assert_array_equal(diagnostics_path, direct_path)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: VineCopula.cvine(3, candidates=[GumbelCopula]),
        lambda: VineCopula.rvine(candidates=[GumbelCopula]),
    ],
    ids=["cvine", "rvine"],
)
def test_generic_vines_accept_vector_jacobi_alpha0_and_sample(factory):
    rng = np.random.default_rng(18)
    latent = 0.8 * rng.normal(size=(35, 1))
    data = norm.cdf(latent + 0.6 * rng.normal(size=(35, 3)))
    vine = factory().fit(
        data,
        method="scar-tm-jacobi",
        alpha0=np.array([1.2, 0.4, 0.25]),
        basis_order=3,
        quad_order=16,
        transition_method="local_fixed",
        sampling_method="lamperti_euler",
        lamperti_substeps=2,
        smart_init=False,
        maxiter=1,
        maxfun=8,
    )

    sampled = vine.sample(8, rng=np.random.default_rng(19))

    assert vine.fit_result.diagnostics["dynamic_attempted_count"] == 3
    assert sampled.shape == (8, 3)
    assert np.all(np.isfinite(sampled))
    assert np.all((sampled >= 0.0) & (sampled <= 1.0))


@pytest.mark.parametrize(
    ("option", "value", "error"),
    [
        ("sampling_method", 1, TypeError),
        ("sampling_method", "exact", ValueError),
        ("lamperti_substeps", False, TypeError),
        ("lamperti_boundary", "wrap", ValueError),
        ("lamperti_eps", np.inf, ValueError),
        ("lamperti_engine", "parallel", ValueError),
        ("lamperti_chunk_observations", 1.5, TypeError),
    ],
)
def test_strategy_rejects_invalid_lamperti_configuration(
        option, value, error):
    with pytest.raises(error):
        get_strategy("scar-tm-jacobi", **{option: value})


@pytest.mark.parametrize(
    ("parameters", "boundary"),
    [
        ((1.2, 0.4, 0.25), "reflect"),
        ((0.5, 0.5, np.sqrt(0.5)), "reflect"),
        ((0.1, 0.5, 0.5), "reflect"),
        ((0.1, 0.2, 1.0), "clip"),
    ],
)
def test_lamperti_legacy_engine_aliases_share_native_path(parameters, boundary):
    kwargs = dict(
        substeps=4,
        boundary=boundary,
        chunk_observations=3,
        return_diagnostics=True,
    )
    python_rng = np.random.default_rng(101)
    numba_rng = np.random.default_rng(101)

    python_path, python_diagnostics = sample_jacobi_lamperti_trajectory(
        *parameters, 40, rng=python_rng, engine="python", **kwargs)
    numba_path, numba_diagnostics = sample_jacobi_lamperti_trajectory(
        *parameters, 40, rng=numba_rng, engine="numba", **kwargs)

    np.testing.assert_allclose(
        numba_path, python_path, rtol=2e-14, atol=2e-14)
    assert (
        numba_diagnostics["boundary_interventions"]
        == python_diagnostics["boundary_interventions"])
    np.testing.assert_array_equal(
        numba_rng.random(8), python_rng.random(8))


@pytest.mark.parametrize("engine", ["python", "numba"])
def test_lamperti_chunk_size_does_not_change_path_or_rng(engine):
    common = dict(
        kappa=0.1,
        m=0.5,
        xi=0.5,
        n=37,
        substeps=5,
        boundary="reflect",
        engine=engine,
        return_diagnostics=True,
    )
    small_rng = np.random.default_rng(102)
    large_rng = np.random.default_rng(102)
    small, small_diagnostics = sample_jacobi_lamperti_trajectory(
        rng=small_rng, chunk_observations=1, **common)
    large, large_diagnostics = sample_jacobi_lamperti_trajectory(
        rng=large_rng, chunk_observations=1000, **common)

    np.testing.assert_array_equal(small, large)
    assert (
        small_diagnostics["boundary_interventions"]
        == large_diagnostics["boundary_interventions"])
    assert small_diagnostics["chunk_observations"] == 1
    assert large_diagnostics["chunk_observations"] == 36
    assert small_diagnostics["sampling_engine"] == engine
    assert large_diagnostics["sampling_engine"] == engine
    np.testing.assert_array_equal(
        small_rng.random(8), large_rng.random(8))


def test_lamperti_chunk_shrinks_to_memory_budget_before_rng_draws():
    n = 10
    substeps = 4
    elements_per_interval = 2 * substeps + 2
    budget = (n + 2 * elements_per_interval) * 8
    path, diagnostics = sample_jacobi_lamperti_trajectory(
        1.2,
        0.4,
        0.25,
        n,
        rng=np.random.default_rng(103),
        substeps=substeps,
        chunk_observations=9,
        memory_budget_bytes=budget,
        return_diagnostics=True,
    )

    assert path.shape == (n,)
    assert diagnostics["chunk_observations_requested"] == 9
    assert diagnostics["chunk_observations"] == 2


@pytest.mark.benchmark
def test_lamperti_native_warm_throughput_report():
    if os.environ.get("PYSCA_RUN_BENCHMARKS") != "1":
        pytest.skip("set PYSCA_RUN_BENCHMARKS=1 to run benchmark gates")
    substeps = 16
    n = 62_501  # exactly one million sequential Euler updates
    sample_jacobi_lamperti_trajectory(
        1.2,
        0.4,
        0.25,
        2,
        rng=np.random.default_rng(200),
        substeps=substeps,
        engine="native",
    )
    measured = interleaved_timings(
        {
            "native_chunked": lambda: sample_jacobi_lamperti_trajectory(
                1.2,
                0.4,
                0.25,
                n,
                rng=np.random.default_rng(201),
                substeps=substeps,
                engine="native",
                chunk_observations=4096,
            ),
            "native_single_interval": (
                lambda: sample_jacobi_lamperti_trajectory(
                    1.2,
                    0.4,
                    0.25,
                    n,
                    rng=np.random.default_rng(201),
                    substeps=substeps,
                    engine="native",
                    chunk_observations=1,
                )
            ),
        },
        repeats=3,
    )
    medians = measured.medians
    chunk_speedup = measured.median_ratio(
        "native_single_interval", "native_chunked")

    print(
        "lamperti_native_warm "
        f"chunked_seconds={medians['native_chunked']:.6f} "
        f"single_interval_seconds={medians['native_single_interval']:.6f} "
        f"chunk_speedup={chunk_speedup:.2f}x"
    )
    assert chunk_speedup >= 2.0


@pytest.mark.validation
def test_lamperti_full_path_marginal_matches_stationary_beta():
    kappa, m, xi = 1.2, 0.4, 0.25
    alpha = 2.0 * kappa * m / xi ** 2
    beta = 2.0 * kappa * (1.0 - m) / xi ** 2
    expected_variance = (
        alpha * beta
        / ((alpha + beta) ** 2 * (alpha + beta + 1.0))
    )
    rng = np.random.default_rng(20260728)
    endpoints = np.array([
        sample_jacobi_lamperti_trajectory(
            kappa,
            m,
            xi,
            5,
            rng=rng,
            substeps=16,
        )[-1]
        for _ in range(2000)
    ])

    assert endpoints.mean() == pytest.approx(m, abs=0.012)
    assert endpoints.var() == pytest.approx(
        expected_variance, rel=0.15)
