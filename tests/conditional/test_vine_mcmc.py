"""Stage-6 contracts and analytical validation for vine DAG+MCMC."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import ks_2samp

from pyscarcopula import BivariateGaussianCopula, ClaytonCopula
from pyscarcopula.vine import cvine_structure, dvine_structure

from ._vine_fixtures import (
    arbitrary_given,
    exact_given,
    gaussian_vine,
    homogeneous_vine,
)
from ._vine_mcmc_oracles import gaussian_mcmc_error
from ._vine_oracles import gaussian_vine_correlation


DIMENSIONS = (4, 8, 12)
CONVERGENCE_SEEDS = (2026401, 2026402, 2026403, 2026404)


def _structure(dimension: int):
    order = list(range(dimension))
    factory = cvine_structure if dimension == 8 else dvine_structure
    return factory(dimension, order)


def _assert_fixed_bit_exact(samples, given):
    for variable, value in given.items():
        np.testing.assert_array_equal(
            samples[:, variable],
            np.full(len(samples), value, dtype=np.float64),
        )


@pytest.mark.parametrize("dimension", DIMENSIONS, ids=lambda d: f"d={d}")
def test_dag_mcmc_diagnostics_are_internally_consistent(dimension):
    vine = gaussian_vine(_structure(dimension))
    given = arbitrary_given(vine)
    n = 23
    samples, diagnostics = vine.predict(
        n,
        given=given,
        mcmc_steps=7,
        mcmc_burnin=5,
        return_diagnostics=True,
        rng=np.random.default_rng(2026410 + dimension),
    )

    assert diagnostics["conditional_method"] == "dag_mcmc"
    assert diagnostics["suffix_start_col"] is None
    assert diagnostics["matrix_rebuilt"] is False
    assert diagnostics["dag_steps"]
    assert diagnostics["dag_edges_used"]
    _assert_fixed_bit_exact(samples, given)

    mcmc = diagnostics["mcmc"]
    free = sorted(set(range(dimension)) - set(given))
    assert mcmc["step_unit"] == "single_coordinate_update"
    assert mcmc["n_free"] == len(free)
    assert mcmc["n_steps"] == 7
    assert mcmc["burnin_steps"] == 5
    assert mcmc["total_steps"] == 12
    assert mcmc["completed_sweeps"] == 12 // len(free)
    assert mcmc["partial_sweep_steps"] == 12 % len(free)
    assert sorted(mcmc["accepted"]) == free
    assert sorted(mcmc["proposed"]) == free
    assert sum(mcmc["proposed"].values()) == n * mcmc["total_steps"]

    for variable in free:
        accepted = mcmc["accepted"][variable]
        proposed = mcmc["proposed"][variable]
        assert 0 <= accepted <= proposed
        assert mcmc["acceptance_rate"][variable] == pytest.approx(
            accepted / proposed
        )
        assert mcmc["accepted_per_chain"][variable] == pytest.approx(
            accepted / n
        )
        assert mcmc["proposals_per_chain"][variable] == pytest.approx(
            proposed / n
        )

    rates = list(mcmc["acceptance_rate"].values())
    assert mcmc["acceptance_min"] == pytest.approx(min(rates))
    assert mcmc["acceptance_mean"] == pytest.approx(np.mean(rates))
    assert mcmc["acceptance_max"] == pytest.approx(max(rates))
    assert mcmc["minimum_accepted_moves_per_chain"] == pytest.approx(
        min(mcmc["accepted_per_chain"].values())
    )
    assert mcmc["convergence_warning"] is bool(mcmc["warning_codes"])


@pytest.mark.parametrize("dimension", DIMENSIONS, ids=lambda d: f"d={d}")
def test_default_mcmc_budget_is_dimension_scaled_in_coordinate_updates(
    dimension,
):
    vine = gaussian_vine(_structure(dimension))
    given = arbitrary_given(vine)
    _, diagnostics = vine.predict(
        5,
        given=given,
        return_diagnostics=True,
        rng=np.random.default_rng(2026420 + dimension),
    )

    n_free = dimension - len(given)
    expected_steps = max(80, 30 * n_free)
    expected_burnin = max(40, 10 * n_free)
    mcmc = diagnostics["mcmc"]
    assert mcmc["n_steps"] == expected_steps
    assert mcmc["burnin_steps"] == expected_burnin
    assert mcmc["completed_sweeps"] == (
        expected_steps + expected_burnin
    ) // n_free


def test_dag_mcmc_seed_reproducibility_includes_initializer_and_transitions():
    vine = gaussian_vine(_structure(4))
    given = arbitrary_given(vine, (0.19, 0.81))

    def draw(seed):
        return vine.predict(
            41,
            given=given,
            mcmc_steps=31,
            mcmc_burnin=11,
            return_diagnostics=True,
            rng=np.random.default_rng(seed),
        )

    first, first_diagnostics = draw(2026431)
    repeated, repeated_diagnostics = draw(2026431)
    different, _ = draw(2026432)

    np.testing.assert_array_equal(first, repeated)
    assert first_diagnostics["mcmc"] == repeated_diagnostics["mcmc"]
    assert not np.array_equal(first, different)


@pytest.mark.parametrize(
    ("keyword", "value", "error"),
    [
        ("mcmc_steps", -1, ValueError),
        ("mcmc_burnin", -1, ValueError),
        ("mcmc_steps", 1.5, TypeError),
        ("mcmc_burnin", True, TypeError),
    ],
)
def test_dag_mcmc_public_controls_reject_invalid_values(keyword, value, error):
    vine = gaussian_vine(_structure(4))
    given = arbitrary_given(vine)
    with pytest.raises(error):
        vine.predict(3, given=given, **{keyword: value})


def test_difficult_tail_case_sets_actionable_convergence_warning():
    vine = homogeneous_vine(
        dvine_structure(4, list(range(4))),
        ClaytonCopula,
        parameter=12.0,
    )
    given = arbitrary_given(vine, (0.001, 0.001))
    _, diagnostics = vine.predict(
        384,
        given=given,
        mcmc_steps=80,
        mcmc_burnin=40,
        return_diagnostics=True,
        rng=np.random.default_rng(2026441),
    )

    mcmc = diagnostics["mcmc"]
    assert mcmc["acceptance_min"] < 0.02
    assert mcmc["low_acceptance_warning"] is True
    assert mcmc["insufficient_moves_warning"] is True
    assert mcmc["convergence_warning"] is True
    assert set(mcmc["warning_codes"]) == {
        "low_acceptance",
        "insufficient_accepted_moves",
    }


@pytest.mark.validation
@pytest.mark.parametrize("dimension", DIMENSIONS, ids=lambda d: f"d={d}")
def test_default_budget_passes_gaussian_analytical_gate_across_four_chains(
    dimension,
):
    vine = gaussian_vine(_structure(dimension))
    levels = {
        4: (0.50, 0.50),
        8: (0.12, 0.88),
        12: (0.20, 0.80),
    }[dimension]
    given = arbitrary_given(vine, levels)
    correlation = gaussian_vine_correlation(vine)
    chains = []
    acceptance = []

    for seed in CONVERGENCE_SEEDS:
        samples, diagnostics = vine.predict(
            300,
            given=given,
            return_diagnostics=True,
            rng=np.random.default_rng(seed + dimension),
        )
        _assert_fixed_bit_exact(samples, given)
        chains.append(samples)
        acceptance.append(diagnostics["mcmc"]["acceptance_min"])

    error = gaussian_mcmc_error(np.vstack(chains), correlation, given)
    assert error.mean_rms < 0.08, error
    assert error.covariance_frobenius < 0.16, error
    assert error.marginal_ks < 0.065, error
    assert error.score < 0.25, error
    assert min(acceptance) > 0.25


@pytest.mark.validation
def test_gaussian_error_decreases_with_budget_in_aggregate():
    cases = (
        (4, 0.70, 96),
        (8, 0.35, 80),
        (12, 0.25, 48),
    )
    factors = (0, 1, 2, 4)
    scores = {factor: [] for factor in factors}

    for dimension, partial_correlation, n_per_chain in cases:
        vine = homogeneous_vine(
            dvine_structure(dimension, list(range(dimension))),
            BivariateGaussianCopula,
            parameter=partial_correlation,
        )
        given = arbitrary_given(vine, (0.12, 0.88))
        correlation = gaussian_vine_correlation(vine)
        n_free = dimension - len(given)
        default_steps = max(80, 30 * n_free)
        default_burnin = max(40, 10 * n_free)

        for factor in factors:
            chains = []
            for seed in CONVERGENCE_SEEDS:
                sample, diagnostics = vine.predict(
                    n_per_chain,
                    given=given,
                    mcmc_steps=factor * default_steps,
                    mcmc_burnin=factor * default_burnin,
                    return_diagnostics=True,
                    rng=np.random.default_rng(
                        seed + 100 * dimension + factor
                    ),
                )
                assert diagnostics["mcmc"]["n_steps"] == (
                    factor * default_steps
                )
                chains.append(sample)
            error = gaussian_mcmc_error(
                np.vstack(chains), correlation, given
            )
            scores[factor].append(error.score)

    aggregate = {
        factor: float(np.mean(scores[factor]))
        for factor in factors
    }
    dag_only = aggregate[0]
    assert aggregate[1] < 0.35 * dag_only, aggregate
    assert aggregate[2] < 0.35 * dag_only, aggregate
    assert aggregate[4] < 0.35 * dag_only, aggregate
    assert np.mean([aggregate[2], aggregate[4]]) <= 1.20 * aggregate[1], (
        aggregate
    )
    slope = np.polyfit(np.array(factors, dtype=float), [
        aggregate[factor] for factor in factors
    ], 1)[0]
    assert slope < 0.0, aggregate


@pytest.mark.validation
def test_forced_dag_mcmc_agrees_with_exact_suffix_for_same_gaussian_target(
    monkeypatch,
):
    vine = gaussian_vine(_structure(4))
    given = exact_given(vine, "direct")
    exact, exact_diagnostics = vine.predict(
        1_500,
        given=given,
        return_diagnostics=True,
        rng=np.random.default_rng(2026451),
    )
    assert exact_diagnostics["conditional_method"] == "suffix"

    monkeypatch.setattr(vine, "_suffix_sampling_state", lambda _given: None)
    approximate, mcmc_diagnostics = vine.predict(
        1_500,
        given=given,
        mcmc_steps=320,
        mcmc_burnin=160,
        return_diagnostics=True,
        rng=np.random.default_rng(2026452),
    )
    assert mcmc_diagnostics["conditional_method"] == "dag_mcmc"
    _assert_fixed_bit_exact(exact, given)
    _assert_fixed_bit_exact(approximate, given)

    free = sorted(set(range(vine.d)) - set(given))
    marginal_ks = [
        ks_2samp(exact[:, variable], approximate[:, variable]).statistic
        for variable in free
    ]
    assert max(marginal_ks) < 0.065, marginal_ks

