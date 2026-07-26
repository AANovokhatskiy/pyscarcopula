"""Stage-0 compatibility contract for the pre-unification vine API.

These characterization tests intentionally describe the current
``RVineCopula`` behavior.  They are the refactoring guardrail for introducing
``VineCopula`` without changing numerical or runtime semantics.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

import pyscarcopula.api as api_module
import pyscarcopula.stattests as statt_module
from pyscarcopula import (
    BivariateGaussianCopula,
    CVineCopula,
    IndependentCopula,
    RVineCopula,
)
from pyscarcopula.api import fit as api_fit
from pyscarcopula.api import predict as api_predict
from pyscarcopula.api import sample as api_sample
from pyscarcopula.io import load_model
from pyscarcopula.stattests import gof_test
from pyscarcopula.vine._pair_copula import PairCopula
from pyscarcopula.vine._structure import RVineMatrix


FIXTURE_DIR = Path(__file__).parents[2] / "fixtures" / "persistence"
RVINE_V2_FIXTURE = FIXTURE_DIR / "v2_rvine_current_path.json"


def _structure_data():
    """Small, explicit rank sample with stable structure-selection ties."""
    return np.array([
        [1, 1, 8, 2],
        [2, 3, 7, 1],
        [3, 2, 5, 4],
        [4, 5, 6, 3],
        [5, 4, 3, 6],
        [6, 7, 4, 5],
        [7, 6, 2, 8],
        [8, 8, 1, 7],
    ], dtype=np.float64) / 9.0


def _characterized_vine():
    return RVineCopula(
        candidates=[IndependentCopula],
        allow_rotations=False,
    ).fit(_structure_data(), method="mle")


def _normalized_trees(vine):
    return [
        [
            (tuple(sorted(conditioned)), tuple(sorted(conditioning)))
            for conditioned, conditioning in level
        ]
        for level in vine.trees
    ]


def test_rvine_auto_structure_and_public_fitted_state_contract():
    vine = _characterized_vine()

    assert vine.d == 4
    np.testing.assert_array_equal(
        vine.matrix,
        np.array([
            [3, 3, 3, 3],
            [2, 2, 2, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
        ]),
    )
    assert _normalized_trees(vine) == [
        [((0, 2), ()), ((2, 3), ()), ((0, 1), ())],
        [((0, 3), (2,)), ((1, 2), (0,))],
        [((1, 3), (0, 2))],
    ]
    assert sorted(vine.pair_copulas) == [
        (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0),
    ]
    assert all(
        isinstance(edge, PairCopula)
        for edge in vine.pair_copulas.values()
    )
    assert vine.method == "MLE"

    assert isinstance(vine.fit_result, OptimizeResult)
    assert vine.fit_result.success is True
    assert vine.fit_result.log_likelihood == 0.0
    assert vine.fit_result.method == "MLE"
    assert vine.fit_result.parameter_count == 0

    diagnostics = vine.fit_diagnostics
    assert diagnostics["target_given_vars"] == ()
    assert diagnostics["conditional_fit_supported"] is True
    assert diagnostics["edge_fits"]["edge_count"] == 6
    assert diagnostics["edge_fits"]["actual_methods"] == {"MLE": 6}
    assert diagnostics["edge_fits"]["family_counts"] == {
        "IndependentCopula": 6,
    }

    natural = vine.natural_order_matrix
    np.testing.assert_array_equal(natural, vine.matrix)
    assert natural is not vine.matrix
    natural[0, 0] = -1
    assert vine.matrix[0, 0] == 3

    converted = vine.to_rvine_matrix()
    assert isinstance(converted, RVineMatrix)
    np.testing.assert_array_equal(
        converted.matrix,
        RVineMatrix.from_natural_order(vine.matrix).matrix,
    )


def test_rvine_likelihood_seeded_sampling_summary_and_repr_contract():
    rng = np.random.default_rng(20260726)
    x0 = rng.standard_normal(160)
    x1 = 0.72 * x0 + np.sqrt(1.0 - 0.72**2) * rng.standard_normal(160)
    ranks = np.column_stack([
        np.argsort(np.argsort(x0)) + 1,
        np.argsort(np.argsort(x1)) + 1,
    ]) / 161.0
    vine = RVineCopula(
        candidates=[BivariateGaussianCopula],
        allow_rotations=False,
    ).fit(
        ranks,
        method="mle",
        copulas=[[(BivariateGaussianCopula, 0)]],
    )

    edge_sum = sum(
        edge.fit_result.log_likelihood
        for edge in vine.pair_copulas.values()
    )
    assert vine.log_likelihood() == pytest.approx(edge_sum)
    assert vine.log_likelihood(ranks) == pytest.approx(
        vine.log_likelihood(),
        rel=1e-10,
        abs=1e-10,
    )

    first = vine.sample(12, rng=np.random.default_rng(77))
    second = vine.sample(12, rng=np.random.default_rng(77))
    np.testing.assert_array_equal(first, second)

    summary = vine.summary(as_string=True)
    assert str(vine) == summary
    assert "RVineCopula" in summary
    assert "log_likelihood" in summary
    assert "\nEdges" in summary
    assert repr(vine).startswith("RVineCopula(d=2, T=160, logL=")
    assert "n_params=1)" in repr(vine)


def test_rvine_suffix_and_arbitrary_conditioning_runtime_contract():
    vine = _characterized_vine()

    suffix, suffix_diagnostics = vine.predict(
        10,
        given={0: 0.35},
        rng=np.random.default_rng(8),
        return_diagnostics=True,
    )
    assert suffix_diagnostics["conditional_method"] == "suffix"
    np.testing.assert_array_equal(suffix[:, 0], np.full(10, 0.35))

    arbitrary_kwargs = {
        "given": {0: 0.25, 3: 0.75},
        "return_diagnostics": True,
        "mcmc_steps": 3,
        "mcmc_burnin": 2,
    }
    arbitrary, arbitrary_diagnostics = vine.predict(
        10,
        rng=np.random.default_rng(9),
        **arbitrary_kwargs,
    )
    repeated, repeated_diagnostics = vine.predict(
        10,
        rng=np.random.default_rng(9),
        **arbitrary_kwargs,
    )

    assert arbitrary_diagnostics["conditional_method"] == "dag_mcmc"
    assert repeated_diagnostics["conditional_method"] == "dag_mcmc"
    np.testing.assert_array_equal(arbitrary, repeated)
    np.testing.assert_array_equal(arbitrary[:, 0], np.full(10, 0.25))
    np.testing.assert_array_equal(arbitrary[:, 3], np.full(10, 0.75))


def test_current_rvine_format_v2_golden_fixture_is_loadable():
    with RVINE_V2_FIXTURE.open(encoding="utf-8") as fixture_file:
        envelope = json.load(fixture_file)
    assert envelope["format"] == "pyscarcopula-model"
    assert envelope["format_version"] == 2
    assert envelope["include_data"] is False
    assert envelope["class"] == "pyscarcopula.vine.rvine.RVineCopula"

    vine = load_model(RVINE_V2_FIXTURE, expected_type=RVineCopula)

    assert vine.d == 2
    assert vine.method == "MLE"
    assert vine._last_u is None
    np.testing.assert_array_equal(vine.matrix, np.array([[1, 1], [0, 0]]))
    assert _normalized_trees(vine) == [[((0, 1), ())]]
    assert sorted(vine.pair_copulas) == [(0, 0)]
    assert vine.fit_result.success is True
    assert vine.fit_diagnostics["edge_fits"]["family_counts"] == {
        "IndependentCopula": 1,
    }
    assert repr(vine) == "RVineCopula(d=2, T=5, logL=0.000, n_params=0)"

    first = vine.sample(8, rng=np.random.default_rng(10))
    second = vine.sample(8, rng=np.random.default_rng(10))
    np.testing.assert_array_equal(first, second)


def test_top_level_vine_dispatch_contract(monkeypatch):
    assert api_module._vine_kind(CVineCopula()) == "cvine"
    assert api_module._vine_kind(RVineCopula()) == "rvine"

    vine = RVineCopula()
    sentinel_result = OptimizeResult(
        log_likelihood=12.5,
        method="MLE",
        success=True,
    )
    fit_calls = []

    def fake_fit(data, method="mle", **kwargs):
        fit_calls.append((data, method, kwargs))
        vine.fit_result = sentinel_result
        return vine

    monkeypatch.setattr(vine, "fit", fake_fit)
    data = _structure_data()
    result = api_fit(
        vine,
        data,
        method="mle",
        to_pobs=True,
        marker="fit",
    )
    assert result is sentinel_result
    assert fit_calls == [
        (
            data,
            "mle",
            {"to_pobs": True, "config": None, "marker": "fit"},
        ),
    ]

    sample_output = np.full((3, 4), 0.2)
    sample_calls = []

    def fake_sample(n, **kwargs):
        sample_calls.append((n, kwargs))
        return sample_output

    monkeypatch.setattr(vine, "sample", fake_sample)
    assert api_sample(
        vine,
        data,
        sentinel_result,
        n=3,
        marker="sample",
    ) is sample_output
    assert sample_calls == [(3, {"marker": "sample"})]

    predict_output = np.full((2, 4), 0.8)
    predict_calls = []

    def fake_predict(n, **kwargs):
        predict_calls.append((n, kwargs))
        return predict_output

    monkeypatch.setattr(vine, "predict", fake_predict)
    assert api_predict(
        vine,
        data,
        sentinel_result,
        n=2,
        given={1: 0.4},
        horizon="current",
        marker="predict",
    ) is predict_output
    assert predict_calls[0][0] == 2
    assert predict_calls[0][1]["u"] is data
    assert predict_calls[0][1]["marker"] == "predict"
    predict_config = predict_calls[0][1]["predict_config"]
    assert predict_config.given == {1: 0.4}
    assert predict_config.horizon == "current"

    gof_sentinel = object()
    gof_calls = []

    def fake_rvine_gof(model, observed, to_pobs, K, grid_range):
        gof_calls.append((model, observed, to_pobs, K, grid_range))
        return gof_sentinel

    monkeypatch.setattr(statt_module, "rvine_gof_test", fake_rvine_gof)
    assert gof_test(
        vine,
        data,
        to_pobs=False,
        K=17,
        grid_range=2.5,
    ) is gof_sentinel
    assert gof_calls == [(vine, data, False, 17, 2.5)]
