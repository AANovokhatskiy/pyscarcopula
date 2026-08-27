"""Permanent top-level API contracts for native-backed vines."""

import numpy as np
import pytest

import pyscarcopula.api as api
import pyscarcopula.stattests as stattests
from pyscarcopula import (
    IndependentCopula,
    PredictConfig,
    VineCopula,
)
from pyscarcopula.contrib.risk_metrics import _get_copula_constructor
from pyscarcopula.vine import cvine_structure, dvine_structure


def _data(rows=70, dimension=4, seed=20260726):
    return np.random.default_rng(seed).uniform(
        0.01, 0.99, size=(rows, dimension))


@pytest.mark.parametrize(
    "structure_factory",
    (cvine_structure, dvine_structure),
)
def test_fixed_vines_work_through_top_level_fit_sample_predict_and_loglik(
        structure_factory):
    data = _data()
    vine = VineCopula(
        structure=structure_factory(4),
        candidates=[IndependentCopula],
    )

    result = api.fit(vine, data, method="mle")

    assert result is vine.fit_result
    assert api.log_likelihood(vine, data, result) == pytest.approx(
        vine.log_likelihood(data))

    direct_sample = vine.sample(
        12,
        rng=np.random.default_rng(11),
        batch_rows=3,
        memory_budget_bytes=1_000_000,
    )
    api_sample = api.sample(
        vine,
        data,
        result,
        12,
        rng=np.random.default_rng(11),
        batch_rows=3,
        memory_budget_bytes=1_000_000,
    )
    np.testing.assert_array_equal(api_sample, direct_sample)

    predict_config = PredictConfig(
        horizon="current",
        return_diagnostics=True,
    )
    direct_predict, direct_diagnostics = vine.predict(
        10,
        u=data,
        rng=np.random.default_rng(12),
        predict_config=predict_config,
    )
    api_predict, api_diagnostics = api.predict(
        vine,
        data,
        result,
        10,
        rng=np.random.default_rng(12),
        predict_config=predict_config,
    )
    np.testing.assert_array_equal(api_predict, direct_predict)
    assert api_diagnostics["conditional_method"] == (
        direct_diagnostics["conditional_method"])
    assert "timings_ms" in api_diagnostics


def test_generic_dispatch_is_type_based_not_structure_label():
    for vine, expected_type in (
        (VineCopula.cvine(4), "cvine"),
        (VineCopula.dvine(4), "dvine"),
        (VineCopula.rvine(), "rvine"),
    ):
        assert api._is_generic_vine(vine)
        assert vine.vine_type == expected_type


def test_unsupported_vine_kwargs_are_rejected_instead_of_dropped():
    data = _data(dimension=3)
    generic = VineCopula.dvine(
        3, candidates=[IndependentCopula]).fit(data)

    with pytest.raises(TypeError, match="unexpected_keyword"):
        api.sample(
            generic, data, generic.fit_result, 4,
            unexpected_keyword=True)
    with pytest.raises(TypeError, match="unexpected_keyword"):
        api.predict(
            generic, data, generic.fit_result, 4,
            unexpected_keyword=True)


@pytest.mark.parametrize(
    ("factory", "expected_type"),
    (
        (VineCopula.cvine, "cvine"),
        (VineCopula.dvine, "dvine"),
        (lambda d, **kwargs: VineCopula.rvine(**kwargs), "rvine"),
    ),
)
def test_gof_forwards_generic_vine_type(
        monkeypatch, factory, expected_type):
    sentinel = object()
    calls = []

    def fake_gof(
            model, data, to_pobs, K, grid_range, *, vine_type):
        calls.append(
            (model, data, to_pobs, K, grid_range, vine_type))
        return sentinel

    monkeypatch.setattr(stattests, "rvine_gof_test", fake_gof)
    data = _data(dimension=3)
    vine = factory(
        3, candidates=[IndependentCopula]).fit(data)

    assert stattests.gof_test(
        vine, data, to_pobs=False, K=17, grid_range=2.5) is sentinel
    assert calls == [
        (vine, data, False, 17, 2.5, expected_type),
    ]


@pytest.mark.parametrize(
    ("factory", "expected_type", "expected_source"),
    (
        (lambda: VineCopula.cvine(
            4, candidates=[IndependentCopula]), "cvine", "fixed"),
        (lambda: VineCopula.dvine(
            4, candidates=[IndependentCopula]), "dvine", "fixed"),
        (lambda: VineCopula.rvine(
            candidates=[IndependentCopula]), "rvine", "auto"),
    ),
)
def test_risk_worker_constructor_preserves_vine_mode(
        factory, expected_type, expected_source):
    source = factory()
    model_class, model_kwargs = _get_copula_constructor(source)
    rebuilt = model_class(**model_kwargs)

    assert rebuilt.vine_type == expected_type
    assert rebuilt.structure_source == expected_source
    if expected_source == "fixed":
        assert rebuilt.structure == source.structure
    else:
        assert rebuilt.structure is None
