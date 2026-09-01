"""Regression tests for public ``VineCopula.fit`` parameter validation."""

import numpy as np
import pytest

from pyscarcopula import (
    BivariateGaussianCopula,
    IndependentCopula,
    NumericalConfig,
    VineCopula,
    api,
)
from pyscarcopula._native import static


DATA = np.random.default_rng(20260901).uniform(0.05, 0.95, size=(24, 3))


def _fixed_specs(family):
    return [
        [(family, 0) for _ in range(3 - tree - 1)]
        for tree in range(2)
    ]


def _fit(entry, vine, **kwargs):
    if entry == "api":
        return api.fit(vine, DATA, method="MLE", **kwargs)
    return vine.fit(DATA, method="MLE", **kwargs)


@pytest.mark.parametrize("entry", ["object", "api"])
def test_vine_fit_rejects_public_prepared_evaluator(entry):
    evaluator = static.prepare(BivariateGaussianCopula(), DATA[:, :2])
    vine = VineCopula.dvine(
        3,
        candidates=[BivariateGaussianCopula],
        allow_rotations=False,
    )

    with pytest.raises(TypeError, match="_prepared_evaluator"):
        _fit(
            entry,
            vine,
            copulas=_fixed_specs(BivariateGaussianCopula),
            _prepared_evaluator=evaluator,
        )

    assert vine.fit_result is None


@pytest.mark.parametrize("entry", ["object", "api"])
@pytest.mark.parametrize("short_path", ["independent", "threshold", "truncated"])
def test_vine_fit_validates_config_before_independent_short_paths(
        entry, short_path):
    vine = VineCopula.dvine(
        3,
        candidates=[BivariateGaussianCopula],
        allow_rotations=False,
    )
    kwargs = {
        "copulas": _fixed_specs(
            IndependentCopula
            if short_path == "independent"
            else BivariateGaussianCopula
        ),
        "config": {"n_threads": 2},
    }
    if short_path == "threshold":
        kwargs["threshold"] = 2.0
    elif short_path == "truncated":
        kwargs["truncation_level"] = 0

    with pytest.raises(TypeError, match="config must be NumericalConfig or None"):
        _fit(entry, vine, **kwargs)

    assert vine.fit_result is None


@pytest.mark.parametrize("entry", ["object", "api"])
@pytest.mark.parametrize("config", [None, NumericalConfig(n_threads=1)])
def test_vine_fit_accepts_supported_config_on_independent_short_path(
        entry, config):
    vine = VineCopula.dvine(
        3,
        candidates=[IndependentCopula],
        allow_rotations=False,
    )

    result = _fit(
        entry,
        vine,
        copulas=_fixed_specs(IndependentCopula),
        config=config,
    )

    fitted = vine if entry == "object" else vine.fit_result
    assert result is fitted
    assert vine.log_likelihood() == 0.0
