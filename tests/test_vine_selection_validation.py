"""Regression tests for public vine family-selection validation."""

import numpy as np
import pytest

from pyscarcopula import IndependentCopula, NumericalConfig
from pyscarcopula.vine import select_best_copula


U1 = np.array([0.15, 0.35, 0.55, 0.75], dtype=np.float64)
U2 = np.array([0.25, 0.45, 0.65, 0.85], dtype=np.float64)


@pytest.mark.parametrize("criterion", ["AIC", "unknown", ""])
def test_independent_only_selection_rejects_invalid_criterion(criterion):
    with pytest.raises(ValueError, match="criterion must be"):
        select_best_copula(
            U1,
            U2,
            candidates=[IndependentCopula],
            criterion=criterion,
        )


@pytest.mark.parametrize("config", [{}, {"n_threads": 2}, object()])
def test_independent_only_selection_rejects_invalid_config(config):
    with pytest.raises(TypeError, match="config must be NumericalConfig or None"):
        select_best_copula(
            U1,
            U2,
            candidates=[IndependentCopula],
            config=config,
        )


@pytest.mark.parametrize("config", [None, NumericalConfig(n_threads=1)])
@pytest.mark.parametrize("criterion", ["aic", "bic", "loglik"])
def test_independent_only_selection_accepts_supported_settings(
        config, criterion):
    selected = select_best_copula(
        U1,
        U2,
        candidates=[IndependentCopula],
        criterion=criterion,
        config=config,
    )

    assert isinstance(selected.copula, IndependentCopula)
    assert selected.result.log_likelihood == 0.0
