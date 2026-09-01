"""Regression tests for top-level VineCopula post-fit config routing."""

from unittest.mock import Mock

import numpy as np
import pytest

from pyscarcopula import VineCopula, api
from pyscarcopula._types import NumericalConfig


@pytest.mark.parametrize(
    ("operation", "invoke"),
    [
        (
            "log_likelihood",
            lambda vine, config: api.log_likelihood(
                vine, np.empty((0, 2)), object(), config=config),
        ),
        (
            "sample",
            lambda vine, config: api.sample(
                vine, np.empty((0, 2)), object(), 2, config=config),
        ),
        (
            "predict",
            lambda vine, config: api.predict(
                vine, np.empty((0, 2)), object(), 2, config=config),
        ),
    ],
)
@pytest.mark.parametrize(
    "config",
    [NumericalConfig(n_threads=2), {"n_threads": 2}],
    ids=["numerical-config", "mapping"],
)
def test_vine_postfit_config_is_rejected_before_delegation(
    monkeypatch,
    operation,
    invoke,
    config,
):
    vine = VineCopula.cvine(d=2)
    delegate = Mock()
    monkeypatch.setattr(vine, operation, delegate)

    with pytest.raises(
        TypeError,
        match=rf"config is not supported for VineCopula {operation}",
    ):
        invoke(vine, config)

    delegate.assert_not_called()

