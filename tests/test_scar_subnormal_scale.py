"""Regression checks for subnormal OU backward scaling."""

import numpy as np
import pytest

from pyscarcopula import GumbelCopula
from pyscarcopula._native import scar_ou
from pyscarcopula.numerical._scar_ou_config import AutoTMConfig


@pytest.mark.parametrize("observation,mu", [
    (0.01, 117.0),
    (0.05, 176.0),
    (0.1, 232.0),
])
@pytest.mark.parametrize("transition", ["local", "auto"])
def test_subnormal_backward_scale_does_not_overflow(observation, mu, transition):
    u = np.array([[0.5, 0.5], [observation, 1 - observation], [0.5, 0.5]])
    config = AutoTMConfig(transition_method=transition, adaptive=False, K=300)
    value, gradient, _ = scar_ou.neg_loglik_with_grad_info(
        .001, mu, .001, u, GumbelCopula(), config)
    assert np.isfinite(value)
    assert np.isfinite(gradient).all()
