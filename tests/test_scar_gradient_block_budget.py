"""Storage budgets preserve the SCAR correlation-gradient calculation."""

import numpy as np
import pytest
from types import SimpleNamespace

from pyscarcopula import StochasticStudentCopula
from pyscarcopula._native import NativeError, scar_ou
from pyscarcopula.numerical._scar_ou_config import AutoTMConfig
from pyscarcopula.strategy.scar_tm import SCARTMStrategy
from pyscarcopula.strategy._base import get_strategy_for_result


@pytest.mark.parametrize("method", ["matrix", "local"])
@pytest.mark.parametrize("rows", [1, 7, 82, 83])
def test_correlation_gradient_across_budget_boundaries(method, rows):
    u = np.random.default_rng(765).uniform(0.05, 0.95, (83, 3))
    correlation = np.full((3, 3), 0.25)
    np.fill_diagonal(correlation, 1)
    copula = StochasticStudentCopula(d=3, R=correlation)
    options = dict(transition_method=method, K=24, max_K=24, adaptive=False)
    reference = scar_ou.prepare_objective(u, copula, AutoTMConfig(**options))
    blocked = scar_ou.prepare_objective(u, copula, AutoTMConfig(
        **options, corr_gradient_block_bytes=rows * 24 * 3 * 8))
    params = (2.0, 0.4, 0.9)
    direction = np.array([0.2, -0.3, 0.5])
    expected = reference.neg_loglik_with_grad_and_corr_info(*params)
    observed = blocked.neg_loglik_with_grad_and_corr_info(*params)
    for first, second in zip(observed[:3], expected[:3]):
        np.testing.assert_allclose(first, second, rtol=1e-12, atol=1e-10)
    assert observed[3] == expected[3]
    directional = blocked.neg_loglik_with_grad_and_corr_directional_info(
        *params, direction)
    np.testing.assert_allclose(directional[2], [expected[2] @ direction],
                               rtol=1e-12, atol=1e-10)
    # Repeated calls reuse their workspace without accumulating stale state.
    again = blocked.neg_loglik_with_grad_and_corr_info(*params)
    for first, second in zip(again[:3], observed[:3]):
        np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("budget", [True, 0, -1, 23, 1024.5, 2**63])
def test_strategy_rejects_invalid_correlation_block_budgets(budget):
    with pytest.raises(ValueError, match="corr_gradient_block_bytes"):
        SCARTMStrategy(corr_gradient_block_bytes=budget)


def test_strategy_routes_block_budget_without_changing_backend_options():
    strategy = SCARTMStrategy(corr_gradient_block_bytes=24 * 1024 * 1024)
    config = strategy._auto_config(kappa=2.0, n_obs=83)
    assert config.corr_gradient_block_bytes == 24 * 1024 * 1024
    assert config.transition_method == "auto"
    assert config.small_kdt == 1e-2
    assert config.r_gh == 3.0


def test_restored_and_bootstrap_strategy_preserve_budget_and_allow_override():
    from pyscarcopula.stattests import _bootstrap_strategy
    result = SimpleNamespace(method="SCAR-TM-OU", diagnostics={
        "corr_gradient_block_bytes": 24 * 1024 * 1024})
    assert get_strategy_for_result(result).corr_gradient_block_bytes == 24 * 1024 * 1024
    assert _bootstrap_strategy(result, None).corr_gradient_block_bytes == 24 * 1024 * 1024
    assert get_strategy_for_result(
        result, corr_gradient_block_bytes=32 * 1024 * 1024
    ).corr_gradient_block_bytes == 32 * 1024 * 1024
    result.diagnostics = {}
    assert get_strategy_for_result(result).corr_gradient_block_bytes == 64 * 1024 * 1024


def test_correlation_block_budget_must_hold_one_grid_row():
    u = np.random.default_rng(76).uniform(0.05, 0.95, (12, 3))
    model = StochasticStudentCopula(d=3, R=np.eye(3))
    prepared = scar_ou.prepare_objective(u, model, AutoTMConfig(
        transition_method="matrix", K=24, max_K=24, adaptive=False,
        corr_gradient_block_bytes=24))
    with pytest.raises(NativeError, match="[Ii]nvalid|INVALID"):
        prepared.neg_loglik_with_grad_and_corr_info(2.0, 0.4, 0.9)
