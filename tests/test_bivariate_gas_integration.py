"""Integration checks for the hybrid Python/C++ bivariate GAS model."""

import numpy as np
import pytest

from pyscarcopula.api import fit
from pyscarcopula.copula.elliptical import BivariateGaussianCopula
from pyscarcopula.copula.gumbel import GumbelCopula
from pyscarcopula.numerical import _cpp_gas
from pyscarcopula.stattests import gof_test
from pyscarcopula.strategy.gas import GASStrategy
from pyscarcopula._types import GASResult, gas_params


def _native_parameter_path(params, copula, scaling, observations):
    state = _cpp_gas.initial_state(*params, copula, scaling)
    g_t = state.g
    r_t = state.parameter
    path = []
    for row in observations:
        path.append(r_t)
        update = _cpp_gas.update_one(
            *params, g_t, row, copula, scaling, 1e-4)
        g_t = update.g_next
        r_t = update.r_next
    return np.asarray(path)


@pytest.mark.parametrize("scaling", ["unit", "fisher"])
def test_python_driven_sampling_matches_native_recursion(
        monkeypatch, scaling):
    copula = GumbelCopula(rotate=180)
    params = (0.03, 0.02, 0.65)
    observations = np.array(
        [
            [0.21, 0.72],
            [0.64, 0.35],
            [0.43, 0.58],
            [0.82, 0.19],
            [0.37, 0.66],
        ],
        dtype=np.float64,
    )
    expected_r = _native_parameter_path(
        params, copula, scaling, observations)
    seen_r = []
    cursor = iter(observations)

    def deterministic_sample(n, r, rng=None):
        assert n == 1
        seen_r.append(float(np.asarray(r)[0]))
        return np.asarray([next(cursor)], dtype=np.float64)

    monkeypatch.setattr(
        copula, "sample_at_parameter", deterministic_sample)
    result = GASResult(
        log_likelihood=0.0,
        method="GAS",
        copula_name=copula.name,
        success=True,
        params=gas_params(*params),
        scaling=scaling,
        score_eps=1e-4,
    )

    samples = GASStrategy(scaling=scaling).sample(
        copula,
        None,
        result,
        len(observations),
        rng=np.random.default_rng(123),
    )

    np.testing.assert_allclose(samples, observations, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        seen_r, expected_r, rtol=2e-8, atol=2e-9)


def test_gas_fit_gof_and_bootstrap_use_compiled_kernels(monkeypatch):
    source = BivariateGaussianCopula()
    u = source.sample_at_parameter(
        45, np.full(45, 0.55), rng=np.random.default_rng(20260612))
    copula = BivariateGaussianCopula()
    result = fit(
        copula,
        u,
        method="gas",
        gamma0=np.array([0.0, 0.05, 0.8]),
        maxiter=40,
        maxfun=80,
    )

    p = result.params
    assert np.isfinite(result.log_likelihood)
    assert result.log_likelihood == pytest.approx(
        _cpp_gas.log_likelihood(
            p.omega,
            p.gamma,
            p.beta,
            u,
            copula,
            result.scaling,
            result.score_eps,
        ),
        rel=1e-12,
        abs=1e-12,
    )

    calls = {"update": 0, "h": 0}
    original_update = _cpp_gas.update_one
    original_h = _cpp_gas.h_path

    def counted_update(*args, **kwargs):
        calls["update"] += 1
        return original_update(*args, **kwargs)

    def counted_h(*args, **kwargs):
        calls["h"] += 1
        return original_h(*args, **kwargs)

    monkeypatch.setattr(_cpp_gas, "update_one", counted_update)
    monkeypatch.setattr(_cpp_gas, "h_path", counted_h)

    gof = gof_test(
        copula,
        u,
        fit_result=result,
        to_pobs=False,
        bootstrap=True,
        n_bootstrap=2,
        bootstrap_refit=False,
        rng=20260612,
    )

    assert np.isfinite(gof.statistic)
    assert 0.0 <= gof.pvalue <= 1.0
    assert gof.n_bootstrap == 2
    assert np.all(np.isfinite(gof.bootstrap_statistics))
    assert calls["h"] == 3
    assert calls["update"] == 2 * (len(u) - 1)
