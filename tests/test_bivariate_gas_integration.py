"""Integration checks for the hybrid Python/C++ bivariate GAS model."""

import numpy as np
import pytest

from pyscarcopula.api import fit
from pyscarcopula.copula.clayton import ClaytonCopula
from pyscarcopula.copula.elliptical import BivariateGaussianCopula
from pyscarcopula.copula.frank import FrankCopula
from pyscarcopula.copula.gumbel import GumbelCopula
from pyscarcopula.copula.joe import JoeCopula
from pyscarcopula._native import gas as _cpp_gas
from pyscarcopula.stattests import gof_test
from pyscarcopula.strategy.gas import GASStrategy
from pyscarcopula._types import GASResult, gas_params


def _stepwise_sample_from_uniforms(params, copula, scaling, draws):
    state = _cpp_gas.initial_state(*params, copula, scaling)
    g_t = state.g
    r_t = state.parameter
    samples = []
    for index, draw in enumerate(draws):
        row = copula._native_adapter().sample_from_uniforms(
            copula,
            draw.reshape(1, 2),
            np.array([r_t]),
        )[0]
        samples.append(row)
        if index + 1 < len(draws):
            update = _cpp_gas.update_one(
                *params, g_t, row, copula, scaling, 1e-4)
            g_t = update.g_next
            r_t = update.r_next
    return np.asarray(samples)


@pytest.mark.parametrize(
    "factory",
    [ClaytonCopula, FrankCopula, GumbelCopula, JoeCopula],
)
@pytest.mark.parametrize("transform_type", ["exp", "logistic"])
def test_native_gas_accepts_new_archimedean_transforms(
        factory, transform_type):
    copula = factory(transform_type=transform_type)
    params = (0.03, 0.02, 0.65)
    state = _cpp_gas.initial_state(*params, copula, "unit")
    update = _cpp_gas.update_one(
        *params, state.g, np.array([0.31, 0.72]),
        copula, "unit", 1e-4)

    assert np.isfinite(state.g)
    assert np.isfinite(state.parameter)
    assert np.isfinite(update.g_next)
    assert np.isfinite(update.r_next)


@pytest.mark.parametrize("scaling", ["unit", "fisher"])
def test_fused_sampling_matches_stepwise_native_recursion(
        monkeypatch, scaling):
    copula = GumbelCopula(rotate=180)
    params = (0.03, 0.02, 0.65)
    draws = np.array(
        [
            [0.21, 0.72],
            [0.64, 0.35],
            [0.43, 0.58],
            [0.82, 0.19],
            [0.37, 0.66],
        ],
        dtype=np.float64,
    )
    expected = _stepwise_sample_from_uniforms(
        params, copula, scaling, draws)
    calls = []
    original_sample = _cpp_gas.sample_bivariate

    def counted_sample(*args, **kwargs):
        calls.append("sample")
        return original_sample(*args, **kwargs)

    class FixedRng:
        def uniform(self, low, high, size):
            assert (low, high, size) == (0.0, 1.0, draws.shape)
            return draws.copy()

    monkeypatch.setattr(_cpp_gas, "sample_bivariate", counted_sample)
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
        len(draws),
        rng=FixedRng(),
    )

    np.testing.assert_allclose(samples, expected, rtol=0.0, atol=0.0)
    assert calls == ["sample"]


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
    assert 0 < result.nfev <= 80
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

    calls = {"sample": 0, "update": 0, "h": 0}
    original_sample = _cpp_gas.sample_bivariate
    original_update = _cpp_gas.update_one
    original_h = _cpp_gas.h_path

    def counted_sample(*args, **kwargs):
        calls["sample"] += 1
        return original_sample(*args, **kwargs)

    def counted_update(*args, **kwargs):
        calls["update"] += 1
        return original_update(*args, **kwargs)

    def counted_h(*args, **kwargs):
        calls["h"] += 1
        return original_h(*args, **kwargs)

    monkeypatch.setattr(_cpp_gas, "sample_bivariate", counted_sample)
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
    assert calls["sample"] == 2
    assert calls["update"] == 0
