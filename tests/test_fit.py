"""Test that fit methods converge and recover known parameters."""
import numpy as np
import pytest
from pyscarcopula import GumbelCopula, ClaytonCopula, FrankCopula, JoeCopula
from pyscarcopula.api import fit, predict, sample
from pyscarcopula._utils import pobs
from pyscarcopula._types import (
    MLEResult, LatentResult, GASResult, NumericalConfig, LBFGSBConfig,
)


class TestMLERecovery:
    """MLE should recover the true constant parameter from generated data."""

    @pytest.mark.parametrize("cls,rot,true_r", [
        (GumbelCopula, 0, 2.5),
        (ClaytonCopula, 0, 3.0),
        (FrankCopula, 0, 8.0),
        (JoeCopula, 0, 2.5),
    ], ids=["Gumbel", "Clayton", "Frank", "Joe"])
    def test_mle_parameter_recovery(self, cls, rot, true_r):
        cop_gen = cls(rotate=rot)
        rng = np.random.default_rng(42)
        samples = cop_gen.sample(2000, r=true_r, rng=rng)
        u = pobs(samples)

        cop = cls(rotate=rot)
        result = fit(cop, u, method='mle')

        assert isinstance(result, MLEResult)
        rel_err = abs(result.copula_param - true_r) / true_r
        assert rel_err < 0.15, \
            f"MLE recovery: true={true_r}, got={result.copula_param:.4f}"


class TestSCARConvergence:
    """SCAR-TM should achieve higher logL than MLE on dynamic data."""

    def test_scar_beats_mle(self, crypto_data):
        u = crypto_data
        cop = GumbelCopula(rotate=180)

        res_mle = fit(cop, u, method='mle')
        res_tm = fit(cop, u, method='scar-tm-ou')

        assert res_tm.log_likelihood >= res_mle.log_likelihood - 1.0

    def test_scar_fit_returns_valid_params(self, crypto_data):
        u = crypto_data[:500]
        cop = GumbelCopula(rotate=180)
        result = fit(cop, u, method='scar-tm-ou', K=150, gtol=1e-2)

        assert isinstance(result, LatentResult)
        assert result.params.kappa > 0
        assert result.params.nu > 0
        assert np.isfinite(result.params.mu)
        assert result.log_likelihood > 0


class TestGASConvergence:
    def test_gas_fit(self, crypto_data):
        u = crypto_data[:500]
        cop = GumbelCopula(rotate=180)
        result = fit(cop, u, method='gas')

        assert isinstance(result, GASResult)
        assert result.log_likelihood > 0


class TestSmartInit:
    @pytest.mark.parametrize("smart", [True, False])
    def test_smart_init_same_optimum(self, crypto_data, smart):
        u = crypto_data[:500]
        cop = GumbelCopula(rotate=180)
        result = fit(cop, u, method='scar-tm-ou', K=150, gtol=1e-2,
                     smart_init=smart, analytical_grad=True)
        assert result.log_likelihood > 100

    def test_use_gas_returns_gas_initial_point(self, monkeypatch):
        from pyscarcopula.strategy import initial_point

        u = pobs(np.random.default_rng(0).standard_normal((20, 2)))
        cop = GumbelCopula(rotate=180)
        gas_alpha = np.array([2.0, 3.0, 4.0])

        monkeypatch.setattr(
            initial_point,
            '_gas_initial_point',
            lambda u_arg, cop_arg, verbose=False: gas_alpha,
        )

        alpha0, info = initial_point.smart_initial_point(
            u, cop, use_gas=True)

        np.testing.assert_allclose(alpha0, gas_alpha)
        assert info['method'] == 'gas'


class TestMCStrategies:
    @pytest.mark.parametrize("method", ['scar-p-ou', 'scar-m-ou'])
    def test_mc_fit_supports_smart_init_false(self, method):
        u = pobs(np.random.default_rng(0).standard_normal((35, 2)))
        cop = GumbelCopula(rotate=180)
        cfg = NumericalConfig(
            scar_optimizer=LBFGSBConfig(maxfun=3, maxiter=3),
            default_n_tr=20,
        )

        result = fit(
            cop, u, method=method, config=cfg, seed=1,
            smart_init=False, M_iterations=1)

        assert isinstance(result, LatentResult)
        assert result.params.kappa > 0
        assert result.params.nu > 0

    def test_scar_p_supports_sample_and_predict(self):
        u = pobs(np.random.default_rng(1).standard_normal((35, 2)))
        cop = GumbelCopula(rotate=180)
        cfg = NumericalConfig(
            scar_optimizer=LBFGSBConfig(maxfun=3, maxiter=3),
            default_n_tr=20,
        )
        result = fit(
            cop, u, method='scar-p-ou', config=cfg, seed=1,
            smart_init=False)

        sim = sample(cop, u, result, 8, rng=np.random.default_rng(2))
        pred = predict(cop, u, result, 8, rng=np.random.default_rng(3))

        assert sim.shape == (8, 2)
        assert pred.shape == (8, 2)
        assert np.all((sim > 0.0) & (sim < 1.0))
        assert np.all((pred > 0.0) & (pred < 1.0))

    def test_scar_m_uses_config_default_m_iterations(self):
        u = pobs(np.random.default_rng(2).standard_normal((35, 2)))
        cop = GumbelCopula(rotate=180)
        cfg = NumericalConfig(
            scar_optimizer=LBFGSBConfig(maxfun=3, maxiter=3),
            default_n_tr=20,
            default_M_iterations=1,
        )

        result = fit(cop, u, method='scar-m-ou', config=cfg, seed=1,
                     smart_init=False)

        assert result.M_iterations == 1
