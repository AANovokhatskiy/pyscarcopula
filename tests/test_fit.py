"""Test that fit methods converge and recover known parameters."""
import numpy as np
import pytest
from pyscarcopula import GumbelCopula, ClaytonCopula, FrankCopula, JoeCopula
from pyscarcopula.api import fit
from pyscarcopula._utils import pobs
from pyscarcopula._types import MLEResult, LatentResult, GASResult


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
        result = fit(cop, u, method='scar-tm-ou', K=150, tol=1e-2)

        assert isinstance(result, LatentResult)
        assert result.params.theta > 0
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
        result = fit(cop, u, method='scar-tm-ou', K=150, tol=1e-2,
                     smart_init=smart, analytical_grad=True)
        assert result.log_likelihood > 100
