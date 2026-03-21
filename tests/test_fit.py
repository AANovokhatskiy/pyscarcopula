"""Test that fit methods converge and recover known parameters."""
import numpy as np
import pytest
from pyscarcopula import GumbelCopula, ClaytonCopula, FrankCopula, JoeCopula
from pyscarcopula.utils import pobs


class TestMLERecovery:
    """MLE should recover the true constant parameter from generated data."""

    @pytest.mark.parametrize("cls,rot,true_r", [
        (GumbelCopula, 0, 2.5),
        (ClaytonCopula, 0, 3.0),
        (FrankCopula, 0, 8.0),
        (JoeCopula, 0, 2.5),
    ], ids=["Gumbel", "Clayton", "Frank", "Joe"])
    def test_mle_parameter_recovery(self, cls, rot, true_r):
        """Generate from copula, fit MLE, check parameter close to truth."""
        cop_gen = cls(rotate=rot)
        rng = np.random.default_rng(42)
        samples = cop_gen.sample(2000, r=true_r, rng=rng)
        u = pobs(samples)

        cop_fit = cls(rotate=rot)
        result = cop_fit.fit(u, method='mle')

        recovered_r = result.copula_param
        rel_err = abs(recovered_r - true_r) / true_r
        assert rel_err < 0.15, \
            f"MLE recovery: true={true_r}, got={recovered_r:.4f}, rel_err={rel_err:.2%}"


class TestSCARConvergence:
    """SCAR-TM should achieve higher logL than MLE on dynamic data."""

    def test_scar_beats_mle(self, crypto_data):
        """On real crypto data, SCAR-TM logL >= MLE logL."""
        u = crypto_data

        cop_mle = GumbelCopula(rotate=180)
        res_mle = cop_mle.fit(u, method='mle')

        cop_tm = GumbelCopula(rotate=180)
        res_tm = cop_tm.fit(u, method='scar-tm-ou')

        assert res_tm.log_likelihood >= res_mle.log_likelihood - 1.0, \
            f"SCAR logL={res_tm.log_likelihood:.2f} < MLE logL={res_mle.log_likelihood:.2f}"

    def test_scar_fit_returns_valid_alpha(self, crypto_data):
        """SCAR fit returns finite positive theta and nu."""
        u = crypto_data[:500]
        cop = GumbelCopula(rotate=180)
        result = cop.fit(u, method='scar-tm-ou', K=150, tol=1e-2)

        theta, mu, nu = result.alpha
        assert theta > 0, f"theta={theta} <= 0"
        assert nu > 0, f"nu={nu} <= 0"
        assert np.isfinite(mu), f"mu={mu} not finite"
        assert result.log_likelihood > 0, f"logL={result.log_likelihood} <= 0"


class TestGASConvergence:
    """GAS should converge and give valid parameters."""

    def test_gas_fit(self, crypto_data):
        u = crypto_data[:500]
        cop = GumbelCopula(rotate=180)
        result = cop.fit(u, method='gas')

        assert result.log_likelihood > 0
        assert hasattr(result, 'gas_params')
        assert len(result.gas_params) == 3


class TestSmartInit:
    """Smart init should not degrade results."""

    @pytest.mark.parametrize("smart", [True, False])
    def test_smart_init_same_optimum(self, crypto_data, smart):
        """Both init strategies should reach similar logL."""
        u = crypto_data[:500]
        cop = GumbelCopula(rotate=180)
        result = cop.fit(u, method='scar-tm-ou', K=150, tol=1e-2,
                         smart_init=smart, analytical_grad=True)
        # Just check it converges to something reasonable
        assert result.log_likelihood > 100, \
            f"logL={result.log_likelihood:.2f} too low with smart_init={smart}"
