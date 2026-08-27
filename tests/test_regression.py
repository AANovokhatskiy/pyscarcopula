"""Regression tests: fixed expected values on crypto dataset.

If these fail after a code change, something broke.
Tolerances are generous (1.0 logL) to allow for optimizer
non-determinism across platforms.
"""
from pyscarcopula import GumbelCopula
from pyscarcopula.api import fit
from pyscarcopula.stattests import gof_test


class TestBivariateRegression:
    """Known values for bivariate BTC-ETH."""

    def test_mle_logL(self, crypto_data):
        cop = GumbelCopula(rotate=180)
        result = fit(cop, crypto_data, method='mle')
        assert abs(result.log_likelihood - 955.6) < 2.0

    def test_mle_param(self, crypto_data):
        cop = GumbelCopula(rotate=180)
        result = fit(cop, crypto_data, method='mle')
        assert abs(result.copula_param - 2.83) < 0.1

    def test_scar_tm_logL(self, crypto_data):
        cop = GumbelCopula(rotate=180)
        result = fit(cop, crypto_data, method='scar-tm-ou')
        assert result.log_likelihood > 1030

    def test_scar_tm_gof(self, crypto_data):
        cop = GumbelCopula(rotate=180)
        result = fit(cop, crypto_data, method='scar-tm-ou')
        gof = gof_test(cop, crypto_data, fit_result=result, to_pobs=False)
        assert gof.statistic < 0.2

    def test_mle_gof_rejected(self, crypto_data):
        cop = GumbelCopula(rotate=180)
        result = fit(cop, crypto_data, method='mle')
        gof = gof_test(cop, crypto_data, fit_result=result, to_pobs=False)
        assert gof.statistic > 0.5
