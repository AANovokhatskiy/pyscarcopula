"""Regression tests: fixed expected values on crypto dataset.

If these fail after a code change, something broke.
Tolerances are generous (1.0 logL) to allow for optimizer
non-determinism across platforms.
"""
import numpy as np
import pytest
from pyscarcopula import GumbelCopula, CVineCopula
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
        assert gof.pvalue > 0.05

    def test_mle_gof_rejected(self, crypto_data):
        cop = GumbelCopula(rotate=180)
        result = fit(cop, crypto_data, method='mle')
        gof = gof_test(cop, crypto_data, fit_result=result, to_pobs=False)
        assert gof.pvalue < 0.05


class TestVineRegression:
    """Known values for 6-crypto vine."""

    def test_vine_mle_logL(self, crypto_data_6d):
        vine = CVineCopula()
        vine.fit(crypto_data_6d, method='mle')
        ll = vine.log_likelihood(crypto_data_6d)
        assert 830 < ll < 920

    def test_vine_scar_logL(self, crypto_data_6d):
        vine = CVineCopula()
        vine.fit(crypto_data_6d, method='scar-tm-ou',
                 truncation_level=2, min_edge_logL=10,
                 tol=5e-2, K=150)
        ll = vine.log_likelihood(crypto_data_6d, K=150)
        assert ll > 860
