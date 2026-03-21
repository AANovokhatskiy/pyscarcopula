"""Regression tests: fixed expected values on crypto dataset.

If these fail after a code change, something broke.
Tolerances are generous (1.0 logL) to allow for optimizer
non-determinism across platforms.
"""
import numpy as np
import pytest
from pyscarcopula import GumbelCopula, CVineCopula
from pyscarcopula.stattests import gof_test


class TestBivariateRegression:
    """Known values for bivariate BTC-ETH."""

    def test_mle_logL(self, crypto_data):
        cop = GumbelCopula(rotate=180)
        result = cop.fit(crypto_data, method='mle')
        # Expected: ~955.6
        assert abs(result.log_likelihood - 955.6) < 2.0, \
            f"MLE logL={result.log_likelihood:.2f}, expected ~955.6"

    def test_mle_param(self, crypto_data):
        cop = GumbelCopula(rotate=180)
        result = cop.fit(crypto_data, method='mle')
        # Expected: theta ~2.83
        assert abs(result.copula_param - 2.83) < 0.1, \
            f"MLE param={result.copula_param:.4f}, expected ~2.83"

    def test_scar_tm_logL(self, crypto_data):
        cop = GumbelCopula(rotate=180)
        result = cop.fit(crypto_data, method='scar-tm-ou')
        # Expected: ~1042
        assert result.log_likelihood > 1030, \
            f"SCAR-TM logL={result.log_likelihood:.2f}, expected >1030"

    def test_scar_tm_gof(self, crypto_data):
        cop = GumbelCopula(rotate=180)
        cop.fit(crypto_data, method='scar-tm-ou')
        gof = gof_test(cop, crypto_data, to_pobs=False)
        # Should pass GoF (p > 0.05)
        assert gof.pvalue > 0.05, \
            f"SCAR-TM GoF p={gof.pvalue:.4f}, expected >0.05"

    def test_mle_gof_rejected(self, crypto_data):
        cop = GumbelCopula(rotate=180)
        cop.fit(crypto_data, method='mle')
        gof = gof_test(cop, crypto_data, to_pobs=False)
        # MLE should be rejected (p < 0.05) on this dynamic data
        assert gof.pvalue < 0.05, \
            f"MLE GoF p={gof.pvalue:.4f}, expected <0.05"


class TestVineRegression:
    """Known values for 6-crypto vine."""

    def test_vine_mle_logL(self, crypto_data_6d):
        vine = CVineCopula()
        vine.fit(crypto_data_6d, method='mle')
        ll = vine.log_likelihood(crypto_data_6d)
        # Expected: ~869
        assert ll > 830 and ll < 920, \
            f"Vine MLE logL={ll:.1f}, expected ~869"

    def test_vine_scar_logL(self, crypto_data_6d):
        vine = CVineCopula()
        vine.fit(crypto_data_6d, method='scar-tm-ou',
                 truncation_level=2, min_edge_logL=10,
                 tol=5e-2, K=150)
        ll = vine.log_likelihood(crypto_data_6d, K=150)
        # Should be better than MLE (~869)
        assert ll > 860, \
            f"Vine SCAR-TM logL={ll:.1f}, expected >860"
