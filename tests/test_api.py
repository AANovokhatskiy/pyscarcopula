"""Test API consistency across copula types."""
import numpy as np
import pytest
from pyscarcopula import (
    GumbelCopula, ClaytonCopula, FrankCopula, JoeCopula,
    IndependentCopula, CVineCopula,
)
from pyscarcopula.stattests import gof_test
from pyscarcopula.utils import pobs


class TestFitResultInterface:
    """All copulas should return fit_result with the same attributes."""

    REQUIRED_ATTRS = ['log_likelihood', 'method', 'name', 'success']

    @pytest.mark.parametrize("cls,rot", [
        (GumbelCopula, 180), (ClaytonCopula, 0),
        (FrankCopula, 0), (JoeCopula, 0),
    ])
    def test_bivariate_fit_result_attrs(self, cls, rot, random_u2):
        cop = cls(rotate=rot)
        cop.fit(random_u2, method='mle')
        for attr in self.REQUIRED_ATTRS:
            assert hasattr(cop.fit_result, attr), \
                f"{cls.__name__} fit_result missing .{attr}"

    def test_vine_fit_result_attrs(self, random_u2):
        # Need 4d for vine
        u4 = pobs(np.random.default_rng(0).standard_normal((200, 4)))
        vine = CVineCopula()
        vine.fit(u4, method='mle')
        for attr in self.REQUIRED_ATTRS:
            assert hasattr(vine.fit_result, attr), \
                f"CVineCopula fit_result missing .{attr}"

    def test_vine_fit_result_log_likelihood(self):
        """vine.fit_result.log_likelihood == sum of edge logLs."""
        u4 = pobs(np.random.default_rng(1).standard_normal((200, 4)))
        vine = CVineCopula()
        vine.fit(u4, method='mle')
        edge_sum = sum(e.fit_result.log_likelihood
                       for tree in vine.edges for e in tree)
        assert abs(vine.fit_result.log_likelihood - edge_sum) < 1e-8


class TestPredictNoMutation:
    """predict() should not change fit state."""

    def test_predict_preserves_state(self, random_u2):
        cop = GumbelCopula(rotate=180)
        cop.fit(random_u2, method='scar-tm-ou', K=50, tol=0.5)
        ll_before = cop.fit_result.log_likelihood
        _ = cop.predict(100)
        ll_after = cop.fit_result.log_likelihood
        assert ll_before == ll_after


class TestUnfittedErrors:
    """Operations on unfitted copula should raise errors."""

    def test_gof_unfitted_raises(self, random_u2):
        cop = GumbelCopula(rotate=180)
        with pytest.raises((ValueError, AttributeError)):
            gof_test(cop, random_u2, to_pobs=False)


class TestIndependentCopula:
    """IndependentCopula special properties."""

    def test_pdf_is_one(self):
        cop = IndependentCopula()
        u1 = np.array([0.1, 0.5, 0.9])
        u2 = np.array([0.3, 0.7, 0.2])
        pdf = cop.pdf(u1, u2, np.zeros(3))
        np.testing.assert_allclose(pdf, 1.0)

    def test_log_pdf_is_zero(self):
        cop = IndependentCopula()
        lp = cop.log_pdf(np.array([0.5]), np.array([0.5]), np.array([0.0]))
        np.testing.assert_allclose(lp, 0.0)

    def test_h_is_identity(self):
        cop = IndependentCopula()
        u2 = np.array([0.1, 0.3, 0.7, 0.9])
        u1 = np.array([0.5, 0.5, 0.5, 0.5])
        h = cop.h(u2, u1, np.zeros(4))
        np.testing.assert_allclose(h, u2)

    def test_fit_zero_logL(self):
        cop = IndependentCopula()
        result = cop.fit(np.random.rand(100, 2))
        assert result.log_likelihood == 0.0
        assert result.method == 'MLE'

    def test_batch_grid(self):
        cop = IndependentCopula()
        u = np.random.rand(50, 2)
        x = np.linspace(-3, 3, 20)
        fi, dfi = cop.pdf_and_grad_on_grid_batch(u, x)
        np.testing.assert_allclose(fi, 1.0)
        np.testing.assert_allclose(dfi, 0.0)
