"""Test API consistency across copula types."""
import numpy as np
import pytest
from pyscarcopula import (
    GumbelCopula, ClaytonCopula, FrankCopula, JoeCopula,
    IndependentCopula, CVineCopula,
)
from pyscarcopula.copula.experimental.equicorr import EquicorrGaussianCopula
from pyscarcopula.api import fit, predict, smoothed_params
from pyscarcopula.stattests import gof_test
from pyscarcopula._utils import pobs
from pyscarcopula._types import MLEResult, LatentResult, GASResult
from pyscarcopula.numerical.predictive_tm import tm_state_distribution


class TestFitResultTypes:
    """fit() returns correct typed result."""

    @pytest.mark.parametrize("cls,rot", [
        (GumbelCopula, 180), (ClaytonCopula, 0),
        (FrankCopula, 0), (JoeCopula, 180),
    ])
    def test_mle_returns_mle_result(self, cls, rot, random_u2):
        cop = cls(rotate=rot)
        result = fit(cop, random_u2, method='mle')
        assert isinstance(result, MLEResult)
        assert result.success
        assert result.log_likelihood > 0

    @pytest.mark.parametrize("cls,rot", [
        (GumbelCopula, 180), (ClaytonCopula, 0),
    ])
    def test_scar_returns_latent_result(self, cls, rot, random_u2):
        cop = cls(rotate=rot)
        result = fit(cop, random_u2, method='scar-tm-ou', K=50, tol=0.5)
        assert isinstance(result, LatentResult)
        assert result.params.theta > 0
        assert result.params.nu > 0

    def test_gas_returns_gas_result(self, random_u2):
        cop = GumbelCopula(rotate=180)
        result = fit(cop, random_u2, method='gas')
        assert isinstance(result, GASResult)
        assert result.scaling == 'unit'

    def test_vine_fit_result(self):
        u4 = pobs(np.random.default_rng(0).standard_normal((200, 4)))
        vine = CVineCopula()
        vine.fit(u4, method='mle')
        assert vine.fit_result.log_likelihood is not None
        assert vine.fit_result.success

    def test_vine_logL_equals_edge_sum(self):
        u4 = pobs(np.random.default_rng(1).standard_normal((200, 4)))
        vine = CVineCopula()
        vine.fit(u4, method='mle')
        edge_sum = sum(e.fit_result.log_likelihood
                       for tree in vine.edges for e in tree)
        assert abs(vine.fit_result.log_likelihood - edge_sum) < 1e-8


class TestSmoothedParams:
    def test_mle_constant(self, random_u2):
        cop = GumbelCopula(rotate=180)
        result = fit(cop, random_u2, method='mle')
        r_t = smoothed_params(cop, random_u2, result)
        assert r_t.shape == (200,)
        assert np.all(r_t == r_t[0])

    def test_scar_varying(self, random_u2):
        cop = GumbelCopula(rotate=180)
        result = fit(cop, random_u2, method='scar-tm-ou', K=50, tol=0.5)
        r_t = smoothed_params(cop, random_u2, result)
        assert r_t.shape == (200,)
        # Should vary (not constant like MLE)
        assert np.std(r_t) > 0


class TestGoFWithFitResult:
    """gof_test with explicit fit_result= parameter."""

    def test_mle_gof(self, random_u2):
        cop = GumbelCopula(rotate=180)
        result = fit(cop, random_u2, method='mle')
        gof = gof_test(cop, random_u2, fit_result=result, to_pobs=False)
        assert 0 <= gof.pvalue <= 1

    def test_scar_gof(self, random_u2):
        cop = GumbelCopula(rotate=180)
        result = fit(cop, random_u2, method='scar-tm-ou', K=50, tol=0.5)
        gof = gof_test(cop, random_u2, fit_result=result, to_pobs=False)
        assert 0 <= gof.pvalue <= 1

    def test_gas_gof(self, random_u2):
        cop = GumbelCopula(rotate=180)
        result = fit(cop, random_u2, method='gas')
        gof = gof_test(cop, random_u2, fit_result=result, to_pobs=False)
        assert 0 <= gof.pvalue <= 1

    def test_unfitted_raises(self, random_u2):
        cop = GumbelCopula(rotate=180)
        with pytest.raises((ValueError, AttributeError)):
            gof_test(cop, random_u2, to_pobs=False)


class TestIndependentCopula:
    def test_pdf_is_one(self):
        cop = IndependentCopula()
        pdf = cop.pdf(np.array([0.1, 0.5, 0.9]),
                      np.array([0.3, 0.7, 0.2]), np.zeros(3))
        np.testing.assert_allclose(pdf, 1.0)

    def test_log_pdf_is_zero(self):
        cop = IndependentCopula()
        lp = cop.log_pdf(np.array([0.5]), np.array([0.5]), np.array([0.0]))
        np.testing.assert_allclose(lp, 0.0)

    def test_h_is_identity(self):
        cop = IndependentCopula()
        u2 = np.array([0.1, 0.3, 0.7, 0.9])
        h = cop.h(u2, np.full(4, 0.5), np.zeros(4))
        np.testing.assert_allclose(h, u2)

    def test_fit_zero_logL(self):
        cop = IndependentCopula()
        result = cop.fit(np.random.rand(100, 2))
        assert result.log_likelihood == 0.0

    def test_batch_grid(self):
        cop = IndependentCopula()
        fi, dfi = cop.pdf_and_grad_on_grid_batch(
            np.random.rand(50, 2), np.linspace(-3, 3, 20))
        np.testing.assert_allclose(fi, 1.0)
        np.testing.assert_allclose(dfi, 0.0)


class TestTransformType:
    """transform_type='softplus' works for all Archimedean copulas."""

    @pytest.mark.parametrize("cls,rot", [
        (GumbelCopula, 180), (ClaytonCopula, 0),
        (FrankCopula, 0), (JoeCopula, 180),
    ])
    def test_softplus_mle(self, cls, rot, random_u2):
        cop = cls(rotate=rot, transform_type='softplus')
        result = fit(cop, random_u2, method='mle')
        assert result.success

    @pytest.mark.parametrize("cls,rot", [
        (GumbelCopula, 180), (ClaytonCopula, 0),
        (FrankCopula, 0), (JoeCopula, 180),
    ])
    def test_softplus_scar(self, cls, rot, random_u2):
        cop = cls(rotate=rot, transform_type='softplus')
        result = fit(cop, random_u2, method='scar-tm-ou', K=50, tol=0.5)
        assert isinstance(result, LatentResult)

    def test_softplus_output_range(self):
        cop = GumbelCopula(rotate=180, transform_type='softplus')
        r = cop.transform(np.linspace(-5, 5, 100))
        assert np.all(r >= 1.0)

    def test_invalid_transform_type(self):
        with pytest.raises(ValueError):
            GumbelCopula(transform_type='invalid')

    def test_vine_softplus(self):
        u4 = pobs(np.random.default_rng(0).standard_normal((150, 4)))
        vine = CVineCopula()
        vine.fit(u4, method='mle', transform_type='softplus')
        assert vine.fit_result.log_likelihood is not None


class TestEquicorrGaussian:
    def test_mle(self):
        u = pobs(np.random.default_rng(42).standard_normal((200, 4)))
        cop = EquicorrGaussianCopula(d=4)
        cop.fit(u, method='mle')
        assert 0 < cop.fit_result.copula_param < 1

    def test_scar(self):
        u = pobs(np.random.default_rng(42).standard_normal((200, 4)))
        cop = EquicorrGaussianCopula(d=4)
        cop.fit(u, method='scar-tm-ou', K=50, tol=0.5)
        assert hasattr(cop.fit_result, 'alpha')

    def test_sample_shape(self):
        cop = EquicorrGaussianCopula(d=5)
        cop.fit(pobs(np.random.default_rng(0).standard_normal((100, 5))),
                method='mle')
        s = cop.sample(100, r=0.5)
        assert s.shape == (100, 5)

    def test_gof(self):
        u = pobs(np.random.default_rng(42).standard_normal((100, 3)))
        cop = EquicorrGaussianCopula(d=3)
        cop.fit(u, method='mle')
        gof = gof_test(cop, u, to_pobs=False)
        assert 0 <= gof.pvalue <= 1

    def test_d1_raises(self):
        with pytest.raises(ValueError):
            EquicorrGaussianCopula(d=1)


class TestConditionalPredict:
    def test_mle_given_first_coordinate_fixed(self, random_u2):
        cop = GumbelCopula(rotate=180)
        result = fit(cop, random_u2, method='mle')
        samples = predict(cop, random_u2, result, 256, given={0: 0.37})
        assert samples.shape == (256, 2)
        np.testing.assert_allclose(samples[:, 0], 0.37)
        assert np.all((samples[:, 1] > 0) & (samples[:, 1] < 1))

    def test_independent_conditional_stays_uniform(self, random_u2):
        cop = IndependentCopula()
        result = cop.fit(random_u2)
        samples = predict(cop, random_u2, result, 4000, given={0: 0.42})
        np.testing.assert_allclose(samples[:, 0], 0.42)
        assert abs(np.mean(samples[:, 1]) - 0.5) < 0.03

    def test_invalid_given_raises(self, random_u2):
        cop = GumbelCopula(rotate=180)
        result = fit(cop, random_u2, method='mle')
        with pytest.raises(ValueError):
            predict(cop, random_u2, result, 16, given={2: 0.5})
        with pytest.raises(ValueError):
            predict(cop, random_u2, result, 16, given={0: 1.0})

    def test_scar_tm_current_and_next_state_distributions_differ(self, random_u2):
        cop = GumbelCopula(rotate=180)
        result = fit(cop, random_u2, method='scar-tm-ou', K=50, tol=0.5)
        p = result.params
        z_cur, prob_cur = tm_state_distribution(
            p.theta, p.mu, p.nu, random_u2, cop, K=50, grid_range=5.0,
            horizon='current')
        z_next, prob_next = tm_state_distribution(
            p.theta, p.mu, p.nu, random_u2, cop, K=50, grid_range=5.0,
            horizon='next')
        np.testing.assert_allclose(z_cur, z_next)
        assert prob_cur.shape == prob_next.shape
        assert np.max(np.abs(prob_cur - prob_next)) > 1e-8
