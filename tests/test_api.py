"""Test API consistency across copula types."""
import numpy as np
import pytest
from pyscarcopula import (
    GumbelCopula, ClaytonCopula, FrankCopula, JoeCopula,
    IndependentCopula, CVineCopula,
)
from pyscarcopula.copula.experimental.equicorr import EquicorrGaussianCopula
from pyscarcopula.api import fit, predict, predictive_mean, smoothed_params
from pyscarcopula.stattests import gof_test
from pyscarcopula._utils import pobs
from pyscarcopula._types import (
    MLEResult, LatentResult, GASResult, gas_params, NumericalConfig,
)
from pyscarcopula.numerical.gas_filter import gas_predict_param
from pyscarcopula.numerical.predictive_tm import (
    sample_grid_distribution, tm_state_distribution,
)
from pyscarcopula.strategy.gas import GASStrategy


class LinearScoreCopula:
    name = 'linear-score'

    def transform(self, x):
        return np.asarray(x, dtype=np.float64)

    def log_pdf(self, u1, u2, r):
        return np.asarray(r) * (np.asarray(u1) + 2.0 * np.asarray(u2))


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
        assert result.params.kappa > 0
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


class TestPredictiveMean:
    def test_mle_constant(self, random_u2):
        cop = GumbelCopula(rotate=180)
        result = fit(cop, random_u2, method='mle')
        r_t = predictive_mean(cop, random_u2, result)
        assert r_t.shape == (200,)
        assert np.all(r_t == r_t[0])

    def test_scar_varying(self, random_u2):
        cop = GumbelCopula(rotate=180)
        result = fit(cop, random_u2, method='scar-tm-ou', K=50, tol=0.5)
        r_t = predictive_mean(cop, random_u2, result)
        assert r_t.shape == (200,)
        # Should vary (not constant like MLE)
        assert np.std(r_t) > 0

    def test_predictive_mean_alias(self, random_u2):
        cop = GumbelCopula(rotate=180)
        result = fit(cop, random_u2, method='mle')
        np.testing.assert_allclose(
            predictive_mean(cop, random_u2, result),
            smoothed_params(cop, random_u2, result))


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

    def test_mle_bootstrap_gof(self, random_u2):
        cop = GumbelCopula(rotate=180)
        result = fit(cop, random_u2, method='mle')
        gof = gof_test(
            cop, random_u2, fit_result=result, to_pobs=False,
            bootstrap=True, n_bootstrap=3, rng=123)
        assert 0 <= gof.pvalue <= 1
        assert gof.n_bootstrap == 3
        assert gof.bootstrap_statistics.shape == (3,)

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

    @pytest.mark.parametrize("cls,rot", [
        (GumbelCopula, 180), (ClaytonCopula, 0),
        (FrankCopula, 0), (JoeCopula, 180),
    ])
    @pytest.mark.parametrize("transform_type", ['softplus', 'xtanh'])
    def test_dtransform_matches_numerical_derivative(
            self, cls, rot, transform_type):
        cop = cls(rotate=rot, transform_type=transform_type)
        x = np.array([-10.0, -3.0, -1.0, -0.5, 0.5, 1.0, 3.0, 10.0])
        eps = 1e-6

        ana = cop.dtransform(x)
        num = (cop.transform(x + eps) - cop.transform(x - eps)) / (2.0 * eps)

        np.testing.assert_allclose(ana, num, rtol=1e-6, atol=1e-8)

    @pytest.mark.parametrize("cls,rot,lower", [
        (GumbelCopula, 180, 1.0001),
        (ClaytonCopula, 0, 0.0001),
        (FrankCopula, 0, 0.0001),
        (JoeCopula, 180, 1.0001),
    ])
    def test_softplus_inv_transform_roundtrip(self, cls, rot, lower):
        cop = cls(rotate=rot, transform_type='softplus')
        r = lower + np.array([0.01, 0.2, 1.0, 4.0, 25.0])
        x = cop.inv_transform(r)

        np.testing.assert_allclose(
            cop.transform(x), r, rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize("cls,rot,lower", [
        (GumbelCopula, 180, 1.0001),
        (ClaytonCopula, 0, 0.0001),
        (FrankCopula, 0, 0.0001),
        (JoeCopula, 180, 1.0001),
    ])
    def test_xtanh_inv_transform_uses_modulus_approximation(
            self, cls, rot, lower):
        cop = cls(rotate=rot, transform_type='xtanh')
        r = lower + np.array([0.01, 0.2, 1.0, 4.0, 25.0])

        np.testing.assert_allclose(cop.inv_transform(r), np.abs(r) + lower)

    @pytest.mark.parametrize("cls,rot,lower", [
        (GumbelCopula, 180, 1.0001),
        (ClaytonCopula, 0, 0.0001),
        (FrankCopula, 0, 0.0001),
        (JoeCopula, 180, 1.0001),
    ])
    def test_softplus_inv_transform_near_lower_bound(self, cls, rot, lower):
        cop = cls(rotate=rot, transform_type='softplus')
        y = np.array([1e-12, 1e-10, 1e-8, 1e-4])
        x = cop.inv_transform(lower + y)

        assert x[0] < -20.0
        np.testing.assert_allclose(
            cop.transform(x) - lower, y, rtol=1e-6, atol=1e-15)

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
    def test_gumbel_sample_uses_passed_rng(self):
        cop = GumbelCopula(rotate=180)
        r = np.full(64, 2.0)

        s1 = cop.sample(64, r, rng=np.random.default_rng(123))
        s2 = cop.sample(64, r, rng=np.random.default_rng(123))
        s3 = cop.sample(64, r, rng=np.random.default_rng(124))

        np.testing.assert_allclose(s1, s2)
        assert not np.allclose(s1, s3)

    def test_sample_grid_distribution_histogram_removes_atoms(self):
        z_grid = np.array([-1.0, 0.0, 1.0])
        prob = np.array([0.0, 1.0, 0.0])

        z_grid_samples = sample_grid_distribution(
            z_grid, prob, 200, np.random.default_rng(1), mode='grid')
        z_hist_samples = sample_grid_distribution(
            z_grid, prob, 200, np.random.default_rng(1), mode='histogram')
        z_default_samples = sample_grid_distribution(
            z_grid, prob, 200, np.random.default_rng(1))

        np.testing.assert_allclose(z_grid_samples, 0.0)
        np.testing.assert_allclose(z_default_samples, z_hist_samples)
        assert np.all(z_hist_samples >= -0.5)
        assert np.all(z_hist_samples <= 0.5)
        assert len(np.unique(np.round(z_hist_samples, 12))) > 100

    def test_bivariate_mle_honors_given(self, random_u2):
        cop = GumbelCopula(rotate=180)
        result = fit(cop, random_u2, method='mle')
        samples = predict(cop, random_u2, result, 256, given={0: 0.37})
        assert samples.shape == (256, 2)
        assert np.all((samples > 0) & (samples < 1))
        assert np.allclose(samples[:, 0], 0.37)

    def test_bivariate_independent_honors_given(self, random_u2):
        cop = IndependentCopula()
        result = cop.fit(random_u2)
        samples = predict(cop, random_u2, result, 4000, given={0: 0.42})
        assert samples.shape == (4000, 2)
        assert np.allclose(samples[:, 0], 0.42)
        assert abs(np.mean(samples[:, 1]) - 0.5) < 0.03

    def test_bivariate_invalid_given_is_rejected(self, random_u2):
        cop = GumbelCopula(rotate=180)
        result = fit(cop, random_u2, method='mle')
        with pytest.raises(ValueError, match="given key"):
            predict(cop, random_u2, result, 16, given={2: 0.5})
        with pytest.raises(ValueError, match="pseudo-observation"):
            predict(cop, random_u2, result, 16, given={0: 1.0})

    def test_scar_tm_current_and_next_state_distributions_differ(self, random_u2):
        cop = GumbelCopula(rotate=180)
        result = fit(cop, random_u2, method='scar-tm-ou', K=50, tol=0.5)
        p = result.params
        z_cur, prob_cur = tm_state_distribution(
            p.kappa, p.mu, p.nu, random_u2, cop, K=50, grid_range=5.0,
            horizon='current')
        z_next, prob_next = tm_state_distribution(
            p.kappa, p.mu, p.nu, random_u2, cop, K=50, grid_range=5.0,
            horizon='next')
        np.testing.assert_allclose(z_cur, z_next)
        assert prob_cur.shape == prob_next.shape
        assert np.max(np.abs(prob_cur - prob_next)) > 1e-8

    def test_gas_predict_uses_final_observation_score(self, monkeypatch):
        cop = LinearScoreCopula()
        u = np.array([[0.2, 0.1], [0.3, 0.4]])
        omega, gamma, beta = 0.1, 0.5, 0.2
        result = GASResult(
            log_likelihood=0.0,
            method='GAS',
            copula_name=cop.name,
            success=True,
            nfev=1,
            message='ok',
            params=gas_params(omega, gamma, beta),
            scaling='unit',
        )

        g0 = omega / (1.0 - beta)
        g1 = omega + beta * g0 + gamma * (u[0, 0] + 2.0 * u[0, 1])
        expected_next = omega + beta * g1 + gamma * (u[1, 0] + 2.0 * u[1, 1])

        captured = {}

        def fake_conditional_sample(copula, n, r_values, given=None, rng=None):
            captured['r'] = r_values.copy()
            return np.zeros((n, 2))

        monkeypatch.setattr(
            'pyscarcopula.strategy.gas.conditional_sample_bivariate',
            fake_conditional_sample)

        GASStrategy().predict(cop, u, result, 4, horizon='next')
        np.testing.assert_allclose(captured['r'], expected_next)
        assert gas_predict_param(
            omega, gamma, beta, u, cop, horizon='current') == pytest.approx(g1)

    def test_gas_fit_forwards_ftol_to_optimizer(self, monkeypatch):
        captured = {}

        class DummyResult:
            x = np.array([0.0, 0.0, 0.0])
            fun = 0.0
            success = True
            nfev = 1
            message = 'ok'

        def fake_minimize(fun, x0, method=None, bounds=None, options=None):
            captured['options'] = options
            return DummyResult()

        monkeypatch.setattr('pyscarcopula.strategy.gas.minimize',
                            fake_minimize)

        cop = IndependentCopula()
        u = np.array([[0.2, 0.4], [0.6, 0.8]])
        cfg = NumericalConfig(default_ftol_gas=1e-11)

        GASStrategy(config=cfg).fit(
            cop, u, gamma0=np.array([0.0, 0.0, 0.0]))
        assert captured['options']['ftol'] == pytest.approx(1e-11)

        GASStrategy().fit(
            cop, u, gamma0=np.array([0.0, 0.0, 0.0]), ftol=1e-9)
        assert captured['options']['ftol'] == pytest.approx(1e-9)
