"""Test API consistency across copula types."""
import os
import numpy as np
import pytest
from pyscarcopula import (
    BivariateGaussianCopula,
    GumbelCopula, ClaytonCopula, FrankCopula, JoeCopula,
    IndependentCopula, CVineCopula, EquicorrGaussianCopula,
    GaussianCopula, StudentCopula, StochasticStudentCopula,
)
from pyscarcopula.api import fit, mixture_h, predict, predictive_mean, sample
from pyscarcopula.stattests import gof_test
from pyscarcopula._utils import pobs
from pyscarcopula._types import (
    MLEResult, LatentResult, GASResult, gas_params, NumericalConfig,
    LBFGSBConfig,
)
from pyscarcopula.numerical.gas_filter import gas_predict_param
from pyscarcopula.numerical._cpp_extension import CppUnsupported
from pyscarcopula.numerical.predictive_tm import (
    sample_grid_distribution, tm_state_distribution,
)
from pyscarcopula.strategy.gas import GASStrategy


class TestPublicPackageSurface:
    def test_multivariate_models_exported_from_package_root(self):
        assert EquicorrGaussianCopula.__name__ == 'EquicorrGaussianCopula'
        assert StochasticStudentCopula.__name__ == 'StochasticStudentCopula'

    def test_blas_thread_policy_env_vars_are_forced(self):
        expected = os.environ.get('PYSCA_BLAS_THREADS', '1')
        for name in (
            'OMP_NUM_THREADS',
            'MKL_NUM_THREADS',
            'OPENBLAS_NUM_THREADS',
            'NUMEXPR_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS',
            'BLIS_NUM_THREADS',
        ):
            assert os.environ[name] == expected


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
        result = fit(cop, random_u2, method='scar-tm-ou')
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
        result = fit(cop, random_u2, method='scar-tm-ou')
        r_t = predictive_mean(cop, random_u2, result)
        assert r_t.shape == (200,)
        # Should vary (not constant like MLE)
        assert np.std(r_t) > 0

    def test_top_level_api_rejects_public_posterior_cache(self, random_u2):
        cop = GumbelCopula(rotate=180)
        result = fit(cop, random_u2, method='mle')

        with pytest.raises(TypeError, match="posterior_cache"):
            predictive_mean(cop, random_u2, result, posterior_cache={})
        with pytest.raises(TypeError, match="posterior_cache"):
            mixture_h(cop, random_u2, result, posterior_cache={})

class TestGoFWithFitResult:
    """gof_test with explicit fit_result= parameter."""

    def test_mle_gof(self, random_u2):
        cop = GumbelCopula(rotate=180)
        result = fit(cop, random_u2, method='mle')
        gof = gof_test(cop, random_u2, fit_result=result, to_pobs=False)
        assert 0 <= gof.pvalue <= 1

    def test_scar_gof(self, random_u2):
        cop = GumbelCopula(rotate=180)
        result = fit(cop, random_u2, method='scar-tm-ou')
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
        result = fit(cop, random_u2, method='scar-tm-ou')
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

        latent = cop.inv_transform(r)
        np.testing.assert_allclose(latent, np.abs(r) + lower)
        assert not np.allclose(
            cop.transform(latent),
            r,
            rtol=1e-10,
            atol=1e-10,
        )

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

    def test_gaussian_transform_type_is_compatibility_only(self):
        import warnings

        from pyscarcopula.numerical import _cpp_copula, _cpp_extension

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            softplus = BivariateGaussianCopula(transform_type="softplus")
            xtanh = BivariateGaussianCopula(transform_type="xtanh")

        assert caught == []
        assert softplus._transform_type == "softplus"
        assert xtanh._transform_type == "xtanh"

        x = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
        rho = np.array([-0.8, -0.2, 0.0, 0.2, 0.8])
        np.testing.assert_array_equal(
            softplus.transform(x), xtanh.transform(x))
        np.testing.assert_array_equal(
            softplus.dtransform(x), xtanh.dtransform(x))
        np.testing.assert_array_equal(
            softplus.inv_transform(rho), xtanh.inv_transform(rho))

        module = _cpp_extension.load()
        softplus_spec = _cpp_copula.make_copula_ops_spec(module, softplus)
        xtanh_spec = _cpp_copula.make_copula_ops_spec(module, xtanh)
        assert softplus_spec.transform == module.Transform.GaussianTanh
        assert xtanh_spec.transform == module.Transform.GaussianTanh

    def test_vine_constructor_flow_keeps_gaussian_transform_fixed(self):
        from pyscarcopula.numerical import _cpp_copula, _cpp_extension
        from pyscarcopula.vine._rvine_dissmann import _make_fixed_copula

        module = _cpp_extension.load()
        for transform_type in ("softplus", "xtanh"):
            copula = _make_fixed_copula(
                (BivariateGaussianCopula, 0), transform_type)
            assert copula._transform_type == transform_type
            spec = _cpp_copula.make_copula_ops_spec(module, copula)
            assert spec.transform == module.Transform.GaussianTanh

    def test_vine_softplus(self):
        u4 = pobs(np.random.default_rng(0).standard_normal((150, 4)))
        vine = CVineCopula()
        vine.fit(u4, method='mle', transform_type='softplus')
        assert vine.fit_result.log_likelihood is not None


class TestEquicorrGaussian:
    def test_api_fit_predict_multivariate_mle(self):
        u = pobs(np.random.default_rng(42).standard_normal((120, 4)))
        cop = EquicorrGaussianCopula(d=4)

        result = fit(cop, u, method='mle')
        samples = predict(cop, u, result, 20, rng=np.random.default_rng(7))

        assert result.success
        assert samples.shape == (20, 4)
        assert np.all((samples > 0.0) & (samples < 1.0))

    def test_api_multivariate_given_is_honored(self):
        u = pobs(np.random.default_rng(43).standard_normal((120, 4)))
        cop = EquicorrGaussianCopula(d=4)
        result = fit(cop, u, method='mle')

        samples = predict(
            cop, u, result, 20, given={0: 0.5}, rng=np.random.default_rng(8))

        assert samples.shape == (20, 4)
        assert np.allclose(samples[:, 0], 0.5)
        assert np.all((samples > 0.0) & (samples < 1.0))

    def test_mle(self):
        u = pobs(np.random.default_rng(42).standard_normal((200, 4)))
        cop = EquicorrGaussianCopula(d=4)
        cop.fit(u, method='mle')
        assert 0 < cop.fit_result.copula_param < 1

    def test_scar(self):
        u = pobs(np.random.default_rng(42).standard_normal((200, 4)))
        cop = EquicorrGaussianCopula(d=4)
        cop.fit(u, method='scar-tm-ou')
        assert cop.fit_result.params.values.shape == (3,)

    def test_sample_shape(self):
        cop = EquicorrGaussianCopula(d=5)
        cop.fit(pobs(np.random.default_rng(0).standard_normal((100, 5))),
                method='mle')
        s = cop.sample_at_parameter(100, r=0.5)
        assert s.shape == (100, 5)

    def test_gof(self):
        u = pobs(np.random.default_rng(42).standard_normal((100, 3)))
        cop = EquicorrGaussianCopula(d=3)
        cop.fit(u, method='mle')
        gof = gof_test(cop, u, to_pobs=False)
        assert 0 <= gof.pvalue <= 1

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: EquicorrGaussianCopula(d=3),
            lambda: StochasticStudentCopula(d=3, R=np.eye(3)),
        ],
    )
    def test_multivariate_gas_fit_and_gof(self, factory):
        u = pobs(np.random.default_rng(49).standard_normal((35, 3)))
        cop = factory()

        result = cop.fit(u, method='gas', maxiter=10, maxfun=10)
        gof = gof_test(cop, u, to_pobs=False)

        assert result.method == 'GAS'
        assert np.isfinite(result.log_likelihood)
        assert 0 <= gof.pvalue <= 1

    def test_d1_raises(self):
        with pytest.raises(ValueError):
            EquicorrGaussianCopula(d=1)

    def test_multivariate_mle_uses_model_optimizer_config(self, monkeypatch):
        captured = {}

        class DummyResult:
            x = np.array([0.0])
            fun = 0.0
            success = True
            nfev = 1
            message = 'ok'

        def fake_minimize(
                fun, x0, jac=None, method=None, bounds=None, options=None):
            captured['options'] = options
            captured['jac'] = jac
            return DummyResult()

        monkeypatch.setattr(
            'pyscarcopula.copula.multivariate.equicorr.minimize',
            fake_minimize)

        u = pobs(np.random.default_rng(46).standard_normal((40, 4)))
        cfg = NumericalConfig(
            equicorr_optimizer=LBFGSBConfig(gtol=2e-4, maxls=31))

        result = fit(EquicorrGaussianCopula(d=4), u, method='mle',
                     config=cfg)

        assert result.success
        assert captured['options']['gtol'] == pytest.approx(2e-4)
        assert captured['options']['maxls'] == 31
        assert captured['jac'] is True
        assert result.diagnostics["optimizer_gradient"] == "analytical"
        assert result.diagnostics["transform_chain_rule"] is True

    def test_stochastic_student_mle_uses_model_optimizer_config(
            self, monkeypatch):
        captured = {}

        class DummyResult:
            x = np.array([5.0])
            fun = 0.0
            success = True
            nfev = 1
            message = 'ok'

        def fake_minimize(
                fun, x0, jac=None, method=None, bounds=None, options=None):
            captured['options'] = options
            captured['x0'] = np.asarray(x0)
            captured['bounds'] = bounds
            captured['jac'] = jac
            return DummyResult()

        monkeypatch.setattr(
            'pyscarcopula.copula.multivariate.stochastic_student.minimize',
            fake_minimize)

        u = pobs(np.random.default_rng(47).standard_normal((40, 3)))
        cfg = NumericalConfig(
            stochastic_student_optimizer=LBFGSBConfig(
                gtol=3e-4, maxiter=41))

        copula = StochasticStudentCopula(d=3)
        result = fit(copula, u, method='mle', config=cfg)

        assert result.success
        assert captured['options']['gtol'] == pytest.approx(3e-4)
        assert captured['options']['maxiter'] == 41
        np.testing.assert_array_equal(captured['x0'], [5.0])
        assert captured['bounds'][0] == (copula._df_offset, None)
        assert captured['jac'] is True

class TestMultivariateCopulaAPI:
    @pytest.mark.parametrize("cls", [GaussianCopula, StudentCopula])
    def test_dense_multivariate_mle_predicts_through_api(self, cls):
        u = pobs(np.random.default_rng(44).standard_normal((90, 3)))
        cop = cls()

        result = fit(cop, u, method='mle')
        samples = predict(cop, u, result, 12, rng=np.random.default_rng(8))

        assert result.success
        assert samples.shape == (12, 3)
        assert np.all((samples > 0.0) & (samples < 1.0))

    def test_student_fit_optimizes_df_directly_above_two(self):
        u = pobs(np.random.default_rng(45).standard_normal((100, 3)))
        cop = StudentCopula()

        cop.fit(u)

        assert np.isfinite(cop.df)
        assert 2.0 < cop.df < 1_000_000.0
        assert (
            cop.fit_result.diagnostics["optimizer_gradient"]
            == "analytical"
        )

    @pytest.mark.parametrize(
        "bad_data",
        [
            np.array([0.1, 0.2, 0.3]),
            np.empty((0, 3)),
            np.ones((5, 1)),
            np.array([[0.2, np.nan], [0.4, 0.6]]),
            np.array([[0.2, np.inf], [0.4, 0.6]]),
        ],
    )
    def test_student_fit_rejects_invalid_input_early(self, bad_data):
        with pytest.raises(ValueError):
            StudentCopula().fit(bad_data)

    def test_student_fit_gof_sample_refit_gof_roundtrip(self):
        R = np.array(
            [
                [1.0, 0.45, -0.25],
                [0.45, 1.0, 0.30],
                [-0.25, 0.30, 1.0],
            ],
            dtype=np.float64,
        )
        df = 6.5
        source = StudentCopula()
        source.shape = R
        source.df = df

        u = source.sample(2000, rng=np.random.default_rng(20260602))
        fitted = StudentCopula()
        fitted.fit(u)
        gof = gof_test(fitted, u, to_pobs=False)

        samples = fitted.sample(1800, rng=np.random.default_rng(20260603))
        refit = StudentCopula()
        refit.fit(samples)
        refit_gof = gof_test(refit, samples, to_pobs=False)

        assert np.isfinite(fitted.df)
        assert np.isfinite(refit.df)
        assert 3.0 < fitted.df < 20.0
        assert 3.0 < refit.df < 20.0
        np.testing.assert_allclose(fitted.shape, R, atol=0.08)
        np.testing.assert_allclose(refit.shape, fitted.shape, atol=0.08)
        assert 0.01 < gof.pvalue <= 1.0
        assert 0.01 < refit_gof.pvalue <= 1.0

class TestConditionalPredict:
    def test_gumbel_sample_uses_passed_rng(self):
        cop = GumbelCopula(rotate=180)
        r = np.full(64, 2.0)

        s1 = cop.sample_at_parameter(
            64, r, rng=np.random.default_rng(123))
        s2 = cop.sample_at_parameter(
            64, r, rng=np.random.default_rng(123))
        s3 = cop.sample_at_parameter(
            64, r, rng=np.random.default_rng(124))

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

    def test_bivariate_mle_sample_honors_given(self, random_u2):
        cop = GumbelCopula(rotate=180)
        result = fit(cop, random_u2, method='mle')
        samples = sample(
            cop, random_u2, result, 256, given={0: 0.37},
            rng=np.random.default_rng(19))
        assert samples.shape == (256, 2)
        assert np.all((samples > 0) & (samples < 1))
        assert np.allclose(samples[:, 0], 0.37)

    def test_gas_model_sample_honors_given(self, random_u2):
        cop = GumbelCopula(rotate=180)
        result = fit(cop, random_u2, method='gas')
        samples = sample(
            cop, random_u2, result, 32, given={0: 0.37},
            rng=np.random.default_rng(20))
        assert samples.shape == (32, 2)
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
        result = fit(cop, random_u2, method='scar-tm-ou')
        p = result.params
        z_cur, prob_cur = tm_state_distribution(
            p.kappa, p.mu, p.nu, random_u2, cop,
            K=result.K, grid_range=result.grid_range,
            horizon='current')
        z_next, prob_next = tm_state_distribution(
            p.kappa, p.mu, p.nu, random_u2, cop,
            K=result.K, grid_range=result.grid_range,
            horizon='next')
        np.testing.assert_allclose(z_cur, z_next)
        assert prob_cur.shape == prob_next.shape
        assert np.max(np.abs(prob_cur - prob_next)) > 1e-8

    def test_gas_predict_rejects_custom_bivariate_copula(self):
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

        with pytest.raises(CppUnsupported, match="C\\+\\+ bivariate GAS"):
            GASStrategy().predict(cop, u, result, 4, horizon='next')
        with pytest.raises(CppUnsupported, match="C\\+\\+ bivariate GAS"):
            gas_predict_param(omega, gamma, beta, u, cop, horizon='current')

    def test_gas_fit_forwards_optimizer_options(self, monkeypatch):
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
        cfg = NumericalConfig(
            gas_optimizer=LBFGSBConfig(
                gtol=2e-4, maxls=33, ftol=1e-11))

        result = GASStrategy(config=cfg).fit(
            cop, u, gamma0=np.array([0.0, 0.0, 0.0]))
        assert captured['options']['gtol'] == pytest.approx(2e-4)
        assert captured['options']['maxls'] == 33
        assert captured['options']['ftol'] == pytest.approx(1e-11)
        assert captured['options']['maxfun'] == 1000
        assert captured['options']['eps'] == pytest.approx(1e-5)
        assert result.score_eps == pytest.approx(cfg.gas_score_eps)

        result = GASStrategy().fit(
            cop, u, gamma0=np.array([0.0, 0.0, 0.0]), gtol=3e-5,
            ftol=1e-9, maxls=44,
            score_eps=2e-5)
        assert captured['options']['gtol'] == pytest.approx(3e-5)
        assert captured['options']['maxls'] == 44
        assert captured['options']['ftol'] == pytest.approx(1e-9)
        assert result.score_eps == pytest.approx(2e-5)

    def test_gas_fit_uses_multivariate_student_optimizer_config(self, monkeypatch):
        captured = []

        class DummyResult:
            x = np.array([0.0, 0.0, 0.0])
            fun = 0.0
            success = True
            nfev = 1
            message = 'ok'

        def fake_minimize(fun, x0, method=None, bounds=None, options=None):
            captured.append(options)
            return DummyResult()

        monkeypatch.setattr('pyscarcopula.strategy.gas.minimize',
                            fake_minimize)
        monkeypatch.setattr(
            'pyscarcopula.strategy.gas.gas_predict_param',
            lambda *args, **kwargs: 0.0)

        cfg = NumericalConfig(
            gas_optimizer=LBFGSBConfig(ftol=1e-12, maxfun=111),
            stochastic_student_gas_optimizer=LBFGSBConfig(
                ftol=1e-9, maxfun=222),
        )
        u = np.full((3, 3), 0.5)

        GASStrategy(config=cfg).fit(
            StochasticStudentCopula(d=3, R=np.eye(3)),
            u,
            gamma0=np.array([0.0, 0.0, 0.0]),
        )
        GASStrategy(config=cfg).fit(
            IndependentCopula(),
            np.full((3, 2), 0.5),
            gamma0=np.array([0.0, 0.0, 0.0]),
        )

        assert captured[0]['ftol'] == pytest.approx(1e-9)
        assert captured[0]['maxfun'] == 222
        assert captured[1]['ftol'] == pytest.approx(1e-12)
        assert captured[1]['maxfun'] == 111

    def test_gas_post_fit_uses_result_score_eps(self, monkeypatch):
        captured = {}

        def fake_gas_loglik(
                omega, gamma, beta, u, copula, scaling, score_eps):
            captured['loglik_score_eps'] = score_eps
            return 12.0

        def fake_gas_predict_param(
                omega, gamma, beta, u, copula, scaling, score_eps,
                horizon='next'):
            captured['predict_score_eps'] = score_eps
            captured['horizon'] = horizon
            return 0.25

        monkeypatch.setattr(
            'pyscarcopula.strategy.gas.gas_loglik', fake_gas_loglik)
        monkeypatch.setattr(
            'pyscarcopula.strategy.gas.gas_predict_param',
            fake_gas_predict_param)

        cop = IndependentCopula()
        u = np.array([[0.2, 0.4], [0.6, 0.8]])
        result = GASResult(
            log_likelihood=0.0,
            method='GAS',
            copula_name=cop.name,
            success=True,
            params=gas_params(0.0, 0.0, 0.0),
            scaling='unit',
            score_eps=3e-6,
        )
        strategy = GASStrategy(config=NumericalConfig(gas_score_eps=9e-4))

        assert strategy.log_likelihood(cop, u, result) == pytest.approx(12.0)
        state = strategy.predictive_state(cop, u, result, horizon='current')

        assert captured['loglik_score_eps'] == pytest.approx(3e-6)
        assert captured['predict_score_eps'] == pytest.approx(3e-6)
        assert captured['horizon'] == 'current'
        np.testing.assert_allclose(state.r, [0.25])
