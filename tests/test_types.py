"""Tests for _types and _utils — the foundation of the new architecture."""
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from pyscarcopula import PredictConfig as PublicPredictConfig
from pyscarcopula._types import (
    NumericalConfig,
    LBFGSBConfig,
    LatentProcessParams, ou_params, jacobi_params, gas_params,
    MLEResult, MultivariateMLEResult, LatentResult, GASResult,
    IndependentResult,
    PredictConfig, PredictiveState,
)
from pyscarcopula._utils import broadcast, pobs, clip_unit


def test_predict_config_exported_from_package_root():
    assert PublicPredictConfig is PredictConfig


# ══════════════════════════════════════════════════════════════════
# NumericalConfig
# ══════════════════════════════════════════════════════════════════

class TestNumericalConfig:
    def test_defaults(self):
        cfg = NumericalConfig()
        assert cfg.default_K == 300
        assert cfg.default_grid_range == 5.0
        assert cfg.default_pts_per_sigma == 4
        assert cfg.mle_optimizer.gtol == 1e-3
        assert cfg.mle_optimizer.maxls == 20
        assert cfg.gas_optimizer.gtol == 1e-3
        assert cfg.gas_optimizer.ftol == 1e-9
        assert cfg.gas_optimizer.maxfun == 4000
        assert cfg.gas_optimizer.maxiter == 1000
        assert cfg.gas_optimizer.maxls == 100
        assert cfg.gas_optimizer.eps == 1e-5
        assert cfg.scar_optimizer.gtol == 1e-3
        assert cfg.scar_optimizer.maxfun == 300
        assert cfg.scar_optimizer.maxiter == 100
        assert cfg.scar_optimizer.maxls == 20
        assert cfg.scar_optimizer.eps == 1e-4
        assert cfg.bivariate_scar_optimizer == cfg.scar_optimizer
        assert cfg.bivariate_log_scar_optimizer.maxfun == 300
        assert cfg.bivariate_log_scar_optimizer.maxiter == 1000
        assert cfg.bivariate_log_scar_optimizer.maxls == 200
        assert cfg.equicorr_optimizer.gtol == 1e-4
        assert cfg.stochastic_student_optimizer.gtol == 1e-4
        assert cfg.stochastic_student_gas_optimizer.ftol == 1e-9
        assert cfg.stochastic_student_gas_optimizer.maxls == 50
        assert cfg.stochastic_student_scar_optimizer.maxfun == 300
        assert cfg.stochastic_student_scar_optimizer.maxiter == 1000
        assert cfg.stochastic_student_scar_optimizer.maxls == 200
        assert cfg.gas_score_eps == 1e-4
        assert cfg.gas_gamma_bound == 20.0
        assert cfg.gas_beta_bound == 0.999
        assert not hasattr(cfg, "eps_clip")
        assert not hasattr(cfg, "eps_log")

    def test_override(self):
        cfg = NumericalConfig(
            default_K=500,
            mle_optimizer=LBFGSBConfig(gtol=1e-5),
            gas_optimizer=LBFGSBConfig(ftol=1e-10, maxfun=250),
            scar_optimizer=LBFGSBConfig(maxls=50),
            bivariate_scar_optimizer=LBFGSBConfig(maxiter=700),
            bivariate_log_scar_optimizer=LBFGSBConfig(maxfun=500),
            equicorr_optimizer=LBFGSBConfig(maxls=35),
            stochastic_student_optimizer=LBFGSBConfig(maxiter=40),
            stochastic_student_gas_optimizer=LBFGSBConfig(ftol=1e-8),
            stochastic_student_scar_optimizer=LBFGSBConfig(maxfun=600),
            gas_score_eps=1e-6,
            gas_gamma_bound=12.0,
            gas_beta_bound=0.95,
        )
        assert cfg.default_K == 500
        assert cfg.mle_optimizer.gtol == 1e-5
        assert cfg.mle_optimizer.maxls == 20
        assert cfg.gas_optimizer.ftol == 1e-10
        assert cfg.gas_optimizer.maxfun == 250
        assert cfg.gas_optimizer.maxls == 100
        assert cfg.scar_optimizer.maxls == 50
        assert cfg.scar_optimizer.maxfun == 300
        assert cfg.bivariate_scar_optimizer.maxiter == 700
        assert cfg.bivariate_scar_optimizer.maxls == 20
        assert cfg.bivariate_log_scar_optimizer.maxfun == 500
        assert cfg.bivariate_log_scar_optimizer.maxiter == 1000
        assert cfg.bivariate_log_scar_optimizer.maxls == 200
        assert cfg.equicorr_optimizer.gtol == 1e-4
        assert cfg.equicorr_optimizer.maxls == 35
        assert cfg.stochastic_student_optimizer.gtol == 1e-4
        assert cfg.stochastic_student_optimizer.maxiter == 40
        assert cfg.stochastic_student_gas_optimizer.ftol == 1e-8
        assert cfg.stochastic_student_gas_optimizer.maxfun == 1000
        assert cfg.stochastic_student_scar_optimizer.maxfun == 600
        assert cfg.stochastic_student_scar_optimizer.maxiter == 1000
        assert cfg.stochastic_student_scar_optimizer.maxls == 200
        assert cfg.gas_score_eps == 1e-6
        assert cfg.gas_gamma_bound == 12.0
        assert cfg.gas_beta_bound == 0.95
        assert cfg.default_grid_range == 5.0  # unchanged

    def test_frozen(self):
        cfg = NumericalConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.default_K = 999


class TestPredictConfig:
    def test_defaults_validate(self):
        cfg = PredictConfig().validated()
        assert cfg.given is None
        assert cfg.horizon == 'next'
        assert cfg.predictive_r_mode is None
        assert cfg.dynamic_conditioning == 'ignore'
        assert cfg.mcmc_steps is None
        assert cfg.mcmc_burnin is None

    def test_normalizes_and_copies_given(self):
        given = {0: 0.4}
        cfg = PredictConfig(
            given=given,
            horizon='CURRENT',
            predictive_r_mode='HISTOGRAM',
            dynamic_conditioning='GIVEN_ONLY',
            mcmc_steps=np.int64(12),
            mcmc_burnin=np.int64(3),
        ).validated()
        assert cfg.horizon == 'current'
        assert cfg.predictive_r_mode == 'histogram'
        assert cfg.dynamic_conditioning == 'given_only'
        assert cfg.given == {0: 0.4}
        assert cfg.given is not given
        assert cfg.mcmc_steps == 12
        assert cfg.mcmc_burnin == 3

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"horizon": "bad"},
            {"predictive_r_mode": "bad"},
            {"dynamic_conditioning": "bad"},
            {"mcmc_steps": -1},
            {"mcmc_burnin": -1},
        ],
    )
    def test_rejects_bad_options(self, kwargs):
        with pytest.raises(ValueError):
            PredictConfig(**kwargs).validated()

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"mcmc_steps": 1.2},
            {"mcmc_burnin": True},
        ],
    )
    def test_rejects_bad_mcmc_option_types(self, kwargs):
        with pytest.raises(TypeError):
            PredictConfig(**kwargs).validated()

    def test_replace_validates(self):
        cfg = PredictConfig().replace(horizon='current')
        assert cfg.horizon == 'current'


class TestPredictiveState:
    def test_point_state(self):
        state = PredictiveState(
            method='MLE',
            horizon='next',
            kind='point',
            r=np.array([0.25]),
        )
        assert state.method == 'MLE'
        assert state.r[0] == pytest.approx(0.25)


# ══════════════════════════════════════════════════════════════════
# LatentProcessParams — the key abstraction for variable param count
# ══════════════════════════════════════════════════════════════════

class TestLatentProcessParams:
    @pytest.mark.parametrize('field', ['values', 'bounds_lower', 'bounds_upper'])
    @pytest.mark.parametrize('representation', ['complex', 'object', 'list', 'nested'])
    @pytest.mark.parametrize('imaginary', [0., .2])
    def test_rejects_complex_values_and_bounds(self, field, representation, imaginary):
        data = np.array([1. + imaginary * 1j, 2.], dtype=np.complex128)
        if representation == 'object':
            data = data.astype(object)
        elif representation == 'list':
            data = data.tolist()
        elif representation == 'nested':
            nested = np.empty(data.size, dtype=object)
            for index, value in enumerate(data):
                nested[index] = np.array(value)
            data = nested
        kwargs = dict(process_type='generic', names=('a', 'b'), values=[1., 2.])
        kwargs[field] = data
        with pytest.raises(TypeError, match=field + '.*real'):
            LatentProcessParams(**kwargs)

    @pytest.mark.parametrize('factory,values,key', [
        (ou_params, [2., .3, .4], 'kappa'),
        (gas_params, [.2, .03, .6], 'omega'),
        (jacobi_params, [2., .4, .3], 'kappa'),
    ])
    @pytest.mark.parametrize('route', ['factory', 'replace'])
    @pytest.mark.parametrize('value', [1. + .2j, np.complex64(1.), np.complex128(1. + .2j)])
    def test_factories_and_replace_reject_complex(self, factory, values, key, route, value):
        original = factory(*values)
        with pytest.raises(TypeError, match='values.*real'):
            if route == 'factory':
                factory(value, *values[1:])
            else:
                original.replace(**{key: value})
        np.testing.assert_array_equal(original.values, values)

    @pytest.mark.parametrize('key', ['gamma_bound', 'beta_bound'])
    @pytest.mark.parametrize('value', [np.complex128(.8 + .2j), np.complex128(.8)])
    @pytest.mark.parametrize('nested', [False, True])
    def test_gas_bounds_reject_complex_before_native_conversion(self, key, value, nested):
        if nested:
            outer = np.empty((), dtype=object)
            outer[()] = np.array(value)
            value = outer
        with pytest.raises(TypeError, match=key + '.*real'):
            gas_params(.2, .03, .6, **{key: value})

    def test_nested_real_values_and_bounds_remain_supported(self):
        values = np.empty(2, dtype=object)
        values[0], values[1] = np.array(.3), np.array(.4)
        bounds = np.empty((), dtype=object)
        bounds[()] = np.array(.8)
        params = LatentProcessParams('generic', ('a', 'b'), values, values, values)
        for actual in (params.values, params.bounds_lower, params.bounds_upper):
            np.testing.assert_array_equal(actual, [.3, .4])
        gas = gas_params(.2, .03, .6, gamma_bound=bounds, beta_bound=bounds)
        np.testing.assert_array_equal(gas.bounds_lower[1:], [-.8, -.8])
        np.testing.assert_array_equal(gas.bounds_upper[1:], [.8, .8])

    def test_real_strided_storage_and_infinite_bounds_keep_existing_contract(self):
        storage = np.array([2., 9., .3, 9., .4, 9.])
        values = storage[::2]
        values.setflags(write=False)
        bounds = np.array([-np.inf, 0., 0.])
        params = LatentProcessParams(
            'ou', ('kappa', 'mu', 'nu'), values, bounds, [np.inf] * 3)
        assert params.values is values
        assert params.bounds_lower is bounds
        assert not params.values.flags.writeable
        replaced = params.replace(mu=.5)
        np.testing.assert_array_equal(params.values, [2., .3, .4])
        np.testing.assert_array_equal(replaced.values, [2., .5, .4])
        assert replaced.bounds_lower is bounds
        np.testing.assert_array_equal(replaced.bounds_upper, [np.inf] * 3)

    @pytest.mark.parametrize('params', [
        ou_params(2., .3, .4), gas_params(.2, .03, .6),
        jacobi_params(2., .4, .3),
    ])
    @pytest.mark.parametrize('unknown', ['parameter_typo', 'maxiter'])
    def test_replace_rejects_unknown_names_without_changing_original(self, params, unknown):
        original = params.values.copy()
        with pytest.raises(TypeError, match=unknown):
            params.replace(**{params.names[0]: 7., unknown: 13.})
        np.testing.assert_array_equal(params.values, original)

    def test_ou_params(self):
        p = ou_params(kappa=49.97, mu=2.42, nu=10.65)
        assert p.process_type == 'ou'
        assert p.n_params == 3
        assert p.kappa == pytest.approx(49.97)
        assert p.mu == pytest.approx(2.42)
        assert p.nu == pytest.approx(10.65)

    def test_gas_params(self):
        p = gas_params(omega=0.07, gamma=0.33, beta=0.97)
        assert p.process_type == 'gas'
        assert p.n_params == 3
        assert p.omega == pytest.approx(0.07)
        assert p.gamma == pytest.approx(0.33)
        assert p.beta == pytest.approx(0.97)

    def test_jacobi_params(self):
        p = jacobi_params(kappa=2.5, m=0.35, xi=0.2)
        assert p.process_type == 'jacobi'
        assert p.n_params == 3
        assert p.kappa == pytest.approx(2.5)
        assert p.m == pytest.approx(0.35)
        assert p.xi == pytest.approx(0.2)
        np.testing.assert_array_equal(
            p.bounds_lower, np.array([0.001, 1e-6, 0.001]))
        np.testing.assert_array_equal(
            p.bounds_upper, np.array([np.inf, 1.0 - 1e-6, np.inf]))

    def test_generic_4_params(self):
        """Future Lévy process with 4 parameters."""
        p = LatentProcessParams(
            process_type='levy',
            names=('alpha', 'beta', 'mu', 'sigma'),
            values=np.array([1.5, 0.3, 0.0, 1.0]),
        )
        assert p.n_params == 4
        assert p.alpha == pytest.approx(1.5)
        assert p.sigma == pytest.approx(1.0)

    def test_generic_2_params(self):
        """Future fBm process with 2 parameters."""
        p = LatentProcessParams(
            process_type='fbm',
            names=('H', 'sigma'),
            values=np.array([0.7, 1.0]),
        )
        assert p.n_params == 2
        assert p.H == pytest.approx(0.7)

    def test_named_access_error(self):
        p = ou_params(1.0, 0.0, 1.0)
        with pytest.raises(AttributeError, match="no parameter 'xyz'"):
            _ = p.xyz

    def test_to_dict(self):
        p = ou_params(1.0, 2.0, 3.0)
        d = p.to_dict()
        assert d == {'kappa': 1.0, 'mu': 2.0, 'nu': 3.0}

    def test_replace(self):
        p = ou_params(1.0, 2.0, 3.0)
        p2 = p.replace(mu=5.0)
        assert p2.mu == pytest.approx(5.0)
        assert p2.kappa == pytest.approx(1.0)
        assert p.mu == pytest.approx(2.0)  # original unchanged (frozen)

    def test_values_array(self):
        p = ou_params(1.0, 2.0, 3.0)
        assert isinstance(p.values, np.ndarray)
        assert p.values.dtype == np.float64
        np.testing.assert_array_equal(p.values, [1.0, 2.0, 3.0])

    def test_bounds(self):
        p = ou_params(1.0, 2.0, 3.0)
        assert p.bounds_lower is not None
        assert p.bounds_lower[0] == pytest.approx(0.001)  # kappa > 0
        assert p.bounds_upper[1] == np.inf  # mu unbounded

    def test_mismatched_names_values(self):
        with pytest.raises(ValueError, match="same length"):
            LatentProcessParams(
                process_type='bad',
                names=('a', 'b'),
                values=np.array([1.0, 2.0, 3.0]),
            )

    def test_frozen(self):
        p = ou_params(1.0, 2.0, 3.0)
        with pytest.raises(FrozenInstanceError):
            p.process_type = 'xxx'

    def test_repr(self):
        p = ou_params(49.97, 2.42, 10.65)
        r = repr(p)
        assert 'ou' in r
        assert 'kappa=49.9700' in r


# ══════════════════════════════════════════════════════════════════
# FitResult types
# ══════════════════════════════════════════════════════════════════

class TestFitResults:
    def test_mle_result(self):
        r = MLEResult(
            log_likelihood=100.0,
            method='MLE',
            copula_name='Gumbel',
            success=True,
            copula_param=2.83,
        )
        assert r.n_params == 1
        assert r.copula_param == pytest.approx(2.83)
        assert r.log_likelihood == pytest.approx(100.0)

    def test_mle_result_supports_additional_static_parameters(self):
        r = MLEResult(
            log_likelihood=100.0,
            method='MLE',
            copula_name='Stochastic Student',
            success=True,
            copula_param=5.0,
            parameter_count=np.int64(4),
            diagnostics={'corr_mode': 'cholesky'},
        )

        assert r.n_params == 4
        assert r.parameter_count == 4
        assert r.diagnostics['corr_mode'] == 'cholesky'
        assert 'n_params: 4' in repr(r)

        with pytest.raises(TypeError):
            MLEResult(
                log_likelihood=0.0,
                method='MLE',
                copula_name='invalid',
                success=False,
                parameter_count=1.5,
            )
        with pytest.raises(ValueError):
            MLEResult(
                log_likelihood=0.0,
                method='MLE',
                copula_name='invalid',
                success=False,
                parameter_count=0,
            )

    def test_multivariate_mle_result_exposes_information_criteria(self):
        correlation = np.eye(3)
        result = MultivariateMLEResult(
            log_likelihood=12.5,
            method="MLE",
            copula_name="Gaussian copula",
            success=True,
            parameter_count=3,
            n_observations=100,
            model_parameters={"correlation_matrix": correlation},
            correlation_matrix=correlation,
        )

        assert isinstance(result, MLEResult)
        assert result.n_params == 3
        assert result.aic == pytest.approx(2 * 3 - 2 * 12.5)
        assert result.bic == pytest.approx(np.log(100) * 3 - 2 * 12.5)
        assert result.correlation_matrix is not correlation
        np.testing.assert_allclose(result.correlation_matrix, correlation)
        correlation[0, 1] = 0.5
        assert result.correlation_matrix[0, 1] == 0.0

        with pytest.raises(ValueError):
            MultivariateMLEResult(
                log_likelihood=0.0,
                method="MLE",
                copula_name="invalid",
                success=False,
                n_observations=0,
            )

    def test_multivariate_mle_result_allows_zero_parameters(self):
        result = MultivariateMLEResult(
            log_likelihood=7.5,
            method="MLE",
            copula_name="Gaussian copula",
            success=True,
            parameter_count=0,
            n_observations=40,
            correlation_matrix=np.eye(2),
        )

        assert result.n_params == 0
        assert result.aic == pytest.approx(-15.0)
        assert result.bic == pytest.approx(-15.0)

    def test_latent_result_ou(self):
        r = LatentResult(
            log_likelihood=1042.47,
            method='SCAR-TM-OU',
            copula_name='Gumbel',
            success=True,
            params=ou_params(49.97, 2.42, 10.65),
            K=300,
            grid_range=5.0,
            pts_per_sigma=4,
        )
        assert r.n_params == 3
        assert r.params.kappa == pytest.approx(49.97)
        assert r.K == 300
        assert r.pts_per_sigma == 4
        np.testing.assert_allclose(
            r.params.values, [49.97, 2.42, 10.65])

    def test_latent_result_supports_additional_static_parameters(self):
        r = LatentResult(
            log_likelihood=1042.47,
            method='SCAR-TM-OU',
            copula_name='Stochastic Student',
            success=True,
            params=ou_params(49.97, 2.42, 10.65),
            parameter_count=np.int64(6),
        )

        assert r.n_params == 6
        assert r.parameter_count == 6
        assert r.params.n_params == 3
        assert 'n_params: 6' in repr(r)

        with pytest.raises(TypeError):
            LatentResult(
                log_likelihood=0.0,
                method='SCAR-TM-OU',
                copula_name='invalid',
                success=False,
                parameter_count=3.5,
            )
        with pytest.raises(ValueError):
            LatentResult(
                log_likelihood=0.0,
                method='SCAR-TM-OU',
                copula_name='invalid',
                success=False,
                parameter_count=2,
            )

    def test_latent_result_future_levy(self):
        """A future Lévy process with 4 params uses the same LatentResult."""
        levy_p = LatentProcessParams(
            process_type='levy',
            names=('alpha', 'beta', 'mu', 'sigma'),
            values=np.array([1.5, 0.3, 0.0, 1.0]),
        )
        r = LatentResult(
            log_likelihood=500.0,
            method='SCAR-TM-LEVY',
            copula_name='Frank',
            success=True,
            params=levy_p,
            K=500,
        )
        assert r.n_params == 4
        assert r.params.alpha == pytest.approx(1.5)
        assert len(r.params.values) == 4

    def test_gas_result(self):
        r = GASResult(
            log_likelihood=1040.0,
            method='GAS',
            copula_name='Gumbel',
            success=True,
            params=gas_params(0.0696, 0.331, 0.9677),
            scaling='unit',
        )
        assert r.n_params == 3
        assert r.omega == pytest.approx(0.0696)
        assert r.gamma == pytest.approx(0.331)
        assert r.beta == pytest.approx(0.9677)
        assert r.scaling == 'unit'
        assert r.diagnostics == {}

    def test_gas_result_fisher_scaling(self):
        r = GASResult(
            log_likelihood=1040.0,
            method='GAS',
            copula_name='Frank',
            success=True,
            params=gas_params(0.16, 5.0, 0.984),
            scaling='fisher',
        )
        assert r.scaling == 'fisher'

    def test_independent_result(self):
        r = IndependentResult(
            log_likelihood=0.0,
            method='MLE',
            copula_name='Independent',
            success=True,
        )
        assert r.n_params == 0
        assert r.log_likelihood == 0.0

    def test_frozen(self):
        r = MLEResult(
            log_likelihood=100.0,
            method='MLE',
            copula_name='Gumbel',
            success=True,
        )
        with pytest.raises(FrozenInstanceError):
            r.log_likelihood = 999.0


# ══════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════

class TestBroadcast:
    def test_scalar_broadcast(self):
        u1, u2, r = broadcast(0.5, 0.3, np.array([1.0, 2.0, 3.0]))
        assert len(u1) == 3
        assert len(u2) == 3
        assert all(u1 == 0.5)

    def test_same_length(self):
        u1, u2, r = broadcast([0.1, 0.2], [0.3, 0.4], [1.0, 2.0])
        assert len(u1) == 2

    def test_single_r(self):
        u1, u2, r = broadcast([0.1, 0.2, 0.3], [0.4, 0.5, 0.6], 1.5)
        assert len(r) == 3
        assert all(r == 1.5)


class TestPobs:
    def test_basic(self):
        data = np.array([[3.0, 1.0], [1.0, 3.0], [2.0, 2.0]])
        u = pobs(data)
        assert u.shape == (3, 2)
        # Ranks / (n+1): ranks are [3,1,2] for col 0
        np.testing.assert_allclose(u[:, 0], [3/4, 1/4, 2/4])

    def test_range(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((100, 4))
        u = pobs(data)
        assert np.all(u > 0)
        assert np.all(u < 1)


class TestClipUnit:
    def test_basic(self):
        x = np.array([-0.1, 0.0, 0.5, 1.0, 1.1])
        c = clip_unit(x)
        assert c[0] == pytest.approx(1e-10)
        assert c[-1] == pytest.approx(1 - 1e-10)
        assert c[2] == pytest.approx(0.5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
