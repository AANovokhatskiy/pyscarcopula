"""Test that fit methods converge and recover known parameters."""
import importlib
import json

import numpy as np
import pytest
from pyscarcopula import GumbelCopula, ClaytonCopula, FrankCopula, JoeCopula
from pyscarcopula.api import fit, predict, sample
from pyscarcopula._utils import pobs
from pyscarcopula._types import (
    MLEResult, LatentResult, GASResult, NumericalConfig, LBFGSBConfig,
)
from pyscarcopula.numerical import _cpp_scar_ou
from pyscarcopula.strategy import scar_tm
from pyscarcopula.strategy.scar_tm import SCARTMStrategy


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
        samples = cop_gen.sample_at_parameter(2000, r=true_r, rng=rng)
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
        result = fit(cop, u, method='scar-tm-ou')

        assert isinstance(result, LatentResult)
        assert result.params.kappa > 0
        assert result.params.nu > 0
        assert np.isfinite(result.params.mu)
        assert result.log_likelihood > 0


class TestSCARNativeExecution:
    @pytest.mark.parametrize("backend", ["auto", "python", "cpp"])
    def test_backend_argument_is_rejected(self, backend):
        with pytest.raises(TypeError, match="backend selection was removed"):
            SCARTMStrategy(backend=backend)

    def test_xtanh_copula_uses_native_objective(self):
        u = pobs(np.random.default_rng(3).standard_normal((18, 2)))
        cop = GumbelCopula(rotate=180, transform_type='xtanh')
        alpha = np.array([0.8, 0.2, 1.0], dtype=np.float64)

        strategy = SCARTMStrategy(
            transition_method='matrix',
            K=24,
            adaptive=False,
            max_K=None,
            analytical_grad=False,
        )
        assert strategy._uses_cpp(cop)
        assert np.isfinite(strategy.objective(cop, u, alpha))

    def test_native_fit_records_transition_fallback_diagnostics(
            self, monkeypatch):
        u = pobs(np.random.default_rng(6).standard_normal((10, 2)))
        cop = ClaytonCopula(rotate=0)
        calls = []

        def fake_objective(kappa, mu, nu, u_arg, copula_arg, config):
            calls.append(float(kappa))
            value = (
                (float(kappa) - 0.8) ** 2
                + float(mu) ** 2
                + (float(nu) - 1.0) ** 2
                + 1.0
            )
            grad = np.array(
                [2.0 * (float(kappa) - 0.8), 2.0 * float(mu),
                 2.0 * (float(nu) - 1.0)],
                dtype=np.float64,
            )
            info = {
                "backend": "local",
                "transition_method": "auto",
                "fallback_chain": ["spectral", "matrix"],
                "fallback_from": "matrix",
                "matrix_fallback_reason": "unknown",
                "engine": "cpp",
                "kappa_dt": float(kappa) / (len(u_arg) - 1),
                "n_obs": len(u_arg),
            }
            return value, grad, info

        monkeypatch.setattr(
            _cpp_scar_ou, "neg_loglik_with_grad_info", fake_objective)
        monkeypatch.setattr(_cpp_scar_ou, "supported", lambda copula: True)

        result = SCARTMStrategy(
            smart_init=False,
            analytical_grad=True,
            K=20,
            adaptive=False,
            max_K=None,
        ).fit(
            cop,
            u,
            alpha0=np.array([1.0, 0.1, 1.2]),
            maxiter=2,
            maxfun=12,
        )

        assert calls
        diag = result.diagnostics
        assert diag["selected_engine"] == "cpp"
        assert diag["objective_evaluations"] == len(calls)
        assert diag["cpp_evaluations"] == len(calls)
        assert diag["fallback_spectral_to_matrix"] == len(calls)
        assert diag["fallback_matrix_to_local"] == len(calls)
        assert diag["matrix_fallback_unknown"] == len(calls)
        assert diag["local_evaluations"] == len(calls)
        assert diag["last_backend"] == "local"

    def test_cpp_result_info_uses_exact_fallback_reason(self):
        info = _cpp_scar_ou._result_info(
            {
                "backend": 1,
                "status": 0,
                "fallback_from": 2,
                "fallback_chain": [0, 2],
                "matrix_fallback_reason": 2,
            },
            "auto",
            1.0,
            11,
            _cpp_scar_ou.AutoTMConfig(transition_method="auto"),
        )

        assert info["backend"] == "local"
        assert info["fallback_from"] == "matrix"
        assert info["fallback_chain"] == ["spectral", "matrix"]
        assert info["matrix_fallback_reason"] == "capped"


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
        result = fit(cop, u, method='scar-tm-ou',
                     smart_init=smart, analytical_grad=True)
        assert result.log_likelihood > 100

    def test_use_gas_returns_gas_initial_point(self, monkeypatch):
        from pyscarcopula.strategy import initial_point

        u = pobs(np.random.default_rng(0).standard_normal((20, 2)))
        cop = GumbelCopula(rotate=180)
        gas_alpha = np.array([2.0, 3.0, 4.0])

        monkeypatch.setattr(
            initial_point,
            '_gas_initial_point',
            lambda u_arg, cop_arg, verbose=False: gas_alpha,
        )

        alpha0, info = initial_point.smart_initial_point(
            u, cop, use_gas=True)

        np.testing.assert_allclose(alpha0, gas_alpha)
        assert info['method'] == 'gas'

    def test_gas_initial_point_uses_lightweight_loglik_then_one_filter(
            self, monkeypatch):
        from pyscarcopula.strategy import initial_point
        gas_module = importlib.import_module(
            'pyscarcopula.numerical.gas_filter')

        u = np.full((8, 2), 0.5)
        cop = GumbelCopula()
        calls = []

        monkeypatch.setattr(initial_point, '_mle_mu', lambda *args: 0.2)

        def fake_loglik(omega, gamma, beta, u_arg, copula, scaling):
            calls.append(('loglik', omega, gamma, beta))
            return gamma

        def fake_filter(omega, gamma, beta, u_arg, copula, scaling):
            calls.append(('filter', omega, gamma, beta))
            return np.linspace(0.0, 0.7, len(u_arg)), np.zeros(len(u_arg)), 0.0

        monkeypatch.setattr(gas_module, 'gas_loglik', fake_loglik)
        monkeypatch.setattr(gas_module, 'gas_filter', fake_filter)

        result = initial_point._gas_initial_point(u, cop)

        assert result.shape == (3,)
        assert sum(call[0] == 'loglik' for call in calls) == 20
        assert sum(call[0] == 'filter' for call in calls) == 1

    def test_stochastic_student_starts_near_static_df_mle(self, monkeypatch):
        from pyscarcopula.copula.multivariate import StochasticStudentCopula
        from pyscarcopula.strategy import initial_point

        class StudentSubclass(StochasticStudentCopula):
            pass

        u = pobs(np.random.default_rng(1).standard_normal((20, 2)))
        copula = StudentSubclass(d=2, R=np.eye(2))
        df0 = 5.0
        inverse_mu0 = float(copula.inv_transform([df0])[0])

        monkeypatch.setattr(
            initial_point,
            '_mle_info',
            lambda copula_arg, u_arg: (df0, inverse_mu0, 12.0),
        )

        alpha0, info = initial_point.smart_initial_point(u, copula)

        expected_kappa = -np.log(0.96) * (len(u) - 1)
        np.testing.assert_allclose(
            alpha0,
            [expected_kappa, inverse_mu0, 0.1],
        )
        np.testing.assert_allclose(copula.transform(alpha0[1]), [df0])
        assert info['method'] == 'stochastic_student_mle'
        assert info['df_mle'] == df0
        assert info['mu_mle'] == inverse_mu0

    def test_smart_initial_point_preserves_primary_failure_reason(
            self, monkeypatch):
        from pyscarcopula.strategy import initial_point

        u = pobs(np.random.default_rng(3).standard_normal((20, 2)))
        copula = GumbelCopula()
        fallback = np.array([1.5, 0.2, 0.8])

        def fail_primary(*args, **kwargs):
            raise ValueError("strength initialization failed")

        monkeypatch.setattr(
            initial_point, '_strength_aware_initial_point', fail_primary)
        monkeypatch.setattr(
            initial_point,
            '_heuristic_initial_point',
            lambda *args, **kwargs: fallback,
        )

        alpha0, info = initial_point.smart_initial_point(u, copula)
        diagnostics = info['initialization']

        np.testing.assert_allclose(alpha0, fallback)
        assert diagnostics['requested_method'] == 'strength_aware'
        assert diagnostics['selected_method'] == 'heuristic'
        assert diagnostics['attempts'][0] == {
            'method': 'strength_aware',
            'success': False,
            'error_type': 'ValueError',
            'error_message': 'strength initialization failed',
        }
        assert diagnostics['attempts'][1]['success'] is True
        json.dumps(diagnostics)

    def test_scar_tm_resolver_records_final_mle_fallback(
            self, monkeypatch):
        from pyscarcopula.strategy.mle import MLEStrategy

        u = pobs(np.random.default_rng(4).standard_normal((20, 2)))
        copula = GumbelCopula()

        def fail_smart(*args, **kwargs):
            raise RuntimeError("smart initialization failed")

        class Result:
            copula_param = 2.0

        monkeypatch.setattr(scar_tm, 'smart_initial_point', fail_smart)
        monkeypatch.setattr(
            MLEStrategy, 'fit', lambda self, cop, data: Result())

        alpha0, diagnostics = scar_tm._resolve_initial_point(
            copula, u, NumericalConfig(), True, False, None)

        assert diagnostics['selected_method'] == 'mle_default'
        assert diagnostics['attempts'][0]['error_type'] == 'RuntimeError'
        assert diagnostics['attempts'][0][
            'error_message'] == 'smart initialization failed'
        assert diagnostics['attempts'][1] == {
            'method': 'mle_default',
            'success': True,
        }
        np.testing.assert_allclose(diagnostics['alpha0'], alpha0)

    def test_stochastic_student_subclass_rejects_scar_mixture_h(self):
        from pyscarcopula.copula.multivariate import StochasticStudentCopula

        class StudentSubclass(StochasticStudentCopula):
            pass

        copula = StudentSubclass(d=2, R=np.eye(2))
        strategy = SCARTMStrategy()

        with pytest.raises(NotImplementedError, match="StochasticStudent"):
            strategy.mixture_h(
                copula,
                np.full((3, 2), 0.5),
                LatentResult(
                    log_likelihood=0.0,
                    method="SCAR-TM-OU",
                    copula_name=copula.name,
                    success=True,
                ),
            )


class TestMCStrategies:
    @pytest.mark.parametrize("method", ['scar-p-ou', 'scar-m-ou'])
    def test_mc_fit_supports_smart_init_false(self, method):
        u = pobs(np.random.default_rng(0).standard_normal((35, 2)))
        cop = GumbelCopula(rotate=180)
        cfg = NumericalConfig(
            scar_optimizer=LBFGSBConfig(maxfun=3, maxiter=3),
            default_n_tr=20,
        )

        result = fit(
            cop, u, method=method, config=cfg, seed=1,
            smart_init=False, M_iterations=1)

        assert isinstance(result, LatentResult)
        assert result.params.kappa > 0
        assert result.params.nu > 0
        initialization = result.diagnostics['initialization']
        assert initialization['requested_method'] == 'mle_default'
        assert initialization['selected_method'] == 'mle_default'
        assert initialization['attempts'] == [{
            'method': 'mle_default',
            'success': True,
        }]

    def test_scar_p_supports_sample_and_predict(self):
        u = pobs(np.random.default_rng(1).standard_normal((35, 2)))
        cop = GumbelCopula(rotate=180)
        cfg = NumericalConfig(
            scar_optimizer=LBFGSBConfig(maxfun=3, maxiter=3),
            default_n_tr=20,
        )
        result = fit(
            cop, u, method='scar-p-ou', config=cfg, seed=1,
            smart_init=False)

        sim = sample(cop, u, result, 8, rng=np.random.default_rng(2))
        pred = predict(cop, u, result, 8, rng=np.random.default_rng(3))

        assert sim.shape == (8, 2)
        assert pred.shape == (8, 2)
        assert np.all((sim > 0.0) & (sim < 1.0))
        assert np.all((pred > 0.0) & (pred < 1.0))

    def test_scar_m_uses_config_default_m_iterations(self):
        u = pobs(np.random.default_rng(2).standard_normal((35, 2)))
        cop = GumbelCopula(rotate=180)
        cfg = NumericalConfig(
            scar_optimizer=LBFGSBConfig(maxfun=3, maxiter=3),
            default_n_tr=20,
            default_M_iterations=1,
        )

        result = fit(cop, u, method='scar-m-ou', config=cfg, seed=1,
                     smart_init=False)

        assert result.M_iterations == 1
