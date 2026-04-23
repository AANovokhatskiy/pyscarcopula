import numpy as np
import pytest
from scipy.stats import cramervonmises, kendalltau, kstest, norm, spearmanr

from pyscarcopula import (
    BivariateGaussianCopula,
    CVineCopula,
    FrankCopula,
    GumbelCopula,
    IndependentCopula,
    RVineCopula,
)
from pyscarcopula._types import (
    GASResult,
    IndependentResult,
    LatentResult,
    MLEResult,
    gas_params,
    ou_params,
)
from pyscarcopula.api import fit as api_fit
from pyscarcopula.api import predict as api_predict
from pyscarcopula.api import sample as api_sample
from pyscarcopula.api import smoothed_params
from pyscarcopula.contrib.risk_metrics import (
    _calculate_cvar_fixed,
    _process_chunk_fixed,
)
from pyscarcopula.stattests import rvine_rosenblatt_transform
from pyscarcopula.strategy._base import get_strategy_for_result
from pyscarcopula.vine._edge import VineEdge


pytestmark = pytest.mark.validation


def _analytical_conditional_mvn(sigma, given_idx, given_vals, free_idx):
    s11 = sigma[np.ix_(free_idx, free_idx)]
    s12 = sigma[np.ix_(free_idx, given_idx)]
    s22 = sigma[np.ix_(given_idx, given_idx)]
    s22_inv = np.linalg.inv(s22)

    mu_cond = s12 @ s22_inv @ given_vals
    sigma_cond = s11 - s12 @ s22_inv @ s12.T
    return mu_cond, sigma_cond


def _mvn_pobs(sigma, n, seed):
    rng = np.random.default_rng(seed)
    x = rng.multivariate_normal(np.zeros(sigma.shape[0]), sigma, size=n)
    return np.clip(norm.cdf(x), 1e-10, 1.0 - 1e-10)


def _dynamic_gaussian_chain(n, seed):
    rng = np.random.default_rng(seed)
    copula = BivariateGaussianCopula()
    phase = np.linspace(0.0, 4.0 * np.pi, n)
    rho = np.where(np.sin(phase) > 0.0, 0.80, 0.25)
    u = np.empty((n, 3), dtype=np.float64)
    u[:, 0] = rng.uniform(0.01, 0.99, n)
    u[:, 1] = copula.h_inverse(
        rng.uniform(0.01, 0.99, n),
        u[:, 0],
        rho,
    )
    u[:, 2] = rng.uniform(0.01, 0.99, n)
    return np.clip(u, 1e-9, 1.0 - 1e-9)


def _family_chain_sample(copula, param, n, d, seed):
    rng = np.random.default_rng(seed)
    u = np.empty((n, d), dtype=np.float64)
    u[:, 0] = rng.uniform(0.01, 0.99, n)
    for j in range(1, d):
        z = rng.uniform(0.01, 0.99, n)
        u[:, j] = copula.h_inverse(z, u[:, j - 1], np.full(n, param))
    return np.clip(u, 1e-9, 1.0 - 1e-9)


def _assert_conditional_mvn_moments(samples, sigma, given):
    d = sigma.shape[0]
    given_idx = sorted(given)
    free_idx = [i for i in range(d) if i not in given_idx]
    x_given = np.array([norm.ppf(given[i]) for i in given_idx])
    mu_cond, sigma_cond = _analytical_conditional_mvn(
        sigma, given_idx, x_given, free_idx)

    x_pred = norm.ppf(np.clip(samples[:, free_idx], 1e-10, 1.0 - 1e-10))
    for j, var_idx in enumerate(free_idx):
        sigma_j = np.sqrt(sigma_cond[j, j])
        mean_err = abs(np.mean(x_pred[:, j]) - mu_cond[j]) / sigma_j
        var_err = abs(np.var(x_pred[:, j]) - sigma_cond[j, j]) / sigma_cond[j, j]
        z_j = (x_pred[:, j] - mu_cond[j]) / sigma_j
        stat, _ = kstest(z_j, "norm")

        assert mean_err < 0.12, f"conditional mean mismatch for X_{var_idx}"
        assert var_err < 0.25, f"conditional variance mismatch for X_{var_idx}"
        assert stat < 0.08, f"conditional KS statistic too large for X_{var_idx}"


def _gaussian_conditional_u_mean(rho, given_u):
    x_given = norm.ppf(given_u)
    mean_x = rho * x_given
    var_x = 1.0 - rho ** 2
    return norm.cdf(mean_x / np.sqrt(1.0 + var_x))


class _LinearScoreCopula:
    name = 'LinearScoreCopula'

    def transform(self, f):
        return np.asarray(f, dtype=np.float64)

    def sample(self, n, r, rng=None):
        return np.tile(np.array([[0.25, 0.75]], dtype=np.float64), (n, 1))

    def log_pdf(self, u1, u2, r):
        return np.asarray(r, dtype=np.float64)


class _RiskMetricsFakeCopula:
    def __init__(self, rotate=0):
        self._rotate = rotate

    def fit(self, data, method='mle', **kwargs):
        self.mean_ = np.mean(data, axis=0)
        return self

    def predict(self, n, u=None, rng=None, **kwargs):
        if rng is None:
            rng = np.random.default_rng()
        shift = 0.05 * getattr(self, 'mean_', np.zeros(2))
        return np.clip(rng.uniform(0.05, 0.95, size=(n, 2)) + shift, 1e-6, 1 - 1e-6)


class _IdentityMarginalModel:
    def ppf(self, u, params):
        return np.asarray(u, dtype=np.float64)


def _vine_edge(tree, idx, copula, fit_result):
    edge = VineEdge(tree=tree, idx=idx)
    edge.copula = copula
    edge.fit_result = fit_result
    return edge


def _mle_gaussian_edge(tree, idx, rho):
    copula = BivariateGaussianCopula()
    return _vine_edge(
        tree,
        idx,
        copula,
        MLEResult(
            log_likelihood=0.0,
            method='MLE',
            copula_name=copula.name,
            success=True,
            copula_param=float(rho),
        ),
    )


def _independent_edge(tree, idx):
    copula = IndependentCopula()
    return _vine_edge(
        tree,
        idx,
        copula,
        IndependentResult(
            log_likelihood=0.0,
            method='INDEPENDENT',
            copula_name=copula.name,
            success=True,
        ),
    )


class TestConditionalSamplingPlanLayer:

    @pytest.mark.parametrize(
        ('copula_factory', 'param'),
        [
            (BivariateGaussianCopula, 0.65),
            (lambda: GumbelCopula(rotate=180), 2.2),
            (FrankCopula, 5.0),
        ],
    )
    def test_rvine_family_sweep_predict_is_reproducible_and_well_formed(
            self, copula_factory, param):
        copula = copula_factory()
        u_train = _family_chain_sample(copula, param, 900, 4, seed=101)
        vine = RVineCopula(candidates=[type(copula)]).fit(u_train, method='mle')
        peel_order = [
            int(vine.matrix[vine.d - 1 - col, col])
            for col in range(vine.d)
        ]
        given_var = peel_order[-1]

        samples1 = vine.predict(
            800,
            given={given_var: 0.35},
            rng=np.random.default_rng(102),
        )
        samples2 = vine.predict(
            800,
            given={given_var: 0.35},
            rng=np.random.default_rng(102),
        )

        assert samples1.shape == (800, 4)
        assert np.all(np.isfinite(samples1))
        assert np.all(samples1 > 0.0)
        assert np.all(samples1 < 1.0)
        assert np.allclose(samples1[:, given_var], 0.35)
        np.testing.assert_allclose(samples1, samples2, rtol=0.0, atol=0.0)

    @pytest.mark.parametrize(
        ('copula_factory', 'param'),
        [
            (BivariateGaussianCopula, 0.65),
            (lambda: GumbelCopula(rotate=180), 2.2),
            (FrankCopula, 5.0),
        ],
    )
    def test_rvine_family_sweep_conditional_shifts_neighbor_coordinate(
            self, copula_factory, param):
        copula = copula_factory()
        u_train = _family_chain_sample(copula, param, 900, 3, seed=103)
        vine = RVineCopula(candidates=[type(copula)]).fit(u_train, method='mle')

        low = vine.predict(
            1200,
            given={0: 0.2},
            rng=np.random.default_rng(104),
        )
        high = vine.predict(
            1200,
            given={0: 0.8},
            rng=np.random.default_rng(105),
        )

        assert np.allclose(low[:, 0], 0.2)
        assert np.allclose(high[:, 0], 0.8)
        assert np.mean(high[:, 1]) > np.mean(low[:, 1]) + 0.05

    def test_rvine_mle_supported_conditional_matches_mvn_oracle(self):
        sigma = np.array([
            [1.0, 0.6, 0.3, 0.1],
            [0.6, 1.0, 0.5, 0.2],
            [0.3, 0.5, 1.0, 0.4],
            [0.1, 0.2, 0.4, 1.0],
        ])
        u_train = _mvn_pobs(sigma, 5000, seed=110)
        vine = RVineCopula(candidates=[BivariateGaussianCopula]).fit(
            u_train, method='mle')
        peel_order = [
            int(vine.matrix[vine.d - 1 - col, col])
            for col in range(vine.d)
        ]
        given = {peel_order[-1]: 0.35}

        samples = vine.predict(
            1500,
            given=given,
            rng=np.random.default_rng(111),
        )

        assert np.allclose(samples[:, peel_order[-1]], 0.35)
        _assert_conditional_mvn_moments(samples, sigma, given)

    def test_rvine_mle_rosenblatt_residuals_are_uniform_and_independent(self):
        sigma = np.array([
            [1.0, 0.6, 0.3, 0.1],
            [0.6, 1.0, 0.5, 0.2],
            [0.3, 0.5, 1.0, 0.4],
            [0.1, 0.2, 0.4, 1.0],
        ])
        u_train = _mvn_pobs(sigma, 2500, seed=112)
        vine = RVineCopula(candidates=[BivariateGaussianCopula]).fit(
            u_train, method='mle')

        residuals = rvine_rosenblatt_transform(vine, u_train)

        assert residuals.shape == u_train.shape
        assert np.all(residuals > 0.0)
        assert np.all(residuals < 1.0)
        for col in range(residuals.shape[1]):
            result = cramervonmises(residuals[:, col], "uniform")
            assert result.pvalue > 0.01

        for i in range(residuals.shape[1]):
            for j in range(i + 1, residuals.shape[1]):
                tau, pvalue = kendalltau(residuals[:, i], residuals[:, j])
                assert abs(tau) < 0.05
                assert pvalue > 0.01

    def test_scar_tm_smoothed_params_track_controlled_dynamic_blocks(self):
        copula = BivariateGaussianCopula()
        rho_true = np.r_[
            np.full(80, -0.45),
            np.full(80, 0.75),
            np.full(80, -0.35),
        ]
        u_train = copula.sample(
            len(rho_true),
            rho_true,
            rng=np.random.default_rng(113),
        )
        result = LatentResult(
            log_likelihood=0.0,
            method='SCAR-TM-OU',
            copula_name=copula.name,
            success=True,
            params=ou_params(1.0, 0.0, 2.0),
            K=60,
            grid_range=4.0,
        )
        smoothed = get_strategy_for_result(result).smoothed_params(
            copula, u_train, result)

        low_first = np.mean(smoothed[20:70])
        high_middle = np.mean(smoothed[100:150])
        low_last = np.mean(smoothed[180:230])

        assert high_middle > low_first + 0.35
        assert high_middle > low_last + 0.20
        assert np.std(smoothed) > 0.15

    def test_scar_tm_fit_recovers_controlled_dynamic_blocks(self):
        copula = BivariateGaussianCopula()
        rho_true = np.r_[
            np.full(80, -0.45),
            np.full(80, 0.75),
            np.full(80, -0.35),
        ]
        u_train = copula.sample(
            len(rho_true),
            rho_true,
            rng=np.random.default_rng(124),
        )
        result = api_fit(
            BivariateGaussianCopula(),
            u_train,
            method='scar-tm-ou',
            K=30,
            grid_range=4.0,
            tol=0.05,
        )
        smoothed = smoothed_params(BivariateGaussianCopula(), u_train, result)

        low_first = np.mean(smoothed[20:70])
        high_middle = np.mean(smoothed[100:150])
        low_last = np.mean(smoothed[180:230])

        assert result.success
        assert high_middle > low_first + 0.45
        assert high_middle > low_last + 0.45
        assert np.std(smoothed) > 0.20

    def test_scar_tm_fit_recovers_full_ou_path(self):
        rng = np.random.default_rng(304)
        T = 350
        theta_true, mu_true, nu_true = 1.0, 1.0, 1.8
        dt = 1.0 / (T - 1)
        rho_ou = np.exp(-theta_true * dt)
        sigma_cond = np.sqrt(
            nu_true ** 2 / (2.0 * theta_true) * (1.0 - rho_ou ** 2))

        x = np.empty(T)
        x[0] = rng.normal(mu_true, nu_true / np.sqrt(2.0 * theta_true))
        for t in range(1, T):
            x[t] = (mu_true
                    + rho_ou * (x[t - 1] - mu_true)
                    + sigma_cond * rng.standard_normal())

        copula = BivariateGaussianCopula()
        rho_true = copula.transform(x)
        u_train = copula.sample(T, rho_true, rng=rng)

        result = api_fit(
            BivariateGaussianCopula(),
            u_train,
            method='scar-tm-ou',
            K=50,
            grid_range=4.5,
            tol=0.05,
        )
        smoothed = smoothed_params(BivariateGaussianCopula(), u_train, result)
        trim = 20
        corr = spearmanr(smoothed[trim:], rho_true[trim:]).statistic
        rmse = np.sqrt(np.mean((smoothed[trim:] - rho_true[trim:]) ** 2))
        std_true = np.std(rho_true[trim:])

        assert result.success
        assert corr > 0.6
        assert rmse / std_true < 0.95

    def test_gas_predictive_conditional_uses_last_score_state(self):
        copula = BivariateGaussianCopula()
        u_train = np.array([
            [0.45, 0.50],
            [0.50, 0.45],
            [0.999, 0.999],
        ])
        result = GASResult(
            log_likelihood=0.0,
            method='GAS',
            copula_name=copula.name,
            success=True,
            params=gas_params(0.0, 2.0, 0.0),
            scaling='unit',
            r_last=0.0,
        )
        strategy = get_strategy_for_result(result)
        r_next = strategy.predictive_params(
            copula, u_train, result, 1, horizon='next')[0]

        samples = api_predict(
            copula,
            u_train,
            result,
            2000,
            rng=np.random.default_rng(114),
            given={0: 0.95},
            horizon='next',
        )

        expected = _gaussian_conditional_u_mean(r_next, 0.95)
        sample_mean = float(np.mean(samples[:, 1]))
        assert r_next > 0.75
        assert sample_mean > 0.70
        assert abs(sample_mean - expected) < 0.04

    def test_gas_fit_recovers_sampled_score_driven_dynamics(self):
        base_copula = BivariateGaussianCopula()
        base_result = GASResult(
            log_likelihood=0.0,
            method='GAS',
            copula_name=base_copula.name,
            success=True,
            params=gas_params(0.0, 1.4, 0.7),
            scaling='unit',
            r_last=0.0,
        )
        sampled = get_strategy_for_result(base_result).sample(
            base_copula,
            None,
            base_result,
            240,
            rng=np.random.default_rng(125),
        )

        fit_result = api_fit(
            BivariateGaussianCopula(),
            sampled,
            method='gas',
            tol=0.05,
        )
        fit_copula = BivariateGaussianCopula()
        smoothed = smoothed_params(fit_copula, sampled, fit_result)
        strategy = get_strategy_for_result(fit_result)
        r_current = strategy.predictive_params(
            fit_copula, sampled, fit_result, 1, horizon='current')[0]
        r_next = strategy.predictive_params(
            fit_copula, sampled, fit_result, 1, horizon='next')[0]

        assert fit_result.success
        assert abs(fit_result.params.alpha) > 0.5
        assert abs(fit_result.params.beta) > 0.4
        assert np.std(smoothed) > 0.08
        assert not np.isclose(r_current, r_next)

    def test_scar_tm_predictive_conditional_uses_predictive_state_distribution(
            self, monkeypatch):
        copula = BivariateGaussianCopula()
        u_train = _mvn_pobs(np.array([[1.0, 0.3], [0.3, 1.0]]), 30, seed=115)
        target_r = 0.85
        target_x = copula.inv_transform(np.array([target_r]))[0]
        result = LatentResult(
            log_likelihood=0.0,
            method='SCAR-TM-OU',
            copula_name=copula.name,
            success=True,
            params=ou_params(1.0, 0.0, 0.5),
            K=5,
            grid_range=3.0,
        )

        def fake_tm_state_distribution(*args, horizon='next', **kwargs):
            assert horizon == 'next'
            return np.array([target_x]), np.array([1.0])

        monkeypatch.setattr(
            'pyscarcopula.numerical.predictive_tm.tm_state_distribution',
            fake_tm_state_distribution,
        )

        samples = api_predict(
            copula,
            u_train,
            result,
            2000,
            rng=np.random.default_rng(116),
            given={0: 0.95},
            horizon='next',
        )

        expected = _gaussian_conditional_u_mean(target_r, 0.95)
        sample_mean = float(np.mean(samples[:, 1]))
        assert sample_mean > 0.70
        assert abs(sample_mean - expected) < 0.04

    def test_bivariate_mle_sample_refit_roundtrip_recovers_parameter(self):
        copula = BivariateGaussianCopula()
        rho_true = 0.55
        u_train = copula.sample(
            1500,
            np.full(1500, rho_true),
            rng=np.random.default_rng(117),
        )
        result = api_fit(copula, u_train, method='mle')
        samples = api_sample(
            copula,
            u_train,
            result,
            1500,
            rng=np.random.default_rng(118),
        )
        result2 = api_fit(BivariateGaussianCopula(), samples, method='mle')

        assert abs(result.copula_param - rho_true) < 0.06
        assert abs(result2.copula_param - result.copula_param) < 0.08

    def test_rvine_gas_sample_refit_keeps_dynamic_edge_alive(self):
        u_train = _dynamic_gaussian_chain(120, seed=119)
        vine = RVineCopula(
            candidates=[BivariateGaussianCopula],
            truncation_level=1,
        ).fit(u_train, method='gas')
        assert any(
            isinstance(pc.fit_result, GASResult)
            for pc in vine.pair_copulas.values()
        )

        samples = vine.sample(120, rng=np.random.default_rng(120))
        refit = RVineCopula(
            candidates=[BivariateGaussianCopula],
            truncation_level=1,
        ).fit(samples, method='gas')
        gas_params_refit = [
            pc.fit_result.params
            for pc in refit.pair_copulas.values()
            if isinstance(pc.fit_result, GASResult)
        ]

        assert gas_params_refit
        assert np.isfinite(refit.log_likelihood())
        assert any(abs(p.alpha) > 0.2 for p in gas_params_refit)

    def test_cvine_dynamic_prefix_conditional_matches_predictive_edge_state(self):
        copula = BivariateGaussianCopula()
        target_r = 0.85
        gas_edge = _vine_edge(
            0,
            0,
            copula,
            GASResult(
                log_likelihood=0.0,
                method='GAS',
                copula_name=copula.name,
                success=True,
                params=gas_params(0.0, 0.0, 0.0),
                scaling='unit',
                r_last=target_r,
            ),
        )

        vine = CVineCopula(candidates=[BivariateGaussianCopula])
        vine.d = 4
        vine.method = 'MIXED'
        vine.edges = [
            [gas_edge, _mle_gaussian_edge(0, 1, 0.15), _independent_edge(0, 2)],
            [_independent_edge(1, 0), _independent_edge(1, 1)],
            [_independent_edge(2, 0)],
        ]

        samples = vine.predict(
            2000,
            given={0: 0.95},
            rng=np.random.default_rng(121),
        )

        expected = _gaussian_conditional_u_mean(target_r, 0.95)
        sample_mean = float(np.mean(samples[:, 1]))
        assert np.allclose(samples[:, 0], 0.95)
        assert sample_mean > 0.70
        assert abs(sample_mean - expected) < 0.04
        assert abs(np.mean(samples[:, 3]) - 0.5) < 0.04

    def test_cvine_scar_tm_train_pseudo_obs_use_mixture_h(self, monkeypatch):
        u_train = _mvn_pobs(np.eye(3), 12, seed=122)
        copula = BivariateGaussianCopula()
        scar_edge = _vine_edge(
            0,
            0,
            copula,
            LatentResult(
                log_likelihood=0.0,
                method='SCAR-TM-OU',
                copula_name=copula.name,
                success=True,
                params=ou_params(1.0, 0.0, 0.5),
                K=5,
                grid_range=3.0,
            ),
        )
        vine = CVineCopula(candidates=[BivariateGaussianCopula])
        vine.d = 3
        vine.method = 'MIXED'
        vine.edges = [
            [scar_edge, _mle_gaussian_edge(0, 1, 0.0)],
            [_mle_gaussian_edge(1, 0, 0.0)],
        ]
        h_calls = []
        r_calls = []

        def fake_edge_h(edge, u2, u1, u_pair, K=300, grid_range=5.0):
            h_calls.append((edge.fit_result.method, u_pair.copy(), K, grid_range))
            if edge is scar_edge:
                return np.full(len(u2), 0.73)
            return np.asarray(u2, dtype=np.float64)

        def fake_generate_r(edge, n, v_train_pair, K, grid_range,
                            horizon='next', **kwargs):
            r_calls.append((edge.fit_result.method, v_train_pair, K, grid_range))
            return np.full(n, 0.0)

        monkeypatch.setattr('pyscarcopula.vine.cvine._edge_h', fake_edge_h)
        monkeypatch.setattr(
            'pyscarcopula.vine.cvine.generate_r_for_predict',
            fake_generate_r,
        )

        samples = vine.predict(
            5,
            u=u_train,
            given={0: 0.4},
            K=5,
            grid_range=3.0,
            rng=np.random.default_rng(123),
        )

        assert samples.shape == (5, 3)
        assert any(method == 'SCAR-TM-OU' for method, *_ in h_calls)
        assert any(
            method == 'MLE'
            and v_pair is not None
            and np.allclose(v_pair[:, 0], 0.73)
            for method, v_pair, *_ in r_calls
        )

    def test_rvine_rejects_unsupported_arbitrary_conditioning(self):
        u_train = _mvn_pobs(np.eye(4), 400, seed=120)
        vine = RVineCopula(candidates=[IndependentCopula]).fit(
            u_train, method='mle')
        peel_order = [
            int(vine.matrix[vine.d - 1 - col, col])
            for col in range(vine.d)
        ]
        given = {peel_order[0]: 0.25, peel_order[2]: 0.75}

        if vine._suffix_sampling_state(given) is not None:
            return
        with np.testing.assert_raises_regex(ValueError, "R-vine variable order"):
            vine.predict(10, given=given, rng=np.random.default_rng(121))

    def test_extreme_given_values_remain_finite(self):
        sigma = np.array([[1.0, 0.85], [0.85, 1.0]])
        u_train = _mvn_pobs(sigma, 1200, seed=130)
        vine = RVineCopula(candidates=[BivariateGaussianCopula]).fit(
            u_train, method='mle')

        for given_value in (0.001, 0.999):
            samples = vine.predict(
                300,
                given={0: given_value},
                rng=np.random.default_rng(131),
            )
            assert np.all(np.isfinite(samples))
            assert np.all(samples > 0.0)
            assert np.all(samples < 1.0)
            assert np.allclose(samples[:, 0], given_value)

    def test_near_singular_gaussian_conditional_is_concentrated(self):
        copula = BivariateGaussianCopula()
        rho = 0.99
        u_train = copula.sample(
            2000,
            np.full(2000, rho),
            rng=np.random.default_rng(132),
        )
        result = api_fit(copula, u_train, method='mle')

        samples = api_predict(
            copula,
            u_train,
            result,
            2000,
            given={0: 0.5},
            rng=np.random.default_rng(133),
        )

        assert np.all(np.isfinite(samples))
        assert np.all(samples > 0.0)
        assert np.all(samples < 1.0)
        assert np.allclose(samples[:, 0], 0.5)
        assert abs(np.mean(samples[:, 1]) - 0.5) < 0.01
        assert np.std(samples[:, 1]) < 0.08

    def test_near_singular_rvine_conditional_is_concentrated(self):
        copula = BivariateGaussianCopula()
        rho = 0.99
        u_train = copula.sample(
            2500,
            np.full(2500, rho),
            rng=np.random.default_rng(134),
        )
        vine = RVineCopula(candidates=[BivariateGaussianCopula]).fit(
            u_train, method='mle')

        samples = vine.predict(
            2000,
            given={0: 0.5},
            rng=np.random.default_rng(135),
        )

        assert np.all(np.isfinite(samples))
        assert np.all(samples > 0.0)
        assert np.all(samples < 1.0)
        assert np.allclose(samples[:, 0], 0.5)
        assert abs(np.mean(samples[:, 1]) - 0.5) < 0.01
        assert np.std(samples[:, 1]) < 0.08

    def test_predict_reproducible_with_fixed_rng(self):
        sigma = np.array([[1.0, 0.7], [0.7, 1.0]])
        u_train = _mvn_pobs(sigma, 600, seed=140)
        vine = RVineCopula(candidates=[BivariateGaussianCopula]).fit(
            u_train, method='mle')

        s1 = vine.predict(
            500,
            given={0: 0.2},
            rng=np.random.default_rng(141),
        )
        s2 = vine.predict(
            500,
            given={0: 0.2},
            rng=np.random.default_rng(141),
        )

        np.testing.assert_allclose(s1, s2, rtol=0.0, atol=0.0)

    def test_gumbel_sample_reproducible_with_fixed_rng(self):
        copula = GumbelCopula()
        r = np.full(300, 2.0)

        s1 = copula.sample(300, r, rng=np.random.default_rng(150))
        s2 = copula.sample(300, r, rng=np.random.default_rng(150))

        np.testing.assert_allclose(s1, s2, rtol=0.0, atol=0.0)

    def test_scar_tm_predict_reproducible_with_fixed_rng(self):
        copula = BivariateGaussianCopula()
        u_train = _mvn_pobs(np.array([[1.0, 0.5], [0.5, 1.0]]), 80, seed=160)
        result = LatentResult(
            log_likelihood=0.0,
            method='SCAR-TM-OU',
            copula_name=copula.name,
            success=True,
            params=ou_params(1.2, 0.1, 0.4),
            K=25,
            grid_range=3.0,
        )

        s1 = api_predict(
            copula,
            u_train,
            result,
            400,
            rng=np.random.default_rng(161),
            given={0: 0.3},
            horizon='current',
            K=25,
            grid_range=3.0,
        )
        s2 = api_predict(
            copula,
            u_train,
            result,
            400,
            rng=np.random.default_rng(161),
            given={0: 0.3},
            horizon='current',
            K=25,
            grid_range=3.0,
        )

        np.testing.assert_allclose(s1, s2, rtol=0.0, atol=0.0)

    def test_gas_current_and_next_horizons_use_different_score_states(self):
        copula = BivariateGaussianCopula()
        result = GASResult(
            log_likelihood=0.0,
            method='GAS',
            copula_name=copula.name,
            success=True,
            params=gas_params(0.0, 0.8, 0.4),
            scaling='unit',
            r_last=0.0,
        )
        u_train = np.array([
            [0.40, 0.45],
            [0.15, 0.90],
        ])
        strategy = get_strategy_for_result(result)

        r_current = strategy.predictive_params(
            copula, u_train, result, 4, horizon='current')
        r_next = strategy.predictive_params(
            copula, u_train, result, 4, horizon='next')

        assert np.allclose(r_current, r_current[0])
        assert np.allclose(r_next, r_next[0])
        assert not np.isclose(r_current[0], r_next[0])

    def test_scar_tm_current_and_next_horizons_route_to_predictive_state(
            self, monkeypatch):
        copula = BivariateGaussianCopula()
        u_train = _mvn_pobs(np.array([[1.0, 0.5], [0.5, 1.0]]), 20, seed=170)
        result = LatentResult(
            log_likelihood=0.0,
            method='SCAR-TM-OU',
            copula_name=copula.name,
            success=True,
            params=ou_params(2.0, 0.0, 0.5),
            K=5,
            grid_range=3.0,
        )
        calls = []

        def fake_tm_state_distribution(*args, horizon='next', **kwargs):
            calls.append(horizon)
            if horizon == 'current':
                return np.array([0.75]), np.array([1.0])
            return np.array([0.25]), np.array([1.0])

        monkeypatch.setattr(
            'pyscarcopula.numerical.predictive_tm.tm_state_distribution',
            fake_tm_state_distribution,
        )
        strategy = get_strategy_for_result(result)

        r_current = strategy.predictive_params(
            copula,
            u_train,
            result,
            3,
            rng=np.random.default_rng(171),
            horizon='current',
        )
        r_next = strategy.predictive_params(
            copula,
            u_train,
            result,
            3,
            rng=np.random.default_rng(171),
            horizon='next',
        )

        assert calls == ['current', 'next']
        np.testing.assert_allclose(r_current, copula.transform(np.full(3, 0.75)))
        np.testing.assert_allclose(r_next, copula.transform(np.full(3, 0.25)))
        assert not np.allclose(r_current, r_next)

    def test_gas_sample_uses_score_driven_recursion(self):
        copula = _LinearScoreCopula()
        result = GASResult(
            log_likelihood=0.0,
            method='GAS',
            copula_name=copula.name,
            success=True,
            params=gas_params(0.0, 0.2, 0.5),
            scaling='unit',
            r_last=0.0,
        )
        strategy = get_strategy_for_result(result)
        seen_r = []

        def sample_and_record(n, r, rng=None):
            seen_r.extend(np.asarray(r, dtype=np.float64).tolist())
            return np.tile(np.array([[0.25, 0.75]], dtype=np.float64), (n, 1))

        copula.sample = sample_and_record

        samples = strategy.sample(
            copula, None, result, 4, rng=np.random.default_rng(180))

        assert samples.shape == (4, 2)
        np.testing.assert_allclose(samples, np.tile([0.25, 0.75], (4, 1)))
        assert len(seen_r) == 4
        assert len(set(np.round(seen_r, 12))) > 1

    def test_risk_metrics_worker_chunks_use_reproducible_per_window_rng(self):
        rng = np.random.default_rng(190)
        data = rng.normal(0.0, 0.01, size=(8, 2))
        window_len = 4
        n_windows = data.shape[0] - window_len + 1
        marginal_model = _IdentityMarginalModel()
        marg_params = [None] * data.shape[0]
        portfolio_weight = np.array([0.5, 0.5])

        seeds1 = np.random.SeedSequence(191).spawn(n_windows)
        worker_args = [
            (
                0,
                3,
                data,
                'mle',
                _RiskMetricsFakeCopula,
                {'rotate': 0},
                marginal_model,
                marg_params,
                0.8,
                window_len,
                20,
                portfolio_weight,
                {},
                seeds1[0:3],
            ),
            (
                3,
                n_windows,
                data,
                'mle',
                _RiskMetricsFakeCopula,
                {'rotate': 0},
                marginal_model,
                marg_params,
                0.8,
                window_len,
                20,
                portfolio_weight,
                {},
                seeds1[3:n_windows],
            ),
        ]
        chunk_results = []
        for args in worker_args:
            chunk_results.extend(_process_chunk_fixed(args))

        var1 = np.zeros(data.shape[0])
        cvar1 = np.zeros(data.shape[0])
        for idx, var, cvar in chunk_results:
            var1[idx] = var
            cvar1[idx] = cvar

        seeds3 = np.random.SeedSequence(191).spawn(n_windows)
        var_seq, cvar_seq, _ = _calculate_cvar_fixed(
            _RiskMetricsFakeCopula(),
            data,
            'mle',
            marginal_model,
            marg_params,
            0.8,
            window_len,
            20,
            portfolio_weight,
            n_jobs=1,
            window_seed_sequences=seeds3,
        )

        np.testing.assert_allclose(var1, var_seq, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(cvar1, cvar_seq, rtol=0.0, atol=0.0)
