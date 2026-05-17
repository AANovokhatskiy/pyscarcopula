"""Test C-vine copula: structure, truncation, independence selection, GoF, sample/predict."""
import importlib
import numpy as np
import pytest
import pyscarcopula.vine as vine_pkg
from pyscarcopula.vine.cvine import CVineCopula
from pyscarcopula.vine._edge_adapter import (
    edge_method,
    edge_n_params,
    edge_param,
    edge_view,
)
from pyscarcopula.vine._pair_copula import PairCopula
from pyscarcopula.vine._rvine_edges import (
    _edge_h,
    _edge_log_likelihood,
    _edge_r_for_sample,
)
from pyscarcopula.vine._selection import (
    select_best_copula, _default_candidates, _kendall_tau,
)
from pyscarcopula.vine._helpers import _clip_unit
from pyscarcopula.copula.elliptical import BivariateGaussianCopula
from pyscarcopula.copula.independent import IndependentCopula
from pyscarcopula.stattests import gof_test
from pyscarcopula._utils import pobs
from pyscarcopula._types import (
    GASResult, IndependentResult, MLEResult, gas_params,
)

# ══════════════════════════════════════════════════════════════
# Edge module tests
# ══════════════════════════════════════════════════════════════

class TestPairCopulaEdge:
    def test_package_all_excludes_internal_edge_helpers(self):
        assert '_edge_h' not in vine_pkg.__all__
        assert '_edge_log_likelihood' not in vine_pkg.__all__
        assert not hasattr(vine_pkg, '_edge_h')
        assert not hasattr(vine_pkg, '_edge_log_likelihood')
        assert 'VineEdge' not in vine_pkg.__all__
        assert not hasattr(vine_pkg, 'VineEdge')

    def test_edge_creation(self):
        edge = PairCopula(tree=0, idx=0)
        assert edge.tree == 0
        assert edge.idx == 0
        assert edge.copula is None
        assert edge.method is None

    def test_edge_adapter_view_for_pair_copula(self):
        cop = BivariateGaussianCopula()
        edge = PairCopula(
            tree=1,
            idx=2,
            copula=cop,
            fit_result=MLEResult(
                log_likelihood=1.5,
                method='MLE',
                copula_name=cop.name,
                success=True,
                copula_param=0.35,
            ),
        )

        view = edge_view(edge)

        assert view.copula is cop
        assert view.fit_result is edge.fit_result
        assert view.tree == 1
        assert view.idx == 2
        assert edge_method(edge) == 'MLE'
        assert edge_param(edge) == pytest.approx(0.35)
        assert edge_n_params(edge) == 1

    def test_get_r_predict_gas_uses_cached_r_last(self):
        cop = BivariateGaussianCopula()
        edge = PairCopula(tree=0, idx=0)
        edge.copula = cop
        edge.fit_result = GASResult(
            log_likelihood=0.0,
            method='GAS',
            copula_name=cop.name,
            success=True,
            params=gas_params(0.1, 0.2, 0.3),
            r_last=0.42,
        )

        r = edge.get_r_predict(5)
        np.testing.assert_allclose(r, np.full(5, 0.42))

    def test_get_r_gas_uses_score_driven_path(self):
        cop = BivariateGaussianCopula()
        edge = PairCopula(tree=0, idx=0)
        edge.copula = cop
        edge.fit_result = GASResult(
            log_likelihood=0.0,
            method='GAS',
            copula_name=cop.name,
            success=True,
            params=gas_params(0.1, 0.2, 0.3),
        )
        u = np.array([[0.2, 0.3], [0.4, 0.6], [0.8, 0.7]])

        r = edge.get_r(u)

        assert r.shape == (3,)
        assert np.all(np.isfinite(r))

    def test_edge_h_scar_tm_uses_mixture_h(self, monkeypatch):
        from pyscarcopula._types import LatentResult, ou_params

        cop = BivariateGaussianCopula()
        edge = PairCopula(tree=0, idx=0)
        edge.copula = cop
        edge.fit_result = LatentResult(
            log_likelihood=0.0,
            method='SCAR-TM-OU',
            copula_name=cop.name,
            success=True,
            params=ou_params(1.0, 0.0, 0.5),
        )
        u_pair = np.array([[0.2, 0.3], [0.4, 0.6]])
        calls = []

        def fake_tm_forward_mixture_h(kappa, mu, nu, u_arg, cop_arg,
                                      *args, **kwargs):
            calls.append((kappa, mu, nu, u_arg.copy(), cop_arg, args, kwargs))
            return np.array([0.11, 0.89])

        monkeypatch.setattr(
            'pyscarcopula.strategy.scar_tm.tm_forward_mixture_h',
            fake_tm_forward_mixture_h,
        )

        out = _edge_h(
            edge,
            u_pair[:, 1],
            u_pair[:, 0],
            u_pair=u_pair,
            K=7,
            grid_range=2.5,
        )

        np.testing.assert_allclose(out, np.array([0.11, 0.89]))
        assert calls
        np.testing.assert_allclose(calls[0][3], u_pair)
        assert calls[0][5][0] == 7
        assert calls[0][5][1] == 2.5

    def test_edge_gas_helpers_use_result_score_eps(self, monkeypatch):
        cop = BivariateGaussianCopula()
        edge = PairCopula(tree=0, idx=0)
        edge.copula = cop
        edge.fit_result = GASResult(
            log_likelihood=0.0,
            method='GAS',
            copula_name=cop.name,
            success=True,
            params=gas_params(0.1, 0.2, 0.3),
            score_eps=3e-6,
        )
        u_pair = np.array([[0.2, 0.3], [0.4, 0.6]])
        captured = []

        def fake_mixture_h(omega, gamma, beta, u_arg, cop_arg, scaling,
                           score_eps):
            captured.append(('h', score_eps))
            return np.array([0.1, 0.2])

        def fake_gas_filter(omega, gamma, beta, u_arg, cop_arg, scaling,
                            score_eps):
            captured.append(('ll', score_eps))
            return np.zeros(len(u_arg)), np.zeros(len(u_arg)), 7.0

        gas_module = importlib.import_module('pyscarcopula.strategy.gas')
        monkeypatch.setattr(
            gas_module, 'gas_mixture_h', fake_mixture_h)
        monkeypatch.setattr(
            gas_module, 'gas_filter', fake_gas_filter)

        _edge_h(edge, u_pair[:, 1], u_pair[:, 0], u_pair=u_pair)
        assert _edge_log_likelihood(edge, u_pair) == pytest.approx(7.0)
        assert captured == [('h', pytest.approx(3e-6)),
                            ('ll', pytest.approx(3e-6))]

    def test_edge_method_after_fit(self):
        u = pobs(np.random.default_rng(42).standard_normal((200, 4)))
        vine = CVineCopula()
        vine.fit(u, method='mle')
        edge = vine.edges[0][0]
        assert edge.method is not None
        assert edge.method.upper() == 'MLE'


# ══════════════════════════════════════════════════════════════
# Selection module tests
# ══════════════════════════════════════════════════════════════

class TestSelection:
    def test_default_candidates_not_empty(self):
        cands = _default_candidates()
        assert len(cands) >= 4

    def test_kendall_tau_range(self):
        rng = np.random.default_rng(42)
        u1 = rng.uniform(0, 1, 200)
        u2 = rng.uniform(0, 1, 200)
        tau = _kendall_tau(u1, u2)
        assert -1.0 <= tau <= 1.0

    def test_select_best_copula_returns_fitted(self):
        rng = np.random.default_rng(42)
        u = pobs(rng.standard_normal((200, 2)))
        cop, result = select_best_copula(
            u[:, 0], u[:, 1], _default_candidates())
        assert cop is not None
        assert result is not None
        assert hasattr(result, 'log_likelihood')

    def test_independent_data_selects_independence(self):
        rng = np.random.default_rng(99)
        u1 = rng.uniform(0, 1, 500)
        u2 = rng.uniform(0, 1, 500)
        cop, result = select_best_copula(u1, u2, _default_candidates())
        assert isinstance(cop, IndependentCopula)

    def test_gaussian_selection_allows_negative_dependence(self):
        rng = np.random.default_rng(101)
        z1 = rng.standard_normal(600)
        z2 = -0.7 * z1 + 0.4 * rng.standard_normal(600)
        u = pobs(np.column_stack((z1, z2)))

        cop, result = select_best_copula(
            u[:, 0], u[:, 1], [BivariateGaussianCopula])

        assert isinstance(cop, BivariateGaussianCopula)
        assert result.copula_param < -0.3


# ══════════════════════════════════════════════════════════════
# Helpers module tests
# ══════════════════════════════════════════════════════════════

class TestHelpers:
    def test_clip_unit(self):
        x = np.array([-0.1, 0.0, 0.5, 1.0, 1.1])
        clipped = _clip_unit(x)
        assert np.all(clipped > 0)
        assert np.all(clipped < 1)

    def test_edge_r_for_sample_mle(self):
        u = pobs(np.random.default_rng(42).standard_normal((200, 3)))
        vine = CVineCopula()
        vine.fit(u, method='mle')
        edge = vine.edges[0][0]
        rng = np.random.default_rng(0)
        r = _edge_r_for_sample(edge, 100, rng)
        assert r.shape == (100,)

    def test_edge_r_for_sample_rejects_stepwise_state(self):
        cop = BivariateGaussianCopula()
        edge = PairCopula(tree=0, idx=0)
        edge.copula = cop
        edge.fit_result = GASResult(
            log_likelihood=0.0,
            method='GAS',
            copula_name=cop.name,
            success=True,
            params=gas_params(0.1, 0.2, 0.3),
        )

        with pytest.raises(ValueError, match="stepwise score updates"):
            _edge_r_for_sample(edge, 10, np.random.default_rng(0))


# ══════════════════════════════════════════════════════════════
# CVineCopula structure tests
# ══════════════════════════════════════════════════════════════

class TestVineStructure:
    @pytest.mark.parametrize("d", [3, 4, 6])
    def test_number_of_edges(self, d):
        u = pobs(np.random.default_rng(42).standard_normal((200, d)))
        vine = CVineCopula()
        vine.fit(u, method='mle')
        n_edges = sum(len(tree) for tree in vine.edges)
        assert n_edges == d * (d - 1) // 2

    @pytest.mark.parametrize("d", [3, 4, 6])
    def test_number_of_trees(self, d):
        u = pobs(np.random.default_rng(42).standard_normal((200, d)))
        vine = CVineCopula()
        vine.fit(u, method='mle')
        assert len(vine.edges) == d - 1

    def test_edges_per_tree(self):
        d = 6
        u = pobs(np.random.default_rng(42).standard_normal((200, d)))
        vine = CVineCopula()
        vine.fit(u, method='mle')
        for j, tree_edges in enumerate(vine.edges):
            assert len(tree_edges) == d - j - 1

    def test_fit_rejects_one_dimensional_data(self):
        with pytest.raises(ValueError, match="data must be 2D"):
            CVineCopula().fit(np.array([0.1, 0.2, 0.3]))

    def test_fit_rejects_d1(self):
        with pytest.raises(ValueError, match="need d >= 2"):
            CVineCopula().fit(np.full((5, 1), 0.5))

    def test_dynamic_fit_failure_falls_back_to_mle(self, monkeypatch):
        calls = []

        def fake_fit(copula, u_pair, method='mle', **kwargs):
            calls.append(str(method).lower())
            if str(method).lower() == 'mle':
                return MLEResult(
                    log_likelihood=1.0,
                    method='MLE',
                    copula_name=copula.name,
                    success=True,
                    copula_param=0.25,
                )
            return GASResult(
                log_likelihood=-1.0,
                method='GAS',
                copula_name=copula.name,
                success=False,
                params=gas_params(0.0, 0.0, 0.0),
                message='failed',
            )

        monkeypatch.setattr('pyscarcopula.api.fit', fake_fit)

        u = pobs(np.random.default_rng(11).standard_normal((30, 2)))
        vine = CVineCopula()
        vine.fit(
            u,
            method='gas',
            copulas=[[(BivariateGaussianCopula, 0)]],
        )

        assert calls == ['mle', 'gas']
        assert vine.edges[0][0].fit_result.method == 'MLE'


# ══════════════════════════════════════════════════════════════
# Truncation tests
# ══════════════════════════════════════════════════════════════

class TestVineTruncation:
    def test_truncation_level(self, crypto_data_6d):
        vine = CVineCopula()
        vine.fit(crypto_data_6d, method='scar-tm-ou',
                 K=50, gtol=0.5, truncation_level=1)
        for j, tree_edges in enumerate(vine.edges):
            for edge in tree_edges:
                if j >= 1:
                    assert edge.fit_result.method.upper() == 'MLE'

    def test_min_edge_logL(self, crypto_data_6d):
        vine = CVineCopula()
        vine.fit(crypto_data_6d, method='scar-tm-ou',
                 K=50, gtol=0.5, min_edge_logL=50)
        n_mle = sum(1 for tree in vine.edges for e in tree
                    if e.fit_result.method.upper() == 'MLE')
        assert n_mle > 0


# ══════════════════════════════════════════════════════════════
# Independence selection tests
# ══════════════════════════════════════════════════════════════

class TestVineIndependence:
    def test_independent_data_selects_independence(self):
        rng = np.random.default_rng(99)
        u = rng.uniform(0, 1, (500, 4))
        vine = CVineCopula()
        vine.fit(u, method='mle')
        n_indep = sum(1 for tree in vine.edges for e in tree
                      if isinstance(e.copula, IndependentCopula))
        total = sum(len(tree) for tree in vine.edges)
        assert n_indep >= total // 2


# ══════════════════════════════════════════════════════════════
# GoF tests
# ══════════════════════════════════════════════════════════════

class TestVineGoF:
    def test_vine_gof_runs(self, crypto_data_6d):
        vine = CVineCopula()
        vine.fit(crypto_data_6d, method='mle')
        gof = gof_test(vine, crypto_data_6d, to_pobs=False)
        assert 0 <= gof.pvalue <= 1


# ══════════════════════════════════════════════════════════════
# Sample / predict tests
# ══════════════════════════════════════════════════════════════

class TestVineSamplePredict:
    def test_sample_rejects_nonpositive_n(self):
        vine = CVineCopula()
        vine.edges = [[]]
        vine.d = 2
        with pytest.raises(ValueError, match="n must be positive"):
            vine.sample(0)

    def test_predict_rejects_nonpositive_n(self):
        vine = CVineCopula()
        vine.edges = [[]]
        vine.d = 2
        with pytest.raises(ValueError, match="n must be positive"):
            vine.predict(0)

    def test_stepwise_gas_sample_uses_result_score_eps(self):
        class LinearCopula:
            name = 'linear'

            def __init__(self):
                self.r_seen = []

            def transform(self, x):
                return np.asarray(x, dtype=np.float64)

            def h_inverse(self, v, u_given, r):
                return np.asarray(v, dtype=np.float64)

            def h(self, u2, u1, r):
                return np.asarray(u2, dtype=np.float64)

            def log_pdf(self, u1, u2, r):
                r = np.asarray(r, dtype=np.float64)
                self.r_seen.extend(float(x) for x in r.ravel())
                return r * (np.asarray(u1) + np.asarray(u2))

        cop = LinearCopula()
        edge = PairCopula(tree=0, idx=0, copula=cop)
        edge.fit_result = GASResult(
            log_likelihood=0.0,
            method='GAS',
            copula_name=cop.name,
            success=True,
            params=gas_params(0.1, 0.2, 0.3),
            score_eps=2e-5,
        )
        vine = CVineCopula()
        vine.edges = [[edge]]
        vine.d = 2

        vine.sample(1, rng=np.random.default_rng(123))

        assert cop.r_seen[1] - cop.r_seen[2] == pytest.approx(4e-5)

    def test_predict_shape(self):
        d = 4
        u = pobs(np.random.default_rng(0).standard_normal((200, d)))
        vine = CVineCopula()
        vine.fit(u, method='mle')
        assert vine.predict(100).shape == (100, d)

    def test_sample_shape(self):
        d = 4
        u = pobs(np.random.default_rng(0).standard_normal((200, d)))
        vine = CVineCopula()
        vine.fit(u, method='mle')
        assert vine.sample(100, u_train=u).shape == (100, d)

    def test_samples_in_unit_cube(self):
        d = 4
        u = pobs(np.random.default_rng(0).standard_normal((200, d)))
        vine = CVineCopula()
        vine.fit(u, method='mle')
        samples = vine.predict(1000)
        assert np.all(samples >= 0) and np.all(samples <= 1)

    def test_log_likelihood_consistent(self):
        """log_likelihood(data) should match fit_result.log_likelihood."""
        d = 3
        u = pobs(np.random.default_rng(42).standard_normal((200, d)))
        vine = CVineCopula()
        vine.fit(u, method='mle')
        ll_fit = vine.fit_result.log_likelihood
        ll_eval = vine.log_likelihood(u)
        assert abs(ll_fit - ll_eval) < 1e-6

    def test_predict_given_prefix_fixes_columns(self):
        d = 4
        u = pobs(np.random.default_rng(7).standard_normal((200, d)))
        vine = CVineCopula()
        vine.fit(u, method='mle')
        samples = vine.predict(120, given={0: 0.2, 1: 0.8})
        assert samples.shape == (120, d)
        np.testing.assert_allclose(samples[:, 0], 0.2)
        np.testing.assert_allclose(samples[:, 1], 0.8)
        assert np.all((samples[:, 2:] > 0) & (samples[:, 2:] < 1))

    def test_predict_uses_passed_rng(self):
        d = 4
        u = pobs(np.random.default_rng(71).standard_normal((200, d)))
        vine = CVineCopula()
        vine.fit(u, method='mle')

        s1 = vine.predict(50, rng=np.random.default_rng(222))
        s2 = vine.predict(50, rng=np.random.default_rng(222))
        s3 = vine.predict(50, rng=np.random.default_rng(223))

        np.testing.assert_allclose(s1, s2)
        assert not np.allclose(s1, s3)

    def test_predict_all_given_returns_constant_rows(self):
        d = 4
        u = pobs(np.random.default_rng(8).standard_normal((200, d)))
        vine = CVineCopula()
        vine.fit(u, method='mle')
        given = {0: 0.1, 1: 0.3, 2: 0.6, 3: 0.9}
        samples = vine.predict(25, given=given)
        expected = np.array([0.1, 0.3, 0.6, 0.9])
        assert samples.shape == (25, d)
        np.testing.assert_allclose(samples, np.tile(expected, (25, 1)))

    def test_predict_given_non_prefix_fixes_column(self):
        d = 4
        u = pobs(np.random.default_rng(9).standard_normal((200, d)))
        vine = CVineCopula()
        vine.fit(u, method='mle')
        samples = vine.predict(50, given={1: 0.4})
        assert samples.shape == (50, d)
        np.testing.assert_allclose(samples[:, 1], 0.4)
        assert np.all((samples[:, [0, 2, 3]] > 0) & (samples[:, [0, 2, 3]] < 1))

    def test_predict_given_invalid_value_raises(self):
        d = 4
        u = pobs(np.random.default_rng(10).standard_normal((200, d)))
        vine = CVineCopula()
        vine.fit(u, method='mle')
        with pytest.raises(ValueError):
            vine.predict(50, given={0: 1.0})

    def test_predict_given_dynamic_edges_uses_strategy_adapter(self):
        d = 3
        u = pobs(np.random.default_rng(11).standard_normal((200, d)))
        vine = CVineCopula()
        vine.fit(u, method='scar-m-ou', gtol=0.5)
        samples = vine.predict(50, given={0: 0.5},
                               rng=np.random.default_rng(111))
        assert samples.shape == (50, d)
        np.testing.assert_allclose(samples[:, 0], 0.5)

    def test_predict_given_non_prefix_independence_sanity(self):
        from pyscarcopula.copula.independent import IndependentCopula

        d = 4
        u = np.random.default_rng(12).uniform(0, 1, size=(300, d))
        copulas = [[(IndependentCopula, 0) for _ in range(d - j - 1)]
                   for j in range(d - 1)]
        vine = CVineCopula(candidates=[IndependentCopula])
        vine.fit(u, method='mle', copulas=copulas)
        samples = vine.predict(150, given={2: 0.65})
        np.testing.assert_allclose(samples[:, 2], 0.65)
        assert abs(np.mean(samples[:, 0]) - 0.5) < 0.08
        assert abs(np.mean(samples[:, 1]) - 0.5) < 0.08
        assert abs(np.mean(samples[:, 3]) - 0.5) < 0.08

    def test_predict_given_allows_independent_result_method_name(self):
        d = 3
        vine = CVineCopula(candidates=[IndependentCopula])
        vine.d = d
        vine.method = 'MIXED'
        vine.edges = [[] for _ in range(d - 1)]
        for tree in range(d - 1):
            for idx in range(d - tree - 1):
                edge = PairCopula(tree=tree, idx=idx)
                edge.copula = IndependentCopula()
                edge.fit_result = IndependentResult(
                    log_likelihood=0.0,
                    method='INDEPENDENT',
                    copula_name=edge.copula.name,
                    success=True,
                )
                vine.edges[tree].append(edge)

        samples = vine.predict(
            100,
            given={0: 0.4},
            rng=np.random.default_rng(121),
        )

        assert samples.shape == (100, d)
        np.testing.assert_allclose(samples[:, 0], 0.4)
        assert abs(np.mean(samples[:, 1]) - 0.5) < 0.10
        assert abs(np.mean(samples[:, 2]) - 0.5) < 0.10

    def test_predict_given_prefix_with_gas(self):
        d = 4
        u = pobs(np.random.default_rng(13).standard_normal((220, d)))
        vine = CVineCopula()
        vine.fit(u, method='gas')
        samples = vine.predict(80, given={0: 0.25, 1: 0.75})
        assert samples.shape == (80, d)
        np.testing.assert_allclose(samples[:, 0], 0.25)
        np.testing.assert_allclose(samples[:, 1], 0.75)
        assert np.all((samples[:, 2:] > 0) & (samples[:, 2:] < 1))

    def test_predict_given_non_prefix_with_gas(self):
        d = 4
        u = pobs(np.random.default_rng(14).standard_normal((220, d)))
        vine = CVineCopula()
        vine.fit(u, method='gas')
        samples = vine.predict(40, given={2: 0.6})
        assert samples.shape == (40, d)
        np.testing.assert_allclose(samples[:, 2], 0.6)
        assert np.all((samples[:, [0, 1, 3]] > 0) & (samples[:, [0, 1, 3]] < 1))

    def test_predict_with_gas_builds_train_pseudo_obs(self, monkeypatch):
        d = 3
        u = pobs(np.random.default_rng(141).standard_normal((180, d)))
        vine = CVineCopula()
        vine.fit(u, method='gas')

        calls = []

        def fake_edge_r_for_predict(edge, n, u_train_pair=None,
                                    horizon='next', **kwargs):
            calls.append((edge.fit_result.method, u_train_pair, horizon))
            return np.full(n, 0.1)

        monkeypatch.setattr(
            'pyscarcopula.vine.cvine._edge_r_for_predict',
            fake_edge_r_for_predict)

        samples = vine.predict(8, u=u, given={0: 0.4}, horizon='current')

        assert samples.shape == (8, d)
        assert any(
            method == 'GAS'
            and v_pair is not None
            and v_pair.shape[1] == 2
            and horizon == 'current'
            for method, v_pair, horizon in calls
        )

    def test_predict_given_prefix_with_scar_tm(self):
        d = 3
        u = pobs(np.random.default_rng(15).standard_normal((180, d)))
        vine = CVineCopula()
        vine.fit(u, method='scar-tm-ou', K=40, gtol=0.5)
        samples = vine.predict(30, given={0: 0.35}, K=40, grid_range=5.0)
        assert samples.shape == (30, d)
        np.testing.assert_allclose(samples[:, 0], 0.35)
        assert np.all((samples[:, 1:] > 0) & (samples[:, 1:] < 1))

    def test_predict_given_non_prefix_with_scar_tm(self):
        d = 3
        u = pobs(np.random.default_rng(16).standard_normal((180, d)))
        vine = CVineCopula()
        vine.fit(u, method='scar-tm-ou', K=40, gtol=0.5)
        samples = vine.predict(20, given={1: 0.55}, K=40, grid_range=5.0,
                               horizon='current')
        assert samples.shape == (20, d)
        np.testing.assert_allclose(samples[:, 1], 0.55)
        assert np.all((samples[:, [0, 2]] > 0) & (samples[:, [0, 2]] < 1))

