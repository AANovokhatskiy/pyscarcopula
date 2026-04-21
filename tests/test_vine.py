"""Test C-vine copula: structure, truncation, independence selection, GoF, sample/predict."""
import numpy as np
import pytest
from pyscarcopula.vine.cvine import CVineCopula
from pyscarcopula.vine._edge import VineEdge
from pyscarcopula.vine._selection import (
    select_best_copula, _default_candidates, _kendall_tau,
)
from pyscarcopula.vine._helpers import _clip_unit, generate_r_for_sample
from pyscarcopula.copula.elliptical import BivariateGaussianCopula
from pyscarcopula.copula.independent import IndependentCopula
from pyscarcopula.stattests import gof_test
from pyscarcopula._utils import pobs

# ══════════════════════════════════════════════════════════════
# Edge module tests
# ══════════════════════════════════════════════════════════════

class TestVineEdge:
    def test_edge_creation(self):
        edge = VineEdge(tree=0, idx=0)
        assert edge.tree == 0
        assert edge.idx == 0
        assert edge.copula is None
        assert edge.method is None

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

    def test_generate_r_for_sample_mle(self):
        u = pobs(np.random.default_rng(42).standard_normal((200, 3)))
        vine = CVineCopula()
        vine.fit(u, method='mle')
        edge = vine.edges[0][0]
        rng = np.random.default_rng(0)
        r = generate_r_for_sample(edge, 100, rng)
        assert r.shape == (100,)


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


# ══════════════════════════════════════════════════════════════
# Truncation tests
# ══════════════════════════════════════════════════════════════

class TestVineTruncation:
    def test_truncation_level(self, crypto_data_6d):
        vine = CVineCopula()
        vine.fit(crypto_data_6d, method='scar-tm-ou',
                 K=50, tol=0.5, truncation_level=1)
        for j, tree_edges in enumerate(vine.edges):
            for edge in tree_edges:
                if j >= 1:
                    assert edge.fit_result.method.upper() == 'MLE'

    def test_min_edge_logL(self, crypto_data_6d):
        vine = CVineCopula()
        vine.fit(crypto_data_6d, method='scar-tm-ou',
                 K=50, tol=0.5, min_edge_logL=50)
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

    def test_predict_given_dynamic_edges_not_implemented(self):
        d = 3
        u = pobs(np.random.default_rng(11).standard_normal((200, d)))
        vine = CVineCopula()
        vine.fit(u, method='scar-m-ou', tol=0.5)
        with pytest.raises(NotImplementedError):
            vine.predict(50, given={0: 0.5})

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

    def test_predict_given_prefix_with_scar_tm(self):
        d = 3
        u = pobs(np.random.default_rng(15).standard_normal((180, d)))
        vine = CVineCopula()
        vine.fit(u, method='scar-tm-ou', K=40, tol=0.5)
        samples = vine.predict(30, given={0: 0.35}, K=40, grid_range=5.0)
        assert samples.shape == (30, d)
        np.testing.assert_allclose(samples[:, 0], 0.35)
        assert np.all((samples[:, 1:] > 0) & (samples[:, 1:] < 1))

    def test_predict_given_non_prefix_with_scar_tm(self):
        d = 3
        u = pobs(np.random.default_rng(16).standard_normal((180, d)))
        vine = CVineCopula()
        vine.fit(u, method='scar-tm-ou', K=40, tol=0.5)
        samples = vine.predict(20, given={1: 0.55}, K=40, grid_range=5.0,
                               horizon='current')
        assert samples.shape == (20, d)
        np.testing.assert_allclose(samples[:, 1], 0.55)
        assert np.all((samples[:, [0, 2]] > 0) & (samples[:, [0, 2]] < 1))

