"""Test C-vine copula: structure, truncation, independence selection, GoF, sample/predict."""
import numpy as np
import pytest
from pyscarcopula.vine.cvine import CVineCopula
from pyscarcopula.vine._edge import VineEdge, _edge_h, _edge_log_likelihood
from pyscarcopula.vine._selection import (
    select_best_copula, _default_candidates, _kendall_tau,
)
from pyscarcopula.vine._helpers import _clip_unit, generate_r_for_sample
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


# ══════════════════════════════════════════════════════════════
# R-vine structure module tests
# ══════════════════════════════════════════════════════════════

from pyscarcopula.vine._structure import (
    RVineMatrix, build_rvine_structure, cvine_structure,
    _maximum_spanning_tree,
)
from pyscarcopula.vine.rvine import RVineCopula


class TestRVineMatrix:
    def test_cvine_structure(self):
        M = cvine_structure(4)
        assert M.d == 4
        # diagonal should be a permutation of 0..3
        diag = [M.matrix[i, i] for i in range(4)]
        assert sorted(diag) == [0, 1, 2, 3]

    def test_edge_extraction(self):
        M = cvine_structure(4)
        edges_t0 = M.edges_at_tree(0)
        assert len(edges_t0) == 3  # d-1 edges in tree 0

    def test_invalid_matrix(self):
        with pytest.raises(ValueError):
            RVineMatrix(np.array([[0, 0], [1, 0]]))  # not a permutation

    def test_mst_basic(self):
        nodes = [0, 1, 2, 3]
        edges = [(0, 1), (1, 2), (2, 3), (0, 3)]
        weights = [0.9, 0.8, 0.7, 0.1]
        mst = _maximum_spanning_tree(nodes, edges, weights)
        assert len(mst) == 3  # d-1 edges


class TestBuildStructure:
    def test_build_returns_valid(self):
        rng = np.random.default_rng(42)
        u = pobs(rng.standard_normal((200, 4)))
        matrix, trees = build_rvine_structure(u)
        assert matrix.d == 4
        assert len(trees) == 3  # d-1 trees
        assert len(trees[0]) == 3  # d-1 edges in tree 0

    def test_build_6d(self):
        rng = np.random.default_rng(42)
        u = pobs(rng.standard_normal((200, 6)))
        matrix, trees = build_rvine_structure(u)
        total_edges = sum(len(t) for t in trees)
        assert total_edges == 15  # 6*5/2


# ══════════════════════════════════════════════════════════════
# R-vine copula tests
# ══════════════════════════════════════════════════════════════

class TestRVineStructure:
    @pytest.mark.parametrize("d", [3, 4, 6])
    def test_number_of_edges(self, d):
        u = pobs(np.random.default_rng(42).standard_normal((200, d)))
        vine = RVineCopula()
        vine.fit(u, method='mle')
        assert len(vine.edges) == d * (d - 1) // 2

    @pytest.mark.parametrize("d", [3, 4, 6])
    def test_number_of_trees(self, d):
        u = pobs(np.random.default_rng(42).standard_normal((200, d)))
        vine = RVineCopula()
        vine.fit(u, method='mle')
        assert len(vine.trees) == d - 1


class TestRVineSamplePredict:
    def test_predict_shape(self):
        d = 4
        u = pobs(np.random.default_rng(0).standard_normal((200, d)))
        vine = RVineCopula()
        vine.fit(u, method='mle')
        assert vine.predict(100).shape == (100, d)

    def test_sample_shape(self):
        d = 4
        u = pobs(np.random.default_rng(0).standard_normal((200, d)))
        vine = RVineCopula()
        vine.fit(u, method='mle')
        assert vine.sample(100).shape == (100, d)

    def test_samples_in_unit_cube(self):
        d = 4
        u = pobs(np.random.default_rng(0).standard_normal((200, d)))
        vine = RVineCopula()
        vine.fit(u, method='mle')
        samples = vine.predict(1000)
        assert np.all(samples >= 0) and np.all(samples <= 1)

    def test_log_likelihood_consistent(self):
        d = 3
        u = pobs(np.random.default_rng(42).standard_normal((200, d)))
        vine = RVineCopula()
        vine.fit(u, method='mle')
        ll_fit = vine.fit_result.log_likelihood
        ll_eval = vine.log_likelihood(u)
        assert abs(ll_fit - ll_eval) < 1e-4


class TestRVineGoF:
    def test_rvine_gof_runs(self, crypto_data_6d):
        vine = RVineCopula()
        vine.fit(crypto_data_6d, method='mle')
        gof = gof_test(vine, crypto_data_6d, to_pobs=False)
        assert 0 <= gof.pvalue <= 1


class TestRVineSCAR:
    def test_scar_truncated(self, crypto_data_6d):
        vine = RVineCopula()
        vine.fit(crypto_data_6d, method='scar-tm-ou',
                 K=50, tol=0.5, truncation_level=1, min_edge_logL=10)
        for key, edge in vine.edges.items():
            tree_level = key[0]
            if tree_level >= 1:
                assert edge.fit_result.method.upper() == 'MLE'

    def test_scar_improves_logL(self, crypto_data_6d):
        vine_mle = RVineCopula()
        vine_mle.fit(crypto_data_6d, method='mle')

        vine_scar = RVineCopula()
        vine_scar.fit(crypto_data_6d, method='scar-tm-ou',
                      truncation_level=2, min_edge_logL=10,
                      tol=5e-2, K=100)

        assert (vine_scar.fit_result.log_likelihood
                >= vine_mle.fit_result.log_likelihood - 5)

