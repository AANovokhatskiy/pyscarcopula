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
from pyscarcopula.vine.rvine import RVineCopula, _build_matrix_edge_map
from pyscarcopula.vine._structure import (
    RVineMatrix, cvine_structure,
    _maximum_spanning_tree, validate_rvine_matrix,
)

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


# ══════════════════════════════════════════════════════════════
# R-vine matrix validation tests
# ══════════════════════════════════════════════════════════════

class TestRVineMatrixValidation:
    def test_cvine_structure_valid(self):
        """C-vine structure should always satisfy the proximity condition."""
        for d in [3, 4, 5, 6]:
            M = cvine_structure(d)
            assert validate_rvine_matrix(M), f"C-vine d={d} failed validation"

    @pytest.mark.parametrize("d", [3, 4, 5, 6])
    def test_fit_produces_valid_matrix(self, d):
        """RVineCopula.fit() should produce valid R-vine matrices."""
        rng = np.random.default_rng(42)
        u = pobs(rng.standard_normal((200, d)))
        vine = RVineCopula()
        vine.fit(u, method='mle')
        assert validate_rvine_matrix(vine._structure), \
            f"RVineCopula.fit() produced invalid matrix for d={d}"

    def test_known_valid_matrix(self):
        """A hand-constructed valid 4x4 C-vine matrix should pass."""
        M = np.array([
            [0, 0, 0, 0],
            [1, 1, 0, 0],
            [2, 2, 2, 0],
            [3, 3, 3, 3],
        ])
        assert validate_rvine_matrix(M)

    def test_known_invalid_matrix(self):
        """A matrix violating proximity should fail validation.
        M[2,0]=2 but column 1 rows 1..2 = {1, 3} — 2 not present."""
        M = np.array([
            [0, 0, 0, 0],
            [1, 1, 0, 0],
            [2, 3, 2, 0],
            [3, 2, 3, 3],
        ])
        assert not validate_rvine_matrix(M)

    def test_truncated_rvine_matrix_valid(self, crypto_data_6d):
        """Truncated R-vine must produce valid matrix without warnings."""
        import warnings
        vine = RVineCopula()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            vine.fit(crypto_data_6d, method='mle', truncation_level=1)
        assert validate_rvine_matrix(vine._structure)


# ═════════════════════════════════════════════════════════════
#  R-vine invariant tests
# ═════════════════════════════════════════════════════════════

class TestRVineRoundtrip:
    @pytest.mark.parametrize("d", [4, 5, 6])
    def test_trees_matrix_roundtrip_exact(self, d):
        rng = np.random.default_rng(123)
        u = pobs(rng.standard_normal((250, d)))

        vine = RVineCopula()
        vine.fit(u, method="mle")

        assert validate_rvine_matrix(vine._structure)

        for tl, tree_edges in enumerate(vine.trees):
            tree_edge_sets = {
                (frozenset({v1, v2}), frozenset(cond))
                for v1, v2, cond in tree_edges
            }
            matrix_edge_sets = {
                (frozenset({v1, v2}), frozenset(cond))
                for v1, v2, cond in vine._structure.edges_at_tree(tl)
            }
            assert tree_edge_sets == matrix_edge_sets, \
                f"Roundtrip mismatch at tree {tl}"


class TestRVineEdgeMap:
    @pytest.mark.parametrize("d", [4, 5, 6])
    def test_matrix_edge_map_is_complete_and_consistent(self, d):
        rng = np.random.default_rng(321)
        u = pobs(rng.standard_normal((250, d)))

        vine = RVineCopula()
        vine.fit(u, method="mle")

        edge_map = _build_matrix_edge_map(vine._structure, vine.trees, vine.edges)
        expected = d * (d - 1) // 2
        assert len(edge_map) == expected

        for t in range(d - 1):
            for s, (v1, v2, cond) in enumerate(vine._structure.edges_at_tree(t)):
                key = edge_map[(t, s)]
                vv1, vv2, ccond = vine.trees[key[0]][key[1]]
                assert frozenset({v1, v2}) == frozenset({vv1, vv2})
                assert frozenset(cond) == frozenset(ccond)


class TestRVineGivenStructure:
    def test_fit_with_given_structure_path(self):
        rng = np.random.default_rng(777)
        u = pobs(rng.standard_normal((300, 4)))

        vine0 = RVineCopula()
        vine0.fit(u, method="mle")

        M = vine0._structure.matrix
        structure = RVineMatrix(M)

        vine1 = RVineCopula(structure=structure)
        vine1.fit(u, method="mle")

        assert validate_rvine_matrix(vine1._structure)
        assert np.isfinite(vine1.fit_result.log_likelihood)

        ll_eval = vine1.log_likelihood(u)
        assert np.isfinite(ll_eval)
        assert abs(ll_eval - vine1.fit_result.log_likelihood) < 1e-4

        gof = gof_test(vine1, u, to_pobs=False)
        assert 0 <= gof.pvalue <= 1


class TestRVineGoFRegression:
    def test_gof_on_model_samples_not_systematically_zero(self):
        rng = np.random.default_rng(2024)
        u = pobs(rng.standard_normal((350, 5)))

        vine = RVineCopula()
        vine.fit(u, method="mle")

        pvals = []
        for seed in range(10):
            # use fresh model samples; sample() uses internal RNG, so just repeat
            u_sim = vine.sample(400)
            gof = gof_test(vine, u_sim, to_pobs=False)
            pvals.append(gof.pvalue)

        pvals = np.asarray(pvals)
        assert np.all((0 <= pvals) & (pvals <= 1))
        # weak but useful regression guard against the old "everything ~ 0" bug
        assert pvals.mean() > 0.15
        assert np.sum(pvals < 1e-6) == 0


class TestRVineRefitSanity:
    def test_sample_refit_gof_reasonable_on_average(self):
        rng = np.random.default_rng(999)
        u = pobs(rng.standard_normal((300, 4)))

        base = RVineCopula()
        base.fit(u, method="mle")

        pvals = []
        for _ in range(12):
            u_sim = base.sample(350)

            refit = RVineCopula()
            refit.fit(u_sim, method="mle")

            assert validate_rvine_matrix(refit._structure)
            assert np.isfinite(refit.fit_result.log_likelihood)

            gof = gof_test(refit, u_sim, to_pobs=False)
            pvals.append(gof.pvalue)

        pvals = np.asarray(pvals)
        assert np.all((0 <= pvals) & (pvals <= 1))
        # soft calibration sanity check; not too strict to avoid flaky CI
        assert pvals.mean() > 0.20

class TestRVineMatrixValidationExtra:
    def test_invalid_matrix_duplicate_in_column(self):
        M = np.array([
            [0, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 2, 2, 0],  # duplicate "1" in column 0
            [3, 3, 3, 3],
        ])
        with pytest.raises(ValueError):
            RVineMatrix(M)

    def test_invalid_matrix_value_out_of_range(self):
        M = np.array([
            [0, 0, 0],
            [1, 1, 0],
            [5, 2, 2],  # out of range for d=3
        ])
        with pytest.raises(ValueError):
            RVineMatrix(M)