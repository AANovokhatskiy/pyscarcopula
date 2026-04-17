"""Test C-vine copula: structure, truncation, independence selection, GoF, sample/predict."""
import numpy as np
import pytest
from pyscarcopula.vine.cvine import CVineCopula
from pyscarcopula.vine._edge import VineEdge, _edge_h, _edge_log_likelihood
from pyscarcopula.vine._selection import (
    select_best_copula, _default_candidates, _kendall_tau,
)
from pyscarcopula.vine._helpers import _clip_unit, generate_r_for_sample
from pyscarcopula.vine._conditional_rvine import (
    _copula_h_inverse_scalar,
    _copula_h_scalar,
    _copula_pdf_scalar,
    _sample_from_tabulated_weight,
    sample_rvine_conditional_with_r,
)
from pyscarcopula.copula.clayton import ClaytonCopula
from pyscarcopula.copula.elliptical import BivariateGaussianCopula
from pyscarcopula.copula.frank import FrankCopula
from pyscarcopula.copula.gumbel import GumbelCopula
from pyscarcopula.copula.independent import IndependentCopula
from pyscarcopula.copula.joe import JoeCopula
from pyscarcopula.stattests import gof_test
from pyscarcopula._types import MLEResult
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

    def test_conditional_structure_prioritizes_conditioning_edge(self):
        rng = np.random.default_rng(2201)
        z = rng.standard_normal((300, 4))
        z[:, 2] = z[:, 0] + 0.05 * rng.standard_normal(300)
        z[:, 3] = z[:, 1] + 0.05 * rng.standard_normal(300)
        u = pobs(z)

        vine = RVineCopula(
            structure_mode='conditional',
            conditional_vars={0, 1},
            conditional_structure_policy='priority',
            candidates=[IndependentCopula],
        )
        vine.fit(u, method='mle')

        first_tree_pairs = {
            frozenset({v1, v2})
            for v1, v2, _ in vine.trees[0]
        }
        assert frozenset({0, 1}) in first_tree_pairs
        assert validate_rvine_matrix(vine._structure)
        assert vine.is_conditioning_optimized_for({0: 0.2, 1: 0.8})
        assert vine.is_conditioning_optimized_for({0: 0.2})
        assert not vine.is_conditioning_optimized_for({2: 0.2})

    def test_dissmann_matrix_conversion_failure_falls_back(self, monkeypatch):
        import pyscarcopula.vine.rvine as rvine_module

        original = rvine_module._trees_to_matrix
        calls = {'n': 0}

        def fail_once(*args, **kwargs):
            calls['n'] += 1
            if calls['n'] == 1:
                raise RuntimeError("synthetic conversion failure")
            return original(*args, **kwargs)

        monkeypatch.setattr(rvine_module, '_trees_to_matrix', fail_once)
        u = pobs(np.random.default_rng(2210).standard_normal((180, 5)))
        vine = RVineCopula(candidates=[IndependentCopula])

        with pytest.warns(RuntimeWarning, match='Dissmann R-vine structure'):
            vine.fit(u, method='mle')

        assert calls['n'] == 1
        assert vine.edges is not None
        assert vine.conditional_structure_status is None
        assert validate_rvine_matrix(vine._structure)
        samples = vine.predict(8)
        assert samples.shape == (8, 5)

    def test_conditional_structure_predict_fixes_given_columns(self):
        d = 5
        u = pobs(np.random.default_rng(2202).standard_normal((260, d)))
        vine = RVineCopula(
            structure_mode='conditional',
            conditional_vars={0, 3},
            candidates=[IndependentCopula],
        )
        with pytest.warns(RuntimeWarning, match='posterior dimension'):
            vine.fit(u, method='mle')

        samples = vine.predict(
            12, given={0: 0.25, 3: 0.75}, quad_order=3,
            conditional_method='grid')
        assert samples.shape == (12, d)
        np.testing.assert_allclose(samples[:, 0], 0.25)
        np.testing.assert_allclose(samples[:, 3], 0.75)
        assert np.all((samples[:, [1, 2, 4]] > 0)
                      & (samples[:, [1, 2, 4]] < 1))

    def test_conditional_plan_reports_posterior_workload(self):
        d = 4
        rng = np.random.default_rng(2204)
        z = rng.standard_normal((240, d))
        z[:, 2] = z[:, 0] + 0.05 * rng.standard_normal(240)
        z[:, 3] = z[:, 1] + 0.05 * rng.standard_normal(240)
        u = pobs(z)
        vine = RVineCopula(
            structure_mode='conditional',
            conditional_vars={0, 1},
            conditional_structure_policy='priority',
            candidates=[IndependentCopula],
        )
        vine.fit(u, method='mle')

        plan = vine.conditional_plan({0: 0.2, 1: 0.8}, quad_order=3)

        assert plan['given_vars'] == (0, 1)
        assert plan['optimized_for_given']
        assert plan['posterior_dim'] == len(plan['posterior_indices'])
        assert plan['graph_feasible'] == (plan['posterior_dim'] == 0)
        assert len(plan['graph_steps']) == d
        assert all('idx' in step and 'action' in step and 'posterior' in step
                   for step in plan['graph_steps'])
        assert plan['posterior_vars'] == tuple(
            plan['order_vars'][idx] for idx in plan['posterior_indices'])
        assert plan['structure_status'] == 'priority'
        if plan['posterior_dim']:
            assert plan['joint_grid_points'] == 3 ** plan['posterior_dim']
        else:
            assert plan['joint_grid_points'] == 0

    def test_conditional_structure_cvine_fallback_keeps_optimization(self):
        d = 6
        u = pobs(np.random.default_rng(2206).standard_normal((250, d)))
        vine = RVineCopula(
            structure_mode='conditional',
            conditional_vars={0, 3},
            candidates=[IndependentCopula],
        )

        with pytest.warns(RuntimeWarning, match='conditional C-vine'):
            vine.fit(u, method='mle')

        plan = vine.conditional_plan({0: 0.2, 3: 0.8}, quad_order=4)
        assert vine.conditional_structure_status == 'cvine_fallback'
        assert plan['structure_status'] == 'cvine_fallback'
        assert plan['optimized_for_given']
        assert plan['posterior_dim'] == 0
        assert plan['joint_grid_points'] == 0

    def test_predict_given_graph_method_uses_no_posterior_plan(self):
        d = 6
        u = pobs(np.random.default_rng(2208).standard_normal((250, d)))
        vine = RVineCopula(
            structure_mode='conditional',
            conditional_vars={0, 3},
            candidates=[IndependentCopula],
        )

        with pytest.warns(RuntimeWarning):
            vine.fit(u, method='mle')

        plan = vine.conditional_plan({0: 0.2, 3: 0.8}, quad_order=4)
        assert plan['posterior_dim'] == 0
        assert plan['graph_feasible']
        assert not any(step['posterior'] for step in plan['graph_steps'])

        samples = vine.predict(
            16, given={0: 0.2, 3: 0.8},
            conditional_method='graph')
        assert samples.shape == (16, d)
        np.testing.assert_allclose(samples[:, 0], 0.2)
        np.testing.assert_allclose(samples[:, 3], 0.8)
        assert np.all((samples[:, [1, 2, 4, 5]] > 0)
                      & (samples[:, [1, 2, 4, 5]] < 1))

    def test_predict_given_graph_method_uses_higher_tree_executor(self):
        d = 4
        rng = np.random.default_rng(2209)
        z0 = rng.standard_normal(260)
        z1 = 0.7 * z0 + 0.3 * rng.standard_normal(260)
        z2 = 0.5 * z0 + 0.4 * z1 + 0.3 * rng.standard_normal(260)
        z3 = 0.4 * z1 + 0.5 * z2 + 0.3 * rng.standard_normal(260)
        u = pobs(np.column_stack((z0, z1, z2, z3)))
        vine = RVineCopula(
            structure=cvine_structure(d),
            candidates=[BivariateGaussianCopula],
        )
        vine.fit(u, method='mle')

        plan = vine.conditional_plan({0: 0.2}, quad_order=4)
        flex_plan = vine.flexible_graph_plan({0: 0.2})
        assert plan['posterior_dim'] > 0
        assert not plan['graph_feasible']
        assert any(step['posterior'] for step in plan['graph_steps'])
        assert flex_plan['complete']
        assert not flex_plan['higher_tree_independent']
        assert flex_plan['sampleable']
        assert any(
            step['action'] == 'sample_pseudo'
            for step in flex_plan['steps'])

        samples = vine.predict(
            8, given={0: 0.2},
            conditional_method='graph')
        assert samples.shape == (8, d)
        np.testing.assert_allclose(samples[:, 0], 0.2)
        assert np.all((samples[:, 1:] > 0) & (samples[:, 1:] < 1))

    @pytest.mark.parametrize(
        "copula_cls,param,rotation",
        [
            (ClaytonCopula, 0.8, 90),
            (ClaytonCopula, 0.8, 270),
            (GumbelCopula, 1.4, 90),
            (JoeCopula, 1.4, 270),
        ],
    )
    def test_graph_higher_tree_executor_handles_rotations(
            self, copula_cls, param, rotation):
        d = 4
        u = pobs(np.random.default_rng(2214).standard_normal((260, d)))
        vine = RVineCopula(
            structure=cvine_structure(d),
            candidates=[IndependentCopula],
        )
        vine.fit(u, method='mle')

        for edge in vine.edges.values():
            copula = copula_cls(rotate=rotation)
            edge.copula = copula
            edge.fit_result = MLEResult(
                log_likelihood=0.0,
                method='MLE',
                copula_name=copula.name,
                success=True,
                copula_param=param,
            )

        n = 24
        r_all = {
            key: np.full(n, param, dtype=np.float64)
            for key in vine.edges
        }
        plan = vine.flexible_graph_plan({0: 0.35})
        assert plan['sampleable']
        assert any(step['action'] == 'sample_pseudo'
                   for step in plan['steps'])

        samples = sample_rvine_conditional_with_r(
            vine, n, r_all, {0: 0.35}, np.random.default_rng(2215),
            conditional_method='graph')
        assert samples.shape == (n, d)
        np.testing.assert_allclose(samples[:, 0], 0.35)
        assert np.all(np.isfinite(samples))
        assert np.all((samples[:, 1:] > 0) & (samples[:, 1:] < 1))

    def test_graph_higher_tree_executor_accepts_dynamic_r_arrays(self):
        d = 4
        u = pobs(np.random.default_rng(2216).standard_normal((260, d)))
        vine = RVineCopula(
            structure=cvine_structure(d),
            candidates=[BivariateGaussianCopula],
        )
        vine.fit(u, method='mle')

        n = 32
        r_path = np.linspace(0.15, 0.65, n, dtype=np.float64)
        r_all = {key: r_path.copy() for key in vine.edges}
        plan = vine.flexible_graph_plan({0: 0.45})
        assert plan['sampleable']
        assert any(step['action'] == 'sample_pseudo'
                   for step in plan['steps'])

        samples = sample_rvine_conditional_with_r(
            vine, n, r_all, {0: 0.45}, np.random.default_rng(2217),
            conditional_method='graph')
        assert samples.shape == (n, d)
        np.testing.assert_allclose(samples[:, 0], 0.45)
        assert np.all(np.isfinite(samples))
        assert np.all((samples[:, 1:] > 0) & (samples[:, 1:] < 1))

    def test_flexible_graph_rejects_multi_given_posterior_pattern(self):
        d = 6
        u = pobs(np.random.default_rng(2218).standard_normal((260, d)))
        vine = RVineCopula(
            structure=cvine_structure(d),
            candidates=[BivariateGaussianCopula],
        )
        vine.fit(u, method='mle')

        given = {0: 0.2, 3: 0.8}
        matrix_plan = vine.conditional_plan(given, quad_order=4)
        flex_plan = vine.flexible_graph_plan(given)

        assert matrix_plan['posterior_dim'] > 0
        assert flex_plan['complete']
        assert not flex_plan['flexible_given_supported']
        assert not flex_plan['sampleable']
        with pytest.raises(ValueError, match='flexible_graph_plan'):
            vine.predict(4, given=given, conditional_method='graph')

    def test_flexible_graph_plan_reaches_tree0_connected_bases(self):
        d = 5
        u = pobs(np.random.default_rng(2210).standard_normal((240, d)))
        vine = RVineCopula(
            structure=cvine_structure(d),
            candidates=[IndependentCopula],
        )
        vine.fit(u, method='mle')

        plan = vine.flexible_graph_plan({0: 0.5})

        assert plan['complete']
        assert plan['higher_tree_independent']
        assert plan['sampleable']
        assert plan['missing_base_vars'] == ()
        assert plan['sampled_base_vars'] == tuple(range(d))
        assert any(step['action'] == 'sample_base' for step in plan['steps'])

    def test_predict_given_graph_method_uses_flexible_tree0_executor(self):
        d = 5
        u = pobs(np.random.default_rng(2213).standard_normal((240, d)))
        vine = RVineCopula(
            structure=cvine_structure(d),
            candidates=[IndependentCopula],
        )
        vine.fit(u, method='mle')

        matrix_plan = vine.conditional_plan({0: 0.5}, quad_order=4)
        flex_plan = vine.flexible_graph_plan({0: 0.5})
        assert matrix_plan['posterior_dim'] > 0
        assert flex_plan['sampleable']

        samples = vine.predict(
            20, given={0: 0.5}, conditional_method='graph')
        assert samples.shape == (20, d)
        np.testing.assert_allclose(samples[:, 0], 0.5)
        assert np.all((samples[:, 1:] > 0) & (samples[:, 1:] < 1))

    def test_flexible_graph_plan_reports_frontier_shape(self):
        d = 5
        u = pobs(np.random.default_rng(2211).standard_normal((240, d)))
        vine = RVineCopula()
        vine.fit(u, method='mle')

        plan = vine.flexible_graph_plan({0: 0.5})

        assert plan['given_vars'] == (0,)
        assert 'known_nodes' in plan
        assert 'higher_tree_frontier' in plan
        assert isinstance(plan['complete'], bool)
        assert isinstance(plan['steps'], tuple)

    def test_flexible_graph_plan_validates_given(self):
        u = pobs(np.random.default_rng(2212).standard_normal((200, 4)))
        vine = RVineCopula()
        vine.fit(u, method='mle')

        with pytest.raises(ValueError):
            vine.flexible_graph_plan({0: 1.0})

    def test_conditional_structure_policy_rejects_worse_priority_plan(self):
        u = pobs(np.random.default_rng(2207).standard_normal((250, 6)))
        vine = RVineCopula(
            structure_mode='conditional',
            conditional_vars={0, 2},
            candidates=[IndependentCopula],
        )

        with pytest.warns(RuntimeWarning, match='posterior dimension'):
            vine.fit(u, method='mle')

        plan = vine.conditional_plan({0: 0.2, 2: 0.8}, quad_order=4)
        assert vine.conditional_structure_status == 'cvine_fallback'
        assert plan['optimized_for_given']
        assert plan['posterior_dim'] == 0

    def test_conditional_plan_validates_given(self):
        u = pobs(np.random.default_rng(2205).standard_normal((200, 4)))
        vine = RVineCopula()
        vine.fit(u, method='mle')

        with pytest.raises(ValueError):
            vine.conditional_plan({4: 0.5})

    def test_conditional_structure_requires_conditional_mode(self):
        with pytest.raises(ValueError):
            RVineCopula(conditional_vars={0, 1})

    def test_conditional_structure_rejects_unknown_policy(self):
        with pytest.raises(ValueError):
            RVineCopula(
                structure_mode='conditional',
                conditional_vars={0, 1},
                conditional_structure_policy='unknown',
            )

    def test_conditional_structure_rejects_explicit_structure(self):
        with pytest.raises(ValueError):
            RVineCopula(
                structure=cvine_structure(4),
                structure_mode='conditional',
                conditional_vars={0, 1},
            )

    def test_conditional_structure_validates_variables_at_fit(self):
        u = pobs(np.random.default_rng(2203).standard_normal((200, 4)))
        vine = RVineCopula(
            structure_mode='conditional',
            conditional_vars={0, 4},
        )
        with pytest.raises(ValueError):
            vine.fit(u, method='mle')


class TestRVineSamplePredict:
    def test_tabulated_weight_sampler_avoids_endpoints(self):
        def weight_fn(w):
            if w <= 0.0 or w >= 1.0:
                raise ZeroDivisionError
            return 1.0

        rng = np.random.default_rng(123)
        sample = _sample_from_tabulated_weight(weight_fn, rng, 33)
        assert 0.0 <= sample <= 1.0

    def test_conditional_scalar_fast_calls_match_class_methods(self):
        copulas = [
            (ClaytonCopula, 0.7, (0, 90, 180, 270)),
            (FrankCopula, 2.0, (0,)),
            (GumbelCopula, 1.5, (0, 90, 180, 270)),
            (IndependentCopula, 0.0, (0,)),
            (JoeCopula, 1.5, (0, 90, 180, 270)),
            (BivariateGaussianCopula, 0.4, (0,)),
        ]
        points = [(0.37, 0.62), (0.13, 0.91), (0.82, 0.21)]

        for copula_cls, r, rotations in copulas:
            for rotation in rotations:
                copula = copula_cls(rotate=rotation)
                for u, v in points:
                    ra = np.array([r], dtype=np.float64)
                    ua = np.array([u], dtype=np.float64)
                    va = np.array([v], dtype=np.float64)
                    assert np.isclose(
                        _copula_pdf_scalar(copula, u, v, r),
                        copula.pdf(ua, va, ra)[0])
                    assert np.isclose(
                        _copula_h_scalar(copula, u, v, r),
                        copula.h(ua, va, ra)[0])
                    assert np.isclose(
                        _copula_h_inverse_scalar(copula, u, v, r),
                        copula.h_inverse(ua, va, ra)[0])

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

    def test_predict_given_with_mle(self):
        d = 4
        u = pobs(np.random.default_rng(17).standard_normal((220, d)))
        vine = RVineCopula()
        vine.fit(u, method='mle')
        samples = vine.predict(50, given={1: 0.4})
        assert samples.shape == (50, d)
        np.testing.assert_allclose(samples[:, 1], 0.4)
        assert np.all((samples[:, [0, 2, 3]] > 0) & (samples[:, [0, 2, 3]] < 1))

    def test_predict_given_accepts_quad_order(self):
        d = 4
        u = pobs(np.random.default_rng(171).standard_normal((220, d)))
        vine = RVineCopula()
        vine.fit(u, method='mle')
        samples = vine.predict(5, given={0: 0.2, 2: 0.7}, quad_order=4)
        assert samples.shape == (5, d)
        np.testing.assert_allclose(samples[:, 0], 0.2)
        np.testing.assert_allclose(samples[:, 2], 0.7)

    def test_predict_given_multi_condition_fast_path(self):
        d = 6
        u = pobs(np.random.default_rng(172).standard_normal((240, d)))
        vine = RVineCopula(structure=cvine_structure(d))
        vine.fit(u, method='mle')
        samples = vine.predict(
            4, given={0: 0.2, 3: 0.8}, quad_order=3,
            conditional_method='grid')
        assert samples.shape == (4, d)
        np.testing.assert_allclose(samples[:, 0], 0.2)
        np.testing.assert_allclose(samples[:, 3], 0.8)
        assert np.all((samples[:, [1, 2, 4, 5]] > 0)
                      & (samples[:, [1, 2, 4, 5]] < 1))

    def test_predict_given_exact_method(self):
        d = 4
        u = pobs(np.random.default_rng(173).standard_normal((220, d)))
        vine = RVineCopula(structure=cvine_structure(d))
        vine.fit(u, method='mle')
        samples = vine.predict(
            3, given={0: 0.2, 2: 0.7}, quad_order=2,
            conditional_method='exact')
        assert samples.shape == (3, d)
        np.testing.assert_allclose(samples[:, 0], 0.2)
        np.testing.assert_allclose(samples[:, 2], 0.7)

    def test_predict_given_invalid_conditional_method_raises(self):
        d = 4
        u = pobs(np.random.default_rng(174).standard_normal((220, d)))
        vine = RVineCopula()
        vine.fit(u, method='mle')
        with pytest.raises(ValueError):
            vine.predict(3, given={0: 0.2}, conditional_method='unknown')

    def test_predict_given_with_gas(self):
        d = 4
        u = pobs(np.random.default_rng(18).standard_normal((220, d)))
        vine = RVineCopula()
        vine.fit(u, method='gas')
        samples = vine.predict(30, given={2: 0.65})
        assert samples.shape == (30, d)
        np.testing.assert_allclose(samples[:, 2], 0.65)
        assert np.all((samples[:, [0, 1, 3]] > 0) & (samples[:, [0, 1, 3]] < 1))

    def test_predict_given_with_scar_tm(self):
        d = 3
        u = pobs(np.random.default_rng(19).standard_normal((180, d)))
        vine = RVineCopula()
        vine.fit(u, method='scar-tm-ou', truncation_level=1, tol=0.5, K=40)
        samples = vine.predict(20, given={0: 0.3}, K=40, horizon='current')
        assert samples.shape == (20, d)
        np.testing.assert_allclose(samples[:, 0], 0.3)
        assert np.all((samples[:, 1:] > 0) & (samples[:, 1:] < 1))

    def test_scar_fit_preserves_grid_settings(self):
        d = 3
        rng = np.random.default_rng(191)
        z0 = rng.standard_normal(160)
        z1 = 0.75 * z0 + 0.35 * rng.standard_normal(160)
        z2 = 0.65 * z1 + 0.35 * rng.standard_normal(160)
        u = pobs(np.column_stack((z0, z1, z2)))
        vine = RVineCopula(candidates=[BivariateGaussianCopula])
        vine.fit(
            u,
            method='scar-tm-ou',
            truncation_level=1,
            K=25,
            grid_range=4.0,
            tol=0.8,
        )

        dynamic_edges = [
            edge for edge in vine.edges.values()
            if edge.fit_result.method.upper() == 'SCAR-TM-OU'
        ]
        assert dynamic_edges
        for edge in dynamic_edges:
            assert edge.fit_result.K == 25
            assert edge.fit_result.grid_range == 4.0

    def test_predict_given_unsupported_method_raises(self):
        d = 3
        u = pobs(np.random.default_rng(20).standard_normal((180, d)))
        vine = RVineCopula()
        vine.fit(u, method='scar-m-ou', truncation_level=1, tol=0.5)
        with pytest.raises(NotImplementedError):
            vine.predict(20, given={0: 0.4})


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
