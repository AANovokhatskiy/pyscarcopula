"""Unit tests for vine._rvine_dissmann (MLE-only Dissmann pipeline)."""
import numpy as np
import pytest

from pyscarcopula import (
    GumbelCopula, ClaytonCopula, FrankCopula, JoeCopula,
    IndependentCopula, BivariateGaussianCopula,
)
from pyscarcopula._types import GASResult, gas_params
from pyscarcopula._utils import pobs
from pyscarcopula.vine._rvine_dissmann import (
    PairCopula,
    select_rvine,
)
from pyscarcopula.vine._rvine_matrix_builder import (
    build_rvine_matrix_with_edge_map,
    validate_natural_order_matrix,
)


# ═══════════════════════════════════════════════════════════
# Synthetic data helpers
# ═══════════════════════════════════════════════════════════

def _sample_dvine_gumbel(T, d, theta, seed=0):
    """Sample a D-vine of Gumbel copulas (tree-0 path 0-1-...-(d-1)).

    Only tree 0 has dependence (Gumbel(theta)); higher trees use the
    empirical h-propagation, so tree t >= 1 is independent conditional
    on lower trees. Returns a (T, d) pseudo-obs array.
    """
    rng = np.random.default_rng(seed)
    cop = GumbelCopula(rotate=0)
    r = np.full(T, theta, dtype=np.float64)

    u = np.zeros((T, d), dtype=np.float64)
    u[:, 0] = rng.uniform(0, 1, T)
    prev = u[:, 0].copy()
    for j in range(1, d):
        w = rng.uniform(0, 1, T)
        u[:, j] = cop.h_inverse(w, prev, r)
        prev = u[:, j].copy()
    return np.clip(u, 1e-9, 1 - 1e-9)


def _sample_independent(T, d, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 1, (T, d))


# ═══════════════════════════════════════════════════════════
# Basic shape & structural contracts
# ═══════════════════════════════════════════════════════════

class TestShapes:

    def test_output_shapes_d4(self):
        u = _sample_dvine_gumbel(400, 4, theta=2.0, seed=1)
        trees_repr, fitted = select_rvine(u)
        assert len(trees_repr) == 3
        assert len(fitted) == 3
        for t, (level_repr, level_fit) in enumerate(zip(trees_repr, fitted)):
            assert len(level_repr) == 4 - 1 - t
            assert len(level_fit) == 4 - 1 - t
            for edge in level_repr:
                conditioned, conditioning = edge
                assert len(conditioned) == 2
                assert len(conditioning) == t
            for pc in level_fit:
                assert isinstance(pc, PairCopula)

    def test_output_shapes_d2(self):
        u = _sample_dvine_gumbel(300, 2, theta=2.5, seed=2)
        trees_repr, fitted = select_rvine(u)
        assert len(trees_repr) == 1
        assert len(fitted) == 1
        assert len(trees_repr[0]) == 1
        assert len(fitted[0]) == 1

    def test_rejects_1d(self):
        u = np.random.default_rng(0).uniform(0, 1, (100,))
        with pytest.raises(ValueError, match="must be 2D"):
            select_rvine(u)

    def test_rejects_d1(self):
        u = np.random.default_rng(0).uniform(0, 1, (100, 1))
        with pytest.raises(ValueError, match="d >= 2"):
            select_rvine(u)


# ═══════════════════════════════════════════════════════════
# Output must be consumable by the matrix builder
# ═══════════════════════════════════════════════════════════

class TestMatrixBuilderCompatibility:

    @pytest.mark.parametrize("d,seed",
                             [(3, 0), (4, 1), (5, 2), (6, 3),
                              (7, 4), (8, 5), (10, 6)])
    def test_trees_repr_builds_valid_matrix(self, d, seed):
        u = _sample_dvine_gumbel(500, d, theta=2.0, seed=seed)
        trees_repr, fitted = select_rvine(u)
        M, edge_map = build_rvine_matrix_with_edge_map(d, trees_repr)
        assert M.shape == (d, d)
        assert validate_natural_order_matrix(M)
        # Anti-diagonal = peeled-leaf sequence, a permutation of 0..d-1.
        anti_diag = {int(M[d - 1 - c, c]) for c in range(d)}
        assert anti_diag == set(range(d))
        # edge_map must cover every (t, col) position.
        for t in range(d - 1):
            for col in range(d - 1 - t):
                assert (t, col) in edge_map

    def test_edge_map_allows_fitted_lookup(self):
        d = 5
        u = _sample_dvine_gumbel(500, d, theta=2.0, seed=0)
        trees_repr, fitted = select_rvine(u)
        M, edge_map = build_rvine_matrix_with_edge_map(d, trees_repr)

        # Each matrix position (t, col) encodes an edge — look it up in
        # the original trees_repr via edge_map, then in fitted.
        for (t, col), orig_idx in edge_map.items():
            conditioned, conditioning = trees_repr[t][orig_idx]
            pc = fitted[t][orig_idx]
            assert isinstance(pc, PairCopula)
            # Natural-order: leaf at M[d-1-col, col], tree-t other
            # endpoint at M[d-2-col-t, col], conditioning in rows
            # d-1-col-t..d-2-col.
            leaf = int(M[d - 1 - col, col])
            tail = int(M[d - 2 - col - t, col])
            assert {leaf, tail} == set(int(x) for x in conditioned)
            cond_from_matrix = {
                int(M[r, col])
                for r in range(d - 1 - col - t, d - 1 - col)
            }
            assert cond_from_matrix == set(int(x) for x in conditioning)


# ═══════════════════════════════════════════════════════════
# Fit content: what copulas get selected
# ═══════════════════════════════════════════════════════════

class TestFitContent:

    def test_independent_data_fits_near_zero_logL(self):
        """All-independent data should yield tiny per-edge logL (or Independent)."""
        u = _sample_independent(800, 4, seed=0)
        trees_repr, fitted = select_rvine(u)
        for level in fitted:
            for pc in level:
                # Either Independent, or a parametric fit with very small logL.
                if isinstance(pc.copula, IndependentCopula):
                    assert pc.log_likelihood == 0.0
                    assert pc.param == 0.0
                else:
                    assert pc.log_likelihood < 15.0  # loose but non-trivial

    def test_strong_dependence_picks_parametric(self):
        """Gumbel D-vine tree-0 edges should fit a parametric copula."""
        u = _sample_dvine_gumbel(800, 4, theta=3.0, seed=0)
        trees_repr, fitted = select_rvine(u)
        # At least one tree-0 edge must be non-independent with strong logL.
        tree0_nonindep = [
            pc for pc in fitted[0]
            if not isinstance(pc.copula, IndependentCopula)
        ]
        assert len(tree0_nonindep) >= 1
        best_ll = max(pc.log_likelihood for pc in tree0_nonindep)
        assert best_ll > 50.0

    def test_paircopula_fields_wellformed(self):
        u = _sample_dvine_gumbel(400, 3, theta=2.0, seed=0)
        _, fitted = select_rvine(u)
        for level in fitted:
            for pc in level:
                assert isinstance(pc.param, float)
                assert isinstance(pc.log_likelihood, float)
                assert isinstance(pc.nfev, int)
                assert isinstance(pc.tau, float)
                assert -1.0 <= pc.tau <= 1.0
                expected_nparams = (
                    0 if isinstance(pc.copula, IndependentCopula) else 1
                )
                assert pc.n_params == expected_nparams


# ═══════════════════════════════════════════════════════════
# Truncation
# ═══════════════════════════════════════════════════════════

class TestTruncation:

    def test_truncation_forces_independent_above_level(self):
        u = _sample_dvine_gumbel(500, 5, theta=3.0, seed=0)
        trees_repr, fitted = select_rvine(
            u,
            truncation_level=2,
            truncation_fill='independent',
        )
        # Levels >= 2 must all be IndependentCopula.
        for t in (2, 3):
            for pc in fitted[t]:
                assert isinstance(pc.copula, IndependentCopula)
                assert pc.log_likelihood == 0.0
                assert pc.param == 0.0
                assert pc.nfev == 0

    def test_truncation_level_zero_makes_all_independent(self):
        u = _sample_dvine_gumbel(400, 4, theta=2.5, seed=0)
        _, fitted = select_rvine(
            u,
            truncation_level=0,
            truncation_fill='independent',
        )
        for level in fitted:
            for pc in level:
                assert isinstance(pc.copula, IndependentCopula)

    def test_truncation_leaves_lower_trees_unfit(self):
        u = _sample_dvine_gumbel(800, 4, theta=3.0, seed=0)
        _, fitted = select_rvine(u, truncation_level=1)
        # Tree 0 should have at least one non-independent fit.
        tree0_nonindep = [
            pc for pc in fitted[0]
            if not isinstance(pc.copula, IndependentCopula)
        ]
        assert len(tree0_nonindep) >= 1


# ═══════════════════════════════════════════════════════════
# min_edge_logL pruning
# ═══════════════════════════════════════════════════════════

class TestMinEdgeLogL:

    def test_high_threshold_prunes_all(self):
        u = _sample_dvine_gumbel(400, 4, theta=2.0, seed=0)
        _, fitted = select_rvine(u, min_edge_logL=1e9)
        for level in fitted:
            for pc in level:
                assert isinstance(pc.copula, IndependentCopula)
                assert pc.log_likelihood == 0.0

    def test_low_threshold_keeps_strong_edges(self):
        u = _sample_dvine_gumbel(800, 4, theta=3.0, seed=0)
        _, fitted_strict = select_rvine(u, min_edge_logL=None)
        _, fitted_loose = select_rvine(u, min_edge_logL=-1e9)
        # A threshold of -inf must keep every fit that passed selection.
        for lvl_s, lvl_l in zip(fitted_strict, fitted_loose):
            for ps, pl in zip(lvl_s, lvl_l):
                assert type(ps.copula) is type(pl.copula)


# ═══════════════════════════════════════════════════════════
# PairCopula.h behaviour
# ═══════════════════════════════════════════════════════════

class TestThreshold:

    def test_high_threshold_prunes_all_before_fit(self):
        u = _sample_dvine_gumbel(400, 4, theta=2.0, seed=0)
        _, fitted = select_rvine(u, threshold=1.1)
        for level in fitted:
            for pc in level:
                assert isinstance(pc.copula, IndependentCopula)
                assert pc.log_likelihood == 0.0
                assert pc.nfev == 0

    def test_threshold_zero_keeps_default_behavior(self):
        u = _sample_dvine_gumbel(600, 4, theta=3.0, seed=0)
        _, fitted_none = select_rvine(u, threshold=None)
        _, fitted_zero = select_rvine(u, threshold=0.0)
        non_none = sum(
            not isinstance(pc.copula, IndependentCopula)
            for level in fitted_none for pc in level
        )
        non_zero = sum(
            not isinstance(pc.copula, IndependentCopula)
            for level in fitted_zero for pc in level
        )
        assert non_zero == non_none


class TestDynamicFallback:

    def test_failed_dynamic_fit_falls_back_to_mle(self, monkeypatch):
        from pyscarcopula.vine import _rvine_dissmann as dissmann

        original_fit_with_strategy = dissmann._fit_with_strategy

        def fake_failed_fit(copula, u_pair, method, config, fit_kwargs):
            if method.lower() == 'gas':
                return GASResult(
                    log_likelihood=-1e6,
                    method='GAS',
                    copula_name=copula.name,
                    success=False,
                    nfev=200,
                    message='forced failure',
                    params=gas_params(0.0, 0.0, 0.95),
                )
            return original_fit_with_strategy(
                copula, u_pair, method, config, fit_kwargs)

        monkeypatch.setattr(dissmann, "_fit_with_strategy", fake_failed_fit)
        u = _sample_dvine_gumbel(300, 3, theta=2.0, seed=0)

        _, fitted = select_rvine(
            u,
            method='gas',
            candidates=[BivariateGaussianCopula],
            threshold=None,
        )

        for level in fitted:
            for pc in level:
                assert pc.fit_result.method == 'MLE'
                assert pc.fit_result.success


class TestPairCopulaH:

    def test_h_independent_is_passthrough(self):
        pc = PairCopula(
            copula=IndependentCopula(),
            param=0.0, log_likelihood=0.0, nfev=0, tau=0.0,
        )
        u = np.linspace(0.1, 0.9, 25)
        v = np.linspace(0.9, 0.1, 25)
        out = pc.h(u, v)
        np.testing.assert_allclose(out, u)
        # Returns a fresh array (not the input).
        assert out is not u

    def test_h_parametric_matches_direct_call(self):
        rng = np.random.default_rng(0)
        u = rng.uniform(0.05, 0.95, 50)
        v = rng.uniform(0.05, 0.95, 50)
        theta = 2.5
        cop = GumbelCopula(rotate=0)
        pc = PairCopula(
            copula=cop, param=theta, log_likelihood=-1.0, nfev=0, tau=0.3,
        )
        expected = cop.h(u, v, np.full(len(u), theta))
        np.testing.assert_allclose(pc.h(u, v), expected)

    def test_h_output_shape(self):
        pc = PairCopula(
            copula=GumbelCopula(rotate=0),
            param=2.0, log_likelihood=0.0, nfev=0, tau=0.2,
        )
        u = np.linspace(0.1, 0.9, 17)
        v = np.linspace(0.2, 0.8, 17)
        out = pc.h(u, v)
        assert out.shape == (17,)
        assert out.dtype == np.float64


# ═══════════════════════════════════════════════════════════
# Criterion plumbing
# ═══════════════════════════════════════════════════════════

class TestCriterion:

    @pytest.mark.parametrize("crit", ["aic", "bic", "loglik"])
    def test_criterion_is_accepted(self, crit):
        u = _sample_dvine_gumbel(300, 4, theta=2.0, seed=0)
        trees_repr, fitted = select_rvine(u, criterion=crit)
        assert len(trees_repr) == 3
        assert len(fitted) == 3

    def test_custom_candidates_subset(self):
        u = _sample_dvine_gumbel(400, 4, theta=2.5, seed=0)
        _, fitted = select_rvine(u, candidates=[GumbelCopula])
        # No edge should fit Clayton/Frank/Joe/Gaussian (only Gumbel or Indep).
        for level in fitted:
            for pc in level:
                cls = type(pc.copula)
                assert cls in (GumbelCopula, IndependentCopula)


# ═══════════════════════════════════════════════════════════
# End-to-end: fitted vine evaluates a finite total loglik
# ═══════════════════════════════════════════════════════════

class TestEndToEnd:

    def test_total_loglik_is_finite(self):
        u = _sample_dvine_gumbel(500, 5, theta=2.5, seed=0)
        _, fitted = select_rvine(u)
        total_ll = sum(pc.log_likelihood for lvl in fitted for pc in lvl)
        assert np.isfinite(total_ll)

    def test_recovers_dvine_path_structure(self):
        """Gumbel D-vine with strong tree-0 dependence → tree-0 edges should
        form a path on the input ordering (each variable appears in exactly
        two edges, except endpoints)."""
        d = 5
        u = _sample_dvine_gumbel(1000, d, theta=3.5, seed=0)
        trees_repr, _ = select_rvine(u)
        tree0_pairs = [conditioned for conditioned, _ in trees_repr[0]]
        degree = {v: 0 for v in range(d)}
        for pair in tree0_pairs:
            for v in pair:
                degree[v] += 1
        # Exactly 2 endpoints (degree 1), rest interior (degree 2).
        assert sum(1 for v, k in degree.items() if k == 1) == 2
        assert sum(1 for v, k in degree.items() if k == 2) == d - 2
