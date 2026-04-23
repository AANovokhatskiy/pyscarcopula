"""Unit tests for the MLE-only RVineCopula (Block 1 refactor)."""
import re

import numpy as np
import pytest
from scipy.stats import kstest, norm

from pyscarcopula import (
    GumbelCopula, ClaytonCopula, FrankCopula, JoeCopula,
    IndependentCopula, BivariateGaussianCopula, RVineCopula,
)
from pyscarcopula._types import (
    GASResult,
    IndependentResult,
    LatentResult,
    MLEResult,
    gas_params,
    ou_params,
)
from pyscarcopula._utils import pobs
from pyscarcopula.api import predict as api_predict
from pyscarcopula.api import sample as api_sample
from pyscarcopula.stattests import gof_test, rvine_rosenblatt_transform
from pyscarcopula.vine._rvine_dissmann import PairCopula
from pyscarcopula.vine._rvine_edges import (
    _edge_h,
    _edge_h_inverse,
    _edge_r_for_predict,
    _edge_r_for_sample,
)
from pyscarcopula.vine._rvine_matrix_builder import validate_natural_order_matrix


# ═══════════════════════════════════════════════════════════
# Synthetic data helpers
# ═══════════════════════════════════════════════════════════

def _sample_dvine_gumbel(T, d, theta, seed=0):
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


def _independent(T, d, seed=0):
    return np.random.default_rng(seed).uniform(0, 1, (T, d))


def _sample_dynamic_gaussian_chain(T, d, seed=0):
    rng = np.random.default_rng(seed)
    cop = BivariateGaussianCopula()
    phase = np.linspace(0.0, 2.0 * np.pi, T)
    r = 0.55 + 0.25 * np.sin(phase)
    u = np.zeros((T, d), dtype=np.float64)
    u[:, 0] = rng.uniform(0.01, 0.99, T)
    for j in range(1, d):
        w = rng.uniform(0.01, 0.99, T)
        sign = -1.0 if j % 2 else 1.0
        u[:, j] = cop.h_inverse(w, u[:, j - 1], sign * r)
    return np.clip(u, 1e-9, 1 - 1e-9)


# ═══════════════════════════════════════════════════════════
# Fit contract: returns self, populates all attributes
# ═══════════════════════════════════════════════════════════

class TestFitContract:

    def test_fit_returns_self(self):
        u = _sample_dvine_gumbel(300, 4, 2.0, seed=0)
        v = RVineCopula()
        assert v.fit(u) is v

    def test_fit_populates_state(self):
        u = _sample_dvine_gumbel(300, 5, 2.0, seed=0)
        v = RVineCopula().fit(u)
        assert v.d == 5
        assert v.matrix.shape == (5, 5)
        assert validate_natural_order_matrix(v.matrix)
        assert len(v.trees) == 4
        # Every matrix edge position must have a PairCopula.
        for t in range(4):
            for col in range(4 - t):
                assert (t, col) in v.pair_copulas
                assert isinstance(v.pair_copulas[(t, col)], PairCopula)
                assert v.pair_copulas[(t, col)].fit_result is not None
                assert isinstance(
                    v.pair_copulas[(t, col)].fit_result,
                    (MLEResult, IndependentResult),
                )

    def test_fit_chainable(self):
        u = _sample_dvine_gumbel(200, 3, 2.0, seed=0)
        s = RVineCopula().fit(u).summary(as_string=True)
        assert "RVineCopula" in s
        assert "log_likelihood" in s

    def test_unfitted_raises(self):
        v = RVineCopula()
        with pytest.raises(RuntimeError, match="call fit"):
            v.log_likelihood()
        with pytest.raises(RuntimeError, match="call fit"):
            _ = v.n_parameters
        with pytest.raises(RuntimeError, match="call fit"):
            _ = v.aic
        with pytest.raises(RuntimeError, match="call fit"):
            v.family_matrix()

    def test_to_pobs_flag(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((400, 4))
        # to_pobs=True should work on raw Gaussian data.
        v = RVineCopula().fit(x, to_pobs=True)
        assert v.d == 4
        assert v._T == 400

    def test_rejects_1d(self):
        u = np.random.default_rng(0).uniform(0, 1, 100)
        with pytest.raises(ValueError, match="2D"):
            RVineCopula().fit(u)


# ═══════════════════════════════════════════════════════════
# Matrix and pair copulas agree with each other
# ═══════════════════════════════════════════════════════════

class TestMatrixAgreement:

    def test_matrix_edge_matches_paircopula_position(self):
        d = 5
        u = _sample_dvine_gumbel(400, d, 2.5, seed=1)
        v = RVineCopula().fit(u)
        # Natural-order: leaf at M[d-1-col, col], tree-t other endpoint
        # at M[d-2-col-t, col], conditioning fills rows d-1-col-t..d-2-col.
        for (t, col), pc in v.pair_copulas.items():
            leaf = int(v.matrix[d - 1 - col, col])
            tail = int(v.matrix[d - 2 - col - t, col])
            cond_from_matrix = frozenset(
                int(v.matrix[r, col])
                for r in range(d - 1 - col - t, d - 1 - col)
            )
            # Recover the original edge index and its representation.
            orig_idx = v._edge_map[(t, col)]
            conditioned, conditioning = v.trees[t][orig_idx]
            assert {leaf, tail} == set(int(x) for x in conditioned)
            assert cond_from_matrix == conditioning


# ═══════════════════════════════════════════════════════════
# Log-likelihood
# ═══════════════════════════════════════════════════════════

class TestLogLikelihood:

    def test_cached_equals_sum_of_edges(self):
        u = _sample_dvine_gumbel(400, 4, 2.0, seed=0)
        v = RVineCopula().fit(u)
        edge_sum = sum(pc.log_likelihood for pc in v.pair_copulas.values())
        assert v.log_likelihood() == pytest.approx(edge_sum)

    def test_on_fitting_data_matches_cached(self):
        u = _sample_dvine_gumbel(500, 4, 2.5, seed=0)
        v = RVineCopula().fit(u)
        cached = v.log_likelihood()
        reeval = v.log_likelihood(u)
        assert reeval == pytest.approx(cached, rel=1e-8, abs=1e-6)

    def test_on_new_data_is_different_but_finite(self):
        v = RVineCopula().fit(_sample_dvine_gumbel(500, 4, 2.5, seed=0))
        u_new = _sample_dvine_gumbel(500, 4, 2.5, seed=99)
        ll = v.log_likelihood(u_new)
        assert np.isfinite(ll)
        assert ll != v.log_likelihood()

    def test_on_new_data_wrong_shape_raises(self):
        v = RVineCopula().fit(_sample_dvine_gumbel(300, 4, 2.0, seed=0))
        with pytest.raises(ValueError, match=r"must be \(T, 4\)"):
            v.log_likelihood(_sample_dvine_gumbel(300, 5, 2.0, seed=0))

    def test_independent_fit_gives_zero_cached(self):
        """Very weak data with aggressive min_edge_logL → all Independent → logL=0."""
        u = _independent(600, 4, seed=0)
        v = RVineCopula(min_edge_logL=1e9).fit(u)
        assert v.log_likelihood() == 0.0
        assert v.n_parameters == 0


# ═══════════════════════════════════════════════════════════
# Information criteria
# ═══════════════════════════════════════════════════════════

class TestInformationCriteria:

    def test_ic_formulas(self):
        u = _sample_dvine_gumbel(500, 4, 2.5, seed=0)
        v = RVineCopula().fit(u)
        k = v.n_parameters
        ll = v.log_likelihood()
        assert v.aic == pytest.approx(-2 * ll + 2 * k)
        assert v.bic == pytest.approx(-2 * ll + k * np.log(500))

    def test_n_parameters_counts_non_independent(self):
        u = _sample_dvine_gumbel(500, 4, 2.5, seed=0)
        v = RVineCopula().fit(u)
        expected = sum(
            1 for pc in v.pair_copulas.values()
            if not isinstance(pc.copula, IndependentCopula)
        )
        assert v.n_parameters == expected


# ═══════════════════════════════════════════════════════════
# Introspection helpers
# ═══════════════════════════════════════════════════════════

class TestIntrospection:

    def test_family_matrix_shape_and_entries(self):
        d = 4
        u = _sample_dvine_gumbel(400, d, 2.0, seed=0)
        v = RVineCopula().fit(u)
        F = v.family_matrix()
        assert F.shape == (d, d)
        # Edge positions in natural order: row d-2-col-t, column col.
        allowed = {"GumbelCopula", "ClaytonCopula", "FrankCopula",
                   "JoeCopula", "IndependentCopula", "BivariateGaussianCopula"}
        edge_cells = {(d - 2 - col - t, col) for (t, col) in v.pair_copulas}
        for (t, col) in v.pair_copulas:
            assert F[d - 2 - col - t, col] in allowed
        # Cells not carrying an edge must be empty strings.
        for r in range(d):
            for c in range(d):
                if (r, c) not in edge_cells:
                    assert F[r, c] == ""

    def test_parameter_matrix_shape_and_nan_defaults(self):
        d = 4
        u = _sample_dvine_gumbel(400, d, 2.0, seed=0)
        v = RVineCopula().fit(u)
        P = v.parameter_matrix()
        assert P.shape == (d, d)
        edge_cells = {(d - 2 - col - t, col) for (t, col) in v.pair_copulas}
        # Every non-edge cell is NaN.
        for r in range(d):
            for c in range(d):
                if (r, c) not in edge_cells:
                    assert np.isnan(P[r, c])
        for (t, col), pc in v.pair_copulas.items():
            val = P[d - 2 - col - t, col]
            assert val == pytest.approx(pc.param)

    def test_rotation_matrix_matches_copula_rotate(self):
        d = 4
        u = _sample_dvine_gumbel(400, d, 2.0, seed=0)
        v = RVineCopula().fit(u)
        R = v.rotation_matrix()
        for (t, col), pc in v.pair_copulas.items():
            assert R[d - 2 - col - t, col] == int(
                getattr(pc.copula, "rotate", 0)
            )

    def test_tau_matrix_values(self):
        d = 4
        u = _sample_dvine_gumbel(400, d, 2.0, seed=0)
        v = RVineCopula().fit(u)
        T = v.tau_matrix()
        for (t, col), pc in v.pair_copulas.items():
            val = T[d - 2 - col - t, col]
            assert val == pytest.approx(pc.tau)
            assert -1.0 <= val <= 1.0


# ═══════════════════════════════════════════════════════════
# Truncation and pruning pass through
# ═══════════════════════════════════════════════════════════

class TestTruncationAndPrune:

    def test_truncation_default_forces_independent_above(self):
        u = _sample_dvine_gumbel(500, 5, 3.0, seed=0)
        v = RVineCopula(truncation_level=2).fit(u)
        for (t, col), pc in v.pair_copulas.items():
            if t >= 2:
                assert isinstance(pc.copula, IndependentCopula)

    def test_truncation_can_keep_mle_above(self):
        u = _sample_dvine_gumbel(500, 5, 3.0, seed=0)
        v = RVineCopula(
            truncation_level=2,
            truncation_fill='mle',
        ).fit(u)
        for (t, col), pc in v.pair_copulas.items():
            if t >= 2:
                assert pc.fit_result.method == 'MLE'

    def test_truncation_can_force_independent_above(self):
        u = _sample_dvine_gumbel(500, 5, 3.0, seed=0)
        v = RVineCopula(
            truncation_level=2,
            truncation_fill='independent',
        ).fit(u)
        for (t, col), pc in v.pair_copulas.items():
            if t >= 2:
                assert isinstance(pc.copula, IndependentCopula)

    def test_fit_truncation_kwargs_override_constructor_defaults(self):
        u = _sample_dvine_gumbel(500, 5, 3.0, seed=0)
        v = RVineCopula().fit(
            u,
            truncation_level=2,
            truncation_fill='independent',
        )
        for (t, col), pc in v.pair_copulas.items():
            if t >= 2:
                assert isinstance(pc.copula, IndependentCopula)

    def test_fit_truncation_kwargs_can_override_constructor_values(self):
        u = _sample_dvine_gumbel(500, 5, 3.0, seed=0)
        v = RVineCopula(
            truncation_level=1,
            truncation_fill='independent',
        ).fit(u, truncation_level=None)
        assert any(
            not isinstance(pc.copula, IndependentCopula)
            for (t, _), pc in v.pair_copulas.items()
            if t >= 1
        )

    def test_min_edge_logL_prunes_all(self):
        u = _sample_dvine_gumbel(400, 4, 2.0, seed=0)
        v = RVineCopula(min_edge_logL=1e9).fit(u)
        assert v.n_parameters == 0
        assert v.log_likelihood() == 0.0

    def test_candidates_restriction(self):
        u = _sample_dvine_gumbel(500, 4, 2.5, seed=0)
        v = RVineCopula(candidates=[GumbelCopula]).fit(u)
        for pc in v.pair_copulas.values():
            assert type(pc.copula) in (GumbelCopula, IndependentCopula)


# ═══════════════════════════════════════════════════════════
# Summary / repr
# ═══════════════════════════════════════════════════════════

class TestRVineEdgePropagation:

    def test_mle_edge_h_matches_copula_h(self):
        cop = BivariateGaussianCopula()
        result = MLEResult(
            log_likelihood=0.0,
            method='MLE',
            copula_name=cop.name,
            success=True,
            copula_param=0.45,
        )
        edge = PairCopula(cop, 0.45, 0.0, 0, 0.0, result)
        u_given = np.array([0.2, 0.5, 0.8])
        u_cond = np.array([0.3, 0.4, 0.7])

        expected = cop.h(u_cond, u_given, np.full(3, 0.45))
        assert _edge_h(edge, u_cond, u_given) == pytest.approx(expected)

    def test_mle_edge_h_inverse_roundtrip_with_given_r(self):
        cop = BivariateGaussianCopula()
        result = MLEResult(
            log_likelihood=0.0,
            method='MLE',
            copula_name=cop.name,
            success=True,
            copula_param=0.35,
        )
        edge = PairCopula(cop, 0.35, 0.0, 0, 0.0, result)
        u_given = np.array([0.2, 0.5, 0.8])
        u_cond = np.array([0.3, 0.4, 0.7])
        r = np.full(3, 0.35)

        v = _edge_h(edge, u_cond, u_given)
        roundtrip = _edge_h_inverse(edge, v, u_given, config={'r': r})
        assert roundtrip == pytest.approx(u_cond)

    def test_independent_edge_is_passthrough(self):
        cop = IndependentCopula()
        result = cop.fit(np.array([[0.2, 0.3], [0.5, 0.7]]))
        edge = PairCopula(cop, 0.0, 0.0, 0, 0.0, result)
        u_given = np.array([0.2, 0.5, 0.8])
        u_cond = np.array([0.3, 0.4, 0.7])

        assert _edge_h(edge, u_cond, u_given) == pytest.approx(u_cond)
        assert _edge_h_inverse(edge, u_cond, u_given) == pytest.approx(u_cond)

    def test_mle_r_for_sample_is_constant(self):
        cop = BivariateGaussianCopula()
        result = MLEResult(
            log_likelihood=0.0,
            method='MLE',
            copula_name=cop.name,
            success=True,
            copula_param=0.25,
        )
        edge = PairCopula(cop, 0.25, 0.0, 0, 0.0, result)
        r = _edge_r_for_sample(edge, 5, np.random.default_rng(0))
        assert r == pytest.approx(np.full(5, 0.25))

    def test_gas_r_for_sample_requires_stepwise_updates(self):
        cop = BivariateGaussianCopula()
        result = GASResult(
            log_likelihood=0.0,
            method='GAS',
            copula_name=cop.name,
            success=True,
            params=gas_params(0.1, 0.2, 0.3),
        )
        edge = PairCopula(cop, 0.0, 0.0, 0, 0.0, result)

        with pytest.raises(ValueError, match="stepwise score updates"):
            _edge_r_for_sample(edge, 5, np.random.default_rng(0))

    @pytest.mark.parametrize(
        "copula_class,param,rotations",
        [
            (ClaytonCopula, 2.0, [0, 90, 180, 270]),
            (GumbelCopula, 2.0, [0, 90, 180, 270]),
            (FrankCopula, 5.0, [0]),
            (JoeCopula, 2.0, [0, 90, 180, 270]),
            (BivariateGaussianCopula, 0.45, [0]),
            (IndependentCopula, 0.0, [0]),
        ],
    )
    def test_h_inverse_roundtrip_supported_rotations(
            self, copula_class, param, rotations):
        u = np.array([0.2, 0.4, 0.7])
        given = np.array([0.3, 0.6, 0.8])
        r = np.full(len(u), param)

        for rotation in rotations:
            cop = copula_class(rotate=rotation)
            h_val = cop.h(u, given, r)
            assert cop.h_inverse(h_val, given, r) == pytest.approx(
                u, abs=1e-8)


class TestSampling:

    def test_sample_shape_and_unit_interval(self):
        u = _sample_dvine_gumbel(400, 4, 2.0, seed=0)
        v = RVineCopula().fit(u)
        s = v.sample(100, rng=np.random.default_rng(1))
        assert s.shape == (100, 4)
        assert np.all(s > 0.0)
        assert np.all(s < 1.0)

    def test_sample_rejects_bad_n(self):
        u = _sample_dvine_gumbel(200, 3, 2.0, seed=0)
        v = RVineCopula().fit(u)
        with pytest.raises(ValueError, match="positive int"):
            v.sample(0)

    def test_gas_sample_shape_and_unit_interval(self):
        rng = np.random.default_rng(3)
        cop = BivariateGaussianCopula()
        u12 = cop.sample(80, np.full(80, 0.65), rng=rng)
        u = np.column_stack([u12, rng.uniform(0.01, 0.99, 80)])
        v = RVineCopula(candidates=[BivariateGaussianCopula]).fit(
            u, method='gas')
        s = v.sample(20, rng=np.random.default_rng(4))
        assert s.shape == (20, 3)
        assert np.all(s > 0.0)
        assert np.all(s < 1.0)

    def test_scar_tm_sample_shape_and_unit_interval(self):
        u = _sample_dynamic_gaussian_chain(45, 3, seed=9)
        v = RVineCopula(candidates=[BivariateGaussianCopula]).fit(
            u,
            method='scar-tm-ou',
            K=12,
            grid_range=3.0,
            tol=5e-2,
        )
        s = v.sample(20, rng=np.random.default_rng(10))
        assert s.shape == (20, 3)
        assert np.all(s > 0.0)
        assert np.all(s < 1.0)


class TestDynamicFitSampleRefit:

    def test_gas_fit_sample_refit_smoke(self):
        u = _sample_dynamic_gaussian_chain(70, 4, seed=12)
        v = RVineCopula(candidates=[BivariateGaussianCopula]).fit(
            u, method='gas')
        assert any(
            isinstance(pc.fit_result, GASResult)
            for pc in v.pair_copulas.values()
        )

        s = v.sample(35, rng=np.random.default_rng(13))
        refit = RVineCopula(candidates=[BivariateGaussianCopula]).fit(
            s, method='gas')

        assert refit.d == 4
        assert np.isfinite(refit.log_likelihood())
        assert all(
            isinstance(pc.fit_result, (GASResult, MLEResult, IndependentResult))
            for pc in refit.pair_copulas.values()
        )
        assert all(
            bool(getattr(pc.fit_result, "success", True))
            for pc in refit.pair_copulas.values()
        )

    def test_scar_tm_fit_sample_refit_smoke(self):
        u = _sample_dynamic_gaussian_chain(45, 3, seed=14)
        fit_kwargs = dict(K=12, grid_range=3.0, tol=5e-2)
        v = RVineCopula(candidates=[BivariateGaussianCopula]).fit(
            u, method='scar-tm-ou', **fit_kwargs)
        assert any(
            isinstance(pc.fit_result, LatentResult)
            for pc in v.pair_copulas.values()
        )

        s = v.sample(25, rng=np.random.default_rng(15))
        refit = RVineCopula(candidates=[BivariateGaussianCopula]).fit(
            s,
            method='scar-tm-ou',
            K=10,
            grid_range=3.0,
            tol=8e-2,
        )

        assert refit.d == 3
        assert np.isfinite(refit.log_likelihood())
        assert all(
            isinstance(pc.fit_result, (LatentResult, IndependentResult))
            for pc in refit.pair_copulas.values()
        )


class TestSummary:

    def test_summary_is_string(self):
        u = _sample_dvine_gumbel(300, 4, 2.0, seed=0)
        v = RVineCopula().fit(u)
        s = v.summary(as_string=True)
        assert isinstance(s, str)
        assert "log_likelihood" in s
        assert "AIC" in s and "BIC" in s

    def test_summary_prints_by_default(self, capsys):
        u = _sample_dvine_gumbel(300, 4, 2.0, seed=0)
        v = RVineCopula().fit(u)
        result = v.summary()
        captured = capsys.readouterr()
        assert result is None
        assert "RVineCopula" in captured.out
        assert "log_likelihood" in captured.out
        assert "\nEdges" in captured.out

    def test_str_equals_summary(self):
        u = _sample_dvine_gumbel(300, 4, 2.0, seed=0)
        v = RVineCopula().fit(u)
        assert str(v) == v.summary(as_string=True)

    def test_summary_uses_short_gaussian_name(self):
        rng = np.random.default_rng(0)
        cop = BivariateGaussianCopula()
        u12 = cop.sample(120, np.full(120, 0.45), rng=rng)
        u = np.column_stack([u12, rng.uniform(0.01, 0.99, 120)])
        v = RVineCopula(candidates=[BivariateGaussianCopula]).fit(u)
        s = v.summary(as_string=True)
        assert "GaussianCopula" in s
        assert "BivariateGaussianCopula" not in s

    def test_summary_mle_omits_dynamic_param_columns(self):
        u = _sample_dvine_gumbel(300, 4, 2.0, seed=0)
        v = RVineCopula().fit(u)
        s = v.summary(as_string=True)
        header = next(
            line for line in s.splitlines()
            if "param" in line and "logL" in line
        )
        assert "dyn_params" not in header
        assert "theta" not in header
        assert "mu" not in header
        assert "nu" not in header

    def test_summary_shows_dynamic_params(self):
        u = _sample_dynamic_gaussian_chain(40, 3, seed=3)
        for result, expected in [
            (
                GASResult(
                    log_likelihood=1.0,
                    method='GAS',
                    copula_name='GaussianCopula',
                    success=True,
                    params=gas_params(0.1, 0.2, 0.3),
                ),
                ("omega=  0.100", "alpha=  0.200", "beta=  0.300"),
            ),
            (
                LatentResult(
                    log_likelihood=1.0,
                    method='SCAR-TM-OU',
                    copula_name='GaussianCopula',
                    success=True,
                    params=ou_params(1.1, 1.2, 1.3),
                ),
                ("theta=  1.100", "mu=  1.200", "nu=  1.300"),
            ),
        ]:
            v = RVineCopula(candidates=[BivariateGaussianCopula]).fit(u)
            pc = v.pair_copulas[(0, 0)]
            v.pair_copulas[(0, 0)] = PairCopula(
                copula=pc.copula,
                param=pc.param,
                log_likelihood=1.0,
                nfev=pc.nfev,
                tau=pc.tau,
                fit_result=result,
            )
            s = v.summary(as_string=True)
            header = next(
                line for line in s.splitlines()
                if "dyn_params" in line and "param" in line and "logL" in line
            )
            assert "dyn_params" in header
            for text in expected:
                assert text in s

            line = next(
                line for line in s.splitlines()
                if "GaussianCopula" in line
            )
            dyn_end = header.index("dyn_params") + 48
            param_end = header.index("param") + len("param")
            assert line[dyn_end:param_end].strip() == ""

    def test_summary_rounds_tiny_dynamic_params_and_aligns_columns(self):
        u = _sample_dynamic_gaussian_chain(40, 3, seed=3)
        v = RVineCopula(candidates=[BivariateGaussianCopula]).fit(u)
        pc = v.pair_copulas[(0, 0)]
        latent = LatentResult(
            log_likelihood=pc.log_likelihood,
            method='SCAR-TM-OU',
            copula_name=pc.copula.name,
            success=True,
            params=ou_params(88.0, -3.53e-05, 8.2),
        )
        v.pair_copulas[(0, 0)] = PairCopula(
            copula=pc.copula,
            param=pc.param,
            log_likelihood=pc.log_likelihood,
            nfev=pc.nfev,
            tau=pc.tau,
            fit_result=latent,
        )

        s = v.summary(as_string=True)
        assert "theta= 88.000, mu=  0.000, nu=  8.200" in s
        assert "e-05" not in s

        lines = s.splitlines()
        header = next(line for line in lines if " tau " in line and "logL" in line)
        for line in lines[lines.index(header) + 1:]:
            if line.strip():
                parts = line.split()
                assert np.isfinite(float(parts[-2]))
                assert float(parts[-1]) >= 0.0

    def test_unfitted_summary_as_string(self):
        v = RVineCopula()
        assert "unfitted" in v.summary(as_string=True)
        assert "unfitted" in repr(v)

    def test_repr_fitted_compact(self):
        u = _sample_dvine_gumbel(300, 4, 2.0, seed=0)
        v = RVineCopula().fit(u)
        r = repr(v)
        assert r.startswith("RVineCopula(")
        assert "d=4" in r


# ═══════════════════════════════════════════════════════════
# Predictive sampling
# ═══════════════════════════════════════════════════════════

class TestPredict:

    def test_predict_shape_and_unit_interval(self):
        u = _sample_dvine_gumbel(200, 3, 2.0, seed=0)
        v = RVineCopula().fit(u)
        p = v.predict(50, rng=np.random.default_rng(2))
        assert p.shape == (50, 3)
        assert np.all(p > 0.0)
        assert np.all(p < 1.0)

    def test_predict_empty_given_keeps_unconditional_path(self):
        u = _sample_dvine_gumbel(200, 3, 2.0, seed=0)
        v = RVineCopula().fit(u)
        p = v.predict(25, given={}, rng=np.random.default_rng(2))
        assert p.shape == (25, 3)
        assert np.all(p > 0.0)
        assert np.all(p < 1.0)

    def test_predict_all_given_returns_constant_rows(self):
        u = _sample_dvine_gumbel(200, 3, 2.0, seed=0)
        v = RVineCopula().fit(u)
        given = {0: 0.2, 1: 0.5, 2: 0.8}
        p = v.predict(10, given=given)
        assert p.shape == (10, 3)
        assert np.allclose(p, np.array([[0.2, 0.5, 0.8]] * 10))

    def test_predict_given_peel_suffix_fast_path(self):
        u = _sample_dvine_gumbel(250, 4, 2.0, seed=0)
        v = RVineCopula().fit(u)
        peel_order = [
            int(v.matrix[v.d - 1 - col, col])
            for col in range(v.d)
        ]
        suffix = peel_order[-2:]
        given = {suffix[0]: 0.35, suffix[1]: 0.75}

        p = v.predict(40, given=given, rng=np.random.default_rng(4))

        assert p.shape == (40, 4)
        assert np.all(p > 0.0)
        assert np.all(p < 1.0)
        assert np.allclose(p[:, suffix[0]], 0.35)
        assert np.allclose(p[:, suffix[1]], 0.75)

    def test_api_predict_rvine_given_peel_suffix_fast_path(self):
        u = _sample_dvine_gumbel(250, 4, 2.0, seed=0)
        v = RVineCopula().fit(u)
        peel_order = [
            int(v.matrix[v.d - 1 - col, col])
            for col in range(v.d)
        ]
        given = {peel_order[-1]: 0.6}

        p = api_predict(
            v,
            u,
            result=None,
            n=30,
            given=given,
            rng=np.random.default_rng(5),
        )

        assert p.shape == (30, 4)
        assert np.all(p > 0.0)
        assert np.all(p < 1.0)
        assert np.allclose(p[:, peel_order[-1]], 0.6)

    def test_predict_given_rebuilds_matrix_when_possible(self):
        u = _sample_dvine_gumbel(250, 4, 2.0, seed=0)
        v = RVineCopula().fit(u)
        peel_order = [
            int(v.matrix[v.d - 1 - col, col])
            for col in range(v.d)
        ]
        given_var = peel_order[0]
        assert v._given_suffix_start_col({given_var: 0.45}) is None

        p = v.predict(
            35,
            given={given_var: 0.45},
            rng=np.random.default_rng(6),
        )

        assert p.shape == (35, 4)
        assert np.all(p > 0.0)
        assert np.all(p < 1.0)
        assert np.allclose(p[:, given_var], 0.45)

    @pytest.mark.parametrize(
        "method,fit_kwargs",
        [
            ("gas", {}),
            ("scar-tm-ou", {"K": 15, "grid_range": 3.0}),
        ],
    )
    def test_predict_given_peel_suffix_fast_path_dynamic_methods(
            self, method, fit_kwargs):
        u = _sample_dynamic_gaussian_chain(45, 3, seed=9)
        v = RVineCopula(candidates=[BivariateGaussianCopula]).fit(
            u,
            method=method,
            **fit_kwargs,
        )
        peel_order = [
            int(v.matrix[v.d - 1 - col, col])
            for col in range(v.d)
        ]
        given = {peel_order[-1]: 0.55}

        p = v.predict(
            12,
            given=given,
            horizon='current',
            rng=np.random.default_rng(10),
        )

        assert p.shape == (12, 3)
        assert np.all(p > 0.0)
        assert np.all(p < 1.0)
        assert np.allclose(p[:, peel_order[-1]], 0.55)

    @pytest.mark.parametrize(
        "method,fit_kwargs",
        [
            ("gas", {}),
            ("scar-tm-ou", {"K": 15, "grid_range": 3.0}),
        ],
    )
    def test_predict_given_rebuild_fast_path_dynamic_methods(
            self, method, fit_kwargs):
        u = _sample_dynamic_gaussian_chain(45, 4, seed=12)
        v = RVineCopula(candidates=[BivariateGaussianCopula]).fit(
            u,
            method=method,
            **fit_kwargs,
        )
        peel_order = [
            int(v.matrix[v.d - 1 - col, col])
            for col in range(v.d)
        ]
        given_var = peel_order[0]

        p = v.predict(
            10,
            given={given_var: 0.4},
            horizon='current',
            rng=np.random.default_rng(13),
        )

        assert p.shape == (10, 4)
        assert np.all(p > 0.0)
        assert np.all(p < 1.0)
        assert np.allclose(p[:, given_var], 0.4)

    def test_gas_predict_r_current_and_next_are_static_paths(self):
        cop = BivariateGaussianCopula()
        result = GASResult(
            log_likelihood=0.0,
            method='GAS',
            copula_name=cop.name,
            success=True,
            params=gas_params(0.0, 0.8, 0.4),
            scaling='unit',
            r_last=0.0,
        )
        edge = PairCopula(
            copula=cop,
            param=0.0,
            log_likelihood=0.0,
            nfev=0,
            tau=0.0,
            fit_result=result,
        )
        u_train_pair = np.array([
            [0.40, 0.45],
            [0.15, 0.90],
        ])

        r_current = _edge_r_for_predict(
            edge, 6, u_train_pair=u_train_pair, horizon='current')
        r_next = _edge_r_for_predict(
            edge, 6, u_train_pair=u_train_pair, horizon='next')

        assert r_current.shape == (6,)
        assert r_next.shape == (6,)
        assert np.allclose(r_current, r_current[0])
        assert np.allclose(r_next, r_next[0])
        assert not np.isclose(r_current[0], r_next[0])

    def test_scar_predict_r_samples_from_tm_posterior(self, monkeypatch):
        cop = BivariateGaussianCopula()
        result = LatentResult(
            log_likelihood=0.0,
            method='SCAR-TM-OU',
            copula_name=cop.name,
            success=True,
            params=ou_params(1.0, 0.0, 0.5),
            K=5,
            grid_range=2.0,
        )
        edge = PairCopula(
            copula=cop,
            param=0.0,
            log_likelihood=0.0,
            nfev=0,
            tau=0.0,
            fit_result=result,
        )
        calls = []

        def fake_tm_state_distribution(theta, mu, nu, u, copula_arg,
                                       horizon='next', **kwargs):
            calls.append((theta, mu, nu, u.copy(), copula_arg, horizon, kwargs))
            return np.array([-0.5, 0.5]), np.array([0.0, 1.0])

        monkeypatch.setattr(
            "pyscarcopula.numerical.predictive_tm.tm_state_distribution",
            fake_tm_state_distribution,
        )

        u_train_pair = np.array([[0.2, 0.3], [0.7, 0.8]])
        r = _edge_r_for_predict(
            edge,
            4,
            u_train_pair=u_train_pair,
            horizon='current',
            rng=np.random.default_rng(1),
        )

        assert np.allclose(r, cop.transform(np.full(4, 0.5)))
        assert calls[0][5] == 'current'
        np.testing.assert_allclose(calls[0][3], u_train_pair)

    @pytest.mark.parametrize(
        "given,exc",
        [
            ([(0, 0.5)], TypeError),
            (np.array([0.5]), TypeError),
            ({"0": 0.5}, TypeError),
            ({1.2: 0.5}, TypeError),
            ({True: 0.5}, TypeError),
            ({3: 0.5}, ValueError),
            ({0: 0.0}, ValueError),
            ({0: 1.0}, ValueError),
            ({0: [0.5]}, TypeError),
        ],
    )
    def test_predict_given_validation(self, given, exc):
        u = _sample_dvine_gumbel(200, 3, 2.0, seed=0)
        v = RVineCopula().fit(u)
        with pytest.raises(exc):
            v.predict(10, given=given)

    def test_predict_partial_given_rejects_non_rebuildable_pattern(self):
        u = _sample_dvine_gumbel(250, 4, 2.0, seed=0)
        v = RVineCopula().fit(u)
        peel_order = [
            int(v.matrix[v.d - 1 - col, col])
            for col in range(v.d)
        ]
        given = {peel_order[0]: 0.5, peel_order[2]: 0.6}
        assert v._suffix_sampling_state(given) is None

        with pytest.raises(ValueError, match="R-vine variable order"):
            v.predict(10, given=given, rng=np.random.default_rng(33))


class TestGoF:

    def test_rvine_rosenblatt_transform_shape_and_unit_interval(self):
        u = _sample_dvine_gumbel(250, 4, 2.0, seed=0)
        v = RVineCopula().fit(u)
        e = rvine_rosenblatt_transform(v, u)
        assert e.shape == u.shape
        assert np.all(e > 0.0)
        assert np.all(e < 1.0)

    def test_gof_test_dispatches_to_new_rvine_matrix_path(self):
        u = _sample_dvine_gumbel(250, 4, 2.0, seed=0)
        v = RVineCopula().fit(u)
        result = gof_test(v, u, to_pobs=False)
        assert np.isfinite(result.statistic)
        assert np.isfinite(result.pvalue)

    @pytest.mark.parametrize(
        "method,fit_kwargs",
        [
            ("mle", {}),
            ("gas", {}),
            ("scar-tm-ou", {"K": 20, "grid_range": 3.0}),
        ],
    )
    def test_gof_test_handles_all_rvine_fit_methods(
            self, method, fit_kwargs):
        rng = np.random.default_rng(11)
        cop = BivariateGaussianCopula()
        u12 = cop.sample(45, np.full(45, 0.55), rng=rng)
        u = np.column_stack([u12, rng.uniform(0.01, 0.99, 45)])
        v = RVineCopula(candidates=[BivariateGaussianCopula]).fit(
            u,
            method=method,
            **fit_kwargs,
        )

        result = gof_test(
            v,
            u,
            to_pobs=False,
            K=fit_kwargs.get("K", 20),
            grid_range=fit_kwargs.get("grid_range", 3.0),
        )
        assert np.isfinite(result.statistic)
        assert np.isfinite(result.pvalue)


class TestApiDispatch:

    def test_api_sample_dispatches_to_rvine_and_ignores_result(self):
        u = _sample_dvine_gumbel(250, 3, 2.0, seed=0)
        v = RVineCopula().fit(u)
        s = api_sample(v, u, result=None, n=40, rng=np.random.default_rng(7))
        assert s.shape == (40, 3)
        assert np.all(s > 0.0)
        assert np.all(s < 1.0)

    def test_api_predict_dispatches_to_rvine_and_ignores_result(self):
        u = _sample_dvine_gumbel(250, 3, 2.0, seed=0)
        v = RVineCopula().fit(u)
        p = api_predict(
            v,
            u,
            result=None,
            n=40,
            horizon='current',
            rng=np.random.default_rng(8),
        )
        assert p.shape == (40, 3)
        assert np.all(p > 0.0)
        assert np.all(p < 1.0)

    def test_predict_accepts_u_alias_for_cvine_style_callers(self):
        u = _sample_dvine_gumbel(250, 3, 2.0, seed=0)
        v = RVineCopula().fit(u)
        p = v.predict(30, u=u, rng=np.random.default_rng(9))
        assert p.shape == (30, 3)
        assert np.all(p > 0.0)
        assert np.all(p < 1.0)

    def test_predict_rejects_both_u_aliases(self):
        u = _sample_dvine_gumbel(250, 3, 2.0, seed=0)
        v = RVineCopula().fit(u)
        with pytest.raises(ValueError, match="u_train or u"):
            v.predict(10, u_train=u, u=u)

    def test_predict_rejects_bad_horizon(self):
        u = _sample_dvine_gumbel(200, 3, 2.0, seed=0)
        v = RVineCopula().fit(u)
        with pytest.raises(ValueError, match="horizon"):
            v.predict(10, horizon='bad')

    def test_gas_predict_shape_and_unit_interval(self):
        rng = np.random.default_rng(5)
        cop = BivariateGaussianCopula()
        u12 = cop.sample(80, np.full(80, 0.65), rng=rng)
        u = np.column_stack([u12, rng.uniform(0.01, 0.99, 80)])
        v = RVineCopula(candidates=[BivariateGaussianCopula]).fit(
            u, method='gas')
        p = v.predict(20, horizon='current', rng=np.random.default_rng(6))
        assert p.shape == (20, 3)
        assert np.all(p > 0.0)
        assert np.all(p < 1.0)


# ═══════════════════════════════════════════════════════════
# Constructor validation
# ═══════════════════════════════════════════════════════════

class TestCtorValidation:

    def test_bad_criterion(self):
        with pytest.raises(ValueError, match="criterion"):
            RVineCopula(criterion="foo")

    def test_negative_truncation_level(self):
        with pytest.raises(ValueError, match="truncation_level"):
            RVineCopula(truncation_level=-1)

    def test_truncation_level_type(self):
        with pytest.raises(TypeError):
            RVineCopula(truncation_level=1.5)

    def test_bad_truncation_fill(self):
        with pytest.raises(ValueError, match="truncation_fill"):
            RVineCopula(truncation_fill="foo")

    def test_negative_threshold(self):
        with pytest.raises(ValueError, match="threshold"):
            RVineCopula(threshold=-0.01)


class TestConditionalPredict:

    @pytest.mark.parametrize("d", [3, 4, 6])
    @pytest.mark.parametrize(
        "method,fit_kwargs",
        [
            ("mle", {}),
            ("gas", {}),
            ("scar-tm-ou", {"K": 12, "grid_range": 3.0}),
        ],
    )
    def test_shape_unit_and_given_column_fixed(
            self, d, method, fit_kwargs):
        u = _sample_dynamic_gaussian_chain(45, d, seed=100 + d)
        v = RVineCopula(candidates=[BivariateGaussianCopula]).fit(
            u,
            method=method,
            **fit_kwargs,
        )

        peel_order = [
            int(v.matrix[v.d - 1 - col, col])
            for col in range(v.d)
        ]
        given_var = peel_order[-1]

        samples = v.predict(
            16,
            given={given_var: 0.2},
            horizon='current',
            rng=np.random.default_rng(200 + d),
        )

        assert samples.shape == (16, d)
        assert np.all(samples > 0.0)
        assert np.all(samples < 1.0)
        assert np.allclose(samples[:, given_var], 0.2)

    def test_all_given_returns_constant_rows_identity(self):
        u = _sample_dvine_gumbel(200, 4, 2.0, seed=0)
        v = RVineCopula().fit(u)
        given = {0: 0.15, 1: 0.35, 2: 0.65, 3: 0.85}
        samples = v.predict(7, given=given)
        expected = np.array([[given[i] for i in range(4)]] * 7)
        np.testing.assert_allclose(samples, expected)

    @pytest.mark.parametrize(
        "given,exc",
        [
            ({0: 0.0}, ValueError),
            ({0: 1.0}, ValueError),
            ({4: 0.5}, ValueError),
            ({True: 0.5}, TypeError),
            ([(0, 0.5)], TypeError),
        ],
    )
    def test_invalid_given_rejected(self, given, exc):
        u = _sample_dvine_gumbel(200, 4, 2.0, seed=0)
        v = RVineCopula().fit(u)
        with pytest.raises(exc):
            v.predict(10, given=given)

    def test_non_prefix_non_rebuildable_given_is_rejected(self):
        u = _sample_dvine_gumbel(250, 4, 2.0, seed=0)
        v = RVineCopula().fit(u)
        peel_order = [
            int(v.matrix[v.d - 1 - col, col])
            for col in range(v.d)
        ]
        given = {peel_order[0]: 0.25, peel_order[2]: 0.75}

        with pytest.raises(ValueError, match="R-vine variable order"):
            v.predict(10, given=given, rng=np.random.default_rng(34))

    def test_gaussian_conditional_matches_closed_form_oracle(self):
        rng = np.random.default_rng(31)
        rho = 0.65
        cop = BivariateGaussianCopula()
        u = cop.sample(2500, np.full(2500, rho), rng=rng)
        v = RVineCopula(candidates=[BivariateGaussianCopula]).fit(u)
        edge = next(iter(v.pair_copulas.values()))
        rho_hat = float(edge.param)
        u0 = 0.3

        samples = v.predict(
            1500,
            given={0: u0},
            rng=np.random.default_rng(32),
        )

        z0 = norm.ppf(u0)
        z1 = norm.ppf(np.clip(samples[:, 1], 1e-10, 1.0 - 1e-10))
        pit = norm.cdf((z1 - rho_hat * z0) / np.sqrt(1.0 - rho_hat ** 2))
        stat, pvalue = kstest(pit, "uniform")

        assert stat < 0.045
        assert pvalue > 0.01

    def test_independent_conditional_leaves_unconditioned_uniform(self):
        u = _independent(300, 4, seed=8)
        v = RVineCopula(candidates=[IndependentCopula]).fit(u)

        samples = v.predict(
            2500,
            given={0: 0.25},
            rng=np.random.default_rng(9),
        )

        assert np.allclose(samples[:, 0], 0.25)
        for col in [1, 2, 3]:
            stat, pvalue = kstest(samples[:, col], "uniform")
            assert stat < 0.04
            assert pvalue > 0.01
