"""Tests for _types and _utils — the foundation of the new architecture."""
import numpy as np
import pytest

from pyscarcopula._types import (
    NumericalConfig, DEFAULT_CONFIG,
    LatentProcessParams, ou_params, gas_params,
    MLEResult, LatentResult, GASResult, IndependentResult,
)
from pyscarcopula._utils import broadcast, pobs, clip_unit


# ══════════════════════════════════════════════════════════════════
# NumericalConfig
# ══════════════════════════════════════════════════════════════════

class TestNumericalConfig:
    def test_defaults(self):
        cfg = NumericalConfig()
        assert cfg.default_K == 300
        assert cfg.default_grid_range == 5.0
        assert cfg.default_pts_per_sigma == 4
        assert cfg.gas_score_eps == 1e-4

    def test_override(self):
        cfg = NumericalConfig(default_K=500, gas_score_eps=1e-6)
        assert cfg.default_K == 500
        assert cfg.gas_score_eps == 1e-6
        assert cfg.default_grid_range == 5.0  # unchanged

    def test_frozen(self):
        cfg = NumericalConfig()
        with pytest.raises(Exception):  # FrozenInstanceError
            cfg.default_K = 999


# ══════════════════════════════════════════════════════════════════
# LatentProcessParams — the key abstraction for variable param count
# ══════════════════════════════════════════════════════════════════

class TestLatentProcessParams:
    def test_ou_params(self):
        p = ou_params(theta=49.97, mu=2.42, nu=10.65)
        assert p.process_type == 'ou'
        assert p.n_params == 3
        assert p.theta == pytest.approx(49.97)
        assert p.mu == pytest.approx(2.42)
        assert p.nu == pytest.approx(10.65)

    def test_gas_params(self):
        p = gas_params(omega=0.07, alpha=0.33, beta=0.97)
        assert p.process_type == 'gas'
        assert p.n_params == 3
        assert p.omega == pytest.approx(0.07)
        assert p.alpha == pytest.approx(0.33)
        assert p.beta == pytest.approx(0.97)

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
        assert d == {'theta': 1.0, 'mu': 2.0, 'nu': 3.0}

    def test_replace(self):
        p = ou_params(1.0, 2.0, 3.0)
        p2 = p.replace(mu=5.0)
        assert p2.mu == pytest.approx(5.0)
        assert p2.theta == pytest.approx(1.0)
        assert p.mu == pytest.approx(2.0)  # original unchanged (frozen)

    def test_values_array(self):
        p = ou_params(1.0, 2.0, 3.0)
        assert isinstance(p.values, np.ndarray)
        assert p.values.dtype == np.float64
        np.testing.assert_array_equal(p.values, [1.0, 2.0, 3.0])

    def test_bounds(self):
        p = ou_params(1.0, 2.0, 3.0)
        assert p.bounds_lower is not None
        assert p.bounds_lower[0] == pytest.approx(0.001)  # theta > 0
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
        with pytest.raises(Exception):
            p.process_type = 'xxx'

    def test_repr(self):
        p = ou_params(49.97, 2.42, 10.65)
        r = repr(p)
        assert 'ou' in r
        assert 'theta=49.9700' in r


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
        assert r.params.theta == pytest.approx(49.97)
        assert r.K == 300
        assert r.pts_per_sigma == 4
        # Legacy access
        np.testing.assert_allclose(r.alpha, [49.97, 2.42, 10.65])

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
        assert len(r.alpha) == 4

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
        assert r.alpha_gas == pytest.approx(0.331)
        assert r.beta == pytest.approx(0.9677)
        assert r.scaling == 'unit'

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
        with pytest.raises(Exception):
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
