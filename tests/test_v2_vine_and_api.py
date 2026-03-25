"""Tests for vine.forward_pass and api module."""
import numpy as np
import pytest

from pyscarcopula.vine.forward_pass import (
    VineStep, vine_forward_iter, vine_rosenblatt,
)
from pyscarcopula.api import fit, log_likelihood, smoothed_params, configure


# ══════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════

@pytest.fixture
def random_u4():
    """4-dimensional pseudo-observations."""
    from pyscarcopula._utils import pobs
    rng = np.random.default_rng(42)
    data = rng.standard_normal((200, 4))
    return pobs(data)


@pytest.fixture
def fitted_vine(random_u4):
    """Fitted C-vine for testing."""
    from pyscarcopula import CVineCopula
    vine = CVineCopula()
    vine.fit(random_u4, method='mle')
    return vine


# ══════════════════════════════════════════════════════════════════
# vine_forward_iter tests
# ══════════════════════════════════════════════════════════════════

class TestVineForwardIter:

    def test_yields_correct_number_of_steps(self, fitted_vine, random_u4):
        """d=4 -> 3+2+1 = 6 edges total."""
        from pyscarcopula.copula.vine import _edge_h

        steps = list(vine_forward_iter(random_u4, fitted_vine.edges, _edge_h))
        d = random_u4.shape[1]
        expected = d * (d - 1) // 2  # 6 for d=4
        assert len(steps) == expected

    def test_step_structure(self, fitted_vine, random_u4):
        from pyscarcopula.copula.vine import _edge_h

        steps = list(vine_forward_iter(random_u4, fitted_vine.edges, _edge_h))

        # First step should be tree 0, edge 0
        assert steps[0].tree == 0
        assert steps[0].edge_idx == 0
        assert steps[0].u_pair.shape == (200, 2)

    def test_tree_edge_counts(self, fitted_vine, random_u4):
        """Tree j should have d-j-1 edges."""
        from pyscarcopula.copula.vine import _edge_h

        steps = list(vine_forward_iter(random_u4, fitted_vine.edges, _edge_h))
        d = random_u4.shape[1]

        for j in range(d - 1):
            tree_steps = [s for s in steps if s.tree == j]
            assert len(tree_steps) == d - j - 1

    def test_u_pair_values_in_unit_interval(self, fitted_vine, random_u4):
        from pyscarcopula.copula.vine import _edge_h

        for step in vine_forward_iter(random_u4, fitted_vine.edges, _edge_h):
            assert np.all(step.u_pair > 0)
            assert np.all(step.u_pair < 1)


# ══════════════════════════════════════════════════════════════════
# vine_rosenblatt tests
# ══════════════════════════════════════════════════════════════════

class TestVineRosenblatt:

    def test_output_shape(self, fitted_vine, random_u4):
        from pyscarcopula.copula.vine import _edge_h

        e = vine_rosenblatt(random_u4, fitted_vine.edges, _edge_h)
        assert e.shape == random_u4.shape

    def test_values_in_unit_interval(self, fitted_vine, random_u4):
        from pyscarcopula.copula.vine import _edge_h

        e = vine_rosenblatt(random_u4, fitted_vine.edges, _edge_h)
        assert np.all(e > 0)
        assert np.all(e < 1)

    def test_first_column_equals_first_variable(self, fitted_vine, random_u4):
        """e[:, 0] = u[:, 0] (first Rosenblatt residual)."""
        from pyscarcopula.copula.vine import _edge_h

        e = vine_rosenblatt(random_u4, fitted_vine.edges, _edge_h)
        np.testing.assert_allclose(e[:, 0], random_u4[:, 0], atol=1e-8)


# ══════════════════════════════════════════════════════════════════
# Top-level API tests
# ══════════════════════════════════════════════════════════════════

class TestAPI:

    def test_fit_mle(self):
        from pyscarcopula import GumbelCopula
        from pyscarcopula._types import MLEResult
        from pyscarcopula._utils import pobs

        rng = np.random.default_rng(0)
        data = rng.standard_normal((200, 2))
        u = pobs(data)

        cop = GumbelCopula(rotate=180)
        result = fit(cop, u, method='mle')

        assert isinstance(result, MLEResult)
        assert result.success
        assert result.log_likelihood > 0
        assert result.copula_param > 1.0

    def test_smoothed_params_mle(self):
        from pyscarcopula import GumbelCopula
        from pyscarcopula._utils import pobs

        rng = np.random.default_rng(0)
        u = pobs(rng.standard_normal((100, 2)))
        cop = GumbelCopula(rotate=180)

        result = fit(cop, u, method='mle')
        r_t = smoothed_params(cop, u, result)

        assert r_t.shape == (100,)
        # MLE: constant
        assert np.all(r_t == r_t[0])

    def test_fit_returns_immutable(self):
        from pyscarcopula import GumbelCopula
        from pyscarcopula._utils import pobs

        rng = np.random.default_rng(0)
        u = pobs(rng.standard_normal((100, 2)))
        cop = GumbelCopula(rotate=180)

        result = fit(cop, u, method='mle')

        with pytest.raises(Exception):
            result.log_likelihood = 999.0

    def test_configure_does_not_crash(self):
        configure(blas_threads=2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
