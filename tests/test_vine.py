"""Test C-vine copula: structure, truncation, independence selection."""
import numpy as np
import pytest
from pyscarcopula import CVineCopula, IndependentCopula
from pyscarcopula.stattests import gof_test
from pyscarcopula.utils import pobs


class TestVineStructure:
    """Basic vine structure properties."""

    @pytest.mark.parametrize("d", [3, 4, 6])
    def test_number_of_edges(self, d):
        """C-vine with d variables has d*(d-1)/2 edges."""
        u = pobs(np.random.default_rng(42).standard_normal((200, d)))
        vine = CVineCopula()
        vine.fit(u, method='mle')
        n_edges = sum(len(tree) for tree in vine.edges)
        assert n_edges == d * (d - 1) // 2, \
            f"Expected {d*(d-1)//2} edges, got {n_edges}"

    @pytest.mark.parametrize("d", [3, 4, 6])
    def test_number_of_trees(self, d):
        u = pobs(np.random.default_rng(42).standard_normal((200, d)))
        vine = CVineCopula()
        vine.fit(u, method='mle')
        assert len(vine.edges) == d - 1

    def test_edges_per_tree(self):
        """Tree j should have d-j-1 edges."""
        d = 6
        u = pobs(np.random.default_rng(42).standard_normal((200, d)))
        vine = CVineCopula()
        vine.fit(u, method='mle')
        for j, tree_edges in enumerate(vine.edges):
            expected = d - j - 1
            assert len(tree_edges) == expected, \
                f"Tree {j}: expected {expected} edges, got {len(tree_edges)}"


class TestVineTruncation:
    """Truncation reduces SCAR edges."""

    def test_truncation_level(self, crypto_data_6d):
        vine = CVineCopula()
        vine.fit(crypto_data_6d, method='scar-tm-ou',
                 K=50, tol=0.5, truncation_level=1)

        for j, tree_edges in enumerate(vine.edges):
            for edge in tree_edges:
                if j >= 1:
                    assert edge.fit_result.method.upper() == 'MLE', \
                        f"Tree {j} edge should be MLE, got {edge.fit_result.method}"

    def test_min_edge_logL(self, crypto_data_6d):
        vine = CVineCopula()
        vine.fit(crypto_data_6d, method='scar-tm-ou',
                 K=50, tol=0.5, min_edge_logL=50)

        n_mle = sum(1 for tree in vine.edges for e in tree
                    if e.fit_result.method.upper() == 'MLE')
        # With threshold=50, many edges should fall back to MLE
        assert n_mle > 0, "Expected some MLE edges with min_edge_logL=50"


class TestVineIndependence:
    """Independent copula selection for weak edges."""

    def test_independent_data_selects_independence(self):
        """On fully independent data, some edges should be IndependentCopula."""
        rng = np.random.default_rng(99)
        u = rng.uniform(0, 1, (500, 4))
        vine = CVineCopula()
        vine.fit(u, method='mle')

        n_indep = sum(1 for tree in vine.edges for e in tree
                      if isinstance(e.copula, IndependentCopula))
        # On fully independent data, most edges should be independent
        total = sum(len(tree) for tree in vine.edges)
        assert n_indep >= total // 2, \
            f"Expected majority independent on random data, got {n_indep}/{total}"


class TestVineGoF:
    """GoF test works on fitted vine (including mixed SCAR/MLE)."""

    def test_vine_gof_runs(self, crypto_data_6d):
        vine = CVineCopula()
        vine.fit(crypto_data_6d, method='mle')
        gof = gof_test(vine, crypto_data_6d, to_pobs=False)
        assert hasattr(gof, 'pvalue')
        assert 0 <= gof.pvalue <= 1


class TestVineSamplePredict:
    """Vine sample and predict produce correct shapes."""

    def test_predict_shape(self):
        d = 4
        u = pobs(np.random.default_rng(0).standard_normal((200, d)))
        vine = CVineCopula()
        vine.fit(u, method='mle')
        samples = vine.predict(100)
        assert samples.shape == (100, d)

    def test_sample_shape(self):
        d = 4
        u = pobs(np.random.default_rng(0).standard_normal((200, d)))
        vine = CVineCopula()
        vine.fit(u, method='mle')
        samples = vine.sample(100, u_train=u)
        assert samples.shape == (100, d)

    def test_samples_in_unit_cube(self):
        d = 4
        u = pobs(np.random.default_rng(0).standard_normal((200, d)))
        vine = CVineCopula()
        vine.fit(u, method='mle')
        samples = vine.predict(1000)
        assert np.all(samples >= 0) and np.all(samples <= 1)
