"""Test mathematical properties that any copula must satisfy."""
import numpy as np
import pytest


class TestCopulaPDF:
    """PDF properties: non-negativity, normalization."""

    def test_pdf_nonnegative(self, archimedean):
        cop, r = archimedean
        rng = np.random.default_rng(0)
        u1 = rng.uniform(0.01, 0.99, 500)
        u2 = rng.uniform(0.01, 0.99, 500)
        r_arr = np.full(500, r)
        pdf = cop.pdf(u1, u2, r_arr)
        assert np.all(pdf >= 0), f"Negative pdf values: min={pdf.min()}"

    def test_log_pdf_finite(self, archimedean):
        cop, r = archimedean
        rng = np.random.default_rng(1)
        u1 = rng.uniform(0.01, 0.99, 200)
        u2 = rng.uniform(0.01, 0.99, 200)
        r_arr = np.full(200, r)
        lp = cop.log_pdf(u1, u2, r_arr)
        assert np.all(np.isfinite(lp)), f"Non-finite log_pdf values"

    def test_pdf_log_pdf_consistent(self, archimedean):
        cop, r = archimedean
        rng = np.random.default_rng(2)
        u1 = rng.uniform(0.05, 0.95, 100)
        u2 = rng.uniform(0.05, 0.95, 100)
        r_arr = np.full(100, r)
        pdf = cop.pdf(u1, u2, r_arr)
        lp = cop.log_pdf(u1, u2, r_arr)
        np.testing.assert_allclose(np.log(pdf + 1e-300), lp, rtol=1e-8)


class TestHFunction:
    """h-function: range, monotonicity, inverse round-trip."""

    def test_h_in_unit_interval(self, archimedean):
        cop, r = archimedean
        u = np.linspace(0.01, 0.99, 50)
        v = np.full(50, 0.5)
        r_arr = np.full(50, r)
        h = cop.h(u, v, r_arr)
        assert np.all(h >= -1e-10) and np.all(h <= 1 + 1e-10), \
            f"h outside [0,1]: range [{h.min()}, {h.max()}]"

    def test_h_monotone_in_u(self, archimedean):
        cop, r = archimedean
        u = np.linspace(0.02, 0.98, 100)
        v = np.full(100, 0.5)
        r_arr = np.full(100, r)
        h = cop.h(u, v, r_arr)
        diffs = np.diff(h)
        # Should be monotonically non-decreasing
        assert np.all(diffs >= -1e-8), \
            f"h not monotone: min diff = {diffs.min()}"

    def test_h_inverse_roundtrip(self, archimedean):
        cop, r = archimedean
        rng = np.random.default_rng(10)
        u2 = rng.uniform(0.05, 0.95, 50)
        u1 = rng.uniform(0.05, 0.95, 50)
        r_arr = np.full(50, r)
        h_val = cop.h(u2, u1, r_arr)
        h_val = np.clip(h_val, 1e-10, 1 - 1e-10)
        u2_rec = cop.h_inverse(h_val, u1, r_arr)
        np.testing.assert_allclose(u2_rec, u2, atol=1e-4,
                                   err_msg="h_inverse(h(u2)) != u2")


class TestTransform:
    """Ψ: R → copula parameter domain."""

    def test_transform_monotone_positive(self, archimedean):
        """transform is monotonically increasing for x > 0."""
        cop, _ = archimedean
        x = np.linspace(0.1, 5, 100)
        r = cop.transform(x)
        diffs = np.diff(r)
        assert np.all(diffs >= -1e-10), \
            f"transform not monotone on x>0: min diff={diffs.min()}"

    def test_transform_positive_output(self, archimedean):
        """transform maps R to positive copula parameter domain."""
        cop, _ = archimedean
        x = np.linspace(-3, 3, 100)
        r = cop.transform(x)
        assert np.all(r > 0), f"transform produced non-positive: min={r.min()}"
