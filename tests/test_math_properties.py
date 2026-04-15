"""Test mathematical properties that any copula must satisfy."""
import numpy as np
import pytest

from pyscarcopula import GumbelCopula, FrankCopula, JoeCopula


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


class TestGumbelSampling:
    """Verify Gumbel V (frailty variable) sampling from stable distribution."""

    @pytest.mark.parametrize("r", [1.5, 2.0, 5.0, 10.0, 20.0])
    def test_V_positive(self, r):
        """V samples must be strictly positive."""
        cop = GumbelCopula()
        V = cop.V(10000, r)
        assert np.all(V > 0), f"Negative V at r={r}: min={V.min()}"

    @pytest.mark.parametrize("r", [1.5, 2.0, 5.0])
    def test_V_laplace_transform(self, r):
        """Empirical check: E[exp(-tV)] ≈ exp(-t^(1/r)) for several t."""
        cop = GumbelCopula()
        V = cop.V(100000, r)
        alpha = 1.0 / r
        for t in [0.5, 1.0, 2.0]:
            empirical = np.mean(np.exp(-t * V))
            theoretical = np.exp(-(t ** alpha))
            np.testing.assert_allclose(empirical, theoretical, rtol=0.05,
                                       err_msg=f"Laplace transform mismatch at t={t}, r={r}")

    def test_sample_unit_interval(self):
        """Gumbel samples should be in (0,1)^2."""
        cop = GumbelCopula()
        samples = cop.sample(5000, 3.0)
        assert np.all(samples > 0) and np.all(samples < 1)


class TestHInverseBoundary:
    """Boundary and stress tests for h_inverse across rotations."""

    @pytest.mark.parametrize("rot", [0, 90, 180, 270])
    def test_gumbel_h_inverse_all_rotations(self, rot):
        cop = GumbelCopula(rotate=rot)
        r = 2.5
        rng = np.random.default_rng(42)
        u = rng.uniform(0.05, 0.95, 100)
        v = rng.uniform(0.05, 0.95, 100)
        r_arr = np.full(100, r)
        h_val = cop.h(u, v, r_arr)
        h_val = np.clip(h_val, 1e-10, 1 - 1e-10)
        u_rec = cop.h_inverse(h_val, v, r_arr)
        np.testing.assert_allclose(u_rec, u, atol=1e-4,
                                   err_msg=f"h_inverse roundtrip failed for rot={rot}")

    @pytest.mark.parametrize("rot", [0, 90, 180, 270])
    def test_joe_h_inverse_all_rotations(self, rot):
        cop = JoeCopula(rotate=rot)
        r = 2.5
        rng = np.random.default_rng(42)
        u = rng.uniform(0.05, 0.95, 100)
        v = rng.uniform(0.05, 0.95, 100)
        r_arr = np.full(100, r)
        h_val = cop.h(u, v, r_arr)
        h_val = np.clip(h_val, 1e-10, 1 - 1e-10)
        u_rec = cop.h_inverse(h_val, v, r_arr)
        np.testing.assert_allclose(u_rec, u, atol=1e-4,
                                   err_msg=f"h_inverse roundtrip failed for rot={rot}")

    @pytest.mark.parametrize("r", [1.001, 1.5, 3.0, 5.0])
    def test_gumbel_h_inverse_param_range(self, r):
        cop = GumbelCopula()
        rng = np.random.default_rng(42)
        u = rng.uniform(0.05, 0.95, 50)
        v = rng.uniform(0.05, 0.95, 50)
        r_arr = np.full(50, r)
        h_val = cop.h(u, v, r_arr)
        h_val = np.clip(h_val, 1e-10, 1 - 1e-10)
        u_rec = cop.h_inverse(h_val, v, r_arr)
        np.testing.assert_allclose(u_rec, u, atol=1e-3,
                                   err_msg=f"h_inverse roundtrip failed for r={r}")


class TestFrankStability:
    """Edge-case stability for Frank copula numerical routines."""

    def test_frank_log_pdf_small_param(self):
        """Frank log_pdf should be finite for very small r."""
        cop = FrankCopula()
        u1 = np.array([0.3, 0.5, 0.7])
        u2 = np.array([0.4, 0.6, 0.8])
        r_arr = np.full(3, 0.001)
        lp = cop.log_pdf(u1, u2, r_arr)
        assert np.all(np.isfinite(lp)), f"Non-finite log_pdf at r=0.001: {lp}"

    def test_frank_log_pdf_large_param(self):
        """Frank log_pdf should be finite for large r."""
        cop = FrankCopula()
        u1 = np.array([0.3, 0.5, 0.7])
        u2 = np.array([0.4, 0.6, 0.8])
        r_arr = np.full(3, 100.0)
        lp = cop.log_pdf(u1, u2, r_arr)
        assert np.all(np.isfinite(lp)), f"Non-finite log_pdf at r=100: {lp}"

    def test_frank_h_extreme_v(self):
        """Frank h should handle v near 0 and 1."""
        cop = FrankCopula()
        u = np.array([0.5, 0.5, 0.5, 0.5])
        v = np.array([0.001, 0.01, 0.99, 0.999])
        r_arr = np.full(4, 5.0)
        h = cop.h(u, v, r_arr)
        assert np.all(np.isfinite(h)), f"Non-finite h values: {h}"
        assert np.all(h > 0) and np.all(h < 1), f"h outside (0,1): {h}"


class TestJoeHFormula:
    """Verify Joe h-function matches the analytical derivative of C."""

    @pytest.mark.parametrize("r", [1.5, 3.0, 8.0])
    def test_joe_h_matches_analytical(self, r):
        """h(u,v;r) = -(q0-1) * B^(1/r-1) * q1 / (1-v)
        where q0=(1-u)^r, q1=(1-v)^r, B=q0+q1-q0*q1."""
        cop = JoeCopula()
        u = np.array([0.2, 0.3, 0.5, 0.7, 0.8])
        v = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
        r_arr = np.full(len(u), r)
        h = cop.h(u, v, r_arr)
        for i in range(len(u)):
            q0 = (1 - u[i]) ** r
            q1 = (1 - v[i]) ** r
            B = q0 + q1 - q0 * q1
            expected = -(q0 - 1.0) * B ** (1.0 / r - 1.0) * q1 / (1 - v[i])
            np.testing.assert_allclose(h[i], expected, rtol=1e-8,
                                       err_msg=f"Joe h mismatch at u={u[i]}, v={v[i]}, r={r}")
