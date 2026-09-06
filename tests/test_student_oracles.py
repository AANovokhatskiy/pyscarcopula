"""Independent checks for the Student quantile test reference."""

import numpy as np
import pytest

from student_oracles import student_quantile_beta_oracle


@pytest.mark.parametrize("df", [1.0, 2.0])
def test_beta_quantile_oracle_matches_closed_forms(df):
    probabilities = np.array([1e-10, 0.01, 0.25, 0.4999, 0.5, 0.75, 0.99])
    if df == 1.0:
        tail = np.minimum(probabilities, 1.0 - probabilities)
        expected = np.copysign(1.0 / np.tan(np.pi * tail), probabilities - 0.5)
        expected[probabilities == 0.5] = 0.0
    else:
        expected = (2.0 * probabilities - 1.0) / np.sqrt(
            2.0 * probabilities * (1.0 - probabilities))
    np.testing.assert_allclose(
        student_quantile_beta_oracle(df, probabilities), expected,
        rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize("df,probability,expected", [
    (0.02, 0.1, -6.331487481607516186916025715867582e33),
    (3.0, 0.01, -4.540702858568133520152937821043665),
    (999.0, 1e-10, -6.427943479659739275014231743229003),
    (1e6, 1e-10, -6.361406848876742619639450684367299),
    (1e6, 0.4999, -0.0002506628927537267849996069670540534),
])
def test_beta_quantile_oracle_matches_high_precision_values(df, probability, expected):
    # Independently computed with mpmath at 80 decimal digits, using exact
    # float inputs: solve betainc(df/2, .5, 0, df/(df+x*x), regularized=True)
    # == 2*p for x>0, then negate. No native or stdtrit outputs are frozen here.
    np.testing.assert_allclose(
        student_quantile_beta_oracle(df, [probability]), [expected],
        rtol=2e-13, atol=0.0)


@pytest.mark.parametrize("df", [1e-4, 1e-3, 0.01, 0.02, 0.05])
def test_beta_quantile_oracle_preserves_finite_tail_endpoint(df):
    maximum = np.sqrt(df / np.finfo(np.float64).tiny)
    actual = student_quantile_beta_oracle(df, [1e-10, 1.0 - 1e-10])
    np.testing.assert_array_equal(actual, [-maximum, maximum])
