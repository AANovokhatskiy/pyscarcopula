import numpy as np
import pytest
from scipy.stats import kendalltau

from pyscarcopula._native import statistics


def test_native_reductions_and_information_criteria():
    values = np.array([1.25, -0.5, 2.0, -0.25])
    assert statistics.sum_values(values) == np.sum(values)
    assert statistics.sum_values([]) == 0.0
    assert statistics.sum_int64([3, 0, 7]) == 10
    assert statistics.sum_absolute(values) == 4.0
    assert statistics.add_scores(1.25, -0.5) == 0.75
    assert statistics.information_criterion(-12.5, 3, 100, "aic") == 31.0
    assert statistics.information_criterion(
        -12.5, 3, 100, "bic") == -2.0 * -12.5 + 3.0 * np.log(100)
    assert statistics.candidate_score(
        -12.5, 3, 100, "loglik") == 12.5
    with pytest.raises(ValueError, match="unknown information criterion"):
        statistics.information_criterion(-1.0, 1, 10, "invalid")


def test_native_dense_ranks_and_matrix_reject_ties():
    values = np.array([0.4, 0.1, 0.3, 0.2])
    np.testing.assert_array_equal(
        statistics.dense_ranks_no_ties(values), [4, 1, 3, 2])
    matrix = np.column_stack((values, values[::-1]))
    np.testing.assert_array_equal(
        statistics.dense_rank_matrix_no_ties(matrix),
        [[4, 2], [1, 3], [3, 1], [2, 4]],
    )
    assert statistics.dense_ranks_no_ties([0.1, 0.1, 0.2]) is None
    assert statistics.dense_rank_matrix_no_ties(
        [[0.1, 0.2], [0.1, 0.3]]) is None
    assert statistics.dense_ranks_no_ties([0.1, np.nan]) is None


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ([1.0, 2.0, 3.0, 4.0], [1.0, 3.0, 2.0, 4.0]),
        ([1.0, 1.0, 2.0, 3.0], [1.0, 2.0, 2.0, 3.0]),
        ([3.0, 1.0, 4.0, 2.0], [1.0, 4.0, 2.0, 3.0]),
    ],
)
def test_native_kendall_tau_matches_scipy(first, second):
    expected = kendalltau(first, second).statistic
    assert statistics.kendall_tau(first, second) == pytest.approx(
        expected, rel=1e-15, abs=1e-15)


def test_native_kendall_dense_fast_path_and_degenerate_values():
    first = np.array([4, 1, 3, 2])
    second = np.array([1, 4, 2, 3])
    expected = kendalltau(first, second).statistic
    assert statistics.kendall_tau_from_dense_ranks(
        first, second) == pytest.approx(expected, rel=1e-15, abs=1e-15)
    assert np.isnan(statistics.kendall_tau([1.0, 1.0], [1.0, 2.0]))
    assert np.isnan(statistics.kendall_tau([1.0, np.nan], [1.0, 2.0]))


def test_native_selection_predicates():
    assert statistics.tau_for_itau(-0.4, preserve_sign=True) == -0.4
    assert statistics.tau_for_itau(-0.4, preserve_sign=False) == 0.4
    assert statistics.tau_for_itau(0.0, preserve_sign=True) is None
    assert statistics.tau_for_itau(1.0, preserve_sign=False) == 1.0
    assert statistics.tau_for_itau(-1.0, preserve_sign=False) == 1.0
    assert statistics.tau_for_itau(-1.0, preserve_sign=True) == -1.0
    assert statistics.tau_for_itau(1.0, preserve_sign=True) == 1.0
    assert statistics.tau_for_itau(1.01, preserve_sign=True) is None
    assert statistics.rotation_compatible(0.1, 0)
    assert statistics.rotation_compatible(0.5, 180)
    assert statistics.rotation_compatible(-0.5, 90)
    assert not statistics.rotation_compatible(-0.5, 0)
    assert statistics.absolute_below(-0.1, 0.15)
    assert statistics.absolute_value(-0.5) == 0.5
    assert statistics.is_finite(1.0)
    assert statistics.is_nan(np.nan)
