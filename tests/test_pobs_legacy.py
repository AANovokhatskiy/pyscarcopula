"""Historical outputs captured by executing the released 0.20.1 pobs."""
import json
from pathlib import Path

import numpy as np
import pytest

from pyscarcopula._utils import pobs


REFERENCES = json.loads((Path(__file__).parent / 'fixtures' /
                        'pobs_legacy_reference.json').read_text())['fixtures']


@pytest.mark.parametrize('case', REFERENCES, ids=lambda case: case['name'])
def test_legacy_matches_released_pobs_elementwise(case):
    data = np.array(case['input'])[:, None]
    expected = np.array(case['expected_pobs'])[:, None]
    original = data.copy()
    np.testing.assert_array_equal(pobs(data, ties_method='legacy'), expected)
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.int64, np.uint64, np.bool_])
def test_legacy_and_ordinal_have_explicitly_different_tie_contracts(dtype):
    data = np.ones((16, 1), dtype=dtype)
    historical = np.r_[np.arange(15, 0, -1), 16][:, None] / 17
    ordinal = np.arange(1, 17)[:, None] / 17
    np.testing.assert_array_equal(pobs(data, ties_method='legacy'), historical)
    np.testing.assert_array_equal(pobs(data), ordinal)
    assert not np.array_equal(historical, ordinal)


@pytest.mark.parametrize('dtype', [np.int64, np.uint64])
def test_legacy_preserves_integer_distinctions_above_float_precision(dtype):
    maximum = np.iinfo(dtype).max
    data = np.array([[maximum], [maximum-2], [maximum-1]], dtype=dtype)
    np.testing.assert_array_equal(pobs(data, ties_method='legacy'), [[.75], [.25], [.5]])


@pytest.mark.parametrize('shape', [(0, 0), (0, 3), (17, 0), (1, 3)])
def test_legacy_empty_and_single_observation_shapes(shape):
    values = pobs(np.zeros(shape), ties_method='legacy')
    np.testing.assert_array_equal(values, np.full(shape, .5))
