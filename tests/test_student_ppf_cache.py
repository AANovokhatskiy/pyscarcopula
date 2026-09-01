"""Student quantile cache bounds, ownership, and content-based reuse."""

import numpy as np
import pytest
from scipy.stats import t

from pyscarcopula.copula.multivariate.student_ppf_cache import (
    StudentPPFTable,
    prepare_student_ppf_cache,
)


@pytest.fixture
def observations():
    return np.array([[.2, .4, .7], [.3, .8, .6], [.7, .6, .3], [.4, .2, .8]])


@pytest.fixture
def cache(observations):
    return prepare_student_ppf_cache(None, observations, observations, 3)


@pytest.mark.parametrize("field", ["n_rows", "t_index", "max_rows", "expected_d"])
@pytest.mark.parametrize("bad", [1.8, True, np.float64(1), -1])
def test_cache_block_rejects_invalid_integer_options(cache, field, bad):
    kwargs = dict(n_rows=1, t_index=0, max_rows=4, expected_d=3)
    kwargs[field] = bad
    with pytest.raises((TypeError, ValueError), match=field):
        cache.block(**kwargs)


@pytest.mark.parametrize("kwargs", [
    dict(n_rows=1, t_index=4),
    dict(n_rows=0, t_index=5),
    dict(n_rows=2, t_index=1, max_rows=2),
    dict(n_rows=1, expected_d=2),
])
def test_cache_block_rejects_out_of_range_requests(cache, kwargs):
    with pytest.raises(ValueError):
        cache.block(**kwargs)


@pytest.mark.parametrize("method", ["cache", "table"])
@pytest.mark.parametrize("start,stop", [(3, 2), (-1, 2), (0, 5), (1.8, 3), (1, 3.8), (True, 3)])
def test_quantile_rows_reject_invalid_bounds(cache, method, start, stop):
    rows = cache.ppf_rows if method == "cache" else cache._ppf.rows
    with pytest.raises((TypeError, ValueError)):
        rows(5., start, stop)


def test_cache_blocks_accept_empty_end_and_numpy_indices(cache):
    assert cache.block(np.int64(0), np.int64(4), np.int64(4), np.int64(3)) == (4, 4)
    assert cache.block(0, max_rows=0) == (0, 0)
    assert cache.ppf_rows(5., 4, 4).shape == (0, 3)
    assert cache._ppf.rows(5., 4, 4).shape == (0, 3)
    np.testing.assert_allclose(cache.ppf_rows(5., 1, 3), cache.ppf(5.)[1:3])


@pytest.mark.parametrize("attribute", ["u_snapshot", "ppf_nodes", "ppf_table"])
def test_factory_cache_state_rejects_in_place_mutation(cache, observations, attribute):
    df = float(cache.ppf_nodes[15])
    expected = t.ppf(observations, df)
    values = getattr(cache, attribute)
    with pytest.raises(ValueError, match="read-only"):
        values.flat[0] = 123.
    reused = prepare_student_ppf_cache(cache, observations, observations.copy(), 3)
    assert reused is cache
    np.testing.assert_allclose(reused.ppf(df), expected, rtol=2e-10, atol=2e-10)


@pytest.mark.parametrize("table_bytes", [0, 1024 * 1024])
def test_table_freezes_arrays_for_cached_and_exact_evaluation(observations, table_bytes):
    table = StudentPPFTable(observations, max_table_bytes=table_bytes)
    arrays = [table.u, table.nodes] + ([] if table.table is None else [table.table])
    for array in arrays:
        assert not array.flags.writeable
    assert (table.table is None) == (table_bytes == 0)
    df = float(table.nodes[15])
    expected = t.ppf(observations, df)
    observations[:] = .5
    np.testing.assert_allclose(table(df), expected, rtol=2e-10, atol=2e-10)


def test_cache_freezes_arrays_from_custom_table_factory(observations):
    def factory(snapshot):
        table = StudentPPFTable(snapshot)
        table.nodes = table.nodes.copy()
        table.table = table.table.copy()
        return table

    cache = prepare_student_ppf_cache(None, observations, observations, 3, factory)
    assert not cache.ppf_nodes.flags.writeable
    assert not cache.ppf_table.flags.writeable
    assert cache.ppf_nodes is cache._ppf.nodes
    assert cache.ppf_table is cache._ppf.table


def test_cache_uses_content_and_isolated_snapshot_for_reuse(cache, observations):
    original = observations.copy()
    assert cache.matches(object(), original)
    observations[0, 0] = .6
    assert not cache.matches(observations, observations)
    np.testing.assert_array_equal(cache.u_snapshot, original)
    replaced = prepare_student_ppf_cache(cache, observations, observations, 3)
    assert replaced is not cache
    assert replaced.version > cache.version


@pytest.mark.parametrize("operation", ["prepare", "matches"])
@pytest.mark.parametrize("representation", ["complex", "object"])
def test_cache_cannot_match_complex_values_after_loss(cache, observations, operation, representation):
    values = observations.astype(complex) + .13j
    if representation == "object":
        values = values.astype(object)
    with pytest.raises(TypeError, match="real"):
        if operation == "prepare":
            prepare_student_ppf_cache(cache, observations, values, 3)
        else:
            cache.matches(observations, values)
