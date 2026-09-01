"""Real-input contracts at sparse Jacobi public and native facades."""

import numpy as np
import pytest

from pyscarcopula._native import jacobi
from pyscarcopula.numerical.jacobi_sparse import (
    SparseJacobiTransition, sample_sparse_jacobi_trajectory,
)
from pyscarcopula.numerical._arrays import as_integer_array


def _inputs():
    return dict(tau=np.array([.2, .8]), weights=np.array([.4, .6]),
                indices=np.array([[0, 1], [0, 1]]),
                probabilities=np.array([[.8, .2], [.3, .7]]),
                counts=np.array([2, 2]), uniforms=np.array([.2, .9, .3]))


@pytest.mark.parametrize('entry', ['probabilities', 'vector', 'tau', 'weights'])
@pytest.mark.parametrize('dtype', [np.complex128, object])
def test_sparse_public_inputs_reject_complex_before_rng(entry, dtype):
    values = _inputs()
    transition = SparseJacobiTransition(
        values['indices'], values['probabilities'], values['counts'])
    rng = np.random.default_rng(71)
    initial = rng.bit_generator.state
    value = np.array([.4, .6]) if entry == 'vector' else values[entry]
    complex_value = (value.astype(complex) + .2j).astype(dtype)
    with pytest.raises(TypeError, match=entry):
        if entry == 'probabilities':
            SparseJacobiTransition(values['indices'], complex_value, values['counts'])
        elif entry == 'vector':
            transition.left_multiply(complex_value)
        else:
            values[entry] = complex_value
            sample_sparse_jacobi_trajectory(values['tau'], values['weights'],
                                            transition, 3, rng=rng)
    assert rng.bit_generator.state == initial


@pytest.mark.parametrize('entry', ['tau', 'weights', 'probabilities', 'uniforms'])
def test_sparse_native_sampler_rejects_complex(entry):
    values = _inputs()
    values[entry] = values[entry].astype(complex) + .2j
    with pytest.raises(TypeError, match=entry):
        jacobi.sample_prepared_sparse_trajectory_fixed_draws(**values)


@pytest.mark.parametrize('operation', [
    'validate_sparse_transition', 'sparse_to_dense', 'sparse_left_multiply',
])
def test_sparse_native_operations_reject_complex_probabilities(operation):
    inputs = _inputs()
    args = [inputs['indices'], inputs['probabilities'].astype(complex) + .2j,
            inputs['counts']]
    if operation == 'sparse_left_multiply':
        args.append(np.array([.4, .6]))
    with pytest.raises(TypeError, match='probabilities'):
        getattr(jacobi, operation)(*args)


def test_sparse_native_matvec_rejects_complex_values():
    inputs = _inputs()
    with pytest.raises(TypeError, match='values'):
        jacobi.sparse_left_multiply(inputs['indices'], inputs['probabilities'],
                                   inputs['counts'], np.array([.4 + .2j, .6 + .3j]))


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_real_sparse_inputs_preserve_native_values(dtype):
    inputs = _inputs()
    # Exactly representable row sums also satisfy the native float64 invariant.
    probabilities = np.array([[.75, .25], [.5, .5]], dtype=dtype)
    transition = SparseJacobiTransition(inputs['indices'], probabilities, inputs['counts'])
    vector = np.array([.4, .6], dtype=dtype)
    actual = transition.left_multiply(vector)
    expected = vector.astype(np.float64) @ probabilities.astype(np.float64)
    np.testing.assert_allclose(actual, expected, atol=1e-15, rtol=1e-15)
    np.testing.assert_array_equal(transition.to_dense(), probabilities.astype(np.float64))


SPARSE_STRUCTURAL_ENTRIES = (
    'public', 'validate_sparse_transition', 'sparse_to_dense',
    'sparse_left_multiply', 'sample_prepared_sparse_trajectory_fixed_draws',
    'sparse_full_horizon_diagnostics',
)


def _call_sparse(entry, values):
    structural = {name: values[name] for name in ('indices', 'probabilities', 'counts')}
    if entry == 'public':
        return SparseJacobiTransition(**structural)
    if entry == 'sample_prepared_sparse_trajectory_fixed_draws':
        return jacobi.sample_prepared_sparse_trajectory_fixed_draws(**values)
    if entry == 'sparse_full_horizon_diagnostics':
        return jacobi.sparse_full_horizon_diagnostics(
            2., .4, .3, values['tau'], values['weights'], **structural, steps=2)
    if entry == 'sparse_left_multiply':
        structural['values'] = np.array([.4, .6])
    return getattr(jacobi, entry)(**structural)


@pytest.mark.parametrize('entry', SPARSE_STRUCTURAL_ENTRIES)
@pytest.mark.parametrize('field', ['indices', 'counts'])
@pytest.mark.parametrize('kind', [
    'complex', 'object_complex', 'fractional', 'integral_float',
    'bool', 'object_bool', 'mixed_bool_list', 'object_timedelta',
])
def test_sparse_structural_buffers_reject_nonintegers(entry, field, kind):
    inputs = _inputs()
    values = inputs[field]
    if kind in ('complex', 'object_complex'):
        invalid = values.astype(complex) + .2j
        if kind == 'object_complex':
            invalid = invalid.astype(object)
    elif kind in ('fractional', 'integral_float'):
        invalid = values.astype(float) + (.2 if kind == 'fractional' else 0.)
    elif kind == 'bool':
        invalid = values.astype(bool)
    else:
        invalid = values.astype(object)
        invalid.flat[0] = np.timedelta64(1, 'ns') if kind == 'object_timedelta' else True
        if kind == 'mixed_bool_list':
            invalid = invalid.tolist()
    inputs[field] = invalid
    with pytest.raises(TypeError, match=f'{field} must contain integer'):
        _call_sparse(entry, inputs)


@pytest.mark.parametrize('entry', SPARSE_STRUCTURAL_ENTRIES)
@pytest.mark.parametrize('field', ['indices', 'counts'])
@pytest.mark.parametrize('bad,dtype', [
    (2 ** 63, np.uint64), (2 ** 64 - 1, np.uint64),
    (2 ** 63, object), (-(2 ** 63) - 1, object),
])
def test_sparse_structural_buffers_reject_integer_overflow(entry, field, bad, dtype):
    inputs = _inputs()
    invalid = inputs[field].astype(dtype)
    invalid.flat[0] = bad
    inputs[field] = invalid
    with pytest.raises(ValueError, match=f'{field} values must fit'):
        _call_sparse(entry, inputs)


@pytest.mark.parametrize('dtype', [np.int8, np.int32, np.int64, np.uint64, object])
def test_sparse_structural_integer_types_preserve_values(dtype):
    inputs = _inputs()
    inputs['indices'] = inputs['indices'].astype(dtype)
    inputs['counts'] = inputs['counts'].astype(dtype)
    transition = _call_sparse('public', inputs)
    vector = np.array([.4, .6])
    expected = vector @ inputs['probabilities']
    np.testing.assert_allclose(transition.left_multiply(vector), expected, atol=1e-15)
    np.testing.assert_array_equal(transition.to_dense(), inputs['probabilities'])
    for entry in SPARSE_STRUCTURAL_ENTRIES[1:]:
        _call_sparse(entry, inputs)


def test_sparse_structural_readonly_strided_arrays_retain_storage():
    inputs = _inputs()
    indices_storage = np.zeros((2, 4), dtype=np.intp)
    indices_storage[:, ::2] = inputs['indices']
    counts_storage = np.zeros(4, dtype=np.intp)
    counts_storage[::2] = inputs['counts']
    inputs['indices'] = indices_storage[:, ::2]
    inputs['counts'] = counts_storage[::2]
    inputs['indices'].setflags(write=False)
    inputs['counts'].setflags(write=False)
    transition = _call_sparse('public', inputs)
    assert transition.indices is inputs['indices']
    assert transition.counts is inputs['counts']
    np.testing.assert_array_equal(transition.to_dense(), inputs['probabilities'])
    for entry in SPARSE_STRUCTURAL_ENTRIES[1:]:
        _call_sparse(entry, inputs)
    assert not inputs['indices'].flags.writeable
    assert not inputs['counts'].flags.writeable


def test_sparse_structural_padding_preserves_native_domain_policy():
    inputs = _inputs()
    inputs.update(indices=np.array([[0, -1], [1, -1]]),
                  probabilities=np.array([[1., 0.], [1., 0.]]), counts=np.array([1, 1]))
    transition = _call_sparse('public', inputs)
    np.testing.assert_array_equal(transition.to_dense(), np.eye(2))
    for entry in SPARSE_STRUCTURAL_ENTRIES[1:]:
        _call_sparse(entry, inputs)


@pytest.mark.parametrize('dtype', [np.int32, np.int64])
def test_integer_array_exact_limits_and_object_python_integers(dtype):
    limits = np.iinfo(dtype)
    source = [limits.min, np.int32(0), np.uint64(limits.max)]
    actual = as_integer_array(source, dtype=dtype)
    assert actual.dtype == np.dtype(dtype)
    assert actual.tolist() == [limits.min, 0, limits.max]
    for invalid in ([limits.min - 1], [limits.max + 1]):
        with pytest.raises(ValueError, match='values must fit'):
            as_integer_array(invalid, dtype=dtype)
