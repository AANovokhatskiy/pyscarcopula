"""Real-input contracts at sparse Jacobi public and native facades."""

import numpy as np
import pytest

from pyscarcopula._native import jacobi
from pyscarcopula.numerical.jacobi_sparse import (
    SparseJacobiTransition, sample_sparse_jacobi_trajectory,
)


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
