"""Historical transform numerics are part of GAS optimizer compatibility."""

import math

import numpy as np
import pytest

from pyscarcopula._native import _extension


def _softplus_spec():
    module = _extension.load()
    spec = module.CopulaSpec()
    spec.family = module.CopulaFamily.Clayton
    spec.transform = module.Transform.Softplus
    # Zero offset exposes the primitive, including its tiny negative tail.
    spec.offset = 0.0
    return module, spec


def _hex_values(values):
    return np.array([float.fromhex(value) for value in values.split()])


# Frozen outputs of the 0.20.1 native implementation, independently built from
# its release source. A two-ULP allowance accommodates platform math libraries;
# the historical tail branches differ from algebraic rewrites by far more.
@pytest.mark.parametrize(
    "operation,inputs,expected",
    [
        (
            "copula_transform",
            "-0x1.5ep+9 -0x1.4000000000001p+4 -0x1.4p+4 "
            "-0x1.3ffffffffffffp+4 -0x1.4cccccccccccdp+0 0x0p+0 "
            "0x1.4cccccccccccdp+0 0x1.3ffffffffffffp+4 0x1.4p+4 "
            "0x1.4000000000001p+4 0x1.5ep+9",
            "0x1.14f2b0fb9307fp-1010 0x1.1b48655f37255p-29 "
            "0x1.1b48655a5141ep-29 0x1.1b48655a51430p-29 "
            "0x1.ed95d71a2d859p-3 0x1.62e42fefa39efp-1 "
            "0x1.8a7f87b0127d8p+0 0x1.400000008da42p+4 "
            "0x1.400000008da43p+4 0x1.4000000000001p+4 0x1.5ep+9",
        ),
        (
            "copula_dtransform",
            "-0x1.5ep+9 -0x1.4000000000001p+4 -0x1.4p+4 "
            "-0x1.3ffffffffffffp+4 -0x1.4cccccccccccdp+0 0x0p+0 "
            "0x1.4cccccccccccdp+0 0x1.3ffffffffffffp+4 0x1.4p+4 "
            "0x1.4000000000001p+4 0x1.5ep+9",
            "0x1.14f2b0fb9307fp-1010 0x1.1b48655f37255p-29 "
            "0x1.1b4865556b5d5p-29 0x1.1b4865556b5e6p-29 "
            "0x1.b69c25fe3c688p-3 0x1p-1 0x1.9258f68070e5ep-1 "
            "0x1.ffffffee4b79ap-1 0x1.ffffffee4b79ap-1 0x1p+0 0x1p+0",
        ),
        (
            "copula_inverse_transform",
            "0x0p+0 0x1.56e1fc2f8f359p-997 0x1.5798ee2308c39p-27 "
            "0x1.5798ee2308c3ap-27 0x1.5798ee2308c3bp-27 "
            "0x1.6666666666666p-1 0x1.4cccccccccccdp+0 "
            "0x1.3ffffffffffffp+4 0x1.4p+4 0x1.4000000000001p+4 0x1.5ep+9",
            "-0x1.5963447f87fb5p+9 -0x1.5963447f87fb5p+9 "
            "-0x1.26bb1bbb55516p+4 -0x1.26bb1bb9fdb87p+4 "
            "-0x1.26bb1bb9fdb87p+4 0x1.bf93f91df85fcp-7 "
            "0x1.f6b0753c6cc16p-1 0x1.3fffffff725bcp+4 "
            "0x1.3fffffff725bdp+4 0x1.4000000000001p+4 0x1.5ep+9",
        ),
    ],
)
def test_softplus_preserves_release_boundary_values(operation, inputs, expected):
    module, spec = _softplus_spec()
    actual = getattr(module, operation)(spec, _hex_values(inputs))
    np.testing.assert_array_max_ulp(actual, _hex_values(expected), maxulp=2)


def test_softplus_tail_approximations_have_bounded_analytic_error():
    module, spec = _softplus_spec()
    x = np.array([-700., -30., -20.00001, -20., -1., 0., 1., 20., 20.00001, 30., 700.])
    actual = module.copula_transform(spec, x)
    derivative = module.copula_dtransform(spec, x)
    analytic = np.logaddexp(0.0, x)
    # The historical threshold is an approximation, with a small jump at 20.
    np.testing.assert_allclose(actual, analytic, rtol=0, atol=2.062e-9)
    np.testing.assert_allclose(
        derivative, np.exp(-np.logaddexp(0.0, -x)), rtol=0, atol=2.062e-9)
    assert np.all(np.isfinite(actual))
    assert np.all(np.isfinite(derivative))
    assert np.all(derivative >= 0.0)
    assert np.all(derivative <= 1.0)
    restored = module.copula_inverse_transform(spec, actual)
    # The small-y inverse approximation contributes less than 5e-9.
    np.testing.assert_allclose(restored, x, rtol=0, atol=7.1e-9)


def test_softplus_derivative_matches_value_away_from_legacy_thresholds():
    module, spec = _softplus_spec()
    x = np.array([-30., -10., -1., 0., 1., 10., 30.])
    h = 1e-4
    difference = (
        module.copula_transform(spec, x + h)
        - module.copula_transform(spec, x - h)
    ) / (2 * h)
    np.testing.assert_allclose(
        module.copula_dtransform(spec, x), difference, rtol=2e-8, atol=1e-10)


def test_softplus_special_values_and_public_inverse_floor_are_unchanged():
    module, spec = _softplus_spec()
    for operation in (
            module.copula_transform, module.copula_dtransform,
            module.copula_inverse_transform):
        for value in (-np.inf, np.inf, np.nan):
            with pytest.raises(ValueError, match="finite"):
                operation(spec, np.array([value]))
    np.testing.assert_equal(
        module.copula_inverse_transform(spec, np.array([-1., 0.])),
        [math.log(1e-300), math.log(1e-300)],
    )
    spec.family = module.CopulaFamily.Student
    spec.offset = 2.0 + 1e-6
    actual = module.copula_inverse_transform(spec, np.array([spec.offset - 1., spec.offset]))
    np.testing.assert_array_equal(actual, [math.log(1e-15)] * 2)
