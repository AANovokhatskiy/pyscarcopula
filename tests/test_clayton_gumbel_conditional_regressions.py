"""Independent generator-equation oracles for conditional inversion tails."""

import json
from pathlib import Path

import numpy as np
import pytest

from pyscarcopula import ClaytonCopula, GumbelCopula
from pyscarcopula._native import pair


CASES = json.loads((Path(__file__).parent / "fixtures" /
                   "clayton_gumbel_conditional_reference.json").read_text())["cases"]
FAMILIES = {"Clayton": ClaytonCopula, "Gumbel": GumbelCopula}
LOW = np.nextafter(0.0, 1.0)
HIGH = np.nextafter(1.0, 0.0)
GRID = np.array([LOW, 1e-300, 1e-12, 0.01, 0.5, 0.99, 1 - 1e-12, HIGH])


@pytest.mark.parametrize("case", CASES)
def test_inverse_against_independent_decimal_generator_oracle(case):
    # Rotation 90 implements 1 - H^{-1}(1-q | given). The fixtures compute
    # that complement at 800 decimal digits before converting it to float.
    model = FAMILIES[case["family"]](rotate=90 if case["reflected"] else 0)
    actual = pair.h_inverse(model, [case["q"]], [case["given"]], [case["theta"]])[0]
    expected = case["expected"]
    assert np.isfinite(actual)
    tolerance = 4e-15 if expected > 1e-8 else 0.0
    assert actual == pytest.approx(expected, rel=3e-11, abs=tolerance)
    if case["reflected"] and case["q"] < 1e-100:
        recovered = pair.h(model, [actual], [case["given"]], [case["theta"]])[0]
        assert recovered == pytest.approx(case["q"], rel=3e-10, abs=0)


@pytest.mark.parametrize("factory", [ClaytonCopula, GumbelCopula])
@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize("theta", [2.0, 100.0, 1000.0, 1e308])
def test_conditional_quantiles_are_finite_and_monotone_on_full_float_tails(
        factory, rotation, theta):
    model = factory(rotate=rotation)
    for given in GRID:
        values = pair.h_inverse(model, GRID, np.full(GRID.size, given), [theta])
        assert np.all(np.isfinite(values))
        assert np.all((values >= 0) & (values <= 1))
        # At very large theta multiple probabilities necessarily map to the
        # same representable coordinate; strict monotonicity is impossible.
        assert np.all(np.diff(values) >= -2e-15)


@pytest.mark.parametrize("factory,theta", [(ClaytonCopula, 0.0), (GumbelCopula, 1.0)])
@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_independence_preserves_tail_probabilities(factory, theta, rotation):
    model = factory(rotate=rotation)
    values = pair.h_inverse(model, GRID, np.full(GRID.size, .43), [theta])
    np.testing.assert_allclose(values, GRID, rtol=2e-15, atol=0)


@pytest.mark.parametrize("factory", [ClaytonCopula, GumbelCopula])
@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_large_parameter_limit_preserves_given_coordinate(factory, rotation):
    given = np.array([1e-12, .01, .5, .99, 1-1e-12])
    values = pair.h_inverse(factory(rotate=rotation), np.full(5, .37), given, [1e308])
    expected = 1-given if rotation in (90, 270) else given
    np.testing.assert_allclose(values, expected, rtol=2e-12, atol=3e-15)


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_gumbel_exhausted_inverse_budget_raises(rotation):
    with pytest.raises(RuntimeError, match="Gumbel conditional inverse did not converge"):
        pair.conditional_sample_from_uniforms(
            GumbelCopula(rotate=rotation), np.array([.173, .421, .793]), 2.3,
            given_coordinate=0, given_value=.43,
            bisection_tol=1e-13, bisection_maxiter=1)
