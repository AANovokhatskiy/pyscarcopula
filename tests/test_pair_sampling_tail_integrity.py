"""Sampling integration checks independent of the h/inverse implementation."""
import numpy as np
import pytest

from pyscarcopula import ClaytonCopula, FrankCopula, GumbelCopula, JoeCopula
from pyscarcopula._native import pair

_SUPPORTED = [(family, rotation)
              for family in (ClaytonCopula, FrankCopula, GumbelCopula, JoeCopula)
              for rotation in (0, 90, 180, 270)
              if family is not FrankCopula or rotation == 0]


@pytest.mark.parametrize("family,rotation", _SUPPORTED)
@pytest.mark.parametrize("theta", [2., 100., 1000.])
def test_strong_dependence_samples_retain_uniform_margins(family, rotation, theta):
    model = family(rotate=rotation)
    values = model.sample_at_parameter(16000, theta, np.random.default_rng(99312))
    assert np.all(np.isfinite(values))
    assert np.all((values > 0) & (values < 1))
    # A copula has uniform margins regardless of its dependence parameter.
    ordered = np.sort(values, axis=0)
    rank = np.arange(1, len(values) + 1)[:, None] / len(values)
    distance = max(np.max(rank - ordered), np.max(ordered - rank + 1 / len(values)))
    assert distance < .018
    # The old inverse created large artificial atoms at this numerical floor.
    assert not np.any((values == 1e-10) | (values == 1 - 1e-10))


@pytest.mark.parametrize("family,rotation", _SUPPORTED)
def test_fixed_uniform_tail_transform_is_finite_and_open(family, rotation):
    grid = np.array([np.nextafter(0., 1.), 1e-300, 1e-12, .01, .5,
                     .99, 1 - 1e-12, np.nextafter(1., 0.)])
    first, quantile = np.meshgrid(grid, grid)
    draws = np.column_stack((first.ravel(), quantile.ravel()))
    values = pair.sample_from_uniforms(family(rotate=rotation), draws, [1000.])
    np.testing.assert_array_equal(values[:, 0], draws[:, 0])
    assert np.all(np.isfinite(values))
    assert np.all((values > 0) & (values < 1))


def test_sampling_surface_has_no_experimental_frailty_selector():
    with pytest.raises(TypeError, match="sampling_method"):
        ClaytonCopula().sample_at_parameter(1, 2., sampling_method="marshall_olkin")
    module, _ = pair._module_and_spec(ClaytonCopula())
    assert not hasattr(module, "copula_sample_marshall_olkin")


@pytest.mark.parametrize("n", [-1, True, 1.5])
def test_invalid_sample_size_does_not_consume_rng(n):
    rng = np.random.default_rng(118)
    before = rng.bit_generator.state
    with pytest.raises(ValueError, match="non-negative integer"):
        ClaytonCopula().sample_at_parameter(n, 2., rng)
    assert rng.bit_generator.state == before
