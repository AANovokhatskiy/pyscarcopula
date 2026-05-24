import numpy as np

from pyscarcopula import ClaytonCopula, GumbelCopula, JoeCopula


def _extreme_uv():
    return (
        np.array([1e-12, 1e-8, 1e-4, 0.02, 0.5, 0.98, 1 - 1e-8]),
        np.array([1 - 1e-8, 0.98, 0.5, 0.02, 1e-4, 1e-8, 1e-12]),
    )


def _assert_unit_interval(values):
    values = np.asarray(values, dtype=np.float64)
    assert np.all(np.isfinite(values))
    assert np.all((values > 0.0) & (values < 1.0))


def test_gumbel_h_is_finite_for_large_theta_and_extreme_u():
    copula = GumbelCopula()
    u, v = _extreme_uv()
    theta = np.full(len(u), 500.0)

    _assert_unit_interval(copula.h(u, v, theta))
    h01, h10 = copula.h_pair(u, v, theta)
    _assert_unit_interval(h01)
    _assert_unit_interval(h10)


def test_clayton_h_is_finite_for_large_theta_and_extreme_u():
    copula = ClaytonCopula()
    u, v = _extreme_uv()
    theta = np.full(len(u), 500.0)

    _assert_unit_interval(copula.h(u, v, theta))
    h01, h10 = copula.h_pair(u, v, theta)
    _assert_unit_interval(h01)
    _assert_unit_interval(h10)


def test_joe_h_is_finite_for_large_theta_and_extreme_u():
    copula = JoeCopula()
    u, v = _extreme_uv()
    theta = np.full(len(u), 500.0)

    _assert_unit_interval(copula.h(u, v, theta))
    h01, h10 = copula.h_pair(u, v, theta)
    _assert_unit_interval(h01)
    _assert_unit_interval(h10)
