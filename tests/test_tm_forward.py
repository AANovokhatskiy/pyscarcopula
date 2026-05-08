import numpy as np

from pyscarcopula.numerical.tm_grid import TMGrid


def _direct_forward_density(grid, phi):
    """Direct quadrature for phi_next[i] = sum_j p(i|j) phi[j] w[j]."""
    z = grid.z
    means = grid.rho * z
    diff = z[np.newaxis, :] - means[:, np.newaxis]
    p = (
        np.exp(-0.5 * (diff / grid.sigma_cond) ** 2)
        / (grid.sigma_cond * np.sqrt(2.0 * np.pi))
    )
    return p.T @ (phi * grid.trap_w)


def _density_to_mass(grid, phi):
    mass = phi * grid.trap_w
    return mass / np.sum(mass)


def test_predict_matvec_matches_direct_forward_quadrature():
    grid = TMGrid(
        kappa=3.0,
        mu=0.4,
        nu=1.2,
        n=25,
        K=15,
        grid_range=2.0,
        grid_method="dense",
        adaptive=False,
    )
    phi = grid.p0 * (1.0 + 0.25 * grid.z / grid.sigma)

    direct = _direct_forward_density(grid, phi)
    via_grid = grid.predict_matvec(phi * grid.trap_w)

    np.testing.assert_allclose(via_grid, direct, rtol=1e-12, atol=1e-12)


def test_forward_weights_are_predictive_before_current_observation():
    grid = TMGrid(
        kappa=2.0,
        mu=0.0,
        nu=1.0,
        n=5,
        K=11,
        grid_range=1.8,
        grid_method="dense",
        adaptive=False,
    )
    fi = np.ones((5, grid.K))

    weights = grid.forward_weights(fi)
    expected_0 = _density_to_mass(grid, grid.p0)
    expected_1 = _density_to_mass(grid, _direct_forward_density(grid, grid.p0))

    np.testing.assert_allclose(weights[0], expected_0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(weights[1], expected_1, rtol=1e-12, atol=1e-12)
