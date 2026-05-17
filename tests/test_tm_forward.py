import numpy as np

from pyscarcopula import GumbelCopula
from pyscarcopula.api import fit
from pyscarcopula.numerical.tm_functions import _forward_loglik, tm_loglik
from pyscarcopula.numerical.tm_grid import TMGrid
from pyscarcopula.strategy import scar_tm


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


def test_forward_loglik_matches_backward_pass_on_notebook_dataset(crypto_data):
    """Use the BTC/ETH dataset from examples/02_bivariate.ipynb."""
    copula = GumbelCopula(rotate=180)

    params = {
        "kappa": 59.02,
        "mu": 2.17,
        "nu": 15.80,
        "K": 60,
        "grid_range": 5.0,
        "grid_method": "auto",
        "adaptive": True,
        "pts_per_sigma": 4,
    }

    backward_loglik = -tm_loglik(**params, u=crypto_data, copula=copula)
    forward_loglik = _forward_loglik(**params, u=crypto_data, copula=copula)

    np.testing.assert_allclose(
        forward_loglik,
        backward_loglik,
        rtol=1e-10,
        atol=1e-8,
    )


def test_fit_path_matches_with_forward_and_backward_objectives(
        crypto_data, monkeypatch):
    """Run two independent SCAR-TM fits on examples/02_bivariate.ipynb data."""
    copula = GumbelCopula(rotate=180)
    original_tm_loglik = scar_tm.tm_loglik

    fit_kwargs = {
        "alpha0": np.array([59.02, 2.0, 15.0]),
        "smart_init": False,
        "analytical_grad": False,
        "K": 60,
        "grid_range": 5.0,
        "grid_method": "dense",
        "adaptive": False,
        "pts_per_sigma": 4,
        "maxiter": 5,
        "maxfun": 40,
        "gtol": 1e-3,
        "eps": 1e-4,
    }

    backward_calls = []
    forward_calls = []

    def backward_objective(kappa, mu, nu, u, cop, K=300, grid_range=5.0,
                           grid_method="auto", adaptive=True,
                           pts_per_sigma=4):
        alpha = np.array([kappa, mu, nu], dtype=np.float64)
        value = original_tm_loglik(
            kappa, mu, nu, u, cop, K, grid_range,
            grid_method, adaptive, pts_per_sigma)
        backward_calls.append((alpha, value))
        return value

    def forward_objective(kappa, mu, nu, u, cop, K=300, grid_range=5.0,
                          grid_method="auto", adaptive=True,
                          pts_per_sigma=4):
        alpha = np.array([kappa, mu, nu], dtype=np.float64)
        value = -_forward_loglik(
            kappa, mu, nu, u, cop, K, grid_range,
            grid_method, adaptive, pts_per_sigma)
        forward_calls.append((alpha, value))
        return value

    monkeypatch.setattr(scar_tm, "tm_loglik", backward_objective)
    backward_fit = fit(copula, crypto_data, method="scar-tm-ou", **fit_kwargs)

    monkeypatch.setattr(scar_tm, "tm_loglik", forward_objective)
    forward_fit = fit(copula, crypto_data, method="scar-tm-ou", **fit_kwargs)

    assert backward_fit.success
    assert forward_fit.success
    assert len(backward_calls) == len(forward_calls)

    backward_path = np.array([alpha for alpha, _ in backward_calls])
    forward_path = np.array([alpha for alpha, _ in forward_calls])
    backward_values = np.array([value for _, value in backward_calls])
    forward_values = np.array([value for _, value in forward_calls])

    np.testing.assert_allclose(
        forward_path,
        backward_path,
        rtol=1e-7,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        forward_values,
        backward_values,
        rtol=1e-9,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        forward_fit.params.values,
        backward_fit.params.values,
        rtol=1e-7,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        forward_fit.log_likelihood,
        backward_fit.log_likelihood,
        rtol=1e-9,
        atol=1e-6,
    )
