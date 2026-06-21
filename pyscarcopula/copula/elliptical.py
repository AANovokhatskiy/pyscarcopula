"""Bivariate Gaussian copula."""

import numpy as np
from scipy.stats import norm

from pyscarcopula.copula.base import BivariateCopula


class BivariateGaussianCopula(BivariateCopula):
    """Bivariate Gaussian copula with native numerical operations.

    Parameters
    ----------
    rotate : int, default 0
        Gaussian rotation. Only the unrotated value ``0`` is supported.
    transform_type : {'softplus', 'xtanh'}, default 'softplus'
        Compatibility-only constructor argument used by shared copula and
        vine configuration flows. It does not select the Gaussian parameter
        transform: Gaussian models always use the bounded ``GaussianTanh``
        mapping. The supplied value is retained as configuration metadata but
        must not be interpreted as applying softplus or xtanh mathematics.
    """

    def __init__(self, rotate: int = 0, transform_type: str = "softplus"):
        if rotate != 0:
            raise ValueError("Rotation not supported for Gaussian copula")
        super().__init__(0)
        self._name = "Gaussian copula"
        self._bounds = [(-0.9999, 0.9999)]
        if transform_type not in ("xtanh", "softplus"):
            raise ValueError(
                "transform_type must be 'xtanh' or 'softplus', "
                f"got '{transform_type}'"
            )
        self._transform_type = transform_type

    @property
    def rotatable(self):
        return False

    def tau_to_param(self, tau):
        tau = np.atleast_1d(np.asarray(tau, dtype=np.float64))
        if np.any((tau <= -1.0) | (tau >= 1.0)):
            raise ValueError("Gaussian Kendall tau must be in (-1, 1)")
        return self._native_adapter().tau_to_param(self, tau)

    def param_to_tau(self, r):
        r = np.atleast_1d(np.asarray(r, dtype=np.float64))
        if np.any((r <= -1.0) | (r >= 1.0)):
            raise ValueError(
                "Gaussian correlation parameter must be in (-1, 1)")
        return self._native_adapter().param_to_tau(self, r)

    def sample_at_parameter(self, n, r, rng=None):
        """Sample from the Gaussian copula."""
        parameter = np.atleast_1d(np.asarray(r, dtype=np.float64))
        rho = parameter[0] if parameter.size == 1 else parameter
        if rng is None:
            rng = np.random.default_rng()

        normal = rng.standard_normal((n, 2))
        if np.isscalar(rho):
            rho_value = float(rho)
            second = (
                rho_value * normal[:, 0]
                + np.sqrt(1.0 - rho_value ** 2) * normal[:, 1]
            )
        else:
            rho_values = np.asarray(rho).ravel()
            if rho_values.size != n:
                raise ValueError(
                    f"r must be scalar or array of length {n}, "
                    f"got {rho_values.size}"
                )
            second = (
                rho_values * normal[:, 0]
                + np.sqrt(1.0 - rho_values ** 2) * normal[:, 1]
            )
        return np.column_stack((norm.cdf(normal[:, 0]), norm.cdf(second)))
