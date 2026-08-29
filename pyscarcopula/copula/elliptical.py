"""Bivariate Gaussian copula."""

import numpy as np
from pyscarcopula._native import model_policy

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

    _native_pair_family = "Gaussian"

    def __init__(self, rotate: int = 0, transform_type: str = "softplus"):
        if rotate != 0:
            raise ValueError("Rotation not supported for Gaussian copula")
        super().__init__(0)
        self._name = "Gaussian copula"
        if transform_type not in ("xtanh", "softplus"):
            raise ValueError(
                "transform_type must be 'xtanh' or 'softplus', "
                f"got '{transform_type}'"
            )
        self._transform_type = transform_type
        self._bounds = model_policy.public_bounds(self)

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
