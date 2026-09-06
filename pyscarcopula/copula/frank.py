import numpy as np
from pyscarcopula._native import model_policy
from pyscarcopula.numerical._arrays import as_float64_array

from pyscarcopula.copula.base import BivariateCopula


class FrankCopula(BivariateCopula):
    """Frank copula. Rotation is unsupported because it is symmetric."""

    _native_pair_family = "Frank"

    def __init__(self, rotate: int = 0, transform_type: str = "softplus"):
        if rotate != 0:
            raise ValueError("Rotation not supported for Frank copula")
        super().__init__(0)
        self._name = "Frank copula"
        if transform_type not in ("xtanh", "softplus", "exp", "logistic"):
            raise ValueError(
                "transform_type must be one of "
                "'xtanh', 'softplus', 'exp', or 'logistic', "
                f"got '{transform_type}'"
            )
        self._transform_type = transform_type
        self._bounds = model_policy.public_bounds(self)

    def tau_to_param(self, tau):
        tau = np.atleast_1d(as_float64_array(tau, name="tau"))
        if np.any((tau <= 0.0) | (tau >= 1.0)):
            raise ValueError("Frank Kendall tau must be in (0, 1)")
        return self._native_adapter().tau_to_param(self, tau)

    def param_to_tau(self, r):
        r = np.atleast_1d(as_float64_array(r, name="r"))
        if np.any(r <= 0.0):
            raise ValueError("Frank parameter must be positive")
        return self._native_adapter().param_to_tau(self, r)
