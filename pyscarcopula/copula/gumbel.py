import numpy as np
from pyscarcopula._native import model_policy
from pyscarcopula.numerical._arrays import as_float64_array

from pyscarcopula.copula.base import BivariateCopula


class GumbelCopula(BivariateCopula):

    _native_pair_family = "Gumbel"

    def __init__(self, rotate: int = 0, transform_type: str = "softplus"):
        super().__init__(rotate)
        self._name = "Gumbel copula"
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
            raise ValueError("Gumbel Kendall tau must be in (0, 1)")
        return self._native_adapter().tau_to_param(self, tau)

    def param_to_tau(self, r):
        r = np.atleast_1d(as_float64_array(r, name="r"))
        if np.any(r < 1.0):
            raise ValueError("Gumbel parameter must be >= 1")
        return self._native_adapter().param_to_tau(self, r)
