import numpy as np

from pyscarcopula.copula.base import BivariateCopula


class JoeCopula(BivariateCopula):

    _native_pair_family = "Joe"

    def __init__(self, rotate: int = 0, transform_type: str = "softplus"):
        super().__init__(rotate)
        self._name = "Joe copula"
        if transform_type not in ("xtanh", "softplus", "exp", "logistic"):
            raise ValueError(
                "transform_type must be one of "
                "'xtanh', 'softplus', 'exp', or 'logistic', "
                f"got '{transform_type}'"
            )
        self._transform_type = transform_type
        upper = 21.0001 if transform_type == "logistic" else np.inf
        self._bounds = [(1.0001, upper)]

    def tau_to_param(self, tau):
        tau = np.atleast_1d(np.asarray(tau, dtype=np.float64))
        if np.any((tau <= 0.0) | (tau >= 1.0)):
            raise ValueError("Joe Kendall tau must be in (0, 1)")
        return self._native_adapter().tau_to_param(self, tau)

    def param_to_tau(self, r):
        r = np.atleast_1d(np.asarray(r, dtype=np.float64))
        if np.any(r < 1.0):
            raise ValueError("Joe parameter must be >= 1")
        return self._native_adapter().param_to_tau(self, r)
