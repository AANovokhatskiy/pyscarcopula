import numpy as np

from pyscarcopula.copula.base import BivariateCopula


class ClaytonCopula(BivariateCopula):

    def __init__(self, rotate: int = 0, transform_type: str = "softplus"):
        super().__init__(rotate)
        self._name = "Clayton copula"
        if transform_type not in ("xtanh", "softplus"):
            raise ValueError(
                "transform_type must be 'xtanh' or 'softplus', "
                f"got '{transform_type}'"
            )
        self._transform_type = transform_type
        self._bounds = [(0.0001, np.inf)]

    def tau_to_param(self, tau):
        tau = np.atleast_1d(np.asarray(tau, dtype=np.float64))
        if np.any((tau <= 0.0) | (tau >= 1.0)):
            raise ValueError("Clayton Kendall tau must be in (0, 1)")
        return self._native_adapter().tau_to_param(self, tau)

    def param_to_tau(self, r):
        r = np.atleast_1d(np.asarray(r, dtype=np.float64))
        if np.any(r <= 0.0):
            raise ValueError("Clayton parameter must be positive")
        return self._native_adapter().param_to_tau(self, r)

    @staticmethod
    def psi(t, r):
        return (1.0 + t * r) ** (-1.0 / r)

    def V(self, n, r, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        parameter = np.atleast_1d(np.asarray(r, dtype=np.float64))
        if parameter.size == 1:
            parameter = np.full(n, parameter[0])
        return rng.gamma(1.0 / parameter, scale=parameter)
