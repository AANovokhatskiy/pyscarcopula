import numpy as np

from pyscarcopula.copula.base import BivariateCopula


def _joe_v_from_uniforms(n, r, uniforms):
    """Sample the Sibuya frailty variable used by Joe sampling."""
    out = np.empty(n)
    for index in range(n):
        target = uniforms[index]
        value = 1
        initial_probability = 1.0 / r[index]
        probability = initial_probability
        cumulative = probability
        while target > cumulative:
            probability *= (
                -(initial_probability - float(value + 1) + 1.0)
                / float(value + 1)
            )
            cumulative += probability
            value += 1
        out[index] = float(value)
    return out


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

    @staticmethod
    def psi(t, r):
        return 1.0 - (1.0 - np.exp(-t)) ** (1.0 / r)

    def V(self, n, r, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        parameter = np.atleast_1d(np.asarray(r, dtype=np.float64))
        if parameter.size == 1:
            parameter = np.full(n, parameter[0])
        uniforms = rng.uniform(0.0, 1.0, size=n)
        return _joe_v_from_uniforms(n, parameter, uniforms)

    def sample_at_parameter(self, n, r, rng=None):
        """Sample through native conditional inversion."""
        if rng is None:
            rng = np.random.default_rng()
        parameter = np.atleast_1d(np.asarray(r, dtype=np.float64))
        if parameter.size == 1:
            parameter = np.full(n, parameter[0])
        u1 = rng.uniform(0.0, 1.0, size=n)
        e2 = rng.uniform(0.0, 1.0, size=n)
        u2 = self.h_inverse(e2, u1, parameter)
        return np.column_stack((u1, u2))
