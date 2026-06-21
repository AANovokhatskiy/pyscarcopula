import numpy as np

from pyscarcopula.copula.base import BivariateCopula


def _frank_bivariate_sample_from_uniforms(n, r, u0_data, v_data):
    """Direct conditional-inversion sampling from fixed uniforms."""
    parameter = np.asarray(r, dtype=np.float64)
    if parameter.size == 1:
        parameter = np.full(n, parameter[0])
    t = np.exp(-parameter * u0_data)
    p = np.exp(-parameter)
    f1 = v_data * (1.0 - p)
    f2 = t + v_data * (1.0 - t)
    sampled = -np.log1p(-f1 / f2) / parameter
    sampled = np.where(np.abs(f1 - f2) < 1e-9, u0_data, sampled)
    return np.column_stack((u0_data, sampled))


class FrankCopula(BivariateCopula):
    """Frank copula. Rotation is unsupported because it is symmetric."""

    def __init__(self, rotate: int = 0, transform_type: str = "softplus"):
        if rotate != 0:
            raise ValueError("Rotation not supported for Frank copula")
        super().__init__(0)
        self._name = "Frank copula"
        self._bounds = [(0.0001, np.inf)]
        if transform_type not in ("xtanh", "softplus"):
            raise ValueError(
                "transform_type must be 'xtanh' or 'softplus', "
                f"got '{transform_type}'"
            )
        self._transform_type = transform_type

    def tau_to_param(self, tau):
        tau = np.atleast_1d(np.asarray(tau, dtype=np.float64))
        if np.any((tau <= 0.0) | (tau >= 1.0)):
            raise ValueError("Frank Kendall tau must be in (0, 1)")
        return self._native_adapter().tau_to_param(self, tau)

    def param_to_tau(self, r):
        r = np.atleast_1d(np.asarray(r, dtype=np.float64))
        if np.any(r <= 0.0):
            raise ValueError("Frank parameter must be positive")
        return self._native_adapter().param_to_tau(self, r)

    def sample_at_parameter(self, n, r, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        parameter = np.atleast_1d(np.asarray(r, dtype=np.float64))
        if parameter.size == 1:
            parameter = np.full(n, parameter[0])
        u0 = rng.uniform(0.0, 1.0, size=n)
        v = rng.uniform(0.0, 1.0, size=n)
        return _frank_bivariate_sample_from_uniforms(
            n, parameter, u0, v)
