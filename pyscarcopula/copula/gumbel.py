import numpy as np

from pyscarcopula.copula.base import BivariateCopula


def _generate_levy_stable_from_uniforms(
        alpha, beta, loc, scale, angle, uniform):
    """Generate stable frailty draws used by Gumbel sampling."""
    exponential = -np.log1p(-uniform)
    non_unit = alpha != 1
    unit = np.invert(non_unit)
    shift = np.arctan(
        beta * np.tan(np.pi / 2 * alpha)) / alpha
    factor = (
        1 + beta ** 2 * np.tan(np.pi / 2 * alpha) ** 2
    ) ** (1 / (2 * alpha))
    x0 = (
        factor
        * np.sin(alpha * (angle + shift))
        / np.cos(angle) ** (1 / alpha)
        * (
            np.cos(angle - alpha * (angle + shift)) / exponential
        ) ** ((1 - alpha) / alpha)
    )
    x1 = 2 / np.pi * (
        (np.pi / 2 + beta * angle) * np.tan(angle)
        - beta * np.log(
            np.pi / 2 * exponential * np.cos(angle)
            / (np.pi / 2 + beta * angle)
        )
    )
    value = x0 * non_unit + x1 * unit
    y0 = scale * value + loc
    y1 = scale * value + 2 / np.pi * beta * scale * np.log(scale) + loc
    return y0 * non_unit + y1 * unit


class GumbelCopula(BivariateCopula):

    def __init__(self, rotate: int = 0, transform_type: str = "softplus"):
        super().__init__(rotate)
        self._name = "Gumbel copula"
        self._bounds = [(1.0001, np.inf)]
        if transform_type not in ("xtanh", "softplus"):
            raise ValueError(
                "transform_type must be 'xtanh' or 'softplus', "
                f"got '{transform_type}'"
            )
        self._transform_type = transform_type

    def tau_to_param(self, tau):
        tau = np.atleast_1d(np.asarray(tau, dtype=np.float64))
        if np.any((tau <= 0.0) | (tau >= 1.0)):
            raise ValueError("Gumbel Kendall tau must be in (0, 1)")
        return self._native_adapter().tau_to_param(self, tau)

    def param_to_tau(self, r):
        r = np.atleast_1d(np.asarray(r, dtype=np.float64))
        if np.any(r < 1.0):
            raise ValueError("Gumbel parameter must be >= 1")
        return self._native_adapter().param_to_tau(self, r)

    @staticmethod
    def psi(t, r):
        return np.exp(-t ** (1.0 / r))

    def V(self, n, r, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        parameter = np.atleast_1d(np.asarray(r, dtype=np.float64))
        if parameter.size == 1:
            parameter = np.full(n, parameter[0])
        angle = rng.uniform(-np.pi / 2.0, np.pi / 2.0, size=n)
        uniform = rng.uniform(0.0, 1.0, size=n)
        alpha = 1.0 / parameter
        scale = np.cos(np.pi / (2.0 * parameter)) ** parameter
        result = _generate_levy_stable_from_uniforms(
            alpha, 1.0, 0.0, scale, angle, uniform)
        return np.maximum(result, 1e-300)
