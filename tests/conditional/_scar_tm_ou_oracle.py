"""Independent continuous-quadrature oracle for Gaussian SCAR-TM-OU.

This module must not import pyscarcopula.  It reconstructs the scalar OU
filter from closed-form transition and Gaussian-copula density formulas on
Gauss-Legendre nodes, a representation distinct from the production uniform
transfer-matrix grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp, roots_legendre
from scipy.stats import norm

from ._analytical_oracles import (
    gaussian_conditional_cdf,
    gaussian_conditional_inverse,
    gaussian_copula_log_density,
    gaussian_copula_parameter_from_state,
)


@dataclass(frozen=True)
class QuadratureDistribution:
    nodes: np.ndarray
    probability: np.ndarray

    def __post_init__(self):
        nodes = np.asarray(self.nodes, dtype=np.float64)
        probability = np.asarray(self.probability, dtype=np.float64)
        if nodes.ndim != 1 or probability.shape != nodes.shape:
            raise ValueError("nodes and probability must be equal-length vectors")
        if np.any(np.diff(nodes) <= 0.0):
            raise ValueError("nodes must be strictly increasing")
        if np.any(~np.isfinite(probability)) or np.any(probability < 0.0):
            raise ValueError("probability must be finite and non-negative")
        total = float(np.sum(probability))
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("probability must have positive finite mass")
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "probability", probability / total)

    @property
    def mean(self) -> float:
        return float(self.probability @ self.nodes)

    @property
    def variance(self) -> float:
        centered = self.nodes - self.mean
        return float(self.probability @ (centered * centered))

    def expectation(self, values) -> float:
        evaluated = np.asarray(values, dtype=np.float64)
        if evaluated.shape != self.nodes.shape:
            raise ValueError("expectation values must match nodes")
        return float(self.probability @ evaluated)

    def quantile(self, probabilities) -> np.ndarray:
        q = np.asarray(probabilities, dtype=np.float64)
        if np.any((q < 0.0) | (q > 1.0)):
            raise ValueError("quantile probabilities must be in [0, 1]")
        cumulative = np.cumsum(self.probability)
        return np.interp(q, cumulative, self.nodes)


@dataclass(frozen=True)
class ScarOuFilterResult:
    current: QuadratureDistribution
    predictive: QuadratureDistribution
    dt: float
    log_evidence: float
    stationary_tail_mass_bound: float
    maximum_transition_mass_deficit: float


class GaussianScarOuReference:
    """Full-history Gaussian SCAR-OU filter on Legendre quadrature nodes."""

    def __init__(
            self,
            kappa: float,
            mu: float,
            nu: float,
            n_observations: int,
            *,
            n_nodes: int = 501,
            range_sigma: float = 7.0):
        self.kappa = float(kappa)
        self.mu = float(mu)
        self.nu = float(nu)
        self.n_observations = int(n_observations)
        self.n_nodes = int(n_nodes)
        self.range_sigma = float(range_sigma)
        if not np.isfinite(self.kappa) or self.kappa <= 0.0:
            raise ValueError("kappa must be positive and finite")
        if not np.isfinite(self.mu):
            raise ValueError("mu must be finite")
        if not np.isfinite(self.nu) or self.nu <= 0.0:
            raise ValueError("nu must be positive and finite")
        if self.n_observations < 2:
            raise ValueError("n_observations must be at least 2")
        if self.n_nodes < 31:
            raise ValueError("n_nodes must be at least 31")
        if not np.isfinite(self.range_sigma) or self.range_sigma <= 3.0:
            raise ValueError("range_sigma must be finite and greater than 3")

        self.dt = 1.0 / (self.n_observations - 1)
        self.stationary_sigma = self.nu / np.sqrt(2.0 * self.kappa)
        self.transition_rho = np.exp(-self.kappa * self.dt)
        self.transition_sigma = self.stationary_sigma * np.sqrt(
            1.0 - self.transition_rho ** 2
        )

        raw_nodes, raw_weights = roots_legendre(self.n_nodes)
        half_width = self.range_sigma * self.stationary_sigma
        self.nodes = self.mu + half_width * raw_nodes
        self.quadrature_weights = half_width * raw_weights
        stationary_density = norm.pdf(
            self.nodes, loc=self.mu, scale=self.stationary_sigma
        )
        self._stationary_probability = self._normalize(
            stationary_density * self.quadrature_weights
        )

        transition_mean = (
            self.mu
            + self.transition_rho * (self.nodes - self.mu)
        )
        transition_density = norm.pdf(
            self.nodes[None, :],
            loc=transition_mean[:, None],
            scale=self.transition_sigma,
        )
        self._transition = (
            transition_density * self.quadrature_weights[None, :]
        )
        row_mass = np.sum(self._transition, axis=1)
        relevant = self._stationary_probability > 1e-12
        self.maximum_transition_mass_deficit = float(
            np.max(np.abs(row_mass[relevant] - 1.0))
        )
        self.stationary_tail_mass_bound = float(
            2.0 * norm.sf(self.range_sigma)
        )

    @staticmethod
    def _normalize(probability) -> np.ndarray:
        values = np.asarray(probability, dtype=np.float64)
        total = float(np.sum(values))
        if not np.isfinite(total) or total <= 0.0:
            raise FloatingPointError("quadrature probability lost all mass")
        return values / total

    @property
    def stationary(self) -> QuadratureDistribution:
        return QuadratureDistribution(
            self.nodes, self._stationary_probability.copy()
        )

    def propagate(
            self, distribution: QuadratureDistribution) -> QuadratureDistribution:
        if not np.array_equal(distribution.nodes, self.nodes):
            raise ValueError("distribution nodes do not match this oracle")
        probability = distribution.probability @ self._transition
        return QuadratureDistribution(self.nodes, probability)

    def filter(self, history) -> ScarOuFilterResult:
        observations = np.asarray(history, dtype=np.float64)
        if observations.shape != (self.n_observations, 2):
            raise ValueError(
                f"history must have shape ({self.n_observations}, 2)"
            )
        if np.any(~np.isfinite(observations)) or np.any(
                (observations <= 0.0) | (observations >= 1.0)):
            raise ValueError("history must be finite and in (0, 1)")

        probability = self._stationary_probability.copy()
        state_rho = gaussian_copula_parameter_from_state(self.nodes)
        log_evidence = 0.0
        for time, row in enumerate(observations):
            if time:
                probability = self._normalize(probability @ self._transition)
            log_weights = (
                np.log(np.maximum(probability, np.finfo(np.float64).tiny))
                + gaussian_copula_log_density(row, state_rho)
            )
            normalization = float(logsumexp(log_weights))
            probability = np.exp(log_weights - normalization)
            log_evidence += normalization

        current = QuadratureDistribution(self.nodes, probability)
        predictive = self.propagate(current)
        return ScarOuFilterResult(
            current=current,
            predictive=predictive,
            dt=self.dt,
            log_evidence=log_evidence,
            stationary_tail_mass_bound=self.stationary_tail_mass_bound,
            maximum_transition_mass_deficit=self.maximum_transition_mass_deficit,
        )

    @staticmethod
    def parameter_values(distribution: QuadratureDistribution) -> np.ndarray:
        return gaussian_copula_parameter_from_state(distribution.nodes)

    def mixture_cdf(
            self,
            free_u,
            given_u: float,
            distribution: QuadratureDistribution) -> np.ndarray:
        values = np.asarray(free_u, dtype=np.float64)
        flat = values.ravel()
        state_rho = self.parameter_values(distribution)
        output = np.empty_like(flat)
        block_size = max(1, 1_000_000 // len(state_rho))
        for start in range(0, len(flat), block_size):
            stop = min(len(flat), start + block_size)
            block = flat[start:stop]
            conditional = gaussian_conditional_cdf(
                block[:, None], float(given_u), state_rho[None, :]
            )
            output[start:stop] = conditional @ distribution.probability
        output[flat <= 0.0] = 0.0
        output[flat >= 1.0] = 1.0
        return output.reshape(values.shape)

    def mixture_cdf_grid(
            self,
            given_u: float,
            distribution: QuadratureDistribution,
            *,
            grid_size: int = 4097) -> tuple[np.ndarray, np.ndarray]:
        grid = np.linspace(0.0, 1.0, int(grid_size), dtype=np.float64)
        cdf = self.mixture_cdf(grid, given_u, distribution)
        cdf = np.maximum.accumulate(np.clip(cdf, 0.0, 1.0))
        cdf[0] = 0.0
        cdf[-1] = 1.0
        return grid, cdf

    def mixture_pit(
            self,
            free_u,
            given_u: float,
            distribution: QuadratureDistribution,
            *,
            grid_size: int = 8193) -> np.ndarray:
        grid, cdf = self.mixture_cdf_grid(
            given_u, distribution, grid_size=grid_size
        )
        return np.interp(np.asarray(free_u, dtype=np.float64), grid, cdf)

    def sample_mixture(
            self,
            n: int,
            given_u: float,
            distribution: QuadratureDistribution,
            rng: np.random.Generator) -> np.ndarray:
        indices = rng.choice(
            len(distribution.nodes), size=int(n), p=distribution.probability
        )
        state_rho = gaussian_copula_parameter_from_state(
            distribution.nodes[indices]
        )
        q = rng.random(int(n))
        return gaussian_conditional_inverse(q, given_u, state_rho)


def simulate_gaussian_scar_history(
        n_observations: int,
        kappa: float,
        mu: float,
        nu: float,
        *,
        seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Independent exact-OU and Gaussian-copula history generator."""

    n_observations = int(n_observations)
    dt = 1.0 / (n_observations - 1)
    stationary_sigma = nu / np.sqrt(2.0 * kappa)
    transition_rho = np.exp(-kappa * dt)
    transition_sigma = stationary_sigma * np.sqrt(1.0 - transition_rho ** 2)
    rng = np.random.default_rng(seed)
    states = np.empty(n_observations, dtype=np.float64)
    states[0] = rng.normal(mu, stationary_sigma)
    for time in range(1, n_observations):
        states[time] = rng.normal(
            mu + transition_rho * (states[time - 1] - mu),
            transition_sigma,
        )
    rho = gaussian_copula_parameter_from_state(states)
    normal = rng.standard_normal((n_observations, 2))
    normal[:, 1] = (
        rho * normal[:, 0]
        + np.sqrt(1.0 - rho * rho) * normal[:, 1]
    )
    return norm.cdf(normal), states
