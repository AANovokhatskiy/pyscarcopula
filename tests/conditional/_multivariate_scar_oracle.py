"""Independent multivariate SCAR-TM-OU conditional-mixture oracle.

The reference uses closed-form copula densities and Gauss-Legendre
quadrature.  It intentionally does not import :mod:`pyscarcopula` and does
not share the production transfer-matrix representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
from scipy.special import gammaln, logsumexp, roots_legendre
from scipy.stats import norm
from scipy.stats import t as t_dist

from ._scar_tm_ou_oracle import QuadratureDistribution, ScarOuFilterResult


def equicorrelation_matrix(dimension: int, rho: float) -> np.ndarray:
    """Return the correlation matrix with one common off-diagonal value."""

    matrix = np.full((int(dimension), int(dimension)), float(rho))
    np.fill_diagonal(matrix, 1.0)
    np.linalg.cholesky(matrix)
    return matrix


def equicorr_parameter_from_state(state, dimension: int) -> np.ndarray:
    """Independent form of the dimension-dependent equicorrelation link."""

    values = np.asarray(state, dtype=np.float64)
    lower = -1.0 / (int(dimension) - 1.0)
    return lower + 0.5 * (1.0 - lower) * (1.0 + np.tanh(values))


def student_df_parameter_from_state(state) -> np.ndarray:
    """Independent finite-variance Student link ``2 + 1e-6 + softplus``."""

    values = np.asarray(state, dtype=np.float64)
    return 2.0 + 1e-6 + np.logaddexp(0.0, values)


def equicorr_gaussian_log_density(row, rho) -> np.ndarray:
    """Equicorrelation Gaussian copula log-density from eigenvalues."""

    observation = np.asarray(row, dtype=np.float64)
    parameter = np.asarray(rho, dtype=np.float64)
    dimension = observation.size
    lower = -1.0 / (dimension - 1.0)
    if observation.shape != (dimension,) or dimension < 2:
        raise ValueError("row must be a vector of dimension at least two")
    if np.any((observation <= 0.0) | (observation >= 1.0)):
        raise ValueError("row must be in (0, 1)")
    if np.any((parameter <= lower) | (parameter >= 1.0)):
        raise ValueError("rho is outside the equicorrelation interval")
    latent = norm.ppf(observation)
    one_minus = 1.0 - parameter
    common = 1.0 + (dimension - 1.0) * parameter
    log_det = (dimension - 1.0) * np.log(one_minus) + np.log(common)
    diagonal = parameter / one_minus
    shared = -parameter / (one_minus * common)
    quadratic = (
        diagonal * float(latent @ latent)
        + shared * float(np.sum(latent) ** 2)
    )
    return -0.5 * (log_det + quadratic)


def student_copula_log_density(row, df, correlation) -> np.ndarray:
    """Multivariate Student copula log-density from density factorization."""

    observation = np.asarray(row, dtype=np.float64)
    parameter = np.atleast_1d(np.asarray(df, dtype=np.float64))
    matrix = np.asarray(correlation, dtype=np.float64)
    dimension = observation.size
    if matrix.shape != (dimension, dimension):
        raise ValueError("correlation shape does not match row")
    if np.any((observation <= 0.0) | (observation >= 1.0)):
        raise ValueError("row must be in (0, 1)")
    if np.any(~np.isfinite(parameter)) or np.any(parameter <= 2.0):
        raise ValueError("df must be finite and greater than two")
    sign, log_det = np.linalg.slogdet(matrix)
    if sign <= 0.0:
        raise ValueError("correlation must be positive definite")
    inverse = np.linalg.inv(matrix)
    latent = t_dist.ppf(observation[None, :], df=parameter[:, None])
    quadratic = np.einsum("ki,ij,kj->k", latent, inverse, latent)
    multivariate = (
        gammaln(0.5 * (parameter + dimension))
        - gammaln(0.5 * parameter)
        - 0.5 * (dimension * np.log(parameter * np.pi) + log_det)
        - 0.5 * (parameter + dimension) * np.log1p(quadratic / parameter)
    )
    marginal = (
        gammaln(0.5 * (parameter[:, None] + 1.0))
        - gammaln(0.5 * parameter[:, None])
        - 0.5 * np.log(parameter[:, None] * np.pi)
        - 0.5 * (parameter[:, None] + 1.0)
        * np.log1p(latent * latent / parameter[:, None])
    )
    return multivariate - np.sum(marginal, axis=1)


def _one_free_partition(
        dimension: int, given: Mapping[int, float]) -> tuple[int, np.ndarray, np.ndarray]:
    given_indices = np.array(sorted(given), dtype=np.int64)
    free_indices = [index for index in range(dimension) if index not in given]
    if len(free_indices) != 1:
        raise ValueError("mixture CDF oracle requires exactly one free coordinate")
    values = np.array([given[int(index)] for index in given_indices])
    return free_indices[0], given_indices, values


def equicorr_conditional_cdf(
        free_u, rho, dimension: int, given: Mapping[int, float]) -> np.ndarray:
    """Conditional CDF for the single free coordinate at each ``rho``."""

    values = np.atleast_1d(np.asarray(free_u, dtype=np.float64))
    parameters = np.atleast_1d(np.asarray(rho, dtype=np.float64))
    _, given_indices, given_u = _one_free_partition(dimension, given)
    free_index = next(index for index in range(dimension) if index not in given)
    z_given = norm.ppf(given_u)
    output = np.empty((len(values), len(parameters)), dtype=np.float64)
    z_free = norm.ppf(np.clip(values, 1e-12, 1.0 - 1e-12))
    for column, parameter in enumerate(parameters):
        matrix = equicorrelation_matrix(dimension, parameter)
        r_gg = matrix[np.ix_(given_indices, given_indices)]
        r_fg = matrix[free_index, given_indices]
        solved_cross = np.linalg.solve(r_gg, r_fg)
        mean = float(r_fg @ np.linalg.solve(r_gg, z_given))
        variance = float(1.0 - r_fg @ solved_cross)
        output[:, column] = norm.cdf((z_free - mean) / np.sqrt(variance))
    output[values <= 0.0] = 0.0
    output[values >= 1.0] = 1.0
    return output


def student_conditional_cdf(
        free_u, df, correlation, given: Mapping[int, float]) -> np.ndarray:
    """Conditional CDF for one free Student-copula coordinate per ``df``."""

    values = np.atleast_1d(np.asarray(free_u, dtype=np.float64))
    parameters = np.atleast_1d(np.asarray(df, dtype=np.float64))
    matrix = np.asarray(correlation, dtype=np.float64)
    dimension = len(matrix)
    free_index, given_indices, given_u = _one_free_partition(dimension, given)
    r_gg = matrix[np.ix_(given_indices, given_indices)]
    r_fg = matrix[free_index, given_indices]
    schur = float(1.0 - r_fg @ np.linalg.solve(r_gg, r_fg))
    output = np.empty((len(values), len(parameters)), dtype=np.float64)
    for column, parameter in enumerate(parameters):
        x_given = t_dist.ppf(given_u, df=parameter)
        solved = np.linalg.solve(r_gg, x_given)
        location = float(r_fg @ solved)
        delta = float(x_given @ solved)
        conditional_df = parameter + len(given_indices)
        scale = np.sqrt((parameter + delta) / conditional_df * schur)
        x_free = t_dist.ppf(
            np.clip(values, 1e-12, 1.0 - 1e-12), df=parameter
        )
        output[:, column] = t_dist.cdf(
            (x_free - location) / scale, df=conditional_df
        )
    output[values <= 0.0] = 0.0
    output[values >= 1.0] = 1.0
    return output


@dataclass(frozen=True)
class ScalarScarOuReference:
    """Continuous-quadrature scalar OU filter with pluggable emissions."""

    kappa: float
    mu: float
    nu: float
    n_observations: int
    parameter_from_state: Callable[[np.ndarray], np.ndarray]
    emission_log_density: Callable[[np.ndarray, np.ndarray], np.ndarray]
    n_nodes: int = 401
    range_sigma: float = 7.0

    def __post_init__(self):
        if self.kappa <= 0.0 or self.nu <= 0.0:
            raise ValueError("kappa and nu must be positive")
        if self.n_observations < 2 or self.n_nodes < 31:
            raise ValueError("insufficient observations or quadrature nodes")
        dt = 1.0 / (self.n_observations - 1)
        stationary_sigma = self.nu / np.sqrt(2.0 * self.kappa)
        transition_rho = np.exp(-self.kappa * dt)
        transition_sigma = stationary_sigma * np.sqrt(1.0 - transition_rho ** 2)
        raw_nodes, raw_weights = roots_legendre(self.n_nodes)
        half_width = self.range_sigma * stationary_sigma
        nodes = self.mu + half_width * raw_nodes
        weights = half_width * raw_weights
        stationary = norm.pdf(nodes, self.mu, stationary_sigma) * weights
        stationary /= np.sum(stationary)
        means = self.mu + transition_rho * (nodes - self.mu)
        transition = norm.pdf(
            nodes[None, :], means[:, None], transition_sigma
        ) * weights[None, :]
        object.__setattr__(self, "dt", dt)
        object.__setattr__(self, "stationary_sigma", stationary_sigma)
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "parameters", self.parameter_from_state(nodes))
        object.__setattr__(self, "_stationary_probability", stationary)
        object.__setattr__(self, "_transition", transition)
        object.__setattr__(
            self, "stationary_tail_mass_bound", float(2.0 * norm.sf(self.range_sigma))
        )

    @staticmethod
    def _normalize(values) -> np.ndarray:
        probability = np.asarray(values, dtype=np.float64)
        total = float(np.sum(probability))
        if not np.isfinite(total) or total <= 0.0:
            raise FloatingPointError("quadrature probability lost all mass")
        return probability / total

    @property
    def stationary(self) -> QuadratureDistribution:
        return QuadratureDistribution(
            self.nodes, self._stationary_probability.copy()
        )

    def propagate(self, distribution: QuadratureDistribution) -> QuadratureDistribution:
        return QuadratureDistribution(
            self.nodes, distribution.probability @ self._transition
        )

    def filter(self, history) -> ScarOuFilterResult:
        observations = np.asarray(history, dtype=np.float64)
        if observations.ndim != 2 or len(observations) != self.n_observations:
            raise ValueError("history has the wrong shape")
        probability = self._stationary_probability.copy()
        log_evidence = 0.0
        for time, row in enumerate(observations):
            if time:
                probability = self._normalize(probability @ self._transition)
            log_weights = (
                np.log(np.maximum(probability, np.finfo(float).tiny))
                + self.emission_log_density(row, self.parameters)
            )
            normalization = float(logsumexp(log_weights))
            probability = np.exp(log_weights - normalization)
            log_evidence += normalization
        current = QuadratureDistribution(self.nodes, probability)
        row_mass = np.sum(self._transition, axis=1)
        relevant = self._stationary_probability > 1e-12
        return ScarOuFilterResult(
            current=current,
            predictive=self.propagate(current),
            dt=self.dt,
            log_evidence=log_evidence,
            stationary_tail_mass_bound=self.stationary_tail_mass_bound,
            maximum_transition_mass_deficit=float(
                np.max(np.abs(row_mass[relevant] - 1.0))
            ),
        )

    def mixture_cdf(
            self,
            free_u,
            distribution: QuadratureDistribution,
            conditional_cdf: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ) -> np.ndarray:
        values = np.asarray(free_u, dtype=np.float64)
        flat = values.ravel()
        output = np.empty_like(flat)
        block_size = max(1, 750_000 // self.n_nodes)
        for start in range(0, len(flat), block_size):
            stop = min(len(flat), start + block_size)
            component = conditional_cdf(flat[start:stop], self.parameters)
            output[start:stop] = component @ distribution.probability
        output[flat <= 0.0] = 0.0
        output[flat >= 1.0] = 1.0
        return output.reshape(values.shape)


def _simulate_ou_states(n, kappa, mu, nu, rng) -> np.ndarray:
    dt = 1.0 / (n - 1)
    stationary_sigma = nu / np.sqrt(2.0 * kappa)
    persistence = np.exp(-kappa * dt)
    innovation = stationary_sigma * np.sqrt(1.0 - persistence ** 2)
    states = np.empty(n)
    states[0] = rng.normal(mu, stationary_sigma)
    for index in range(1, n):
        states[index] = rng.normal(
            mu + persistence * (states[index - 1] - mu), innovation
        )
    return states


def simulate_equicorr_scar_history(
        n, dimension, kappa, mu, nu, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate an exact-OU equicorrelation Gaussian-copula history."""

    rng = np.random.default_rng(seed)
    states = _simulate_ou_states(n, kappa, mu, nu, rng)
    parameters = equicorr_parameter_from_state(states, dimension)
    observations = np.empty((n, dimension))
    for index, parameter in enumerate(parameters):
        latent = rng.multivariate_normal(
            np.zeros(dimension), equicorrelation_matrix(dimension, parameter)
        )
        observations[index] = norm.cdf(latent)
    return observations, states


def simulate_student_scar_history(
        n, correlation, kappa, mu, nu, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate an exact-OU stochastic-df Student-copula history."""

    matrix = np.asarray(correlation, dtype=np.float64)
    rng = np.random.default_rng(seed)
    states = _simulate_ou_states(n, kappa, mu, nu, rng)
    parameters = student_df_parameter_from_state(states)
    observations = np.empty((n, len(matrix)))
    root = np.linalg.cholesky(matrix)
    for index, parameter in enumerate(parameters):
        latent = root @ rng.standard_normal(len(matrix))
        latent *= np.sqrt(parameter / rng.chisquare(parameter))
        observations[index] = t_dist.cdf(latent, df=parameter)
    return observations, states
