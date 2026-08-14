"""Independent diagnostics and density references for vine DAG+MCMC tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.special import expit, logit
from scipy.stats import kstest, norm

from ._analytical_oracles import gaussian_conditional_parameters


@dataclass(frozen=True)
class GaussianMCMCError:
    """Scale-free error summary relative to a conditional Gaussian oracle."""

    mean_rms: float
    covariance_frobenius: float
    marginal_ks: float

    @property
    def score(self) -> float:
        return self.mean_rms + self.covariance_frobenius + self.marginal_ks


def gaussian_mcmc_error(samples, correlation, given) -> GaussianMCMCError:
    """Measure a vine sample in whitened conditional-Gaussian coordinates."""

    values = np.asarray(samples, dtype=np.float64)
    oracle = gaussian_conditional_parameters(correlation, given)
    latent = norm.ppf(np.clip(
        values[:, oracle.free_indices], 1e-12, 1.0 - 1e-12
    ))
    whitened = oracle.whiten(latent)
    dimension = whitened.shape[1]
    mean_rms = float(
        np.linalg.norm(np.mean(whitened, axis=0)) / np.sqrt(dimension)
    )
    covariance = np.atleast_2d(np.cov(whitened, rowvar=False, ddof=1))
    covariance_frobenius = float(
        np.linalg.norm(covariance - np.eye(dimension), ord="fro")
        / np.sqrt(dimension)
    )
    marginal_ks = float(max(
        kstest(whitened[:, column], "norm").statistic
        for column in range(dimension)
    ))
    return GaussianMCMCError(
        mean_rms=mean_rms,
        covariance_frobenius=covariance_frobenius,
        marginal_ks=marginal_ks,
    )


def pyvine_conditional_logit_mcmc(
    model,
    n: int,
    given: Mapping[int, float],
    rng: np.random.Generator,
    *,
    sweeps: int = 180,
    proposal_scale: float = 0.9,
):
    """Sample a fixed pyvine density with a logit random-walk reference.

    This intentionally differs from production's independence proposal.  The
    target density is evaluated only through ``pyvinecopulib.Vinecop.pdf``;
    the logit Jacobian is included in the Metropolis ratio.
    """

    dimension = int(model.dim)
    n = int(n)
    sweeps = int(sweeps)
    if n <= 0 or sweeps < 0 or proposal_scale <= 0.0:
        raise ValueError("n and proposal_scale must be positive; sweeps >= 0")
    free = [index for index in range(dimension) if index not in given]
    current = rng.uniform(0.05, 0.95, size=(n, dimension))
    for variable, value in given.items():
        current[:, int(variable)] = float(value)

    tiny = np.finfo(np.float64).tiny

    def log_density(rows):
        density = np.asarray(
            model.pdf(np.asfortranarray(rows)), dtype=np.float64
        )
        if np.any(~np.isfinite(density)) or np.any(density < 0.0):
            raise AssertionError("pyvine reference returned invalid density")
        return np.log(np.maximum(density, tiny))

    current_logp = log_density(current)
    accepted = {int(variable): 0 for variable in free}
    proposed = {int(variable): 0 for variable in free}

    for _sweep in range(sweeps):
        for variable in free:
            current_u = current[:, variable].copy()
            proposed_u = expit(
                logit(current_u) + rng.normal(0.0, proposal_scale, size=n)
            )
            proposal = current.copy()
            proposal[:, variable] = proposed_u
            proposal_logp = log_density(proposal)
            log_alpha = (
                proposal_logp
                - current_logp
                + np.log(proposed_u)
                + np.log1p(-proposed_u)
                - np.log(current_u)
                - np.log1p(-current_u)
            )
            accept = np.log(rng.uniform(tiny, 1.0, size=n)) < log_alpha
            if np.any(accept):
                current[accept, variable] = proposed_u[accept]
                current_logp[accept] = proposal_logp[accept]
            accepted[int(variable)] += int(np.sum(accept))
            proposed[int(variable)] += n

    rates = {
        variable: accepted[variable] / proposed[variable]
        if proposed[variable] else 0.0
        for variable in free
    }
    return current, {
        "accepted": accepted,
        "proposed": proposed,
        "acceptance_rate": rates,
        "acceptance_min": min(rates.values()) if rates else None,
        "acceptance_mean": (
            float(np.mean(list(rates.values()))) if rates else None
        ),
        "sweeps": sweeps,
        "proposal": "logit_random_walk",
        "proposal_scale": float(proposal_scale),
    }


__all__ = [
    "GaussianMCMCError",
    "gaussian_mcmc_error",
    "pyvine_conditional_logit_mcmc",
]
