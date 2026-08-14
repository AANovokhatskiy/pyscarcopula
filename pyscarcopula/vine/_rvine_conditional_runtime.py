"""Conditional sampling runtime helpers for ``RVineCopula``."""

import numpy as np

from pyscarcopula._constants import PSEUDO_OBS_EPS
from pyscarcopula._utils import clip_pseudo_observations
from pyscarcopula.vine._helpers import _open_unit_uniform
from pyscarcopula.vine._rvine_dag import execute_conditional_plan


def sample_dag_given_with_r(n, r_all, rng, given, plan, pair_copulas):
    """Execute a DAG conditional sampling plan with precomputed parameters."""
    missing = sorted(set(plan.edges_used) - set(r_all))
    if missing:
        raise KeyError(
            "RVineCopula._sample_dag_given_with_r: missing predicted "
            f"parameters for DAG edges {missing}"
        )
    r_payload = {
        key: {
            'edge': pair_copulas[key],
            'r': r_all[key],
        }
        for key in plan.edges_used
    }
    return execute_conditional_plan(plan, r_payload, given, n, rng)


def sample_arbitrary_given_mcmc(
        d, n, r_all, rng, given, log_pdf_rows, initial=None,
        n_steps=None, burnin_steps=None):
    """Metropolis-within-Gibbs fallback for arbitrary conditional patterns.

    One step is one coordinate update across all ``n`` parallel chains.  It
    is deliberately not called a sweep: a full sweep consists of
    ``len(free_vars)`` steps.  Keeping that unit explicit prevents callers
    from under-budgeting higher-dimensional conditional draws.
    """
    free_vars = [var for var in range(d) if var not in given]
    if not free_vars:
        out = np.empty((n, d), dtype=np.float64)
        for var in range(d):
            out[:, var] = given[var]
        return out, _empty_mcmc_diagnostics()

    if initial is None:
        current = _open_unit_uniform(rng, size=(n, d))
        for var, value in given.items():
            current[:, var] = value
    else:
        current = np.asarray(initial, dtype=np.float64).copy()
        for var, value in given.items():
            current[:, var] = value

    current_logp = log_pdf_rows(current, r_all)
    n_steps = (
        max(80, 30 * len(free_vars))
        if n_steps is None else int(n_steps)
    )
    burnin_steps = (
        max(40, 10 * len(free_vars))
        if burnin_steps is None else int(burnin_steps)
    )
    total_steps = burnin_steps + n_steps
    accepted = {int(var): 0 for var in free_vars}
    proposed = {int(var): 0 for var in free_vars}

    for step_idx in range(total_steps):
        var = free_vars[step_idx % len(free_vars)]
        proposal = current.copy()
        proposal[:, var] = _open_unit_uniform(rng, size=n)
        proposal_logp = log_pdf_rows(proposal, r_all)
        log_alpha = proposal_logp - current_logp
        accept = np.log(
            rng.uniform(PSEUDO_OBS_EPS, 1.0, size=n)) < log_alpha
        if np.any(accept):
            current[accept, var] = proposal[accept, var]
            current_logp[accept] = proposal_logp[accept]
        accepted[int(var)] += int(np.sum(accept))
        proposed[int(var)] += int(n)

    rates = {
        var: accepted[var] / proposed[var] if proposed[var] else 0.0
        for var in free_vars
    }
    proposals_per_chain = {
        var: proposed[var] / n
        for var in free_vars
    }
    accepted_per_chain = {
        var: accepted[var] / n
        for var in free_vars
    }
    rate_values = np.array(list(rates.values()), dtype=np.float64)
    has_proposals = any(proposed[var] > 0 for var in free_vars)
    acceptance_min = float(np.min(rate_values)) if has_proposals else None
    acceptance_mean = float(np.mean(rate_values)) if has_proposals else None
    acceptance_max = float(np.max(rate_values)) if has_proposals else None
    low_acceptance_warning = (
        bool(has_proposals)
        and acceptance_min is not None
        and acceptance_min < 0.02
    )
    minimum_accepted_moves_per_chain = (
        float(min(accepted_per_chain.values()))
        if has_proposals else None
    )
    insufficient_moves_warning = (
        minimum_accepted_moves_per_chain is not None
        and minimum_accepted_moves_per_chain < 5.0
    )
    warning_codes = []
    if low_acceptance_warning:
        warning_codes.append('low_acceptance')
    if insufficient_moves_warning:
        warning_codes.append('insufficient_accepted_moves')
    convergence_warning = bool(warning_codes)
    return clip_pseudo_observations(current), {
        'accepted': accepted,
        'proposed': proposed,
        'acceptance_rate': rates,
        'accepted_per_chain': accepted_per_chain,
        'proposals_per_chain': proposals_per_chain,
        'acceptance_min': acceptance_min,
        'acceptance_mean': acceptance_mean,
        'acceptance_max': acceptance_max,
        'low_acceptance_warning': low_acceptance_warning,
        'insufficient_moves_warning': insufficient_moves_warning,
        'minimum_accepted_moves_per_chain': minimum_accepted_moves_per_chain,
        'convergence_warning': convergence_warning,
        'warning_codes': tuple(warning_codes),
        'step_unit': 'single_coordinate_update',
        'n_free': len(free_vars),
        'n_steps': n_steps,
        'burnin_steps': burnin_steps,
        'total_steps': total_steps,
        'completed_sweeps': total_steps // len(free_vars),
        'partial_sweep_steps': total_steps % len(free_vars),
    }


def _empty_mcmc_diagnostics():
    return {
        'accepted': {},
        'proposed': {},
        'acceptance_rate': {},
        'accepted_per_chain': {},
        'proposals_per_chain': {},
        'acceptance_min': None,
        'acceptance_mean': None,
        'acceptance_max': None,
        'low_acceptance_warning': False,
        'insufficient_moves_warning': False,
        'minimum_accepted_moves_per_chain': None,
        'convergence_warning': False,
        'warning_codes': (),
        'step_unit': 'single_coordinate_update',
        'n_free': 0,
        'n_steps': 0,
        'burnin_steps': 0,
        'total_steps': 0,
        'completed_sweeps': 0,
        'partial_sweep_steps': 0,
    }
