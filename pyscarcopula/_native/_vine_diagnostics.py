"""Stable Python result construction for native R-vine MCMC counters."""

import numpy as np


def _mcmc_diagnostics(
        free_vars, accepted, proposed, n, n_steps, burnin_steps):
    """Build the stable public diagnostics from raw coordinate counters."""
    free_vars = [int(var) for var in free_vars]
    accepted = {int(var): int(accepted[var]) for var in free_vars}
    proposed = {int(var): int(proposed[var]) for var in free_vars}
    rates = {
        var: accepted[var] / proposed[var] if proposed[var] else 0.0
        for var in free_vars
    }
    proposals_per_chain = {
        var: proposed[var] / n if n else 0.0
        for var in free_vars
    }
    accepted_per_chain = {
        var: accepted[var] / n if n else 0.0
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
    total_steps = int(burnin_steps) + int(n_steps)
    return {
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
        'n_steps': int(n_steps),
        'burnin_steps': int(burnin_steps),
        'total_steps': total_steps,
        'completed_sweeps': total_steps // len(free_vars),
        'partial_sweep_steps': total_steps % len(free_vars),
    }


def _empty_mcmc_diagnostics():
    """Return diagnostics for a conditional problem without free variables."""
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
