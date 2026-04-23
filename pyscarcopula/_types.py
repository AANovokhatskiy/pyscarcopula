"""
pyscarcopula._types — typed results and configuration.

Design decisions:
  - FitResult is a hierarchy of frozen dataclasses (not dynamically patched OptimizeResult).
  - LatentProcessParams is a flexible container: OU has 3 params (theta, mu, nu),
    but a future Lévy or fBm process may have 2, 4, or more.
  - NumericalConfig gathers all magic numbers in one place.
  - All results are immutable — no accidental overwrites.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np


# ══════════════════════════════════════════════════════════════════
# Numerical configuration — all magic numbers live here
# ══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class NumericalConfig:
    """Central registry of numerical constants.

    Users can override defaults at construction time:
        config = NumericalConfig(default_K=500, bisection_tol=1e-12)
    """

    # Clipping / numerical safety
    eps_clip: float = 1e-10
    eps_log: float = 1e-300
    fail_value: float = 1e10

    # Transfer matrix defaults
    default_K: int = 300
    default_grid_range: float = 5.0
    default_pts_per_sigma: int = 4
    default_grid_method: str = 'auto'
    default_adaptive: bool = True

    # Optimizer defaults
    default_tol_mle: float = 1e-4
    default_tol_scar: float = 1e-3
    default_tol_gas: float = 1e-3
    default_maxfun: int = 100

    # Bisection (h-function inversion)
    bisection_tol: float = 1e-10
    bisection_maxiter: int = 60

    # GAS score computation
    gas_score_eps: float = 1e-4

    # MC samplers
    default_n_tr: int = 500
    default_M_iterations: int = 3


DEFAULT_CONFIG = NumericalConfig()


# ══════════════════════════════════════════════════════════════════
# Latent process parameters — flexible for any process type
# ══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class LatentProcessParams:
    """Parameters of a latent stochastic process.

    This is intentionally generic: different processes have different
    parameter sets and different numbers of parameters.

    For OU process: names=('theta', 'mu', 'nu'), values=(49.97, 2.42, 10.65)
    For future Lévy: names=('alpha', 'beta', 'mu', 'sigma'), values=(1.5, 0.3, 0, 1)
    For future fBm:  names=('H', 'mu', 'sigma'), values=(0.7, 0, 1)

    The named access (params.theta) goes through __getattr__,
    the positional access (params.values[0]) is always available.
    """

    process_type: str                          # 'ou', 'levy', 'fbm', ...
    names: tuple[str, ...]                     # parameter names
    values: np.ndarray                         # parameter values (1D)
    bounds_lower: np.ndarray | None = None     # per-param lower bounds
    bounds_upper: np.ndarray | None = None     # per-param upper bounds

    def __post_init__(self):
        # Ensure values is a proper numpy array
        object.__setattr__(self, 'values',
                           np.asarray(self.values, dtype=np.float64))
        if self.bounds_lower is not None:
            object.__setattr__(self, 'bounds_lower',
                               np.asarray(self.bounds_lower, dtype=np.float64))
        if self.bounds_upper is not None:
            object.__setattr__(self, 'bounds_upper',
                               np.asarray(self.bounds_upper, dtype=np.float64))
        if len(self.names) != len(self.values):
            raise ValueError(
                f"names ({len(self.names)}) and values ({len(self.values)}) "
                f"must have the same length")

    def __getattr__(self, name: str):
        """Named access: params.theta, params.mu, etc."""
        if name in ('process_type', 'names', 'values',
                     'bounds_lower', 'bounds_upper'):
            raise AttributeError(name)  # let dataclass handle these
        try:
            idx = self.names.index(name)
            return float(self.values[idx])
        except ValueError:
            raise AttributeError(
                f"'{type(self).__name__}' has no parameter '{name}'. "
                f"Available: {self.names}")

    @property
    def n_params(self) -> int:
        return len(self.names)

    def to_dict(self) -> dict[str, float]:
        return dict(zip(self.names, self.values))

    def replace(self, **kwargs) -> LatentProcessParams:
        """Return a new LatentProcessParams with some values changed."""
        d = self.to_dict()
        d.update(kwargs)
        new_values = np.array([d[n] for n in self.names])
        return LatentProcessParams(
            process_type=self.process_type,
            names=self.names,
            values=new_values,
            bounds_lower=self.bounds_lower,
            bounds_upper=self.bounds_upper,
        )

    def __repr__(self):
        pairs = ', '.join(f'{n}={v:.4f}' for n, v in zip(self.names, self.values))
        return f"LatentProcessParams({self.process_type}: {pairs})"


def ou_params(theta: float, mu: float, nu: float) -> LatentProcessParams:
    """Convenience constructor for OU process parameters."""
    return LatentProcessParams(
        process_type='ou',
        names=('theta', 'mu', 'nu'),
        values=np.array([theta, mu, nu]),
        bounds_lower=np.array([0.001, -np.inf, 0.001]),
        bounds_upper=np.array([np.inf, np.inf, np.inf]),
    )


def gas_params(omega: float, alpha: float, beta: float) -> LatentProcessParams:
    """Convenience constructor for GAS process parameters.

    Bounds match the original GASProcess.fit():
      omega: unbounded
      alpha: [-5, 5] (score sensitivity — can be negative)
      beta:  (-0.999, 0.999) (persistence, |beta| < 1 for stationarity)
    """
    return LatentProcessParams(
        process_type='gas',
        names=('omega', 'alpha', 'beta'),
        values=np.array([omega, alpha, beta]),
        bounds_lower=np.array([-np.inf, -5.0, -0.999]),
        bounds_upper=np.array([np.inf, 5.0, 0.999]),
    )


# ══════════════════════════════════════════════════════════════════
# Fit results — typed, immutable, no more hasattr()
# ══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class FitResultBase:
    """Common fields for all fit results."""

    log_likelihood: float
    method: str                 # 'MLE', 'SCAR-TM-OU', 'GAS', ...
    copula_name: str
    success: bool
    nfev: int = 0
    message: str = ''

    def _repr_lines(self) -> list[str]:
        """Override in subclasses to add method-specific fields."""
        return []

    def __repr__(self) -> str:
        lines = [
            f"         copula: {self.copula_name}",
            f"         method: {self.method}",
            f" log_likelihood: {self.log_likelihood:.6f}",
        ]
        lines.extend(self._repr_lines())
        lines.extend([
            f"        success: {self.success}",
            f"           nfev: {self.nfev}",
            f"        message: {self.message}",
        ])
        return '\n'.join(lines)


@dataclass(frozen=True, repr=False)
class MLEResult(FitResultBase):
    """Result of MLE fit: single constant copula parameter."""

    copula_param: float = 0.0

    @property
    def n_params(self) -> int:
        return 1

    def _repr_lines(self) -> list[str]:
        return [f"   copula_param: {self.copula_param:.6f}"]


@dataclass(frozen=True, repr=False)
class LatentResult(FitResultBase):
    """Result of any latent-process fit (SCAR-OU, SCAR-Lévy, ...).

    The process parameters live in `params` (LatentProcessParams),
    which is generic over the number and names of parameters.
    This way OU's 3 params and a future Lévy's 4 params use the same type.
    """

    params: LatentProcessParams = field(
        default_factory=lambda: ou_params(1.0, 0.0, 1.0))

    # Grid / solver metadata (varies by method)
    K: int | None = None                     # grid size (TM)
    grid_range: float | None = None          # TM grid range
    pts_per_sigma: int | None = None         # TM adaptive resolution
    n_tr: int | None = None                  # MC trajectory count
    M_iterations: int | None = None          # EIS iterations

    @property
    def alpha(self) -> np.ndarray:
        """Legacy access: result.alpha -> array of all param values."""
        return self.params.values.copy()

    @property
    def n_params(self) -> int:
        return self.params.n_params

    def _repr_lines(self) -> list[str]:
        lines = []
        for name, val in zip(self.params.names, self.params.values):
            lines.append(f"         {name:>5s}: {val:.6f}")
        if self.K is not None:
            lines.append(f"              K: {self.K}")
        if self.grid_range is not None:
            lines.append(f"     grid_range: {self.grid_range}")
        if self.n_tr is not None:
            lines.append(f"           n_tr: {self.n_tr}")
        if self.M_iterations is not None:
            lines.append(f"   M_iterations: {self.M_iterations}")
        return lines


@dataclass(frozen=True, repr=False)
class GASResult(FitResultBase):
    """Result of GAS fit."""

    params: LatentProcessParams = field(
        default_factory=lambda: gas_params(0.0, 0.0, 0.0))
    scaling: str = 'unit'                    # 'unit' or 'fisher'
    r_last: float = 0.0                      # one-step-ahead r value

    @property
    def omega(self) -> float:
        return self.params.omega

    @property
    def alpha_gas(self) -> float:
        return self.params.alpha

    @property
    def beta(self) -> float:
        return self.params.beta

    @property
    def n_params(self) -> int:
        return self.params.n_params

    def _repr_lines(self) -> list[str]:
        lines = []
        for name, val in zip(self.params.names, self.params.values):
            lines.append(f"         {name:>5s}: {val:.6f}")
        lines.append(f"        scaling: {self.scaling}")
        return lines


@dataclass(frozen=True, repr=False)
class IndependentResult(FitResultBase):
    """Result for IndependentCopula: 0 params, logL=0.

    copula_param is always 0.0 — present for interface uniformity
    so that code like edge.fit_result.copula_param works without
    isinstance checks.
    """

    copula_param: float = 0.0

    @property
    def n_params(self) -> int:
        return 0


# Union type for consumers
FitResult = MLEResult | LatentResult | GASResult | IndependentResult
