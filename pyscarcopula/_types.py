"""
pyscarcopula._types - typed results and configuration.

Design decisions:
  - FitResult is a hierarchy of frozen dataclasses (not dynamically patched OptimizeResult).
  - LatentProcessParams is a flexible container for processes with different
    parameter names and counts.
  - NumericalConfig gathers all magic numbers in one place.
  - All results are immutable by convention.
"""

from __future__ import annotations
from dataclasses import dataclass, field, replace as dataclass_replace
from typing import Any
import numpy as np

from pyscarcopula._native.threads import validate_n_threads


# Numerical configuration

_LBFGSB_OPTION_NAMES = (
    'gtol',
    'ftol',
    'maxfun',
    'maxiter',
    'maxls',
    'eps',
    'maxcor',
    'finite_diff_rel_step',
)


@dataclass(frozen=True)
class LBFGSBConfig:
    """Options for SciPy's L-BFGS-B optimizer.

    ``None`` means "inherit the library default" when the configuration is
    merged into :class:`NumericalConfig`. All supplied values must be
    strictly positive.
    """

    gtol: float | None = None
    ftol: float | None = None
    maxfun: int | None = None
    maxiter: int | None = None
    maxls: int | None = None
    eps: float | None = None
    maxcor: int | None = None
    finite_diff_rel_step: float | None = None

    def merged(self, override: "LBFGSBConfig") -> "LBFGSBConfig":
        """Return a copy with non-``None`` values from ``override``."""
        values = {
            name: getattr(override, name)
            for name in _LBFGSB_OPTION_NAMES
            if getattr(override, name) is not None
        }
        return dataclass_replace(self, **values)

    def options(
        self,
        **overrides: float | int | None,
    ) -> dict[str, float | int]:
        """Return validated options suitable for ``scipy.optimize.minimize``.

        Keyword arguments override fields on this object. Unknown option
        names raise :class:`TypeError` instead of being silently ignored.
        """
        values = {
            name: getattr(self, name)
            for name in _LBFGSB_OPTION_NAMES
        }
        unknown = set(overrides).difference(values)
        if unknown:
            bad = ', '.join(sorted(unknown))
            raise TypeError(f"Unknown L-BFGS-B option(s): {bad}")
        values.update({
            name: value
            for name, value in overrides.items()
            if value is not None
        })
        return {
            name: self._validated_option(name, value)
            for name, value in values.items()
            if value is not None
        }

    @staticmethod
    def _validated_option(name: str, value: float | int) -> float | int:
        if name in ('maxfun', 'maxiter', 'maxls', 'maxcor'):
            value = int(value)
        else:
            value = float(value)
        if value <= 0:
            raise ValueError(f"{name} must be positive")
        return value


DEFAULT_MLE_OPTIMIZER = LBFGSBConfig(
    gtol=1e-3,
    maxls=20,
)
DEFAULT_GAS_OPTIMIZER = LBFGSBConfig(
    gtol=1e-3,
    ftol=1e-9,
    maxfun=4000,
    maxiter=1000,
    maxls=100,
    eps=1e-5,
)
DEFAULT_SCAR_OPTIMIZER = LBFGSBConfig(
    gtol=1e-3,
    maxfun=300,
    maxiter=100,
    maxls=20,
    eps=1e-4,
)
DEFAULT_BIVARIATE_LOG_SCAR_OPTIMIZER = LBFGSBConfig(
    gtol=1e-3,
    maxfun=300,
    maxiter=1000,
    maxls=200,
    eps=1e-4,
)
DEFAULT_EQUICORR_OPTIMIZER = LBFGSBConfig(
    gtol=1e-4,
)
DEFAULT_STOCHASTIC_STUDENT_OPTIMIZER = LBFGSBConfig(
    gtol=1e-4,
)
DEFAULT_STATIC_STUDENT_OPTIMIZER = LBFGSBConfig(
    gtol=1e-2,
    eps=1e-4,
)
DEFAULT_STOCHASTIC_STUDENT_GAS_OPTIMIZER = LBFGSBConfig(
    gtol=1e-3,
    ftol=1e-9,
    maxfun=1000,
    maxiter=1000,
    maxls=50,
    eps=1e-5,
)
DEFAULT_STOCHASTIC_STUDENT_SCAR_OPTIMIZER = LBFGSBConfig(
    gtol=1e-3,
    maxfun=300,
    maxiter=1000,
    maxls=200,
    eps=1e-4,
)
@dataclass(frozen=True)
class NumericalConfig:
    """Shared optimizer and algorithm configuration.

    This immutable object centralizes numerical tolerances, grid sizes, Monte
    Carlo defaults, and optimizer options. Unspecified optimizer fields are
    filled from the method-specific defaults during initialization.

    Examples
    --------
    >>> config = NumericalConfig(default_K=500, bisection_tol=1e-12)
    """

    # Numerical failure policy
    fail_value: float = 1e10

    # Per-call native parallelism. Parallel execution is strictly opt-in:
    # process environment variables never change this default.
    n_threads: int = 1

    # Transfer matrix defaults
    default_K: int = 300
    default_grid_range: float = 5.0
    default_pts_per_sigma: int = 4
    default_grid_method: str = 'auto'
    default_adaptive: bool = True

    # Optimizer defaults. Public fit kwargs use scipy L-BFGS-B option names.
    mle_optimizer: LBFGSBConfig = field(
        default_factory=lambda: DEFAULT_MLE_OPTIMIZER)
    gas_optimizer: LBFGSBConfig = field(
        default_factory=lambda: DEFAULT_GAS_OPTIMIZER)
    scar_optimizer: LBFGSBConfig = field(
        default_factory=lambda: DEFAULT_SCAR_OPTIMIZER)
    bivariate_scar_optimizer: LBFGSBConfig | None = None
    bivariate_log_scar_optimizer: LBFGSBConfig = field(
        default_factory=lambda: DEFAULT_BIVARIATE_LOG_SCAR_OPTIMIZER)

    # Multivariate copula optimizer defaults
    equicorr_optimizer: LBFGSBConfig = field(
        default_factory=lambda: DEFAULT_EQUICORR_OPTIMIZER)
    stochastic_student_optimizer: LBFGSBConfig = field(
        default_factory=lambda: DEFAULT_STOCHASTIC_STUDENT_OPTIMIZER)
    static_student_optimizer: LBFGSBConfig = field(
        default_factory=lambda: DEFAULT_STATIC_STUDENT_OPTIMIZER)
    stochastic_student_gas_optimizer: LBFGSBConfig = field(
        default_factory=lambda: DEFAULT_STOCHASTIC_STUDENT_GAS_OPTIMIZER)
    stochastic_student_scar_optimizer: LBFGSBConfig = field(
        default_factory=lambda: DEFAULT_STOCHASTIC_STUDENT_SCAR_OPTIMIZER)

    # Bisection (h-function inversion)
    bisection_tol: float = 1e-10
    bisection_maxiter: int = 60

    # GAS score computation
    gas_score_eps: float = 1e-4
    gas_gamma_bound: float = 20.0
    gas_beta_bound: float = 0.999

    def __post_init__(self) -> None:
        resolved_threads = validate_n_threads(self.n_threads)
        object.__setattr__(self, 'n_threads', resolved_threads)
        object.__setattr__(
            self, 'mle_optimizer',
            DEFAULT_MLE_OPTIMIZER.merged(self.mle_optimizer))
        object.__setattr__(
            self, 'gas_optimizer',
            DEFAULT_GAS_OPTIMIZER.merged(self.gas_optimizer))
        object.__setattr__(
            self, 'scar_optimizer',
            DEFAULT_SCAR_OPTIMIZER.merged(self.scar_optimizer))
        bivariate_scar_optimizer = self.bivariate_scar_optimizer
        object.__setattr__(
            self, 'bivariate_scar_optimizer',
            self.scar_optimizer
            if bivariate_scar_optimizer is None
            else DEFAULT_SCAR_OPTIMIZER.merged(
                bivariate_scar_optimizer))
        object.__setattr__(
            self, 'bivariate_log_scar_optimizer',
            DEFAULT_BIVARIATE_LOG_SCAR_OPTIMIZER.merged(
                self.bivariate_log_scar_optimizer))
        object.__setattr__(
            self, 'equicorr_optimizer',
            DEFAULT_EQUICORR_OPTIMIZER.merged(self.equicorr_optimizer))
        object.__setattr__(
            self, 'stochastic_student_optimizer',
            DEFAULT_STOCHASTIC_STUDENT_OPTIMIZER.merged(
                self.stochastic_student_optimizer))
        object.__setattr__(
            self, 'static_student_optimizer',
            DEFAULT_STATIC_STUDENT_OPTIMIZER.merged(
                self.static_student_optimizer))
        object.__setattr__(
            self, 'stochastic_student_gas_optimizer',
            DEFAULT_STOCHASTIC_STUDENT_GAS_OPTIMIZER.merged(
                self.stochastic_student_gas_optimizer))
        object.__setattr__(
            self, 'stochastic_student_scar_optimizer',
            DEFAULT_STOCHASTIC_STUDENT_SCAR_OPTIMIZER.merged(
                self.stochastic_student_scar_optimizer))


DEFAULT_CONFIG = NumericalConfig()


@dataclass(frozen=True)
class PredictConfig:
    """Prediction-time options shared by copulas, vines, and strategies.

    Parameters
    ----------
    given : dict[int, float] or None
        Coordinates fixed during conditional sampling, expressed in
        pseudo-observation space.
    horizon : {"current", "next"}
        Whether to use the filtered current state or advance it one step.
    predictive_r_mode : {"grid", "histogram"} or None
        Sampling representation for a predictive scalar parameter.
    dynamic_conditioning : {"ignore", "given_only"}
        Policy for updating dynamic vine edges from conditioned values.
    return_diagnostics : bool
        Request prediction diagnostics from models that support them.
    mcmc_steps, mcmc_burnin : int or None
        Optional non-negative counts of single-coordinate updates for
        conditional MCMC sampling. A full sweep updates every free variable.

    Notes
    -----
    Call :meth:`validated` after direct construction, or use :meth:`replace`,
    to normalize string values and validate integer controls.
    """

    given: dict[int, float] | None = None
    horizon: str = 'next'
    predictive_r_mode: str | None = None
    dynamic_conditioning: str = 'ignore'
    return_diagnostics: bool = False
    mcmc_steps: int | None = None
    mcmc_burnin: int | None = None

    def validated(self) -> "PredictConfig":
        """Return a normalized, validated copy of this configuration."""
        horizon = str(self.horizon).lower()
        if horizon not in ('current', 'next'):
            raise ValueError("horizon must be 'current' or 'next'")

        mode = self.predictive_r_mode
        if mode is not None:
            mode = str(mode).lower()
            if mode not in ('grid', 'histogram'):
                raise ValueError(
                    "predictive_r_mode must be 'grid' or 'histogram'"
                )

        dynamic_conditioning = str(self.dynamic_conditioning).lower()
        if dynamic_conditioning not in ('ignore', 'given_only'):
            raise ValueError(
                "dynamic_conditioning must be 'ignore' or 'given_only'"
            )

        mcmc_steps = self._validate_optional_nonnegative_int(
            self.mcmc_steps, 'mcmc_steps')
        mcmc_burnin = self._validate_optional_nonnegative_int(
            self.mcmc_burnin, 'mcmc_burnin')

        given = dict(self.given) if isinstance(self.given, dict) else self.given
        return dataclass_replace(
            self,
            given=given,
            horizon=horizon,
            predictive_r_mode=mode,
            dynamic_conditioning=dynamic_conditioning,
            mcmc_steps=mcmc_steps,
            mcmc_burnin=mcmc_burnin,
        )

    def replace(self, **kwargs: Any) -> "PredictConfig":
        """Return a validated copy with selected fields replaced."""
        return dataclass_replace(self, **kwargs).validated()

    @staticmethod
    def _validate_optional_nonnegative_int(
        value: int | None,
        name: str,
    ) -> int | None:
        if value is None:
            return None
        if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)):
            raise TypeError(f"{name} must be a non-negative integer or None")
        value = int(value)
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
        return value


@dataclass(frozen=True)
class PredictiveState:
    """Strategy-level predictive state used to sample copula parameters.

    This is an internal interchange object. Depending on ``kind``, the state
    is represented either by a point/path in ``r`` or by probabilities on
    ``z_grid``. ``metadata`` carries strategy-specific diagnostics.
    """

    method: str
    horizon: str
    kind: str
    r: np.ndarray | None = None
    z_grid: np.ndarray | None = None
    prob: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# Latent process parameters

@dataclass(frozen=True)
class LatentProcessParams:
    """Parameters of a latent stochastic process.

    This is intentionally generic: different processes have different
    parameter sets and different numbers of parameters.

    For OU process: names=('kappa', 'mu', 'nu'), values=(49.97, 2.42, 10.65)
    For fBm-style models: names=('H', 'mu', 'sigma'), values=(0.7, 0, 1)

    The named access (params.kappa) goes through __getattr__,
    the positional access (params.values[0]) is always available.
    """

    process_type: str                          # 'ou', 'levy', 'fbm', ...
    names: tuple[str, ...]                     # parameter names
    values: np.ndarray                         # parameter values (1D)
    bounds_lower: np.ndarray | None = None     # per-param lower bounds
    bounds_upper: np.ndarray | None = None     # per-param upper bounds

    def __post_init__(self) -> None:
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
        """Named access: params.kappa, params.mu, etc."""
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
        """Number of named process parameters."""
        return len(self.names)

    def to_dict(self) -> dict[str, float]:
        """Return process parameters keyed by their names."""
        return {
            name: float(value)
            for name, value in zip(self.names, self.values)
        }

    def replace(self, **kwargs: float) -> LatentProcessParams:
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

    def __repr__(self) -> str:
        pairs = ', '.join(f'{n}={v:.4f}' for n, v in zip(self.names, self.values))
        return f"LatentProcessParams({self.process_type}: {pairs})"


def ou_params(kappa: float, mu: float, nu: float) -> LatentProcessParams:
    """Convenience constructor for OU process parameters."""
    return LatentProcessParams(
        process_type='ou',
        names=('kappa', 'mu', 'nu'),
        values=np.array([kappa, mu, nu]),
        bounds_lower=np.array([0.001, -np.inf, 0.001]),
        bounds_upper=np.array([np.inf, np.inf, np.inf]),
    )


def jacobi_params(kappa: float, m: float, xi: float) -> LatentProcessParams:
    """Convenience constructor for Jacobi diffusion parameters.

    The process evolves Kendall's tau directly:
    d tau_t = kappa * (m - tau_t) dt + xi * sqrt(tau_t * (1 - tau_t)) dW_t.
    """
    return LatentProcessParams(
        process_type='jacobi',
        names=('kappa', 'm', 'xi'),
        values=np.array([kappa, m, xi]),
        bounds_lower=np.array([0.001, 1e-6, 0.001]),
        bounds_upper=np.array([np.inf, 1.0 - 1e-6, np.inf]),
    )


def gas_params(omega: float, gamma: float, beta: float,
               gamma_bound: float = 10.0,
               beta_bound: float = 0.999) -> LatentProcessParams:
    """Convenience constructor for GAS process parameters.

    Bounds used by GASProcess.fit():
      omega: unbounded
      gamma: [-gamma_bound, gamma_bound] (score sensitivity can be negative)
      beta:  (-beta_bound, beta_bound) (persistence, |beta| < 1)
    """
    gamma_bound = float(gamma_bound)
    beta_bound = float(beta_bound)
    return LatentProcessParams(
        process_type='gas',
        names=('omega', 'gamma', 'beta'),
        values=np.array([omega, gamma, beta]),
        bounds_lower=np.array([-np.inf, -gamma_bound, -beta_bound]),
        bounds_upper=np.array([np.inf, gamma_bound, beta_bound]),
    )


# Fit results

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
    """Result of an MLE fit with optional additional static parameters."""

    copula_param: float = 0.0
    parameter_count: int = 1
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.parameter_count, (bool, np.bool_)) or not isinstance(
                self.parameter_count, (int, np.integer)):
            raise TypeError("parameter_count must be a positive integer")
        if int(self.parameter_count) <= 0:
            raise ValueError("parameter_count must be positive")
        object.__setattr__(self, 'parameter_count', int(self.parameter_count))

    @property
    def n_params(self) -> int:
        return self.parameter_count

    def _repr_lines(self) -> list[str]:
        lines = [f"   copula_param: {self.copula_param:.6f}"]
        if self.parameter_count != 1:
            lines.append(f"       n_params: {self.parameter_count}")
        return lines


@dataclass(frozen=True, repr=False)
class MultivariateMLEResult(MLEResult):
    """MLE result with explicit multivariate parameters and correlation."""

    copula_param: float | None = None
    n_observations: int = 1
    model_parameters: dict[str, Any] = field(default_factory=dict)
    correlation_matrix: np.ndarray | None = None

    def __post_init__(self):
        if isinstance(self.parameter_count, (bool, np.bool_)) or not isinstance(
                self.parameter_count, (int, np.integer)):
            raise TypeError(
                "parameter_count must be a non-negative integer")
        if int(self.parameter_count) < 0:
            raise ValueError("parameter_count must be non-negative")
        object.__setattr__(self, "parameter_count", int(self.parameter_count))
        if (
                isinstance(self.n_observations, (bool, np.bool_))
                or not isinstance(self.n_observations, (int, np.integer))):
            raise TypeError("n_observations must be a positive integer")
        if int(self.n_observations) <= 0:
            raise ValueError("n_observations must be positive")
        object.__setattr__(
            self, "n_observations", int(self.n_observations))

        if self.correlation_matrix is not None:
            correlation = np.array(
                self.correlation_matrix, dtype=np.float64, copy=True)
            if (
                    correlation.ndim != 2
                    or correlation.shape[0] != correlation.shape[1]):
                raise ValueError(
                    "correlation_matrix must be a square matrix")
            object.__setattr__(
                self, "correlation_matrix", correlation)

    @property
    def aic(self) -> float:
        """Akaike information criterion."""
        return 2.0 * self.parameter_count - 2.0 * self.log_likelihood

    @property
    def bic(self) -> float:
        """Bayesian information criterion."""
        return (
            np.log(self.n_observations) * self.parameter_count
            - 2.0 * self.log_likelihood
        )

    def _repr_lines(self) -> list[str]:
        lines = []
        for name, value in self.model_parameters.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                lines.append(f" {name:>14s}: {float(value):.6f}")
        lines.append(f"       n_params: {self.parameter_count}")
        lines.extend([
            f"            aic: {self.aic:.6f}",
            f"            bic: {self.bic:.6f}",
        ])
        return lines


@dataclass(frozen=True, repr=False)
class LatentResult(FitResultBase):
    """Result of any latent-process fit.

    The process parameters live in `params` (LatentProcessParams),
    which is generic over the number and names of parameters.
    This keeps the result type independent of the parameter count.
    """

    params: LatentProcessParams = field(
        default_factory=lambda: ou_params(1.0, 0.0, 1.0))

    # Grid / solver metadata (varies by method)
    K: int | None = None                     # grid size (TM)
    grid_range: float | None = None          # TM grid range
    pts_per_sigma: int | None = None         # TM adaptive resolution
    transition_method: str | None = None     # TM transition selector
    max_K: int | None = None                 # TM adaptive grid cap
    r_gh: float | None = None                # TM local-transition threshold
    gh_order: int | None = None              # TM local quadrature order
    auto_small_kdt: float | None = None      # auto GH threshold
    spectral_basis_order: int | str | None = None  # Hermite basis size/mode
    spectral_quad_order: int | None = None   # Hermite quadrature size
    diagnostics: dict[str, Any] = field(default_factory=dict)
    parameter_count: int | None = None        # latent plus fitted static params
    # Jacobi options that change likelihood, prediction, or admissibility.
    # Appended to preserve positional compatibility of older result fields.
    tau_eps: float = 1e-6
    theta_cap: float | None = None
    clip_negative: bool = False
    negative_mass_tol: float = 1e-5
    stationary_shape_max: float | None = 500.0
    # Unconditional Jacobi sampling configuration.
    sampling_method: str = "tm_grid"
    lamperti_substeps: int = 8
    lamperti_boundary: str = "reflect"
    lamperti_eps: float = 1e-10
    lamperti_engine: str = "numba"
    lamperti_chunk_observations: int = 4096
    memory_budget_bytes: int | None = None
    transition_storage: str = "dense"
    stationarity_correction: str = "none"
    # Appended to preserve positional compatibility of existing result fields.
    grid_method: str | None = None
    adaptive: bool | None = None

    def __post_init__(self):
        parameter_count = self.parameter_count
        if parameter_count is None:
            parameter_count = self.params.n_params
        if isinstance(parameter_count, (bool, np.bool_)) or not isinstance(
                parameter_count, (int, np.integer)):
            raise TypeError("parameter_count must be an integer or None")
        parameter_count = int(parameter_count)
        if parameter_count < self.params.n_params:
            raise ValueError(
                "parameter_count cannot be smaller than the latent "
                "parameter count")
        object.__setattr__(self, 'parameter_count', parameter_count)

    @property
    def n_params(self) -> int:
        return self.parameter_count

    def _repr_lines(self) -> list[str]:
        lines = []
        for name, val in zip(self.params.names, self.params.values):
            lines.append(f"         {name:>5s}: {val:.6f}")
        if self.parameter_count != self.params.n_params:
            lines.append(f"       n_params: {self.parameter_count}")
        if self.K is not None:
            lines.append(f"              K: {self.K}")
        if self.grid_range is not None:
            lines.append(f"     grid_range: {self.grid_range}")
        if self.transition_method is not None:
            lines.append(f"transition_method: {self.transition_method}")
        if self.transition_storage != "dense":
            lines.append(f"transition_storage: {self.transition_storage}")
        if self.stationarity_correction != "none":
            lines.append(
                f"stationarity_correction: {self.stationarity_correction}")
        if self.max_K is not None:
            lines.append(f"          max_K: {self.max_K}")
        if self.gh_order is not None:
            lines.append(f"       gh_order: {self.gh_order}")
        if self.auto_small_kdt is not None:
            lines.append(f" auto_small_kdt: {self.auto_small_kdt}")
        if self.spectral_basis_order is not None:
            lines.append(f"spectral_basis_order: {self.spectral_basis_order}")
        if self.spectral_quad_order is not None:
            lines.append(f"spectral_quad_order: {self.spectral_quad_order}")
        return lines


@dataclass(frozen=True, repr=False)
class GASResult(FitResultBase):
    """Result of GAS fit."""

    params: LatentProcessParams = field(
        default_factory=lambda: gas_params(0.0, 0.0, 0.0))
    scaling: str = 'unit'                    # 'unit' or 'fisher'
    score_eps: float = DEFAULT_CONFIG.gas_score_eps
    r_last: float = 0.0                      # one-step-ahead r value
    diagnostics: dict[str, Any] = field(default_factory=dict)
    parameter_count: int | None = None        # GAS plus fitted static params

    def __post_init__(self):
        parameter_count = self.parameter_count
        if parameter_count is None:
            parameter_count = self.params.n_params
        if isinstance(parameter_count, (bool, np.bool_)) or not isinstance(
                parameter_count, (int, np.integer)):
            raise TypeError("parameter_count must be an integer or None")
        parameter_count = int(parameter_count)
        if parameter_count < self.params.n_params:
            raise ValueError(
                "parameter_count cannot be smaller than the GAS "
                "parameter count")
        object.__setattr__(self, 'parameter_count', parameter_count)

    @property
    def omega(self) -> float:
        return self.params.omega

    @property
    def gamma(self) -> float:
        return self.params.gamma

    @property
    def beta(self) -> float:
        return self.params.beta

    @property
    def n_params(self) -> int:
        return self.parameter_count

    def _repr_lines(self) -> list[str]:
        lines = []
        for name, val in zip(self.params.names, self.params.values):
            lines.append(f"         {name:>5s}: {val:.6f}")
        if self.parameter_count != self.params.n_params:
            lines.append(f"       n_params: {self.parameter_count}")
        lines.append(f"        scaling: {self.scaling}")
        lines.append(
            f"      score_eps: "
            f"{getattr(self, 'score_eps', DEFAULT_CONFIG.gas_score_eps)}")
        return lines


@dataclass(frozen=True, repr=False)
class IndependentResult(FitResultBase):
    """Result for IndependentCopula: 0 params, logL=0.

    copula_param is always 0.0; present for interface uniformity.
    so that code like edge.fit_result.copula_param works without
    isinstance checks.
    """

    copula_param: float = 0.0

    @property
    def n_params(self) -> int:
        return 0


# Union type for consumers
FitResult = (
    MLEResult
    | MultivariateMLEResult
    | LatentResult
    | GASResult
    | IndependentResult
)
