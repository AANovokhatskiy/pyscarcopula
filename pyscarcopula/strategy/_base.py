"""
pyscarcopula.strategy._base — estimation strategy interface.

Each estimation method (MLE, SCAR-TM, GAS, future SCAR-Lévy, ...)
is a separate class implementing this protocol. This replaces the
string-based dispatch in the old BivariateCopula.fit().

Adding a new method = adding a new file in strategy/.
No existing code needs to change.

The protocol is intentionally minimal. Strategies that don't support
a particular operation (e.g. MLE has no smoothed_params in the
time-varying sense) should raise NotImplementedError with a clear message.
"""

from __future__ import annotations
from typing import Protocol, runtime_checkable
import numpy as np

from pyscarcopula._types import (
    FitResult,
    NumericalConfig,
    DEFAULT_CONFIG,
    PredictiveState,
)


@runtime_checkable
class FitStrategy(Protocol):
    """Interface for copula parameter estimation.

    Every strategy receives a stateless copula object and data,
    and returns an immutable FitResult. No copula mutation.

    Methods
    -------
    fit(copula, u, **kwargs) -> FitResult
        Estimate parameters.

    log_likelihood(copula, u, result) -> float
        Evaluate log-likelihood at fitted parameters.

    smoothed_params(copula, u, result) -> ndarray (T,)
        Time-varying copula parameter. For MLE this is constant.

    rosenblatt_e2(copula, u, result) -> ndarray (T,)
        Second Rosenblatt residual for GoF test.
    """

    def fit(self, copula, u: np.ndarray, **kwargs) -> FitResult:
        """Fit copula to pseudo-observations.

        Parameters
        ----------
        copula : CopulaProtocol
        u : (T, 2) pseudo-observations

        Returns
        -------
        FitResult (immutable)
        """
        ...

    def log_likelihood(self, copula, u: np.ndarray,
                       result: FitResult) -> float:
        """Evaluate log-likelihood at fitted parameters."""
        ...

    def smoothed_params(self, copula, u: np.ndarray,
                        result: FitResult) -> np.ndarray:
        """E[Psi(x_k) | u_{1:k-1}] — time-varying copula parameter.

        For MLE: constant array.
        For SCAR-TM: forward pass on transfer matrix grid.
        For GAS: deterministic Psi(f_t) path.
        """
        ...

    def rosenblatt_e2(self, copula, u: np.ndarray,
                      result: FitResult) -> np.ndarray:
        """Second Rosenblatt residual for GoF.

        MLE:  e2 = h(u2, u1, r_mle)
        SCAR: e2 = E[h(u2, u1, Psi(x_k)) | u_{1:k-1}] (mixture)
        GAS:  e2 = h(u2, u1, Psi(f_t))
        """
        ...

    def mixture_h(self, copula, u: np.ndarray,
                  result: FitResult) -> np.ndarray:
        """h-function for vine pseudo-observations.

        MLE:  h(u2, u1; theta_mle) — constant parameter
        SCAR: E[h(u2, u1; Psi(x)) | data] — mixture over predictive
        GAS:  h(u2, u1; Psi(f_t)) — along GAS-filtered path

        This is the key function that propagates pseudo-obs through
        the vine tree. Different methods produce different pseudo-obs,
        which affects higher-tree calibration (SCAR-TM is best here).
        """
        ...

    def objective(self, copula, u: np.ndarray,
                  alpha: np.ndarray, **kwargs) -> float:
        """Minus log-likelihood at raw parameter array.

        This is the function that optimizers minimize during fit().
        Exposed for manual exploration, plotting, diagnostics.

        Parameters
        ----------
        copula : CopulaProtocol
        u : (T, 2) pseudo-observations
        alpha : (n_params,) raw parameters

        Returns
        -------
        float : -logL (returns 1e10 on failure)
        """
        ...

    def sample(self, copula, u: np.ndarray,
               result: FitResult, n: int, **kwargs) -> np.ndarray:
        """Generate n observations reproducing the fitted model.

        Simulates a path of length n with time-varying parameter:
          MLE:     r = const for all t
          SCAR-TM: r(t) = Psi(x(t)), x(t) simulated from OU process
          GAS:     r(t) = Psi(f(t)), f(t) via score-driven recursion
                   on the generated observations

        fit(copula, sample(...)) should recover similar parameters.

        Parameters
        ----------
        copula : CopulaProtocol
        u : (T, 2) — not used by all methods, but needed for GAS init
        result : FitResult from fit()
        n : int — number of observations to generate

        Returns
        -------
        (n, 2) pseudo-observations
        """
        ...

    def predict(self, copula, u: np.ndarray,
                result: FitResult, n: int, **kwargs) -> np.ndarray:
        """Sample n observations for next-step prediction.

        Conditional on observed data u_{1:T}, generate n i.i.d.
        samples from the predictive copula distribution at T+1:
          MLE:     r = theta_mle (constant)
          SCAR-TM: mixture sampling from posterior p(x_T | data)
          GAS:     r = Psi(f_T), last filtered value

        Used for risk metrics (VaR/CVaR estimation).

        Parameters
        ----------
        copula : CopulaProtocol
        u : (T, 2) pseudo-observations (conditioning data)
        result : FitResult from fit()
        n : int — number of samples

        Returns
        -------
        (n, 2) pseudo-observations
        """
        ...

    def predictive_params(self, copula, u: np.ndarray | None,
                          result: FitResult, n: int, **kwargs) -> np.ndarray:
        """Generate copula parameter values for predictive sampling.

        This is the vine-facing counterpart of ``predict``: it returns the
        predictive copula parameters rather than drawing observations.
        """
        ...

    def predictive_state(self, copula, u: np.ndarray | None,
                         result: FitResult, **kwargs) -> PredictiveState:
        """Return a strategy-specific predictive state before sampling."""
        ...

    def condition_state(self, copula, state: PredictiveState,
                        observation: np.ndarray | None,
                        result: FitResult, **kwargs) -> PredictiveState:
        """Condition a predictive state on a partial prediction-time observation."""
        ...

    def sample_params(self, copula, state: PredictiveState, n: int,
                      rng=None, **kwargs) -> np.ndarray:
        """Sample copula parameters from a predictive state."""
        ...


# ══════════════════════════════════════════════════════════════════
# Strategy registry — the one and only place with method dispatch
# ══════════════════════════════════════════════════════════════════

_REGISTRY: dict[str, type] = {}


def register_strategy(method_name: str):
    """Decorator to register a strategy class for a method name.

    Usage:
        @register_strategy('SCAR-TM-OU')
        class SCARTMStrategy:
            ...
    """
    def decorator(cls):
        _REGISTRY[method_name.upper()] = cls
        return cls
    return decorator


def get_strategy(method: str, config: NumericalConfig | None = None,
                 **kwargs) -> FitStrategy:
    """Get a strategy instance for the given method name.

    This is the ONLY place in the codebase that maps strings to classes.
    Everything else works with typed strategy objects.

    Parameters
    ----------
    method : str
        'mle', 'scar-tm-ou', 'scar-p-ou', 'scar-m-ou', 'gas'
    config : NumericalConfig or None
    **kwargs : forwarded to strategy constructor

    Returns
    -------
    FitStrategy instance

    Raises
    ------
    ValueError if method is unknown
    """
    m = method.upper()

    # Lazy registration: import strategies if requested method is missing.
    # Using `m not in _REGISTRY` instead of `not _REGISTRY` to avoid a
    # race condition: if one strategy module is imported early (e.g. GAS
    # init triggers MLE import), _REGISTRY is non-empty but incomplete,
    # and other methods like SCAR-M-OU would not be found.
    if m not in _REGISTRY:
        _import_all_strategies()

    if m not in _REGISTRY:
        available = sorted(_REGISTRY.keys())
        raise ValueError(
            f"Unknown method '{method}'. Available: {available}")

    cls = _REGISTRY[m]
    cfg = config or DEFAULT_CONFIG
    return cls(config=cfg, **kwargs)


def get_strategy_for_result(result: FitResult,
                            config: NumericalConfig | None = None,
                            **kwargs) -> FitStrategy:
    """Instantiate the strategy matching an existing FitResult."""
    result_kwargs = {}
    for name in ('K', 'grid_range', 'pts_per_sigma', 'scaling'):
        value = getattr(result, name, None)
        if value is not None:
            result_kwargs[name] = value
    result_kwargs.update(kwargs)
    return get_strategy(result.method, config=config, **result_kwargs)


def _import_all_strategies():
    """Import all strategy modules to trigger @register_strategy."""
    # These imports cause the @register_strategy decorators to fire,
    # populating _REGISTRY. This is the lazy-import replacement for
    # the old if/elif chain — but structured as a one-time registration.
    try:
        from pyscarcopula.strategy import mle       # noqa: F401
    except ImportError:
        pass
    try:
        from pyscarcopula.strategy import scar_tm   # noqa: F401
    except ImportError:
        pass
    try:
        from pyscarcopula.strategy import scar_mc   # noqa: F401
    except ImportError:
        pass
    try:
        from pyscarcopula.strategy import gas       # noqa: F401
    except ImportError:
        pass


def list_methods() -> list[str]:
    """List all registered estimation methods."""
    _import_all_strategies()
    return sorted(_REGISTRY.keys())
