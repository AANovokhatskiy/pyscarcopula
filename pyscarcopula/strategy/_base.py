"""Estimation strategy interface and registry.

Each estimation method is a separate class implementing this protocol.
Adding a method means adding a strategy module and registering it.
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

    predictive_mean(copula, u, result) -> ndarray (T,)
        Predictive mean copula parameter. For MLE this is constant.

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

    def predictive_mean(self, copula, u: np.ndarray,
                        result: FitResult) -> np.ndarray:
        """Predictive mean of the time-varying copula parameter.

        For MLE: constant array.
        For SCAR-TM: E[Psi(x_k) | u_{1:k-1}].
        For GAS: deterministic Psi(g_t) path.
        """
        ...

    def smoothed_params(self, copula, u: np.ndarray,
                        result: FitResult) -> np.ndarray:
        """E[Psi(x_k) | u_{1:k-1}] for the time-varying copula parameter.

        The SCAR-TM quantity is predictive, not two-sided smoothing.
        """
        ...

    def rosenblatt_e2(self, copula, u: np.ndarray,
                      result: FitResult) -> np.ndarray:
        """Second Rosenblatt residual for GoF.

        MLE:  e2 = h(u2, u1, r_mle)
        SCAR: e2 = E[h(u2, u1, Psi(x_k)) | u_{1:k-1}] (mixture)
        GAS:  e2 = h(u2, u1, Psi(g_t))
        """
        ...

    def mixture_h(self, copula, u: np.ndarray,
                  result: FitResult) -> np.ndarray:
        """h-function for vine pseudo-observations.

        MLE:  h(u2, u1; theta_mle), constant parameter
        SCAR: E[h(u2, u1; Psi(x)) | data], mixture over predictive state
        GAS:  h(u2, u1; Psi(g_t)), along the GAS-filtered path

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
          GAS:     r(t) = Psi(g(t)), g(t) via score-driven recursion
                   on the generated observations

        fit(copula, sample(...)) should recover similar parameters.

        Parameters
        ----------
        copula : CopulaProtocol
        u : (T, 2)
            Not used by all methods, but needed for GAS initialization.
        result : FitResult from fit()
        n : int
            Number of observations to generate.

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
          GAS:     r = Psi(g_T), last filtered value

        Used for risk metrics (VaR/CVaR estimation).

        Parameters
        ----------
        copula : CopulaProtocol
        u : (T, 2) pseudo-observations (conditioning data)
        result : FitResult from fit()
        n : int
            Number of samples.

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

    def model_sample_params(self, copula, result: FitResult, n: int,
                            rng=None, **kwargs) -> np.ndarray:
        """Generate parameter path for unconditional model reproduction."""
        ...

    def model_sample_state(self, copula, result: FitResult,
                           **kwargs) -> PredictiveState | None:
        """Return state for stepwise model reproduction, if required."""
        ...


# Strategy registry

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
    method = result.method.upper()

    for name in ('K', 'grid_range', 'pts_per_sigma', 'scaling'):
        value = getattr(result, name, None)
        if value is not None:
            result_kwargs[name] = value

    if method == 'SCAR-TM-OU':
        transition_method = getattr(result, 'transition_method', None)
        if transition_method is None:
            result_kwargs['transition_method'] = 'matrix'
            result_kwargs['max_K'] = None
        else:
            result_kwargs['transition_method'] = transition_method
            result_kwargs['max_K'] = getattr(result, 'max_K', None)

        for name in ('r_gh', 'gh_order'):
            value = getattr(result, name, None)
            if value is not None:
                result_kwargs[name] = value

    result_kwargs.update(kwargs)
    return get_strategy(result.method, config=config, **result_kwargs)


def _import_all_strategies():
    """Import all strategy modules to trigger @register_strategy."""
    # These imports cause the @register_strategy decorators to fire.
    # Import failures are intentionally not swallowed: a broken strategy
    # module should fail loudly instead of becoming "Unknown method".
    from pyscarcopula.strategy import mle       # noqa: F401
    from pyscarcopula.strategy import scar_tm   # noqa: F401
    from pyscarcopula.strategy import scar_mc   # noqa: F401
    from pyscarcopula.strategy import gas       # noqa: F401


def list_methods() -> list[str]:
    """List all registered estimation methods."""
    _import_all_strategies()
    return sorted(_REGISTRY.keys())
