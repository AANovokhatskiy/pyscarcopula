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

from pyscarcopula._types import FitResult, NumericalConfig, DEFAULT_CONFIG


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

    # Lazy registration: import strategies on first use
    if not _REGISTRY:
        _import_all_strategies()

    if m not in _REGISTRY:
        available = sorted(_REGISTRY.keys())
        raise ValueError(
            f"Unknown method '{method}'. Available: {available}")

    cls = _REGISTRY[m]
    cfg = config or DEFAULT_CONFIG
    return cls(config=cfg, **kwargs)


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
    if not _REGISTRY:
        _import_all_strategies()
    return sorted(_REGISTRY.keys())
