"""
pyscarcopula.api — stateless top-level API.

This module provides a functional interface that does NOT mutate
copula objects. It replaces the old pattern of copula.fit() + copula.fit_result.

Usage:
    from pyscarcopula.api import fit, smoothed_params

    result = fit(copula, data, method='scar-tm-ou', to_pobs=True)
    r_t = smoothed_params(copula, data, result)

All functions accept a copula (stateless) + data + result (immutable),
and return new immutable values. No side effects.
"""

import numpy as np
from pyscarcopula._types import FitResult, NumericalConfig, DEFAULT_CONFIG
from pyscarcopula._utils import pobs as _pobs
from pyscarcopula.strategy._base import get_strategy


def fit(copula, data, method='scar-tm-ou', to_pobs=False,
        config: NumericalConfig | None = None, **kwargs) -> FitResult:
    """Fit a copula to data. Does not mutate the copula.

    Parameters
    ----------
    copula : CopulaProtocol
        Stateless copula object.
    data : (T, 2) array
        Log-returns or pseudo-observations.
    method : str
        'mle', 'scar-tm-ou', 'scar-p-ou', 'scar-m-ou', 'gas'
    to_pobs : bool
        Transform data to pseudo-observations first.
    config : NumericalConfig or None
        Override default numerical constants.
    **kwargs
        Forwarded to the strategy's fit() method.

    Returns
    -------
    FitResult (immutable dataclass)
    """
    u = np.asarray(data, dtype=np.float64)
    if to_pobs:
        u = _pobs(u)

    strategy = get_strategy(method, config=config, **kwargs)
    return strategy.fit(copula, u, **kwargs)


def log_likelihood(copula, data, result: FitResult,
                   config: NumericalConfig | None = None, **kwargs) -> float:
    """Evaluate log-likelihood at fitted parameters.

    Parameters
    ----------
    copula : CopulaProtocol
    data : (T, 2) pseudo-observations
    result : FitResult from fit()

    Returns
    -------
    float
    """
    u = np.asarray(data, dtype=np.float64)
    strategy = get_strategy(result.method, config=config, **kwargs)
    return strategy.log_likelihood(copula, u, result)


def smoothed_params(copula, data, result: FitResult,
                    config: NumericalConfig | None = None, **kwargs) -> np.ndarray:
    """Time-varying copula parameter.

    For MLE: constant array.
    For SCAR-TM: E[Psi(x_k) | u_{1:k-1}] via transfer matrix.
    For GAS: Psi(f_t) along filtered path.

    Parameters
    ----------
    copula : CopulaProtocol
    data : (T, 2) pseudo-observations
    result : FitResult from fit()

    Returns
    -------
    (T,) array of copula parameters
    """
    u = np.asarray(data, dtype=np.float64)
    strategy = get_strategy(result.method, config=config, **kwargs)
    return strategy.smoothed_params(copula, u, result)


def mixture_h(copula, data, result: FitResult,
              config: NumericalConfig | None = None, **kwargs) -> np.ndarray:
    """h-function for vine pseudo-observation propagation.

    MLE:  h(u2, u1; theta_mle)
    SCAR: E[h(u2, u1; Psi(x)) | data]  (mixture over predictive dist)
    GAS:  h(u2, u1; Psi(f_t))

    Parameters
    ----------
    copula : CopulaProtocol
    data : (T, 2) pseudo-observations
    result : FitResult from fit()

    Returns
    -------
    (T,) array
    """
    u = np.asarray(data, dtype=np.float64)
    strategy = get_strategy(result.method, config=config, **kwargs)
    return strategy.mixture_h(copula, u, result)


def configure(blas_threads: int = 1):
    """Optionally limit BLAS threads.

    Call BEFORE any computation. Replaces the old side-effect in __init__.py.
    """
    import os
    for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ.setdefault(var, str(blas_threads))
