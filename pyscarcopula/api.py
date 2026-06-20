"""
pyscarcopula.api - stateless top-level API.

This module provides a functional interface that does NOT mutate
copula objects. Object methods such as BivariateCopula.fit() are
stateful convenience wrappers around this API.

Usage:
    from pyscarcopula.api import fit, sample, predict, predictive_mean

    copula = GumbelCopula(rotate=180)
    result = fit(copula, u, method='scar-tm-ou')

    # Simulate from fitted model (fit(copula, v) should recover similar params)
    v = sample(copula, u, result, n=2000)

    # Predict next observation (for risk metrics)
    u_pred = predict(copula, u, result, n=100000)

    # Predictive mean parameter path
    r_t = predictive_mean(copula, u, result)

All functions accept a copula + data + immutable result and return new
values. The functions in this module do not store fit state on the copula.

Note: BivariateCopula also has convenience methods (copula.predict,
copula.sample) that work after copula.fit(). These delegate
to this API internally but require copula.fit() to have been called.
"""

import numpy as np
from pyscarcopula._types import (
    FitResult,
    NumericalConfig,
    PredictConfig,
)
from pyscarcopula._utils import pobs as _pobs
from pyscarcopula.strategy._base import (
    ensure_strategy_supported,
    get_copula_capabilities,
    get_strategy,
    get_strategy_for_result,
    validate_copula_data,
)


def _as_float64_array_no_copy(value):
    if type(value) is np.ndarray and value.dtype == np.float64:
        return value
    return np.asarray(value, dtype=np.float64)


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
        'mle', 'scar-tm-ou', 'scar-tm-jacobi', 'scar-p-ou',
        'scar-m-ou', 'gas'
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
    u = _as_float64_array_no_copy(data)
    if to_pobs:
        u = _pobs(u)

    validate_copula_data(copula, u)
    ensure_strategy_supported(copula, method)
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
    **kwargs
        Forwarded to the strategy constructor when applicable.

    Returns
    -------
    float
    """
    u = np.asarray(data, dtype=np.float64)
    validate_copula_data(copula, u)
    strategy = get_strategy_for_result(result, config=config, **kwargs)
    return strategy.log_likelihood(copula, u, result)


def predictive_mean(copula, data, result: FitResult,
                    config: NumericalConfig | None = None, **kwargs) -> np.ndarray:
    """Predictive mean of the time-varying copula parameter.

    For MLE: constant array.
    For SCAR-TM-OU: E[Psi(x_k) | u_{1:k-1}] via transfer matrix.
    For SCAR-TM-JACOBI: E[theta(tau_k) | u_{1:k-1}].
    For GAS: Psi(g_t) along filtered path.

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
    validate_copula_data(copula, u)
    strategy = get_strategy_for_result(result, config=config, **kwargs)
    return strategy.predictive_mean(copula, u, result)


def mixture_h(copula, data, result: FitResult,
              config: NumericalConfig | None = None, **kwargs) -> np.ndarray:
    """h-function for vine pseudo-observation propagation.

    MLE:  h(u2, u1; theta_mle)
    SCAR: E[h(u2, u1; Psi(x_k)) | u_{1:k-1}] using predictive weights
    GAS:  h(u2, u1; Psi(g_t))

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
    validate_copula_data(copula, u)
    capabilities = get_copula_capabilities(copula)
    if capabilities is not None and not capabilities.supports_pair_ops:
        raise NotImplementedError(
            f"{type(copula).__name__} does not expose pair h-functions")
    strategy = get_strategy_for_result(result, config=config, **kwargs)
    runtime_kwargs = {
        name: kwargs[name]
        for name in ('state_cache', 'current_cache_key', 'next_cache_key')
        if name in kwargs
    }
    return strategy.mixture_h(copula, u, result, **runtime_kwargs)


def configure(blas_threads: int = 1):
    """Force the package BLAS thread policy.

    Call before importing NumPy/SciPy-heavy modules for reliable backend
    behavior. Package import applies the same policy using
    ``PYSCA_BLAS_THREADS`` or the default value ``1``.
    """
    import os
    value = str(int(blas_threads))
    os.environ['PYSCA_BLAS_THREADS'] = value
    for var in (
        'OMP_NUM_THREADS',
        'MKL_NUM_THREADS',
        'OPENBLAS_NUM_THREADS',
        'NUMEXPR_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS',
        'BLIS_NUM_THREADS',
    ):
        os.environ[var] = value


def sample(copula, data, result: FitResult, n: int,
           config: NumericalConfig | None = None, **kwargs) -> np.ndarray:
    """Generate n observations reproducing the fitted model.

    Simulates a path of length n with time-varying parameter:
      MLE:     r = const for all t
      SCAR-TM: r(t) = Psi(x(t)), x(t) simulated from OU process
      GAS:     r(t) = Psi(g(t)), g(t) via score-driven recursion

    fit(copula, sample(...)) should recover similar parameters.

    Parameters
    ----------
    copula : CopulaProtocol
    data : (T, d) pseudo-observations
        Used by non-vine strategies where model reproduction requires fitted
        data. Vine objects retain their fitted edge state and ignore this
        stateless-dispatch argument.
    result : FitResult from fit()
    n : int
        Number of observations to generate.

    Returns
    -------
    (n, d) pseudo-observations
    """
    vine_kind = _vine_kind(copula)
    if vine_kind == 'cvine':
        return copula.sample(n, **kwargs)
    if vine_kind == 'rvine':
        return copula.sample(n, **kwargs)

    u = np.asarray(data, dtype=np.float64)
    validate_copula_data(copula, u)
    strategy = get_strategy_for_result(result, config=config, **kwargs)
    return strategy.sample(copula, u, result, n, **kwargs)


def _resolve_predict_config(predict_config, given, horizon, kwargs):
    predictive_r_mode = kwargs.pop('predictive_r_mode', None)
    dynamic_conditioning = kwargs.pop('dynamic_conditioning', 'ignore')
    return_diagnostics = kwargs.pop('return_diagnostics', False)
    mcmc_steps = kwargs.pop('mcmc_steps', None)
    mcmc_burnin = kwargs.pop('mcmc_burnin', None)
    if predict_config is None:
        return PredictConfig(
            given=given,
            horizon=horizon,
            predictive_r_mode=predictive_r_mode,
            dynamic_conditioning=dynamic_conditioning,
            return_diagnostics=return_diagnostics,
            mcmc_steps=mcmc_steps,
            mcmc_burnin=mcmc_burnin,
        ).validated()
    if not isinstance(predict_config, PredictConfig):
        raise TypeError("predict_config must be PredictConfig or None")

    out = predict_config.validated()
    if given is not None:
        out = out.replace(given=given)
    if str(horizon).lower() != 'next':
        out = out.replace(horizon=horizon)
    if predictive_r_mode is not None:
        out = out.replace(predictive_r_mode=predictive_r_mode)
    if str(dynamic_conditioning).lower() != 'ignore':
        out = out.replace(dynamic_conditioning=dynamic_conditioning)
    if return_diagnostics:
        out = out.replace(return_diagnostics=True)
    if mcmc_steps is not None:
        out = out.replace(mcmc_steps=mcmc_steps)
    if mcmc_burnin is not None:
        out = out.replace(mcmc_burnin=mcmc_burnin)
    return out


def predict(copula, data, result: FitResult, n: int,
            config: NumericalConfig | None = None, given=None,
            horizon='next', predict_config: PredictConfig | None = None,
            **kwargs) -> np.ndarray:
    """Sample n observations from the predictive copula distribution.

    For edge models, the predictive parameter semantics are:
      MLE:     r = theta_mle (constant)
      SCAR-TM: mixture sampling from p(x_T | data) or p(x_{T+1} | data)
      GAS:     point estimate Psi(g_T) or Psi(g_{T+1})

    ``given`` is a conditional sampling argument in pseudo-observation
    space. For bivariate copulas it may fix coordinate 0 or 1; for vines it
    fixes vine-level coordinates. For `RVineCopula`, exact conditional
    generation requires the fixed variables to be representable at the end of
    the R-vine variable order, either in the fitted matrix itself or after
    rebuilding an equivalent natural-order matrix. If the model was fitted
    with `given_vars=...`, that target set is the advertised fit-time
    contract for the current exact sampler.

    Parameters
    ----------
    copula : CopulaProtocol
    data : pseudo-observations used as prediction history
        Passed to both C-vines and R-vines as their canonical ``u`` history.
    result : FitResult from fit()
        Ignored for vine copulas, which hold fitted edge state internally.
    given : dict[int, float] or None
        Fixed pseudo-observation coordinates.
    horizon : {'current', 'next'}
        Predictive state timing for GAS and SCAR-TM.
    n : int
        Number of samples.

    Returns
    -------
    (n, d) pseudo-observations
    """
    pcfg = _resolve_predict_config(predict_config, given, horizon, kwargs)
    vine_kind = _vine_kind(copula)
    if vine_kind == 'cvine':
        return copula.predict(
            n,
            u=data,
            given=pcfg.given,
            horizon=pcfg.horizon,
            predictive_r_mode=pcfg.predictive_r_mode,
            **kwargs,
        )
    if vine_kind == 'rvine':
        return copula.predict(
            n, u=data, predict_config=pcfg, **kwargs)

    u = np.asarray(data, dtype=np.float64)
    validate_copula_data(copula, u)
    strategy = get_strategy_for_result(result, config=config, **kwargs)
    return strategy.predict(
        copula,
        u,
        result,
        n,
        given=pcfg.given,
        horizon=pcfg.horizon,
        predictive_r_mode=pcfg.predictive_r_mode,
        **kwargs,
    )


def _is_vine_copula(obj) -> bool:
    return _vine_kind(obj) is not None


def _vine_kind(obj):
    try:
        from pyscarcopula.vine.cvine import CVineCopula
        from pyscarcopula.vine.rvine import RVineCopula
    except ImportError:
        return None

    if isinstance(obj, CVineCopula):
        return 'cvine'
    if isinstance(obj, RVineCopula):
        return 'rvine'
    return None
