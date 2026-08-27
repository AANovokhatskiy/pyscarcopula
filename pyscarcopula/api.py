"""
pyscarcopula.api - top-level API helpers.

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

The functions in this module accept a copula object, data, and fitted result
objects where needed.

Note: BivariateCopula also has convenience methods (copula.predict,
copula.sample) that work after copula.fit(). These delegate
to this API internally but require copula.fit() to have been called.
"""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pyscarcopula._types import (
    FitResult,
    NumericalConfig,
    PredictConfig,
)
from pyscarcopula.copula._protocol import CommonCopulaProtocol
from pyscarcopula._native.registry import registry_entry_for
from pyscarcopula._utils import pobs as _pobs
from pyscarcopula.numerical._arrays import as_float64_array
from pyscarcopula.strategy._base import (
    ensure_strategy_supported,
    get_copula_capabilities,
    get_strategy,
    get_strategy_for_result,
    validate_copula_data,
)


FloatArray: TypeAlias = NDArray[np.float64]
PredictOutput: TypeAlias = FloatArray | tuple[FloatArray, dict[str, Any]]


def _as_float64_array_no_copy(value: ArrayLike) -> FloatArray:
    return as_float64_array(value, name="data")


def _reject_public_posterior_cache(kwargs: dict[str, Any]) -> None:
    if "posterior_cache" in kwargs:
        raise TypeError(
            "posterior_cache is an internal runtime cache and is not "
            "accepted by the top-level API")


def _prepared_equicorr_or_none(copula, data):
    from pyscarcopula.copula.multivariate.equicorr import (
        EquicorrGaussianCopula,
    )
    from pyscarcopula.copula.multivariate.equicorr_prepared import (
        EquicorrPreparedData,
    )

    if not isinstance(data, EquicorrPreparedData):
        return None
    if not isinstance(copula, EquicorrGaussianCopula):
        raise TypeError(
            "EquicorrPreparedData is accepted only by "
            "EquicorrGaussianCopula")
    if copula.d != data.dimension:
        raise ValueError(
            "prepared dimension does not match copula dimension")
    return data


def fit(
    copula: CommonCopulaProtocol,
    data: ArrayLike,
    method: str = 'scar-tm-ou',
    to_pobs: bool = False,
    config: NumericalConfig | None = None,
    **kwargs: Any,
) -> FitResult:
    """Fit a copula to data.

    Parameters
    ----------
    copula : CommonCopulaProtocol
        Copula instance to fit. The fitted result and training data are also
        stored on the instance for its stateful convenience methods.
    data : array_like of shape (n_observations, n_dimensions)
        Raw observations or pseudo-observations. Bivariate strategies require
        two columns; multivariate and vine models determine their own width.
    method : str
        Estimation strategy name, such as ``"mle"``, ``"scar-tm-ou"``,
        ``"scar-tm-jacobi"``, or ``"gas"``.
    to_pobs : bool
        If true, rank-transform each data column before fitting.
    config : NumericalConfig or None
        Numerical and optimizer settings. ``None`` selects library defaults.
    **kwargs
        Strategy-specific fitting options.

    Returns
    -------
    FitResult
        Immutable result object appropriate for the selected strategy.

    Raises
    ------
    ValueError
        If the data shape or values are invalid.
    TypeError
        If the selected strategy is incompatible with ``copula``.
    NotImplementedError
        If the requested strategy/model combination is recognized but not
        implemented.
    """
    if _is_vine_copula(copula):
        fitted = copula.fit(
            data,
            method=method,
            to_pobs=to_pobs,
            config=config,
            **kwargs,
        )
        return fitted.fit_result

    prepared = _prepared_equicorr_or_none(copula, data)
    prepared_input = prepared is not None
    if prepared_input:
        if to_pobs:
            raise ValueError(
                "to_pobs=True is unavailable for prepared statistics")
        if int(getattr(copula, "d", -1)) != data.dimension:
            raise ValueError(
                "prepared dimension does not match copula dimension")
        if method.upper() not in {"MLE", "GAS", "SCAR-TM-OU"}:
            raise ValueError(
                "prepared Equicorr statistics currently support only "
                "MLE, GAS, and SCAR-TM-OU strategies")
        u = data
    else:
        u = _as_float64_array_no_copy(data)
        if to_pobs:
            u = _pobs(u)
        validate_copula_data(copula, u)
    ensure_strategy_supported(copula, method)
    strategy = get_strategy(method, config=config, **kwargs)
    result = strategy.fit(copula, u, **kwargs)
    # Mirror the state synchronization of copula.fit() so convenience
    # methods (predict/sample without explicit data or result) see the
    # strategy result rather than a stale intermediate (e.g. MLE) one.
    copula.fit_result = result
    if prepared_input:
        copula._last_prepared = u
        copula._last_u = None
    else:
        copula._last_u = u
        if hasattr(copula, "_last_prepared"):
            copula._last_prepared = None
    if (
            getattr(result, "params", None) is not None
            and hasattr(copula, "_last_latent_result")):
        copula._last_latent_result = result
    return result


def log_likelihood(
    copula: CommonCopulaProtocol,
    data: ArrayLike,
    result: FitResult,
    config: NumericalConfig | None = None,
    **kwargs: Any,
) -> float:
    """Evaluate log-likelihood at fitted parameters.

    Parameters
    ----------
    copula : CommonCopulaProtocol
        Copula associated with ``result``.
    data : array_like of shape (n_observations, n_dimensions)
        Pseudo-observations at which to evaluate the fitted model.
    result : FitResult
        Result returned by :func:`fit`.
    config : NumericalConfig or None
        Numerical and optimizer settings.
    **kwargs
        Forwarded to the strategy constructor when applicable.

    Returns
    -------
    float
        Total log-likelihood over all observations.
    """
    # CVineCopula is an approved breaking removal for Stage 8.6.  Until that
    # stage it retains its legacy runtime and is intentionally outside the
    # exact-type native registry introduced for the generic VineCopula.
    if _is_legacy_cvine(copula):
        return float(copula.log_likelihood(data, **kwargs))
    registry_entry_for(copula)
    if _is_generic_vine(copula):
        return float(copula.log_likelihood(data, **kwargs))

    prepared = _prepared_equicorr_or_none(copula, data)
    if prepared is None:
        u = _as_float64_array_no_copy(data)
        validate_copula_data(copula, u)
    else:
        u = prepared
    strategy = get_strategy_for_result(result, config=config, **kwargs)
    return strategy.log_likelihood(copula, u, result)


def predictive_mean(
    copula: CommonCopulaProtocol,
    data: ArrayLike,
    result: FitResult,
    config: NumericalConfig | None = None,
    **kwargs: Any,
) -> FloatArray:
    """Predictive mean of the time-varying copula parameter.

    For MLE: constant array.
    For SCAR-TM-OU: E[Psi(x_k) | u_{1:k-1}] via transfer matrix.
    For SCAR-TM-JACOBI: E[theta(tau_k) | u_{1:k-1}].
    For GAS: Psi(g_t) along filtered path.

    Parameters
    ----------
    copula : CommonCopulaProtocol
        Fitted copula family.
    data : array_like of shape (n_observations, n_dimensions)
        Prediction history in pseudo-observation space.
    result : FitResult
        Result returned by :func:`fit`.
    config : NumericalConfig or None
        Numerical and optimizer settings.

    Returns
    -------
    ndarray
        Predictive parameter path of shape ``(n_observations,)``.
    """
    prepared = _prepared_equicorr_or_none(copula, data)
    if prepared is None:
        u = _as_float64_array_no_copy(data)
        validate_copula_data(copula, u)
    else:
        u = prepared
    _reject_public_posterior_cache(kwargs)
    strategy = get_strategy_for_result(result, config=config, **kwargs)
    return strategy.predictive_mean(copula, u, result)


def mixture_h(
    copula: CommonCopulaProtocol,
    data: ArrayLike,
    result: FitResult,
    config: NumericalConfig | None = None,
    **kwargs: Any,
) -> FloatArray:
    """h-function for vine pseudo-observation propagation.

    MLE:  h(u2, u1; theta_mle)
    SCAR: E[h(u2, u1; Psi(x_k)) | u_{1:k-1}] using predictive weights
    GAS:  h(u2, u1; Psi(g_t))

    Parameters
    ----------
    copula : CommonCopulaProtocol
        Fitted bivariate copula family.
    data : array_like of shape (n_observations, 2)
        Pair pseudo-observations.
    result : FitResult
        Result returned by :func:`fit`.
    config : NumericalConfig or None
        Numerical and optimizer settings.

    Returns
    -------
    ndarray
        Conditional CDF values of shape ``(n_observations,)``.

    Raises
    ------
    NotImplementedError
        If ``copula`` does not provide pair-copula h-functions.
    """
    registry_entry_for(copula)
    prepared = _prepared_equicorr_or_none(copula, data)
    if prepared is None:
        u = _as_float64_array_no_copy(data)
        validate_copula_data(copula, u)
    else:
        u = prepared
    capabilities = get_copula_capabilities(copula)
    if capabilities is not None and not capabilities.supports_pair_ops:
        raise NotImplementedError(
            f"{type(copula).__name__} does not expose pair h-functions")
    _reject_public_posterior_cache(kwargs)
    runtime_names = ('state_cache', 'current_cache_key', 'next_cache_key')
    strategy_kwargs = {
        name: value
        for name, value in kwargs.items()
        if name not in runtime_names
    }
    strategy = get_strategy_for_result(result, config=config, **strategy_kwargs)
    runtime_kwargs = {
        name: kwargs[name]
        for name in runtime_names
        if name in kwargs
    }
    return strategy.mixture_h(copula, u, result, **runtime_kwargs)


def sample(
    copula: CommonCopulaProtocol,
    data: ArrayLike,
    result: FitResult,
    n: int,
    config: NumericalConfig | None = None,
    **kwargs: Any,
) -> FloatArray:
    """Generate n observations reproducing the fitted model.

    Simulates a path of length n with time-varying parameter:
      MLE:     r = const for all t
      SCAR-TM: r(t) = Psi(x(t)), x(t) simulated from OU process
      SCAR-TM-JACOBI: r(t) from the fitted discrete Jacobi Markov model
      GAS:     r(t) = Psi(g(t)), g(t) via score-driven recursion

    fit(copula, sample(...)) should recover similar parameters.

    Parameters
    ----------
    copula : CommonCopulaProtocol
        Fitted copula family or vine model.
    data : array_like of shape (n_observations, n_dimensions)
        Used by non-vine strategies where model reproduction requires fitted
        data. Vine objects retain their fitted edge state and ignore this
        stateless-dispatch argument.
    result : FitResult
        Result returned by :func:`fit`. Vine models retain their edge results
        internally.
    n : int
        Number of observations to generate.

    Returns
    -------
    ndarray
        Simulated pseudo-observations of shape ``(n, n_dimensions)``.
    """
    if _is_generic_vine(copula):
        return copula.sample(n, **kwargs)
    if _is_legacy_cvine(copula):
        return copula.sample(n, **kwargs)

    prepared = _prepared_equicorr_or_none(copula, data)
    if prepared is None:
        u = _as_float64_array_no_copy(data)
        validate_copula_data(copula, u)
    else:
        u = prepared
    strategy = get_strategy_for_result(result, config=config, **kwargs)
    return strategy.sample(copula, u, result, n, **kwargs)


def _resolve_predict_config(
    predict_config: PredictConfig | None,
    given: dict[int, float] | None,
    horizon: str,
    kwargs: dict[str, Any],
) -> PredictConfig:
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


def predict(
    copula: CommonCopulaProtocol,
    data: ArrayLike,
    result: FitResult,
    n: int,
    config: NumericalConfig | None = None,
    given: dict[int, float] | None = None,
    horizon: str = 'next',
    predict_config: PredictConfig | None = None,
    **kwargs: Any,
) -> PredictOutput:
    """Sample n observations from the predictive copula distribution.

    For edge models, the predictive parameter semantics are:
      MLE:     r = theta_mle (constant)
      SCAR-TM: mixture sampling from p(x_T | data) or p(x_{T+1} | data)
      GAS:     point estimate Psi(g_T) or Psi(g_{T+1})

    ``given`` is a conditional sampling argument in pseudo-observation
    space. For bivariate copulas it may fix coordinate 0 or 1; for vines it
    fixes vine-level coordinates. For `VineCopula`, exact conditional
    generation requires the fixed variables to be representable at the end of
    the R-vine variable order, either in the fitted matrix itself or after
    rebuilding an equivalent natural-order matrix. If the model was fitted
    with `given_vars=...`, that target set is the advertised fit-time
    contract for the current exact sampler.

    Parameters
    ----------
    copula : CommonCopulaProtocol
        Fitted copula family or vine model.
    data : array_like
        Pseudo-observations used as prediction history.
        Passed to both C-vines and R-vines as their canonical ``u`` history.
    result : FitResult
        Ignored for vine copulas, which hold fitted edge state internally.
    given : dict[int, float] or None
        Fixed pseudo-observation coordinates.
    horizon : {'current', 'next'}
        Predictive state timing for GAS and SCAR-TM.
    n : int
        Number of samples.
    config : NumericalConfig or None
        Numerical and optimizer settings.
    predict_config : PredictConfig or None
        Bundled prediction options. Explicit non-default arguments override
        corresponding fields in this object.
    **kwargs
        Strategy- or vine-specific prediction options.

    Returns
    -------
    ndarray or (ndarray, dict)
        Predictive pseudo-observations of shape ``(n, n_dimensions)``. When
        diagnostics are requested by a supporting vine model, returns the
        samples together with a diagnostics mapping.
    """
    pcfg = _resolve_predict_config(predict_config, given, horizon, kwargs)
    if _is_generic_vine(copula):
        return copula.predict(
            n, u=data, predict_config=pcfg, **kwargs)
    if _is_legacy_cvine(copula):
        unsupported = []
        if pcfg.dynamic_conditioning != 'ignore':
            unsupported.append('dynamic_conditioning')
        if pcfg.return_diagnostics:
            unsupported.append('return_diagnostics')
        if pcfg.mcmc_steps is not None:
            unsupported.append('mcmc_steps')
        if pcfg.mcmc_burnin is not None:
            unsupported.append('mcmc_burnin')
        if unsupported:
            names = ', '.join(unsupported)
            raise TypeError(
                "legacy CVineCopula.predict does not support: "
                f"{names}")
        return copula.predict(
            n,
            u=data,
            given=pcfg.given,
            horizon=pcfg.horizon,
            predictive_r_mode=pcfg.predictive_r_mode,
            **kwargs,
        )

    prepared = _prepared_equicorr_or_none(copula, data)
    if prepared is None:
        u = _as_float64_array_no_copy(data)
        validate_copula_data(copula, u)
    else:
        u = prepared
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


def _is_vine_copula(obj: object) -> bool:
    return _is_generic_vine(obj) or _is_legacy_cvine(obj)


def _is_generic_vine(obj: object) -> bool:
    try:
        from pyscarcopula.vine.vine import VineCopula
    except ImportError:
        return False
    return isinstance(obj, VineCopula)


def _is_legacy_cvine(obj: object) -> bool:
    try:
        from pyscarcopula.vine.cvine import CVineCopula
    except ImportError:
        return False
    if isinstance(obj, CVineCopula):
        return True
    return False
