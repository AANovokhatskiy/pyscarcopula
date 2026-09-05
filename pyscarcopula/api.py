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

from contextlib import nullcontext
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pyscarcopula._types import (
    FitResult,
    NumericalConfig,
    PredictConfig,
)
from pyscarcopula._native.registry import registry_entry_for
from pyscarcopula._utils import pobs as _pobs
from pyscarcopula.numerical._arrays import as_float64_array
from pyscarcopula.strategy._base import (
    ensure_strategy_supported,
    get_strategy,
    get_strategy_for_result,
    is_multivariate_copula,
    partition_strategy_fit_kwargs,
    partition_strategy_operation_kwargs,
    validate_copula_data,
    validate_raw_copula_data,
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


def _reject_vine_postfit_config(
    config: NumericalConfig | None,
    operation: str,
) -> None:
    """Reject a configuration that VineCopula post-fit methods cannot use."""
    if config is not None:
        raise TypeError(
            f"config is not supported for VineCopula {operation}")


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
    copula: object,
    data: ArrayLike,
    method: str = 'scar-tm-ou',
    to_pobs: bool = False,
    config: NumericalConfig | None = None,
    **kwargs: Any,
) -> FitResult:
    """Fit a copula to data.

    Parameters
    ----------
    copula : object
        Exact registered built-in copula to fit. The fitted result and training
        data are also stored on the instance for its stateful convenience
        methods.
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
    registry_entry_for(copula)
    if _is_generic_vine(copula):
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
        if to_pobs:
            u = validate_raw_copula_data(copula, data)
            u = _pobs(u)
        else:
            u = _as_float64_array_no_copy(data)
        validate_copula_data(copula, u)
    constructor_kwargs, fit_kwargs = partition_strategy_fit_kwargs(
        method,
        kwargs,
    )
    ensure_strategy_supported(copula, method)
    strategy = get_strategy(
        method,
        config=config,
        **constructor_kwargs,
    )
    multivariate = is_multivariate_copula(copula)
    multivariate_mle = multivariate and method.upper() == "MLE"
    transaction = copula._fit_transaction() if multivariate else nullcontext()
    with transaction:
        # A fit owns its training snapshot before any native call releases
        # the GIL. Immutable prepared statistics already own their buffers.
        if not prepared_input:
            u = u.copy()
        if multivariate and not multivariate_mle:
            copula._prepare_dynamic_fit(u)
        result = strategy.fit(copula, u, **fit_kwargs)
        if multivariate_mle and not result.success:
            # Static MLE publishes only independently accepted candidates.
            return result
        if multivariate and not multivariate_mle:
            result = copula._finalize_dynamic_fit(result)
        # Keep the returned dynamic candidate (including its success flag),
        # never an intermediate MLE used to initialize the optimizer.
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
    copula: object,
    data: ArrayLike,
    result: FitResult,
    config: NumericalConfig | None = None,
    **kwargs: Any,
) -> float:
    """Evaluate log-likelihood at fitted parameters.

    Parameters
    ----------
    copula : object
        Copula associated with ``result``.
    data : array_like of shape (n_observations, n_dimensions)
        Pseudo-observations at which to evaluate the fitted model.
    result : FitResult
        Result returned by :func:`fit`.
    config : NumericalConfig or None
        Numerical and optimizer settings. VineCopula post-fit dispatch does
        not support this argument and accepts only ``None``.
    **kwargs
        Forwarded to the strategy constructor when applicable.

    Returns
    -------
    float
        Total log-likelihood over all observations.
    """
    registry_entry_for(copula)
    if _is_generic_vine(copula):
        _reject_vine_postfit_config(config, "log_likelihood")
        return float(copula.log_likelihood(data, **kwargs))

    prepared = _prepared_equicorr_or_none(copula, data)
    if prepared is None:
        u = _as_float64_array_no_copy(data)
        validate_copula_data(copula, u)
    else:
        u = prepared
    constructor_kwargs, operation_kwargs = partition_strategy_operation_kwargs(
        result.method, "log_likelihood", kwargs)
    strategy = get_strategy_for_result(
        result, config=config, **constructor_kwargs)
    return strategy.log_likelihood(copula, u, result, **operation_kwargs)


def predictive_mean(
    copula: object,
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
    copula : object
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
    registry_entry_for(copula)
    prepared = _prepared_equicorr_or_none(copula, data)
    if prepared is None:
        u = _as_float64_array_no_copy(data)
        validate_copula_data(copula, u)
    else:
        u = prepared
    _reject_public_posterior_cache(kwargs)
    constructor_kwargs, operation_kwargs = partition_strategy_operation_kwargs(
        result.method, "predictive_mean", kwargs)
    strategy = get_strategy_for_result(
        result, config=config, **constructor_kwargs)
    return strategy.predictive_mean(copula, u, result, **operation_kwargs)


def mixture_h(
    copula: object,
    data: ArrayLike,
    result: FitResult,
    config: NumericalConfig | None = None,
    **kwargs: Any,
) -> FloatArray:
    """h-function for vine pseudo-observation propagation.

    MLE:  h_{2|1}(u2 | u1; theta_mle)
    SCAR: E[h_{2|1}(u2 | u1; Psi(x_k)) | u_{1:k-1}] using predictive weights
    GAS:  h_{2|1}(u2 | u1; Psi(g_t))

    The conditional direction uses the original copula's variable order,
    including for asymmetric 90/270-degree rotations.

    Parameters
    ----------
    copula : object
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
    if is_multivariate_copula(copula):
        raise NotImplementedError(
            f"{type(copula).__name__} does not expose pair h-functions")
    _reject_public_posterior_cache(kwargs)
    constructor_kwargs, operation_kwargs = partition_strategy_operation_kwargs(
        result.method, "mixture_h", kwargs)
    strategy = get_strategy_for_result(
        result, config=config, **constructor_kwargs)
    return strategy.mixture_h(copula, u, result, **operation_kwargs)


def sample(
    copula: object,
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
    copula : object
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
    config : NumericalConfig or None
        Numerical and optimizer settings. VineCopula post-fit dispatch does
        not support this argument and accepts only ``None``.

    Returns
    -------
    ndarray
        Simulated pseudo-observations of shape ``(n, n_dimensions)``.
    """
    registry_entry_for(copula)
    if _is_generic_vine(copula):
        _reject_vine_postfit_config(config, "sample")
        return copula.sample(n, **kwargs)

    if "n_threads" in kwargs:
        from pyscarcopula._native.threads import validate_n_threads
        kwargs["n_threads"] = validate_n_threads(kwargs["n_threads"])
    prepared = _prepared_equicorr_or_none(copula, data)
    if prepared is None:
        u = _as_float64_array_no_copy(data)
        validate_copula_data(copula, u)
    else:
        u = prepared
    constructor_kwargs, operation_kwargs = partition_strategy_operation_kwargs(
        result.method, "sample", kwargs)
    strategy = get_strategy_for_result(
        result, config=config, **constructor_kwargs)
    return strategy.sample(copula, u, result, n, **operation_kwargs)


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


def _validate_non_vine_predict_config(pcfg: PredictConfig, explicit_options=()):
    """Reject vine-only prediction controls at API and model entry points."""
    unsupported = set(explicit_options).union(
        name for name, active in (
            ('dynamic_conditioning', pcfg.dynamic_conditioning != 'ignore'),
            ('return_diagnostics', bool(pcfg.return_diagnostics)),
            ('mcmc_steps', pcfg.mcmc_steps is not None),
            ('mcmc_burnin', pcfg.mcmc_burnin is not None),
        ) if active
    )
    if unsupported:
        raise TypeError(
            f"prediction option(s) supported only by vine models: "
            f"{sorted(unsupported)}")


def predict(
    copula: object,
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
    copula : object
        Fitted copula family or vine model.
    data : array_like
        Pseudo-observations used as prediction history.
        Passed to regular-vine runtimes as their canonical ``u`` history.
    result : FitResult
        Ignored for vine copulas, which hold fitted edge state internally.
    given : dict[int, float] or None
        Fixed pseudo-observation coordinates.
    horizon : {'current', 'next'}
        Predictive state timing for GAS and SCAR-TM.
    n : int
        Number of samples.
    config : NumericalConfig or None
        Numerical and optimizer settings. VineCopula post-fit dispatch does
        not support this argument and accepts only ``None``.
    predict_config : PredictConfig or None
        Bundled prediction options. Explicit non-default arguments override
        corresponding fields in this object.
        ``dynamic_conditioning``, ``return_diagnostics``, ``mcmc_steps`` and
        ``mcmc_burnin`` are vine-only options. Non-vine models reject these
        direct keywords and non-default values in ``predict_config``.
    **kwargs
        Strategy- or vine-specific prediction options.

    Returns
    -------
    ndarray or (ndarray, dict)
        Predictive pseudo-observations of shape ``(n, n_dimensions)``. When
        diagnostics are requested by a supporting vine model, returns the
        samples together with a diagnostics mapping.
    """
    _reject_public_posterior_cache(kwargs)
    vine_option_names = {
        'dynamic_conditioning', 'return_diagnostics',
        'mcmc_steps', 'mcmc_burnin',
    }
    explicit_vine_options = vine_option_names.intersection(kwargs)
    pcfg = _resolve_predict_config(predict_config, given, horizon, kwargs)
    registry_entry_for(copula)
    if _is_generic_vine(copula):
        _reject_vine_postfit_config(config, "predict")
        return copula.predict(
            n, u=data, predict_config=pcfg, **kwargs)

    _validate_non_vine_predict_config(pcfg, explicit_vine_options)

    if "n_threads" in kwargs:
        from pyscarcopula._native.threads import validate_n_threads
        kwargs["n_threads"] = validate_n_threads(kwargs["n_threads"])
    prepared = _prepared_equicorr_or_none(copula, data)
    if prepared is None:
        u = _as_float64_array_no_copy(data)
        validate_copula_data(copula, u)
    else:
        u = prepared
    constructor_kwargs, operation_kwargs = partition_strategy_operation_kwargs(
        result.method, "predict", kwargs)
    strategy = get_strategy_for_result(
        result, config=config, **constructor_kwargs)
    return strategy.predict(
        copula,
        u,
        result,
        n,
        given=pcfg.given,
        horizon=pcfg.horizon,
        predictive_r_mode=pcfg.predictive_r_mode,
        **operation_kwargs,
    )


def _is_generic_vine(obj: object) -> bool:
    try:
        from pyscarcopula.vine.vine import VineCopula
    except ImportError:
        return False
    return isinstance(obj, VineCopula)
