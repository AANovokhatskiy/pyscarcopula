"""Estimation strategy interface and registry.

Each estimation method is a separate class implementing this protocol.
Adding a method means adding a strategy module and registering it.
"""

from __future__ import annotations
from functools import lru_cache
import inspect
from typing import Protocol, runtime_checkable
import numpy as np
from pyscarcopula.numerical._arrays import (
    as_float64_array,
    as_pseudo_observation_array,
)
from pyscarcopula._native.registry import (
    native_id_for,
    query_capability,
    registry_entry_for,
    strategy_support,
)

from pyscarcopula._types import (
    FitResult,
    NumericalConfig,
    DEFAULT_CONFIG,
    PredictiveState,
)


def reject_legacy_tol(kwargs):
    """Reject the removed SciPy-style ``tol`` alias consistently."""
    if 'tol' in kwargs:
        raise TypeError("tol is not supported; use gtol")


def reject_unknown_mle_kwargs(kwargs, *, allowed=()):
    """Reject unsupported MLE keywords while retaining common fit options."""
    reject_legacy_tol(kwargs)
    unexpected = sorted(set(kwargs).difference(allowed))
    if unexpected:
        raise TypeError(
            f"unexpected MLE keyword argument(s): {unexpected}")


def reject_unknown_strategy_kwargs(method, kwargs):
    """Reject leftover constructor or fit keywords for one strategy."""
    reject_legacy_tol(kwargs)
    unexpected = sorted(kwargs)
    if unexpected:
        raise TypeError(
            f"unexpected {str(method).upper()} keyword argument(s): "
            f"{unexpected}")


def lbfgsb_overrides(
        *,
        gtol=None,
        ftol=None,
        maxfun=None,
        maxiter=None,
        maxls=None,
        eps=None,
        maxcor=None,
        finite_diff_rel_step=None):
    """Collect common L-BFGS-B option overrides for config objects."""
    return {
        'gtol': gtol,
        'ftol': ftol,
        'maxfun': maxfun,
        'maxiter': maxiter,
        'maxls': maxls,
        'eps': eps,
        'maxcor': maxcor,
        'finite_diff_rel_step': finite_diff_rel_step,
    }


def lbfgsb_options(optimizer_config, **overrides):
    """Return optimizer options from a config using common override keys."""
    return optimizer_config.options(**overrides)


_PAIR_NATIVE_IDS = frozenset({
    "Independent",
    "Clayton",
    "Frank",
    "Gumbel",
    "Joe",
    "BivariateGaussian",
})
_MULTIVARIATE_NATIVE_IDS = frozenset({
    "Gaussian",
    "Student",
    "EquicorrGaussian",
    "StochasticStudent",
})


def is_pair_copula(copula) -> bool:
    """Whether an exact registered model is a built-in pair copula."""
    return native_id_for(copula) in _PAIR_NATIVE_IDS


def has_dynamic_scalar_parameter(copula) -> bool:
    """Query native transform support for the retained dynamic families."""
    return any(
        query_capability(
            copula,
            "parameter_transform_bounds_initialization",
            dynamics,
        ).supported
        for dynamics in ("GAS", "SCAR-TM-OU", "SCAR-TM-JACOBI")
    )


def supports_conditional_sampling(copula) -> bool:
    """Query the native conditional-sampling capability."""
    return bool(query_capability(
        copula, "conditional_sampling_transform").supported)


def copula_dimension(copula, u=None) -> int | None:
    """Resolve the dimension of an exact registered built-in model."""
    native_id = native_id_for(copula)
    if native_id in _PAIR_NATIVE_IDS:
        return 2
    dimension = getattr(copula, "dimension", None)
    if dimension is None:
        dimension = getattr(copula, "d", None)
    if dimension is not None:
        return int(dimension)
    if u is not None:
        array = np.asarray(u)
        if array.ndim == 2:
            return int(array.shape[1])
    return None


def _validate_copula_data_shape(copula, array):
    """Validate observation layout against an explicit copula dimension."""
    if array.ndim != 2:
        raise ValueError(f"copula data must be 2D, got shape {array.shape}")
    if array.shape[0] == 0:
        raise ValueError("copula data must contain at least one observation")
    dimension = copula_dimension(copula)
    if dimension is not None and array.shape[1] != dimension:
        raise ValueError(
            f"{type(copula).__name__} expects {dimension} columns, "
            f"got shape {array.shape}")
    return array


def validate_raw_copula_data(copula, data):
    """Validate finite real raw observations before a rank transform."""
    registry_entry_for(copula)
    array = as_float64_array(data, name="data")
    _validate_copula_data_shape(copula, array)
    if not np.all(np.isfinite(array)):
        raise ValueError("data must contain only finite values")
    return array


def validate_copula_data(copula, u):
    """Validate 2D pseudo-observations for an exact copula dimension."""
    registry_entry_for(copula)
    array = as_pseudo_observation_array(u)
    return _validate_copula_data_shape(copula, array)


def is_multivariate_copula(copula) -> bool:
    """Whether an exact registered model is a non-vine multivariate copula."""
    return native_id_for(copula) in _MULTIVARIATE_NATIVE_IDS


def _uses_data_estimated_correlation(copula) -> bool:
    corr_num_params = getattr(copula, "_corr_num_params", None)
    if callable(corr_num_params) and int(corr_num_params()) > 0:
        return True

    if getattr(copula, "_corr_mode", None) != "fixed":
        return False

    preprocessing = getattr(copula, "_corr_preprocessing", None)
    if getattr(preprocessing, "source", None) == "kendall":
        return True
    return getattr(copula, "_R", None) is None


def _allows_gas_static_correlation(copula, method) -> bool:
    if str(method).upper() != "GAS":
        return False
    corr_mode = getattr(copula, "_corr_mode", None)
    if corr_mode not in {"fixed", "shrinkage"}:
        return False
    corr_num_params = getattr(copula, "_corr_num_params", None)
    if not callable(corr_num_params):
        return corr_mode == "fixed"
    n_corr = int(corr_num_params())
    if corr_mode == "fixed":
        return n_corr == 0
    return n_corr > 0


def ensure_strategy_supported(copula, method):
    """Reject incompatible built-in strategy selections deterministically."""
    registry_entry_for(copula)
    normalized = validate_strategy_method(str(method))
    joint_factor_dynamic_methods = {
        "GAS",
        "SCAR-TM-OU",
    }
    if (
            normalized in joint_factor_dynamic_methods
            and getattr(copula, "_corr_mode", None) == "factor"
            and getattr(copula, "factor_estimation", None) == "joint"):
        # Future work: analytical loading derivatives must be propagated
        # through the sequential GAS/SCAR recursion and tiled emissions.
        raise NotImplementedError(
            "factor_estimation='joint' is currently supported only for "
            "static MLE; dynamic GAS/SCAR joint loading estimation is "
            "not implemented")
    native_support = strategy_support(copula, normalized)
    if native_support is not None and not native_support.supported:
        if normalized == "GAS":
            raise TypeError(f"{type(copula).__name__} does not support GAS")
        if normalized == "SCAR-TM-OU":
            raise TypeError(
                f"{type(copula).__name__} does not support SCAR-TM-OU")
        if normalized == "SCAR-TM-JACOBI":
            raise TypeError(
                f"{type(copula).__name__} does not support pair "
                "Jacobi dynamics")
        raise TypeError(
            f"{type(copula).__name__} does not support {normalized}")
    if (
            normalized == "GAS"
            and getattr(copula, "_corr_mode", None) == "cholesky"
            and _uses_data_estimated_correlation(copula)):
        raise NotImplementedError(
            "GAS joint static correlation currently supports only "
            "corr_mode='shrinkage'")
    if (
            normalized not in {"MLE", "SCAR-TM-OU"}
            and not _allows_gas_static_correlation(copula, normalized)
            and _uses_data_estimated_correlation(copula)):
        raise NotImplementedError(
            "data-estimated static correlation is implemented for "
            "MLE and SCAR-TM-OU only")


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
        copula : exact registered built-in copula
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

    def mixture_h_pair(self, copula, u: np.ndarray,
                       result: FitResult,
                       **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """Return both vine h-directions from one canonical edge posterior.

        ``u`` is ordered as ``(u1, u2)``. The returned arrays are
        ``h(u2 | u1)`` and ``h(u1 | u2)`` respectively. Strategies that do
        not implement this optional optimization are handled by the vine
        adapter through the legacy two-call fallback.
        """
        ...

    def objective(self, copula, u: np.ndarray,
                  alpha: np.ndarray, **kwargs) -> float:
        """Minus log-likelihood at raw parameter array.

        This is the function that optimizers minimize during fit().
        Exposed for manual exploration, plotting, diagnostics.

        Parameters
        ----------
        copula : exact registered built-in copula
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
          SCAR-TM-JACOBI: fitted quadrature-grid Jacobi Markov trajectory
          GAS:     r(t) = Psi(g(t)), g(t) via score-driven recursion
                   on the generated observations

        fit(copula, sample(...)) should recover similar parameters.

        Parameters
        ----------
        copula : exact registered built-in copula
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
        copula : exact registered built-in copula
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


def _explicit_keyword_names(callable_object, *, excluded):
    """Return explicit keyword parameters, excluding variadic ``**kwargs``."""
    return frozenset(
        name
        for name, parameter in inspect.signature(
            callable_object).parameters.items()
        if name not in excluded
        and parameter.kind in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
    )


def _accepts_var_keywords(callable_object):
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in inspect.signature(
            callable_object).parameters.values()
    )


@lru_cache(maxsize=None)
def _strategy_keyword_contract(method: str):
    """Return constructor and fit keyword names for a registered strategy."""
    normalized = validate_strategy_method(method)
    cls = _REGISTRY[normalized]
    constructor_names = set(_explicit_keyword_names(
        cls.__init__, excluded={"self", "config"}))
    constructor_names.update(getattr(
        cls, "_constructor_keyword_aliases", ()))
    fit_names = _explicit_keyword_names(
        cls.fit, excluded={"self", "copula", "u"})
    return (
        frozenset(constructor_names),
        fit_names,
        _accepts_var_keywords(cls.__init__),
        _accepts_var_keywords(cls.fit),
        bool(getattr(cls, "_strict_keyword_contract", False)),
    )


def partition_strategy_fit_kwargs(
        method: str,
        kwargs,
        *,
        reject_unknown: bool = True):
    """Partition public fit keywords into constructor and fit options.

    Explicit strategy signatures are the source of truth. Compatibility
    aliases handled inside a constructor can be declared through
    ``_constructor_keyword_aliases`` on the strategy class.
    """
    normalized = validate_strategy_method(method)
    if reject_unknown:
        reject_legacy_tol(kwargs)
    (
        constructor_names,
        fit_names,
        constructor_var_kwargs,
        fit_var_kwargs,
        strict_contract,
    ) = _strategy_keyword_contract(normalized)
    recognized = constructor_names.union(fit_names)
    unexpected = sorted(set(kwargs).difference(recognized))
    variadic_contract = constructor_var_kwargs or fit_var_kwargs
    if (
            reject_unknown
            and unexpected
            and (strict_contract or not variadic_contract)):
        raise TypeError(
            f"unexpected {normalized} keyword argument(s): {unexpected}")
    constructor_kwargs = {
        name: value
        for name, value in kwargs.items()
        if name in constructor_names
        or (
            not strict_contract
            and constructor_var_kwargs
            and name not in recognized
        )
    }
    fit_kwargs = {
        name: value
        for name, value in kwargs.items()
        if name in fit_names
        or (
            not strict_contract
            and fit_var_kwargs
            and name not in recognized
        )
    }
    return constructor_kwargs, fit_kwargs


def register_strategy(method_name: str):
    """Decorator to register a strategy class for a method name.

    Usage:
        @register_strategy('SCAR-TM-OU')
        class SCARTMStrategy:
            ...
    """
    def decorator(cls):
        _REGISTRY[method_name.upper()] = cls
        _strategy_keyword_contract.cache_clear()
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
        'mle', 'scar-tm-ou', 'scar-tm-jacobi', or 'gas'
    config : NumericalConfig or None
    **kwargs : forwarded to strategy constructor

    Returns
    -------
    FitStrategy instance

    Raises
    ------
    ValueError if method is unknown
    """
    m = validate_strategy_method(method)

    cls = _REGISTRY[m]
    cfg = config or DEFAULT_CONFIG
    return cls(config=cfg, **kwargs)


def get_strategy_for_result(result: FitResult,
                            config: NumericalConfig | None = None,
                            **kwargs) -> FitStrategy:
    """Instantiate the strategy matching an existing FitResult."""
    result_kwargs = {}
    method = result.method.upper()

    for name in (
            'K', 'grid_range', 'grid_method', 'adaptive',
            'pts_per_sigma', 'scaling'):
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

        for name in (
                'r_gh', 'gh_order',
                'auto_small_kdt',
                'spectral_basis_order', 'spectral_quad_order',
                ):
            value = getattr(result, name, None)
            if value is not None:
                result_kwargs[name] = value

    if method == 'SCAR-TM-JACOBI':
        transition_method = getattr(result, 'transition_method', None)
        if transition_method is not None:
            result_kwargs['transition_method'] = transition_method
        gh_order = getattr(result, 'gh_order', None)
        if gh_order is not None:
            result_kwargs['gh_order'] = gh_order
        basis_order = getattr(result, 'spectral_basis_order', None)
        if basis_order is not None:
            result_kwargs['basis_order'] = basis_order
        quad_order = getattr(result, 'spectral_quad_order', None)
        if quad_order is not None:
            result_kwargs['quad_order'] = quad_order
        result_kwargs.update({
            'tau_eps': getattr(result, 'tau_eps', 1e-6),
            'theta_cap': getattr(result, 'theta_cap', None),
            'clip_negative': getattr(result, 'clip_negative', False),
            'negative_mass_tol': getattr(
                result, 'negative_mass_tol', 1e-5),
            'stationary_shape_max': getattr(
                result, 'stationary_shape_max', 500.0),
            'transition_storage': getattr(
                result, 'transition_storage', 'dense'),
            'stationarity_correction': getattr(
                result, 'stationarity_correction', 'none'),
            'sampling_method': getattr(
                result, 'sampling_method', 'tm_grid'),
            'lamperti_substeps': getattr(
                result, 'lamperti_substeps', 8),
            'lamperti_boundary': getattr(
                result, 'lamperti_boundary', 'reflect'),
            'lamperti_eps': getattr(
                result, 'lamperti_eps', 1e-10),
            'lamperti_engine': getattr(
                result, 'lamperti_engine', 'native'),
            'lamperti_chunk_observations': getattr(
                result, 'lamperti_chunk_observations', 4096),
        })
        memory_budget_bytes = getattr(
            result, 'memory_budget_bytes', None)
        if memory_budget_bytes is not None:
            result_kwargs['memory_budget_bytes'] = memory_budget_bytes

    result_kwargs.update(kwargs)
    constructor_kwargs, _ = partition_strategy_fit_kwargs(
        result.method,
        result_kwargs,
        reject_unknown=False,
    )
    return get_strategy(
        result.method,
        config=config,
        **constructor_kwargs,
    )


def _import_all_strategies():
    """Import all strategy modules to trigger @register_strategy."""
    # These imports cause the @register_strategy decorators to fire.
    # Import failures are intentionally not swallowed: a broken strategy
    # module should fail loudly instead of becoming "Unknown method".
    from pyscarcopula.strategy import mle       # noqa: F401
    from pyscarcopula.strategy import scar_tm   # noqa: F401
    from pyscarcopula.strategy import scar_jacobi  # noqa: F401
    from pyscarcopula.strategy import gas       # noqa: F401


def validate_strategy_method(method: str) -> str:
    """Return the canonical registered name or reject it before execution."""
    normalized = method.upper()
    if normalized not in _REGISTRY:
        _import_all_strategies()
    if normalized not in _REGISTRY:
        available = sorted(_REGISTRY)
        raise ValueError(
            f"Unknown method '{method}'. Available: {available}")
    return normalized


def list_methods() -> list[str]:
    """List all registered estimation methods."""
    _import_all_strategies()
    return sorted(_REGISTRY.keys())
