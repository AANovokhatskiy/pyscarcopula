"""
Goodness-of-fit tests for copula models (bivariate and vine).

Bivariate:
    MLE:  e2 = h_{2|1}(u2 | u1; r)
    SCAR: e2 = E[h_{2|1}(u2 | u1; Psi(x_k)) | u_{1:k-1}]  (mixture)
    Directions refer to the original column order, including rotated copulas.

C-Vine (d dimensions):
    Rosenblatt transform through the tree:
    e_1 = u_1
    e_i = h_{i-1}( ... h_1(h_0(u_i | u_1) | ...) ... )
    where each h uses the copula parameter from the corresponding edge.

    MLE:  constant r per edge
    SCAR: predictive E[Psi(x_k) | u_{1:k-1}] per edge

Under the correctly specified model: e ~ iid U[0,1]^d.
The implemented diagnostic maps each row to a scalar radial summary,
Phi^{-1}(e) -> chi2(d) CDF, and applies a one-sample CvM test against
U[0,1].  It is not a separate omnibus test of componentwise or serial
independence.

Usage:
    from pyscarcopula.stattests import gof_test, rvine_gof_test
"""

import numpy as np
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable
from scipy.stats import cramervonmises

from pyscarcopula._parallel import (
    create_worker_model,
    get_copula_constructor,
    resolve_parallelism,
    spawn_seed_sequences,
    validate_model_fit_kwargs,
    with_n_threads,
)
from pyscarcopula._utils import (
    clip_pseudo_observations,
    clip_pseudo_observations_no_copy,
    clip_rosenblatt_output,
)
from pyscarcopula.numerical._arrays import (
    as_float64_array,
    as_float64_scalar,
    as_pseudo_observation_array,
    validate_float64_allocation,
    validate_positive_int,
)
from pyscarcopula._native import _extension as _cpp_extension
from pyscarcopula._native.errors import NativeUnsupported


@dataclass(frozen=True)
class BootstrapGoFResult:
    """Goodness-of-fit result with bootstrap-calibrated p-value."""

    statistic: float
    pvalue: float
    bootstrap_statistics: np.ndarray
    n_bootstrap: int
    calibration: str = 'parametric_bootstrap'
    bootstrap_diagnostics: tuple[dict, ...] = ()
    n_jobs_requested: int = 1
    n_jobs: int = 1
    n_threads: int = 1
    backend: str = 'sequential'
    rng_policy: str = 'seed_sequence_per_replication'
    worker_model_ownership: str = 'per_task'


# ══════════════════════════════════════════════════════════════════
# CvM test (shared)
# ══════════════════════════════════════════════════════════════════

def cvm_test(e):
    """
    One-sample Cramér-von Mises test of the radial Rosenblatt summary.

    Parameters
    ----------
    e : array-like, shape (T, d)
        Under H0:
            y_t = chi2.cdf(sum_j Phi^{-1}(e_tj)^2, df=d) ~ U[0,1]

    Returns
    -------
    CramerVonMisesResult
        Has .statistic and .pvalue.  SciPy uses the conventional
        sample-size-scaled statistic

            W^2 = T * integral_0^1 (F_T(y) - y)^2 dy.

        The scalar reduction is a diagnostic consequence of the joint null;
        it does not separately test componentwise or serial independence.
    """
    e = as_pseudo_observation_array(e, name="e")
    if e.ndim != 2:
        raise ValueError(f"e must have shape (T, d), got {e.shape}")

    T, d = e.shape
    if T < 2:
        raise ValueError("e must contain at least two observations")
    if d == 0:
        raise ValueError("e must contain at least one dimension")

    from pyscarcopula._native import multivariate as multivariate_native
    y = multivariate_native.radial_uniform_summary(e)

    return cramervonmises(y, "uniform")


def _grid_transition_method(transition_method):
    if str(transition_method).lower() == 'spectral':
        return 'auto'
    return transition_method


def _prepare_gof_data(data, *, expected_dim, to_pobs):
    """Validate and normalize observations at the public GoF boundary."""
    from pyscarcopula._utils import pobs as compute_pobs

    if not isinstance(to_pobs, (bool, np.bool_)):
        raise TypeError("to_pobs must be a boolean")

    u = as_float64_array(data, name="data")
    if u.ndim != 2:
        expected = (
            "(T, d)" if expected_dim is None else f"(T, {expected_dim})")
        raise ValueError(f"data must have shape {expected}, got {u.shape}")
    if len(u) < 2:
        raise ValueError("data must contain at least two observations")
    if u.shape[1] == 0:
        raise ValueError("data must contain at least one dimension")
    if expected_dim is not None and u.shape[1] != expected_dim:
        raise ValueError(
            f"data must have shape (T, {expected_dim}), got {u.shape}")
    if not np.all(np.isfinite(u)):
        raise ValueError("data must contain only finite values")

    if to_pobs:
        u = compute_pobs(u)
    return as_pseudo_observation_array(u, name="data")


def _validated_boolean(name, value):
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a boolean")
    return bool(value)


# ══════════════════════════════════════════════════════════════════
# Bivariate Rosenblatt
# ══════════════════════════════════════════════════════════════════

def rosenblatt_transform_mle(copula, u, r):
    """Rosenblatt for constant copula parameter (MLE). Returns (T, 2)."""
    u = as_pseudo_observation_array(u, name="u")
    r = as_float64_scalar(r, name="r")
    T = len(u)
    e = np.empty((T, 2))
    e[:, 0] = u[:, 0]
    _, e[:, 1] = copula.h_pair(
        u[:, 0], u[:, 1], np.full(T, r))
    return clip_rosenblatt_output(e)


def rosenblatt_transform_scar(copula, u, alpha, K=300, grid_range=5.0,
                              grid_method='auto', adaptive=True,
                              pts_per_sigma=4, transition_method='matrix',
                              max_K=None, r_gh=3.0, gh_order=5):
    """Mixture Rosenblatt for SCAR (bivariate). Returns (T, 2)."""
    from pyscarcopula._native import scar_ou as _cpp_scar_ou
    from pyscarcopula.numerical._scar_ou_config import AutoTMConfig

    kappa, mu, nu = alpha
    config = AutoTMConfig(
        transition_method=transition_method,
        K=K,
        grid_range=grid_range,
        grid_method=grid_method,
        adaptive=adaptive,
        pts_per_sigma=pts_per_sigma,
        max_K=max_K,
        r_gh=r_gh,
        gh_order=gh_order,
    )
    e = np.empty((len(u), 2), dtype=np.float64)
    e[:, 0] = u[:, 0]
    e[:, 1] = _cpp_scar_ou.mixture_h(
        kappa, mu, nu, u, copula, config)
    return e


def rosenblatt_transform_gas(copula, u, gas_params, scaling='unit',
                             score_eps=1e-4):
    """Rosenblatt for GAS (bivariate). Returns (T, 2)."""
    from pyscarcopula.numerical.gas_filter import gas_rosenblatt
    omega, gamma, beta = gas_params
    return gas_rosenblatt(
        omega, gamma, beta, u, copula, scaling, score_eps)


# ══════════════════════════════════════════════════════════════════
# Unified tests
# ══════════════════════════════════════════════════════════════════

def gof_test(model, data, to_pobs=True, K=300, grid_range=5.0,
             fit_result=None, bootstrap=False, n_bootstrap=199,
             bootstrap_refit=True, bootstrap_fit_kwargs=None, rng=None,
             n_jobs=1):
    """
    Unified goodness-of-fit test for any copula model.

    Dispatches based on model type:
      - BivariateCopula  -> bivariate Rosenblatt (MLE or SCAR mixture)
      - VineCopula       -> generic regular-vine Rosenblatt
      - GaussianCopula   -> Cholesky-based Rosenblatt
      - StudentCopula    -> conditional t-distribution Rosenblatt

    Parameters
    ----------
    model : BivariateCopula, VineCopula, GaussianCopula, or StudentCopula
    data : (T, d) array
    to_pobs : bool
    K : int — grid size (SCAR-TM-OU only)
    grid_range : float (SCAR-TM-OU only)
    fit_result : FitResult or None
        If provided, use this instead of model.fit_result.
        Enables the stateless API: gof_test(copula, u, fit_result=result)
    bootstrap : bool
        If True, calibrate a supported bivariate, regular-vine, static
        multivariate, or dynamic multivariate CvM statistic by parametric
        bootstrap instead of using the one-sample asymptotic p-value.
    n_bootstrap : int
        Number of bootstrap replications.
    bootstrap_refit : bool
        If True, re-estimate the model on each bootstrap sample.
    bootstrap_fit_kwargs : dict or None
        Extra keyword arguments for each bootstrap fit.
    rng : int, Generator, SeedSequence, or None
        Random seed/source for bootstrap simulation.
    n_jobs : int
        Bootstrap worker processes. ``1`` executes sequentially; ``-1`` uses
        all available CPUs. Ignored when ``bootstrap=False``.

    Returns
    -------
    CramérVonMisesResult or BootstrapGoFResult
        The bootstrap result additionally contains calibration samples,
        per-replication diagnostics, and resolved parallelism metadata.
    """
    from pyscarcopula.copula.base import BivariateCopula
    from pyscarcopula.copula.multivariate import GaussianCopula, StudentCopula
    from pyscarcopula.vine.vine import VineCopula
    from pyscarcopula.copula.multivariate import (
        EquicorrGaussianCopula,
        StochasticStudentCopula,
    )

    bootstrap = _validated_boolean("bootstrap", bootstrap)
    bootstrap_refit = _validated_boolean(
        "bootstrap_refit", bootstrap_refit)

    if isinstance(model, StochasticStudentCopula):
        if bootstrap:
            return _gof_dynamic_multivariate(
                'stochastic_student',
                model,
                data,
                to_pobs,
                K,
                grid_range,
                fit_result,
                n_bootstrap,
                bootstrap_refit,
                bootstrap_fit_kwargs,
                rng,
                n_jobs,
            )
        return stochastic_student_gof_test(model, data, to_pobs, K,
                                           grid_range, fit_result=fit_result)
    elif isinstance(model, EquicorrGaussianCopula):
        if bootstrap:
            return _gof_dynamic_multivariate(
                'equicorr',
                model,
                data,
                to_pobs,
                K,
                grid_range,
                fit_result,
                n_bootstrap,
                bootstrap_refit,
                bootstrap_fit_kwargs,
                rng,
                n_jobs,
            )
        return equicorr_gof_test(model, data, to_pobs, K, grid_range,
                                 fit_result=fit_result)
    elif isinstance(model, BivariateCopula):
        return _gof_bivariate(model, data, to_pobs, K, grid_range,
                              fit_result=fit_result, bootstrap=bootstrap,
                              n_bootstrap=n_bootstrap,
                              bootstrap_refit=bootstrap_refit,
                              bootstrap_fit_kwargs=bootstrap_fit_kwargs,
                              rng=rng, n_jobs=n_jobs)
    elif isinstance(model, VineCopula):
        u = _prepare_gof_data(
            data, expected_dim=getattr(model, "d", None), to_pobs=to_pobs)
        result = rvine_gof_test(
            model,
            u,
            False,
            K,
            grid_range,
            vine_type=model.vine_type,
        )
        if not bootstrap:
            return result
        rvine_fit_result = (
            fit_result
            if fit_result is not None
            else getattr(model, 'fit_result', None)
        )
        if rvine_fit_result is None:
            rvine_fit_result = model
        return _bootstrap_gof(
            'rvine',
            model,
            u,
            rvine_fit_result,
            float(result.statistic),
            K=K,
            grid_range=grid_range,
            n_bootstrap=n_bootstrap,
            bootstrap_refit=bootstrap_refit,
            bootstrap_fit_kwargs=bootstrap_fit_kwargs,
            rng=rng,
            n_jobs=n_jobs,
        )
    elif isinstance(model, GaussianCopula):
        if bootstrap:
            return _gof_static_multivariate(
                'gaussian',
                model,
                data,
                to_pobs,
                fit_result,
                n_bootstrap,
                bootstrap_refit,
                bootstrap_fit_kwargs,
                rng,
                n_jobs,
            )
        return gaussian_gof_test(
            model, data, to_pobs, fit_result=fit_result)
    elif isinstance(model, StudentCopula):
        if bootstrap:
            return _gof_static_multivariate(
                'student',
                model,
                data,
                to_pobs,
                fit_result,
                n_bootstrap,
                bootstrap_refit,
                bootstrap_fit_kwargs,
                rng,
                n_jobs,
            )
        return student_gof_test(
            model, data, to_pobs, fit_result=fit_result)
    else:
        raise TypeError(f"Unsupported model type: {type(model).__name__}")

# ══════════════════════════════════════════════════════════════════
# Bivariate gof_test
# ══════════════════════════════════════════════════════════════════

def _gof_bivariate(copula, data, to_pobs=True, K=300, grid_range=5.0,
                   fit_result=None, bootstrap=False, n_bootstrap=199,
                   bootstrap_refit=True, bootstrap_fit_kwargs=None,
                   rng=None, n_jobs=1):
    """
    Goodness-of-fit for a fitted BivariateCopula.

    MLE: constant parameter Rosenblatt.
    SCAR: mixture Rosenblatt (integrates h over predictive distribution).
    GAS: deterministic Rosenblatt (h evaluated at filtered r_t).

    Parameters
    ----------
    copula : BivariateCopula
    data : (T, 2) array
    to_pobs : bool
    K : int
    grid_range : float
    fit_result : FitResult or None
        If None, uses copula.fit_result (set by copula.fit()).
    """
    u = _prepare_gof_data(data, expected_dim=2, to_pobs=to_pobs)

    fr = fit_result if fit_result is not None else getattr(copula, 'fit_result', None)
    if fr is None:
        raise ValueError("No fit_result provided and copula has no fit_result. "
                         "Call copula.fit() first or pass fit_result=.")

    if bootstrap:
        n_bootstrap = validate_positive_int(n_bootstrap, "n_bootstrap")
    e = _bivariate_rosenblatt_from_result(copula, u, fr, K, grid_range)
    result = cvm_test(e)

    if not bootstrap:
        return result

    return _bootstrap_gof_bivariate(
        copula, u, fr, float(result.statistic), K, grid_range,
        n_bootstrap=n_bootstrap, bootstrap_refit=bootstrap_refit,
        bootstrap_fit_kwargs=bootstrap_fit_kwargs, rng=rng, n_jobs=n_jobs)


def _bivariate_rosenblatt_from_result(copula, u, fit_result,
                                      K=300, grid_range=5.0):
    method = fit_result.method.upper()

    if method == 'MLE':
        r = getattr(fit_result, 'copula_param', 0.0)
        return rosenblatt_transform_mle(copula, u, r)
    if method == 'GAS':
        scaling = getattr(fit_result, 'scaling', 'unit')
        return rosenblatt_transform_gas(
            copula, u, fit_result.params.values, scaling,
            score_eps=getattr(fit_result, 'score_eps', 1e-4))

    if getattr(fit_result, 'params', None) is None:
        raise ValueError(
            f"Cannot compute bivariate Rosenblatt transform for {method}")

    from pyscarcopula.strategy._base import get_strategy_for_result

    # K/grid_range are OU-grid settings. Jacobi restores its own quadrature
    # settings from the result and must not receive unrelated OU options.
    grid_kwargs = (
        {} if method == 'SCAR-TM-JACOBI'
        else {'K': K, 'grid_range': grid_range})
    strategy = get_strategy_for_result(fit_result, **grid_kwargs)
    e = np.empty((len(u), 2), dtype=np.float64)
    e[:, 0] = u[:, 0]
    e[:, 1] = strategy.rosenblatt_e2(copula, u, fit_result)
    return clip_rosenblatt_output(e)


def _bootstrap_fit_kwargs(fit_result, fit_kwargs):
    """Restore fitted defaults while retaining explicit refit settings."""
    out = dict(fit_kwargs)
    method = fit_result.method.upper()
    if method == 'GAS' and out.get('score_eps') is None:
        from pyscarcopula._types import DEFAULT_CONFIG

        config = out.get('config')
        out['score_eps'] = (
            config.gas_score_eps if config is not None else
            getattr(fit_result, 'score_eps', DEFAULT_CONFIG.gas_score_eps))
    if 'alpha0' in out or 'gamma0' in out:
        return out

    if method == 'MLE' and hasattr(fit_result, 'copula_param'):
        out['alpha0'] = np.array([fit_result.copula_param], dtype=np.float64)
    else:
        params = getattr(fit_result, 'params', None)
        if params is not None:
            key = 'gamma0' if method == 'GAS' else 'alpha0'
            out[key] = np.asarray(params.values, dtype=np.float64)
    return out


def _fit_result_diagnostics(result):
    """Return normalized diagnostics for one bootstrap refit."""
    log_likelihood = getattr(result, 'log_likelihood', np.nan)
    if callable(log_likelihood):
        log_likelihood = getattr(result, '_log_likelihood', np.nan)
    row = {
        'bootstrap_fit_method': getattr(result, 'method', ''),
        'bootstrap_fit_log_likelihood': float(log_likelihood),
        'bootstrap_fit_success': bool(getattr(result, 'success', False)),
        'bootstrap_fit_nfev': int(getattr(result, 'nfev', 0)),
        'bootstrap_fit_message': str(getattr(result, 'message', '')),
    }
    copula_param = getattr(result, 'copula_param', None)
    if copula_param is not None:
        row['bootstrap_param_theta'] = float(copula_param)

    params = getattr(result, 'params', None)
    if params is not None:
        values = np.asarray(params.values, dtype=np.float64)
        row['bootstrap_params_json'] = {
            name: float(value)
            for name, value in zip(params.names, values)
        }
        for name, value in zip(params.names, values):
            row[f'bootstrap_param_{name}'] = float(value)
    return row


def _bootstrap_strategy(fit_result, config, **constructor_kwargs):
    if (
            fit_result.method.upper() == 'MLE'
            and not hasattr(fit_result, 'copula_param')):
        return None

    from pyscarcopula.strategy._base import get_strategy_for_result

    return get_strategy_for_result(
        fit_result, config=config, **constructor_kwargs)


def _bootstrap_capture_none(copula, fit_result):
    return None


def _bootstrap_prepare_bivariate(
        copula_class, constructor_kwargs, fit_result, fitted_snapshot):
    return create_worker_model(copula_class, constructor_kwargs)


def _bootstrap_simulate_bivariate(
        copula, u, fit_result, rng, K, grid_range, n_threads, config):
    strategy = _bootstrap_strategy(fit_result, config)
    if strategy is None:
        return copula.sample_at_parameter(len(u), rng=rng)
    return strategy.sample(copula, u, fit_result, len(u), rng=rng)


def _bootstrap_refit_bivariate(
        copula_class, constructor_kwargs, u_boot, fit_result, fit_kwargs,
        K, grid_range, n_threads, config):
    copula = create_worker_model(copula_class, constructor_kwargs)
    from pyscarcopula.strategy._base import partition_strategy_fit_kwargs

    strategy_kwargs, fit_kwargs = partition_strategy_fit_kwargs(
        fit_result.method, fit_kwargs)
    strategy = _bootstrap_strategy(fit_result, config, **strategy_kwargs)
    if strategy is None:
        result = copula.fit(
            u_boot, method='mle', to_pobs=False, **fit_kwargs)
    else:
        result = strategy.fit(
            copula,
            u_boot,
            **_bootstrap_fit_kwargs(fit_result, fit_kwargs),
        )
    return copula, result


def _bootstrap_statistic_bivariate(
        copula, u_boot, fit_result, K, grid_range):
    e_boot = _bivariate_rosenblatt_from_result(
        copula, u_boot, fit_result, K, grid_range)
    return float(cvm_test(e_boot).statistic)


def _bootstrap_prepare_gaussian(
        copula_class, constructor_kwargs, fit_result, fitted_snapshot):
    copula = create_worker_model(copula_class, constructor_kwargs)
    if getattr(copula, 'corr_mode', 'dense') == 'factor':
        loadings = fit_result.model_parameters.get('factor_loadings')
        if loadings is None:
            raise ValueError(
                "factor Gaussian fit result does not contain factor_loadings")
        copula._set_factor_loadings(
            np.asarray(loadings, dtype=np.float64),
            diagnostics={'source': 'bootstrap_fitted_result'},
        )
    else:
        correlation = getattr(fit_result, 'correlation_matrix', None)
        if correlation is None:
            raise ValueError(
                "Gaussian fit result does not contain a correlation matrix")
        correlation = np.asarray(correlation, dtype=np.float64)
        copula._set_dimension(correlation.shape[0], allow_change=True)
        copula.corr = correlation.copy()
    copula.fit_result = fit_result
    return copula


def _bootstrap_simulate_static_multivariate(
        copula, u, fit_result, rng, K, grid_range, n_threads, config):
    return copula.sample(len(u), rng=rng, n_threads=n_threads)


def _bootstrap_refit_static_multivariate(
        copula_class, constructor_kwargs, u_boot, fit_result, fit_kwargs,
        K, grid_range, n_threads, config):
    copula = create_worker_model(copula_class, constructor_kwargs)
    result = copula.fit(
        u_boot,
        method='mle',
        to_pobs=False,
        config=config,
        **fit_kwargs,
    )
    return copula, result


def _bootstrap_statistic_gaussian(
        copula, u_boot, fit_result, K, grid_range):
    if getattr(copula, 'corr_mode', 'dense') == 'factor':
        e_boot = factor_gaussian_rosenblatt_transform(
            copula.correlation_operator_, u_boot)
    else:
        correlation = getattr(fit_result, 'correlation_matrix', None)
        if correlation is None:
            correlation = copula.corr
        e_boot = gaussian_rosenblatt_transform(correlation, u_boot)
    return float(cvm_test(e_boot).statistic)


def _bootstrap_prepare_student(
        copula_class, constructor_kwargs, fit_result, fitted_snapshot):
    copula = create_worker_model(copula_class, constructor_kwargs)
    df = getattr(fit_result, 'copula_param', None)
    if df is None:
        raise ValueError("Student fit result does not contain df")
    if getattr(copula, 'corr_mode', 'fixed') == 'factor':
        loadings = getattr(
            fit_result, 'model_parameters', {}).get('factor_loadings')
        if loadings is None:
            raise ValueError(
                "factor Student fit result does not contain factor_loadings")
        copula._set_factor_loadings(
            np.asarray(loadings, dtype=np.float64),
            diagnostics={'source': 'bootstrap_fitted_result'},
        )
        copula.df = float(df)
        copula.fit_result = fit_result
        return copula

    correlation = getattr(fit_result, 'correlation_matrix', None)
    if correlation is None:
        raise ValueError(
            "Student fit result does not contain correlation and df")
    correlation = np.asarray(correlation, dtype=np.float64)
    copula._set_dimension(correlation.shape[0], allow_change=True)
    copula.shape = correlation.copy()
    copula.df = float(df)
    copula.fit_result = fit_result
    return copula


def _bootstrap_simulate_fitted_model(
        copula, u, fit_result, rng, K, grid_range, n_threads, config):
    """Simulate one bootstrap sample from a fitted static model."""
    return copula.sample(len(u), rng=rng)


def _bootstrap_statistic_student(
        copula, u_boot, fit_result, K, grid_range):
    df = getattr(fit_result, 'copula_param', None)
    if getattr(copula, 'corr_mode', 'fixed') == 'factor':
        e_boot = factor_student_rosenblatt_transform(
            copula.correlation_operator_, float(df), u_boot)
        return float(cvm_test(e_boot).statistic)
    correlation = getattr(fit_result, 'correlation_matrix', None)
    e_boot = student_rosenblatt_transform(
        correlation, float(df), u_boot)
    return float(cvm_test(e_boot).statistic)


def _bootstrap_capture_rvine(copula, fit_result):
    """Capture fitted Python state without persisting transient native cache."""
    return deepcopy(copula)


def _bootstrap_prepare_rvine(
        copula_class, constructor_kwargs, fit_result, fitted_snapshot):
    """Clone a fitted R-vine snapshot for one bootstrap worker."""
    return deepcopy(fitted_snapshot)


def _bootstrap_refit_rvine(
        copula_class, constructor_kwargs, u_boot, fit_result, fit_kwargs,
        K, grid_range, n_threads, config):
    """Refit an R-vine worker model to one bootstrap sample."""
    copula = create_worker_model(copula_class, constructor_kwargs)
    method = str(getattr(fit_result, 'method', 'MLE')).lower()
    fitted = copula.fit(
        u_boot,
        method=method,
        to_pobs=False,
        config=config,
        **fit_kwargs,
    )
    return fitted, fitted.fit_result


def _bootstrap_statistic_rvine(
        copula, u_boot, fit_result, K, grid_range):
    """Compute the CvM statistic from static R-vine residuals."""
    residuals = rvine_rosenblatt_transform(
        copula,
        u_boot,
        K=K,
        grid_range=grid_range,
        vine_type=copula.vine_type,
    )
    return float(cvm_test(residuals).statistic)


def _bootstrap_prepare_equicorr(
        copula_class, constructor_kwargs, fit_result, fitted_snapshot):
    copula = create_worker_model(copula_class, constructor_kwargs)
    copula.fit_result = fit_result
    return copula


def _bootstrap_simulate_dynamic_multivariate(
        copula, u, fit_result, rng, K, grid_range, n_threads, config):
    return copula.sample(len(u), u=u, rng=rng, n_threads=n_threads)


def _bootstrap_refit_dynamic(
        copula_class, constructor_kwargs, u_boot, fit_result, fit_kwargs,
        K, grid_range, n_threads, config):
    copula = create_worker_model(copula_class, constructor_kwargs)
    method = fit_result.method.upper()
    if method == 'MLE':
        result = copula.fit(
            u_boot,
            method='mle',
            to_pobs=False,
            config=config,
            **fit_kwargs,
        )
        return copula, result

    if hasattr(copula, '_ensure_corr_initialized'):
        copula._ensure_corr_initialized(u_boot)

    from pyscarcopula.strategy._base import (
        get_strategy_for_result,
        partition_strategy_fit_kwargs,
    )

    strategy_kwargs, fit_kwargs = partition_strategy_fit_kwargs(
        fit_result.method, fit_kwargs)
    strategy = get_strategy_for_result(
        fit_result,
        config=config,
        **strategy_kwargs,
    )
    result = strategy.fit(
        copula,
        u_boot,
        **_bootstrap_fit_kwargs(fit_result, fit_kwargs),
    )
    copula.fit_result = result
    copula._last_u = u_boot
    return copula, result


def _bootstrap_statistic_equicorr(
        copula, u_boot, fit_result, K, grid_range):
    e_boot = equicorr_rosenblatt_transform(
        copula, u_boot, fit_result, K, grid_range)
    return float(cvm_test(e_boot).statistic)


def _bootstrap_capture_stochastic_student(copula, fit_result):
    corr_mode = copula.corr_mode
    model_parameters = getattr(fit_result, 'model_parameters', {})
    owns_fit_result = getattr(copula, 'fit_result', None) is fit_result
    if corr_mode == 'factor':
        loadings = model_parameters.get('factor_loadings')
        if loadings is None and (
                owns_fit_result
                or getattr(copula, '_constructor_factor_loadings', None)
                is not None):
            loadings = copula.factor_loadings_
        if loadings is None:
            raise ValueError(
                "fitted factor loadings are required for bootstrap GoF")
        return {
            'corr_mode': corr_mode,
            'factor_loadings': np.asarray(
                loadings, dtype=np.float64).copy(),
        }

    correlation = getattr(fit_result, 'correlation_matrix', None)
    if correlation is None:
        correlation = model_parameters.get('correlation_matrix')
    if correlation is None and (
            corr_mode == 'fixed' or owns_fit_result):
        correlation = copula.R
    if correlation is None:
        raise ValueError(
            "fitted correlation state is required for bootstrap GoF; "
            "pass the fitted StochasticStudentCopula instance")
    return {
        'corr_mode': corr_mode,
        'correlation_matrix': np.asarray(
            correlation, dtype=np.float64).copy(),
    }


def _bootstrap_prepare_stochastic_student(
        copula_class, constructor_kwargs, fit_result, fitted_snapshot):
    copula = create_worker_model(copula_class, constructor_kwargs)
    if fitted_snapshot['corr_mode'] == 'factor':
        copula._set_factor_loadings(
            fitted_snapshot['factor_loadings'],
            diagnostics={'source': 'bootstrap_fitted_snapshot'},
        )
    else:
        copula._set_R(
            fitted_snapshot['correlation_matrix'],
            source='bootstrap_fitted_snapshot',
        )
    copula.fit_result = fit_result
    return copula


def _bootstrap_statistic_stochastic_student(
        copula, u_boot, fit_result, K, grid_range):
    e_boot = stochastic_student_rosenblatt_transform(
        copula, u_boot, fit_result, K, grid_range)
    return float(cvm_test(e_boot).statistic)


@dataclass(frozen=True)
class _BootstrapAdapter:
    capture: Callable
    prepare: Callable
    simulate: Callable
    refit: Callable
    statistic: Callable


_BOOTSTRAP_ADAPTERS = {
    'bivariate': _BootstrapAdapter(
        capture=_bootstrap_capture_none,
        prepare=_bootstrap_prepare_bivariate,
        simulate=_bootstrap_simulate_bivariate,
        refit=_bootstrap_refit_bivariate,
        statistic=_bootstrap_statistic_bivariate,
    ),
    'gaussian': _BootstrapAdapter(
        capture=_bootstrap_capture_none,
        prepare=_bootstrap_prepare_gaussian,
        simulate=_bootstrap_simulate_static_multivariate,
        refit=_bootstrap_refit_static_multivariate,
        statistic=_bootstrap_statistic_gaussian,
    ),
    'student': _BootstrapAdapter(
        capture=_bootstrap_capture_none,
        prepare=_bootstrap_prepare_student,
        simulate=_bootstrap_simulate_static_multivariate,
        refit=_bootstrap_refit_static_multivariate,
        statistic=_bootstrap_statistic_student,
    ),
    'rvine': _BootstrapAdapter(
        capture=_bootstrap_capture_rvine,
        prepare=_bootstrap_prepare_rvine,
        simulate=_bootstrap_simulate_fitted_model,
        refit=_bootstrap_refit_rvine,
        statistic=_bootstrap_statistic_rvine,
    ),
    'equicorr': _BootstrapAdapter(
        capture=_bootstrap_capture_none,
        prepare=_bootstrap_prepare_equicorr,
        simulate=_bootstrap_simulate_dynamic_multivariate,
        refit=_bootstrap_refit_dynamic,
        statistic=_bootstrap_statistic_equicorr,
    ),
    'stochastic_student': _BootstrapAdapter(
        capture=_bootstrap_capture_stochastic_student,
        prepare=_bootstrap_prepare_stochastic_student,
        simulate=_bootstrap_simulate_dynamic_multivariate,
        refit=_bootstrap_refit_dynamic,
        statistic=_bootstrap_statistic_stochastic_student,
    ),
}


def _bootstrap_gof_worker(task):
    """Run one independently seeded parametric-bootstrap replication."""
    (
        iteration,
        adapter_name,
        copula_class,
        constructor_kwargs,
        u,
        fit_result,
        fitted_snapshot,
        observed_statistic,
        K,
        grid_range,
        bootstrap_refit,
        fit_kwargs,
        seed_sequence,
        n_threads,
    ) = task

    try:
        adapter = _BOOTSTRAP_ADAPTERS[adapter_name]
        iter_start = time.perf_counter()
        rng = np.random.default_rng(seed_sequence)
        config = fit_kwargs['config']
        refit_kwargs = dict(fit_kwargs)
        refit_kwargs.pop('config')
        copula = adapter.prepare(
            copula_class, constructor_kwargs, fit_result, fitted_snapshot)
        u_boot = adapter.simulate(
            copula,
            u,
            fit_result,
            rng,
            K,
            grid_range,
            n_threads,
            config,
        )

        fit_start = time.perf_counter()
        if bootstrap_refit:
            copula, boot_result = adapter.refit(
                copula_class,
                constructor_kwargs,
                u_boot,
                fit_result,
                refit_kwargs,
                K,
                grid_range,
                n_threads,
                config,
            )
        else:
            boot_result = fit_result
        fit_elapsed = time.perf_counter() - fit_start

        stat_start = time.perf_counter()
        statistic = adapter.statistic(
            copula, u_boot, boot_result, K, grid_range)
        stat_elapsed = time.perf_counter() - stat_start

        row = {
            'bootstrap_iteration': int(iteration + 1),
            'bootstrap_statistic': statistic,
            'bootstrap_exceeds_observed': bool(
                statistic >= float(observed_statistic)),
            'bootstrap_fit_time_sec': float(fit_elapsed),
            'bootstrap_stat_time_sec': float(stat_elapsed),
            'bootstrap_total_time_sec': float(
                time.perf_counter() - iter_start),
            'bootstrap_refit': bool(bootstrap_refit),
        }
        row.update(_fit_result_diagnostics(boot_result))
        return statistic, row
    except Exception as exc:
        raise RuntimeError(
            f"bootstrap iteration {iteration + 1} failed") from exc


def _bootstrap_gof(
        adapter_name, copula, u, fit_result, observed_statistic,
        K=300, grid_range=5.0, n_bootstrap=199,
        bootstrap_refit=True, bootstrap_fit_kwargs=None, rng=None,
        n_jobs=1):
    """Run calibrated GoF through a model-specific bootstrap adapter."""
    n_bootstrap = validate_positive_int(n_bootstrap, "n_bootstrap")
    validate_float64_allocation(
        (n_bootstrap,), name="bootstrap_statistics")

    fit_kwargs = (
        {} if bootstrap_fit_kwargs is None
        else dict(bootstrap_fit_kwargs)
    )
    if 'to_pobs' in fit_kwargs:
        raise TypeError(
            "bootstrap_fit_kwargs cannot override to_pobs; "
            "bootstrap samples are already pseudo-observations")
    validate_model_fit_kwargs(copula, fit_result.method, fit_kwargs)
    if fit_result.method.upper() == 'GAS':
        # Resolve fitted defaults before with_n_threads adds a default config;
        # only a config supplied by the caller overrides the fitted score step.
        fit_kwargs = _bootstrap_fit_kwargs(fit_result, fit_kwargs)
    n_threads, parallel_diagnostics = resolve_parallelism(
        n_jobs, n_bootstrap, None, (fit_kwargs,))
    fit_kwargs = with_n_threads(fit_kwargs, n_threads)
    requested_jobs = parallel_diagnostics['n_jobs_requested']
    resolved_jobs = parallel_diagnostics['n_jobs']
    seed_sequences = spawn_seed_sequences(rng, n_bootstrap)
    copula_class, constructor_kwargs = get_copula_constructor(copula)
    fitted_snapshot = _BOOTSTRAP_ADAPTERS[adapter_name].capture(
        copula, fit_result)
    tasks = [
        (
            iteration,
            adapter_name,
            copula_class,
            constructor_kwargs,
            u,
            fit_result,
            fitted_snapshot,
            observed_statistic,
            K,
            grid_range,
            bootstrap_refit,
            fit_kwargs,
            seed_sequences[iteration],
            n_threads,
        )
        for iteration in range(n_bootstrap)
    ]

    if resolved_jobs == 1:
        worker_results = [
            _bootstrap_gof_worker(task) for task in tasks
        ]
        backend = 'sequential'
    else:
        from joblib import Parallel, delayed, parallel_config

        with parallel_config(
                backend='loky', inner_max_num_threads=n_threads):
            worker_results = Parallel(n_jobs=resolved_jobs)(
                delayed(_bootstrap_gof_worker)(task)
                for task in tasks
            )
        backend = 'loky'

    boot_stats = np.asarray(
        [item[0] for item in worker_results], dtype=np.float64)
    diagnostics = tuple(item[1] for item in worker_results)

    pvalue = (
        1.0 + np.sum(boot_stats >= float(observed_statistic))
    ) / (len(boot_stats) + 1.0)
    return BootstrapGoFResult(
        statistic=float(observed_statistic),
        pvalue=float(pvalue),
        bootstrap_statistics=boot_stats,
        n_bootstrap=len(boot_stats),
        bootstrap_diagnostics=diagnostics,
        n_jobs_requested=requested_jobs,
        n_jobs=resolved_jobs,
        n_threads=n_threads,
        backend=backend,
    )


def _bootstrap_gof_bivariate(copula, u, fit_result, observed_statistic,
                             K=300, grid_range=5.0, n_bootstrap=199,
                             bootstrap_refit=True,
                             bootstrap_fit_kwargs=None, rng=None,
                             n_jobs=1):
    """Parametric bootstrap calibration for bivariate GoF."""
    return _bootstrap_gof(
        'bivariate',
        copula,
        u,
        fit_result,
        observed_statistic,
        K=K,
        grid_range=grid_range,
        n_bootstrap=n_bootstrap,
        bootstrap_refit=bootstrap_refit,
        bootstrap_fit_kwargs=bootstrap_fit_kwargs,
        rng=rng,
        n_jobs=n_jobs,
    )


def _gof_static_multivariate(
        adapter_name, copula, data, to_pobs, fit_result,
        n_bootstrap, bootstrap_refit, bootstrap_fit_kwargs, rng, n_jobs):
    """GoF with optional bootstrap for static multivariate copulas."""
    n_bootstrap = validate_positive_int(n_bootstrap, "n_bootstrap")
    validate_float64_allocation(
        (n_bootstrap,), name="bootstrap_statistics")
    u = _prepare_gof_data(
        data, expected_dim=copula.dimension, to_pobs=to_pobs)
    fr = (
        fit_result
        if fit_result is not None
        else getattr(copula, 'fit_result', None)
    )
    if fr is None:
        raise ValueError(
            "No fit_result provided and copula has no fit_result. "
            "Call copula.fit() first or pass fit_result=.")

    copula_class, constructor_kwargs = get_copula_constructor(copula)
    adapter = _BOOTSTRAP_ADAPTERS[adapter_name]
    fitted_copula = adapter.prepare(
        copula_class, constructor_kwargs, fr, adapter.capture(copula, fr))
    observed_statistic = adapter.statistic(
        fitted_copula, u, fr, 300, 5.0)

    return _bootstrap_gof(
        adapter_name,
        copula,
        u,
        fr,
        observed_statistic,
        n_bootstrap=n_bootstrap,
        bootstrap_refit=bootstrap_refit,
        bootstrap_fit_kwargs=bootstrap_fit_kwargs,
        rng=rng,
        n_jobs=n_jobs,
    )


def _gof_dynamic_multivariate(
        adapter_name, copula, data, to_pobs, K, grid_range, fit_result,
        n_bootstrap, bootstrap_refit, bootstrap_fit_kwargs, rng, n_jobs):
    """GoF bootstrap for dynamic multivariate copulas."""
    n_bootstrap = validate_positive_int(n_bootstrap, "n_bootstrap")
    validate_float64_allocation(
        (n_bootstrap,), name="bootstrap_statistics")
    u = _prepare_gof_data(
        data, expected_dim=copula.dimension, to_pobs=to_pobs)
    fr = (
        fit_result
        if fit_result is not None
        else getattr(copula, 'fit_result', None)
    )
    if fr is None:
        raise ValueError(
            "No fit_result provided and copula has no fit_result. "
            "Call copula.fit() first or pass fit_result=.")

    copula_class, constructor_kwargs = get_copula_constructor(copula)
    adapter = _BOOTSTRAP_ADAPTERS[adapter_name]
    fitted_copula = adapter.prepare(
        copula_class, constructor_kwargs, fr, adapter.capture(copula, fr))
    observed_statistic = adapter.statistic(
        fitted_copula, u, fr, K, grid_range)
    return _bootstrap_gof(
        adapter_name,
        copula,
        u,
        fr,
        observed_statistic,
        K=K,
        grid_range=grid_range,
        n_bootstrap=n_bootstrap,
        bootstrap_refit=bootstrap_refit,
        bootstrap_fit_kwargs=bootstrap_fit_kwargs,
        rng=rng,
        n_jobs=n_jobs,
    )


# ══════════════════════════════════════════════════════════════════
# Vine Rosenblatt transform
# ══════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════
# Vine gof_test
# ══════════════════════════════════════════════════════════════════

def _prepare_rvine_rosenblatt_observations(vine, u):
    """Validate pseudo-observations before applying endpoint safeguards."""
    observations = as_pseudo_observation_array(u, name="u")
    if observations.ndim != 2:
        raise ValueError(f"u must have shape (T, {vine.d})")
    _, dimension = observations.shape
    if dimension != vine.d:
        raise ValueError(
            f"u has d={dimension}, but fitted vine has d={vine.d}")
    return clip_pseudo_observations(observations)


def _rvine_rosenblatt_transform_native(
        module, vine, u, *, K=300, grid_range=5.0):
    """Run supported static or dynamic edges through the native traversal."""
    from pyscarcopula._native import vine as _cpp_rvine

    active_keys = _cpp_rvine.density_active_keys(
        vine._trees, vine._edge_map)
    if not _cpp_rvine.native_edges_supported(
            vine.pair_copulas, active_keys):
        raise NativeUnsupported(
            "native R-vine Rosenblatt requires exact registered built-in "
            "edge copulas"
        )
    layout = _cpp_rvine.static_rosenblatt_parameter_layout(
        vine.pair_copulas, active_keys)
    if layout is None:
        return _cpp_rvine.rosenblatt(
            module,
            vine.pair_copulas,
            vine.d,
            vine._trees,
            vine._edge_map,
            vine.matrix,
            u,
            active_keys=active_keys,
            dynamic_strategy_kwargs={"K": K, "grid_range": grid_range},
        )
    parameter_paths, parameter_sources = layout
    observations = _cpp_rvine._rvine_observations(
        u, vine.d, "Rosenblatt")
    residual_node_keys = _cpp_rvine.rosenblatt_residual_node_keys(
        vine.matrix)
    context, parameters = vine._native_density_context(
        module,
        vine.pair_copulas,
        vine._edge_map,
        parameter_paths,
        len(observations),
        active_keys=active_keys,
        normalized_paths=parameter_paths,
        parameter_sources=parameter_sources,
        residual_node_keys=residual_node_keys,
        cache_slot='rosenblatt',
    )
    if context is None:
        raise NativeUnsupported(
            "native R-vine Rosenblatt could not compile the edge context")
    residuals = _cpp_rvine.rosenblatt(
        module,
        vine.pair_copulas,
        vine.d,
        vine._trees,
        vine._edge_map,
        vine.matrix,
        observations,
        active_keys=context['active_keys'],
        parameter_paths=parameter_paths,
        parameter_sources=context['parameter_sources'],
        residual_node_keys=residual_node_keys,
        native_plan=context['plan'],
        native_edges=context['edges'],
        parameter_pack=parameters,
    )
    return clip_rosenblatt_output(residuals)


def rvine_rosenblatt_transform(
        vine, u, K=300, grid_range=5.0, *, vine_type=None):
    """Apply the mandatory native R-vine Rosenblatt traversal."""
    if vine_type is None:
        vine_type = getattr(vine, "vine_type", "rvine")
    if vine_type not in {"cvine", "dvine", "rvine"}:
        raise ValueError(
            "vine_type must be 'cvine', 'dvine' or 'rvine', "
            f"got {vine_type!r}")
    if getattr(vine, 'matrix', None) is None:
        raise ValueError("Fit the vine first")
    observations = _prepare_rvine_rosenblatt_observations(vine, u)
    from pyscarcopula._native import vine as _cpp_rvine
    active_keys = _cpp_rvine.density_active_keys(
        vine._trees, vine._edge_map)
    if not _cpp_rvine.native_edges_supported(
            vine.pair_copulas, active_keys):
        raise NativeUnsupported(
            "native R-vine Rosenblatt requires exact registered built-in "
            "edge copulas"
        )
    module = _cpp_extension.load()
    if not hasattr(module, "rvine_rosenblatt_transform"):
        raise NativeUnsupported(
            "native R-vine Rosenblatt requires "
            "_native._scar_cpp.rvine_rosenblatt_transform")
    return _rvine_rosenblatt_transform_native(
        module, vine, observations, K=K, grid_range=grid_range)


def rvine_gof_test(
        vine, data, to_pobs=True, K=500, grid_range=7.0, *,
        vine_type=None):
    """
    Goodness-of-fit test for a fitted R-vine copula.

    Parameters
    ----------
    vine : VineCopula (fitted)
    data : (T, d)
    to_pobs : bool
    K : int
    grid_range : float
    vine_type : {'cvine', 'dvine', 'rvine'} or None
        Structural mode forwarded by :func:`gof_test`. ``None`` derives it
        from the fitted model.

    Returns
    -------
    CramérVonMisesResult
    """
    u = _prepare_gof_data(
        data, expected_dim=getattr(vine, "d", None), to_pobs=to_pobs)

    if getattr(vine, 'matrix', None) is None:
        raise ValueError("Fit the vine first")

    if vine_type is None:
        vine_type = getattr(vine, "vine_type", "rvine")
    e = rvine_rosenblatt_transform(
        vine,
        u,
        K=K,
        grid_range=grid_range,
        vine_type=vine_type,
    )
    return cvm_test(e)


# ══════════════════════════════════════════════════════════════════
# Gaussian copula Rosenblatt
# ══════════════════════════════════════════════════════════════════

def gaussian_rosenblatt_transform(R, u):
    """
    Rosenblatt transform for d-dimensional Gaussian copula.

    x = Phi^{-1}(u), x ~ N(0, R).
    Conditional: x_i | x_{1:i-1} ~ N(mu_{i|1..i-1}, sigma^2_{i|1..i-1})
    e_i = Phi((x_i - mu_{i|1..i-1}) / sigma_{i|1..i-1})

    Uses Cholesky: R = L L^T, then z = L^{-1} x has independent
    components, and e_i = Phi(z_i).

    Parameters
    ----------
    R : (d, d) correlation matrix
    u : (T, d) pseudo-observations

    Returns
    -------
    e : (T, d)
    """
    from pyscarcopula._native import multivariate as multivariate_native
    return multivariate_native.gaussian_rosenblatt(R, u)


def factor_gaussian_rosenblatt_transform(correlation, u):
    """Rosenblatt transform for a Gaussian factor correlation.

    The sequential conditional distribution is evaluated with the
    rank-dimensional posterior of the latent factor. Storage is
    ``O(T*k + k*k)`` and no dense correlation or Cholesky factor is formed.
    """
    from pyscarcopula._native import multivariate as multivariate_native
    return multivariate_native.factor_gaussian_rosenblatt(correlation, u)


def gaussian_gof_test(copula, data, to_pobs=True, *, fit_result=None):
    """
    Goodness-of-fit test for a fitted GaussianCopula.

    Parameters
    ----------
    copula : GaussianCopula (fitted, has .corr)
    data : (T, d)
    to_pobs : bool
    fit_result : FitResult or None
        Explicit fitted correlation state, taking precedence over the model.

    Returns
    -------
    CramérVonMisesResult
    """
    u = _prepare_gof_data(
        data, expected_dim=copula.dimension, to_pobs=to_pobs)

    if fit_result is not None:
        from pyscarcopula.strategy.multivariate_mle import (
            sampling_model_from_result,
        )
        copula = sampling_model_from_result(copula, fit_result)

    if getattr(copula, "corr_mode", "dense") == "factor":
        try:
            correlation = copula.correlation_operator_
        except (AttributeError, ValueError) as exc:
            raise ValueError("Fit the copula first") from exc
        e = factor_gaussian_rosenblatt_transform(correlation, u)
    else:
        if copula.corr is None:
            raise ValueError("Fit the copula first")
        e = gaussian_rosenblatt_transform(copula.corr, u)
    return cvm_test(e)


# ══════════════════════════════════════════════════════════════════
# Student-t copula Rosenblatt
# ══════════════════════════════════════════════════════════════════

def student_rosenblatt_transform(R, df, u):
    """Evaluate the dense Student Rosenblatt transform natively."""
    from pyscarcopula._native import multivariate as multivariate_native
    return multivariate_native.dense_student_rosenblatt(R, df, u)


def factor_student_rosenblatt_transform(correlation, df, u):
    """Rosenblatt transform for a Student factor correlation.

    Sequential conditioning uses the rank-dimensional Woodbury posterior.
    Storage is ``O(T*k + k*k)`` and no dense correlation matrix is formed.
    ``df`` may be scalar or contain one value per observation.
    """
    from pyscarcopula._native import multivariate as multivariate_native
    return multivariate_native.factor_student_rosenblatt(
        correlation, df, u)


def student_gof_test(copula, data, to_pobs=True, *, fit_result=None):
    """
    Goodness-of-fit test for a fitted StudentCopula.

    Parameters
    ----------
    copula : StudentCopula (fitted, has .shape and .df)
    data : (T, d)
    to_pobs : bool
    fit_result : FitResult or None
        Explicit fitted correlation and degrees of freedom, taking precedence
        over the model.

    Returns
    -------
    CramérVonMisesResult
    """
    u = _prepare_gof_data(
        data, expected_dim=copula.dimension, to_pobs=to_pobs)

    if fit_result is not None:
        from pyscarcopula.strategy.multivariate_mle import (
            sampling_model_from_result,
        )
        copula = sampling_model_from_result(copula, fit_result)

    if getattr(copula, "corr_mode", "fixed") == "factor":
        try:
            correlation = copula.correlation_operator_
        except (AttributeError, ValueError) as exc:
            raise ValueError("Fit the copula first") from exc
        if copula.df is None:
            raise ValueError("Fit the copula first")
        e = factor_student_rosenblatt_transform(
            correlation, copula.df, u)
    else:
        if copula.shape is None or copula.df is None:
            raise ValueError("Fit the copula first")
        e = student_rosenblatt_transform(copula.shape, copula.df, u)
    return cvm_test(e)


# ══════════════════════════════════════════════════════════════════
# Equicorrelation Gaussian copula GoF
# ══════════════════════════════════════════════════════════════════

def _gas_parameter_path(copula, u, fit_result):
    """Return deterministic GAS parameter path r_t for a fitted model."""
    from pyscarcopula.numerical.gas_filter import gas_filter

    omega, gamma, beta = fit_result.params.values
    scaling = getattr(fit_result, 'scaling', 'unit')
    score_eps = getattr(fit_result, 'score_eps', 1e-4)
    _, r_path, _ = gas_filter(
        omega, gamma, beta, u, copula, scaling=scaling,
        score_eps=score_eps)
    return np.asarray(r_path, dtype=np.float64)


def _native_grid_config_from_result(fit_result, K, grid_range):
    """Build the native grid config with the preserved OU-grid defaults."""
    from pyscarcopula.numerical._scar_ou_config import AutoTMConfig
    from pyscarcopula.strategy._base import get_ou_strategy_for_result

    strategy = get_ou_strategy_for_result(
        fit_result, K=K, grid_range=grid_range)
    return AutoTMConfig(
        K=strategy.K,
        grid_range=strategy.grid_range,
        grid_method=strategy.grid_method,
        adaptive=strategy.adaptive,
        pts_per_sigma=strategy.pts_per_sigma,
        transition_method=_grid_transition_method(strategy.transition_method),
        small_kdt=strategy.auto_small_kdt,
        max_K=strategy.max_K,
        r_gh=strategy.r_gh,
        gh_order=strategy.gh_order,
    )


def equicorr_rosenblatt_transform(copula, u, fit_result, K=300, grid_range=5.0):
    """
    Rosenblatt transform for EquicorrGaussianCopula.

    MLE: constant rho, Cholesky-based sequential conditioning.
    SCAR: mixture over predictive rho(t) distribution from TM forward pass.

    For equicorrelation R = (1-rho)*I + rho*11':
        E[x_i | x_0,...,x_{i-1}] = rho * sum(x_0,...,x_{i-1}) / (1 + (i-1)*rho)
        Var(x_i | x_0,...,x_{i-1}) = 1 - i*rho^2 / (1 + (i-1)*rho)

    Parameters
    ----------
    copula : EquicorrGaussianCopula
    u : (T, d)
    fit_result : FitResult
        Estimation result. Required.
    K, grid_range : TM grid params (SCAR only)

    Returns
    -------
    e : (T, d) — should be iid U[0,1]^d under correct model
    """
    T, d = u.shape
    method = fit_result.method.upper()
    if method not in ('MLE', 'GAS'):
        from pyscarcopula._native import scar_ou as _cpp_scar_ou

        config = _native_grid_config_from_result(fit_result, K, grid_range)
        kappa, mu, nu = fit_result.params.values
        return _cpp_scar_ou.gaussian_rosenblatt(
            kappa,
            mu,
            nu,
            u,
            copula,
            config,
        )

    if method == 'MLE':
        rho = fit_result.copula_param
    elif method == 'GAS':
        rho = _gas_parameter_path(copula, u, fit_result)
    else:
        raise AssertionError(f"unsupported equicorrelation method: {method}")

    from pyscarcopula._native import multivariate as multivariate_native
    return multivariate_native.equicorr_gaussian_rosenblatt(rho, u)


def equicorr_gof_test(copula, data, to_pobs=True,
                      K=300, grid_range=5.0, fit_result=None):
    """
    Goodness-of-fit test for EquicorrGaussianCopula.

    Parameters
    ----------
    copula : EquicorrGaussianCopula
    data : (T, d)
    to_pobs : bool
    K : int
    grid_range : float
    fit_result : FitResult or None
        If None, uses copula.fit_result (set by copula.fit()).

    Returns
    -------
    CramérVonMisesResult
    """
    u = _prepare_gof_data(
        data, expected_dim=copula.dimension, to_pobs=to_pobs)

    fr = fit_result if fit_result is not None else getattr(copula, 'fit_result', None)
    if fr is None:
        raise ValueError("No fit_result provided and copula has no fit_result. "
                         "Call copula.fit() first or pass fit_result=.")

    e = equicorr_rosenblatt_transform(copula, u, fr, K, grid_range)
    return cvm_test(e)


# ══════════════════════════════════════════════════════════════════
# Stochastic Student-t copula GoF
# ══════════════════════════════════════════════════════════════════

def stochastic_student_rosenblatt_transform(copula, u, fit_result,
                                             K=300, grid_range=5.0):
    """
    Rosenblatt transform for StochasticStudentCopula.

    MLE: constant df, standard sequential conditioning.
    SCAR: mixture over predictive df(t) distribution from TM forward pass.

    For Student-t copula with shape matrix R and df:
        x = t_df^{-1}(u)
        x_i | x_0,...,x_{i-1} ~ t_{df+i}(mu_i, sigma^2_i * scale)
    where:
        mu_i = R_{i,0:i} R_{0:i,0:i}^{-1} x_{0:i}
        sigma^2_i = R_{ii} - R_{i,0:i} R_{0:i,0:i}^{-1} R_{0:i,i}
        scale = (df + x_{0:i}^T R_{0:i,0:i}^{-1} x_{0:i}) / (df + i)

    Parameters
    ----------
    copula : StochasticStudentCopula
    u : (T, d) pseudo-observations
    fit_result : FitResult
    K, grid_range : TM grid params (SCAR only)

    Returns
    -------
    e : (T, d) — should be iid U[0,1]^d under correct model
    """
    T, d = u.shape
    method = fit_result.method.upper()

    if method not in ('MLE', 'GAS'):
        from pyscarcopula._native import scar_ou as _cpp_scar_ou

        config = _native_grid_config_from_result(fit_result, K, grid_range)
        kappa, mu, nu_ou = fit_result.params.values
        return _cpp_scar_ou.student_rosenblatt(
            kappa,
            mu,
            nu_ou,
            u,
            copula,
            config,
        )

    if method == 'MLE':
        df = fit_result.copula_param
        if getattr(copula, 'corr_mode', None) == 'factor':
            e = factor_student_rosenblatt_transform(
                copula.correlation_operator_, df, u)
        else:
            e = student_rosenblatt_transform(copula.R, df, u)
        return clip_pseudo_observations(e)

    if method == 'GAS':
        df_path = _gas_parameter_path(copula, u, fit_result)
        if getattr(copula, 'corr_mode', None) == 'factor':
            e = factor_student_rosenblatt_transform(
                copula.correlation_operator_, df_path, u)
        else:
            e = student_rosenblatt_transform(copula.R, df_path, u)
        return clip_pseudo_observations(e)

    raise AssertionError(f"unsupported Student estimation method: {method}")


def stochastic_student_gof_test(copula, data, to_pobs=True,
                                 K=300, grid_range=5.0, fit_result=None):
    """
    Goodness-of-fit test for StochasticStudentCopula.

    Parameters
    ----------
    copula : StochasticStudentCopula
    data : (T, d)
    to_pobs : bool
    K : int
    grid_range : float
    fit_result : FitResult or None

    Returns
    -------
    CramérVonMisesResult
    """
    u = _prepare_gof_data(
        data, expected_dim=copula.dimension, to_pobs=to_pobs)

    fr = fit_result if fit_result is not None else getattr(copula, 'fit_result', None)
    if fr is None:
        raise ValueError("No fit_result provided and copula has no fit_result. "
                         "Call copula.fit() first or pass fit_result=.")

    e = stochastic_student_rosenblatt_transform(copula, u, fr, K, grid_range)
    return cvm_test(e)


# ══════════════════════════════════════════════════════════════════
