"""
Goodness-of-fit tests for copula models (bivariate and vine).

Bivariate:
    MLE:  e2 = h(u2, u1, r)
    SCAR: e2 = E[h(u2, u1, Psi(x_k)) | u_{1:k-1}]  (mixture)

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
    from pyscarcopula.stattests import gof_test, vine_gof_test
"""

import numpy as np
import time
from dataclasses import dataclass
from typing import Callable
from scipy.stats import chi2, norm, cramervonmises

from pyscarcopula._parallel import (
    create_worker_model,
    get_copula_constructor,
    resolve_parallelism,
    spawn_seed_sequences,
    with_n_threads,
)
from pyscarcopula._utils import (
    clip_pseudo_observations,
    clip_pseudo_observations_no_copy,
    clip_rosenblatt_output,
)
from pyscarcopula.numerical._arrays import (
    as_float64_array,
    as_pseudo_observation_array,
    validate_float64_allocation,
    validate_positive_int,
)


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

    # Avoid inf in norm.ppf at exactly 0 or 1
    e = clip_pseudo_observations(e)

    z = norm.ppf(e)                       # (T, d)
    q = np.sum(z * z, axis=1)             # (T,)
    y = chi2.cdf(q, df=d)                 # should be U[0,1] under H0

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
    T = len(u)
    e = np.empty((T, 2))
    e[:, 0] = u[:, 0]
    e[:, 1] = copula.h(u[:, 1], u[:, 0], np.full(T, float(r)))
    return clip_rosenblatt_output(e)


def rosenblatt_transform_scar(copula, u, alpha, K=300, grid_range=5.0,
                              grid_method='auto', adaptive=True,
                              pts_per_sigma=4, transition_method='matrix',
                              max_K=None, r_gh=3.0, gh_order=5):
    """Mixture Rosenblatt for SCAR (bivariate). Returns (T, 2)."""
    from pyscarcopula.numerical import _cpp_scar_ou
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


def rosenblatt_transform_gas(copula, u, gas_params, scaling='unit'):
    """Rosenblatt for GAS (bivariate). Returns (T, 2)."""
    from pyscarcopula.numerical.gas_filter import gas_rosenblatt
    omega, gamma, beta = gas_params
    return gas_rosenblatt(omega, gamma, beta, u, copula, scaling)


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
      - CVineCopula      -> legacy C-vine Rosenblatt
      - GaussianCopula   -> Cholesky-based Rosenblatt
      - StudentCopula    -> conditional t-distribution Rosenblatt

    Parameters
    ----------
    model : BivariateCopula, VineCopula, CVineCopula, GaussianCopula,
        or StudentCopula
    data : (T, d) array
    to_pobs : bool
    K : int — grid size (SCAR only)
    grid_range : float (SCAR only)
    fit_result : FitResult or None
        If provided, use this instead of model.fit_result.
        Enables the stateless API: gof_test(copula, u, fit_result=result)
    bootstrap : bool
        If True, calibrate a supported bivariate, static multivariate, or
        dynamic multivariate CvM statistic by parametric bootstrap instead of
        using the one-sample asymptotic p-value.
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
    from pyscarcopula.vine.cvine import CVineCopula
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
        if bootstrap:
            raise NotImplementedError(
                "Bootstrap GoF is not implemented for VineCopula.")
        return rvine_gof_test(
            model,
            data,
            to_pobs,
            K,
            grid_range,
            vine_type=model.vine_type,
        )
    elif isinstance(model, CVineCopula):
        if bootstrap:
            raise NotImplementedError(
                "Bootstrap GoF is not implemented for CVineCopula.")
        return vine_gof_test(model, data, to_pobs, K, grid_range)
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
        return gaussian_gof_test(model, data, to_pobs)
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
        return student_gof_test(model, data, to_pobs)
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
            copula, u, fit_result.params.values, scaling)

    if getattr(fit_result, 'params', None) is None:
        raise ValueError(
            f"Cannot compute bivariate Rosenblatt transform for {method}")

    from pyscarcopula.strategy._base import get_strategy_for_result

    strategy = get_strategy_for_result(
        fit_result, K=K, grid_range=grid_range)
    e = np.empty((len(u), 2), dtype=np.float64)
    e[:, 0] = u[:, 0]
    e[:, 1] = strategy.rosenblatt_e2(copula, u, fit_result)
    return clip_rosenblatt_output(e)


def _bootstrap_fit_kwargs(fit_result, fit_kwargs):
    """Warm-start bootstrap refits from the original fitted parameters."""
    out = dict(fit_kwargs)
    if 'alpha0' in out or 'gamma0' in out:
        return out

    method = fit_result.method.upper()
    if method == 'MLE' and hasattr(fit_result, 'copula_param'):
        out['alpha0'] = np.array([fit_result.copula_param], dtype=np.float64)
    else:
        params = getattr(fit_result, 'params', None)
        if params is not None:
            key = 'gamma0' if method == 'GAS' else 'alpha0'
            out[key] = np.asarray(params.values, dtype=np.float64)
    return out


def _fit_result_diagnostics(result):
    row = {
        'bootstrap_fit_method': getattr(result, 'method', ''),
        'bootstrap_fit_log_likelihood': float(
            getattr(result, 'log_likelihood', np.nan)),
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


def _bootstrap_strategy(fit_result, config):
    if (
            fit_result.method.upper() == 'MLE'
            and not hasattr(fit_result, 'copula_param')):
        return None

    from pyscarcopula.strategy._base import get_strategy_for_result

    return get_strategy_for_result(fit_result, config=config)


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
    strategy = _bootstrap_strategy(fit_result, config)
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


def _bootstrap_simulate_gaussian(
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


def _bootstrap_simulate_student(
        copula, u, fit_result, rng, K, grid_range, n_threads, config):
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


def _bootstrap_prepare_equicorr(
        copula_class, constructor_kwargs, fit_result, fitted_snapshot):
    copula = create_worker_model(copula_class, constructor_kwargs)
    copula.fit_result = fit_result
    return copula


def _bootstrap_simulate_equicorr(
        copula, u, fit_result, rng, K, grid_range, n_threads, config):
    return copula.sample(len(u), u=u, rng=rng)


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

    from pyscarcopula.strategy._base import get_strategy_for_result

    strategy = get_strategy_for_result(
        fit_result,
        config=config,
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


def _bootstrap_simulate_stochastic_student(
        copula, u, fit_result, rng, K, grid_range, n_threads, config):
    return copula.sample(
        len(u), u=u, rng=rng, n_threads=n_threads)


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
        simulate=_bootstrap_simulate_gaussian,
        refit=_bootstrap_refit_static_multivariate,
        statistic=_bootstrap_statistic_gaussian,
    ),
    'student': _BootstrapAdapter(
        capture=_bootstrap_capture_none,
        prepare=_bootstrap_prepare_student,
        simulate=_bootstrap_simulate_student,
        refit=_bootstrap_refit_static_multivariate,
        statistic=_bootstrap_statistic_student,
    ),
    'equicorr': _BootstrapAdapter(
        capture=_bootstrap_capture_none,
        prepare=_bootstrap_prepare_equicorr,
        simulate=_bootstrap_simulate_equicorr,
        refit=_bootstrap_refit_dynamic,
        statistic=_bootstrap_statistic_equicorr,
    ),
    'stochastic_student': _BootstrapAdapter(
        capture=_bootstrap_capture_stochastic_student,
        prepare=_bootstrap_prepare_stochastic_student,
        simulate=_bootstrap_simulate_stochastic_student,
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

def _vine_edge_h(edge, u2, u1, u_pair, K=300, grid_range=5.0):
    """Delegate to the shared pair-edge runtime."""
    from pyscarcopula.vine._rvine_edges import _edge_h
    return _edge_h(
        edge,
        u2,
        u1,
        u_pair=u_pair,
        K=K,
        grid_range=grid_range,
    )


def vine_rosenblatt_transform(vine, u, K=300, grid_range=5.0):
    """
    Rosenblatt transform for a fitted C-vine copula.

    Each edge in the vine is an independent bivariate copula
    (possibly with its own latent OU process). The vine Rosenblatt
    simply applies h-functions level by level, reusing the bivariate
    approach on every edge — no vine-specific modifications needed.

    ```
    v[0][i] = u_i
    v[j+1][i] = h(v[j][i+1] | v[j][0]; edge_{j,i})
    e_0 = u_0
    e_{j+1} = h(v[j][1] | v[j][0]; edge_{j,0})
    ```

    Parameters
    ----------
    vine : CVineCopula (fitted)
    u : (T, d) pseudo-observations
    K : int — grid size for SCAR mixture
    grid_range : float

    Returns
    -------
    e : (T, d) — should be iid U[0,1]^d under correct model
    """
    T, d = u.shape
    v = [[None] * d for _ in range(d)]
    for i in range(d):
        v[0][i] = clip_pseudo_observations(u[:, i].copy())

    e = np.empty((T, d))
    e[:, 0] = v[0][0]

    for j in range(d - 1):
        n_edges = d - j - 1

        # e_{j+1}: first edge of tree j
        u1 = clip_pseudo_observations(v[j][0])
        u2 = clip_pseudo_observations(v[j][1])
        u_pair = np.column_stack((u1, u2))
        edge = vine.edges[j][0]
        e[:, j + 1] = clip_pseudo_observations(
            _vine_edge_h(edge, u2, u1, u_pair, K, grid_range))

        # Propagate v to next level (all edges, same approach)
        if j < d - 2:
            for i in range(n_edges):
                u1 = clip_pseudo_observations(v[j][0])
                u2 = clip_pseudo_observations(v[j][i + 1])
                u_pair = np.column_stack((u1, u2))
                edge_i = vine.edges[j][i]
                v[j + 1][i] = clip_pseudo_observations(
                    _vine_edge_h(edge_i, u2, u1, u_pair, K, grid_range))

    return clip_rosenblatt_output(e)


# ══════════════════════════════════════════════════════════════════
# Vine gof_test
# ══════════════════════════════════════════════════════════════════

def vine_gof_test(vine, data, to_pobs=True, K=500, grid_range=7.0):
    """
    Goodness-of-fit test for a fitted C-vine copula.

    Applies the d-dimensional Rosenblatt transform through the vine
    tree structure, then tests e ~ iid U[0,1]^d via CvM.

    For SCAR edges: uses mixture h-function (avoids Jensen bias).
    For MLE edges: uses constant parameter h-function.

    Parameters
    ----------
    vine : CVineCopula (fitted)
    data : (T, d)
    to_pobs : bool
    K : int — grid size for SCAR mixture Rosenblatt
    grid_range : float

    Returns
    -------
    CramérVonMisesResult with .statistic and .pvalue
    """
    u = _prepare_gof_data(
        data, expected_dim=getattr(vine, "d", None), to_pobs=to_pobs)

    if vine.edges is None:
        raise ValueError("Fit the vine first")

    e = vine_rosenblatt_transform(vine, u, K=K, grid_range=grid_range)
    return cvm_test(e)


def rvine_rosenblatt_transform(
        vine, u, K=300, grid_range=5.0, *, vine_type=None):
    """
    Rosenblatt transform for a fitted R-vine copula.

    Mirrors ``VineCopula.sample`` for the natural-order matrix:
    columns are traversed right-to-left, and each anti-diagonal leaf is
    transformed by h-functions from tree 0 up to the column's top tree.
    """
    from pyscarcopula.vine._rvine_edges import (
        _edge_h,
        _edge_h_pair_for_variables,
    )

    if vine_type is None:
        vine_type = getattr(vine, "vine_type", "rvine")
    if vine_type not in {"cvine", "dvine", "rvine"}:
        raise ValueError(
            "vine_type must be 'cvine', 'dvine' or 'rvine', "
            f"got {vine_type!r}")

    if getattr(vine, 'matrix', None) is None:
        raise ValueError("Fit the vine first")

    u = np.asarray(u, dtype=np.float64)
    T, d = u.shape
    if d != vine.d:
        raise ValueError(f"u has d={d}, but fitted vine has d={vine.d}")

    M = vine.matrix

    if d == 2:
        edge = vine.pair_copulas[(0, 0)]
        u1 = clip_pseudo_observations(u[:, 0])
        u2 = clip_pseudo_observations(u[:, 1])
        u_pair = np.column_stack((u1, u2))
        e = np.empty((T, d), dtype=np.float64)
        e[:, 0] = u1
        e[:, 1] = clip_pseudo_observations(
            _edge_h(edge, u2, u1, u_pair=u_pair, K=K,
                    grid_range=grid_range))
        return clip_rosenblatt_output(e)

    pseudo = {
        (var, frozenset()): clip_pseudo_observations(u[:, var].copy())
        for var in range(d)
    }

    e = np.empty((T, d), dtype=np.float64)

    last_var = int(M[0, d - 1])
    e[:, d - 1] = pseudo[(last_var, frozenset())]

    for col in range(d - 2, -1, -1):
        leaf = int(M[d - 1 - col, col])
        top_tree = d - 2 - col
        cur = pseudo[(leaf, frozenset())]

        for t in range(top_tree + 1):
            row = d - 2 - col - t
            partner = int(M[row, col])
            conditioning = frozenset(
                int(M[r, col])
                for r in range(row + 1, d - 1 - col)
            )
            next_leaf_cond = conditioning | {partner}
            next_partner_cond = conditioning | {leaf}

            edge = vine.pair_copulas[(t, col)]
            leaf_val = pseudo.get((leaf, conditioning))
            partner_val = pseudo.get((partner, conditioning))
            if leaf_val is None:
                raise RuntimeError(
                    "Missing leaf pseudo-observation during Rosenblatt: "
                    f"var={leaf}, cond_set={sorted(conditioning)}, "
                    f"column={col}, tree={t}"
                )
            if partner_val is None:
                raise RuntimeError(
                    "Missing partner pseudo-observation during Rosenblatt: "
                    f"var={partner}, cond_set={sorted(conditioning)}, "
                    f"column={col}, tree={t}"
                )

            leaf_next, partner_next = _edge_h_pair_for_variables(
                edge,
                leaf,
                leaf_val,
                partner,
                partner_val,
                K=K,
                grid_range=grid_range,
            )
            cur = clip_pseudo_observations(leaf_next)
            pseudo[(leaf, next_leaf_cond)] = cur
            pseudo[(partner, next_partner_cond)] = (
                clip_pseudo_observations(partner_next))

        e[:, col] = cur

    return clip_rosenblatt_output(e)

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
    u_c = clip_pseudo_observations_no_copy(u)
    x = norm.ppf(u_c)

    L = np.linalg.cholesky(R)
    # z = L^{-1} x, so z_i are independent N(0,1)
    # e_i = Phi(z_i)
    z = np.linalg.solve(L, x.T).T  # (T, d)
    e = norm.cdf(z)

    return clip_pseudo_observations(e)


def factor_gaussian_rosenblatt_transform(correlation, u):
    """Rosenblatt transform for a Gaussian factor correlation.

    The sequential conditional distribution is evaluated with the
    rank-dimensional posterior of the latent factor. Storage is
    ``O(T*k + k*k)`` and no dense correlation or Cholesky factor is formed.
    """
    u_c = clip_pseudo_observations_no_copy(u)
    x = norm.ppf(u_c)
    if x.ndim != 2 or x.shape[1] != correlation.dimension:
        raise ValueError(
            "data width must match factor correlation dimension")

    rows, dimension = x.shape
    rank = correlation.rank
    loadings = correlation.loadings
    uniqueness = correlation.uniqueness
    factor_mean = np.zeros((rows, rank), dtype=np.float64)
    factor_covariance = np.eye(rank, dtype=np.float64)
    transformed = np.empty_like(x)

    for index in range(dimension):
        loading = loadings[index]
        covariance_loading = factor_covariance @ loading
        conditional_variance = (
            uniqueness[index] + loading @ covariance_loading)
        residual = x[:, index] - factor_mean @ loading
        transformed[:, index] = norm.cdf(
            residual / np.sqrt(conditional_variance))
        factor_mean += (
            residual / conditional_variance)[:, None] * (
                covariance_loading[None, :])
        factor_covariance -= np.outer(
            covariance_loading,
            covariance_loading,
        ) / conditional_variance

    return clip_pseudo_observations(transformed)


def gaussian_gof_test(copula, data, to_pobs=True):
    """
    Goodness-of-fit test for a fitted GaussianCopula.

    Parameters
    ----------
    copula : GaussianCopula (fitted, has .corr)
    data : (T, d)
    to_pobs : bool

    Returns
    -------
    CramérVonMisesResult
    """
    u = _prepare_gof_data(
        data, expected_dim=copula.dimension, to_pobs=to_pobs)

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
    """
    Rosenblatt transform for d-dimensional Student-t copula.

    x = t_df^{-1}(u), x ~ t_d(0, R, df).

    Sequential conditioning using the property that for
    multivariate t with shape R and df degrees of freedom:

        x_i | x_0,...,x_{i-1} ~ t_{df+i}(mu_i, sigma^2_i * scale)

    where:
        mu_i = R_{i,0:i} R_{0:i,0:i}^{-1} x_{0:i}
        sigma^2_i = R_{ii} - R_{i,0:i} R_{0:i,0:i}^{-1} R_{0:i,i}
        scale = (df + x_{0:i}^T R_{0:i,0:i}^{-1} x_{0:i}) / (df + i)

    Here i is the zero-based coordinate index, so the conditioning set has
    size i.

    Parameters
    ----------
    R : (d, d) shape matrix (correlation)
    df : float — degrees of freedom
    u : (T, d) pseudo-observations

    Returns
    -------
    e : (T, d)
    """
    from scipy.stats import t as t_dist

    u_c = clip_pseudo_observations(u)
    x = t_dist.ppf(u_c, df=df)

    T, d = x.shape
    e = np.empty((T, d))

    # First variable: e_0 = t_df.cdf(x_0)
    e[:, 0] = t_dist.cdf(x[:, 0], df=df)

    for i in range(1, d):
        # Conditional distribution of x_i | x_{0:i-1}
        R_11 = R[:i, :i]          # (i, i)
        R_21 = R[i, :i]           # (i,)
        R_22 = R[i, i]            # scalar

        R_11_inv = np.linalg.inv(R_11)
        beta = R_21 @ R_11_inv    # (i,) — regression coefficients

        # Conditional variance (without scale)
        sigma2_cond = R_22 - R_21 @ R_11_inv @ R_21  # scalar
        sigma_cond = np.sqrt(max(sigma2_cond, 1e-12))

        # For each observation
        x_prev = x[:, :i]                          # (T, i)
        mu_cond = x_prev @ beta                     # (T,)

        # Quadratic form: x_{1:i-1}^T R_{1:i-1}^{-1} x_{1:i-1}
        quad = np.sum(x_prev @ R_11_inv * x_prev, axis=1)  # (T,)

        # Scale factor and conditional df
        df_cond = df + i
        scale = (df + quad) / df_cond

        # Standardized residual
        z = (x[:, i] - mu_cond) / (sigma_cond * np.sqrt(scale))

        e[:, i] = t_dist.cdf(z, df=df_cond)

    return clip_pseudo_observations(e)


def factor_student_rosenblatt_transform(correlation, df, u):
    """Rosenblatt transform for a Student factor correlation.

    Sequential conditioning uses the rank-dimensional Woodbury posterior.
    Storage is ``O(T*k + k*k)`` and no dense correlation matrix is formed.
    ``df`` may be scalar or contain one value per observation.
    """
    from scipy.stats import t as t_dist

    u_c = clip_pseudo_observations_no_copy(u)
    if u_c.ndim != 2 or u_c.shape[1] != correlation.dimension:
        raise ValueError(
            "data width must match factor correlation dimension")
    rows, dimension = u_c.shape
    df_path = np.asarray(df, dtype=np.float64)
    if df_path.ndim == 0:
        df_path = np.full(rows, float(df_path), dtype=np.float64)
    else:
        df_path = np.ravel(df_path)
        if len(df_path) != rows:
            raise ValueError("df must be scalar or have one value per row")
    if (
            not np.all(np.isfinite(df_path))
            or np.any(df_path <= 2.0)):
        raise ValueError("df must be finite and greater than 2")

    x = t_dist.ppf(u_c, df=df_path[:, None])
    loadings = correlation.loadings
    uniqueness = correlation.uniqueness
    rank = correlation.rank
    factor_covariance = np.eye(rank, dtype=np.float64)
    projected = np.zeros((rows, rank), dtype=np.float64)
    diagonal_quadratic = np.zeros(rows, dtype=np.float64)
    transformed = np.empty_like(x)
    transformed[:, 0] = u_c[:, 0]

    for index in range(dimension):
        loading = loadings[index]
        covariance_loading = factor_covariance @ loading
        conditional_variance = float(
            uniqueness[index] + loading @ covariance_loading)
        if not np.isfinite(conditional_variance) or (
                conditional_variance <= 0.0):
            raise ValueError(
                "factor correlation produced non-positive "
                "conditional variance")

        if index > 0:
            solved_projection = projected @ factor_covariance
            conditional_mean = solved_projection @ loading
            quadratic = diagonal_quadratic - np.einsum(
                "ij,ij->i",
                projected,
                solved_projection,
                optimize=False,
            )
            quadratic = np.maximum(quadratic, 0.0)
            conditional_df = df_path + index
            scale = (df_path + quadratic) / conditional_df
            standardized = (
                (x[:, index] - conditional_mean)
                / np.sqrt(conditional_variance * scale)
            )
            transformed[:, index] = t_dist.cdf(
                standardized, df=conditional_df)

        projected += (
            x[:, index] / uniqueness[index]
        )[:, None] * loading[None, :]
        diagonal_quadratic += (
            x[:, index] * x[:, index] / uniqueness[index])
        factor_covariance -= np.outer(
            covariance_loading,
            covariance_loading,
        ) / conditional_variance

    return clip_pseudo_observations(transformed)


def student_gof_test(copula, data, to_pobs=True):
    """
    Goodness-of-fit test for a fitted StudentCopula.

    Parameters
    ----------
    copula : StudentCopula (fitted, has .shape and .df)
    data : (T, d)
    to_pobs : bool

    Returns
    -------
    CramérVonMisesResult
    """
    u = _prepare_gof_data(
        data, expected_dim=copula.dimension, to_pobs=to_pobs)

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


def _tm_grid_kwargs_from_result(fit_result):
    """SCAR-TM numerical options stored on a fitted result."""
    out = {}
    for name in (
            'grid_method', 'adaptive', 'pts_per_sigma',
            'transition_method', 'max_K',
            'r_gh', 'gh_order'):
        value = getattr(fit_result, name, None)
        if value is not None:
            out[name] = value
    if 'transition_method' in out:
        out['transition_method'] = _grid_transition_method(
            out['transition_method'])
    return out


def _native_grid_config_from_result(fit_result, K, grid_range):
    """Build the native grid config with legacy TMGrid default semantics."""
    from pyscarcopula.numerical._scar_ou_config import AutoTMConfig

    options = _tm_grid_kwargs_from_result(fit_result)
    return AutoTMConfig(
        K=K,
        grid_range=grid_range,
        grid_method=options.get('grid_method', 'auto'),
        adaptive=options.get('adaptive', True),
        pts_per_sigma=options.get('pts_per_sigma', 4),
        transition_method=options.get('transition_method', 'matrix'),
        max_K=options.get('max_K', None),
        r_gh=options.get('r_gh', 3.0),
        gh_order=options.get('gh_order', 5),
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
        from pyscarcopula.numerical import _cpp_scar_ou

        kappa, mu, nu = fit_result.params.values
        return _cpp_scar_ou.gaussian_rosenblatt(
            kappa,
            mu,
            nu,
            u,
            copula,
            _native_grid_config_from_result(
                fit_result, K, grid_range),
        )

    u_c = clip_pseudo_observations(u)
    x_norm = norm.ppf(u_c)

    if method == 'MLE':
        rho = fit_result.copula_param
        e = np.empty((T, d))
        e[:, 0] = u[:, 0]
        for i in range(1, d):
            sx = np.sum(x_norm[:, :i], axis=1)
            cond_mean = rho * sx / (1.0 + (i - 1) * rho)
            cond_var = 1.0 - i * rho ** 2 / (1.0 + (i - 1) * rho)
            cond_var = max(cond_var, 1e-10)
            z_i = (x_norm[:, i] - cond_mean) / np.sqrt(cond_var)
            e[:, i] = norm.cdf(z_i)
        return clip_pseudo_observations(e)

    if method == 'GAS':
        rho_path = _gas_parameter_path(copula, u, fit_result)
        e = np.empty((T, d))
        e[:, 0] = u[:, 0]
        for i in range(1, d):
            rho = rho_path
            sx = np.sum(x_norm[:, :i], axis=1)
            cond_mean = rho * sx / (1.0 + (i - 1) * rho)
            cond_var = 1.0 - i * rho ** 2 / (1.0 + (i - 1) * rho)
            cond_var = np.maximum(cond_var, 1e-10)
            z_i = (x_norm[:, i] - cond_mean) / np.sqrt(cond_var)
            e[:, i] = norm.cdf(z_i)
        return clip_pseudo_observations(e)

    raise ValueError(
        f"Unsupported EquicorrGaussianCopula fit method: {fit_result.method}")


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
        from pyscarcopula.numerical import _cpp_scar_ou

        kappa, mu, nu_ou = fit_result.params.values
        return _cpp_scar_ou.student_rosenblatt(
            kappa,
            mu,
            nu_ou,
            u,
            copula,
            _native_grid_config_from_result(
                fit_result, K, grid_range),
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
            e = np.empty((T, d))
            for t_idx, df_t in enumerate(df_path):
                e[t_idx] = student_rosenblatt_transform(
                    copula.R, float(df_t), u[t_idx:t_idx + 1])[0]
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
