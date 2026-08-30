"""
Stochastic Student-t copula: d-dimensional t-copula with OU-driven df.

Static correlation can be fixed, shrinkage-estimated, fully estimated through
a Cholesky-like parameterization, or represented by a compact factor operator.
The degrees-of-freedom parameter nu(t) = Psi(x(t)) varies over time,
where x(t) is a latent Ornstein-Uhlenbeck process.

Psi(x) = 2 + 1e-6 + softplus(x), mapping R above the finite-variance bound.

This class inherits MultivariateCopula. Its latent process remains scalar
(one OU process drives df), so compatible dynamic strategies can use the
explicit capability contract without exposing pair-copula operations.

Usage:
    from pyscarcopula.copula.multivariate.stochastic_student import StochasticStudentCopula

    cop = StochasticStudentCopula(d=6)
    result = cop.fit(returns, method='scar-tm-ou', to_pobs=True)
    samples = cop.sample_at_parameter(10000, r=5.0)
    pred = cop.predict(100000)

    from pyscarcopula.stattests import gof_test
    gof_test(cop, returns, to_pobs=True)
"""

from __future__ import annotations

from dataclasses import replace
import threading
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from pyscarcopula.copula.multivariate.base import (
    MultivariateCopula,
    factor_uniqueness,
    model_state_locked,
)
from pyscarcopula._types import DEFAULT_CONFIG, FitResultBase, NumericalConfig
from pyscarcopula._native import model_policy
from pyscarcopula._utils import pobs
from pyscarcopula.copula.multivariate.conditional import (
    fill_given,
    sample_factor_student_conditional,
    sample_student_conditional,
    validate_multivariate_given,
)
from pyscarcopula.copula.multivariate.corr_param import (
    _corr_gradient_to_raw_params,
    _corr_from_cholesky_params,
    _make_shrinkage_corr_from_validated,
    cholesky_corr_n_params,
    estimate_kendall_correlation,
    logit,
    pack_cholesky_corr,
    project_to_corr,
    preprocess_correlation_matrix,
    sigmoid,
)
from pyscarcopula.copula.multivariate.correlation_policy import (
    CorrelationEstimator,
    CorrelationMode,
    CorrelationPolicy,
    FactorEstimation,
    factor_parameter_count,
    normalize_correlation_mode,
    normalize_factor_estimation,
    validate_joint_factor_rank,
)
from pyscarcopula.copula.multivariate.student_ppf_cache import (
    StudentPPFTable as _PPFTable,
    prepare_student_ppf_cache,
)
from pyscarcopula.copula.multivariate.factor_correlation import (
    FactorCorrelation,
)
from pyscarcopula.copula.multivariate.factor_estimation import (
    FactorLoadingParameterization,
    estimate_factor_loadings,
)
from pyscarcopula.numerical._arrays import (
    as_float64_array,
    validate_integer,
    validate_sampling_memory_budget as _sampling_memory_budget,
    validate_sampling_n_threads as _sampling_n_threads,
)
from pyscarcopula.copula.multivariate.factor_student import (
    FactorStudentEvaluator,
)


_LBFGSB_FIT_KEYS = (
    'gtol',
    'ftol',
    'maxfun',
    'maxiter',
    'maxls',
    'eps',
    'maxcor',
    'finite_diff_rel_step',
)

_STUDENT_FIT_INITIAL, _STUDENT_FIT_BOUNDS = (
    model_policy.student_fit_policy(2, stochastic=True))
_DF_OFFSET, _DF_FIT_UPPER = _STUDENT_FIT_BOUNDS
def _as_float64_array_no_copy(value):
    return as_float64_array(value, name="data")


def _validate_fit_data(u, d):
    if u.ndim != 2 or u.shape[1] != int(d):
        raise ValueError(f"data must have shape (n_observations, {int(d)})")
    if u.shape[0] == 0:
        raise ValueError("data must contain at least one observation")
    if not np.all(np.isfinite(u)):
        raise ValueError("data must contain only finite values")


def _validated_student_sampling_parameters(r, n):
    values = np.atleast_1d(np.asarray(r, dtype=np.float64)).ravel()
    if values.size not in (1, int(n)):
        raise ValueError(
            f"r must be scalar or array of length {int(n)}, "
            f"got {values.size}")
    if np.any(~np.isfinite(values)) or np.any(values <= 2.0):
        raise ValueError("r must be finite and greater than 2")
    return values


# ══════════════════════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════
# Class
# ══════════════════════════════════════════════════════════════

class StochasticStudentCopula(MultivariateCopula):
    """
    d-dimensional Student-t copula with stochastic degrees of freedom.

    The static correlation matrix R is fixed or estimated jointly.
    The df parameter is driven by a latent OU process:
        df(t) = Psi(x(t)) = 2 + 1e-6 + softplus(x(t))

    Compatible with SCAR-TM-OU: one latent OU process drives df(t).
    Also supports MLE (constant df) and GAS methods.

    Parameters
    ----------
    d : int
        Dimension (number of variables). Must be >= 2.
    R : (d, d) ndarray or None
        Fixed correlation matrix or initialization matrix for an estimated
        correlation mode.
    corr_mode : {'fixed', 'shrinkage', 'cholesky', 'factor'}
        Static correlation parameterization.
    corr_base : (d, d) ndarray or None
        Explicit initialization/base matrix for an estimated correlation
        mode. Initialization priority is ``corr_base``, then ``R``, then a
        Kendall estimate from the fit data.
    factor_rank : int or None
        Required rank ``k`` for ``corr_mode='factor'``.
    factor_loadings : (d, k) ndarray or None
        Optional fixed loadings. If omitted, call :meth:`initialize_factor`
        for deterministic two-stage initialization.
    factor_estimation : {'two-stage', 'joint'}
        Factor estimation policy. ``joint`` estimates identifiable loadings
        together with constant ``df`` in static MLE; dynamic GAS/SCAR joint
        estimation is not yet supported.
    factor_tile_size : int
        Dimension tile used by initialization and exact Student grid kernels.
    factor_joint_penalty : float
        L2 loading penalty used only by joint static MLE.
    factor_joint_condition_max : float
        Maximum accepted Woodbury core condition estimate in joint static MLE.

    Notes
    -----
    SCAR-TM-OU accepts and returns physical ``(kappa, mu, nu)`` parameters,
    but this model is optimized internally in
    ``(log(kappa), mu, log(sigma_x))``, where
    ``sigma_x = nu / sqrt(2 * kappa)``. The internal representation is
    reported in the fit diagnostics.
    """

    _gas_optimizer_config = 'stochastic_student_gas_optimizer'
    _scar_optimizer_config = 'stochastic_student_scar_optimizer'
    _df_offset = _DF_OFFSET
    _scar_static_df_mle_initialization = True
    _scar_log_stationary_scale_optimization = True
    _scar_stationary_scale_bounds = model_policy.stationary_scale_bounds()
    _supports_scar_mixture_h = False
    def __init__(
            self,
            d: int,
            R: ArrayLike | None = None,
            *,
            corr_mode: CorrelationMode = 'fixed',
            corr_base: ArrayLike | None = None,
            corr_shrinkage_init: float = 0.8,
            cholesky_d_max: int = 10,
            allow_large_cholesky: bool = False,
            factor_rank: int | None = None,
            factor_loadings: ArrayLike | None = None,
            factor_estimation: FactorEstimation = "two-stage",
            factor_tile_size: int = 16384,
            factor_uniqueness_min: float = 1e-8,
            factor_joint_max_params: int = 100000,
            factor_joint_penalty: float = 1e-6,
            factor_joint_condition_max: float = 1e12,
            factor_seed: int = 0,
            factor_oversampling: int = 8) -> None:
        if isinstance(d, (bool, np.bool_)) or not isinstance(
                d, (int, np.integer)):
            raise TypeError(f"d must be an integer >= 2, got {d!r}")
        d = int(d)
        if d < 2:
            raise ValueError(f"d must be >= 2, got {d}")
        corr_mode = normalize_correlation_mode(corr_mode)
        factor_estimation = normalize_factor_estimation(factor_estimation)
        if corr_mode == 'fixed' and corr_base is not None:
            raise ValueError("corr_base is only valid for estimated corr modes")
        if corr_mode == 'factor' and (R is not None or corr_base is not None):
            raise ValueError(
                "R and corr_base are forbidden when corr_mode='factor'")
        if corr_mode != 'factor' and (
                factor_rank is not None or factor_loadings is not None):
            raise ValueError(
                "factor_rank and factor_loadings require corr_mode='factor'")
        if not (0.0 < float(corr_shrinkage_init) < 1.0):
            raise ValueError("corr_shrinkage_init must be in (0, 1)")
        if (
                corr_mode == 'cholesky'
                and d > int(cholesky_d_max)
                and not allow_large_cholesky):
            raise ValueError(
                "corr_mode='cholesky' is limited to "
                f"d <= {int(cholesky_d_max)} by default")
        super().__init__(
            dimension=d, name=f"Stochastic Student-t copula (d={d})")
        self._d = d
        self._bounds = model_policy.public_bounds(self)
        self._corr_mode = corr_mode
        self._corr_preprocessing = None
        self._corr_base_preprocessing = (
            None
            if corr_base is None
            else preprocess_correlation_matrix(
                corr_base, source="corr_base"))
        self._corr_base = (
            None
            if self._corr_base_preprocessing is None
            else self._corr_base_preprocessing.correlation)
        self._corr_shrinkage_init = float(corr_shrinkage_init)
        self._corr_params_raw = np.empty(0, dtype=np.float64)
        self._corr_alpha = None
        self._cholesky_d_max = int(cholesky_d_max)
        self._allow_large_cholesky = bool(allow_large_cholesky)
        self._factor_rank = None
        self._factor_loadings = None
        self._factor_correlation = None
        self._factor_operator = None
        self._factor_initialization_diagnostics = {}
        self._constructor_factor_loadings = None
        self._factor_estimation = factor_estimation
        self._factor_tile_size = validate_integer(
            factor_tile_size, "factor_tile_size", minimum=1)
        self._factor_uniqueness_min = float(factor_uniqueness_min)
        self._factor_joint_max_params = validate_integer(
            factor_joint_max_params, "factor_joint_max_params", minimum=1)
        self._factor_joint_penalty = float(factor_joint_penalty)
        if (
                not np.isfinite(self._factor_joint_penalty)
                or self._factor_joint_penalty < 0.0):
            raise ValueError(
                "factor_joint_penalty must be finite and non-negative")
        self._factor_joint_condition_max = float(
            factor_joint_condition_max)
        if (
                not np.isfinite(self._factor_joint_condition_max)
                or self._factor_joint_condition_max <= 1.0):
            raise ValueError(
                "factor_joint_condition_max must be finite and greater "
                "than 1")
        self._factor_seed = validate_integer(factor_seed, "factor_seed")
        self._factor_oversampling = validate_integer(
            factor_oversampling, "factor_oversampling")
        if corr_mode == 'factor':
            if (
                    isinstance(factor_rank, (bool, np.bool_))
                    or not isinstance(factor_rank, (int, np.integer))):
                raise TypeError(
                    "factor_rank must be an integer when corr_mode='factor'")
            self._factor_rank = int(factor_rank)
            if not 1 <= self._factor_rank < d:
                raise ValueError("factor_rank must satisfy 1 <= k < d")
            if self._factor_estimation == 'joint':
                validate_joint_factor_rank(d, self._factor_rank)
            if not (
                    np.isfinite(self._factor_uniqueness_min)
                    and 0.0 < self._factor_uniqueness_min < 1.0):
                raise ValueError(
                    "factor_uniqueness_min must be finite and in (0, 1)")
            if (
                    self._factor_estimation == 'joint'
                    and d * self._factor_rank
                    > self._factor_joint_max_params):
                raise ValueError(
                    "joint factor estimation exceeds "
                    "factor_joint_max_params")

        # Correlation matrix — set during fit or at init
        self._R = None
        self._L = None
        self._L_inv = None
        self._log_det = None
        self._corr_cache_version = 0
        self._last_latent_result = None
        self._constructor_R = None
        self._constructor_corr_base = (
            None if self._corr_base is None else self._corr_base.copy())
        if R is not None:
            R = np.asarray(R, dtype=np.float64)
            if R.shape != (d, d):
                raise ValueError(f"R must be ({d}, {d}), got {R.shape}")
            self._set_R(R, source="supplied")
            self._constructor_R = self._R.copy()
        if factor_loadings is not None:
            self._set_factor_loadings(
                factor_loadings,
                diagnostics={"source": "supplied"},
            )
            self._constructor_factor_loadings = (
                self._factor_loadings.copy())

        # Transient full-sample PPF cache.
        self._ppf_cache = None

    @property
    def d(self):
        return self._d

    @property
    def R(self):
        if self._corr_mode == 'factor':
            raise RuntimeError(
                "R is not materialized for corr_mode='factor'; use "
                "correlation_operator_ or to_correlation_matrix()")
        if self._R is None:
            return None
        return self._R.copy()

    @property
    def corr_mode(self) -> CorrelationMode:
        return self._corr_mode

    @property
    def corr_estimator_(self) -> CorrelationEstimator:
        if self._corr_mode == "factor":
            if self._factor_estimation == "joint":
                return "factor_joint"
            return "factor_two_stage"
        if self._corr_mode in {"shrinkage", "cholesky"}:
            return "joint_mle"
        preprocessing = self._corr_preprocessing
        if preprocessing is not None and preprocessing.source == "supplied":
            return "supplied"
        return "kendall_plugin"

    @property
    def correlation_policy_(self) -> CorrelationPolicy:
        """Return an immutable view of the active correlation policy."""
        preprocessing = self._corr_preprocessing
        supplied = None
        if preprocessing is not None and preprocessing.source == "supplied":
            supplied = self._R
        return CorrelationPolicy.create(
            mode=self._corr_mode,
            estimator=self.corr_estimator_,
            dimension=self._d,
            supplied_correlation=supplied,
            base_correlation=self._corr_base,
            preprocessing=preprocessing,
            raw_parameters=self._corr_params_raw,
            factor_rank=self._factor_rank,
            factor_estimation=(
                self._factor_estimation
                if self._corr_mode == "factor" else None),
            shrinkage_initial=self._corr_shrinkage_init,
        )

    @property
    def factor_rank(self):
        return getattr(self, "_factor_rank", None)

    @property
    def factor_estimation(self) -> FactorEstimation:
        return getattr(self, "_factor_estimation", "two-stage")

    @property
    def factor_tile_size(self):
        return getattr(self, "_factor_tile_size", 16384)

    @property
    def factor_uniqueness_min(self):
        return getattr(self, "_factor_uniqueness_min", 1e-8)

    @property
    def factor_joint_max_params(self):
        return getattr(self, "_factor_joint_max_params", 100000)

    @property
    def factor_joint_penalty(self):
        return getattr(self, "_factor_joint_penalty", 1e-6)

    @property
    def factor_joint_condition_max(self):
        return getattr(self, "_factor_joint_condition_max", 1e12)

    @property
    def factor_loadings_(self):
        if self._corr_mode != 'factor':
            return None
        if self._factor_loadings is None:
            return None
        return self._factor_loadings.copy()

    factor_uniqueness_ = property(factor_uniqueness)

    @property
    def correlation_operator_(self):
        if self._corr_mode != 'factor':
            raise AttributeError(
                "correlation_operator_ is only available in factor mode")
        if self._factor_operator is None:
            raise ValueError(
                "factor correlation is not initialized; call "
                "initialize_factor()")
        return self._factor_operator

    def to_correlation_matrix(
            self, *, max_dimension=2048, memory_budget_bytes=None):
        """Explicitly materialize a small factor correlation matrix."""
        return self.correlation_operator_.to_dense(
            max_dimension=max_dimension,
            memory_budget_bytes=memory_budget_bytes,
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_state_lock", None)
        state.pop("_emission_cache", None)
        # Native factor workspaces and MappingProxy diagnostics are rebuilt
        # from the compact O(d*k) loading array after deserialization.
        state["_factor_correlation"] = None
        state["_factor_operator"] = None
        state["_ppf_cache"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._state_lock = threading.RLock()
        self._corr_cache_version = int(
            getattr(self, "_corr_cache_version", 0))
        self._corr_preprocessing = getattr(
            self, "_corr_preprocessing", None)
        self._corr_base_preprocessing = getattr(
            self, "_corr_base_preprocessing", None)
        self.__dict__.pop("_ppf_table", None)
        self.__dict__.pop("_ppf_table_u", None)
        self.__dict__.pop("_ppf_table_u_id", None)
        self.__dict__.pop("_emission_cache", None)
        self._ppf_cache = None
        self._factor_correlation = None
        self._factor_operator = None
        factor_loadings = getattr(self, "_factor_loadings", None)
        if (
                getattr(self, "_corr_mode", None) == "factor"
                and factor_loadings is not None):
            self._set_factor_loadings(
                factor_loadings,
                diagnostics=getattr(
                    self, "_factor_initialization_diagnostics",
                    {"source": "restored"}),
            )
        self._L = None
        if (
                getattr(self, "_corr_mode", None) != "factor"
                and self._R is not None):
            from pyscarcopula._native import multivariate as multivariate_native
            self._L_inv, self._log_det = (
                multivariate_native.prepare_dense_correlation(self._R))
        else:
            self._L_inv = None
            self._log_det = None

    def _set_factor_loadings(self, loadings, *, diagnostics=None):
        if self._corr_mode != 'factor':
            raise ValueError(
                "factor loadings require corr_mode='factor'")
        loadings = np.asarray(loadings, dtype=np.float64)
        expected = (self._d, self._factor_rank)
        if loadings.shape != expected:
            raise ValueError(
                f"factor_loadings must have shape {expected}, "
                f"got {loadings.shape}")
        factor = FactorCorrelation(
            loadings,
            uniqueness_min=self._factor_uniqueness_min,
            diagnostics={} if diagnostics is None else diagnostics,
        )
        operator = factor.prepare()
        self._factor_loadings = factor.loadings
        self._factor_correlation = factor
        self._factor_operator = operator
        self._factor_initialization_diagnostics = dict(
            factor.diagnostics)
        self._corr_cache_version += 1

    @model_state_locked
    def initialize_factor(self, data, *, to_pobs=False):
        """Initialize fixed factor loadings without a dense covariance.

        Supplied constructor loadings are retained. Otherwise a deterministic
        randomized SVD over tiled normal scores supplies either fixed
        two-stage loadings or the starting point for joint static MLE.
        """
        operator = self._initialize_factor_from_data(data, to_pobs=to_pobs)
        # Explicit initialization is fixed policy, unlike the data-derived
        # loadings prepared internally for each independent fit.
        self._constructor_factor_loadings = self._factor_loadings.copy()
        return operator

    def _initialize_factor_from_data(self, data, *, to_pobs=False):
        if self._corr_mode != 'factor':
            raise ValueError(
                "initialize_factor requires corr_mode='factor'")
        u = _as_float64_array_no_copy(data)
        _validate_fit_data(u, self._d)
        if to_pobs:
            u = pobs(u)
            _validate_fit_data(u, self._d)
        elif np.any((u < 0.0) | (u > 1.0)):
            raise ValueError(
                "two-stage factor initialization expects "
                "pseudo-observations in [0, 1]; use to_pobs=True")
        if self._factor_operator is not None:
            return self._factor_operator
        loadings, diagnostics = estimate_factor_loadings(
            u,
            self._factor_rank,
            uniqueness_min=self._factor_uniqueness_min,
            dimension_tile=self._factor_tile_size,
            seed=self._factor_seed,
            oversampling=self._factor_oversampling,
        )
        if self._factor_estimation == 'joint':
            diagnostics = {
                **diagnostics,
                "source": "joint_randomized_svd_start",
            }
        self._set_factor_loadings(
            loadings, diagnostics=diagnostics)
        return self._factor_operator

    def factor_diagnostics(self):
        """Return compact factor representation and initialization metadata."""
        if self._corr_mode != 'factor':
            return {}
        identifiable = factor_parameter_count(self._d, self._factor_rank)
        diagnostics = {
            'factor_rank': self._factor_rank,
            'factor_estimation': self._factor_estimation,
            'factor_n_params': identifiable,
            'factor_tile_size': self._factor_tile_size,
            'factor_uniqueness_min': self._factor_uniqueness_min,
            'factor_joint_max_params': self._factor_joint_max_params,
            'factor_joint_penalty': self._factor_joint_penalty,
            'factor_joint_condition_max':
                self._factor_joint_condition_max,
            'factor_initialized': self._factor_operator is not None,
            'representation': 'factor_woodbury',
        }
        if self._factor_operator is not None:
            diagnostics.update(dict(self._factor_operator.diagnostics))
            diagnostics.update({
                f"initialization_{key}": value
                for key, value
                in self._factor_correlation.diagnostics.items()
            })
        return diagnostics

    def _set_R(self, R, *, source="supplied"):
        """Set correlation matrix and precompute Cholesky."""
        R = np.asarray(R, dtype=np.float64)
        if R.shape != (self._d, self._d):
            raise ValueError(
                f"R must be ({self._d}, {self._d}), got {R.shape}")
        preprocessing = preprocess_correlation_matrix(
            R, source=source, eps=1e-8)
        self._set_generated_R(preprocessing.correlation)
        self._corr_preprocessing = preprocessing

    def _set_generated_R(self, R):
        """Commit an internally validated SPD correlation matrix."""
        R = np.asarray(R, dtype=np.float64)
        if R.shape != (self._d, self._d):
            raise ValueError(
                f"R must be ({self._d}, {self._d}), got {R.shape}")
        if not np.all(np.isfinite(R)):
            raise ValueError("R must contain only finite values")
        from pyscarcopula._native import multivariate as multivariate_native
        inverse_cholesky, log_determinant = (
            multivariate_native.prepare_dense_correlation(R))
        self._R = np.ascontiguousarray(R)
        self._L = None
        self._L_inv = inverse_cholesky
        self._log_det = log_determinant
        self._corr_cache_version += 1

    def _initial_corr(self, u):
        """Initial SPD correlation estimate from pseudo-observations."""
        u = np.asarray(u, dtype=np.float64)
        if u.ndim != 2 or u.shape[1] != self._d:
            raise ValueError(
                f"u must have shape (T, {self._d}), got {u.shape}")
        preprocessing = estimate_kendall_correlation(u, eps=1e-8)
        self._corr_preprocessing = preprocessing
        return preprocessing.correlation

    def _ensure_corr_initialized(self, u=None):
        """Initialize correlation from corr_base, R, or data, in that order."""
        if self._corr_mode == 'factor':
            if self._factor_operator is None:
                if u is None:
                    raise ValueError(
                        "factor correlation is not initialized")
                self._initialize_factor_from_data(u)
            return
        if self._R is None:
            if self._corr_base is not None:
                self._set_generated_R(self._corr_base)
                self._corr_preprocessing = self._corr_base_preprocessing
            elif u is not None:
                self._set_generated_R(self._initial_corr(u))
            else:
                raise ValueError("Correlation matrix R not set")

        if self._corr_mode in {'shrinkage', 'cholesky'}:
            if self._corr_base is None:
                self._corr_base = self._R.copy()

    def _corr_num_params(self):
        if self._corr_mode in {'fixed', 'factor'}:
            return 0
        if self._corr_mode == 'shrinkage':
            return 1
        return cholesky_corr_n_params(self._d)

    def _prepare_dynamic_fit(self, u):
        """Rebuild data-derived correlation without inheriting a prior fit."""
        self._ppf_cache = None
        self._corr_params_raw = np.empty(0, dtype=np.float64)
        self._corr_alpha = None
        if self._corr_mode == 'factor':
            self._factor_operator = None
            self._factor_correlation = None
            self._factor_loadings = None
            self._factor_initialization_diagnostics = {}
            if self._constructor_factor_loadings is not None:
                self._set_factor_loadings(
                    self._constructor_factor_loadings,
                    diagnostics={"source": "supplied"})
        else:
            self._R = self._L_inv = self._L = self._log_det = None
            self._corr_preprocessing = None
            self._corr_base = (
                None if self._constructor_corr_base is None
                else self._constructor_corr_base.copy())
            if self._constructor_R is not None:
                self._set_R(self._constructor_R, source="supplied")
        self._ensure_corr_initialized(u)

    def _finalize_dynamic_fit(self, result):
        diagnostics = dict(result.diagnostics)
        diagnostics.update(self._corr_count_diagnostics())
        diagnostics.update(self.correlation_preprocessing_diagnostics())
        return replace(
            result,
            parameter_count=(
                result.params.n_params + self._corr_effective_num_params()),
            diagnostics=diagnostics)

    def _corr_plugin_num_params(self):
        if self._corr_mode == 'factor':
            return factor_parameter_count(self._d, self._factor_rank)
        preprocessing = self._corr_preprocessing
        if (
                self._corr_mode == 'fixed'
                and preprocessing is not None
                and preprocessing.source == 'kendall'):
            return cholesky_corr_n_params(self._d)
        return 0

    def _corr_effective_num_params(self):
        return self._corr_num_params() + self._corr_plugin_num_params()

    def _corr_count_diagnostics(self):
        n_corr = self._corr_num_params()
        plugin_n = self._corr_plugin_num_params()
        diagnostics = {
            'corr_mode': self._corr_mode,
            'corr_estimator': self.corr_estimator_,
            'corr_n_params': n_corr,
            'corr_plugin_n_params': plugin_n,
            'corr_effective_n_params': n_corr + plugin_n,
        }
        if self._corr_mode == 'factor':
            diagnostics.update(self.factor_diagnostics())
        return diagnostics

    def _default_corr_params(self):
        if self._corr_mode in {'fixed', 'factor'}:
            return np.empty(0, dtype=np.float64)
        if self._corr_mode == 'shrinkage':
            return np.array(
                [float(logit(self._corr_shrinkage_init))],
                dtype=np.float64,
            )
        base = self._corr_base if self._corr_base is not None else self._R
        if base is None:
            raise ValueError("R not set")
        return pack_cholesky_corr(base)

    def _initial_corr_params(self, u):
        self._ensure_corr_initialized(u)
        raw = self._default_corr_params()
        self._set_corr_from_params(raw)
        return raw

    def _pack_corr_params(self):
        expected = self._corr_num_params()
        raw = getattr(self, "_corr_params_raw", None)
        if raw is not None:
            raw = np.asarray(raw, dtype=np.float64).reshape(-1)
            if raw.size == expected:
                return raw.copy()
        return self._default_corr_params()

    def _set_corr_from_params(self, params):
        params = np.asarray(params, dtype=np.float64).reshape(-1)
        expected = self._corr_num_params()
        if params.size != expected:
            raise ValueError(
                f"expected {expected} correlation parameters, "
                f"got {params.size}")
        if self._corr_mode in {'fixed', 'factor'}:
            self._corr_params_raw = np.empty(0, dtype=np.float64)
            self._corr_alpha = None
            return
        if self._corr_mode == 'shrinkage':
            if self._corr_base is None:
                raise ValueError("corr_base not initialized")
            R = _make_shrinkage_corr_from_validated(
                float(params[0]), self._corr_base)
            self._set_generated_R(R)
            self._corr_params_raw = params.copy()
            self._corr_alpha = float(sigmoid(params[0]))
            return
        R = _corr_from_cholesky_params(params, self._d)
        self._set_generated_R(R)
        self._corr_params_raw = params.copy()
        self._corr_alpha = None

    def _snapshot_corr_state(self):
        """Capture mutable correlation state for later restore."""
        return (
            self._R,
            self._L,
            self._L_inv,
            self._log_det,
            self._corr_cache_version,
            self._corr_params_raw,
            self._corr_alpha,
        )

    def _restore_corr_state(self, state):
        (self._R, self._L, self._L_inv, self._log_det,
         self._corr_cache_version,
         self._corr_params_raw, self._corr_alpha) = state

    def _split_joint_params(self, params):
        params = np.asarray(params, dtype=np.float64).reshape(-1)
        n_corr = self._corr_num_params()
        if params.size < n_corr:
            raise ValueError("parameter vector is shorter than corr params")
        if n_corr == 0:
            return params, np.empty(0, dtype=np.float64)
        return params[:-n_corr], params[-n_corr:]

    def corr_params(self):
        if (
                self._corr_mode == "factor"
                and self._factor_estimation == "joint"):
            return np.asarray(
                self._corr_params_raw, dtype=np.float64).reshape(-1).copy()
        return self._pack_corr_params()

    def corr_alpha(self):
        return self._corr_alpha

    # ── Transform: x -> df ──────────────────────────────────
    # Psi(x) = 2 + 1e-6 + softplus(x), ensuring finite variance.

    def transform(self, x):
        """Map latent values to degrees of freedom above the finite-variance bound."""
        from pyscarcopula._native import multivariate as multivariate_native
        return multivariate_native.transform(self, x)

    def transform_scalar(self, x):
        return float(self.transform(np.array([x], dtype=np.float64))[0])

    def inv_transform(self, df):
        """Map degrees of freedom above the model offset to latent values."""
        from pyscarcopula._native import multivariate as multivariate_native
        return multivariate_native.inverse_transform(self, df)

    def dtransform(self, x):
        """d(Psi)/dx = sigmoid(x)."""
        from pyscarcopula._native import multivariate as multivariate_native
        return multivariate_native.dtransform(self, x)

    def dtransform_scalar(self, x):
        return float(self.dtransform(np.array([x], dtype=np.float64))[0])

    @model_state_locked
    def prepare_emission_cache(self, u):
        """Return the reusable full-sample Student PPF cache."""
        if self._corr_mode == 'factor':
            if self._factor_operator is None:
                raise ValueError("factor correlation is not initialized")
            values = np.asarray(u)
            if values.ndim != 2 or values.shape[1] != self._d:
                raise ValueError(
                    f"u must have shape (T, {self._d})")
            # The factor grid evaluates exact PPFs in bounded tiles and
            # intentionally has no O(T*K*d) full-sample cache.
            return None
        if self._R is None:
            raise ValueError("R not set")
        source = u
        self._ppf_cache = prepare_student_ppf_cache(
            self._ppf_cache,
            source,
            u,
            self._d,
            table_factory=_PPFTable,
        )
        return self._ppf_cache

    # ── Density ──────────────────────────────────────────────

    @model_state_locked
    def log_likelihood(self, u, r=None, *, n_threads=1):
        """
        Log-likelihood for d-dimensional data.

        u : (T, d) pseudo-observations
        r : float (df) or None — if None, evaluates the fitted strategy
        """
        if r is None:
            return self._fitted_log_likelihood(u, n_threads=n_threads)
        if (
                self._corr_mode == 'factor'
                and self._factor_operator is None):
            raise ValueError(
                "Factor correlation not set. Call initialize_factor() first.")
        if self._corr_mode != 'factor' and self._R is None:
            raise ValueError("Correlation matrix R not set. Call fit() first.")

        u = as_float64_array(u, name="u")

        if self._corr_mode == 'factor':
            return FactorStudentEvaluator(
                self._factor_operator, u).evaluate(
                    float(r), n_threads=n_threads).log_likelihood

        from pyscarcopula._native import static as static_likelihood
        return static_likelihood.prepare(
            self, u, n_threads=n_threads).log_likelihood(float(r))

    def log_pdf_rows(
            self, u, r, t_index=None, cache=None, *, n_threads=1):
        """Return one log-density per row for scalar/row-wise df values."""
        if self._corr_mode == 'factor':
            if self._factor_operator is None:
                raise ValueError(
                    "Factor correlation not set. "
                    "Call initialize_factor() first.")
            return FactorStudentEvaluator(
                self._factor_operator, u).log_pdf_rows(
                    r, n_threads=n_threads)
        if self._R is None:
            raise ValueError("Correlation matrix R not set. Call fit() first.")
        from pyscarcopula._native import multivariate as multivariate_native
        values, _ = multivariate_native.log_pdf_and_dlog_rows(
            self, u, r, t_index=t_index, cache=cache,
            n_threads=n_threads)
        return values

    def dlog_pdf_dr_rows(
            self, u, r, t_index=None, cache=None, *, n_threads=1):
        """Return d log c(u_t; df_t) / d df_t for each row."""
        if self._corr_mode == 'factor':
            if self._factor_operator is None:
                raise ValueError(
                    "Factor correlation not set. "
                    "Call initialize_factor() first.")
            return FactorStudentEvaluator(
                self._factor_operator, u).dlog_pdf_ddf_rows(
                    r, n_threads=n_threads)
        if self._R is None:
            raise ValueError("Correlation matrix R not set. Call fit() first.")

        from pyscarcopula._native import multivariate as multivariate_native
        _, values = multivariate_native.log_pdf_and_dlog_rows(
            self, u, r, t_index=t_index, cache=cache,
            n_threads=n_threads)
        return values

    def log_pdf_and_dlog_dr_rows(
            self, u, r, t_index=None, cache=None, *, n_threads=1):
        """Return per-row log-density and d log c(u_t; df_t) / d df_t."""
        if self._corr_mode == 'factor':
            if self._factor_operator is None:
                raise ValueError(
                    "Factor correlation not set. "
                    "Call initialize_factor() first.")
            return FactorStudentEvaluator(
                self._factor_operator, u).log_pdf_and_dlog_ddf_rows(
                    r, n_threads=n_threads)
        if self._R is None:
            raise ValueError("Correlation matrix R not set. Call fit() first.")

        from pyscarcopula._native import multivariate as multivariate_native
        return multivariate_native.log_pdf_and_dlog_rows(
            self, u, r, t_index=t_index, cache=cache,
            n_threads=n_threads)

    def pdf_on_grid(self, u_row, z_grid, *, n_threads=1):
        """Copula density on latent grid for one observation.

        u_row : (d,) single observation
        z_grid : (K,) latent grid values

        Returns: (K,) copula densities c(u_row; R, Psi(z_j))
        """
        if (
                self._corr_mode == 'factor'
                and self._factor_operator is None):
            raise ValueError("factor correlation is not initialized")
        if self._corr_mode != 'factor' and self._R is None:
            raise ValueError("R not set")

        if self._corr_mode == 'factor':
            fi, _ = self.pdf_and_grad_on_grid_batch(
                np.asarray(u_row, dtype=np.float64)[None, :],
                z_grid,
                n_threads=n_threads,
            )
            return fi[0]

        from pyscarcopula._native import multivariate as multivariate_native
        fi, _ = multivariate_native.pdf_and_grad_grid(
            self,
            np.asarray(u_row, dtype=np.float64)[None, :],
            z_grid,
            n_threads=n_threads,
        )
        return fi[0]

    def pdf_and_grad_on_grid(self, u_row, z_grid, *, n_threads=1):
        """
        Compute fi(z) and dfi/dz on the grid analytically.

        fi(z) = c(u_row; R, Psi(z))
        dfi/dz = fi * d(log c)/d(df) * d(Psi)/dz

        u_row : (d,), z_grid : (K,)
        Returns: (fi, dfi_dz) each of shape (K,)
        """
        if (
                self._corr_mode == 'factor'
                and self._factor_operator is None):
            raise ValueError("factor correlation is not initialized")
        if self._corr_mode != 'factor' and self._R is None:
            raise ValueError("R not set")

        if self._corr_mode == 'factor':
            fi, dfi = self.pdf_and_grad_on_grid_batch(
                np.asarray(u_row, dtype=np.float64)[None, :],
                z_grid,
                n_threads=n_threads,
            )
            return fi[0], dfi[0]

        from pyscarcopula._native import multivariate as multivariate_native
        fi, dfi = multivariate_native.pdf_and_grad_grid(
            self,
            np.asarray(u_row, dtype=np.float64)[None, :],
            z_grid,
            n_threads=n_threads,
        )
        return fi[0], dfi[0]

    def pdf_and_grad_on_grid_batch(
            self, u, x_grid, t_index=0, cache=None, *, n_threads=1):
        """
        Batch evaluation for all T observations.

        Uses a reusable PPF table when it fits the memory budget. Native
        out-of-table evaluation uses analytical df derivatives and a
        controlled normal-quantile asymptotic above the final df node.

        u : (T, d) pseudo-observations
        x_grid : (K,) latent grid values
        n_threads : int, keyword-only
            Maximum native worker count. ``1`` preserves the sequential
            fast path; small batches stay sequential for any value.

        Returns: (fi, dfi) each (T, K)
        """
        if (
                self._corr_mode == 'factor'
                and self._factor_operator is None):
            raise ValueError("factor correlation is not initialized")
        if self._corr_mode != 'factor' and self._R is None:
            raise ValueError("R not set")

        if self._corr_mode == 'factor':
            evaluator = FactorStudentEvaluator(
                self._factor_operator, u)
            return evaluator.stochastic_pdf_and_gradient_grid(
                x_grid,
                offset=self._df_offset,
                dimension_tile=self._factor_tile_size,
                n_threads=n_threads,
            )

        from pyscarcopula._native import multivariate as multivariate_native
        return multivariate_native.pdf_and_grad_grid(
            self,
            u,
            x_grid,
            t_index=t_index,
            cache=cache,
            n_threads=n_threads,
        )

    def copula_grid_batch(
            self, u, x_grid, t_index=0, cache=None, *, n_threads=1):
        """Batch version of pdf_on_grid (value only)."""
        if (
                self._corr_mode == 'factor'
                and self._factor_operator is None):
            raise ValueError("factor correlation is not initialized")
        if self._corr_mode != 'factor' and self._R is None:
            raise ValueError("R not set")

        if self._corr_mode == 'factor':
            fi, _ = self.pdf_and_grad_on_grid_batch(
                u,
                x_grid,
                t_index=t_index,
                cache=cache,
                n_threads=n_threads,
            )
            return fi

        from pyscarcopula._native import multivariate as multivariate_native
        fi, _ = multivariate_native.pdf_and_grad_grid(
            self,
            u,
            x_grid,
            t_index=t_index,
            cache=cache,
            n_threads=n_threads,
        )
        return fi

    # ── MLE fit ──────────────────────────────────────────────

    def _fit_joint_factor_mle_shared(
            self, u, config, optimizer_options, loadings, initialization):
        """Joint factor adapter using the common optimizer/acceptance gate."""
        from pyscarcopula._types import MultivariateMLEResult
        from pyscarcopula.strategy.multivariate_mle import (
            StaticMLEEvaluation,
            StaticMLEProblem,
            run_static_multivariate_mle,
        )

        optimizer_options = dict(optimizer_options)
        optimizer_options.setdefault("ftol", 1e-10)
        # The loading pullback can be steep near the uniqueness boundary.
        # A longer line search prevents platform-dependent ABNORMAL
        # terminations without overriding a larger caller-supplied value.
        optimizer_options["maxls"] = max(
            int(optimizer_options.get("maxls", 20)), 100)
        parameterization, factor0 = (
            FactorLoadingParameterization.from_loadings(
                loadings,
                uniqueness_min=self._factor_uniqueness_min,
            ))
        expected = parameterization.n_parameters
        if expected > self._factor_joint_max_params:
            raise ValueError(
                "joint factor estimation exceeds factor_joint_max_params")
        penalty = self._factor_joint_penalty
        condition_max = self._factor_joint_condition_max
        policy = CorrelationPolicy.create(
            mode="factor",
            estimator="factor_joint",
            dimension=self._d,
            factor_rank=self._factor_rank,
            factor_estimation="joint",
        )
        factor_evaluator = FactorStudentEvaluator(
            FactorCorrelation(
                loadings,
                uniqueness_min=self._factor_uniqueness_min,
                diagnostics={"source": "joint_static_mle_initial"},
            ),
            u,
        )

        def evaluate(parameters):
            df = float(parameters[0])
            native = (
                factor_evaluator
                .penalized_parameterized_objective_and_gradient(
                    df,
                    parameters[1:],
                    parameterization,
                    penalty=penalty,
                    condition_max=condition_max,
                    n_threads=config.n_threads,
                ))
            native_diagnostics = dict(native.diagnostics)
            return StaticMLEEvaluation(
                objective=native.objective,
                gradient=native.gradient,
                state={
                    "df": df,
                    "loadings": native.loadings.copy(),
                    "log_likelihood": float(native.log_likelihood),
                    "native_diagnostics": native_diagnostics,
                    "operator_diagnostics": native_diagnostics,
                },
            )

        outcome = run_static_multivariate_mle(
            StaticMLEProblem(
                family="student_factor",
                initial_parameters=np.concatenate([
                    np.array([_STUDENT_FIT_INITIAL], dtype=np.float64), factor0]),
                bounds=(_STUDENT_FIT_BOUNDS,)
                + ((None, None),) * expected,
                evaluate=evaluate,
            ),
            optimizer_options=optimizer_options,
            fail_value=config.fail_value,
        )
        state = (
            None if outcome.evaluation is None
            else outcome.evaluation.state)
        final_loadings = (
            np.asarray(loadings, dtype=np.float64).copy()
            if state is None
            else np.asarray(state["loadings"], dtype=np.float64).copy())
        df_hat = float(outcome.parameters[0])
        corr_raw = outcome.parameters[1:].copy()
        log_likelihood = (
            -outcome.final_objective
            if state is None else float(state["log_likelihood"]))
        message = outcome.message
        if (
                outcome.optimizer_success
                and not outcome.accepted
                and outcome.final_gradient_inf_norm > outcome.gradient_gate):
            message += "; rejected by joint factor gradient gate"
        diagnostics = {
            "n_threads": config.n_threads,
            "parameterization": "natural_df_and_triangular_factor",
            "gradient_mode": "analytical_joint_factor",
            "model_score": "not_applicable",
            "optimizer_gradient": "analytical",
            "gradient_kind": "analytical",
            "df_gradient": "analytical",
            "correlation_gradient": "analytical_factor",
            "joint_static": True,
            "corr_params_raw": corr_raw.copy(),
            "corr_alpha": None,
            "joint_factor": True,
            "joint_dynamic_supported": False,
            "joint_identification": (
                "pivoted_lower_triangular_positive_diag"),
            "joint_anchor_rows": parameterization.anchors.copy(),
            "joint_penalty": penalty,
            "joint_condition_max": condition_max,
            "joint_initial_objective": outcome.initial_objective,
            "joint_final_objective": outcome.final_objective,
            "joint_gradient_inf_norm": outcome.final_gradient_inf_norm,
            "joint_gradient_gate": outcome.gradient_gate,
            "joint_evaluations": outcome.evaluations,
            "joint_native_diagnostics": (
                {} if state is None else state["native_diagnostics"]),
            "joint_operator_diagnostics": (
                {} if state is None else state["operator_diagnostics"]),
            "factor_rank": self._factor_rank,
            "factor_estimation": "joint",
            "factor_n_params": policy.effective_n_params,
            "factor_initialized": state is not None,
            "representation": "factor_woodbury",
            **{
                f"initialization_{key}": value
                for key, value in initialization.items()
            },
            "initialization_joint_start_source": initialization.get(
                "source", "supplied"),
            **policy.diagnostics(),
            **outcome.diagnostics(),
        }
        result = MultivariateMLEResult(
            log_likelihood=log_likelihood,
            method="MLE",
            copula_name=self._name,
            success=outcome.accepted,
            nfev=outcome.nfev,
            message=message,
            copula_param=df_hat,
            parameter_count=1 + policy.effective_n_params,
            n_observations=len(u),
            model_parameters={
                "df": df_hat,
                "corr_mode": "factor",
                "corr_estimator": "factor_joint",
                "corr_params_raw": corr_raw.copy(),
                "corr_alpha": None,
                "factor_rank": self._factor_rank,
                "factor_loadings": final_loadings.copy(),
                "factor_uniqueness": FactorCorrelation(
                    final_loadings,
                    uniqueness_min=self._factor_uniqueness_min,
                ).uniqueness.copy(),
                "factor_estimation": "joint",
                "factor_anchor_rows": parameterization.anchors.copy(),
            },
            correlation_matrix=None,
            diagnostics=diagnostics,
        )
        if outcome.accepted:
            self._set_factor_loadings(
                final_loadings,
                diagnostics={
                    "source": "joint_static_mle",
                    "joint_start_source": initialization.get(
                        "source", "supplied"),
                    "joint_anchor_rows": parameterization.anchors.tolist(),
                    "joint_penalty": penalty,
                    "joint_condition_max": condition_max,
                },
            )
            self._corr_params_raw = corr_raw.copy()
            self._corr_alpha = None
            self.fit_result = result
            self._last_u = u.copy()
        return result

    def _fit_mle_shared(
            self, u, config: NumericalConfig, optimizer_options):
        """Build and run an unpublished static Student candidate."""
        from pyscarcopula.strategy.multivariate_mle import (
            StaticMLEEvaluation,
            StaticMLEProblem,
            make_student_static_mle_evaluator,
            run_static_multivariate_mle,
        )

        if self._corr_mode == "factor":
            if self._constructor_factor_loadings is None:
                loadings, initialization = estimate_factor_loadings(
                    u,
                    self._factor_rank,
                    uniqueness_min=self._factor_uniqueness_min,
                    dimension_tile=self._factor_tile_size,
                    seed=self._factor_seed,
                    oversampling=self._factor_oversampling,
                )
                if self._factor_estimation == "joint":
                    initialization = {
                        **initialization,
                        "source": "joint_randomized_svd_start",
                    }
            else:
                loadings = self._constructor_factor_loadings.copy()
                initialization = {"source": "supplied"}
            if self._factor_estimation == "joint":
                return self._fit_joint_factor_mle_shared(
                    u,
                    config,
                    optimizer_options,
                    loadings,
                    initialization,
                )

            factor = FactorCorrelation(
                loadings,
                uniqueness_min=self._factor_uniqueness_min,
                diagnostics=initialization,
            )
            operator = factor.prepare()
            factor_evaluator = FactorStudentEvaluator(operator, u)
            policy = CorrelationPolicy.create(
                mode="factor",
                estimator="factor_two_stage",
                dimension=self._d,
                factor_rank=self._factor_rank,
                factor_estimation="two-stage",
            )

            def evaluate(parameters):
                value, gradient = factor_evaluator.objective_and_gradient(
                    float(parameters[0]), n_threads=config.n_threads)
                return StaticMLEEvaluation(
                    objective=value,
                    gradient=gradient,
                    state={"df": float(parameters[0])},
                )

            outcome = run_static_multivariate_mle(
                StaticMLEProblem(
                    family="student",
                    initial_parameters=np.array([_STUDENT_FIT_INITIAL]),
                    bounds=(_STUDENT_FIT_BOUNDS,),
                    evaluate=evaluate,
                ),
                optimizer_options=optimizer_options,
                fail_value=config.fail_value,
            )
            correlation = None
            preprocessing = None
            corr_raw = np.empty(0, dtype=np.float64)
            corr_alpha = None
            candidate_factor = factor
        else:
            if self._constructor_corr_base is not None:
                initial_correlation = self._constructor_corr_base.copy()
                preprocessing = self._corr_base_preprocessing
            elif self._constructor_R is not None:
                initial_correlation = self._constructor_R.copy()
                preprocessing = self._corr_preprocessing
                if preprocessing is None:
                    preprocessing = preprocess_correlation_matrix(
                        initial_correlation, source="supplied")
            else:
                preprocessing = estimate_kendall_correlation(u, eps=1e-8)
                initial_correlation = preprocessing.correlation.copy()

            estimator = (
                "joint_mle"
                if self._corr_mode in {"shrinkage", "cholesky"}
                else (
                    "supplied"
                    if preprocessing.source in {"supplied", "corr_base"}
                    else "kendall_plugin"))
            policy = CorrelationPolicy.create(
                mode=self._corr_mode,
                estimator=estimator,
                dimension=self._d,
                supplied_correlation=(
                    initial_correlation if estimator == "supplied" else None),
                base_correlation=(
                    initial_correlation
                    if self._corr_mode in {"shrinkage", "cholesky"}
                    else None),
                preprocessing=preprocessing,
                shrinkage_initial=self._corr_shrinkage_init,
            )
            corr0 = policy.initial_raw_parameters()
            n_corr = policy.optimized_n_params
            evaluate = make_student_static_mle_evaluator(
                initial_correlation,
                policy,
                u,
                n_threads=config.n_threads,
                fail_value=config.fail_value,
            )

            outcome = run_static_multivariate_mle(
                StaticMLEProblem(
                    family="student",
                    initial_parameters=np.concatenate([
                        np.array([_STUDENT_FIT_INITIAL]), corr0]),
                    bounds=(_STUDENT_FIT_BOUNDS,)
                    + ((None, None),) * n_corr,
                    evaluate=evaluate,
                ),
                optimizer_options=optimizer_options,
                fail_value=config.fail_value,
            )
            correlation = (
                initial_correlation.copy()
                if outcome.evaluation is None
                else np.asarray(
                    outcome.evaluation.correlation,
                    dtype=np.float64).copy())
            corr_raw = outcome.parameters[1:].copy()
            corr_alpha = (
                float(sigmoid(corr_raw[0]))
                if self._corr_mode == "shrinkage" and corr_raw.size
                else None)
            candidate_factor = None

        df_hat = float(outcome.parameters[0])
        n_corr = policy.optimized_n_params
        diagnostics = {
            "n_threads": config.n_threads,
            "parameterization": "natural_df",
            "gradient_mode": (
                "analytical_df" if n_corr == 0 else "analytical_joint"),
            "model_score": "not_applicable",
            "optimizer_gradient": "analytical",
            "gradient_kind": "analytical",
            "setup_derivative": "not_applicable",
            "filter_derivative": "not_applicable",
            "df_gradient": "analytical",
            "correlation_gradient": (
                "not_applicable" if n_corr == 0 else "analytical"),
            **policy.diagnostics(),
            **outcome.diagnostics(),
            "corr_params_raw": corr_raw.copy(),
            "corr_alpha": corr_alpha,
        }
        if candidate_factor is not None:
            diagnostics.update({
                "factor_rank": self._factor_rank,
                "factor_estimation": self._factor_estimation,
                "factor_n_params": policy.effective_n_params,
                "factor_initialized": True,
                "representation": "factor_woodbury",
                **dict(candidate_factor.prepare().diagnostics),
                **{
                    f"initialization_{key}": value
                    for key, value in initialization.items()
                },
            })
        if preprocessing is not None:
            diagnostics.update(preprocessing.diagnostics())
        if correlation is not None:
            diagnostics["corr_matrix"] = correlation.copy()

        model_parameters = {
            "df": df_hat,
            "corr_mode": self._corr_mode,
            "corr_estimator": policy.estimator,
            "corr_params_raw": corr_raw.copy(),
            "corr_alpha": corr_alpha,
        }
        if candidate_factor is not None:
            model_parameters.update({
                "factor_rank": self._factor_rank,
                "factor_loadings": candidate_factor.loadings.copy(),
                "factor_uniqueness": candidate_factor.uniqueness.copy(),
                "factor_estimation": self._factor_estimation,
            })
        else:
            model_parameters["correlation_matrix"] = correlation.copy()

        from pyscarcopula._types import MultivariateMLEResult
        result = MultivariateMLEResult(
            log_likelihood=-outcome.final_objective,
            method="MLE",
            copula_name=self._name,
            success=outcome.accepted,
            nfev=outcome.nfev,
            message=outcome.message,
            copula_param=df_hat,
            parameter_count=1 + policy.effective_n_params,
            n_observations=len(u),
            model_parameters=model_parameters,
            correlation_matrix=(
                None if candidate_factor is not None else correlation.copy()),
            diagnostics=diagnostics,
        )
        if outcome.accepted:
            if candidate_factor is not None:
                self._set_factor_loadings(
                    candidate_factor.loadings, diagnostics=initialization)
            else:
                self._set_generated_R(correlation)
                self._corr_preprocessing = preprocessing
                if self._corr_mode in {"shrinkage", "cholesky"}:
                    self._corr_base = initial_correlation.copy()
                self._corr_params_raw = corr_raw.copy()
                self._corr_alpha = corr_alpha
            self.fit_result = result
            self._last_u = u.copy()
        return result

    def _fit_mle(self, u, config: NumericalConfig | None = None,
                 gtol=None, ftol=None, maxfun=None, maxiter=None,
                 maxls=None, eps=None, maxcor=None,
                 finite_diff_rel_step=None):
        """Fit constant natural ``df`` and optional correlation parameters.

        Unlike dynamic SCAR/GAS fitting, static MLE has no latent state and
        therefore optimizes degrees of freedom directly without ``transform``.
        """
        config = config or DEFAULT_CONFIG
        optimizer_options = config.stochastic_student_optimizer.options(
            gtol=gtol,
            ftol=ftol,
            maxfun=maxfun,
            maxiter=maxiter,
            maxls=maxls,
            eps=eps,
            maxcor=maxcor,
            finite_diff_rel_step=finite_diff_rel_step,
        )

        return self._fit_mle_shared(u, config, optimizer_options)
    def correlation_preprocessing_diagnostics(self):
        """Return diagnostics for the correlation used to initialize fitting."""
        if self._corr_preprocessing is None:
            return {}
        return self._corr_preprocessing.diagnostics()

    # ── Fit (MLE + SCAR) ────────────────────────────────────

    @model_state_locked
    def fit(
            self,
            data: ArrayLike,
            method: str = 'scar-tm-ou',
            to_pobs: bool = False,
            config: NumericalConfig | None = None,
            **kwargs: Any) -> FitResultBase:
        """
        Fit the stochastic Student-t copula.

        Step 1: Initialize dense R or compact factor loadings if needed.
        Step 2: Estimate OU params for df(t) via the chosen method.

        Parameters
        ----------
        data : (T, d) array
        method : str — 'mle', 'scar-tm-ou', 'gas', etc.
        to_pobs : bool
        **kwargs : forwarded to strategy

        Returns
        -------
        FitResult
        """
        from pyscarcopula.strategy._base import (
            partition_strategy_fit_kwargs,
        )
        partition_strategy_fit_kwargs(method, kwargs)

        u = _as_float64_array_no_copy(data)
        _validate_fit_data(u, self._d)
        if to_pobs:
            u = pobs(u)
            _validate_fit_data(u, self._d)

        from pyscarcopula.strategy._base import ensure_strategy_supported
        ensure_strategy_supported(self, method)
        if (
                self._corr_mode == "factor"
                and self._factor_estimation == "joint"
                and method.upper() != "MLE"):
            raise NotImplementedError(
                "joint factor estimation is currently supported only for "
                "static MLE; GAS/SCAR require loading gradients through "
                "their sequential filters")

        optimizer_kwargs = {}
        if method.upper() == 'MLE':
            optimizer_kwargs = {
                key: kwargs.pop(key)
                for key in _LBFGSB_FIT_KEYS
                if key in kwargs
            }
            if kwargs:
                unexpected = ", ".join(sorted(kwargs))
                raise TypeError(
                    f"unexpected MLE keyword argument(s): {unexpected}")

        if method.upper() == 'MLE':
            return self._fit_mle(u, config=config, **optimizer_kwargs)

        from pyscarcopula.api import fit as _api_fit
        return _api_fit(self, u, method=method, config=config, **kwargs)

    # ── Sampling ─────────────────────────────────────────────

    def _factor_sampling_peak_bytes(self, rows, *, conditional=False):
        rows = int(rows)
        # Conservative peak: native draw buffers, pybind result ownership,
        # latent arrays, CDF output, and conditional sufficient statistics can
        # coexist briefly even though steady-state storage is smaller.
        multiplier = (
            8 * self._d + 6 * self._factor_rank + 16
            if conditional
            else 6 * self._d + 3 * self._factor_rank + 12
        )
        small = (
            3 * self._factor_rank * self._factor_rank * 8
            if conditional
            else 0
        )
        return rows * multiplier * 8 + small

    def sample_at_parameter(
            self,
            n,
            r,
            rng=None,
            *,
            n_threads=1,
            memory_budget_bytes=None):
        """
        Sample from d-dimensional Student-t copula.

        Parameters
        ----------
        n : int — number of observations
        r : float, (n,) array, or None — degrees of freedom.
            If scalar: all samples use same df.
            If array of length n: each sample uses its own df(t).
        rng : np.random.Generator or None

        Returns
        -------
        (n, d) pseudo-observations in [0, 1]^d
        """
        n = validate_integer(n, "n")
        n_threads = _sampling_n_threads(n_threads)
        r_arr = _validated_student_sampling_parameters(r, n)
        _sampling_memory_budget(
            memory_budget_bytes, n * self._d * 8,
            "use sample_at_parameter_batches() or increase memory_budget_bytes")
        if rng is None:
            rng = np.random.default_rng()
        from pyscarcopula._native import multivariate as multivariate_native
        if self._corr_mode == 'factor':
            if self._factor_operator is None:
                raise ValueError(
                    "factor correlation is not initialized; call fit() or "
                    "initialize_factor() first")
            _sampling_memory_budget(
                memory_budget_bytes,
                self._factor_sampling_peak_bytes(n),
                "use sample_at_parameter_batches(), reduce batch_rows, "
                "or increase memory_budget_bytes",
            )
            df_path = r_arr
            if df_path.size == 1:
                df_path = np.full(n, df_path[0], dtype=np.float64)
            elif df_path.size != n:
                raise ValueError(
                    f"r must be scalar or array of length {n}, "
                    f"got {df_path.size}")
            factor_draws = rng.standard_normal(
                (n, self._factor_operator.rank))
            residual_draws = rng.standard_normal((n, self._d))
            chi_square_uniforms = rng.uniform(0.0, 1.0, size=n)
            return multivariate_native.factor_student_sample_from_normal_uniforms(
                self._factor_operator,
                df_path,
                factor_draws,
                residual_draws,
                chi_square_uniforms,
                n_threads=n_threads,
            )
        if self._R is None:
            raise ValueError("Correlation matrix R not set. Call fit() first.")

        is_scalar = (r_arr.size == 1)
        normal_draws = rng.standard_normal((n, self._d))
        chi_square_uniforms = rng.uniform(0.0, 1.0, size=n)
        if not is_scalar and len(r_arr) != n:
            raise ValueError(
                f"r must be scalar or array of length {n}, got {len(r_arr)}")

        return multivariate_native.student_sample_from_normal_uniforms(
            self._R,
            r_arr,
            normal_draws,
            chi_square_uniforms,
            n_threads=n_threads,
        )

    def sample_at_parameter_batches(
            self,
            n,
            r,
            *,
            batch_rows=128,
            rng=None,
            n_threads=1,
            memory_budget_bytes=None):
        """Yield Student samples in bounded row blocks."""
        n = validate_integer(n, "n")
        batch_rows = validate_integer(batch_rows, "batch_rows", minimum=1)
        n_threads = _sampling_n_threads(n_threads)
        parameters = np.atleast_1d(
            np.asarray(r, dtype=np.float64)).ravel()
        if parameters.size not in (1, n):
            raise ValueError(
                f"r must be scalar or array of length {n}, "
                f"got {parameters.size}")
        if (
                not np.all(np.isfinite(parameters))
                or np.any(parameters <= 2.0)):
            raise ValueError("r must be finite and greater than 2")
        if rng is None:
            rng = np.random.default_rng()
        if self._corr_mode == "factor":
            _sampling_memory_budget(
                memory_budget_bytes,
                self._factor_sampling_peak_bytes(
                    min(n, batch_rows)),
                "reduce batch_rows or increase memory_budget_bytes",
            )

        for start in range(0, n, batch_rows):
            stop = min(n, start + batch_rows)
            block_parameters = (
                parameters
                if parameters.size == 1
                else parameters[start:stop]
            )
            yield self.sample_at_parameter(
                stop - start,
                block_parameters,
                rng=rng,
                n_threads=n_threads,
                memory_budget_bytes=memory_budget_bytes,
            )

    @model_state_locked
    def sample(
            self,
            n,
            u=None,
            rng=None,
            *,
            n_threads=1,
            memory_budget_bytes=None):
        """Generate observations reproducing the fitted model."""
        if self.fit_result is None:
            raise ValueError("Fit first")
        n = validate_integer(n, "n")
        n_threads = _sampling_n_threads(n_threads)
        _sampling_memory_budget(
            memory_budget_bytes, n * self._d * 8,
            "use sample_batches() or increase memory_budget_bytes")
        if self._corr_mode == "factor":
            blocks = self.sample_batches(
                n,
                u=u,
                rng=rng,
                batch_rows=max(1, n),
                n_threads=n_threads,
                memory_budget_bytes=memory_budget_bytes,
            )
            try:
                return next(blocks)
            except StopIteration:
                return np.empty((0, self._d), dtype=np.float64)
        from pyscarcopula.api import sample as _api_sample

        u_data = u if u is not None else getattr(self, "_last_u", None)
        if u_data is None:
            raise ValueError(
                "No data for sample. "
                "Either call fit() first or pass u= explicitly.")
        return _api_sample(
            self, u_data, self.fit_result, n, rng=rng,
            n_threads=n_threads, memory_budget_bytes=memory_budget_bytes)

    @model_state_locked
    def sample_batches(
            self,
            n,
            u=None,
            rng=None,
            *,
            batch_rows=128,
            given=None,
            n_threads=1,
            memory_budget_bytes=None):
        """Yield fitted Student-model samples in bounded row blocks."""
        if self.fit_result is None:
            raise ValueError("Fit first")
        n = validate_integer(n, "n")
        batch_rows = validate_integer(batch_rows, "batch_rows", minimum=1)
        n_threads = _sampling_n_threads(n_threads)
        _sampling_memory_budget(
            memory_budget_bytes, min(n, batch_rows) * self._d * 8,
            "reduce batch_rows or increase memory_budget_bytes")
        if rng is None:
            rng = np.random.default_rng()
        given = validate_multivariate_given(given, self._d)
        if self._corr_mode == "factor":
            _sampling_memory_budget(
                memory_budget_bytes,
                self._factor_sampling_peak_bytes(
                    min(n, batch_rows),
                    conditional=bool(given),
                ),
                "reduce batch_rows or increase memory_budget_bytes",
            )

        from pyscarcopula.strategy._base import get_strategy_for_result
        result = self.fit_result
        strategy = get_strategy_for_result(result)
        state = strategy.model_sample_state(self, result)
        from pyscarcopula.strategy.predict_helpers import sample_model_batches
        return sample_model_batches(
            self, strategy, result, state, n, batch_rows=batch_rows,
            given=given, rng=rng, n_threads=n_threads,
            memory_budget_bytes=memory_budget_bytes)

    # ── Predict ──────────────────────────────────────────────

    @model_state_locked
    def sample_conditional(
            self,
            n,
            r=None,
            given=None,
            rng=None,
            *,
            n_threads=1,
            memory_budget_bytes=None):
        """Sample conditionally with ``given={var_index: u_value}``."""
        n = validate_integer(n, "n")
        n_threads = _sampling_n_threads(n_threads)
        _sampling_memory_budget(
            memory_budget_bytes,
            n * self._d * 8,
            "use sample_batches()/predict_batches(), reduce batch_rows, "
            "or increase memory_budget_bytes",
        )
        if self._corr_mode != 'factor' and self._R is None:
            raise ValueError("Correlation matrix R not set. Call fit() first.")
        if rng is None:
            rng = np.random.default_rng()
        given = validate_multivariate_given(given, self._d)
        if not given:
            if r is None:
                return self.sample(
                    n,
                    rng=rng,
                    n_threads=n_threads,
                    memory_budget_bytes=memory_budget_bytes,
                )
            return self.sample_at_parameter(
                n,
                r=r,
                rng=rng,
                n_threads=n_threads,
                memory_budget_bytes=memory_budget_bytes,
            )
        if len(given) == self._d:
            if r is not None:
                _validated_student_sampling_parameters(r, n)
            return fill_given(n, self._d, given)
        if r is None:
            return self.predict(
                n, given=given, rng=rng, n_threads=n_threads,
                memory_budget_bytes=memory_budget_bytes)
        _validated_student_sampling_parameters(r, n)
        if self._corr_mode == "factor":
            if self._factor_operator is None:
                raise ValueError(
                    "factor correlation is not initialized; call fit() or "
                    "initialize_factor() first")
            _sampling_memory_budget(
                memory_budget_bytes,
                self._factor_sampling_peak_bytes(
                    n, conditional=True),
                "use sample_batches()/predict_batches(), reduce batch_rows, "
                "or increase memory_budget_bytes",
            )
            return sample_factor_student_conditional(
                n,
                self._factor_operator,
                r,
                given=given,
                rng=rng,
                n_threads=n_threads,
            )
        return sample_student_conditional(
            n,
            self._R,
            r,
            given=given,
            rng=rng,
            n_threads=n_threads,
        )

    @model_state_locked
    def predict(
            self,
            n,
            u=None,
            rng=None,
            given=None,
            horizon='next',
            predictive_r_mode=None,
            predict_config=None,
            *,
            n_threads=1,
            memory_budget_bytes=None):
        """
        Sample n observations for next-step prediction.

        For MLE, use the fitted constant degrees of freedom. Dynamic
        strategies delegate predictive parameter generation to their common
        strategy implementation, including ``current``/``next`` timing.

        Parameters
        ----------
        n : int
        u : (T, d) or None — conditioning data.
        rng : np.random.Generator or None
        """
        from pyscarcopula.api import _resolve_predict_config
        config = _resolve_predict_config(
            predict_config, given, horizon, {
                "predictive_r_mode": predictive_r_mode,
            })
        given = config.given
        horizon = config.horizon
        predictive_r_mode = config.predictive_r_mode
        if self.fit_result is None:
            raise ValueError("Fit first")
        n = validate_integer(n, "n")
        n_threads = _sampling_n_threads(n_threads)
        if rng is None:
            rng = np.random.default_rng()
        sampling_options = {}
        if n_threads != 1:
            sampling_options["n_threads"] = n_threads
        if memory_budget_bytes is not None:
            sampling_options["memory_budget_bytes"] = memory_budget_bytes

        from pyscarcopula._types import MLEResult
        if isinstance(self.fit_result, MLEResult):
            return self.sample_conditional(
                n,
                r=self.fit_result.copula_param,
                given=given,
                rng=rng,
                **sampling_options,
            )

        if n == 0:
            validate_multivariate_given(given, self._d)
            return np.empty((0, self._d), dtype=np.float64)

        observations = u if u is not None else getattr(self, '_last_u', None)
        from pyscarcopula.strategy._base import get_strategy_for_result
        strategy = get_strategy_for_result(self.fit_result)
        parameters = strategy.predictive_params(
            self,
            observations,
            self.fit_result,
            n,
            rng=rng,
            horizon=horizon,
            predictive_r_mode=predictive_r_mode,
            memory_budget_bytes=memory_budget_bytes,
        )
        return self.sample_conditional(
            n,
            r=parameters,
            given=given,
            rng=rng,
            **sampling_options,
        )

    @model_state_locked
    def predict_batches(
            self,
            n,
            u=None,
            rng=None,
            *,
            batch_rows=128,
            given=None,
            horizon="next",
            predictive_r_mode=None,
            predict_config=None,
            n_threads=1,
            memory_budget_bytes=None):
        """Yield fitted predictive samples from one frozen state."""
        if self.fit_result is None:
            raise ValueError("Fit first")
        n = validate_integer(n, "n")
        batch_rows = validate_integer(batch_rows, "batch_rows", minimum=1)
        n_threads = _sampling_n_threads(n_threads)
        if rng is None:
            rng = np.random.default_rng()

        from pyscarcopula.api import _resolve_predict_config
        config = _resolve_predict_config(
            predict_config,
            given,
            horizon,
            {"predictive_r_mode": predictive_r_mode},
        )
        if self._corr_mode == "factor":
            _sampling_memory_budget(
                memory_budget_bytes,
                self._factor_sampling_peak_bytes(
                    min(n, batch_rows),
                    conditional=bool(config.given),
                ),
                "reduce batch_rows or increase memory_budget_bytes",
            )
        observations = u if u is not None else getattr(
            self, "_last_u", None)

        from pyscarcopula.strategy._base import get_strategy_for_result
        result = self.fit_result
        strategy = get_strategy_for_result(result)
        state = strategy.predictive_state(
            self,
            observations,
            result,
            horizon=config.horizon,
            predictive_r_mode=config.predictive_r_mode,
        )

        from pyscarcopula.strategy.predict_helpers import sample_predictive_batches
        return sample_predictive_batches(
            self, strategy, state, n, batch_rows=batch_rows,
            given=config.given, rng=rng,
            predictive_r_mode=config.predictive_r_mode,
            n_threads=n_threads, memory_budget_bytes=memory_budget_bytes)

    # Predictive mean path

    @model_state_locked
    def predictive_mean(self, u=None):
        """Return the predictive df path for the fitted strategy."""
        if self.fit_result is None:
            raise ValueError("Fit the model before calling predictive_mean")
        u_data = u if u is not None else getattr(self, '_last_u', None)
        if u_data is None:
            raise ValueError("No data. Pass u= or call fit() first.")

        from pyscarcopula.api import predictive_mean as _predictive_mean

        return _predictive_mean(self, u_data, self.fit_result)

    @model_state_locked
    def xT_distribution(self, u, K=300, grid_range=5.0):
        """Distribution of x_T on grid (for predict)."""
        if self.fit_result is None:
            raise ValueError("Fit with SCAR first")
        kappa, mu, nu_ou = self.fit_result.params.values
        from pyscarcopula._native import scar_ou as _cpp_scar_ou
        from pyscarcopula.numerical._scar_ou_config import AutoTMConfig
        return _cpp_scar_ou.state_distribution(
            kappa,
            mu,
            nu_ou,
            u,
            self,
            AutoTMConfig(K=K, grid_range=grid_range),
        )

    @model_state_locked
    def posterior_state_weights(
            self, u, params=None, *, K=None, grid_range=None,
            grid_method=None, adaptive=None, pts_per_sigma=None,
            transition_method='matrix', max_K=None, r_gh=3.0, gh_order=5):
        """Return ``P(x_t = grid_i | u_1:T)`` on the TM grid."""
        u = np.asarray(u, dtype=np.float64)
        if u.ndim != 2 or u.shape[1] != self._d:
            raise ValueError(
                f"u must have shape (T, {self._d}), got {u.shape}")
        if len(u) < 2:
            raise ValueError("u must contain at least two observations")
        if not np.all(np.isfinite(u)):
            raise ValueError("u must contain only finite values")

        if params is None:
            if self.fit_result is None:
                raise ValueError("Fit first or pass params")
            fit_result = self.fit_result
            if getattr(fit_result, "params", None) is None:
                fit_result = getattr(self, "_last_latent_result", None)
            if getattr(fit_result, "params", None) is None:
                raise ValueError(
                    "posterior_state_weights requires latent SCAR/GAS "
                    "parameters; call fit(..., method='scar-tm-ou') or "
                    "pass params= explicitly"
                )
            params = fit_result.params.values
        params = np.asarray(params, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(params)):
            raise ValueError("params must contain only finite values")

        self._ensure_corr_initialized(u)
        n_corr = self._corr_num_params()
        joint_size = 3 + n_corr
        corr_state = None
        if params.size == 3:
            latent_params = params
        elif n_corr and params.size == joint_size:
            latent_params, corr_params = self._split_joint_params(params)
            # Temporarily switch the correlation to the passed joint params;
            # this is a query method and must not mutate the model.
            corr_state = self._snapshot_corr_state()
            self._set_corr_from_params(corr_params)
        else:
            expected = "3" if n_corr == 0 else f"3 or {joint_size}"
            raise ValueError(
                f"params must contain {expected} values for "
                f"corr_mode={self._corr_mode!r}, got {params.size}")

        try:
            return self._posterior_state_weights_tm(
                u, latent_params, K=K, grid_range=grid_range,
                grid_method=grid_method, adaptive=adaptive,
                pts_per_sigma=pts_per_sigma,
                transition_method=transition_method, max_K=max_K,
                r_gh=r_gh, gh_order=gh_order)
        finally:
            if corr_state is not None:
                self._restore_corr_state(corr_state)

    def _posterior_state_weights_tm(
            self, u, latent_params, *, K, grid_range, grid_method,
            adaptive, pts_per_sigma, transition_method, max_K,
            r_gh, gh_order):
        """Native C++ forward-backward sweep for posterior state weights."""

        config = DEFAULT_CONFIG
        K = config.default_K if K is None else int(K)
        grid_range = (
            config.default_grid_range if grid_range is None
            else float(grid_range))
        grid_method = (
            config.default_grid_method if grid_method is None
            else grid_method)
        adaptive = (
            config.default_adaptive if adaptive is None
            else bool(adaptive))
        pts_per_sigma = (
            config.default_pts_per_sigma if pts_per_sigma is None
            else int(pts_per_sigma))

        from pyscarcopula._native import scar_ou as _cpp_scar_ou
        from pyscarcopula.numerical._scar_ou_config import AutoTMConfig
        _, weights = _cpp_scar_ou.smoothed_state_distribution(
            latent_params[0],
            latent_params[1],
            latent_params[2],
            u,
            self,
            AutoTMConfig(
                K=K,
                grid_range=grid_range,
                grid_method=grid_method,
                adaptive=adaptive,
                pts_per_sigma=pts_per_sigma,
                transition_method=transition_method,
                max_K=max_K,
                r_gh=r_gh,
                gh_order=gh_order,
            ),
        )
        return weights
