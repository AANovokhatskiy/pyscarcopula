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

from dataclasses import replace
import threading

import numpy as np
from scipy.stats import norm, t as t_dist
from scipy.optimize import minimize

from pyscarcopula.copula.base import CopulaCapabilities
from pyscarcopula.copula.multivariate.base import (
    MultivariateCopula,
    model_state_locked,
)
from pyscarcopula._types import DEFAULT_CONFIG, NumericalConfig
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

_DF_OFFSET = 2.0 + 1e-6
def _as_float64_array_no_copy(value):
    if type(value) is np.ndarray and value.dtype == np.float64:
        return value
    return np.asarray(value, dtype=np.float64)


def _validate_fit_data(u, d):
    if u.ndim != 2 or u.shape[1] != int(d):
        raise ValueError(f"data must have shape (n_observations, {int(d)})")
    if u.shape[0] == 0:
        raise ValueError("data must contain at least one observation")
    if not np.all(np.isfinite(u)):
        raise ValueError("data must contain only finite values")


def _factor_integer(name, value, *, minimum=0):
    if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def _sampling_integer(name, value, *, minimum=0):
    if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def _sampling_n_threads(value):
    value = _sampling_integer("n_threads", value, minimum=1)
    if value > 256:
        raise ValueError("n_threads must be an integer in [1, 256]")
    return value


def _sampling_memory_budget(memory_budget_bytes, required, guidance):
    if memory_budget_bytes is None:
        return
    budget = _sampling_integer(
        "memory_budget_bytes", memory_budget_bytes)
    if budget < int(required):
        raise MemoryError(
            f"sampling requires approximately {int(required)} bytes; "
            f"{guidance}")


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

def _softplus(x):
    """Numerically stable softplus: log(1 + exp(x))."""
    return np.logaddexp(0.0, x)


def _softplus_scalar(x):
    return float(np.logaddexp(0.0, float(x)))


def _softplus_deriv(x):
    """d softplus / dx = sigmoid(x) = 1 / (1 + exp(-x))."""
    out = np.empty_like(x, dtype=np.float64)
    positive = x >= 0.0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def _softplus_deriv_scalar(x):
    x = float(x)
    if x >= 0.0:
        return float(1.0 / (1.0 + np.exp(-x)))
    exp_x = np.exp(x)
    return float(exp_x / (1.0 + exp_x))


def _inv_softplus(y):
    """Inverse of softplus: x = log(exp(y) - 1)."""
    y = np.asarray(y, dtype=np.float64)
    return np.where(y > 30, y, np.log(np.expm1(np.clip(y, 1e-15, 500))))


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
    Also supports MLE (constant df), GAS, and SCAR-MC methods.

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
    _df_offset = _DF_OFFSET
    _scar_static_df_mle_initialization = True
    _scar_log_stationary_scale_optimization = True
    _supports_scar_mixture_h = False
    _capabilities = CopulaCapabilities(
        supports_gas=True,
        supports_scar_ou=True,
        supports_scar_mc=True,
        supports_latent_grid=True,
        supports_conditional_sampling=True,
        has_dynamic_scalar_parameter=True,
    )

    def __init__(self, d, R=None, *, corr_mode='fixed',
                 corr_base=None, corr_shrinkage_init=0.8,
                 cholesky_d_max=10, allow_large_cholesky=False,
                 factor_rank=None, factor_loadings=None,
                 factor_estimation="two-stage",
                 factor_tile_size=16384,
                 factor_uniqueness_min=1e-8,
                 factor_joint_max_params=100000,
                 factor_joint_penalty=1e-6,
                 factor_joint_condition_max=1e12,
                 factor_seed=0, factor_oversampling=8):
        if isinstance(d, (bool, np.bool_)) or not isinstance(
                d, (int, np.integer)):
            raise TypeError(f"d must be an integer >= 2, got {d!r}")
        d = int(d)
        if d < 2:
            raise ValueError(f"d must be >= 2, got {d}")
        corr_mode = str(corr_mode).lower()
        if corr_mode not in {'fixed', 'shrinkage', 'cholesky', 'factor'}:
            raise ValueError(
                "corr_mode must be 'fixed', 'shrinkage', 'cholesky', "
                "or 'factor'")
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
        self._bounds = [(-10.0, 10.0)]  # bounds in x-space (latent)
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
        self._factor_estimation = str(factor_estimation).lower()
        self._factor_tile_size = _factor_integer(
            "factor_tile_size", factor_tile_size, minimum=1)
        self._factor_uniqueness_min = float(factor_uniqueness_min)
        self._factor_joint_max_params = _factor_integer(
            "factor_joint_max_params",
            factor_joint_max_params,
            minimum=1,
        )
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
        self._factor_seed = _factor_integer(
            "factor_seed", factor_seed)
        self._factor_oversampling = _factor_integer(
            "factor_oversampling", factor_oversampling)
        if corr_mode == 'factor':
            if (
                    isinstance(factor_rank, (bool, np.bool_))
                    or not isinstance(factor_rank, (int, np.integer))):
                raise TypeError(
                    "factor_rank must be an integer when corr_mode='factor'")
            self._factor_rank = int(factor_rank)
            if not 1 <= self._factor_rank < d:
                raise ValueError("factor_rank must satisfy 1 <= k < d")
            if self._factor_estimation not in {'two-stage', 'joint'}:
                raise ValueError(
                    "factor_estimation must be 'two-stage' or 'joint'")
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
    def corr_mode(self):
        return self._corr_mode

    @property
    def factor_rank(self):
        return getattr(self, "_factor_rank", None)

    @property
    def factor_estimation(self):
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

    @property
    def factor_uniqueness_(self):
        if self._factor_correlation is None:
            return None
        return self._factor_correlation.uniqueness.copy()

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
        if "_L" not in self.__dict__:
            # States pickled before the Cholesky factor was cached.
            self._L = (
                np.linalg.cholesky(self._R) if self._R is not None else None)

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
        identifiable = (
            self._d * self._factor_rank
            - self._factor_rank * (self._factor_rank - 1) // 2
        )
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
        """Commit an internally generated SPD correlation matrix."""
        R = np.asarray(R, dtype=np.float64)
        if R.shape != (self._d, self._d):
            raise ValueError(
                f"R must be ({self._d}, {self._d}), got {R.shape}")
        if not np.all(np.isfinite(R)):
            raise ValueError("R must contain only finite values")
        try:
            L = np.linalg.cholesky(R)
        except np.linalg.LinAlgError as exc:
            raise ValueError("R must be positive definite") from exc
        self._R = R
        self._L = L
        self._L_inv = np.linalg.inv(L)
        self._log_det = 2.0 * np.sum(np.log(np.diag(L)))
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
                self.initialize_factor(u)
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

    def _corr_plugin_num_params(self):
        if self._corr_mode == 'factor':
            return (
                self._d * self._factor_rank
                - self._factor_rank * (self._factor_rank - 1) // 2
            )
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
        return self._pack_corr_params()

    def corr_alpha(self):
        return self._corr_alpha

    # ── Transform: x -> df ──────────────────────────────────
    # Psi(x) = 2 + 1e-6 + softplus(x), ensuring finite variance.

    def transform(self, x):
        """Map latent values to degrees of freedom above the finite-variance bound."""
        from pyscarcopula.numerical import multivariate_native
        return multivariate_native.transform(self, x)

    def transform_scalar(self, x):
        return float(self.transform(np.array([x], dtype=np.float64))[0])

    def inv_transform(self, df):
        """Map degrees of freedom above the model offset to latent values."""
        from pyscarcopula.numerical import multivariate_native
        return multivariate_native.inverse_transform(self, df)

    def dtransform(self, x):
        """d(Psi)/dx = sigmoid(x)."""
        from pyscarcopula.numerical import multivariate_native
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
        r : float (df) or None — if None, uses fitted df
        """
        if (
                self._corr_mode == 'factor'
                and self._factor_operator is None):
            raise ValueError(
                "Factor correlation not set. Call initialize_factor() first.")
        if self._corr_mode != 'factor' and self._R is None:
            raise ValueError("Correlation matrix R not set. Call fit() first.")

        u = np.asarray(u, dtype=np.float64)

        if r is None:
            from pyscarcopula._types import MLEResult
            if isinstance(self.fit_result, MLEResult):
                r = self.fit_result.copula_param
            else:
                r = float(self.transform(
                    np.array([self.fit_result.params.mu]))[0])

        if self._corr_mode == 'factor':
            return FactorStudentEvaluator(
                self._factor_operator, u).evaluate(
                    float(r), n_threads=n_threads).log_likelihood

        from pyscarcopula.numerical import static_likelihood
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
        from pyscarcopula.numerical import multivariate_native
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

        from pyscarcopula.numerical import multivariate_native
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

        from pyscarcopula.numerical import multivariate_native
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

        from pyscarcopula.numerical import multivariate_native
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

        from pyscarcopula.numerical import multivariate_native
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
            density, df_gradient = evaluator.pdf_and_grad_on_grid(
                self.transform(np.asarray(x_grid, dtype=np.float64)),
                dimension_tile=self._factor_tile_size,
                n_threads=n_threads,
            )
            return (
                density,
                df_gradient * self.dtransform(
                    np.asarray(x_grid, dtype=np.float64))[None, :],
            )

        from pyscarcopula.numerical import multivariate_native
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

        from pyscarcopula.numerical import multivariate_native
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

    def _fit_joint_factor_mle(
            self, u, config, optimizer_options):
        """Jointly estimate static df and identifiable factor loadings."""
        from pyscarcopula._types import MultivariateMLEResult
        from pyscarcopula.numerical._cpp_extension import CppError

        optimizer_options = dict(optimizer_options)
        optimizer_options.setdefault("ftol", 1e-10)

        parameterization, factor0 = (
            FactorLoadingParameterization.from_loadings(
                self._factor_loadings,
                uniqueness_min=self._factor_uniqueness_min,
            ))
        expected = self._corr_plugin_num_params()
        if parameterization.n_parameters != expected:
            raise RuntimeError(
                "joint factor parameterization has an invalid size")
        if expected > self._factor_joint_max_params:
            raise ValueError(
                "joint factor estimation exceeds "
                "factor_joint_max_params")

        x0 = np.concatenate([
            np.array([5.0], dtype=np.float64),
            factor0,
        ])
        fail_value = float(getattr(config, "fail_value", 1e10))
        penalty = self._factor_joint_penalty
        condition_max = self._factor_joint_condition_max
        best = {
            "value": np.inf,
            "x": x0.copy(),
            "loadings": self._factor_loadings.copy(),
            "evaluation": None,
            "operator_diagnostics": {},
        }
        evaluations = 0

        def failure(x):
            direction = np.asarray(x, dtype=np.float64) - x0
            norm = np.linalg.norm(direction)
            if not np.isfinite(norm) or norm == 0.0:
                direction = np.ones_like(x0)
            else:
                direction = direction / norm
            return fail_value, direction * np.sqrt(fail_value)

        def evaluate_valid(x):
            nonlocal evaluations
            evaluations += 1
            x = np.asarray(x, dtype=np.float64)
            if (
                    x.shape != x0.shape
                    or np.any(~np.isfinite(x))
                    or x[0] <= _DF_OFFSET):
                raise ValueError("invalid joint factor optimizer point")
            loadings = parameterization.loadings(x[1:])
            factor = FactorCorrelation(
                loadings,
                uniqueness_min=self._factor_uniqueness_min,
                diagnostics={"source": "joint_static_mle_trial"},
            )
            operator = factor.prepare()
            operator_diagnostics = dict(operator.diagnostics)
            if (
                    operator_diagnostics["condition_estimate_m"]
                    > condition_max):
                raise ValueError(
                    "joint factor Woodbury core exceeds condition gate")
            evaluation = FactorStudentEvaluator(
                operator, u).joint_likelihood_and_gradient(
                    float(x[0]),
                    n_threads=config.n_threads,
                )
            value = (
                -evaluation.log_likelihood
                + penalty * float(np.sum(loadings * loadings))
            )
            loading_objective_gradient = (
                -evaluation.dlog_likelihood_dloadings
                + 2.0 * penalty * loadings
            )
            gradient = np.empty_like(x)
            gradient[0] = -evaluation.dlog_likelihood_ddf
            gradient[1:] = parameterization.pullback(
                x[1:], loading_objective_gradient)
            if (
                    not np.isfinite(value)
                    or value >= fail_value
                    or np.any(~np.isfinite(gradient))):
                raise FloatingPointError(
                    "non-finite joint factor objective or gradient")
            if value < best["value"]:
                best.update({
                    "value": float(value),
                    "x": x.copy(),
                    "loadings": loadings.copy(),
                    "evaluation": evaluation,
                    "operator_diagnostics": operator_diagnostics,
                })
            return float(value), gradient

        def objective_and_gradient(x):
            try:
                return evaluate_valid(x)
            except (
                    FloatingPointError,
                    OverflowError,
                    ValueError,
                    RuntimeError,
                    np.linalg.LinAlgError,
                    CppError):
                return failure(x)

        initial_objective, _ = objective_and_gradient(x0)
        result = minimize(
            objective_and_gradient,
            x0,
            jac=True,
            method="L-BFGS-B",
            bounds=[(_DF_OFFSET, None)]
                + [(None, None)] * expected,
            options=optimizer_options,
        )

        try:
            final_objective, final_gradient = evaluate_valid(result.x)
            final_x = np.asarray(result.x, dtype=np.float64).copy()
            final_loadings = parameterization.loadings(final_x[1:])
            final_factor = FactorCorrelation(
                final_loadings,
                uniqueness_min=self._factor_uniqueness_min,
            ).prepare()
            final_evaluation = FactorStudentEvaluator(
                final_factor, u).joint_likelihood_and_gradient(
                    float(final_x[0]),
                    n_threads=config.n_threads,
                )
            final_operator_diagnostics = dict(
                final_factor.diagnostics)
        except (
                FloatingPointError,
                OverflowError,
                ValueError,
                RuntimeError,
                np.linalg.LinAlgError,
                CppError):
            final_x = best["x"]
            final_loadings = best["loadings"]
            final_evaluation = best["evaluation"]
            final_operator_diagnostics = best[
                "operator_diagnostics"]
            final_objective, final_gradient = (
                objective_and_gradient(final_x))

        if final_evaluation is None:
            raise RuntimeError(
                "joint factor MLE found no valid evaluation")
        gradient_inf_norm = float(np.max(np.abs(final_gradient)))
        gtol = float(optimizer_options.get("gtol", 1e-5))
        gradient_gate = max(1e-4, 40.0 * gtol)
        accepted = bool(
            result.success
            and np.isfinite(final_objective)
            and gradient_inf_norm <= gradient_gate
        )
        message = str(getattr(result, "message", ""))
        if result.success and not accepted:
            message = (
                f"{message}; rejected by joint factor gradient gate "
                f"({gradient_inf_norm:.6g} > {gradient_gate:.6g})"
            )

        start_source = self._factor_initialization_diagnostics.get(
            "source", "supplied")
        self._set_factor_loadings(
            final_loadings,
            diagnostics={
                "source": "joint_static_mle",
                "joint_start_source": start_source,
                "joint_anchor_rows":
                    parameterization.anchors.tolist(),
                "joint_penalty": penalty,
                "joint_condition_max": condition_max,
            },
        )
        df_hat = float(final_x[0])
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
            "joint_factor": True,
            "joint_dynamic_supported": False,
            "joint_identification":
                "pivoted_lower_triangular_positive_diag",
            "joint_anchor_rows": parameterization.anchors.copy(),
            "joint_penalty": penalty,
            "joint_condition_max": condition_max,
            "joint_initial_objective": float(initial_objective),
            "joint_final_objective": float(final_objective),
            "joint_gradient_inf_norm": gradient_inf_norm,
            "joint_gradient_gate": gradient_gate,
            "joint_evaluations": evaluations,
            "joint_native_diagnostics":
                dict(final_evaluation.diagnostics),
            "joint_operator_diagnostics":
                final_operator_diagnostics,
            **self._corr_count_diagnostics(),
        }
        result_object = MultivariateMLEResult(
            log_likelihood=final_evaluation.log_likelihood,
            method="MLE",
            copula_name=self._name,
            success=accepted,
            nfev=int(getattr(result, "nfev", evaluations)),
            message=message,
            copula_param=df_hat,
            parameter_count=1 + expected,
            n_observations=len(u),
            model_parameters={
                "df": df_hat,
                "corr_mode": "factor",
                "factor_rank": self._factor_rank,
                "factor_loadings": self.factor_loadings_,
                "factor_uniqueness": self.factor_uniqueness_,
                "factor_estimation": "joint",
                "factor_anchor_rows":
                    parameterization.anchors.copy(),
            },
            correlation_matrix=None,
            diagnostics=diagnostics,
        )
        self.fit_result = result_object
        return result_object

    def _fit_mle(self, u, config: NumericalConfig | None = None,
                 gtol=None, ftol=None, maxfun=None, maxiter=None,
                 maxls=None, eps=None, maxcor=None,
                 finite_diff_rel_step=None):
        """Fit constant natural ``df`` and optional correlation parameters.

        Unlike dynamic SCAR/GAS fitting, static MLE has no latent state and
        therefore optimizes degrees of freedom directly without ``transform``.
        """
        from pyscarcopula._types import MLEResult
        from pyscarcopula.numerical import static_likelihood
        from pyscarcopula.numerical._cpp_extension import CppError

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

        self._ensure_corr_initialized(u)
        if (
                self._corr_mode == "factor"
                and self._factor_estimation == "joint"):
            return self._fit_joint_factor_mle(
                u, config, optimizer_options)
        corr0 = self._initial_corr_params(u)
        n_corr = self._corr_num_params()
        counted_corr = self._corr_effective_num_params()
        fail_value = float(getattr(config, 'fail_value', 1e10))

        if self._corr_mode == 'factor':
            fixed_evaluator = FactorStudentEvaluator(
                self._factor_operator, u)
        else:
            fixed_evaluator = (
                static_likelihood.prepare(
                    self, u, n_threads=config.n_threads)
                if n_corr == 0 else None)

        def _failure_result(x):
            # Non-zero, large-magnitude gradient pointing back toward the
            # starting point: a zero gradient would make L-BFGS-B report
            # convergence at a point where evaluation actually failed.
            direction = x - x0
            norm = np.linalg.norm(direction)
            if not np.isfinite(norm) or norm == 0.0:
                direction = np.ones_like(x)
            else:
                direction = direction / norm
            return fail_value, direction * np.sqrt(fail_value)

        def objective_and_gradient(x):
            try:
                if n_corr:
                    self._set_corr_from_params(x[1:])
                    evaluator = static_likelihood.prepare(
                        self, u, n_threads=config.n_threads)
                    value, df_gradient, corr_gradient = (
                        evaluator.objective_and_joint_gradient(
                            float(x[0]), fail_value=fail_value))
                else:
                    evaluator = fixed_evaluator
                    if self._corr_mode == 'factor':
                        value, df_gradient = (
                            evaluator.objective_and_gradient(
                                float(x[0]),
                                n_threads=config.n_threads,
                            ))
                    else:
                        value, df_gradient = (
                            evaluator.objective_and_gradient(
                                float(x[0]), fail_value=fail_value))
                if not np.isfinite(value) or value >= fail_value:
                    # Evaluator-reported failure comes back as fail_value
                    # with a zero gradient, which L-BFGS-B would read as
                    # convergence — replace with a large non-zero gradient.
                    return _failure_result(x)
                gradient = np.empty_like(x)
                gradient[0] = df_gradient[0]
                if n_corr:
                    gradient[1:] = _corr_gradient_to_raw_params(
                        self._corr_mode,
                        x[1:],
                        self.R,
                        corr_gradient,
                        self._corr_base,
                    )
                return value, gradient
            except (FloatingPointError, OverflowError, ValueError,
                    np.linalg.LinAlgError, CppError):
                return _failure_result(x)

        # Static MLE starts and remains in natural degrees-of-freedom units.
        x0 = np.concatenate([np.array([5.0]), corr0])
        bounds = [(_DF_OFFSET, None)] + [(None, None)] * n_corr
        res = minimize(
            objective_and_gradient,
            x0,
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options=optimizer_options,
        )
        gradient_mode = (
            'analytical_df' if n_corr == 0 else 'analytical_joint')

        if n_corr:
            self._set_corr_from_params(res.x[1:])

        df_hat = float(res.x[0])
        diagnostics = {
            'n_threads': config.n_threads,
            'parameterization': 'natural_df',
            'gradient_mode': gradient_mode,
            'model_score': 'not_applicable',
            'optimizer_gradient': 'analytical',
            'gradient_kind': 'analytical',
            'setup_derivative': 'not_applicable',
            'filter_derivative': 'not_applicable',
            'df_gradient': 'analytical',
            'correlation_gradient': (
                'not_applicable' if n_corr == 0 else 'analytical'),
            **self._corr_count_diagnostics(),
            'corr_params_raw': self.corr_params(),
            'corr_alpha': self.corr_alpha(),
            **self.correlation_preprocessing_diagnostics(),
        }
        if self._corr_mode != 'factor':
            diagnostics['corr_matrix'] = self._R.copy()

        from pyscarcopula._types import MultivariateMLEResult

        model_parameters = {
            'df': df_hat,
            'corr_mode': self._corr_mode,
            'corr_alpha': self.corr_alpha(),
        }
        if self._corr_mode == 'factor':
            model_parameters.update({
                'factor_rank': self._factor_rank,
                'factor_loadings': self.factor_loadings_,
                'factor_uniqueness': self.factor_uniqueness_,
                'factor_estimation': self._factor_estimation,
            })
        else:
            model_parameters[
                'correlation_matrix'] = self._R.copy()

        result = MultivariateMLEResult(
            log_likelihood=-res.fun,
            method='MLE',
            copula_name=self._name,
            success=res.success,
            nfev=res.nfev,
            message=str(getattr(res, 'message', '')),
            copula_param=df_hat,
            parameter_count=1 + counted_corr,
            n_observations=len(u),
            model_parameters=model_parameters,
            correlation_matrix=(
                None
                if self._corr_mode == 'factor'
                else self._R.copy()),
            diagnostics=diagnostics,
        )
        self.fit_result = result
        return result

    def correlation_preprocessing_diagnostics(self):
        """Return diagnostics for the correlation used to initialize fitting."""
        if self._corr_preprocessing is None:
            return {}
        return self._corr_preprocessing.diagnostics()

    # ── Fit (MLE + SCAR) ────────────────────────────────────

    @model_state_locked
    def fit(self, data, method='scar-tm-ou', to_pobs=False, **kwargs):
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
        config = kwargs.pop('config', None)
        if 'tol' in kwargs:
            raise TypeError("tol is not supported; use gtol")

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

        self._last_u = u

        self._ensure_corr_initialized(u)

        if method.upper() == 'MLE':
            return self._fit_mle(u, config=config, **optimizer_kwargs)

        # Step 2: SCAR / GAS — use strategy
        from pyscarcopula.api import fit as _api_fit
        result = _api_fit(self, u, method=method, config=config, **kwargs)
        diagnostics = dict(result.diagnostics)
        diagnostics.update(self._corr_count_diagnostics())
        diagnostics.update(self.correlation_preprocessing_diagnostics())
        counted_corr = self._corr_effective_num_params()
        if hasattr(result, 'parameter_count'):
            result = replace(
                result,
                parameter_count=result.params.n_params + counted_corr,
                diagnostics=diagnostics,
            )
        else:
            result.diagnostics.update(diagnostics)
        self._last_latent_result = result
        self.fit_result = result
        return result

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
        n = _sampling_integer("n", n)
        n_threads = _sampling_n_threads(n_threads)
        r_arr = _validated_student_sampling_parameters(r, n)
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
            if rng is None:
                rng = np.random.default_rng()
            df_path = r_arr
            if df_path.size == 1:
                df_path = np.full(n, df_path[0], dtype=np.float64)
            elif df_path.size != n:
                raise ValueError(
                    f"r must be scalar or array of length {n}, "
                    f"got {df_path.size}")
            latent = self._factor_operator.sample_normal(
                n,
                rng=rng,
                n_threads=n_threads,
            )
            chi_square = np.empty(n, dtype=np.float64)
            for df_value in np.unique(df_path):
                rows = np.flatnonzero(df_path == df_value)
                chi_square[rows] = rng.chisquare(
                    float(df_value), size=len(rows))
            latent *= np.sqrt(df_path / chi_square)[:, None]
            return t_dist.cdf(latent, df=df_path[:, None])
        if self._R is None:
            raise ValueError("Correlation matrix R not set. Call fit() first.")
        if rng is None:
            rng = np.random.default_rng()

        is_scalar = (r_arr.size == 1)

        d = self._d
        L = self._L

        if is_scalar:
            # All samples share same df — vectorized
            df_val = float(r_arr[0])
            # multivariate t: x = sqrt(df/chi2) * L @ z, z ~ N(0,I)
            z = rng.standard_normal((n, d))
            chi2_samples = rng.chisquare(df_val, size=n)
            scale = np.sqrt(df_val / chi2_samples)  # (n,)
            x = scale[:, np.newaxis] * (z @ L.T)  # (n, d)
            u = t_dist.cdf(x, df=df_val)
        else:
            # Each sample has its own df — vectorized where possible
            if len(r_arr) != n:
                raise ValueError(
                    f"r must be scalar or array of length {n}, got {len(r_arr)}")

            z = rng.standard_normal((n, d))
            x_normal = z @ L.T  # (n, d) — correlated normal

            u = np.empty((n, d))
            # Group by unique df values for efficiency
            unique_dfs, inverse = np.unique(r_arr, return_inverse=True)
            for idx, df_val in enumerate(unique_dfs):
                mask = (inverse == idx)
                n_mask = np.sum(mask)
                chi2_samples = rng.chisquare(df_val, size=n_mask)
                scale = np.sqrt(df_val / chi2_samples)
                x_t = scale[:, np.newaxis] * x_normal[mask]
                u[mask] = t_dist.cdf(x_t, df=df_val)

        return u

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
        n = _sampling_integer("n", n)
        batch_rows = _sampling_integer(
            "batch_rows", batch_rows, minimum=1)
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
        n = _sampling_integer("n", n)
        n_threads = _sampling_n_threads(n_threads)
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
        return _api_sample(self, u_data, self.fit_result, n, rng=rng)

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
        n = _sampling_integer("n", n)
        batch_rows = _sampling_integer(
            "batch_rows", batch_rows, minimum=1)
        n_threads = _sampling_n_threads(n_threads)
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
        if state is None:
            parameters = strategy.model_sample_params(
                self, result, n, rng=rng)

            def independent_blocks():
                for start in range(0, n, batch_rows):
                    stop = min(n, start + batch_rows)
                    yield self.sample_conditional(
                        stop - start,
                        r=parameters[start:stop],
                        given=given,
                        rng=rng,
                        n_threads=n_threads,
                        memory_budget_bytes=memory_budget_bytes,
                    )

            return independent_blocks()

        def recursive_blocks():
            current = state
            for start in range(0, n, batch_rows):
                stop = min(n, start + batch_rows)
                block = np.empty(
                    (stop - start, self._d), dtype=np.float64)
                for row in range(stop - start):
                    parameter = strategy.sample_params(
                        self, current, 1, rng=rng)[0]
                    observation = self.sample_conditional(
                        1,
                        r=parameter,
                        given=given,
                        rng=rng,
                        n_threads=n_threads,
                        memory_budget_bytes=memory_budget_bytes,
                    )
                    block[row] = observation[0]
                    current = strategy.condition_state(
                        self, current, observation, result)
                yield block

        return recursive_blocks()

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
        n = _sampling_integer("n", n)
        n_threads = _sampling_n_threads(n_threads)
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
            _sampling_memory_budget(
                memory_budget_bytes,
                n * self._d * 8,
                "use sample_batches()/predict_batches(), reduce batch_rows, "
                "or increase memory_budget_bytes",
            )
            return fill_given(n, self._d, given)
        if r is None:
            from pyscarcopula._types import MLEResult
            if isinstance(self.fit_result, MLEResult):
                r = self.fit_result.copula_param
            else:
                r = self.transform(
                    np.array([self.fit_result.params.mu]))[0]
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
        n = _sampling_integer("n", n)
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
        n = _sampling_integer("n", n)
        batch_rows = _sampling_integer(
            "batch_rows", batch_rows, minimum=1)
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

        def blocks():
            for start in range(0, n, batch_rows):
                count = min(batch_rows, n - start)
                parameters = strategy.sample_params(
                    self,
                    state,
                    count,
                    rng=rng,
                    predictive_r_mode=config.predictive_r_mode,
                )
                yield self.sample_conditional(
                    count,
                    r=parameters,
                    given=config.given,
                    rng=rng,
                    n_threads=n_threads,
                    memory_budget_bytes=memory_budget_bytes,
                )

        return blocks()

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
        from pyscarcopula.numerical import _cpp_scar_ou
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
        """TM forward-backward sweep for ``posterior_state_weights``."""

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

        from pyscarcopula.numerical.tm_grid import TMGrid
        grid = TMGrid(
            latent_params[0], latent_params[1], latent_params[2],
            len(u), K, grid_range, grid_method, adaptive, pts_per_sigma,
            transition_method=transition_method, max_K=max_K,
            r_gh=r_gh, gh_order=gh_order)
        fi_grid = grid.copula_grid(u, self)

        beta = np.ones((len(u), grid.K), dtype=np.float64)
        for t in range(len(u) - 2, -1, -1):
            beta[t] = grid.matvec(fi_grid[t + 1] * beta[t + 1])
            mx = np.max(np.abs(beta[t]))
            if mx > 0.0:
                beta[t] /= mx

        weights = np.empty((len(u), grid.K), dtype=np.float64)
        phi = grid.p0.copy()
        for t in range(len(u)):
            raw = phi * fi_grid[t] * beta[t] * grid.trap_w
            raw = np.where(np.isfinite(raw) & (raw > 0.0), raw, 0.0)
            total = np.sum(raw)
            if total > 0.0:
                weights[t] = raw / total
            else:
                weights[t] = np.full(grid.K, 1.0 / grid.K)
            if t < len(u) - 1:
                phi = grid.advance_forward_phi(phi, fi_grid[t])
                if phi is None:
                    phi = np.full(grid.K, 1.0)

        return weights
