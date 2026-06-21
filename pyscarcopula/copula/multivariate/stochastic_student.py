"""
Stochastic Student-t copula: d-dimensional t-copula with OU-driven df.

The static correlation matrix R can be fixed, shrinkage-estimated, or fully
estimated through a Cholesky-like parameterization.
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

import numpy as np
from scipy.stats import t as t_dist
from scipy.optimize import minimize

from pyscarcopula.copula.base import CopulaCapabilities
from pyscarcopula.copula.multivariate.base import MultivariateCopula
from pyscarcopula._types import DEFAULT_CONFIG, NumericalConfig
from pyscarcopula._utils import pobs
from pyscarcopula.copula.multivariate.conditional import (
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
    corr_mode : {'fixed', 'shrinkage', 'cholesky'}
        Static correlation parameterization.
    corr_base : (d, d) ndarray or None
        Explicit initialization/base matrix for an estimated correlation
        mode. Initialization priority is ``corr_base``, then ``R``, then a
        Kendall estimate from the fit data.
    """

    _gas_optimizer_config = 'stochastic_student_gas_optimizer'
    _df_offset = _DF_OFFSET
    _scar_static_df_mle_initialization = True
    _supports_scar_mixture_h = False
    _capabilities = CopulaCapabilities(
        supports_gas=True,
        supports_scar_ou=True,
        supports_latent_grid=True,
        supports_conditional_sampling=True,
        has_dynamic_scalar_parameter=True,
    )

    def __init__(self, d, R=None, *, corr_mode='fixed',
                 corr_base=None, corr_shrinkage_init=0.8,
                 cholesky_d_max=10, allow_large_cholesky=False):
        if isinstance(d, (bool, np.bool_)) or not isinstance(
                d, (int, np.integer)):
            raise TypeError(f"d must be an integer >= 2, got {d!r}")
        d = int(d)
        if d < 2:
            raise ValueError(f"d must be >= 2, got {d}")
        corr_mode = str(corr_mode).lower()
        if corr_mode not in {'fixed', 'shrinkage', 'cholesky'}:
            raise ValueError(
                "corr_mode must be 'fixed', 'shrinkage', or 'cholesky'")
        if corr_mode == 'fixed' and corr_base is not None:
            raise ValueError("corr_base is only valid for estimated corr modes")
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

        # Correlation matrix — set during fit or at init
        self._R = None
        self._L_inv = None
        self._log_det = None
        self._corr_cache_version = 0
        if R is not None:
            R = np.asarray(R, dtype=np.float64)
            if R.shape != (d, d):
                raise ValueError(f"R must be ({d}, {d}), got {R.shape}")
            self._set_R(R, source="supplied")

        # Transient full-sample PPF cache.
        self._ppf_cache = None

    @property
    def d(self):
        return self._d

    @property
    def R(self):
        return self._R

    @property
    def corr_mode(self):
        return self._corr_mode

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_emission_cache", None)
        state["_ppf_cache"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
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
        if self._corr_mode == 'fixed':
            return 0
        if self._corr_mode == 'shrinkage':
            return 1
        return cholesky_corr_n_params(self._d)

    def _default_corr_params(self):
        if self._corr_mode == 'fixed':
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
        if self._corr_mode == 'fixed':
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

    def prepare_emission_cache(self, u):
        """Return the reusable full-sample Student PPF cache."""
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

    def log_likelihood(self, u, r=None):
        """
        Log-likelihood for d-dimensional data.

        u : (T, d) pseudo-observations
        r : float (df) or None — if None, uses fitted df
        """
        if self._R is None:
            raise ValueError("Correlation matrix R not set. Call fit() first.")

        u = np.asarray(u, dtype=np.float64)

        if r is None:
            from pyscarcopula._types import MLEResult
            if isinstance(self.fit_result, MLEResult):
                r = self.fit_result.copula_param
            else:
                r = float(self.transform(
                    np.array([self.fit_result.params.mu]))[0])

        from pyscarcopula.numerical import static_likelihood
        return static_likelihood.prepare(self, u).log_likelihood(float(r))

    def log_pdf_rows(self, u, r, t_index=None, cache=None):
        """Return one log-density per row for scalar/row-wise df values."""
        if self._R is None:
            raise ValueError("Correlation matrix R not set. Call fit() first.")
        from pyscarcopula.numerical import multivariate_native
        values, _ = multivariate_native.log_pdf_and_dlog_rows(
            self, u, r, t_index=t_index, cache=cache)
        return values

    def dlog_pdf_dr_rows(self, u, r, t_index=None, cache=None):
        """Return d log c(u_t; df_t) / d df_t for each row."""
        if self._R is None:
            raise ValueError("Correlation matrix R not set. Call fit() first.")

        from pyscarcopula.numerical import multivariate_native
        _, values = multivariate_native.log_pdf_and_dlog_rows(
            self, u, r, t_index=t_index, cache=cache)
        return values

    def log_pdf_and_dlog_dr_rows(self, u, r, t_index=None, cache=None):
        """Return per-row log-density and d log c(u_t; df_t) / d df_t."""
        if self._R is None:
            raise ValueError("Correlation matrix R not set. Call fit() first.")

        from pyscarcopula.numerical import multivariate_native
        return multivariate_native.log_pdf_and_dlog_rows(
            self, u, r, t_index=t_index, cache=cache)

    def pdf_on_grid(self, u_row, z_grid):
        """Copula density on latent grid for one observation.

        u_row : (d,) single observation
        z_grid : (K,) latent grid values

        Returns: (K,) copula densities c(u_row; R, Psi(z_j))
        """
        if self._R is None:
            raise ValueError("R not set")

        from pyscarcopula.numerical import multivariate_native
        fi, _ = multivariate_native.pdf_and_grad_grid(
            self, np.asarray(u_row, dtype=np.float64)[None, :], z_grid)
        return fi[0]

    def pdf_and_grad_on_grid(self, u_row, z_grid):
        """
        Compute fi(z) and dfi/dz on the grid analytically.

        fi(z) = c(u_row; R, Psi(z))
        dfi/dz = fi * d(log c)/d(df) * d(Psi)/dz

        u_row : (d,), z_grid : (K,)
        Returns: (fi, dfi_dz) each of shape (K,)
        """
        if self._R is None:
            raise ValueError("R not set")

        from pyscarcopula.numerical import multivariate_native
        fi, dfi = multivariate_native.pdf_and_grad_grid(
            self, np.asarray(u_row, dtype=np.float64)[None, :], z_grid)
        return fi[0], dfi[0]

    def pdf_and_grad_on_grid_batch(self, u, x_grid, t_index=0, cache=None):
        """
        Batch evaluation for all T observations.

        Optimized: uses precomputed PPF lookup table (~300× faster ppf calls),
        inlined density computation, single table build per fit.

        u : (T, d) pseudo-observations
        x_grid : (K,) latent grid values

        Returns: (fi, dfi) each (T, K)
        """
        if self._R is None:
            raise ValueError("R not set")

        from pyscarcopula.numerical import multivariate_native
        return multivariate_native.pdf_and_grad_grid(
            self, u, x_grid, t_index=t_index, cache=cache)

    def copula_grid_batch(self, u, x_grid, t_index=0, cache=None):
        """Batch version of pdf_on_grid (value only)."""
        if self._R is None:
            raise ValueError("R not set")

        from pyscarcopula.numerical import multivariate_native
        fi, _ = multivariate_native.pdf_and_grad_grid(
            self, u, x_grid, t_index=t_index, cache=cache)
        return fi

    # ── MLE fit ──────────────────────────────────────────────

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
        corr0 = self._initial_corr_params(u)
        n_corr = self._corr_num_params()
        fail_value = float(getattr(config, 'fail_value', 1e10))

        fixed_evaluator = (
            static_likelihood.prepare(self, u) if n_corr == 0 else None)

        def objective_and_gradient(x):
            try:
                if n_corr:
                    self._set_corr_from_params(x[1:])
                    evaluator = static_likelihood.prepare(self, u)
                    value, df_gradient, corr_gradient = (
                        evaluator.objective_and_joint_gradient(
                            float(x[0]), fail_value=fail_value))
                else:
                    evaluator = fixed_evaluator
                    value, df_gradient = evaluator.objective_and_gradient(
                        float(x[0]), fail_value=fail_value)
                if not np.isfinite(value):
                    return fail_value, np.zeros_like(x)
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
            except Exception:
                return fail_value, np.zeros_like(x)

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
            'corr_mode': self._corr_mode,
            'corr_n_params': n_corr,
            'corr_params_raw': self.corr_params(),
            'corr_alpha': self.corr_alpha(),
            'corr_matrix': self._R.copy(),
            **self.correlation_preprocessing_diagnostics(),
        }

        from pyscarcopula._types import MultivariateMLEResult

        result = MultivariateMLEResult(
            log_likelihood=-res.fun,
            method='MLE',
            copula_name=self._name,
            success=res.success,
            nfev=res.nfev,
            message=str(getattr(res, 'message', '')),
            copula_param=df_hat,
            parameter_count=1 + n_corr,
            n_observations=len(u),
            model_parameters={
                'df': df_hat,
                'corr_mode': self._corr_mode,
                'corr_alpha': self.corr_alpha(),
                'correlation_matrix': self._R.copy(),
            },
            correlation_matrix=self._R,
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

    def fit(self, data, method='scar-tm-ou', to_pobs=False, **kwargs):
        """
        Fit the stochastic Student-t copula.

        Step 1: Estimate R via Kendall tau (if not already set).
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
        if to_pobs:
            u = pobs(u)

        self._last_u = u

        if method.upper() == 'GAS' and self._corr_num_params() > 0:
            raise NotImplementedError(
                "joint static correlation estimation is implemented for "
                "MLE and SCAR-TM-OU, not GAS")

        self._ensure_corr_initialized(u)

        if method.upper() == 'MLE':
            optimizer_kwargs = {
                key: kwargs.pop(key)
                for key in _LBFGSB_FIT_KEYS
                if key in kwargs
            }
            return self._fit_mle(u, config=config, **optimizer_kwargs)

        # Step 2: SCAR / GAS — use strategy
        from pyscarcopula.api import fit as _api_fit
        result = _api_fit(self, u, method=method, config=config, **kwargs)
        result.diagnostics.update(
            self.correlation_preprocessing_diagnostics())
        self.fit_result = result
        return result

    # ── Sampling ─────────────────────────────────────────────

    def sample_at_parameter(self, n, r, rng=None):
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
        if self._R is None:
            raise ValueError("Correlation matrix R not set. Call fit() first.")
        if rng is None:
            rng = np.random.default_rng()

        r_arr = np.atleast_1d(np.asarray(r, dtype=np.float64))
        is_scalar = (r_arr.size == 1)

        d = self._d
        L = np.linalg.cholesky(self._R)

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

    def sample(self, n, u=None, rng=None):
        """Generate observations reproducing the fitted model."""
        if self.fit_result is None:
            raise ValueError("Fit first")
        from pyscarcopula.api import sample as _api_sample

        u_data = u if u is not None else getattr(self, "_last_u", None)
        if u_data is None:
            raise ValueError(
                "No data for sample. "
                "Either call fit() first or pass u= explicitly.")
        return _api_sample(self, u_data, self.fit_result, n, rng=rng)

    # ── Predict ──────────────────────────────────────────────

    def sample_conditional(self, n, r=None, given=None, rng=None):
        """Sample conditionally with ``given={var_index: u_value}``."""
        if self._R is None:
            raise ValueError("Correlation matrix R not set. Call fit() first.")
        if rng is None:
            rng = np.random.default_rng()
        given = validate_multivariate_given(given, self._d)
        if not given:
            if r is None:
                return self.sample(n, rng=rng)
            return self.sample_at_parameter(n, r=r, rng=rng)
        if r is None:
            from pyscarcopula._types import MLEResult
            if isinstance(self.fit_result, MLEResult):
                r = self.fit_result.copula_param
            else:
                r = self.transform(
                    np.array([self.fit_result.params.mu]))[0]
        return sample_student_conditional(
            n, self._R, r, given=given, rng=rng)

    def predict(self, n, u=None, rng=None, given=None, horizon='next',
                predictive_r_mode=None, predict_config=None):
        """
        Sample n observations for next-step prediction.

        For MLE: constant df.
        For SCAR-TM: mixture sampling from posterior p(x_T | data).

        Parameters
        ----------
        n : int
        u : (T, d) or None — conditioning data.
        rng : np.random.Generator or None
        """
        if predict_config is not None:
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
        if rng is None:
            rng = np.random.default_rng()

        from pyscarcopula._types import MLEResult
        if isinstance(self.fit_result, MLEResult):
            return self.sample_conditional(
                n, r=self.fit_result.copula_param, given=given, rng=rng)

        u_data = u if u is not None else getattr(self, '_last_u', None)
        if u_data is not None:
            # Mixture sampling from posterior
            z_grid, prob = self.xT_distribution(u_data)
            idx = rng.choice(len(z_grid), size=n, p=prob)
            df_samples = self.transform(z_grid[idx])  # (n,)
            return self.sample_conditional(
                n, r=df_samples, given=given, rng=rng)
        else:
            # Fallback: stationary OU sample
            kappa, mu, nu_ou = self.fit_result.params.values
            sigma2 = nu_ou ** 2 / (2.0 * kappa)
            x_T = rng.normal(mu, np.sqrt(sigma2))
            df_val = self.transform(np.array([x_T]))[0]
            return self.sample_conditional(n, r=df_val, given=given, rng=rng)

    # Predictive mean path

    def predictive_mean(self, u=None):
        """Return predictive mean df(t) from TM forward pass."""
        if self.fit_result is None:
            raise ValueError("Fit with SCAR first")
        u_data = u if u is not None else getattr(self, '_last_u', None)
        if u_data is None:
            raise ValueError("No data. Pass u= or call fit() first.")

        kappa, mu, nu_ou = self.fit_result.params.values
        from pyscarcopula.numerical import _cpp_scar_ou
        return _cpp_scar_ou.predictive_mean(
            kappa, mu, nu_ou, u_data, self)

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
            params = self.fit_result.params.values
        params = np.asarray(params, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(params)):
            raise ValueError("params must contain only finite values")

        self._ensure_corr_initialized(u)
        n_corr = self._corr_num_params()
        joint_size = 3 + n_corr
        if params.size == 3:
            latent_params = params
        elif n_corr and params.size == joint_size:
            latent_params, corr_params = self._split_joint_params(params)
            self._set_corr_from_params(corr_params)
        else:
            expected = "3" if n_corr == 0 else f"3 or {joint_size}"
            raise ValueError(
                f"params must contain {expected} values for "
                f"corr_mode={self._corr_mode!r}, got {params.size}")

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
