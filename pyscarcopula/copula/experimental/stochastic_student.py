"""
Stochastic Student-t copula: d-dimensional t-copula with OU-driven df.

The correlation matrix R is estimated once (Kendall tau) and fixed.
The degrees-of-freedom parameter nu(t) = Psi(x(t)) varies over time,
where x(t) is a latent Ornstein-Uhlenbeck process.

Psi(x) = 2 + softplus(x) = 2 + log(1 + exp(x)),  maps R -> (2, inf).

This class inherits BivariateCopula to plug into the existing
SCAR-TM-OU / SCAR-MC / GAS strategy infrastructure. Despite handling
d-dimensional data, the latent process is scalar (one OU drives df),
so all transfer-matrix machinery works unchanged.

Usage:
    from pyscarcopula.copula.experimental.stochastic_student import StochasticStudentCopula

    cop = StochasticStudentCopula(d=6)
    result = cop.fit(returns, method='scar-tm-ou', to_pobs=True)
    samples = cop.sample(10000, r=5.0)
    pred = cop.predict(100000)

    from pyscarcopula.stattests import gof_test
    gof_test(cop, returns, to_pobs=True)
"""

import numpy as np
from scipy.stats import t as t_dist, norm, kendalltau
from scipy.special import gammaln
from scipy.optimize import minimize

from pyscarcopula.copula.base import BivariateCopula
from pyscarcopula._utils import pobs
from pyscarcopula.copula.experimental.stochastic_student_dcc import _PPFTable


# ══════════════════════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════════════════════

def _softplus(x):
    """Numerically stable softplus: log(1 + exp(x))."""
    return np.where(x > 30, x, np.log1p(np.exp(np.clip(x, -500, 30))))


def _softplus_deriv(x):
    """d softplus / dx = sigmoid(x) = 1 / (1 + exp(-x))."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _inv_softplus(y):
    """Inverse of softplus: x = log(exp(y) - 1)."""
    y = np.asarray(y, dtype=np.float64)
    return np.where(y > 30, y, np.log(np.expm1(np.clip(y, 1e-15, 500))))


def _kendall_tau_matrix(u):
    """Estimate correlation matrix via Kendall's tau: R_ij = sin(pi/2 * tau_ij)."""
    d = u.shape[1]
    R = np.eye(d)
    for i in range(d):
        for j in range(i + 1, d):
            tau, _ = kendalltau(u[:, i], u[:, j])
            R[i, j] = np.sin(np.pi / 2.0 * tau)
            R[j, i] = R[i, j]
    return R


def _ensure_positive_definite(R):
    """Project to nearest PD matrix if needed."""
    eigvals = np.linalg.eigvalsh(R)
    if np.min(eigvals) > 0:
        return R
    # Nearest PD via eigenvalue clipping
    vals, vecs = np.linalg.eigh(R)
    vals = np.maximum(vals, 1e-6)
    R_pd = vecs @ np.diag(vals) @ vecs.T
    # Re-normalize to correlation matrix
    diag = np.sqrt(np.diag(R_pd))
    R_pd = R_pd / np.outer(diag, diag)
    np.fill_diagonal(R_pd, 1.0)
    return R_pd


def _multivariate_t_logpdf(x, R, df, L_inv=None, log_det=None):
    """
    Log-density of multivariate t-distribution.

    x : (T, d) or (d,)
    R : (d, d) shape matrix (correlation)
    df : float — degrees of freedom
    L_inv, log_det : precomputed from Cholesky (optional, for speed)

    Returns: (T,) or scalar
    """
    x = np.atleast_2d(x)
    T, d = x.shape

    if L_inv is None or log_det is None:
        L = np.linalg.cholesky(R)
        L_inv = np.linalg.inv(L)
        log_det = 2.0 * np.sum(np.log(np.diag(L)))

    # Quadratic form: x^T R^{-1} x = ||L^{-1} x||^2
    y = x @ L_inv.T  # (T, d)
    quad = np.sum(y ** 2, axis=1)  # (T,)

    # Log normalizing constant
    log_norm = (gammaln(0.5 * (df + d)) - gammaln(0.5 * df)
                - 0.5 * d * np.log(df * np.pi) - 0.5 * log_det)

    log_pdf = log_norm - 0.5 * (df + d) * np.log(1.0 + quad / df)
    return log_pdf


def _student_copula_logpdf(u, R, df, L_inv=None, log_det=None):
    """
    Log-density of d-dimensional Student-t copula.

    c(u; R, df) = f_d(t_df^{-1}(u); R, df) / prod_j f_1(t_df^{-1}(u_j); df)

    u : (T, d) pseudo-observations
    R : (d, d) shape matrix
    df : float — degrees of freedom

    Returns: (T,) log copula densities
    """
    eps = 1e-10
    u_c = np.clip(u, eps, 1.0 - eps)
    d = u_c.shape[1]

    x = t_dist.ppf(u_c, df=df)  # (T, d)

    log_joint = _multivariate_t_logpdf(x, R, df, L_inv, log_det)
    log_marginals = np.sum(t_dist.logpdf(x, df=df), axis=1)  # (T,)

    return log_joint - log_marginals


def _student_copula_dlogpdf_ddf(u, R, df, L_inv=None, log_det=None, eps_fd=1e-5):
    """
    d(log c) / d(df) via central finite differences.

    u : (T, d), R : (d, d), df : float

    Returns: (T,)
    """
    lp = _student_copula_logpdf(u, R, df + eps_fd, L_inv, log_det)
    lm = _student_copula_logpdf(u, R, df - eps_fd, L_inv, log_det)
    return (lp - lm) / (2.0 * eps_fd)


def _log_copula_inlined(x, df, d, L_inv, log_det):
    """
    Inlined log copula density given precomputed x = t_ppf(u, df).

    Avoids scipy wrapper overhead. Used by batch evaluation.

    x : (T, d) — precomputed t-quantiles
    df : float
    d : int — dimension
    L_inv : (d, d) — inverse Cholesky of R
    log_det : float — log-det of R

    Returns: (T,) log copula density
    """
    # Joint multivariate t log-pdf
    y = x @ L_inv.T              # (T, d)
    quad = np.sum(y * y, axis=1) # (T,)

    log_norm_joint = (gammaln(0.5 * (df + d)) - gammaln(0.5 * df)
                      - 0.5 * d * np.log(df * np.pi) - 0.5 * log_det)
    log_joint = log_norm_joint - 0.5 * (df + d) * np.log(1.0 + quad / df)

    # Sum of marginal log-pdfs (inlined, no scipy wrapper)
    log_norm_marg = (gammaln(0.5 * (df + 1.0)) - gammaln(0.5 * df)
                     - 0.5 * np.log(df * np.pi))
    log_marg = np.sum(
        log_norm_marg - 0.5 * (df + 1.0) * np.log(1.0 + x * x / df),
        axis=1)

    return log_joint - log_marg


# ══════════════════════════════════════════════════════════════
# Class
# ══════════════════════════════════════════════════════════════

class StochasticStudentCopula(BivariateCopula):
    """
    d-dimensional Student-t copula with stochastic degrees of freedom.

    The correlation matrix R is estimated once and fixed.
    The df parameter is driven by a latent OU process:
        df(t) = Psi(x(t)) = 2 + softplus(x(t))

    Compatible with SCAR-TM-OU: one latent OU process drives df(t).
    Also supports MLE (constant df), GAS, and SCAR-MC methods.

    Parameters
    ----------
    d : int
        Dimension (number of variables). Must be >= 2.
    R : (d, d) ndarray or None
        Correlation matrix. If None, estimated during fit().
    """

    def __init__(self, d, R=None, rotate=0):
        super().__init__(rotate=0)
        if d < 2:
            raise ValueError(f"d must be >= 2, got {d}")
        self._d = d
        self._name = f"Stochastic Student-t copula (d={d})"
        self._bounds = [(-10.0, 10.0)]  # bounds in x-space (latent)

        # Correlation matrix — set during fit or at init
        if R is not None:
            R = np.asarray(R, dtype=np.float64)
            if R.shape != (d, d):
                raise ValueError(f"R must be ({d}, {d}), got {R.shape}")
            self._R = _ensure_positive_definite(R)
        else:
            self._R = None

        # Precomputed Cholesky decomposition (set when R is set)
        self._L_inv = None
        self._log_det = None

        # PPF lookup table (built lazily in batch methods)
        self._ppf_table = None
        self._ppf_table_u_id = None

    @property
    def d(self):
        return self._d

    @property
    def R(self):
        return self._R

    def _set_R(self, R):
        """Set correlation matrix and precompute Cholesky."""
        self._R = _ensure_positive_definite(R)
        L = np.linalg.cholesky(self._R)
        self._L_inv = np.linalg.inv(L)
        self._log_det = 2.0 * np.sum(np.log(np.diag(L)))

    # ── Transform: x -> df ──────────────────────────────────
    # Psi(x) = 2 + softplus(x) = 2 + log(1 + exp(x))
    # Maps R -> (2, inf), ensuring finite variance.

    def transform(self, x):
        """x -> df: maps R to (2, inf)."""
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        return 2.0 + _softplus(x)

    def inv_transform(self, df):
        """df -> x: maps (2, inf) to R."""
        df = np.atleast_1d(np.asarray(df, dtype=np.float64))
        return _inv_softplus(np.maximum(df - 2.0, 1e-15))

    def dtransform(self, x):
        """d(Psi)/dx = sigmoid(x)."""
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        return _softplus_deriv(x)

    def _get_ppf_table(self, u):
        """Get or build PPF lookup table for given data."""
        u_id = id(u)
        if self._ppf_table is not None and self._ppf_table_u_id == u_id:
            return self._ppf_table
        self._ppf_table = _PPFTable(u)
        self._ppf_table_u_id = u_id
        return self._ppf_table

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
            r = self.fit_result.copula_param if self.fit_result else 5.0

        ll = _student_copula_logpdf(u, self._R, r, self._L_inv, self._log_det)
        return np.sum(ll)

    def pdf_on_grid(self, u_row, z_grid):
        """Copula density on latent grid for one observation.

        u_row : (d,) single observation
        z_grid : (K,) latent grid values

        Returns: (K,) copula densities c(u_row; R, Psi(z_j))
        """
        if self._R is None:
            raise ValueError("R not set")

        u_row = np.asarray(u_row, dtype=np.float64)
        z_grid = np.asarray(z_grid, dtype=np.float64)
        K = len(z_grid)
        df_grid = self.transform(z_grid)

        u_tiled = np.tile(u_row, (1, 1))  # (1, d)
        result = np.empty(K)

        for k in range(K):
            ll = _student_copula_logpdf(u_tiled, self._R, df_grid[k],
                                        self._L_inv, self._log_det)
            result[k] = np.exp(ll[0])

        return result

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

        u_row = np.asarray(u_row, dtype=np.float64)
        z_grid = np.asarray(z_grid, dtype=np.float64)
        K = len(z_grid)
        df_grid = self.transform(z_grid)
        dpsi = self.dtransform(z_grid)  # (K,)

        u_tiled = np.tile(u_row, (1, 1))  # (1, d)
        fi = np.empty(K)
        dlogc_ddf = np.empty(K)

        for k in range(K):
            ll = _student_copula_logpdf(u_tiled, self._R, df_grid[k],
                                        self._L_inv, self._log_det)
            fi[k] = np.exp(ll[0])

            dll = _student_copula_dlogpdf_ddf(u_tiled, self._R, df_grid[k],
                                              self._L_inv, self._log_det)
            dlogc_ddf[k] = dll[0]

        dfi_dz = fi * dlogc_ddf * dpsi
        return fi, dfi_dz

    def pdf_and_grad_on_grid_batch(self, u, x_grid):
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

        u = np.asarray(u, dtype=np.float64)
        x_grid = np.asarray(x_grid, dtype=np.float64)
        T = len(u)
        K = len(x_grid)
        d = self._d
        df_grid = self.transform(x_grid)
        dpsi = self.dtransform(x_grid)
        L_inv = self._L_inv
        log_det = self._log_det
        eps = 1e-5

        ppf = self._get_ppf_table(u)

        fi = np.empty((T, K))
        dfi = np.empty((T, K))

        for k in range(K):
            df_c = df_grid[k]
            df_p = df_c + eps
            df_m = max(df_c - eps, 2.001)

            x_c = ppf(df_c)
            x_p = ppf(df_p)
            x_m = ppf(df_m)

            lc_c = _log_copula_inlined(x_c, df_c, d, L_inv, log_det)
            lc_p = _log_copula_inlined(x_p, df_p, d, L_inv, log_det)
            lc_m = _log_copula_inlined(x_m, df_m, d, L_inv, log_det)

            fi[:, k] = np.exp(lc_c)
            dfi[:, k] = fi[:, k] * (lc_p - lc_m) / (df_p - df_m) * dpsi[k]

        return fi, dfi

    def copula_grid_batch(self, u, x_grid):
        """Batch version of pdf_on_grid (value only)."""
        if self._R is None:
            raise ValueError("R not set")

        u = np.asarray(u, dtype=np.float64)
        x_grid = np.asarray(x_grid, dtype=np.float64)
        K = len(x_grid)
        d = self._d
        df_grid = self.transform(x_grid)
        L_inv = self._L_inv
        log_det = self._log_det

        ppf = self._get_ppf_table(u)

        fi = np.empty((len(u), K))
        for k in range(K):
            x = ppf(df_grid[k])
            fi[:, k] = np.exp(
                _log_copula_inlined(x, df_grid[k], d, L_inv, log_det))

        return fi

    # ── MLE fit ──────────────────────────────────────────────

    def _fit_mle(self, u):
        """Fit constant df via profile MLE (R estimated via Kendall tau)."""
        from pyscarcopula._types import MLEResult

        # Estimate R if not set
        if self._R is None:
            R_hat = _kendall_tau_matrix(u)
            self._set_R(R_hat)

        def neg_ll(x):
            df = self.transform(np.array([x[0]]))[0]
            ll = _student_copula_logpdf(u, self._R, df,
                                        self._L_inv, self._log_det)
            return -np.sum(ll)

        # Initial guess: df=5 -> x = inv_transform(5)
        x0 = np.array([float(self.inv_transform(np.array([5.0]))[0])])
        res = minimize(neg_ll, x0, method='L-BFGS-B',
                       bounds=[(-8.0, 15.0)], options={'gtol': 1e-4})

        df_hat = self.transform(res.x)[0]

        result = MLEResult(
            log_likelihood=-res.fun,
            method='MLE',
            copula_name=self._name,
            success=res.success,
            nfev=res.nfev,
            message=str(getattr(res, 'message', '')),
            copula_param=df_hat,
        )
        self.fit_result = result
        return result

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
        u = np.asarray(data, dtype=np.float64)
        if to_pobs:
            u = pobs(u)

        self._last_u = u

        # Step 1: Estimate R if not set
        if self._R is None:
            R_hat = _kendall_tau_matrix(u)
            self._set_R(R_hat)

        if method.upper() == 'MLE':
            return self._fit_mle(u)

        # Step 2: SCAR / GAS — use strategy
        from pyscarcopula.api import fit as _api_fit
        result = _api_fit(self, u, method=method, **kwargs)
        self.fit_result = result
        return result

    # ── Sampling ─────────────────────────────────────────────

    def sample(self, n, r=None, rng=None):
        """
        Sample from d-dimensional Student-t copula.

        Parameters
        ----------
        n : int — number of observations
        r : float, (n,) array, or None — degrees of freedom.
            If scalar: all samples use same df.
            If array of length n: each sample uses its own df(t).
            If None: uses fitted df.
        rng : np.random.Generator or None

        Returns
        -------
        (n, d) pseudo-observations in [0, 1]^d
        """
        if self._R is None:
            raise ValueError("Correlation matrix R not set. Call fit() first.")
        if rng is None:
            rng = np.random.default_rng()

        if r is None:
            if self.fit_result is not None:
                from pyscarcopula._types import MLEResult
                if isinstance(self.fit_result, MLEResult):
                    r = self.fit_result.copula_param
                else:
                    # SCAR: use stationary OU mean
                    r = self.transform(
                        np.array([self.fit_result.params.mu]))[0]
            else:
                r = 5.0

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

    # ── Predict ──────────────────────────────────────────────

    def predict(self, n, u=None, rng=None):
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
        if self.fit_result is None:
            raise ValueError("Fit first")
        if rng is None:
            rng = np.random.default_rng()

        from pyscarcopula._types import MLEResult
        if isinstance(self.fit_result, MLEResult):
            return self.sample(n, r=self.fit_result.copula_param, rng=rng)

        u_data = u if u is not None else getattr(self, '_last_u', None)
        if u_data is not None:
            # Mixture sampling from posterior
            z_grid, prob = self.xT_distribution(u_data)
            idx = rng.choice(len(z_grid), size=n, p=prob)
            df_samples = self.transform(z_grid[idx])  # (n,)
            return self.sample(n, r=df_samples, rng=rng)
        else:
            # Fallback: stationary OU sample
            theta, mu, nu_ou = self.fit_result.params.values
            sigma2 = nu_ou ** 2 / (2.0 * theta)
            x_T = rng.normal(mu, np.sqrt(sigma2))
            df_val = self.transform(np.array([x_T]))[0]
            return self.sample(n, r=df_val, rng=rng)

    # ── Smoothed params ──────────────────────────────────────

    def smoothed_params(self, u=None):
        """Return smoothed df(t) from TM forward pass."""
        if self.fit_result is None:
            raise ValueError("Fit with SCAR first")
        u_data = u if u is not None else getattr(self, '_last_u', None)
        if u_data is None:
            raise ValueError("No data. Pass u= or call fit() first.")

        theta, mu, nu_ou = self.fit_result.params.values
        from pyscarcopula.numerical.tm_functions import tm_forward_smoothed
        return tm_forward_smoothed(theta, mu, nu_ou, u_data, self)

    def xT_distribution(self, u, K=300, grid_range=5.0):
        """Distribution of x_T on grid (for predict)."""
        if self.fit_result is None:
            raise ValueError("Fit with SCAR first")
        theta, mu, nu_ou = self.fit_result.params.values
        from pyscarcopula.numerical.tm_functions import tm_xT_distribution
        return tm_xT_distribution(theta, mu, nu_ou, u, self, K, grid_range)