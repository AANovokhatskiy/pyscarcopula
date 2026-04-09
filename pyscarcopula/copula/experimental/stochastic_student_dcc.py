"""
Stochastic Student-t copula with OU-driven df and DCC-driven correlation path.

Model
-----
For each time t,
    U_t | x_t, R_t ~ t-copula(R_t, nu_t)
    nu_t = 2 + softplus(x_t)
where x_t is a latent Ornstein-Uhlenbeck process, and R_t is a deterministic
correlation path produced by a DCC(1,1) filter fitted on standardized
residuals z_t.

Design goals
------------
1. Reuse the same multivariate Student-t copula density as in the fixed-R model.
2. Keep the latent state scalar, so existing SCAR-TM-OU / GAS / MC strategies
   can still optimize only the df-process.
3. Make DCC estimation a separate step and cache the path/parameters in the
   class instance to avoid repeated fitting.
4. Provide practical sampling modes:
   - in_sample: replay the fitted R_t path
   - last_R:    freeze correlation at the last R_T
   - dcc_forecast_mean: multi-step forecast using E[z z^T] recursion

Notes
-----
- The DCC layer is intentionally simple: Gaussian quasi-likelihood on already
  standardized residuals. This is usually good enough for a first working
  prototype.
- The copula density remains the standard multivariate t-copula density.
- The class can accept external standardized residuals z_t, or compute them
  internally from raw returns via GARCH(1,1) standardization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import kendalltau
from scipy.stats import t as t_dist

from pyscarcopula._utils import pobs
from pyscarcopula.copula.base import BivariateCopula

try:
    from pyscarcopula._types import MLEResult
except Exception:  # pragma: no cover - fallback only for standalone use
    @dataclass
    class MLEResult:
        log_likelihood: float
        method: str
        copula_name: str
        success: bool
        nfev: int
        message: str
        copula_param: float


@dataclass
class DCCFitResult:
    a: float
    b: float
    log_likelihood: float
    success: bool
    nfev: int
    message: str
    method: str = "DCC(1,1)-QMLE"


# ══════════════════════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════════════════════


def _softplus(x):
    """Numerically stable softplus: log(1 + exp(x))."""
    return np.where(x > 30, x, np.log1p(np.exp(np.clip(x, -500, 30))))


def _softplus_deriv(x):
    """d softplus / dx = sigmoid(x)."""
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
            if np.isnan(tau):
                tau = 0.0
            R[i, j] = np.sin(np.pi / 2.0 * tau)
            R[j, i] = R[i, j]
    return R


def _ensure_positive_definite(R):
    """Project to nearest PD correlation matrix if needed."""
    R = np.asarray(R, dtype=np.float64)
    eigvals = np.linalg.eigvalsh(R)
    if np.min(eigvals) > 1e-10:
        R = 0.5 * (R + R.T)
        diag = np.sqrt(np.maximum(np.diag(R), 1e-12))
        R = R / np.outer(diag, diag)
        np.fill_diagonal(R, 1.0)
        return R

    vals, vecs = np.linalg.eigh(0.5 * (R + R.T))
    vals = np.maximum(vals, 1e-6)
    R_pd = vecs @ np.diag(vals) @ vecs.T
    diag = np.sqrt(np.maximum(np.diag(R_pd), 1e-12))
    R_pd = R_pd / np.outer(diag, diag)
    np.fill_diagonal(R_pd, 1.0)
    return R_pd


def _safe_cholesky(R):
    """Robust Cholesky for correlation matrices."""
    try:
        return np.linalg.cholesky(R)
    except np.linalg.LinAlgError:
        return np.linalg.cholesky(_ensure_positive_definite(R))


def _multivariate_t_logpdf(x, R, df, L_inv=None, log_det=None):
    """
    Log-density of multivariate t-distribution.

    x : (T, d) or (d,)
    R : (d, d) shape matrix (correlation)
    df : float — degrees of freedom
    L_inv, log_det : precomputed from Cholesky (optional)

    Returns: (T,)
    """
    x = np.atleast_2d(x)
    _, d = x.shape

    if L_inv is None or log_det is None:
        L = _safe_cholesky(R)
        L_inv = np.linalg.inv(L)
        log_det = 2.0 * np.sum(np.log(np.diag(L)))

    y = x @ L_inv.T
    quad = np.sum(y * y, axis=1)

    log_norm = (
        gammaln(0.5 * (df + d))
        - gammaln(0.5 * df)
        - 0.5 * d * np.log(df * np.pi)
        - 0.5 * log_det
    )

    return log_norm - 0.5 * (df + d) * np.log1p(quad / df)


def _student_copula_logpdf(u, R, df, L_inv=None, log_det=None):
    """Log-density of d-dimensional Student-t copula."""
    eps = 1e-10
    u_c = np.clip(np.asarray(u, dtype=np.float64), eps, 1.0 - eps)
    x = t_dist.ppf(u_c, df=df)
    log_joint = _multivariate_t_logpdf(x, R, df, L_inv, log_det)
    log_marginals = np.sum(t_dist.logpdf(x, df=df), axis=1)
    return log_joint - log_marginals


def _student_copula_dlogpdf_ddf(u, R, df, L_inv=None, log_det=None, eps_fd=1e-5):
    """Finite-difference derivative of log copula density wrt df."""
    df_p = df + eps_fd
    df_m = max(df - eps_fd, 2.001)
    lp = _student_copula_logpdf(u, R, df_p, L_inv, log_det)
    lm = _student_copula_logpdf(u, R, df_m, L_inv, log_det)
    return (lp - lm) / (df_p - df_m)


def _log_copula_inlined(x, df, d, L_inv, log_det):
    """Inlined log copula density with precomputed x=t_ppf(u,df)."""
    y = x @ L_inv.T
    quad = np.sum(y * y, axis=1)

    log_norm_joint = (
        gammaln(0.5 * (df + d))
        - gammaln(0.5 * df)
        - 0.5 * d * np.log(df * np.pi)
        - 0.5 * log_det
    )
    log_joint = log_norm_joint - 0.5 * (df + d) * np.log1p(quad / df)

    log_norm_marg = (
        gammaln(0.5 * (df + 1.0))
        - gammaln(0.5 * df)
        - 0.5 * np.log(df * np.pi)
    )
    log_marg = np.sum(
        log_norm_marg - 0.5 * (df + 1.0) * np.log1p(x * x / df), axis=1
    )

    return log_joint - log_marg


def _log_copula_inlined_timevarying(x, df, d, L_inv_path, log_det_path):
    """
    Log copula density for a full time path of correlation matrices.

    x : (T, d)
    L_inv_path : (T, d, d)
    log_det_path : (T,)
    returns : (T,)
    """
    y = np.einsum("tij,tj->ti", L_inv_path, x)
    quad = np.sum(y * y, axis=1)

    log_norm_joint = (
        gammaln(0.5 * (df + d))
        - gammaln(0.5 * df)
        - 0.5 * d * np.log(df * np.pi)
        - 0.5 * log_det_path
    )
    log_joint = log_norm_joint - 0.5 * (df + d) * np.log1p(quad / df)

    log_norm_marg = (
        gammaln(0.5 * (df + 1.0))
        - gammaln(0.5 * df)
        - 0.5 * np.log(df * np.pi)
    )
    log_marg = np.sum(
        log_norm_marg - 0.5 * (df + 1.0) * np.log1p(x * x / df), axis=1
    )

    return log_joint - log_marg


# ── PPF lookup table ────────────────────────────────────────


class _PPFTable:
    """
    Precomputed inverse-CDF table for Student-t distribution.

    Builds t_ppf(u, df) on a dense grid of df values once,
    then provides fast linear interpolation.  Typical speedup
    vs scipy.stats.t.ppf: ~300×.

    Parameters
    ----------
    u : (T, d)  pseudo-observations (fixed for the lifetime of the table)
    df_lo, df_hi : float  range of df values to cover
    n_lo, n_hi : int  number of nodes in [df_lo, 5] and [5, df_hi]
    """

    def __init__(self, u, df_lo=2.005, df_hi=250.0, n_lo=120, n_hi=80):
        u_c = np.clip(u, 1e-10, 1.0 - 1e-10)
        nodes_lo = np.linspace(df_lo, 5.0, n_lo)
        nodes_hi = np.geomspace(5.0, df_hi, n_hi)
        self.nodes = np.unique(np.concatenate([nodes_lo, nodes_hi]))
        self.table = np.empty((len(self.nodes),) + u_c.shape, dtype=np.float64)
        for i, df_val in enumerate(self.nodes):
            self.table[i] = t_dist.ppf(u_c, df=df_val)

    def __call__(self, df):
        """Interpolate ppf at given df.  Returns array of same shape as u."""
        idx = np.searchsorted(self.nodes, df) - 1
        idx = max(0, min(idx, len(self.nodes) - 2))
        alpha = (df - self.nodes[idx]) / (self.nodes[idx + 1] - self.nodes[idx])
        return (1.0 - alpha) * self.table[idx] + alpha * self.table[idx + 1]


# ── DCC recursion & likelihood (vectorized) ─────────────────


def _dcc_normalize(Q):
    """Map covariance-like Q to correlation matrix R."""
    q = np.sqrt(np.maximum(np.diag(Q), 1e-12))
    R = Q / np.outer(q, q)
    R = 0.5 * (R + R.T)
    np.fill_diagonal(R, 1.0)
    return _ensure_positive_definite(R)


def _dcc_recursion(z, a, b, Qbar):
    """
    Build DCC(1,1) path.

    Parameters
    ----------
    z : (T, d) standardized residuals
    a, b : DCC parameters
    Qbar : unconditional covariance/correlation target

    Returns
    -------
    Q_path, R_path  — each (T, d, d)
    """
    z = np.asarray(z, dtype=np.float64)
    T, d = z.shape
    Qbar = _ensure_positive_definite(Qbar)

    Q_path = np.empty((T, d, d), dtype=np.float64)
    R_path = np.empty((T, d, d), dtype=np.float64)

    c = 1.0 - a - b
    Q_prev = Qbar.copy()
    for t in range(T):
        if t == 0:
            Q_t = Qbar.copy()
        else:
            outer = np.outer(z[t - 1], z[t - 1])
            Q_t = c * Qbar + a * outer + b * Q_prev
            Q_t = 0.5 * (Q_t + Q_t.T)
        R_t = _dcc_normalize(Q_t)
        Q_path[t] = Q_t
        R_path[t] = R_t
        Q_prev = Q_t

    return Q_path, R_path


def _dcc_loglik_gaussian(z, R_path):
    """
    Gaussian quasi log-likelihood for DCC path (vectorized).

    ℓ = -½ Σ_t (log|R_t| + z_t^T R_t^{-1} z_t - z_t^T z_t)

    The -z^T z term is constant w.r.t. DCC parameters but included
    for correct absolute log-likelihood reporting.
    """
    z = np.asarray(z, dtype=np.float64)
    T, d = z.shape

    # Batch Cholesky: (T, d, d)
    L = np.linalg.cholesky(R_path)
    log_det = 2.0 * np.sum(np.log(
        np.diagonal(L, axis1=1, axis2=2)
    ), axis=1)                              # (T,)

    L_inv = np.linalg.inv(L)                # (T, d, d)
    y = np.einsum("tij,tj->ti", L_inv, z)   # (T, d)
    quad = np.sum(y * y, axis=1)             # (T,)
    z_sq = np.sum(z * z, axis=1)             # (T,)

    return float(-0.5 * np.sum(log_det + quad - z_sq))


# ── GARCH(1,1) standardization ──────────────────────────────


def _garch11_filter(r, omega, alpha, beta):
    """
    Univariate GARCH(1,1) variance filter.

    sigma2_t = omega + alpha * r_{t-1}^2 + beta * sigma2_{t-1}

    Returns
    -------
    sigma2 : (T,) conditional variance path
    """
    T = len(r)
    sigma2 = np.empty(T, dtype=np.float64)
    sigma2[0] = omega / max(1.0 - alpha - beta, 1e-6)  # unconditional
    for t in range(1, T):
        sigma2[t] = omega + alpha * r[t - 1] ** 2 + beta * sigma2[t - 1]
    return np.maximum(sigma2, 1e-12)


def _fit_garch11(r, omega0=None, alpha0=0.05, beta0=0.90):
    """
    Fit GARCH(1,1) to univariate return series via MLE.

    Returns
    -------
    (omega, alpha, beta), sigma2_path
    """
    r = np.asarray(r, dtype=np.float64)
    T = len(r)
    var_r = np.var(r, ddof=1)
    if omega0 is None:
        omega0 = var_r * (1.0 - alpha0 - beta0)

    def neg_ll(params):
        omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or (alpha + beta) >= 0.9999:
            return 1e12
        s2 = _garch11_filter(r, omega, alpha, beta)
        return 0.5 * np.sum(np.log(s2) + r ** 2 / s2)

    x0 = np.array([omega0, alpha0, beta0])
    bounds = [(1e-10, 10 * var_r), (1e-8, 0.5), (0.5, 0.9999)]
    res = minimize(neg_ll, x0, method="L-BFGS-B", bounds=bounds,
                   options={"gtol": 1e-5, "maxiter": 300})
    omega, alpha, beta = res.x
    sigma2 = _garch11_filter(r, omega, alpha, beta)
    return (omega, alpha, beta), sigma2


# ══════════════════════════════════════════════════════════════
# Class
# ══════════════════════════════════════════════════════════════


class StochasticStudentDCCCopula(BivariateCopula):
    """
    d-dimensional Student-t copula with stochastic df and DCC(1,1) correlation.

    Workflow
    --------
    1. Fit a DCC path R_t on standardized residuals z_t using fit_R_t().
       Alternatively, pass raw returns to fit_R_t(returns=...) and let the
       class compute GARCH(1,1) standardized residuals internally.
    2. Fit the latent OU process for df_t on pseudo-observations u_t using fit().

    Notes
    -----
    - The copula density is the same multivariate Student-t density as in the
      fixed-R model.
    - Existing SCAR/GAS strategies can be reused because the latent process is
      still scalar; only the emission density changes from R to R_t.
    """

    def __init__(self, d, rotate=0):
        super().__init__(rotate=0)
        if d < 2:
            raise ValueError(f"d must be >= 2, got {d}")
        self._d = int(d)
        self._name = f"Stochastic Student-t copula with DCC(1,1) (d={d})"
        self._bounds = [(-10.0, 10.0)]

        # DCC state
        self._dcc_a: Optional[float] = None
        self._dcc_b: Optional[float] = None
        self._dcc_Qbar: Optional[np.ndarray] = None
        self._dcc_Q_path: Optional[np.ndarray] = None
        self._R_path: Optional[np.ndarray] = None
        self._L_inv_path: Optional[np.ndarray] = None
        self._log_det_path: Optional[np.ndarray] = None
        self._last_z_for_dcc: Optional[np.ndarray] = None
        self.dcc_result: Optional[DCCFitResult] = None

        # GARCH params cache (per column)
        self._garch_params: Optional[list] = None

        # Convenience handles to last in-sample matrix
        self._R_last: Optional[np.ndarray] = None
        self._L_inv_last: Optional[np.ndarray] = None
        self._log_det_last: Optional[float] = None
        self._Q_last: Optional[np.ndarray] = None

        # Data cache
        self._last_u: Optional[np.ndarray] = None

        # PPF lookup table (built lazily in batch methods)
        self._ppf_table: Optional[_PPFTable] = None
        self._ppf_table_u_id: Optional[int] = None

    @property
    def d(self):
        return self._d

    @property
    def R_path(self):
        return self._R_path

    @property
    def dcc_params(self):
        if self._dcc_a is None or self._dcc_b is None:
            return None
        return {"a": self._dcc_a, "b": self._dcc_b}

    # ── Transform: x -> df ──────────────────────────────────

    def transform(self, x):
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        return 2.0 + _softplus(x)

    def inv_transform(self, df):
        df = np.atleast_1d(np.asarray(df, dtype=np.float64))
        return _inv_softplus(np.maximum(df - 2.0, 1e-15))

    def dtransform(self, x):
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        return _softplus_deriv(x)

    # ── DCC state handling ──────────────────────────────────

    def _set_R_path(self, R_path, Q_path=None, Qbar=None, dcc_params=None):
        """Store the correlation path and its fast Cholesky helpers (vectorized)."""
        R_path = np.asarray(R_path, dtype=np.float64)
        if R_path.ndim != 3 or R_path.shape[1:] != (self._d, self._d):
            raise ValueError(
                f"R_path must have shape (T, {self._d}, {self._d}), got {R_path.shape}"
            )

        T = R_path.shape[0]

        # Ensure all matrices are PD — check batch eigenvalues
        R_clean = np.empty_like(R_path)
        needs_fix = np.zeros(T, dtype=bool)

        # Symmetrize all
        R_sym = 0.5 * (R_path + np.swapaxes(R_path, -2, -1))

        # Check which need PD projection
        min_eigs = np.linalg.eigvalsh(R_sym)[:, 0]  # smallest eigenvalue
        needs_fix = min_eigs <= 1e-10

        # Fast path for good matrices
        good_mask = ~needs_fix
        if np.any(good_mask):
            R_good = R_sym[good_mask]
            diag_good = np.sqrt(np.maximum(
                np.diagonal(R_good, axis1=1, axis2=2), 1e-12
            ))  # (n_good, d)
            R_clean[good_mask] = R_good / (diag_good[:, :, None] * diag_good[:, None, :])
            for idx in np.where(good_mask)[0]:
                np.fill_diagonal(R_clean[idx], 1.0)

        # Slow path for matrices needing PD projection
        for idx in np.where(needs_fix)[0]:
            R_clean[idx] = _ensure_positive_definite(R_path[idx])

        # Batch Cholesky and inverse
        L = np.linalg.cholesky(R_clean)
        L_inv_path = np.linalg.inv(L)
        log_det_path = 2.0 * np.sum(
            np.log(np.diagonal(L, axis1=1, axis2=2)), axis=1
        )

        self._R_path = R_clean
        self._L_inv_path = L_inv_path
        self._log_det_path = log_det_path

        if Q_path is not None:
            self._dcc_Q_path = np.asarray(Q_path, dtype=np.float64)
            self._Q_last = self._dcc_Q_path[-1].copy()
        else:
            self._dcc_Q_path = None
            self._Q_last = None

        if Qbar is not None:
            self._dcc_Qbar = np.asarray(Qbar, dtype=np.float64)

        if dcc_params is not None:
            self._dcc_a = float(dcc_params[0])
            self._dcc_b = float(dcc_params[1])

        self._R_last = self._R_path[-1].copy()
        self._L_inv_last = self._L_inv_path[-1].copy()
        self._log_det_last = float(self._log_det_path[-1])

        # Invalidate PPF cache
        self._ppf_table = None
        self._ppf_table_u_id = None

    # ── Standardized residuals ──────────────────────────────

    @staticmethod
    def compute_standardized_residuals(returns):
        """
        Compute GARCH(1,1) standardized residuals from raw returns.

        Each column of `returns` is independently fitted with a GARCH(1,1)
        model, and the result is z_t = r_t / sigma_t.

        Parameters
        ----------
        returns : (T, d) ndarray
            Raw (log-)return series.

        Returns
        -------
        z : (T, d) ndarray
            Standardized residuals.
        garch_params : list of (omega, alpha, beta) tuples
            Fitted GARCH parameters per column.
        """
        returns = np.asarray(returns, dtype=np.float64)
        T, d = returns.shape
        z = np.empty_like(returns)
        garch_params = []
        for j in range(d):
            params, sigma2 = _fit_garch11(returns[:, j])
            z[:, j] = returns[:, j] / np.sqrt(sigma2)
            garch_params.append(params)
        return z, garch_params

    # ── DCC fitting ─────────────────────────────────────────

    def fit_R_t(
        self,
        z=None,
        returns=None,
        a0=0.03,
        b0=0.95,
        fix_params: Optional[Tuple[float, float]] = None,
        gtol=1e-6,
        qbar_method="kendall",
    ):
        """
        Fit DCC(1,1) on standardized residuals and cache the full R_t path.

        Parameters
        ----------
        z : (T, d) ndarray or None
            Standardized residuals used for DCC estimation.
            If None, computed from `returns` via GARCH(1,1).
        returns : (T, d) ndarray or None
            Raw returns. Used only if z is None.
        a0, b0 : float
            Initial values for DCC parameters.
        fix_params : (a, b) or None
            If provided, skip optimization and use these DCC parameters.
        gtol : float
            Optimizer tolerance.
        qbar_method : str
            'kendall' — use Kendall-tau based correlation as Qbar target
            (more robust for heavy tails).
            'pearson' — use sample covariance of z as Qbar target
            (classical DCC).
        """
        if z is None and returns is None:
            raise ValueError("Provide either z (standardized residuals) or returns")

        if z is None:
            z, garch_params = self.compute_standardized_residuals(returns)
            self._garch_params = garch_params

        z = np.asarray(z, dtype=np.float64)
        if z.ndim != 2 or z.shape[1] != self._d:
            raise ValueError(f"z must have shape (T, {self._d}), got {z.shape}")
        if len(z) < 2:
            raise ValueError("Need at least 2 observations to fit DCC")

        self._last_z_for_dcc = z.copy()

        # Normalize columns to unit variance for DCC.
        std = np.std(z, axis=0, ddof=1)
        std = np.where(std <= 1e-12, 1.0, std)
        z_std = z / std

        # Compute Qbar — the unconditional correlation target
        if qbar_method.lower() == "kendall":
            # Kendall-tau based: sin(pi/2 * tau) — more robust for heavy tails
            Qbar = _kendall_tau_matrix(z_std)
        else:
            # Classical: sample covariance
            Qbar = np.cov(z_std.T)
        Qbar = _ensure_positive_definite(Qbar)

        if fix_params is not None:
            a_hat, b_hat = float(fix_params[0]), float(fix_params[1])
            if a_hat < 0 or b_hat < 0 or (a_hat + b_hat) >= 0.999:
                raise ValueError("Need a >= 0, b >= 0, a + b < 1")
            Q_path, R_path = _dcc_recursion(z_std, a_hat, b_hat, Qbar)
            ll = _dcc_loglik_gaussian(z_std, R_path)
            result = DCCFitResult(
                a=a_hat,
                b=b_hat,
                log_likelihood=ll,
                success=True,
                nfev=0,
                message="DCC parameters fixed by user",
            )
            self._set_R_path(R_path, Q_path=Q_path, Qbar=Qbar, dcc_params=(a_hat, b_hat))
            self.dcc_result = result
            return result

        def objective(raw):
            a, b = float(raw[0]), float(raw[1])
            if a < 0.0 or b < 0.0 or (a + b) >= 0.999:
                return 1e12 + 1e8 * max(a + b - 0.999, 0.0)
            _, R_path = _dcc_recursion(z_std, a, b, Qbar)
            return -_dcc_loglik_gaussian(z_std, R_path)

        x0 = np.array([a0, b0], dtype=np.float64)
        bounds = [(1e-8, 0.999), (1e-8, 0.999)]
        res = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"gtol": gtol, "maxiter": 500},
        )

        a_hat, b_hat = float(res.x[0]), float(res.x[1])
        # enforce stationarity in final values if optimizer landed on boundary
        if a_hat + b_hat >= 0.999:
            s = a_hat + b_hat
            a_hat *= 0.999 / s
            b_hat *= 0.999 / s

        Q_path, R_path = _dcc_recursion(z_std, a_hat, b_hat, Qbar)
        ll = _dcc_loglik_gaussian(z_std, R_path)

        result = DCCFitResult(
            a=a_hat,
            b=b_hat,
            log_likelihood=ll,
            success=bool(res.success),
            nfev=int(getattr(res, "nfev", 0)),
            message=str(getattr(res, "message", "")),
        )
        self._set_R_path(R_path, Q_path=Q_path, Qbar=Qbar, dcc_params=(a_hat, b_hat))
        self.dcc_result = result
        return result

    def forecast_R_path(self, n, mode="mean"):
        """
        Forecast future R_t path from the fitted DCC recursion.

        mode='mean' uses E[z_t z_t^T | F_{t-1}] ≈ R_t and is stable for a first
        prototype. Returns arrays of shape (n, d, d).
        """
        if self._dcc_a is None or self._dcc_b is None or self._dcc_Qbar is None:
            raise ValueError("Call fit_R_t() first")
        if self._Q_last is None:
            raise ValueError("No last Q_t available. Call fit_R_t() first")
        if mode.lower() != "mean":
            raise NotImplementedError("Only mode='mean' is implemented")

        a = self._dcc_a
        b = self._dcc_b
        Qbar = self._dcc_Qbar
        Q_prev = self._Q_last.copy()
        R_prev = self._R_last.copy()

        Q_path = np.empty((n, self._d, self._d), dtype=np.float64)
        R_path = np.empty((n, self._d, self._d), dtype=np.float64)

        for h in range(n):
            Q_next = (1.0 - a - b) * Qbar + a * R_prev + b * Q_prev
            Q_next = 0.5 * (Q_next + Q_next.T)
            R_next = _dcc_normalize(Q_next)
            Q_path[h] = Q_next
            R_path[h] = R_next
            Q_prev = Q_next
            R_prev = R_next

        return Q_path, R_path

    # ── PPF table management ────────────────────────────────

    def _get_ppf_table(self, u):
        """Get or build PPF lookup table for given data."""
        u_id = id(u)
        if self._ppf_table is not None and self._ppf_table_u_id == u_id:
            return self._ppf_table
        self._ppf_table = _PPFTable(u)
        self._ppf_table_u_id = u_id
        return self._ppf_table

    # ── Density ──────────────────────────────────────────────

    def _require_R_path(self):
        if self._R_path is None or self._L_inv_path is None or self._log_det_path is None:
            raise ValueError("R_t path not set. Call fit_R_t() or _set_R_path() first.")

    def log_likelihood(self, u, r=None):
        """
        Log-likelihood under time-varying R_t.

        u : (T, d) pseudo-observations
        r : scalar df or None. If None, uses fitted parameter / stationary mean.
        """
        self._require_R_path()
        u = np.asarray(u, dtype=np.float64)
        T = len(u)
        if T != len(self._R_path):
            raise ValueError(f"u has length {T}, but R_path has length {len(self._R_path)}")

        if r is None:
            if isinstance(self.fit_result, MLEResult):
                r = self.fit_result.copula_param
            else:
                r = float(self.transform(np.array([self.fit_result.params.mu]))[0])

        x = t_dist.ppf(np.clip(u, 1e-10, 1.0 - 1e-10), df=r)
        ll = _log_copula_inlined_timevarying(
            x, r, self._d, self._L_inv_path[:T], self._log_det_path[:T]
        )
        return float(np.sum(ll))

    def pdf_on_grid(self, u_row, z_grid, t_index=0):
        """Copula density on latent grid for one observation at time t_index."""
        self._require_R_path()
        u_row = np.asarray(u_row, dtype=np.float64)
        z_grid = np.asarray(z_grid, dtype=np.float64)
        t_index = int(t_index)
        if not (0 <= t_index < len(self._R_path)):
            raise IndexError("t_index out of range")

        df_grid = self.transform(z_grid)
        result = np.empty(len(z_grid), dtype=np.float64)
        u_tiled = u_row[None, :]
        R_t = self._R_path[t_index]
        L_inv_t = self._L_inv_path[t_index]
        log_det_t = self._log_det_path[t_index]

        for k, df_k in enumerate(df_grid):
            result[k] = np.exp(
                _student_copula_logpdf(u_tiled, R_t, df_k, L_inv_t, log_det_t)[0]
            )
        return result

    def pdf_and_grad_on_grid(self, u_row, z_grid, t_index=0):
        """fi(z) and dfi/dz for a single observation at time t_index."""
        self._require_R_path()
        u_row = np.asarray(u_row, dtype=np.float64)
        z_grid = np.asarray(z_grid, dtype=np.float64)
        t_index = int(t_index)
        if not (0 <= t_index < len(self._R_path)):
            raise IndexError("t_index out of range")

        df_grid = self.transform(z_grid)
        dpsi = self.dtransform(z_grid)
        fi = np.empty(len(z_grid), dtype=np.float64)
        dlogc_ddf = np.empty(len(z_grid), dtype=np.float64)
        u_tiled = u_row[None, :]

        R_t = self._R_path[t_index]
        L_inv_t = self._L_inv_path[t_index]
        log_det_t = self._log_det_path[t_index]

        for k, df_k in enumerate(df_grid):
            ll = _student_copula_logpdf(u_tiled, R_t, df_k, L_inv_t, log_det_t)
            fi[k] = np.exp(ll[0])
            dll = _student_copula_dlogpdf_ddf(u_tiled, R_t, df_k, L_inv_t, log_det_t)
            dlogc_ddf[k] = dll[0]

        dfi_dz = fi * dlogc_ddf * dpsi
        return fi, dfi_dz

    def pdf_and_grad_on_grid_batch(self, u, x_grid):
        """
        Batch evaluation for all observations with time-varying R_t.

        Optimized: uses precomputed PPF lookup table (~300× faster ppf calls),
        inlined density computation, single table build per fit.

        Returns: (fi, dfi), each of shape (T, K)
        """
        self._require_R_path()
        u = np.asarray(u, dtype=np.float64)
        x_grid = np.asarray(x_grid, dtype=np.float64)
        T = len(u)
        if T != len(self._R_path):
            raise ValueError(f"u has length {T}, but R_path has length {len(self._R_path)}")

        K = len(x_grid)
        d = self._d
        df_grid = self.transform(x_grid)
        dpsi = self.dtransform(x_grid)
        eps = 1e-5

        ppf = self._get_ppf_table(u)

        fi = np.empty((T, K), dtype=np.float64)
        dfi = np.empty((T, K), dtype=np.float64)

        L_inv_T = self._L_inv_path[:T]
        log_det_T = self._log_det_path[:T]

        for k in range(K):
            df_c = df_grid[k]
            df_p = df_c + eps
            df_m = max(df_c - eps, 2.001)

            x_c = ppf(df_c)
            x_p = ppf(df_p)
            x_m = ppf(df_m)

            lc_c = _log_copula_inlined_timevarying(x_c, df_c, d, L_inv_T, log_det_T)
            lc_p = _log_copula_inlined_timevarying(x_p, df_p, d, L_inv_T, log_det_T)
            lc_m = _log_copula_inlined_timevarying(x_m, df_m, d, L_inv_T, log_det_T)

            fi[:, k] = np.exp(lc_c)
            dfi[:, k] = fi[:, k] * (lc_p - lc_m) / (df_p - df_m) * dpsi[k]

        return fi, dfi

    def copula_grid_batch(self, u, x_grid):
        """Batch version of pdf_on_grid for time-varying R_t."""
        self._require_R_path()
        u = np.asarray(u, dtype=np.float64)
        x_grid = np.asarray(x_grid, dtype=np.float64)
        T = len(u)
        if T != len(self._R_path):
            raise ValueError(f"u has length {T}, but R_path has length {len(self._R_path)}")

        K = len(x_grid)
        d = self._d
        df_grid = self.transform(x_grid)

        ppf = self._get_ppf_table(u)

        fi = np.empty((T, K), dtype=np.float64)
        L_inv_T = self._L_inv_path[:T]
        log_det_T = self._log_det_path[:T]

        for k in range(K):
            x = ppf(df_grid[k])
            fi[:, k] = np.exp(
                _log_copula_inlined_timevarying(x, df_grid[k], d, L_inv_T, log_det_T)
            )
        return fi

    # ── MLE fit ──────────────────────────────────────────────

    def _fit_mle(self, u):
        """Fit constant df under an already estimated time-varying R_t path."""
        self._require_R_path()
        u = np.asarray(u, dtype=np.float64)
        T = len(u)
        if T != len(self._R_path):
            raise ValueError(f"u has length {T}, but R_path has length {len(self._R_path)}")

        def neg_ll(x):
            df = float(self.transform(np.array([x[0]]))[0])
            x_t = t_dist.ppf(np.clip(u, 1e-10, 1.0 - 1e-10), df=df)
            ll = _log_copula_inlined_timevarying(
                x_t, df, self._d, self._L_inv_path[:T], self._log_det_path[:T]
            )
            return -float(np.sum(ll))

        x0 = np.array([float(self.inv_transform(np.array([5.0]))[0])])
        res = minimize(
            neg_ll,
            x0,
            method="L-BFGS-B",
            bounds=[(-8.0, 15.0)],
            options={"gtol": 1e-4, "maxiter": 300},
        )

        df_hat = float(self.transform(res.x)[0])
        result = MLEResult(
            log_likelihood=-float(res.fun),
            method="MLE",
            copula_name=self._name,
            success=bool(res.success),
            nfev=int(getattr(res, "nfev", 0)),
            message=str(getattr(res, "message", "")),
            copula_param=df_hat,
        )
        self.fit_result = result
        return result

    # ── Fit df dynamics ──────────────────────────────────────

    def fit(self, data, method="scar-tm-ou", to_pobs=False, **kwargs):
        """
        Fit df dynamics after the DCC path has already been estimated.

        Parameters
        ----------
        data : (T, d) array
            Pseudo-observations or raw data if to_pobs=True.
        method : str
            'mle', 'scar-tm-ou', 'gas', etc.
        to_pobs : bool
            Convert data to pseudo-observations before fitting.
        """
        self._require_R_path()
        u = np.asarray(data, dtype=np.float64)
        if to_pobs:
            u = pobs(u)

        if len(u) != len(self._R_path):
            raise ValueError(
                f"data has length {len(u)}, but fitted R_path has length {len(self._R_path)}"
            )

        self._last_u = u

        if method.upper() == "MLE":
            return self._fit_mle(u)

        from pyscarcopula.api import fit as _api_fit

        result = _api_fit(self, u, method=method, **kwargs)
        self.fit_result = result
        return result

    # ── Sampling helpers ─────────────────────────────────────

    def _infer_df_scalar(self):
        if isinstance(self.fit_result, MLEResult):
            return float(self.fit_result.copula_param)
        return float(self.transform(np.array([self.fit_result.params.mu]))[0])

    def _infer_df_path(self, n=None, df_mode="fitted", u=None, rng=None):
        """
        Infer a df path for sampling.

        Modes
        -----
        fitted   : constant df for MLE, stationary-mean df for SCAR
        smoothed : in-sample smoothed df_t for SCAR; constant for MLE
        posterior: draw x_T from posterior and keep df constant over the horizon
        """
        if rng is None:
            rng = np.random.default_rng()

        if n is None:
            if self._R_path is None:
                raise ValueError("Need n or fitted R_path")
            n = len(self._R_path)

        if self.fit_result is None:
            return np.full(n, 5.0, dtype=np.float64)

        if isinstance(self.fit_result, MLEResult):
            return np.full(n, float(self.fit_result.copula_param), dtype=np.float64)

        mode = df_mode.lower()
        if mode == "fitted":
            df_val = float(self.transform(np.array([self.fit_result.params.mu]))[0])
            return np.full(n, df_val, dtype=np.float64)

        if mode == "smoothed":
            u_data = u if u is not None else self._last_u
            if u_data is None:
                raise ValueError("No data cached for smoothed df path")
            x_sm = np.asarray(self.smoothed_params(u=u_data), dtype=np.float64)
            df_sm = self.transform(x_sm)
            if len(df_sm) != n:
                raise ValueError(
                    f"smoothed df path has length {len(df_sm)}, but n={n}; use mode='in_sample'"
                )
            return df_sm

        if mode == "posterior":
            u_data = u if u is not None else self._last_u
            if u_data is None:
                raise ValueError("No data cached for posterior df path")
            z_grid, prob = self.xT_distribution(u_data)
            idx = rng.choice(len(z_grid), size=1, p=prob)
            df_val = float(self.transform(z_grid[idx])[0])
            return np.full(n, df_val, dtype=np.float64)

        raise ValueError(f"Unknown df_mode: {df_mode}")

    def _sample_t_copula_with_R_path(self, R_path, df_path, rng=None):
        """Draw one observation per time point from a t-copula path."""
        if rng is None:
            rng = np.random.default_rng()
        R_path = np.asarray(R_path, dtype=np.float64)
        df_path = np.asarray(df_path, dtype=np.float64)
        n = len(R_path)
        if len(df_path) != n:
            raise ValueError(f"df_path length {len(df_path)} does not match R_path length {n}")

        d = self._d
        # Batch Cholesky for sampling
        L_all = np.linalg.cholesky(R_path)  # (n, d, d)
        z = rng.standard_normal((n, d))      # (n, d)
        chi2 = rng.chisquare(df_path)        # (n,)
        scale = np.sqrt(df_path / chi2)       # (n,)

        # x_t = scale_t * L_t @ z_t  — batched via einsum
        x = scale[:, None] * np.einsum("tij,tj->ti", L_all, z)  # (n, d)

        # CDF — group by unique df values for vectorization
        u = np.empty((n, d), dtype=np.float64)
        unique_dfs, inverse = np.unique(df_path, return_inverse=True)
        for idx, df_val in enumerate(unique_dfs):
            mask = inverse == idx
            u[mask] = t_dist.cdf(x[mask], df=df_val)

        return u

    # ── Sampling ─────────────────────────────────────────────

    def sample(
        self,
        n=None,
        mode="last_R",
        df_mode="fitted",
        R_path=None,
        df_path=None,
        rng=None,
    ):
        """
        Sample from the time-varying Student-t copula.

        Parameters
        ----------
        n : int or None
        mode : str
            'last_R' | 'in_sample' | 'dcc_forecast_mean'
        df_mode : str
            'fitted' | 'smoothed' | 'posterior'
        R_path : optional externally supplied path (overrides mode)
        df_path : optional externally supplied df path
        """
        if rng is None:
            rng = np.random.default_rng()

        if R_path is not None:
            R_use = np.asarray(R_path, dtype=np.float64)
            if R_use.ndim != 3 or R_use.shape[1:] != (self._d, self._d):
                raise ValueError(
                    f"R_path must have shape (T, {self._d}, {self._d}), got {R_use.shape}"
                )
            n_eff = len(R_use)
        else:
            self._require_R_path()
            mode = mode.lower()
            if mode == "in_sample":
                R_use = self._R_path
                n_eff = len(R_use)
            elif mode == "last_r":
                if n is None:
                    raise ValueError("n is required for mode='last_R'")
                R_use = np.repeat(self._R_last[None, :, :], int(n), axis=0)
                n_eff = int(n)
            elif mode == "dcc_forecast_mean":
                if n is None:
                    raise ValueError("n is required for mode='dcc_forecast_mean'")
                _, R_use = self.forecast_R_path(int(n), mode="mean")
                n_eff = int(n)
            else:
                raise ValueError(f"Unknown mode: {mode}")

        if df_path is not None:
            df_use = np.asarray(df_path, dtype=np.float64)
            if len(df_use) != n_eff:
                raise ValueError(f"df_path length {len(df_use)} does not match n={n_eff}")
        else:
            df_use = self._infer_df_path(n=n_eff, df_mode=df_mode, rng=rng)

        return self._sample_t_copula_with_R_path(R_use, df_use, rng=rng)

    # ── Predict ──────────────────────────────────────────────

    def predict(self, n, u=None, rng=None, mode="dcc_forecast_mean", df_mode="posterior"):
        """
        Out-of-sample prediction.

        Default behavior:
        - forecast R_t via DCC conditional-mean recursion
        - sample df from x_T posterior and keep it fixed over the horizon
        """
        return self.sample(n=n, mode=mode, df_mode=df_mode, rng=rng)

    # ── Smoothed params / terminal distribution ──────────────

    def smoothed_params(self, u=None):
        """Return smoothed latent x_t values from TM forward pass."""
        if self.fit_result is None:
            raise ValueError("Fit df dynamics first")
        u_data = u if u is not None else self._last_u
        if u_data is None:
            raise ValueError("No data available. Pass u= or call fit() first.")
        theta, mu, nu_ou = self.fit_result.params.values
        from pyscarcopula.numerical.tm_functions import tm_forward_smoothed

        return tm_forward_smoothed(theta, mu, nu_ou, u_data, self)

    def xT_distribution(self, u, K=300, grid_range=5.0):
        """Distribution of x_T on grid, using the fitted df-dynamics model."""
        if self.fit_result is None:
            raise ValueError("Fit df dynamics first")
        theta, mu, nu_ou = self.fit_result.params.values
        from pyscarcopula.numerical.tm_functions import tm_xT_distribution

        return tm_xT_distribution(theta, mu, nu_ou, u, self, K, grid_range)


__all__ = ["StochasticStudentDCCCopula", "DCCFitResult"]
