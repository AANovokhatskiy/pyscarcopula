"""
Equicorrelation Gaussian copula with SCAR-TM support.

R(t) = (1-rho(t))*I + rho(t)*11'  where rho(t) = Psi(x(t))

Single scalar latent process controls the entire correlation matrix.
Exact analytical density: O(d) per evaluation (no matrix inversion).
Transfer matrix method works unchanged — same as bivariate SCAR.

Constraint: rho in (-1/(d-1), 1) for positive definiteness.
Transform: Psi(x) maps R -> (-1/(d-1), 1).
"""

import numpy as np
from numba import njit
from scipy.stats import norm
from scipy.optimize import minimize
from pyscarcopula.copula.base import BivariateCopula, _broadcast


# ══════════════════════════════════════════════════════════════
# Numba kernels
# ══════════════════════════════════════════════════════════════

@njit(cache=True)
def _equicorr_transform(x, d):
    """Map x in R to rho in (-1/(d-1), 1) via scaled tanh."""
    rho_min = -1.0 / (d - 1)
    rho_range = 1.0 - rho_min  # = d/(d-1)
    n = len(x)
    out = np.empty(n)
    for i in range(n):
        # tanh maps R -> (-1, 1), scale to (rho_min, 1)
        out[i] = rho_min + rho_range * 0.5 * (1.0 + np.tanh(x[i]))
    return out


@njit(cache=True)
def _equicorr_dtransform(x, d):
    """d(Psi)/dx."""
    rho_min = -1.0 / (d - 1)
    rho_range = 1.0 - rho_min
    n = len(x)
    out = np.empty(n)
    for i in range(n):
        th = np.tanh(x[i])
        out[i] = rho_range * 0.5 * (1.0 - th * th)
    return out


@njit(cache=True)
def _equicorr_inv_transform(rho, d):
    """Map rho in (-1/(d-1), 1) back to x."""
    rho_min = -1.0 / (d - 1)
    rho_range = 1.0 - rho_min
    n = len(rho)
    out = np.empty(n)
    for i in range(n):
        t = 2.0 * (rho[i] - rho_min) / rho_range - 1.0
        t = min(max(t, -0.999999), 0.999999)
        out[i] = 0.5 * np.log((1.0 + t) / (1.0 - t))  # atanh
    return out


@njit(cache=True)
def _equicorr_log_pdf(z_all, rho_arr, d):
    """
    Equicorrelation Gaussian copula log-density.

    z_all: (T, d) — Phi^{-1}(u)
    rho_arr: (T,) — correlation at each time step
    d: int

    Returns: (T,) log c(u; rho)
    """
    T = z_all.shape[0]
    out = np.empty(T)
    for t in range(T):
        rho = rho_arr[t]
        S1 = 0.0  # sum(z_i^2)
        S2 = 0.0  # (sum z_i)^2
        sz = 0.0
        for j in range(d):
            S1 += z_all[t, j] ** 2
            sz += z_all[t, j]
        S2 = sz * sz

        a_minus_1 = rho / (1.0 - rho)
        b = -rho / ((1.0 - rho) * (1.0 + (d - 1) * rho))

        log_det = (d - 1) * np.log(1.0 - rho) + np.log(1.0 + (d - 1) * rho)
        quad = a_minus_1 * S1 + b * S2

        out[t] = -0.5 * log_det - 0.5 * quad
    return out


@njit(cache=True)
def _equicorr_dlog_pdf_drho(z_all, rho_arr, d):
    """
    d(log c)/d(rho) for equicorrelation Gaussian copula.

    Analytical derivative needed for TM gradient.
    """
    T = z_all.shape[0]
    out = np.empty(T)
    for t in range(T):
        rho = rho_arr[t]
        S1 = 0.0
        sz = 0.0
        for j in range(d):
            S1 += z_all[t, j] ** 2
            sz += z_all[t, j]
        S2 = sz * sz

        omr = 1.0 - rho  # 1-rho
        D = 1.0 + (d - 1) * rho  # 1+(d-1)*rho

        # d(log_det)/drho = -(d-1)/(1-rho) + (d-1)/(1+(d-1)*rho)
        dlog_det = -(d - 1) / omr + (d - 1) / D

        # d(a-1)/drho = 1/(1-rho)^2
        da = 1.0 / (omr * omr)

        # d(b)/drho = derivative of -rho/((1-rho)*(1+(d-1)*rho))
        # b = -rho / (omr * D)
        # db = -(omr*D - rho*(-D + omr*(d-1))) / (omr*D)^2
        #    = -(omr*D + rho*D - rho*omr*(d-1)) / (omr*D)^2
        #    = -(D*(omr+rho) - rho*omr*(d-1)) / (omr*D)^2
        #    = -(D - rho*omr*(d-1)) / (omr*D)^2
        db = -(D - rho * omr * (d - 1)) / (omr * D) ** 2

        dquad = da * S1 + db * S2
        out[t] = -0.5 * dlog_det - 0.5 * dquad
    return out


@njit(cache=True)
def _equicorr_pdf_and_grad_batch(u_all, r_grid, dpsi, d):
    """
    Batch: fi and dfi/dx for all T observations on grid.

    u_all: (T, d) pseudo-observations
    r_grid: (K,) rho values on grid
    dpsi: (K,) d(Psi)/dx on grid
    d: int — dimension

    Returns: (fi, dfi) each (T, K)
    """
    T = u_all.shape[0]
    K = len(r_grid)

    fi = np.empty((T, K))
    dfi = np.empty((T, K))

    for t in range(T):
        # Precompute z and sufficient statistics
        S1 = 0.0
        sz = 0.0
        for j in range(d):
            zj = _ndtri(u_all[t, j])
            S1 += zj * zj
            sz += zj
        S2 = sz * sz

        for k in range(K):
            rho = r_grid[k]
            omr = 1.0 - rho
            D = 1.0 + (d - 1) * rho

            a_minus_1 = rho / omr
            b = -rho / (omr * D)
            log_det = (d - 1) * np.log(omr) + np.log(D)
            quad = a_minus_1 * S1 + b * S2
            log_c = -0.5 * log_det - 0.5 * quad
            c_val = np.exp(log_c)
            fi[t, k] = c_val

            # d(log c)/drho
            dlog_det = -(d - 1) / omr + (d - 1) / D
            da = 1.0 / (omr * omr)
            db = -(D - rho * omr * (d - 1)) / (omr * D) ** 2
            dquad = da * S1 + db * S2
            dlogc_drho = -0.5 * dlog_det - 0.5 * dquad

            dfi[t, k] = c_val * dlogc_drho * dpsi[k]

    return fi, dfi


@njit(cache=True)
def _ndtri(p):
    """Fast Phi^{-1}(p) approximation (Beasley-Springer-Moro)."""
    # Rational approximation, accurate to ~1e-9
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0

    a = [0.0, -3.969683028665376e1, 2.209460984245205e2,
         -2.759285104469687e2, 1.383577518672690e2,
         -3.066479806614716e1, 2.506628277459239e0]
    b = [0.0, -5.447609879822406e1, 1.615858368580409e2,
         -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1]
    c = [0.0, -7.784894002430293e-3, -3.223964580411365e-1,
         -2.400758277161838e0, -2.549732539343734e0,
         4.374664141464968e0, 2.938163982698783e0]
    d_coeff = [0.0, 7.784695709041462e-3, 3.224671290700398e-1,
               2.445134137142996e0, 3.754408661907416e0]

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = np.sqrt(-2.0 * np.log(p))
        return (((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6]) / \
               ((((d_coeff[1]*q+d_coeff[2])*q+d_coeff[3])*q+d_coeff[4])*q+1.0)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[1]*r+a[2])*r+a[3])*r+a[4])*r+a[5])*r+a[6])*q / \
               (((((b[1]*r+b[2])*r+b[3])*r+b[4])*r+b[5])*r+1.0)
    else:
        q = np.sqrt(-2.0 * np.log(1.0 - p))
        return -(((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6]) / \
                ((((d_coeff[1]*q+d_coeff[2])*q+d_coeff[3])*q+d_coeff[4])*q+1.0)


# ══════════════════════════════════════════════════════════════
# Class
# ══════════════════════════════════════════════════════════════

class EquicorrGaussianCopula(BivariateCopula):
    """
    Equicorrelation Gaussian copula for d dimensions.

    R(t) = (1-rho(t))*I + rho(t)*11'

    Single scalar parameter rho(t) controls all pairwise correlations.
    Compatible with SCAR-TM-OU: one latent OU process drives rho(t).

    Analytical density: O(d) per evaluation (no matrix operations).
    Exact d(log c)/drho for analytical gradient.

    Parameters
    ----------
    d : int
        Dimension (number of variables). Must be >= 2.
    """

    def __init__(self, d, rotate=0):
        super().__init__(rotate=0)
        if d < 2:
            raise ValueError(f"d must be >= 2, got {d}")
        self._d = d
        self._name = f"Equicorr Gaussian copula (d={d})"
        self._bounds = [(-10.0, 10.0)]  # bounds in x-space

    @property
    def d(self):
        return self._d

    def transform(self, x):
        return _equicorr_transform(
            np.atleast_1d(np.asarray(x, dtype=np.float64)), self._d)

    def inv_transform(self, r):
        return _equicorr_inv_transform(
            np.atleast_1d(np.asarray(r, dtype=np.float64)), self._d)

    def dtransform(self, x):
        return _equicorr_dtransform(
            np.atleast_1d(np.asarray(x, dtype=np.float64)), self._d)

    # ── Density ──────────────────────────────────────────────

    def log_likelihood(self, u, r=None):
        """
        Log-likelihood for d-dimensional data.

        u : (T, d) pseudo-observations
        r : float or None — if None, uses MLE rho
        """
        u = np.asarray(u, dtype=np.float64)
        z = norm.ppf(np.clip(u, 1e-10, 1 - 1e-10))

        if r is None:
            r = self.fit_result.copula_param if self.fit_result else 0.0

        rho_arr = np.full(len(u), r)
        ll = _equicorr_log_pdf(z, rho_arr, self._d)
        return np.sum(ll)

    def pdf_on_grid(self, u_row, z_grid):
        """Copula density on latent grid for one observation."""
        u_row = np.asarray(u_row, dtype=np.float64)
        z_grid = np.asarray(z_grid, dtype=np.float64)
        rho_grid = self.transform(z_grid)

        z_norm = norm.ppf(np.clip(u_row, 1e-10, 1 - 1e-10))
        z_all = np.tile(z_norm, (len(rho_grid), 1))  # (K, d)

        # Evaluate for each rho
        ll = _equicorr_log_pdf(z_all, rho_grid, self._d)
        return np.exp(ll)

    def pdf_and_grad_on_grid(self, u_row, z_grid):
        u_row = np.asarray(u_row, dtype=np.float64)
        z_grid = np.asarray(z_grid, dtype=np.float64)
        rho_grid = self.transform(z_grid)
        dpsi = self.dtransform(z_grid)

        z_norm = norm.ppf(np.clip(u_row, 1e-10, 1 - 1e-10))
        z_all = np.tile(z_norm, (len(rho_grid), 1))

        ll = _equicorr_log_pdf(z_all, rho_grid, self._d)
        dll = _equicorr_dlog_pdf_drho(z_all, rho_grid, self._d)

        fi = np.exp(ll)
        dfi = fi * dll * dpsi
        return fi, dfi

    def pdf_and_grad_on_grid_batch(self, u, x_grid):
        """Batch evaluation for all T observations."""
        u = np.asarray(u, dtype=np.float64)
        x = np.asarray(x_grid, dtype=np.float64)
        r_grid = self.transform(x)
        dpsi = self.dtransform(x)
        return _equicorr_pdf_and_grad_batch(u, r_grid, dpsi, self._d)

    def copula_grid_batch(self, u, x_grid):
        fi, _ = self.pdf_and_grad_on_grid_batch(u, x_grid)
        return fi

    # ── MLE ──────────────────────────────────────────────────

    def _fit_mle(self, u):
        """Fit constant rho via MLE."""
        from pyscarcopula._types import MLEResult

        u = np.asarray(u, dtype=np.float64)
        z = norm.ppf(np.clip(u, 1e-10, 1 - 1e-10))
        d = self._d

        def neg_ll(x):
            rho = self.transform(np.array([x[0]]))[0]
            rho_arr = np.full(len(u), rho)
            return -np.sum(_equicorr_log_pdf(z, rho_arr, d))

        x0 = np.array([0.5])
        res = minimize(neg_ll, x0, method='L-BFGS-B',
                       bounds=[(-8.0, 8.0)], options={'gtol': 1e-4})

        result = MLEResult(
            log_likelihood=-res.fun,
            method='MLE',
            copula_name=self._name,
            success=res.success,
            nfev=res.nfev,
            message=str(getattr(res, 'message', '')),
            copula_param=self.transform(res.x)[0],
        )
        self.fit_result = result
        return result

    # ── Fit (MLE + SCAR) ─────────────────────────────────────

    def fit(self, data, method='scar-tm-ou', to_pobs=False, **kwargs):
        from pyscarcopula._utils import pobs as _pobs

        u = np.asarray(data, dtype=np.float64)
        if to_pobs:
            u = _pobs(u)

        self._last_u = u  # store for predict

        if method.upper() == 'MLE':
            return self._fit_mle(u)

        # For SCAR/GAS: use strategy
        from pyscarcopula.api import fit as _api_fit
        result = _api_fit(self, u, method=method, **kwargs)
        self.fit_result = result
        return result

    # ── Sampling ─────────────────────────────────────────────

    def sample(self, n, r=None, rng=None):
        """Sample from equicorrelation Gaussian copula."""
        if rng is None:
            rng = np.random.default_rng()
        if r is None:
            r = self.fit_result.copula_param if self.fit_result else 0.5

        d = self._d
        # R = (1-rho)*I + rho*11' = LL' where L = sqrt(1-rho)*I + (sqrt(1+(d-1)*rho) - sqrt(1-rho))/d * 11'
        # Simpler: z = sqrt(1-rho)*eps + sqrt(rho)*eta*1 where eps~N(0,I), eta~N(0,1)
        if r >= 0:
            eps = rng.standard_normal((n, d))
            eta = rng.standard_normal((n, 1))
            z = np.sqrt(1 - r) * eps + np.sqrt(r) * eta
        else:
            # For negative rho, use Cholesky
            R = (1 - r) * np.eye(d) + r * np.ones((d, d))
            L = np.linalg.cholesky(R)
            z = rng.standard_normal((n, d)) @ L.T

        return norm.cdf(z)

    def predict(self, n, u=None, rng=None):
        """Sample from copula using conditional distribution from last fit.

        For MLE: constant rho.
        For SCAR-TM: mixture sampling from posterior p(x_T | data).

        Parameters
        ----------
        n : int
        u : (T, d) or None — data for conditioning.
            If None, uses data from last fit() call.
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
            rho_samples = self.transform(z_grid[idx])
            # Sample each observation with its own rho
            # (sample handles scalar r, so we loop or vectorize)
            result = np.empty((n, self._d))
            for j in range(n):
                result[j] = self.sample(1, r=float(rho_samples[j]), rng=rng)[0]
            return result
        else:
            # Fallback: stationary OU sample
            theta, mu, nu = self.fit_result.params.values
            sigma2 = nu ** 2 / (2.0 * theta)
            x_T = rng.normal(mu, np.sqrt(sigma2))
            rho = self.transform(np.array([x_T]))[0]
            return self.sample(n, r=rho, rng=rng)

    # ── Smoothed params ──────────────────────────────────────

    def smoothed_params(self, u):
        """Return smoothed rho(t) from TM forward pass."""
        if self.fit_result is None:
            raise ValueError("Fit with SCAR first")
        theta, mu, nu = self.fit_result.params.values
        from pyscarcopula.numerical.tm_functions import tm_forward_smoothed
        return tm_forward_smoothed(theta, mu, nu, u, self)

    def xT_distribution(self, u, K=300, grid_range=5.0):
        if self.fit_result is None:
            raise ValueError("Fit with SCAR first")
        theta, mu, nu = self.fit_result.params.values
        from pyscarcopula.numerical.tm_functions import tm_xT_distribution
        return tm_xT_distribution(theta, mu, nu, u, self, K, grid_range)