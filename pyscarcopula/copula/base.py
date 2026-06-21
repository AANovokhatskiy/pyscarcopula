"""Shared base classes for bivariate copulas.

Built-in numerical operations are dispatched through the native copula
adapter. Subclasses provide family metadata plus family-specific sampling and
Kendall-tau helpers where needed.
"""

from dataclasses import dataclass, replace

import numpy as np

from pyscarcopula._constants import PSEUDO_OBS_EPS
from pyscarcopula._utils import pobs, broadcast as _broadcast  # noqa: F401


def _xtanh_transform(x, offset):
    values = np.asarray(x, dtype=np.float64)
    return values * np.tanh(values) + offset


def _xtanh_dtransform(x):
    values = np.asarray(x, dtype=np.float64)
    tanh_values = np.tanh(values)
    return tanh_values + values * (1.0 - tanh_values * tanh_values)


def _inv_xtanh_transform(r, offset):
    """Return the historical modulus-based latent approximation for xtanh.

    ``x * tanh(x) + offset`` is even, so it has no globally unique inverse.
    This helper intentionally preserves the established positive-branch
    approximation ``abs(r) + offset``. It is an initialization convention,
    not a round-trip inverse of :func:`_xtanh_transform`.
    """
    values = np.atleast_1d(np.asarray(r, dtype=np.float64)).ravel()
    return np.abs(values) + offset


def _softplus_transform(x, offset):
    values = np.asarray(x, dtype=np.float64)
    return np.logaddexp(0.0, values) + offset


def _softplus_dtransform(x):
    values = np.asarray(x, dtype=np.float64)
    out = np.empty_like(values)
    positive = values >= 0.0
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    out[~positive] = exp_values / (1.0 + exp_values)
    return out


def _softplus_inv_transform(r, offset):
    values = np.asarray(r, dtype=np.float64) - offset
    return np.where(
        values > 20.0,
        values,
        np.where(
            values <= 0.0,
            np.log(1e-300),
            np.where(
                values < 1e-8,
                np.log(values),
                np.log(np.expm1(values)),
            ),
        ),
    )


@dataclass(frozen=True)
class CopulaCapabilities:
    """Immutable strategy and numerical capability descriptor."""

    dimension: int | None = None
    supports_pair_ops: bool = False
    supports_native_point_ops: bool = False
    supports_native_mle: bool = False
    supports_gas: bool = False
    supports_scar_ou: bool = False
    supports_latent_grid: bool = False
    supports_conditional_sampling: bool = False
    has_dynamic_scalar_parameter: bool = False


class CopulaBase:
    """Common stateful convenience API for all copula dimensions."""

    _capabilities = CopulaCapabilities()

    def __init__(self, *, name="Copula"):
        self._name = name
        self.fit_result = None

    @property
    def name(self):
        return self._name

    @property
    def dimension(self):
        return None

    @property
    def capabilities(self):
        return replace(self._capabilities, dimension=self.dimension)

    def validate_dimension(self, data):
        u = np.asarray(data)
        if u.ndim != 2:
            raise ValueError(
                f"{type(self).__name__}: data must be 2D, got shape {u.shape}")
        dimension = self.dimension
        if dimension is not None and u.shape[1] != dimension:
            raise ValueError(
                f"{type(self).__name__}: data must have {dimension} columns, "
                f"got shape {u.shape}")
        return u

    @staticmethod
    def list_of_methods():
        from pyscarcopula.strategy._base import list_methods
        return list_methods()

    def mlog_likelihood(self, alpha, u, method='mle', **kwargs):
        from pyscarcopula.strategy._base import get_strategy

        u = np.asarray(u, dtype=np.float64)
        alpha = np.atleast_1d(np.asarray(alpha, dtype=np.float64))
        strategy = get_strategy(method, **kwargs)
        return strategy.objective(self, u, alpha, **kwargs)

    def fit(self, data, method='scar-tm-ou', to_pobs=False, **kwargs):
        from pyscarcopula.api import fit as _api_fit

        u = np.asarray(data, dtype=np.float64)
        if to_pobs:
            u = pobs(u)
        result = _api_fit(self, u, method=method, **kwargs)
        self.fit_result = result
        self._last_u = u
        return result

    def predict(self, n, u=None, rng=None, given=None, horizon='next',
                predictive_r_mode=None, predict_config=None):
        if self.fit_result is None:
            raise ValueError("Fit first")
        from pyscarcopula.api import predict as _api_predict

        u_data = u if u is not None else getattr(self, '_last_u', None)
        if u_data is None:
            raise ValueError(
                "No data for predict. "
                "Either call fit() first or pass u= explicitly.")
        return _api_predict(
            self, u_data, self.fit_result, n,
            rng=rng, given=given, horizon=horizon,
            predictive_r_mode=predictive_r_mode,
            predict_config=predict_config)

    def sample(self, n, u=None, rng=None):
        if self.fit_result is None:
            raise ValueError("Fit first")
        from pyscarcopula.api import sample as _api_sample

        u_data = u if u is not None else getattr(self, '_last_u', None)
        if u_data is None:
            raise ValueError(
                "No data for sample. "
                "Either call fit() first or pass u= explicitly.")
        return _api_sample(self, u_data, self.fit_result, n, rng=rng)

    def save(self, path, *, include_data=True):
        from pyscarcopula.io import save_model

        save_model(self, path, include_data=include_data)

    @classmethod
    def load(cls, path):
        from pyscarcopula.io import load_model

        return load_model(path, expected_type=cls)


class BivariateCopula(CopulaBase):
    """
    Base class for bivariate copulas (dim=2).

    Provides copula evaluation, sampling, and backward-compatible object
    methods for fitting and prediction. The object methods delegate to the
    stateless functions in pyscarcopula.api and store fit_result for
    convenience.

    Built-in families use the shared native adapter for density, derivatives,
    transforms, conditional distributions, inverse conditionals, and grids.
    Subclasses retain family metadata, sampling, and Kendall-tau behavior.

    Estimation methods (via .fit()):
        'mle'        — constant parameter (1 param)
        'scar-tm-ou' — transfer matrix (3 params: kappa, mu, nu)
        'scar-tm-jacobi' - TM for Jacobi Kendall-tau dynamics
        'gas'        — GAS score-driven (3 params: omega, gamma, beta)
        'scar-p-ou'  — MC p-sampler, 'scar-m-ou' — MC m-sampler with EIS

    Parameters
    ----------
    rotate : int
        Copula rotation: 0, 90, 180, or 270 degrees.
    """

    _scar_static_df_mle_initialization = False
    _supports_scar_mixture_h = True
    _capabilities = CopulaCapabilities(
        dimension=2,
        supports_pair_ops=True,
        supports_native_point_ops=True,
        supports_gas=True,
        supports_scar_ou=True,
        supports_latent_grid=True,
        supports_conditional_sampling=True,
        has_dynamic_scalar_parameter=True,
    )

    def __init__(self, rotate: int = 0):
        if rotate not in (0, 90, 180, 270):
            raise ValueError(f"rotate must be 0/90/180/270, got {rotate}")
        super().__init__(name="BivariateCopula")
        self._rotate = rotate
        self._bounds = [(-np.inf, np.inf)]

    @property
    def dimension(self):
        return 2

    @property
    def d(self):
        return 2

    @property
    def name(self):
        return self._name

    @property
    def rotate(self):
        return self._rotate

    @property
    def bounds(self):
        return self._bounds

    @staticmethod
    def list_of_methods():
        from pyscarcopula.strategy._base import list_methods
        return list_methods()

    # ── transform ──────────────────────────────────────────────────
    @staticmethod
    def _native_adapter():
        from pyscarcopula.numerical import copula_native

        return copula_native

    def transform(self, x):
        """Map latent values to the copula parameter domain."""
        return self._native_adapter().transform(self, x)

    def inv_transform(self, r):
        """Map copula parameters to the model's latent convention.

        For ``softplus`` and Gaussian transforms this is a numerical inverse.
        For ``xtanh`` it is the established modulus-based positive-branch
        approximation; because ``x * tanh(x)`` is even, no globally unique
        inverse exists and a transform/inverse round trip is not guaranteed.
        """
        return self._native_adapter().inverse_transform(self, r)

    def dtransform(self, x):
        """Evaluate the derivative of the parameter transform."""
        return self._native_adapter().dtransform(self, x)

    def tau_to_param(self, tau):
        """Map Kendall's tau to the copula parameter."""
        raise NotImplementedError(
            f"tau_to_param is not implemented for {type(self).__name__}"
        )

    def param_to_tau(self, r):
        """Map the copula parameter to Kendall's tau."""
        raise NotImplementedError(
            f"param_to_tau is not implemented for {type(self).__name__}"
        )

    # ── PDF / log-PDF ─────────────────────────────────────────────
    def pdf_unrotated(self, u1, u2, r):
        return self._native_adapter().pdf(
            self, u1, u2, r, unrotated=True)

    def log_pdf_unrotated(self, u1, u2, r):
        return self._native_adapter().log_pdf(
            self, u1, u2, r, unrotated=True)

    def pdf(self, u1, u2, r):
        return self._native_adapter().pdf(self, u1, u2, r)

    def log_pdf(self, u1, u2, r):
        return self._native_adapter().log_pdf(self, u1, u2, r)

    def dlog_pdf_dr_unrotated(self, u1, u2, r):
        """Evaluate the native derivative of log density with respect to r."""
        return self._native_adapter().dlog_pdf_dr(
            self, u1, u2, r, unrotated=True)

    def dlog_pdf_dr(self, u1, u2, r):
        """d(log c)/dr with rotation applied."""
        return self._native_adapter().dlog_pdf_dr(self, u1, u2, r)

    def _apply_rotation(self, u1, u2):
        rot = self._rotate
        if rot == 0:
            return u1, u2
        elif rot == 90:
            return 1.0 - u1, u2
        elif rot == 180:
            return 1.0 - u1, 1.0 - u2
        else:
            return u1, 1.0 - u2

    # ── sampling ──────────────────────────────────────────────────
    def psi(self, t, r):
        """Inverse generator (Laplace-Stieltjes)."""
        return np.exp(-t)

    def V(self, n, r, rng=None):
        """Sample from F = LS^{-1}(psi). Override per copula."""
        return np.ones(n)

    def sample_at_parameter(self, n, r, rng=None):
        """
        Sample at an explicitly supplied copula parameter.

        This is the low-level counterpart of :meth:`sample`, which reproduces
        a fitted model.

        r: scalar or array (n,).
        Returns (n, 2).
        """
        if rng is None:
            rng = np.random.default_rng()

        _r = np.atleast_1d(np.asarray(r, dtype=np.float64))
        if _r.size == 1:
            _r = np.full(n, _r[0])

        x = rng.uniform(0, 1, size=(n, 2))
        V_data = np.clip(self.V(n, _r, rng=rng), 1e-50, None)

        u = np.empty((n, 2))
        u[:, 0] = self.psi(-np.log(x[:, 0]) / V_data, _r)
        u[:, 1] = self.psi(-np.log(x[:, 1]) / V_data, _r)

        rot = self._rotate
        if rot == 90:
            u[:, 0] = 1.0 - u[:, 0]
        elif rot == 180:
            u[:, 0] = 1.0 - u[:, 0]
            u[:, 1] = 1.0 - u[:, 1]
        elif rot == 270:
            u[:, 1] = 1.0 - u[:, 1]

        return u

    # ── h-functions ───────────────────────────────────────────────
    def h_unrotated(self, u, v, r):
        return self._native_adapter().h(
            self, u, v, r, unrotated=True)

    def h_inverse_unrotated(self, u, v, r):
        return self._native_adapter().h_inverse(
            self, u, v, r, unrotated=True)

    def _h_inverse_bisection(self, u, v, r, tol=1e-10, maxiter=60):
        """
        Numerical inversion: find t such that h(t, v, r) = u.
        Uses bisection on [eps, 1-eps].
        """
        u = np.atleast_1d(np.asarray(u, dtype=np.float64))
        v = np.atleast_1d(np.asarray(v, dtype=np.float64))
        r = np.atleast_1d(np.asarray(r, dtype=np.float64))
        n = max(len(u), len(v), len(r))
        if len(u) == 1 and n > 1:
            u = np.full(n, u[0])
        if len(v) == 1 and n > 1:
            v = np.full(n, v[0])
        if len(r) == 1 and n > 1:
            r = np.full(n, r[0])

        lo = np.full(n, PSEUDO_OBS_EPS)
        hi = np.full(n, 1.0 - PSEUDO_OBS_EPS)

        for _ in range(maxiter):
            mid = 0.5 * (lo + hi)
            h_mid = self.h_unrotated(mid, v, r)
            mask = h_mid < u
            lo = np.where(mask, mid, lo)
            hi = np.where(mask, hi, mid)
            if np.max(hi - lo) < tol:
                break

        return 0.5 * (lo + hi)

    def h(self, u, v, r):
        return self._native_adapter().h(self, u, v, r)

    def h_pair(self, u, v, r):
        """Evaluate both conditional directions in one native call."""
        return self._native_adapter().h_pair(self, u, v, r)

    def h_inverse(self, u, v, r):
        return self._native_adapter().h_inverse(self, u, v, r)
        
    # ── log-likelihood ────────────────────────────────────────────
    def log_likelihood(self, u, r):
        """u: (T, 2), r: scalar or (T,)."""
        r_arr = np.atleast_1d(np.asarray(r, dtype=np.float64)).ravel()
        if r_arr.size == 1:
            from pyscarcopula.numerical import static_likelihood
            return static_likelihood.prepare(self, u).log_likelihood(
                float(r_arr[0]))
        return np.sum(self.log_pdf(u[:, 0], u[:, 1], r))

    # ── evaluate pdf on a grid of latent states (for transfer matrix) ──
    def pdf_on_grid(self, u_row, z_grid):
        """
        c(u_row; Psi(z_j)) for each z_j in z_grid.
        u_row: (2,), z_grid: (K,). Returns (K,).
        """
        u = np.asarray(u_row, dtype=np.float64).reshape(1, 2)
        return self._native_adapter().pdf_grid(self, u, z_grid)[0]

    def pdf_and_grad_on_grid(self, u_row, z_grid):
        """
        Compute fi(z) and dfi/dz on the grid analytically.

        Uses chain rule: dfi/dz = fi * d(log c)/dr * Psi'(z).

        u_row: (2,), z_grid: (K,).
        Returns (fi, dfi_dz) each of shape (K,).
        """
        u = np.asarray(u_row, dtype=np.float64).reshape(1, 2)
        fi, dfi_dz = self._native_adapter().pdf_and_grad_grid(
            self, u, z_grid)
        return fi[0], dfi_dz[0]

    def pdf_and_grad_on_grid_batch(self, u, x_grid):
        """
        Batch version: compute fi and dfi_dx for all T observations.

        u : (T, 2), x_grid : (K,).
        Returns (fi, dfi_dx) each of shape (T, K).

        Evaluation is fused in the native backend.
        """
        return self._native_adapter().pdf_and_grad_grid(self, u, x_grid)

    def copula_grid_batch(self, u, x_grid):
        """
        Batch version of pdf_on_grid (value only, no gradient).

        u : (T, 2), x_grid : (K,).
        Returns fi of shape (T, K).

        Evaluation is fused in the native backend.
        """
        return self._native_adapter().pdf_grid(self, u, x_grid)

    # ══════════════════════════════════════════════════════════════
    # Negative log-likelihood evaluation (convenience)
    # ══════════════════════════════════════════════════════════════

    # ══════════════════════════════════════════════════════════════
    # Fit — delegates to api.fit() / strategy
    # ══════════════════════════════════════════════════════════════

    def predict(self, n, u=None, rng=None, given=None, horizon='next',
                predictive_r_mode=None, predict_config=None):
        """Sample n observations for next-step prediction.

        Delegates to api.predict() which dispatches to the correct
        strategy (MLE/SCAR-TM/GAS/SCAR-MC). For bivariate copulas,
        ``given`` may fix coordinate 0 or 1 in pseudo-observation space;
        the remaining coordinate is sampled conditionally through the
        fitted copula h-function.

        Parameters
        ----------
        n : int
            Number of predictive samples.
        u : (T, 2) array-like or None
            Prediction history. If ``None``, use data from the last
            :meth:`fit` call.
        rng : numpy.random.Generator or None
            Random number generator.
        given : dict[int, float] or None
            Optional fixed pseudo-observation coordinate. Keys must be 0 or 1,
            values must lie in ``(0, 1)``. If both coordinates are fixed, the
            returned samples repeat those values.
        horizon : {'current', 'next'}
            Predictive state timing for GAS and SCAR-TM.
        predictive_r_mode : {'grid', 'histogram'} or None
            SCAR-TM predictive parameter sampling mode.
        predict_config : PredictConfig or None
            Optional bundled prediction configuration.

        Returns
        -------
        (n, 2) pseudo-observations
        """
        if self.fit_result is None:
            raise ValueError("Fit first")

        from pyscarcopula.api import predict as _api_predict

        u_data = u if u is not None else getattr(self, '_last_u', None)
        if u_data is None:
            raise ValueError(
                "No data for predict. "
                "Either call fit() first or pass u= explicitly.")
        return _api_predict(
            self, u_data, self.fit_result, n,
            rng=rng, given=given, horizon=horizon,
            predictive_r_mode=predictive_r_mode,
            predict_config=predict_config)

    def sample(self, n, u=None, rng=None):
        """Generate n observations reproducing the fitted model.

        Delegates to api.sample() which dispatches to the correct
        strategy. fit(copula, sample(...)) should recover
        similar parameters.

        Parameters
        ----------
        n : int
            Number of observations.
        u : (T, 2) array-like or None
            Reference fitted history. If ``None``, use data from the last
            :meth:`fit` call.
        rng : numpy.random.Generator or None
            Random number generator.

        Returns
        -------
        (n, 2) pseudo-observations
        """
        if self.fit_result is None:
            raise ValueError("Fit first")

        from pyscarcopula.api import sample as _api_sample

        u_data = u if u is not None else getattr(self, '_last_u', None)
        if u_data is None:
            raise ValueError(
                "No data for sample. "
                "Either call fit() first or pass u= explicitly.")
        return _api_sample(self, u_data, self.fit_result, n, rng=rng)
