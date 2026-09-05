"""Shared base classes for bivariate copulas.

Built-in numerical operations are dispatched through the native copula
adapter. Subclasses provide family metadata plus family-specific sampling and
Kendall-tau helpers where needed.
"""

from __future__ import annotations

from os import PathLike
from typing import Any, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pyscarcopula._types import FitResult, PredictConfig
from pyscarcopula._native import model_policy, statistics
from pyscarcopula._utils import broadcast as _broadcast  # noqa: F401
from pyscarcopula.numerical._arrays import as_float64_array


FloatArray = NDArray[np.float64]
CopulaT = TypeVar("CopulaT", bound="CopulaBase")


class CopulaBase:
    """Common stateful convenience API for all copula dimensions.

    The object API stores the latest fit result and training data, then
    delegates numerical work to :mod:`pyscarcopula.api`. Use the functions in
    that module when explicit, stateless data flow is preferable.

    Parameters
    ----------
    name : str
        Human-readable model name used in results and diagnostics.
    """

    def __init__(self, *, name: str = "Copula") -> None:
        self._name = name
        self.fit_result: FitResult | None = None

    @property
    def name(self) -> str:
        """Human-readable copula name."""
        return self._name

    @property
    def dimension(self) -> int | None:
        """Required data width, or ``None`` when not fixed by the class."""
        return None

    def validate_dimension(self, data: ArrayLike) -> np.ndarray:
        """Validate and return a two-dimensional data array.

        Parameters
        ----------
        data : array_like
            Observations with rows as samples and columns as dimensions.

        Returns
        -------
        ndarray
            ``data`` converted with :func:`numpy.asarray`.

        Raises
        ------
        ValueError
            If ``data`` is not two-dimensional or has the wrong width.
        """
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
    def list_of_methods() -> list[str]:
        """Return registered estimation strategy names."""
        from pyscarcopula.strategy._base import list_methods
        return list_methods()

    def mlog_likelihood(
        self,
        alpha: ArrayLike,
        u: ArrayLike,
        method: str = 'mle',
        **kwargs: Any,
    ) -> float:
        """Evaluate a strategy's negative log-likelihood objective.

        Parameters use the strategy's objective representation (physical
        ``[kappa, mu, nu]`` for SCAR-TM-OU). Fit arguments such as ``alpha0``,
        ``initial_mle_result`` and ``maxiter`` are not accepted here.
        """
        from pyscarcopula.strategy._base import (
            get_strategy,
            partition_strategy_operation_kwargs,
        )

        u = as_float64_array(u, name="u")
        alpha = np.atleast_1d(as_float64_array(alpha, name="alpha"))
        config = kwargs.pop("config", None)
        constructor_kwargs, objective_kwargs = (
            partition_strategy_operation_kwargs(method, "objective", kwargs)
        )
        strategy = get_strategy(
            method,
            config=config,
            **constructor_kwargs,
        )
        return strategy.objective(
            self,
            u,
            alpha,
            **objective_kwargs,
        )

    def fit(
        self,
        data: ArrayLike,
        method: str = 'scar-tm-ou',
        to_pobs: bool = False,
        **kwargs: Any,
    ) -> FitResult:
        """Fit this copula and retain the result for convenience methods.

        Parameters
        ----------
        data : array_like
            Raw observations or pseudo-observations.
        method : str
            Registered estimation strategy.
        to_pobs : bool
            Rank-transform columns before fitting.
        **kwargs
            Strategy-specific options forwarded to :func:`pyscarcopula.api.fit`.

        Returns
        -------
        FitResult
            Immutable strategy-specific fit result.
        """
        from pyscarcopula.api import fit as _api_fit

        return _api_fit(
            self,
            data,
            method=method,
            to_pobs=to_pobs,
            **kwargs,
        )

    def predict(
        self,
        n: int,
        u: ArrayLike | None = None,
        rng: np.random.Generator | None = None,
        given: dict[int, float] | None = None,
        horizon: str = 'next',
        predictive_r_mode: str | None = None,
        predict_config: PredictConfig | None = None,
    ) -> FloatArray:
        """Draw samples from the fitted predictive distribution.

        ``u`` defaults to the data retained by :meth:`fit`. ``given`` fixes
        selected coordinates in pseudo-observation space.

        Raises
        ------
        ValueError
            If the model has not been fitted or no prediction history exists.
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

    def sample(
        self,
        n: int,
        u: ArrayLike | None = None,
        rng: np.random.Generator | None = None,
    ) -> FloatArray:
        """Simulate a path reproducing the fitted model.

        ``u`` defaults to the data retained by :meth:`fit` and is used by
        dynamic strategies to initialize their simulation state.

        Raises
        ------
        ValueError
            If the model has not been fitted or no fitted history exists.
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

    def save(
        self,
        path: str | PathLike[str],
        *,
        include_data: bool = True,
    ) -> None:
        """Serialize this model to a versioned JSON document.

        Parameters
        ----------
        path : path-like
            Destination file.
        include_data : bool
            Include retained training data. Disable this to reduce file size
            or avoid persisting source observations.
        """
        from pyscarcopula.io import save_model

        save_model(self, path, include_data=include_data)

    @classmethod
    def load(
        cls: type[CopulaT],
        path: str | PathLike[str],
    ) -> CopulaT:
        """Load a model and require it to be an instance of ``cls``."""
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
    Subclasses retain family metadata and Kendall-tau behavior. Python owns
    RNG draws; the fixed-draw sampling transform is native.

    Estimation methods (via .fit()):
        'mle'        — constant parameter (1 param)
        'scar-tm-ou' — transfer matrix (3 params: kappa, mu, nu)
        'scar-tm-jacobi' - TM for Jacobi Kendall-tau dynamics
        'gas'        — GAS score-driven (3 params: omega, gamma, beta)

    Parameters
    ----------
    rotate : int
        Copula rotation: 0, 90, 180, or 270 degrees.
    """

    _scar_optimizer_config = 'bivariate_scar_optimizer'
    _scar_log_optimizer_config = 'bivariate_log_scar_optimizer'
    _scar_stationary_scale_bounds = model_policy.stationary_scale_bounds()
    _scar_static_df_mle_initialization = False
    _supports_scar_mixture_h = True
    def __init__(self, rotate: int = 0):
        if rotate not in (0, 90, 180, 270):
            raise ValueError(f"rotate must be 0/90/180/270, got {rotate}")
        super().__init__(name="BivariateCopula")
        self._rotate = rotate

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
        from pyscarcopula._native import pair as copula_native

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

    # ── sampling ──────────────────────────────────────────────────
    def sample_at_parameter(self, n, r, rng=None):
        """
        Sample at an explicitly supplied copula parameter.

        This is the low-level counterpart of :meth:`sample`, which reproduces
        a fitted model.

        r: scalar or array (n,).
        Returns (n, 2).

        """
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 0:
            raise ValueError("n must be a non-negative integer")
        if rng is None:
            rng = np.random.default_rng()

        parameter = np.atleast_1d(
            as_float64_array(r, name="r")).ravel()
        family = self._native_pair_family
        if parameter.size == 1:
            parameter = np.full(n, parameter[0])
        elif parameter.size != n:
            raise ValueError(
                f"r must be scalar or array of length {n}, "
                f"got {parameter.size}")

        if family == "Gaussian":
            draws = rng.standard_normal((n, 2))
            return self._native_adapter().sample_from_rng_draws(
                self,
                draws,
                np.empty((0, 0), dtype=np.float64),
                parameter,
            )
        if family in {"Clayton", "Gumbel", "Frank", "Joe", "Independent"}:
            draws = rng.uniform(0, 1, size=(n, 2))
        else:
            raise ValueError(
                f"unsupported native pair sampling family: {family}")
        return self._native_adapter().sample_from_uniforms(
            self, draws, parameter)

    # ── h-functions ───────────────────────────────────────────────
    def h_unrotated(self, u, v, r):
        return self._native_adapter().h(
            self, u, v, r, unrotated=True)

    def h_inverse_unrotated(self, u, v, r):
        return self._native_adapter().h_inverse(
            self, u, v, r, unrotated=True)

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
        r_arr = np.atleast_1d(
            as_float64_array(r, name="r")).ravel()
        if r_arr.size == 1:
            from pyscarcopula._native import static as static_likelihood
            return static_likelihood.prepare(self, u).log_likelihood(
                float(r_arr[0]))
        return statistics.sum_values(self.log_pdf(u[:, 0], u[:, 1], r))

    # ── evaluate pdf on a grid of latent states (for transfer matrix) ──
    def pdf_on_grid(self, u_row, z_grid):
        """
        c(u_row; Psi(z_j)) for each z_j in z_grid.
        u_row: (2,), z_grid: (K,). Returns (K,).
        """
        u = as_float64_array(u_row, name="u_row").reshape(1, 2)
        return self._native_adapter().pdf_grid(self, u, z_grid)[0]

    def pdf_and_grad_on_grid(self, u_row, z_grid):
        """
        Compute fi(z) and dfi/dz on the grid analytically.

        Uses chain rule: dfi/dz = fi * d(log c)/dr * Psi'(z).

        u_row: (2,), z_grid: (K,).
        Returns (fi, dfi_dz) each of shape (K,).
        """
        u = as_float64_array(u_row, name="u_row").reshape(1, 2)
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
