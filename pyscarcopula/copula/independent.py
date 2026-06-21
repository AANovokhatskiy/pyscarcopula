"""
Independence copula: C(u1, u2) = u1 * u2.

Density c(u1, u2) = 1 everywhere.
h-function: h(u2 | u1) = u2 (no conditioning effect).

This copula has zero parameters and log-likelihood = 0.
Used in vine copulas to prune edges with negligible dependence:
if no parametric copula beats independence by AIC, the edge
is set to independent, saving all fit/TM/GAS computation.
"""

import numpy as np
from pyscarcopula.copula.base import BivariateCopula, CopulaCapabilities


class IndependentCopula(BivariateCopula):
    """
    Independence copula: C(u1, u2) = u1 * u2.

    This is a zero-parameter copula with c(u1, u2) = 1.
    It serves as the null model for vine edge selection:
    edges where no parametric copula beats independence by AIC
    are set to independent, eliminating all fit/forward-pass cost.
    """

    def __init__(self, rotate: int = 0):
        # rotation is meaningless for independence, but accept it
        # to keep the interface uniform
        super().__init__(0)
        self._name = "Independent copula"
        self._bounds = []  # no parameters

    # ── transform (trivial) ──────────────────────────────────────

    def transform(self, x):
        return super().transform(x)

    def inv_transform(self, r):
        return super().inv_transform(r)

    def dtransform(self, x):
        return super().dtransform(x)

    # ── PDF / log-PDF ────────────────────────────────────────────

    def pdf_unrotated(self, u1, u2, r):
        return super().pdf_unrotated(u1, u2, r)

    def log_pdf_unrotated(self, u1, u2, r):
        return super().log_pdf_unrotated(u1, u2, r)

    def dlog_pdf_dr_unrotated(self, u1, u2, r):
        return super().dlog_pdf_dr_unrotated(u1, u2, r)

    # ── h-functions (trivial) ────────────────────────────────────

    def h_unrotated(self, u, v, r):
        """h(u | v) = u for independence."""
        return super().h_unrotated(u, v, r)

    def h_inverse_unrotated(self, u, v, r):
        """h_inverse(u | v) = u for independence."""
        return super().h_inverse_unrotated(u, v, r)

    # ── sampling ─────────────────────────────────────────────────

    def sample_at_parameter(self, n, r=None, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        return rng.uniform(0, 1, size=(n, 2))

    # ── log-likelihood ───────────────────────────────────────────

    def log_likelihood(self, u, r=None):
        return 0.0

    # ── grid evaluations (all trivial) ───────────────────────────

    def pdf_on_grid(self, u_row, z_grid):
        return super().pdf_on_grid(u_row, z_grid)

    def pdf_and_grad_on_grid(self, u_row, z_grid):
        return super().pdf_and_grad_on_grid(u_row, z_grid)

    def pdf_and_grad_on_grid_batch(self, u, x_grid):
        return super().pdf_and_grad_on_grid_batch(u, x_grid)

    def copula_grid_batch(self, u, x_grid):
        return super().copula_grid_batch(u, x_grid)

    # ── fit (instant) ────────────────────────────────────────────

    def fit(self, data, method='mle', to_pobs=False, **kwargs):
        """
        'Fit' the independence copula.

        Always instant: logL = 0, no parameters.
        Returns an IndependentResult.
        """
        from pyscarcopula._types import IndependentResult
        result = IndependentResult(
            log_likelihood=0.0,
            method='MLE',
            copula_name=self._name,
            success=True,
        )
        self.fit_result = result
        self._last_u = np.asarray(data, dtype=np.float64)
        return result
    _capabilities = CopulaCapabilities(
        dimension=2,
        supports_pair_ops=True,
        supports_native_point_ops=True,
        supports_conditional_sampling=True,
    )
