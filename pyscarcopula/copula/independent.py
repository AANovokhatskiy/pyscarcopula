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
from pyscarcopula.copula.base import BivariateCopula


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

    @staticmethod
    def transform(x):
        return np.zeros_like(np.asarray(x, dtype=np.float64))

    @staticmethod
    def inv_transform(r):
        return np.zeros_like(np.asarray(r, dtype=np.float64))

    @staticmethod
    def dtransform(x):
        return np.zeros_like(np.asarray(x, dtype=np.float64))

    # ── PDF / log-PDF ────────────────────────────────────────────

    def pdf_unrotated(self, u1, u2, r):
        u1a = np.atleast_1d(np.asarray(u1, dtype=np.float64))
        return np.ones(len(u1a))

    def log_pdf_unrotated(self, u1, u2, r):
        u1a = np.atleast_1d(np.asarray(u1, dtype=np.float64))
        return np.zeros(len(u1a))

    def dlog_pdf_dr_unrotated(self, u1, u2, r):
        u1a = np.atleast_1d(np.asarray(u1, dtype=np.float64))
        return np.zeros(len(u1a))

    # ── h-functions (trivial) ────────────────────────────────────

    def h_unrotated(self, u, v, r):
        """h(u | v) = u for independence."""
        return np.atleast_1d(np.asarray(u, dtype=np.float64)).copy()

    def h_inverse_unrotated(self, u, v, r):
        """h_inverse(u | v) = u for independence."""
        return np.atleast_1d(np.asarray(u, dtype=np.float64)).copy()

    # ── sampling ─────────────────────────────────────────────────

    def sample(self, n, r=None, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        return rng.uniform(0, 1, size=(n, 2))

    # ── log-likelihood ───────────────────────────────────────────

    def log_likelihood(self, u, r=None):
        return 0.0

    # ── grid evaluations (all trivial) ───────────────────────────

    def pdf_on_grid(self, u_row, z_grid):
        return np.ones(len(z_grid))

    def pdf_and_grad_on_grid(self, u_row, z_grid):
        K = len(z_grid)
        return np.ones(K), np.zeros(K)

    def pdf_and_grad_on_grid_batch(self, u, x_grid):
        n = len(u)
        K = len(x_grid)
        return np.ones((n, K)), np.zeros((n, K))

    def copula_grid_batch(self, u, x_grid):
        return np.ones((len(u), len(x_grid)))

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
        return result
