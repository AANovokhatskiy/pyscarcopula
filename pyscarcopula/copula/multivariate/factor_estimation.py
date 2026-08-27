"""Shared low-rank correlation initialization from pseudo-observations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FactorLoadingParameterization:
    """Identifiable smooth coordinates for joint factor optimization.

    Pivoted anchor rows form a lower-triangular block with positive diagonal,
    removing the rotational degrees of freedom. Each raw row is mapped
    smoothly inside the uniqueness boundary.
    """

    dimension: int
    rank: int
    uniqueness_min: float
    anchors: np.ndarray
    free_rows: np.ndarray
    free_columns: np.ndarray
    diagonal_entries: np.ndarray
    max_norm: float

    @classmethod
    def from_loadings(cls, loadings, *, uniqueness_min):
        loadings = np.asarray(loadings, dtype=np.float64)
        if loadings.ndim != 2:
            raise ValueError("loadings must be a two-dimensional array")
        dimension, rank = loadings.shape
        if not 1 <= rank < dimension or not np.all(np.isfinite(loadings)):
            raise ValueError(
                "loadings must have shape (d, k), 1 <= k < d, and be finite")
        from pyscarcopula._native import multivariate as multivariate_native
        native = multivariate_native.factor_parameterization_from_loadings(
            loadings, uniqueness_min)

        parameterization = cls(
            dimension=dimension,
            rank=rank,
            uniqueness_min=float(uniqueness_min),
            anchors=native["anchors"],
            free_rows=native["free_rows"],
            free_columns=native["free_columns"],
            diagonal_entries=native["diagonal_entries"],
            max_norm=native["max_norm"],
        )
        return parameterization, native["parameters"]

    @property
    def n_parameters(self):
        return int(self.free_rows.size)

    def loadings(self, parameters):
        from pyscarcopula._native import multivariate as multivariate_native
        return multivariate_native.factor_parameterization_loadings(
            parameters,
            free_rows=self.free_rows,
            free_columns=self.free_columns,
            diagonal_entries=self.diagonal_entries,
            dimension=self.dimension,
            rank=self.rank,
            max_norm=self.max_norm,
        )

    def pullback(self, parameters, loading_gradient):
        from pyscarcopula._native import multivariate as multivariate_native
        return multivariate_native.factor_parameterization_pullback(
            parameters,
            loading_gradient,
            free_rows=self.free_rows,
            free_columns=self.free_columns,
            diagonal_entries=self.diagonal_entries,
            dimension=self.dimension,
            rank=self.rank,
            max_norm=self.max_norm,
        )


def estimate_factor_loadings(
        u,
        rank,
        *,
        uniqueness_min,
        dimension_tile,
        seed,
        oversampling):
    """Estimate loadings by tiled randomized SVD without dense covariance."""
    u = np.asarray(u, dtype=np.float64)
    if u.ndim != 2 or u.shape[1] < 2 or u.shape[0] == 0:
        raise ValueError(
            "u must have shape (n_observations, dimension), dimension >= 2")
    if not np.all(np.isfinite(u)) or np.any((u < 0.0) | (u > 1.0)):
        raise ValueError(
            "u must contain finite pseudo-observations in [0, 1]")

    n_observations, dimension = u.shape
    rank = int(rank)
    if n_observations <= rank:
        raise ValueError(
            "two-stage factor initialization requires "
            "n_observations > factor_rank")
    subspace_size = min(
        dimension, n_observations, rank + int(oversampling))
    if subspace_size < rank:
        raise ValueError("factor_rank exceeds the estimable data rank")
    rng = np.random.default_rng(int(seed))
    random_projection = rng.standard_normal(
        (dimension, subspace_size))
    from pyscarcopula._native import multivariate as multivariate_native
    loadings, native = (
        multivariate_native.estimate_factor_loadings_from_projection(
            u,
            rank,
            uniqueness_min=uniqueness_min,
            dimension_tile=dimension_tile,
            random_projection=random_projection,
        )
    )
    return loadings, {
        "source": "two_stage_randomized_svd",
        "n_observations": int(n_observations),
        "random_seed": int(seed),
        "oversampling": int(subspace_size - rank),
        "power_iterations": 1,
        "dimension_tile": native["score_tile"],
        "configured_dimension_tile": int(dimension_tile),
        "leading_eigenvalues": native["leading_eigenvalues"],
    }


__all__ = ["estimate_factor_loadings"]
