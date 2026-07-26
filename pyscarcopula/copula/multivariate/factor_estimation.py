"""Shared low-rank correlation initialization from pseudo-observations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import qr
from scipy.special import expit
from scipy.stats import norm


_FACTOR_SCORE_WORK_BYTES = 32 * 1024**2
_JOINT_DIAGONAL_RAW_FLOOR = 1e-10


def _softplus(value):
    return np.log1p(np.exp(-np.abs(value))) + np.maximum(value, 0.0)


def _inverse_softplus(value):
    value = np.asarray(value, dtype=np.float64)
    return value + np.log(-np.expm1(-value))


def _sigmoid(value):
    return expit(value)


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
        max_norm = float(np.sqrt(np.nextafter(
            1.0 - float(uniqueness_min), 0.0)))

        _, _, pivots = qr(
            loadings.T, mode="economic", pivoting=True)
        anchors = np.asarray(pivots[:rank], dtype=np.int64)
        anchor_block = loadings[anchors]
        rotation, _ = np.linalg.qr(anchor_block.T)
        canonical = np.asarray(loadings @ rotation, dtype=np.float64)
        signs = np.where(
            np.diag(canonical[anchors]) < 0.0, -1.0, 1.0)
        canonical *= signs[None, :]

        anchor_floor = min(1e-6, max_norm * 1e-3)
        for order, row in enumerate(anchors):
            canonical[row, order + 1:] = 0.0
            canonical[row, order] = max(
                canonical[row, order], anchor_floor)
            norm_squared = float(canonical[row] @ canonical[row])
            if norm_squared >= max_norm * max_norm:
                canonical[row] *= (
                    max_norm * (1.0 - 1e-8)
                    / np.sqrt(norm_squared))

        row_norm_squared = np.einsum(
            "ij,ij->i", canonical, canonical, optimize=False)
        raw = canonical / np.sqrt(
            np.maximum(
                max_norm * max_norm - row_norm_squared,
                np.finfo(np.float64).tiny,
            ))[:, None]

        anchor_order = np.full(dimension, -1, dtype=np.int64)
        anchor_order[anchors] = np.arange(rank, dtype=np.int64)
        free_rows = []
        free_columns = []
        diagonal_entries = []
        parameters = []
        for row in range(dimension):
            order = int(anchor_order[row])
            stop = order + 1 if order >= 0 else rank
            for column in range(stop):
                is_diagonal = order >= 0 and column == order
                free_rows.append(row)
                free_columns.append(column)
                diagonal_entries.append(is_diagonal)
                value = raw[row, column]
                if is_diagonal:
                    value = _inverse_softplus(
                        max(
                            value - _JOINT_DIAGONAL_RAW_FLOOR,
                            np.finfo(np.float64).eps,
                        ))
                parameters.append(float(value))

        parameterization = cls(
            dimension=dimension,
            rank=rank,
            uniqueness_min=float(uniqueness_min),
            anchors=anchors,
            free_rows=np.asarray(free_rows, dtype=np.int64),
            free_columns=np.asarray(free_columns, dtype=np.int64),
            diagonal_entries=np.asarray(
                diagonal_entries, dtype=np.bool_),
            max_norm=max_norm,
        )
        return parameterization, np.asarray(
            parameters, dtype=np.float64)

    @property
    def n_parameters(self):
        return int(self.free_rows.size)

    def _raw_matrix(self, parameters):
        parameters = np.asarray(parameters, dtype=np.float64)
        if (
                parameters.shape != (self.n_parameters,)
                or np.any(~np.isfinite(parameters))):
            raise ValueError(
                f"factor parameters must have shape "
                f"({self.n_parameters},) and be finite")
        values = parameters.copy()
        values[self.diagonal_entries] = (
            _softplus(values[self.diagonal_entries])
            + _JOINT_DIAGONAL_RAW_FLOOR)
        raw = np.zeros(
            (self.dimension, self.rank), dtype=np.float64)
        raw[self.free_rows, self.free_columns] = values
        return raw

    def loadings(self, parameters):
        raw = self._raw_matrix(parameters)
        denominator = np.sqrt(
            1.0 + np.einsum(
                "ij,ij->i", raw, raw, optimize=False))
        return self.max_norm * raw / denominator[:, None]

    def pullback(self, parameters, loading_gradient):
        raw = self._raw_matrix(parameters)
        loading_gradient = np.asarray(
            loading_gradient, dtype=np.float64)
        if loading_gradient.shape != raw.shape:
            raise ValueError(
                f"loading_gradient must have shape {raw.shape}")
        denominator = np.sqrt(
            1.0 + np.einsum(
                "ij,ij->i", raw, raw, optimize=False))
        projection = np.einsum(
            "ij,ij->i", raw, loading_gradient, optimize=False)
        raw_gradient = self.max_norm * (
            loading_gradient / denominator[:, None]
            - raw
                * (projection / (denominator ** 3))[:, None]
        )
        result = raw_gradient[
            self.free_rows, self.free_columns].copy()
        result[self.diagonal_entries] *= _sigmoid(
            np.asarray(parameters)[self.diagonal_entries])
        return result


def _factor_score_block(u, start, stop):
    probabilities = np.clip(
        u[:, start:stop],
        np.finfo(np.float64).eps,
        1.0 - np.finfo(np.float64).eps,
    )
    scores = norm.ppf(probabilities)
    scores -= np.mean(scores, axis=0)
    scale = np.sqrt(np.mean(scores * scores, axis=0))
    threshold = np.sqrt(np.finfo(np.float64).eps)
    np.divide(
        scores,
        scale,
        out=scores,
        where=scale > threshold,
    )
    scores[:, scale <= threshold] = 0.0
    return scores


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
    score_tile = min(
        int(dimension_tile),
        max(
            1,
            _FACTOR_SCORE_WORK_BYTES
            // (n_observations * np.dtype(np.float64).itemsize),
        ),
    )

    rng = np.random.default_rng(int(seed))
    random_projection = rng.standard_normal(
        (dimension, subspace_size))
    sample_subspace = np.zeros(
        (n_observations, subspace_size), dtype=np.float64)
    for start in range(0, dimension, score_tile):
        stop = min(dimension, start + score_tile)
        scores = _factor_score_block(u, start, stop)
        sample_subspace += scores @ random_projection[start:stop]
    del random_projection
    sample_subspace, _ = np.linalg.qr(
        sample_subspace, mode="reduced")

    variable_subspace = np.empty(
        (dimension, subspace_size), dtype=np.float64)
    for start in range(0, dimension, score_tile):
        stop = min(dimension, start + score_tile)
        scores = _factor_score_block(u, start, stop)
        variable_subspace[start:stop] = scores.T @ sample_subspace
    sample_subspace.fill(0.0)
    for start in range(0, dimension, score_tile):
        stop = min(dimension, start + score_tile)
        scores = _factor_score_block(u, start, stop)
        sample_subspace += scores @ variable_subspace[start:stop]
    sample_subspace, _ = np.linalg.qr(
        sample_subspace, mode="reduced")
    del variable_subspace

    compressed = np.empty(
        (subspace_size, dimension), dtype=np.float64)
    for start in range(0, dimension, score_tile):
        stop = min(dimension, start + score_tile)
        scores = _factor_score_block(u, start, stop)
        compressed[:, start:stop] = sample_subspace.T @ scores
    _, singular_values, right_vectors = np.linalg.svd(
        compressed, full_matrices=False)
    eigenvalues = (
        singular_values[:rank] * singular_values[:rank]
        / float(n_observations)
    )
    scales = np.sqrt(np.maximum(eigenvalues - 1.0, 0.0))
    loadings = right_vectors[:rank].T * scales

    max_norm = np.sqrt(np.nextafter(
        1.0 - float(uniqueness_min), 0.0))
    row_norms = np.linalg.norm(loadings, axis=1)
    row_scale = np.minimum(
        1.0,
        np.divide(
            max_norm,
            row_norms,
            out=np.ones_like(row_norms),
            where=row_norms > 0.0,
        ),
    )
    return loadings * row_scale[:, None], {
        "source": "two_stage_randomized_svd",
        "n_observations": int(n_observations),
        "random_seed": int(seed),
        "oversampling": int(subspace_size - rank),
        "power_iterations": 1,
        "dimension_tile": int(score_tile),
        "configured_dimension_tile": int(dimension_tile),
        "leading_eigenvalues": eigenvalues,
    }


__all__ = ["estimate_factor_loadings"]
