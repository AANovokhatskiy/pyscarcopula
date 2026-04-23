"""Exact conditional predict backend for C-vines."""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss

from pyscarcopula.vine._helpers import _clip_unit


def validate_cvine_given(given, d):
    """Validate `given` for C-vine conditional predict."""
    if given is None:
        return {}

    if not isinstance(given, dict):
        raise TypeError("given must be a dict[int, float] or None")

    out = {}
    for key, value in given.items():
        try:
            idx = int(key)
        except Exception as exc:
            raise TypeError("given keys must be integers") from exc
        if idx < 0 or idx >= d:
            raise ValueError(f"given key must be in [0, {d - 1}], got {key!r}")
        val = float(value)
        if not (0.0 < val < 1.0):
            raise ValueError(
                f"given[{idx}] must be in pseudo-observation space (0, 1), got {val}"
            )
        out[idx] = val

    return out


def ensure_cvine_conditional_supported(vine):
    """Reject conditional predict for unsupported dynamic edge methods."""
    from pyscarcopula._types import IndependentResult
    from pyscarcopula.copula.independent import IndependentCopula

    for tree in vine.edges:
        for edge in tree:
            if (isinstance(edge.copula, IndependentCopula)
                    or isinstance(edge.fit_result, IndependentResult)):
                continue
            method = edge.method.upper() if edge.method is not None else ''
            if method not in ('MLE', 'GAS', 'SCAR-TM-OU'):
                raise NotImplementedError(
                    "Exact conditional predict for CVine is currently "
                    "implemented only for MLE, GAS and SCAR-TM-OU edges"
                )


def is_prefix_conditioning(given):
    """Whether given variables form a prefix {0, ..., k-1}."""
    keys = sorted(given)
    return keys == list(range(len(keys)))


def _edge_r_scalar(r_pred, j, i, sample_idx=0):
    """Extract scalar edge parameter from predict-time r storage."""
    r = np.asarray(r_pred[j][i], dtype=np.float64).ravel()
    if r.size == 0:
        raise ValueError("Empty edge parameter array")
    if sample_idx < 0 or sample_idx >= r.size:
        sample_idx = 0
    return float(r[sample_idx])


def _compute_row_from_x(vine, state_rows, i, x_i, r_pred, sample_idx=0):
    """Build row i pseudo-observations from a realized x_i."""
    row = np.empty(i + 1, dtype=np.float64)
    row[0] = float(_clip_unit(np.asarray([x_i], dtype=np.float64))[0])

    for j_idx in range(i):
        edge = vine.edges[j_idx][i - j_idx - 1]
        r = _edge_r_scalar(r_pred, j_idx, i - j_idx - 1, sample_idx)
        row[j_idx + 1] = float(_clip_unit(
            np.atleast_1d(edge.copula.h(
                np.array([row[j_idx]]),
                np.array([state_rows[j_idx][j_idx]]),
                np.array([r]))
        )[0]))

    return row


def _compute_row_from_w(vine, state_rows, i, w_i, r_pred, sample_idx=0):
    """Build row i pseudo-observations from latent Rosenblatt w_i."""
    val = float(w_i)
    for k in range(i - 1, -1, -1):
        edge = vine.edges[k][i - k - 1]
        r = _edge_r_scalar(r_pred, k, i - k - 1, sample_idx)
        val = float(_clip_unit(np.atleast_1d(edge.copula.h_inverse(
            np.array([val]),
            np.array([state_rows[k][k]]),
            np.array([r]))
        )[0]))

    return _compute_row_from_x(vine, state_rows, i, val, r_pred, sample_idx)


def _conditional_density(vine, state_rows, row, i, r_pred, sample_idx=0):
    """Conditional density of x_i given already known prefix."""
    if i == 0:
        return 1.0

    dens = 1.0
    for k in range(i):
        edge = vine.edges[k][i - k - 1]
        r = _edge_r_scalar(r_pred, k, i - k - 1, sample_idx)
        pdf_val = float(edge.copula.pdf(
            np.array([state_rows[k][k]]),
            np.array([row[k]]),
            np.array([r]))[0])
        dens *= max(pdf_val, 1e-300)
    return dens


def _has_future_given(given, i, d):
    """Whether there is any observed variable at or after index i."""
    for j in range(i, d):
        if j in given:
            return True
    return False


def _quadrature_integral(func, a, b, nodes, weights):
    """Gauss-Legendre integration on [a, b]."""
    if b <= a:
        return 0.0
    mid = 0.5 * (a + b)
    half = 0.5 * (b - a)
    xs = mid + half * nodes
    vals = np.array([func(float(x)) for x in xs], dtype=np.float64)
    return float(half * np.sum(weights * vals))


def _future_likelihood(vine, state_rows, next_i, given, r_pred, nodes, weights,
                       sample_idx=0):
    """Likelihood of future observed variables, integrating unobserved ones out."""
    d = vine.d
    if next_i >= d:
        return 1.0

    if not _has_future_given(given, next_i, d):
        return 1.0

    if next_i in given:
        row = _compute_row_from_x(
            vine, state_rows, next_i, given[next_i], r_pred, sample_idx)
        q = _conditional_density(
            vine, state_rows, row, next_i, r_pred, sample_idx)
        return q * _future_likelihood(
            vine, state_rows + [row], next_i + 1, given, r_pred, nodes, weights,
            sample_idx)

    def _integrand(w):
        row = _compute_row_from_w(
            vine, state_rows, next_i, w, r_pred, sample_idx)
        return _future_likelihood(
            vine, state_rows + [row], next_i + 1, given, r_pred, nodes, weights,
            sample_idx)

    return _quadrature_integral(_integrand, 0.0, 1.0, nodes, weights)


def _sample_w_posterior(vine, state_rows, i, given, r_pred, rng,
                        nodes, weights, sample_idx=0, tol=1e-6, maxiter=40):
    """Sample w_i from its posterior given future observed variables."""
    def weight_fn(w):
        row = _compute_row_from_w(vine, state_rows, i, w, r_pred, sample_idx)
        return _future_likelihood(
            vine, state_rows + [row], i + 1, given, r_pred, nodes, weights,
            sample_idx)

    z_total = _quadrature_integral(weight_fn, 0.0, 1.0, nodes, weights)
    if not np.isfinite(z_total) or z_total <= 0.0:
        return float(rng.uniform(0.0, 1.0))

    target = float(rng.uniform(0.0, z_total))
    lo = 0.0
    hi = 1.0

    for _ in range(maxiter):
        mid = 0.5 * (lo + hi)
        mass = _quadrature_integral(weight_fn, 0.0, mid, nodes, weights)
        if mass < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break

    return 0.5 * (lo + hi)


def sample_cvine_conditional_prefix_with_r(vine, n, r_pred, given, rng):
    """Sample from a fitted C-vine conditional on a prefix of variables."""
    d = vine.d
    n_given = len(given)

    x = np.zeros((n, d), dtype=np.float64)
    v_samp = [[None] * d for _ in range(d)]

    for i in range(n_given):
        x[:, i] = given[i]
        v_samp[i][0] = np.full(n, given[i], dtype=np.float64)

        if i < d - 1:
            for j_idx in range(i):
                edge = vine.edges[j_idx][i - j_idx - 1]
                r = r_pred[j_idx][i - j_idx - 1]
                v_samp[i][j_idx + 1] = _clip_unit(
                    edge.copula.h(v_samp[i][j_idx], v_samp[j_idx][j_idx], r)
                )

    for i in range(n_given, d):
        v_samp[i][0] = rng.uniform(0.0, 1.0, n)

        for k in range(i - 1, -1, -1):
            edge = vine.edges[k][i - k - 1]
            r = r_pred[k][i - k - 1]
            v_samp[i][0] = _clip_unit(
                edge.copula.h_inverse(v_samp[i][0], v_samp[k][k], r)
            )

        x[:, i] = v_samp[i][0]

        if i < d - 1:
            for j_idx in range(i):
                edge = vine.edges[j_idx][i - j_idx - 1]
                r = r_pred[j_idx][i - j_idx - 1]
                v_samp[i][j_idx + 1] = _clip_unit(
                    edge.copula.h(v_samp[i][j_idx], v_samp[j_idx][j_idx], r)
                )

    return x


def sample_cvine_conditional_general_with_r(vine, n, r_pred, given, rng,
                                            quad_order=12):
    """Sample from a fitted C-vine conditional on an arbitrary given set."""
    d = vine.d
    nodes, weights = leggauss(quad_order)
    x = np.zeros((n, d), dtype=np.float64)

    for t in range(n):
        state_rows = []

        for i in range(d):
            if i in given:
                row = _compute_row_from_x(
                    vine, state_rows, i, given[i], r_pred, sample_idx=t)
            elif _has_future_given(given, i + 1, d):
                w_i = _sample_w_posterior(
                    vine, state_rows, i, given, r_pred, rng, nodes, weights,
                    sample_idx=t)
                row = _compute_row_from_w(
                    vine, state_rows, i, w_i, r_pred, sample_idx=t)
            else:
                w_i = float(rng.uniform(0.0, 1.0))
                row = _compute_row_from_w(
                    vine, state_rows, i, w_i, r_pred, sample_idx=t)

            state_rows.append(row)
            x[t, i] = row[0]

    return x
