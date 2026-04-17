"""Grid-based posterior helpers for R-vine conditional sampling."""

from __future__ import annotations

import numpy as np

from pyscarcopula.vine._conditional_scalar import (
    _FAMILY_CLAYTON,
    _FAMILY_FRANK,
    _FAMILY_GAUSSIAN,
    _FAMILY_GUMBEL,
    _FAMILY_INDEPENDENT,
    _FAMILY_JOE,
    _clip_scalar,
    _gauss_h_inv_numba,
    _gauss_h_numba,
    _gauss_log_pdf_numba,
)
from pyscarcopula.copula.clayton import _clayton_h, _clayton_h_inv, _clayton_pdf
from pyscarcopula.copula.frank import _frank_h, _frank_h_inv, _frank_pdf
from pyscarcopula.copula.gumbel import (
    _gumbel_h, _gumbel_h_inverse_newton, _gumbel_pdf,
)
from pyscarcopula.copula.joe import _joe_h, _joe_h_inverse_newton, _joe_pdf
from pyscarcopula.vine._helpers import _clip_unit


def _decode_grid_index(flat_idx, dim, grid_size):
    cells = np.empty(dim, dtype=np.int64)
    val = int(flat_idx)
    for pos in range(dim - 1, -1, -1):
        cells[pos] = val % grid_size
        val //= grid_size
    return cells


def _r_array_vector(r_array, n):
    if r_array.size == 0:
        raise ValueError("Empty edge parameter array")
    if r_array.size == 1:
        return np.full(n, float(r_array[0]), dtype=np.float64)
    if r_array.size < n:
        return np.full(n, float(r_array[0]), dtype=np.float64)
    return r_array[:n]


def _unrotated_pdf_vector(family_id, copula, u, v, r):
    if family_id == _FAMILY_INDEPENDENT:
        return np.ones_like(u, dtype=np.float64)
    if family_id == _FAMILY_CLAYTON:
        return _clayton_pdf(u, v, r)
    if family_id == _FAMILY_FRANK:
        return _frank_pdf(u, v, r)
    if family_id == _FAMILY_GUMBEL:
        return _gumbel_pdf(u, v, r)
    if family_id == _FAMILY_JOE:
        return _joe_pdf(u, v, r)
    if family_id == _FAMILY_GAUSSIAN:
        return np.exp(_gauss_log_pdf_numba(u, v, r))
    return np.asarray(copula.pdf(u, v, r), dtype=np.float64)


def _unrotated_h_vector(family_id, copula, u, v, r):
    if family_id == _FAMILY_INDEPENDENT:
        return u
    if family_id == _FAMILY_CLAYTON:
        return _clayton_h(u, v, r)
    if family_id == _FAMILY_FRANK:
        return _frank_h(u, v, r)
    if family_id == _FAMILY_GUMBEL:
        return _gumbel_h(u, v, r)
    if family_id == _FAMILY_JOE:
        return _joe_h(u, v, r)
    if family_id == _FAMILY_GAUSSIAN:
        return _gauss_h_numba(u, v, r)
    return np.asarray(copula.h_unrotated(u, v, r), dtype=np.float64)


def _unrotated_h_inverse_vector(family_id, copula, u, v, r):
    if family_id == _FAMILY_INDEPENDENT:
        return u
    if family_id == _FAMILY_CLAYTON:
        return _clayton_h_inv(u, v, r)
    if family_id == _FAMILY_FRANK:
        return _frank_h_inv(u, v, r)
    if family_id == _FAMILY_GUMBEL:
        return _gumbel_h_inverse_newton(u, v, r)
    if family_id == _FAMILY_JOE:
        return _joe_h_inverse_newton(u, v, r)
    if family_id == _FAMILY_GAUSSIAN:
        return _gauss_h_inv_numba(u, v, r)
    return np.asarray(copula.h_inverse_unrotated(u, v, r), dtype=np.float64)


def _copula_pdf_vector(family_id, rot, copula, u, v, r):
    if rot == 0:
        u0, v0 = u, v
    elif rot == 90:
        u0, v0 = 1.0 - u, v
    elif rot == 180:
        u0, v0 = 1.0 - u, 1.0 - v
    else:
        u0, v0 = u, 1.0 - v
    return _unrotated_pdf_vector(family_id, copula, u0, v0, r)


def _copula_h_vector(family_id, rot, copula, u, v, r):
    if rot == 0:
        return _unrotated_h_vector(family_id, copula, u, v, r)
    if rot == 90:
        return 1.0 - _unrotated_h_vector(family_id, copula, 1.0 - u, v, r)
    if rot == 180:
        return 1.0 - _unrotated_h_vector(
            family_id, copula, 1.0 - u, 1.0 - v, r)
    return _unrotated_h_vector(family_id, copula, u, 1.0 - v, r)


def _copula_h_inverse_vector(family_id, rot, copula, u, v, r):
    if rot == 0:
        return _unrotated_h_inverse_vector(family_id, copula, u, v, r)
    if rot == 90:
        return 1.0 - _unrotated_h_inverse_vector(
            family_id, copula, 1.0 - u, v, r)
    if rot == 180:
        return 1.0 - _unrotated_h_inverse_vector(
            family_id, copula, 1.0 - u, 1.0 - v, r)
    return _unrotated_h_inverse_vector(family_id, copula, u, 1.0 - v, r)


def column_ops_have_static_r(column_ops):
    for ops in column_ops:
        for op in ops:
            r_array = op[0]
            if r_array.size > 1 and not np.all(r_array == r_array[0]):
                return False
    return True


def column_from_x_many(pseudo, s, x_val, n, column_ops):
    updates = {}
    base = np.asarray(x_val, dtype=np.float64)
    if base.ndim == 0:
        base = np.full(n, float(base), dtype=np.float64)
    else:
        base = base[:n].astype(np.float64, copy=False)
    base = _clip_unit(base)
    base_key = column_ops.base_keys_by_col[s]
    updates[base_key] = base

    ops = column_ops[s]
    density = np.ones(n, dtype=np.float64)
    if not ops:
        return updates, density

    for op in ops:
        (r_array, copula, family_id, rot, _, _, _, _) = op[:8]
        _, partner_key, var_cond_key, var_next_key, partner_next_key = op[8:13]

        partner_val = pseudo.get(partner_key)
        if partner_val is None:
            raise RuntimeError(
                "Missing partner pseudo-observation during vectorized "
                "RVine conditional construction"
            )

        var_at_cond = updates.get(var_cond_key, updates[base_key])
        r = _r_array_vector(r_array, n)
        pdf = _copula_pdf_vector(
            family_id, rot, copula, var_at_cond, partner_val, r)
        cur = _copula_h_vector(
            family_id, rot, copula, var_at_cond, partner_val, r)
        rev = _copula_h_vector(
            family_id, rot, copula, partner_val, var_at_cond, r)

        density *= np.maximum(pdf, 1e-300)
        updates[var_next_key] = _clip_unit(cur)
        updates[partner_next_key] = _clip_unit(rev)

    return updates, density


def column_from_w_many(pseudo, s, w_val, n, column_ops):
    ops = column_ops[s]
    if not ops:
        return column_from_x_many(pseudo, s, w_val, n, column_ops)

    val = np.asarray(w_val, dtype=np.float64)
    if val.ndim == 0:
        val = np.full(n, float(val), dtype=np.float64)
    else:
        val = val[:n].astype(np.float64, copy=False)

    for op in reversed(ops):
        (r_array, copula, family_id, rot, _, _, _, _) = op[:8]
        partner_key = op[9]
        partner_val = pseudo.get(partner_key)
        if partner_val is None:
            raise RuntimeError(
                "Missing partner pseudo-observation during vectorized "
                "RVine conditional inversion"
            )
        r = _r_array_vector(r_array, n)
        val = _clip_unit(
            _copula_h_inverse_vector(
                family_id, rot, copula, val, partner_val, r))

    return column_from_x_many(pseudo, s, val, n, column_ops)


def joint_grid_weights_many(order_cols, order_vars, posterior_idxs, w_values,
                            given, n, column_ops, posterior_pos_by_idx,
                            last_idx):
    pseudo = {}
    density = np.ones(n, dtype=np.float64)

    for idx in range(last_idx + 1):
        s = order_cols[idx]
        var = order_vars[idx]

        if var in given:
            updates, column_density = column_from_x_many(
                pseudo, s, given[var], n, column_ops)
            density *= column_density
        else:
            pos = posterior_pos_by_idx[idx]
            updates, _ = column_from_w_many(
                pseudo, s, w_values[pos], n, column_ops)

        pseudo.update(updates)

    return density


def build_joint_posterior_grid_cache_many(order_cols, order_vars,
                                          posterior_idxs, given, grid_size,
                                          n, column_ops):
    dim = len(posterior_idxs)
    centers = (np.arange(grid_size, dtype=np.float64) + 0.5) / grid_size
    total = int(grid_size ** dim)
    masses = np.empty((n, total), dtype=np.float64)
    cells = np.zeros(dim, dtype=np.int64)
    w_values = np.empty(dim, dtype=np.float64)
    posterior_pos_by_idx = {
        idx: pos for pos, idx in enumerate(posterior_idxs)
    }
    last_idx = max(
        [idx for idx, var in enumerate(order_vars) if var in given]
        + list(posterior_idxs)
    )

    for flat_idx in range(total):
        for pos in range(dim):
            w_values[pos] = centers[cells[pos]]
        mass = joint_grid_weights_many(
            order_cols, order_vars, posterior_idxs, w_values, given, n,
            column_ops, posterior_pos_by_idx, last_idx)
        masses[:, flat_idx] = np.where(
            np.isfinite(mass) & (mass > 0.0), mass, 0.0)
        for pos in range(dim - 1, -1, -1):
            cells[pos] += 1
            if cells[pos] < grid_size:
                break
            cells[pos] = 0

    total_mass = np.sum(masses, axis=1)
    cdf = np.cumsum(masses, axis=1)
    return posterior_idxs, grid_size, dim, total, cdf, total_mass


def draw_joint_posterior_grid_batch(cache, n, rng):
    posterior_idxs, grid_size, dim, total, cdf, total_mass = cache
    out = {
        idx: np.empty(n, dtype=np.float64)
        for idx in posterior_idxs
    }

    static_cdf = cdf.ndim == 1
    for sample_idx in range(n):
        z_total = float(total_mass if np.ndim(total_mass) == 0
                        else total_mass[sample_idx])
        if not np.isfinite(z_total) or z_total <= 0.0:
            for idx in posterior_idxs:
                out[idx][sample_idx] = float(rng.uniform(0.0, 1.0))
            continue

        target = float(rng.uniform(0.0, z_total))
        row_cdf = cdf if static_cdf else cdf[sample_idx]
        flat_idx = int(np.searchsorted(row_cdf, target, side='right'))
        if flat_idx >= total:
            flat_idx = total - 1

        cells = _decode_grid_index(flat_idx, dim, grid_size)
        for pos, idx in enumerate(posterior_idxs):
            w = (float(cells[pos]) + float(rng.uniform(0.0, 1.0))) / grid_size
            out[idx][sample_idx] = _clip_scalar(w)

    return out


def sample_rvine_conditional_grid_many(vine, n, order_cols, order_vars,
                                       given, rng, column_ops, posterior_w):
    d = vine.d
    pseudo = {}
    x = np.zeros((n, d), dtype=np.float64)

    for idx, s in enumerate(order_cols):
        var = order_vars[idx]

        if var in given:
            updates, _ = column_from_x_many(
                pseudo, s, given[var], n, column_ops)
        elif idx in posterior_w:
            updates, _ = column_from_w_many(
                pseudo, s, posterior_w[idx], n, column_ops)
        else:
            updates, _ = column_from_w_many(
                pseudo, s, rng.uniform(0.0, 1.0, size=n), n, column_ops)

        pseudo.update(updates)

    for var in range(d):
        key = column_ops.base_keys_by_var[var]
        if key not in pseudo:
            raise RuntimeError(f"Missing sampled variable {var}")
        x[:, var] = pseudo[key]

    return x
