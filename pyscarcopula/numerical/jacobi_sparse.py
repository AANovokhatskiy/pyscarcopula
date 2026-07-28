"""Sparse local-transition operators for the Jacobi latent process."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from pyscarcopula._utils import clip_h_function_values
from pyscarcopula.numerical._arrays import (
    validate_float64_allocation,
    validate_positive_int,
)
from pyscarcopula.numerical._transition_methods import (
    normalize_jacobi_matrix_transition_method,
)
from pyscarcopula.numerical.jacobi_tm import (
    DEFAULT_JACOBI_MEMORY_BUDGET_BYTES,
    _jacobi_lamperti,
    _jacobi_lamperti_drift_from_tau,
    _jacobi_stationary_shape,
    _emission_grid,
    _fixed_tau_rule,
    _fixed_tau_weight_derivatives,
    _h_grid_on_theta,
    _h_pair_grids_on_theta,
    _normal_hermite_rule,
    _resolve_dt,
    _validate_jacobi_order,
    _validate_nonnegative_int,
    jacobi_rule,
)


@dataclass(frozen=True)
class SparseJacobiTransition:
    """Compact row-wise representation of a local Jacobi transition."""

    indices: np.ndarray
    probabilities: np.ndarray
    counts: np.ndarray

    def __post_init__(self):
        indices = np.asarray(self.indices, dtype=np.intp)
        probabilities = np.asarray(self.probabilities, dtype=np.float64)
        counts = np.asarray(self.counts, dtype=np.intp)
        if indices.ndim != 2 or probabilities.shape != indices.shape:
            raise ValueError(
                "indices and probabilities must have the same 2D shape")
        if counts.shape != (indices.shape[0],):
            raise ValueError("counts must have shape (n_rows,)")
        if np.any(counts < 1) or np.any(counts > indices.shape[1]):
            raise ValueError("counts are outside the sparse row width")
        if np.any(~np.isfinite(probabilities)) or np.any(
                probabilities < 0.0):
            raise ValueError(
                "probabilities must be finite and non-negative")
        n_rows = indices.shape[0]
        for row in range(n_rows):
            count = int(counts[row])
            active_indices = indices[row, :count]
            active_probabilities = probabilities[row, :count]
            if np.any(active_indices < 0) or np.any(
                    active_indices >= n_rows):
                raise ValueError("active sparse indices are out of range")
            if np.any(np.diff(active_indices) <= 0):
                raise ValueError(
                    "active sparse indices must be strictly increasing")
            if not np.isclose(
                    np.sum(active_probabilities), 1.0,
                    rtol=0.0,
                    atol=1e-14):
                raise ValueError("sparse transition rows must sum to one")
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "probabilities", probabilities)
        object.__setattr__(self, "counts", counts)

    @property
    def shape(self):
        n_rows = int(self.indices.shape[0])
        return n_rows, n_rows

    @property
    def max_width(self):
        return int(self.indices.shape[1])

    @property
    def nnz(self):
        return int(np.sum(self.counts))

    @property
    def retained_bytes(self):
        return int(
            self.indices.nbytes
            + self.probabilities.nbytes
            + self.counts.nbytes)

    def left_multiply(self, vector):
        """Return ``vector @ P`` without dense materialization."""
        vector = np.asarray(vector, dtype=np.float64)
        if vector.shape != (self.shape[0],):
            raise ValueError(
                f"vector must have shape ({self.shape[0]},)")
        return _sparse_left_multiply(
            vector, self.indices, self.probabilities, self.counts)

    def to_dense(self, *, memory_budget_bytes=None):
        """Materialize a guarded dense matrix for diagnostics."""
        n_rows = self.shape[0]
        validate_float64_allocation(
            (n_rows, n_rows),
            name="dense Jacobi transition diagnostic",
            memory_budget_bytes=memory_budget_bytes,
        )
        return _sparse_to_dense(
            self.indices, self.probabilities, self.counts)


@njit(cache=True, nogil=True)
def _insert_sparse_mass(row_indices, row_probabilities, count, index, mass):
    for slot in range(count):
        if row_indices[slot] == index:
            row_probabilities[slot] += mass
            return count
    row_indices[count] = index
    row_probabilities[count] = mass
    return count + 1


@njit(cache=True, nogil=True)
def _insert_sparse_mass_with_grad(
        row_indices,
        row_probabilities,
        row_derivatives,
        count,
        index,
        mass,
        derivative):
    for slot in range(count):
        if row_indices[slot] == index:
            row_probabilities[slot] += mass
            for parameter in range(3):
                row_derivatives[parameter, slot] += derivative[parameter]
            return count
    row_indices[count] = index
    row_probabilities[count] = mass
    for parameter in range(3):
        row_derivatives[parameter, count] = derivative[parameter]
    return count + 1


@njit(cache=True, nogil=True)
def _sort_sparse_row(row_indices, row_probabilities, count):
    for position in range(1, count):
        index = row_indices[position]
        probability = row_probabilities[position]
        cursor = position - 1
        while cursor >= 0 and row_indices[cursor] > index:
            row_indices[cursor + 1] = row_indices[cursor]
            row_probabilities[cursor + 1] = row_probabilities[cursor]
            cursor -= 1
        row_indices[cursor + 1] = index
        row_probabilities[cursor + 1] = probability


@njit(cache=True, nogil=True)
def _sort_sparse_row_with_grad(
        row_indices, row_probabilities, row_derivatives, count):
    for position in range(1, count):
        index = row_indices[position]
        probability = row_probabilities[position]
        derivative = row_derivatives[:, position].copy()
        cursor = position - 1
        while cursor >= 0 and row_indices[cursor] > index:
            row_indices[cursor + 1] = row_indices[cursor]
            row_probabilities[cursor + 1] = row_probabilities[cursor]
            row_derivatives[:, cursor + 1] = row_derivatives[:, cursor]
            cursor -= 1
        row_indices[cursor + 1] = index
        row_probabilities[cursor + 1] = probability
        row_derivatives[:, cursor + 1] = derivative


@njit(cache=True, nogil=True)
def _build_sparse_local_kernel(
        tau, y_grid, drift, offsets, gh_weights, xi):
    n_rows = tau.size
    max_width = 2 * gh_weights.size
    indices = np.full((n_rows, max_width), -1, dtype=np.int64)
    probabilities = np.zeros((n_rows, max_width), dtype=np.float64)
    counts = np.zeros(n_rows, dtype=np.int64)
    y_max = np.pi / xi

    for row in range(n_rows):
        center = y_grid[row] + drift[row]
        # ``drift`` already contains the dt multiplier supplied by Python.
        count = 0
        for node in range(gh_weights.size):
            y_next = center + offsets[node]
            if y_next < 0.0:
                y_next = 0.0
            elif y_next > y_max:
                y_next = y_max
            tau_y = np.sin(0.5 * xi * y_next) ** 2
            weight = gh_weights[node]
            if tau_y <= tau[0]:
                count = _insert_sparse_mass(
                    indices[row], probabilities[row], count, 0, weight)
                continue
            if tau_y >= tau[-1]:
                count = _insert_sparse_mass(
                    indices[row],
                    probabilities[row],
                    count,
                    n_rows - 1,
                    weight,
                )
                continue
            right = np.searchsorted(tau, tau_y, side="right")
            left = right - 1
            width = tau[right] - tau[left]
            if width <= 0.0:
                count = _insert_sparse_mass(
                    indices[row],
                    probabilities[row],
                    count,
                    left,
                    weight,
                )
                continue
            fraction = (tau_y - tau[left]) / width
            count = _insert_sparse_mass(
                indices[row],
                probabilities[row],
                count,
                left,
                weight * (1.0 - fraction),
            )
            count = _insert_sparse_mass(
                indices[row],
                probabilities[row],
                count,
                right,
                weight * fraction,
            )

        _sort_sparse_row(indices[row], probabilities[row], count)
        row_sum = 0.0
        for slot in range(count):
            row_sum += probabilities[row, slot]
        for slot in range(count):
            probabilities[row, slot] /= row_sum
        counts[row] = count
    return indices, probabilities, counts


@njit(cache=True, nogil=True)
def _build_sparse_fixed_kernel(
        tau,
        y_grid,
        drift_dt,
        dcenter,
        offsets,
        gh_weights,
        xi):
    n_rows = tau.size
    max_width = 2 * gh_weights.size
    indices = np.full((n_rows, max_width), -1, dtype=np.int64)
    probabilities = np.zeros((n_rows, max_width), dtype=np.float64)
    derivatives = np.zeros(
        (3, n_rows, max_width), dtype=np.float64)
    counts = np.zeros(n_rows, dtype=np.int64)
    y_max = np.pi / xi
    zero_derivative = np.zeros(3, dtype=np.float64)

    for row in range(n_rows):
        center = y_grid[row] + drift_dt[row]
        count = 0
        for node in range(gh_weights.size):
            y_next = center + offsets[node]
            weight = gh_weights[node]
            if y_next <= 0.0:
                count = _insert_sparse_mass_with_grad(
                    indices[row],
                    probabilities[row],
                    derivatives[:, row],
                    count,
                    0,
                    weight,
                    zero_derivative,
                )
                continue
            if y_next >= y_max:
                count = _insert_sparse_mass_with_grad(
                    indices[row],
                    probabilities[row],
                    derivatives[:, row],
                    count,
                    n_rows - 1,
                    weight,
                    zero_derivative,
                )
                continue

            phase = 0.5 * xi * y_next
            tau_y = np.sin(phase) ** 2
            dtau_y = np.empty(3, dtype=np.float64)
            common = 0.5 * np.sin(xi * y_next)
            dtau_y[0] = common * xi * dcenter[0, row]
            dtau_y[1] = common * xi * dcenter[1, row]
            dtau_y[2] = common * (
                y_next + xi * dcenter[2, row])

            if tau_y <= tau[0]:
                count = _insert_sparse_mass_with_grad(
                    indices[row],
                    probabilities[row],
                    derivatives[:, row],
                    count,
                    0,
                    weight,
                    zero_derivative,
                )
                continue
            if tau_y >= tau[-1]:
                count = _insert_sparse_mass_with_grad(
                    indices[row],
                    probabilities[row],
                    derivatives[:, row],
                    count,
                    n_rows - 1,
                    weight,
                    zero_derivative,
                )
                continue

            right = np.searchsorted(tau, tau_y, side="right")
            left = right - 1
            width = tau[right] - tau[left]
            if width <= 0.0:
                count = _insert_sparse_mass_with_grad(
                    indices[row],
                    probabilities[row],
                    derivatives[:, row],
                    count,
                    left,
                    weight,
                    zero_derivative,
                )
                continue
            fraction = (tau_y - tau[left]) / width
            derivative = weight * dtau_y / width
            count = _insert_sparse_mass_with_grad(
                indices[row],
                probabilities[row],
                derivatives[:, row],
                count,
                left,
                weight * (1.0 - fraction),
                -derivative,
            )
            count = _insert_sparse_mass_with_grad(
                indices[row],
                probabilities[row],
                derivatives[:, row],
                count,
                right,
                weight * fraction,
                derivative,
            )

        _sort_sparse_row_with_grad(
            indices[row],
            probabilities[row],
            derivatives[:, row],
            count,
        )
        row_sum = 0.0
        drow_sum = np.zeros(3, dtype=np.float64)
        for slot in range(count):
            row_sum += probabilities[row, slot]
            for parameter in range(3):
                drow_sum[parameter] += derivatives[parameter, row, slot]
        for slot in range(count):
            raw_probability = probabilities[row, slot]
            probabilities[row, slot] = raw_probability / row_sum
            for parameter in range(3):
                derivatives[parameter, row, slot] = (
                    derivatives[parameter, row, slot] * row_sum
                    - raw_probability * drow_sum[parameter]
                ) / (row_sum * row_sum)
        counts[row] = count
    return indices, probabilities, derivatives, counts


@njit(cache=True, nogil=True)
def _sparse_left_multiply(vector, indices, probabilities, counts):
    output = np.zeros(vector.size, dtype=np.float64)
    for row in range(vector.size):
        source_mass = vector[row]
        for slot in range(counts[row]):
            output[indices[row, slot]] += (
                source_mass * probabilities[row, slot])
    return output


@njit(cache=True, nogil=True)
def _sparse_filter_loglik_kernel(
        weights, indices, probabilities, counts, fi_grid):
    predicted = weights.copy()
    posterior = np.empty_like(predicted)
    log_likelihood = 0.0
    for observation in range(fi_grid.shape[0]):
        scale = 0.0
        for node in range(predicted.size):
            value = predicted[node] * fi_grid[observation, node]
            posterior[node] = value
            scale += value
        if not np.isfinite(scale) or scale <= 0.0:
            return -np.inf
        inverse_scale = 1.0 / scale
        for node in range(predicted.size):
            posterior[node] *= inverse_scale
        log_likelihood += np.log(scale)
        if observation < fi_grid.shape[0] - 1:
            predicted.fill(0.0)
            for row in range(posterior.size):
                source_mass = posterior[row]
                for slot in range(counts[row]):
                    predicted[indices[row, slot]] += (
                        source_mass * probabilities[row, slot])
            total = 0.0
            for node in range(predicted.size):
                total += predicted[node]
            if not np.isfinite(total) or total <= 0.0:
                return -np.inf
            inverse_total = 1.0 / total
            for node in range(predicted.size):
                predicted[node] *= inverse_total
    return log_likelihood


@njit(cache=True, nogil=True)
def _sparse_neg_loglik_grad_kernel(
        weights,
        dweights,
        indices,
        probabilities,
        dprobabilities,
        counts,
        fi_grid):
    predicted = weights.copy()
    dpredicted = dweights.copy()
    posterior = np.empty_like(predicted)
    dposterior = np.empty_like(dpredicted)
    next_predicted = np.empty_like(predicted)
    next_dpredicted = np.empty_like(dpredicted)
    log_likelihood = 0.0
    gradient = np.zeros(3, dtype=np.float64)

    for observation in range(fi_grid.shape[0]):
        scale = 0.0
        dscale = np.zeros(3, dtype=np.float64)
        for node in range(predicted.size):
            emission = fi_grid[observation, node]
            posterior[node] = predicted[node] * emission
            scale += posterior[node]
            for parameter in range(3):
                dposterior[parameter, node] = (
                    dpredicted[parameter, node] * emission)
                dscale[parameter] += dposterior[parameter, node]
        if not np.isfinite(scale) or scale <= 0.0:
            return 1e10, np.zeros(3, dtype=np.float64)
        for node in range(predicted.size):
            raw_probability = posterior[node]
            posterior[node] = raw_probability / scale
            for parameter in range(3):
                dposterior[parameter, node] = (
                    dposterior[parameter, node] * scale
                    - raw_probability * dscale[parameter]
                ) / (scale * scale)
        log_likelihood += np.log(scale)
        gradient += dscale / scale

        if observation < fi_grid.shape[0] - 1:
            next_predicted.fill(0.0)
            next_dpredicted.fill(0.0)
            for row in range(posterior.size):
                for slot in range(counts[row]):
                    target = indices[row, slot]
                    probability = probabilities[row, slot]
                    next_predicted[target] += (
                        posterior[row] * probability)
                    for parameter in range(3):
                        next_dpredicted[parameter, target] += (
                            dposterior[parameter, row] * probability
                            + posterior[row]
                            * dprobabilities[parameter, row, slot]
                        )
            total = 0.0
            dtotal = np.zeros(3, dtype=np.float64)
            for node in range(next_predicted.size):
                total += next_predicted[node]
                for parameter in range(3):
                    dtotal[parameter] += next_dpredicted[parameter, node]
            if not np.isfinite(total) or total <= 0.0:
                return 1e10, np.zeros(3, dtype=np.float64)
            for node in range(next_predicted.size):
                raw_probability = next_predicted[node]
                predicted[node] = raw_probability / total
                for parameter in range(3):
                    dpredicted[parameter, node] = (
                        next_dpredicted[parameter, node] * total
                        - raw_probability * dtotal[parameter]
                    ) / (total * total)
    if (
            not np.isfinite(log_likelihood)
            or np.any(~np.isfinite(gradient))):
        return 1e10, np.zeros(3, dtype=np.float64)
    return -log_likelihood, -gradient


@njit(cache=True, nogil=True)
def _sparse_to_dense(indices, probabilities, counts):
    n_rows = indices.shape[0]
    dense = np.zeros((n_rows, n_rows), dtype=np.float64)
    for row in range(n_rows):
        for slot in range(counts[row]):
            dense[row, indices[row, slot]] = probabilities[row, slot]
    return dense


@njit(cache=True, nogil=True)
def _sample_sparse_path_kernel(
        tau, indices, probabilities, counts, uniforms, initial_index):
    path = np.empty(uniforms.size, dtype=np.float64)
    index = initial_index
    path[0] = tau[index]
    for observation in range(1, uniforms.size):
        draw = uniforms[observation]
        cumulative = 0.0
        selected = counts[index] - 1
        for slot in range(counts[index]):
            cumulative += probabilities[index, slot]
            if draw < cumulative:
                selected = slot
                break
        index = indices[index, selected]
        path[observation] = tau[index]
    return path


def _validate_sparse_workspace(
        *,
        quad_order,
        gh_order,
        correction="none",
        memory_budget_bytes=None):
    if memory_budget_bytes is None:
        memory_budget_bytes = DEFAULT_JACOBI_MEMORY_BUDGET_BYTES
    width = 2 * gh_order
    # Counts and indices use intp, conservatively charged as float64 arrays.
    width_arrays = {
        "none": 2,
        "mh": 4,
        "ipfp": 5,
    }[correction]
    elements = (
        8 * quad_order
        + width_arrays * quad_order * width)
    return validate_float64_allocation(
        (elements,),
        name="sparse Jacobi transition workspace",
        memory_budget_bytes=memory_budget_bytes,
    )


def _validate_sparse_filter_workspace(
        *,
        n_obs,
        quad_order,
        gh_order,
        gradient=False,
        correction="none",
        memory_budget_bytes=None):
    if memory_budget_bytes is None:
        memory_budget_bytes = DEFAULT_JACOBI_MEMORY_BUDGET_BYTES
    width = 2 * gh_order + 1
    # Transition indices/probabilities/counts, filter vectors, and the
    # emission plus two directional h grids used by the largest operation.
    transition_arrays = {
        "none": 2,
        "mh": 4,
        "ipfp": 5,
    }[correction]
    elements = (
        transition_arrays * quad_order * width
        + 7 * quad_order
        + 3 * n_obs * quad_order)
    if gradient:
        elements += (
            3 * quad_order * width
            + 8 * quad_order)
    return validate_float64_allocation(
        (elements,),
        name="sparse Jacobi filtering workspace",
        memory_budget_bytes=memory_budget_bytes,
    )


def jacobi_sparse_local_transition(
        kappa,
        m,
        xi,
        *,
        dt=None,
        n_obs=None,
        quad_order=128,
        basis_order=1,
        gh_order=5,
        correction="none",
        memory_budget_bytes=None,
        return_diagnostics=False):
    """Build a direct sparse local-GH Jacobi transition."""
    shapes = _jacobi_stationary_shape(kappa, m, xi)
    if shapes is None:
        raise ValueError("invalid Jacobi parameters")
    alpha, beta = shapes
    quad_order = _validate_jacobi_order(quad_order, "quad_order")
    basis_order = _validate_jacobi_order(basis_order, "basis_order")
    gh_order = _validate_jacobi_order(gh_order, "gh_order")
    if correction not in {"none", "mh", "ipfp"}:
        raise ValueError("correction must be 'none', 'mh', or 'ipfp'")
    _validate_sparse_workspace(
        quad_order=quad_order,
        gh_order=gh_order,
        correction=correction,
        memory_budget_bytes=memory_budget_bytes,
    )
    dt = _resolve_dt(dt, n_obs)
    tau, weights, _ = jacobi_rule(
        alpha,
        beta,
        quad_order,
        basis_order=1,
        memory_budget_bytes=memory_budget_bytes,
    )
    y_grid = _jacobi_lamperti(tau, xi)
    drift_dt = (
        _jacobi_lamperti_drift_from_tau(tau, kappa, m, xi) * dt)
    gh_nodes, gh_weights = _normal_hermite_rule(gh_order)
    offsets = np.sqrt(2.0 * dt) * gh_nodes
    indices, probabilities, counts = _build_sparse_local_kernel(
        tau,
        y_grid,
        drift_dt,
        offsets,
        gh_weights,
        float(xi),
    )
    transition = SparseJacobiTransition(
        indices=indices,
        probabilities=probabilities,
        counts=counts,
    )
    correction_diagnostics = {}
    if correction == "mh":
        transition, correction_diagnostics = _mh_correct_sparse_transition(
            transition, weights)
    elif correction == "ipfp":
        transition, correction_diagnostics = (
            _ipfp_correct_sparse_transition(transition, weights))
    propagated = transition.left_multiply(weights)
    diagnostics = {
        "dt": float(dt),
        "alpha": float(alpha),
        "beta": float(beta),
        "gh_order": int(gh_order),
        "transition_method": "local_sparse",
        "correction": correction,
        "nnz": transition.nnz,
        "max_width": transition.max_width,
        "retained_bytes": transition.retained_bytes,
        "dense_bytes": int(quad_order * quad_order * 8),
        "max_row_sum_error": float(np.max(np.abs(
            np.array([
                np.sum(transition.probabilities[row, :transition.counts[row]])
                for row in range(quad_order)
            ]) - 1.0))),
        "stationary_error": float(
            np.max(np.abs(propagated - weights))),
        **correction_diagnostics,
    }
    if return_diagnostics:
        return tau, weights, transition, diagnostics
    return tau, weights, transition


def jacobi_sparse_fixed_grid_transition(
        kappa,
        m,
        xi,
        *,
        dt=None,
        n_obs=None,
        quad_order=128,
        gh_order=5,
        return_grad=False,
        memory_budget_bytes=None,
        return_diagnostics=False):
    """Build a sparse fixed-grid local-GH transition and derivatives."""
    shapes = _jacobi_stationary_shape(kappa, m, xi)
    if shapes is None:
        raise ValueError("invalid Jacobi parameters")
    alpha, beta = shapes
    quad_order = _validate_jacobi_order(quad_order, "quad_order")
    gh_order = _validate_jacobi_order(gh_order, "gh_order")
    width = 2 * gh_order
    gradient_factor = 3 if return_grad else 0
    validate_float64_allocation(
        (
            (2 + gradient_factor) * quad_order * width
            + 15 * quad_order,
        ),
        name="sparse fixed-grid Jacobi transition workspace",
        memory_budget_bytes=memory_budget_bytes,
    )
    dt = _resolve_dt(dt, n_obs)
    tau, weights = _fixed_tau_rule(alpha, beta, quad_order)
    gh_nodes, gh_weights = _normal_hermite_rule(gh_order)
    y_grid = _jacobi_lamperti(tau, xi)
    drift = _jacobi_lamperti_drift_from_tau(tau, kappa, m, xi)
    denom = np.sqrt(np.maximum(tau * (1.0 - tau), 1e-300))
    asin_sqrt = np.arcsin(np.sqrt(tau))

    dcenter = np.empty((3, quad_order), dtype=np.float64)
    dcenter[0] = (
        (float(m) - tau) / (float(xi) * denom) * dt)
    dcenter[1] = (
        float(kappa) / (float(xi) * denom) * dt)
    dcenter[2] = (
        -2.0 * asin_sqrt / (float(xi) ** 2)
        + (
            -float(kappa) * (float(m) - tau)
            / (float(xi) ** 2 * denom)
            - (1.0 - 2.0 * tau) / (4.0 * denom)
        ) * dt
    )
    offsets = np.sqrt(2.0 * dt) * gh_nodes
    if return_grad:
        indices, probabilities, dprobabilities, counts = (
            _build_sparse_fixed_kernel(
                tau,
                y_grid,
                drift * dt,
                dcenter,
                offsets,
                gh_weights,
                float(xi),
            )
        )
    else:
        indices, probabilities, counts = _build_sparse_local_kernel(
            tau,
            y_grid,
            drift * dt,
            offsets,
            gh_weights,
            float(xi),
        )
        dprobabilities = None
    transition = SparseJacobiTransition(
        indices=indices,
        probabilities=probabilities,
        counts=counts,
    )
    diagnostics = {
        "dt": float(dt),
        "alpha": float(alpha),
        "beta": float(beta),
        "gh_order": int(gh_order),
        "transition_method": "local_fixed_sparse",
        "correction": "none",
        "nnz": transition.nnz,
        "max_width": transition.max_width,
        "retained_bytes": int(
            transition.retained_bytes
            + (dprobabilities.nbytes if dprobabilities is not None else 0)),
        "dense_bytes": int(
            quad_order * quad_order * 8
            * (4 if return_grad else 1)),
        "stationary_error": float(np.max(np.abs(
            transition.left_multiply(weights) - weights))),
    }
    if return_grad and return_diagnostics:
        return (
            tau,
            weights,
            transition,
            dprobabilities,
            diagnostics,
        )
    if return_grad:
        return tau, weights, transition, dprobabilities
    if return_diagnostics:
        return tau, weights, transition, diagnostics
    return tau, weights, transition


def _reverse_probability(transition, source, target):
    count = int(transition.counts[source])
    indices = transition.indices[source, :count]
    slot = int(np.searchsorted(indices, target))
    if slot < count and int(indices[slot]) == target:
        return float(transition.probabilities[source, slot])
    return 0.0


def _mh_correct_sparse_transition(transition, weights):
    n_rows = transition.shape[0]
    max_width = transition.max_width + 1
    indices = np.full((n_rows, max_width), -1, dtype=np.intp)
    probabilities = np.zeros((n_rows, max_width), dtype=np.float64)
    counts = np.zeros(n_rows, dtype=np.intp)
    accepted_mass = 0.0
    proposed_off_diagonal = 0.0
    reverse_missing = 0
    off_diagonal_edges = 0
    row_acceptance_ratios = []

    for row in range(n_rows):
        entries = {}
        row_proposed = 0.0
        row_accepted = 0.0
        count = int(transition.counts[row])
        for slot in range(count):
            target = int(transition.indices[row, slot])
            proposal = float(transition.probabilities[row, slot])
            if target == row:
                continue
            off_diagonal_edges += 1
            proposed_off_diagonal += weights[row] * proposal
            row_proposed += proposal
            reverse = _reverse_probability(transition, target, row)
            if reverse <= 0.0:
                reverse_missing += 1
                accepted = 0.0
            else:
                ratio = (
                    weights[target] * reverse
                    / (weights[row] * proposal))
                accepted = proposal * min(1.0, ratio)
            if accepted > 0.0:
                entries[target] = accepted
                accepted_mass += weights[row] * accepted
                row_accepted += accepted
        if row_proposed > 0.0:
            row_acceptance_ratios.append(row_accepted / row_proposed)
        off_sum = float(sum(entries.values()))
        entries[row] = 1.0 - off_sum
        ordered = sorted(entries.items())
        counts[row] = len(ordered)
        for slot, (target, probability) in enumerate(ordered):
            indices[row, slot] = target
            probabilities[row, slot] = probability

    corrected = SparseJacobiTransition(indices, probabilities, counts)
    balance_error = 0.0
    for row in range(n_rows):
        for slot in range(corrected.counts[row]):
            target = int(corrected.indices[row, slot])
            reverse = _reverse_probability(corrected, target, row)
            balance_error = max(
                balance_error,
                abs(
                    weights[row] * corrected.probabilities[row, slot]
                    - weights[target] * reverse),
            )
    return corrected, {
        "mean_accepted_off_diagonal_mass": float(accepted_mass),
        "mean_proposed_off_diagonal_mass": float(
            proposed_off_diagonal),
        "acceptance_mass_ratio": float(
            accepted_mass / proposed_off_diagonal
            if proposed_off_diagonal > 0.0 else 1.0),
        "min_row_acceptance_ratio": float(
            min(row_acceptance_ratios)
            if row_acceptance_ratios else 1.0),
        "mean_stay_probability": float(np.sum(
            weights * np.array([
                _reverse_probability(corrected, row, row)
                for row in range(n_rows)
            ]))),
        "max_stay_probability": float(max(
            _reverse_probability(corrected, row, row)
            for row in range(n_rows))),
        "reverse_missing_edge_fraction": float(
            reverse_missing / off_diagonal_edges
            if off_diagonal_edges else 0.0),
        "detailed_balance_error": float(balance_error),
    }


def _ipfp_correct_sparse_transition(
        transition, weights, *, tolerance=1e-15, max_iterations=10_000):
    """Balance stationary joint flux on the existing sparse support."""
    weights = np.asarray(weights, dtype=np.float64)
    if (
            weights.shape != (transition.shape[0],)
            or np.any(~np.isfinite(weights))
            or np.any(weights <= 0.0)):
        raise ValueError("IPFP weights must be finite and strictly positive")
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("IPFP tolerance must be finite and positive")
    max_iterations = validate_positive_int(
        max_iterations, "max_iterations")

    flux = transition.probabilities.copy()
    proposal_flux = transition.probabilities.copy()
    for row in range(transition.shape[0]):
        count = int(transition.counts[row])
        flux[row, :count] *= weights[row]
        proposal_flux[row, :count] *= weights[row]

    converged = False
    residual = np.inf
    iterations = 0
    for iteration in range(1, max_iterations + 1):
        column_sums = np.zeros(transition.shape[0], dtype=np.float64)
        for row in range(transition.shape[0]):
            count = int(transition.counts[row])
            np.add.at(
                column_sums,
                transition.indices[row, :count],
                flux[row, :count],
            )
        if np.any(~np.isfinite(column_sums)) or np.any(column_sums <= 0.0):
            raise FloatingPointError(
                "sparse IPFP support cannot reach every stationary node")
        column_scale = weights / column_sums
        for row in range(transition.shape[0]):
            count = int(transition.counts[row])
            targets = transition.indices[row, :count]
            flux[row, :count] *= column_scale[targets]

        row_sums = np.zeros(transition.shape[0], dtype=np.float64)
        for row in range(transition.shape[0]):
            count = int(transition.counts[row])
            row_sums[row] = np.sum(flux[row, :count])
        if np.any(~np.isfinite(row_sums)) or np.any(row_sums <= 0.0):
            raise FloatingPointError("sparse IPFP produced an invalid row")
        for row in range(transition.shape[0]):
            count = int(transition.counts[row])
            flux[row, :count] *= weights[row] / row_sums[row]

        column_sums.fill(0.0)
        for row in range(transition.shape[0]):
            count = int(transition.counts[row])
            np.add.at(
                column_sums,
                transition.indices[row, :count],
                flux[row, :count],
            )
        residual = float(np.max(np.abs(column_sums - weights)))
        iterations = iteration
        if residual <= tolerance:
            converged = True
            break
    if not converged:
        raise FloatingPointError(
            "sparse IPFP did not converge within max_iterations")

    probabilities = np.zeros_like(transition.probabilities)
    for row in range(transition.shape[0]):
        count = int(transition.counts[row])
        probabilities[row, :count] = (
            flux[row, :count] / weights[row])
    corrected = SparseJacobiTransition(
        transition.indices.copy(),
        probabilities,
        transition.counts.copy(),
    )
    positive = proposal_flux > 0.0
    kl_divergence = float(np.sum(
        flux[positive] * np.log(
            flux[positive] / proposal_flux[positive])))
    max_probability_change = 0.0
    for row in range(transition.shape[0]):
        count = int(transition.counts[row])
        max_probability_change = max(
            max_probability_change,
            float(np.max(np.abs(
                corrected.probabilities[row, :count]
                - transition.probabilities[row, :count]))),
        )
    return corrected, {
        "ipfp_iterations": iterations,
        "ipfp_stationary_residual": residual,
        "ipfp_kl_divergence": kl_divergence,
        "ipfp_max_probability_change": max_probability_change,
        "mean_stay_probability": float(np.sum(
            weights * np.array([
                _reverse_probability(corrected, row, row)
                for row in range(corrected.shape[0])
            ]))),
        "max_stay_probability": float(max(
            _reverse_probability(corrected, row, row)
            for row in range(corrected.shape[0]))),
    }


def sparse_jacobi_full_horizon_diagnostics(
        tau, weights, transition, *, steps, kappa, m):
    """Return deterministic stationarity and first-moment diagnostics."""
    steps = _validate_nonnegative_int(steps, "steps")
    propagated = np.asarray(weights, dtype=np.float64).copy()
    one_step = transition.left_multiply(propagated)
    for _ in range(steps):
        propagated = transition.left_multiply(propagated)
    target_mean = float(np.sum(weights * tau))
    target_variance = float(
        np.sum(weights * (tau - target_mean) ** 2))
    propagated_mean = float(np.sum(propagated * tau))
    propagated_variance = float(
        np.sum(propagated * (tau - propagated_mean) ** 2))
    conditional = np.empty_like(tau)
    for row in range(transition.shape[0]):
        count = int(transition.counts[row])
        conditional[row] = np.sum(
            transition.probabilities[row, :count]
            * tau[transition.indices[row, :count]])
    dt = 1.0 / steps if steps > 0 else 0.0
    expected_conditional = (
        m + (tau - m) * np.exp(-kappa * dt)
        if steps > 0 else tau)
    conditional_error = conditional - expected_conditional
    lag_one_covariance = float(np.sum(
        weights * (tau - target_mean) * (conditional - target_mean)))
    lag_one_correlation = (
        lag_one_covariance / target_variance
        if target_variance > 0.0 else 0.0)
    target_lag_one_correlation = (
        float(np.exp(-kappa * dt)) if steps > 0 else 1.0)
    return {
        "steps": steps,
        "one_step_stationary_tv": float(
            0.5 * np.sum(np.abs(one_step - weights))),
        "full_horizon_stationary_tv": float(
            0.5 * np.sum(np.abs(propagated - weights))),
        "target_mean": target_mean,
        "propagated_mean": propagated_mean,
        "target_variance": target_variance,
        "propagated_variance": propagated_variance,
        "relative_variance_error": float(
            abs(propagated_variance - target_variance) / target_variance
            if target_variance > 0.0 else 0.0),
        "conditional_mean_rmse": float(
            np.sqrt(np.sum(weights * conditional_error ** 2))),
        "conditional_mean_max_error": float(
            np.max(np.abs(conditional_error))),
        "lag_one_correlation": float(lag_one_correlation),
        "target_lag_one_correlation": target_lag_one_correlation,
        "lag_one_correlation_error": float(
            lag_one_correlation - target_lag_one_correlation),
    }


def select_sparse_jacobi_order(
        kappa,
        m,
        xi,
        *,
        n_obs,
        quad_orders=(48, 80, 128, 192, 384, 768),
        max_quad_order=None,
        basis_order=32,
        gh_order=5,
        max_full_horizon_tv=0.02,
        max_relative_variance_error=0.10,
        max_conditional_mean_rmse=1e-3,
        max_lag_one_correlation_error=1e-2,
        memory_budget_bytes=None,
        require_pass=False):
    """Select the first sparse local grid satisfying full-horizon gates.

    This is an explicit prototype: it never changes an already fitted
    model's order and performs no random sampling.
    """
    n_obs = validate_positive_int(n_obs, "n_obs")
    if n_obs < 2:
        raise ValueError("n_obs must be at least 2 for adaptive selection")
    steps = max(n_obs - 1, 0)
    basis_order = _validate_jacobi_order(basis_order, "basis_order")
    gh_order = _validate_jacobi_order(gh_order, "gh_order")
    if isinstance(quad_orders, (str, bytes)):
        raise TypeError("quad_orders must be an iterable of integers")
    orders = tuple(
        _validate_jacobi_order(order, "quad_orders")
        for order in quad_orders)
    if not orders:
        raise ValueError("quad_orders must not be empty")
    if any(right <= left for left, right in zip(orders, orders[1:])):
        raise ValueError("quad_orders must be strictly increasing")
    if max_quad_order is not None:
        max_quad_order = _validate_jacobi_order(
            max_quad_order, "max_quad_order")
        orders = tuple(order for order in orders if order <= max_quad_order)
        if not orders:
            raise ValueError("max_quad_order excludes every candidate")

    thresholds = {
        "full_horizon_stationary_tv": float(max_full_horizon_tv),
        "relative_variance_error": float(max_relative_variance_error),
        "conditional_mean_rmse": float(max_conditional_mean_rmse),
        "absolute_lag_one_correlation_error": float(
            max_lag_one_correlation_error),
    }
    if any(
            not np.isfinite(value) or value < 0.0
            for value in thresholds.values()):
        raise ValueError(
            "adaptive Jacobi thresholds must be finite and non-negative")

    candidates = []
    selected = None
    last_successful = None
    for order in orders:
        try:
            tau, weights, transition, construction = (
                jacobi_sparse_local_transition(
                    kappa,
                    m,
                    xi,
                    n_obs=n_obs,
                    quad_order=order,
                    basis_order=min(basis_order, order),
                    gh_order=gh_order,
                    memory_budget_bytes=memory_budget_bytes,
                    return_diagnostics=True,
                )
            )
        except MemoryError as exc:
            candidates.append({
                "quad_order": order,
                "passed": False,
                "memory_limited": True,
                "error": str(exc),
            })
            break
        horizon = sparse_jacobi_full_horizon_diagnostics(
            tau,
            weights,
            transition,
            steps=steps,
            kappa=kappa,
            m=m,
        )
        passed = (
            horizon["full_horizon_stationary_tv"]
            <= thresholds["full_horizon_stationary_tv"]
            and horizon["relative_variance_error"]
            <= thresholds["relative_variance_error"]
            and horizon["conditional_mean_rmse"]
            <= thresholds["conditional_mean_rmse"]
            and abs(horizon["lag_one_correlation_error"])
            <= thresholds["absolute_lag_one_correlation_error"]
        )
        candidates.append({
            "quad_order": order,
            "passed": bool(passed),
            "memory_limited": False,
            "retained_bytes": construction["retained_bytes"],
            **horizon,
        })
        last_successful = (tau, weights, transition)
        if passed:
            selected = last_successful
            break

    if selected is None:
        if require_pass:
            raise RuntimeError(
                "no sparse Jacobi quad_order satisfied the full-horizon gates")
        if last_successful is None:
            raise MemoryError(
                "memory budget prevented every sparse Jacobi candidate")
        selected = last_successful
    tau, weights, transition = selected
    selected_order = transition.shape[0]
    selected_record = next(
        record for record in candidates
        if record.get("quad_order") == selected_order)
    report = {
        "selected_quad_order": selected_order,
        "passed": bool(selected_record["passed"]),
        "exhausted": not bool(selected_record["passed"]),
        "thresholds": thresholds,
        "candidates": candidates,
    }
    return tau, weights, transition, report


def compare_sparse_jacobi_corrections(
        kappa,
        m,
        xi,
        *,
        n_obs,
        quad_orders=(48, 80, 128),
        corrections=("none", "mh", "ipfp"),
        basis_order=32,
        gh_order=5,
        memory_budget_bytes=None):
    """Compare correction candidates and higher orders deterministically."""
    n_obs = validate_positive_int(n_obs, "n_obs")
    if n_obs < 2:
        raise ValueError("n_obs must be at least 2")
    orders = tuple(
        _validate_jacobi_order(order, "quad_orders")
        for order in quad_orders)
    if not orders:
        raise ValueError("quad_orders must not be empty")
    records = []
    for order in orders:
        for correction in corrections:
            try:
                tau, weights, transition, construction = (
                    jacobi_sparse_local_transition(
                        kappa,
                        m,
                        xi,
                        n_obs=n_obs,
                        quad_order=order,
                        basis_order=min(basis_order, order),
                        gh_order=gh_order,
                        correction=correction,
                        memory_budget_bytes=memory_budget_bytes,
                        return_diagnostics=True,
                    )
                )
                horizon = sparse_jacobi_full_horizon_diagnostics(
                    tau,
                    weights,
                    transition,
                    steps=n_obs - 1,
                    kappa=kappa,
                    m=m,
                )
                records.append({
                    "quad_order": order,
                    "correction": correction,
                    "status": "ok",
                    **construction,
                    **horizon,
                })
            except (FloatingPointError, MemoryError) as exc:
                records.append({
                    "quad_order": order,
                    "correction": correction,
                    "status": "unsupported",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })
    return records


def _iter_sparse_filter(weights, transition, fi_grid):
    predicted = np.asarray(weights, dtype=np.float64).copy()
    for observation in range(fi_grid.shape[0]):
        weighted = predicted * fi_grid[observation]
        scale = float(np.sum(weighted))
        if not np.isfinite(scale) or scale <= 0.0:
            raise FloatingPointError("sparse Jacobi filter update failed")
        posterior = weighted / scale
        posterior_total = float(np.sum(posterior))
        if not np.isfinite(posterior_total) or posterior_total <= 0.0:
            raise FloatingPointError("sparse Jacobi posterior is invalid")
        posterior /= posterior_total
        yield observation, predicted, posterior, scale
        if observation < fi_grid.shape[0] - 1:
            predicted = transition.left_multiply(posterior)
            total = float(np.sum(predicted))
            if not np.isfinite(total) or total <= 0.0:
                raise FloatingPointError("sparse Jacobi prediction failed")
            predicted /= total


def _sparse_filter_setup(
        kappa,
        m,
        xi,
        u,
        copula,
        *,
        basis_order=32,
        quad_order=None,
        theta_cap=None,
        transition_method="local",
        gh_order=5,
        correction="none",
        memory_budget_bytes=None):
    u = np.asarray(u, dtype=np.float64)
    if u.ndim != 2 or u.shape[1] != 2 or len(u) < 1:
        return None
    basis_order = _validate_jacobi_order(basis_order, "basis_order")
    if quad_order is None:
        from pyscarcopula.numerical.jacobi_tm import default_quad_order

        quad_order = default_quad_order(basis_order)
    quad_order = _validate_jacobi_order(quad_order, "quad_order")
    gh_order = _validate_jacobi_order(gh_order, "gh_order")
    transition_method = normalize_jacobi_matrix_transition_method(
        transition_method)
    if transition_method not in {"local", "local_fixed"}:
        raise ValueError(
            "sparse filtering requires transition_method='local' "
            "or 'local_fixed'")
    if transition_method == "local_fixed" and correction != "none":
        raise ValueError(
            "stationarity correction is not supported for sparse "
            "local_fixed")
    _validate_sparse_filter_workspace(
        n_obs=len(u),
        quad_order=quad_order,
        gh_order=gh_order,
        correction=correction,
        memory_budget_bytes=memory_budget_bytes,
    )
    if transition_method == "local_fixed":
        tau, weights, transition = (
            jacobi_sparse_fixed_grid_transition(
                kappa,
                m,
                xi,
                n_obs=len(u),
                quad_order=quad_order,
                gh_order=gh_order,
                memory_budget_bytes=memory_budget_bytes,
            )
        )
    else:
        tau, weights, transition = jacobi_sparse_local_transition(
            kappa,
            m,
            xi,
            n_obs=len(u),
            basis_order=basis_order,
            quad_order=quad_order,
            gh_order=gh_order,
            correction=correction,
            memory_budget_bytes=memory_budget_bytes,
        )
    fi_grid, theta = _emission_grid(
        u, copula, tau, theta_cap=theta_cap)
    return u, tau, weights, transition, fi_grid, theta


def jacobi_sparse_matrix_loglik(
        kappa,
        m,
        xi,
        u,
        copula,
        *,
        basis_order=32,
        quad_order=None,
        theta_cap=None,
        transition_method="local",
        gh_order=5,
        correction="none",
        memory_budget_bytes=None):
    """Evaluate a local-GH Jacobi likelihood with sparse filtering."""
    try:
        setup = _sparse_filter_setup(
            kappa,
            m,
            xi,
            u,
            copula,
            basis_order=basis_order,
            quad_order=quad_order,
            theta_cap=theta_cap,
            transition_method=transition_method,
            gh_order=gh_order,
            correction=correction,
            memory_budget_bytes=memory_budget_bytes,
        )
    except MemoryError:
        raise
    except Exception:
        return -np.inf
    if setup is None:
        return -np.inf
    _, _, weights, transition, fi_grid, _ = setup
    return float(_sparse_filter_loglik_kernel(
        weights,
        transition.indices,
        transition.probabilities,
        transition.counts,
        fi_grid,
    ))


def jacobi_sparse_matrix_neg_loglik_with_grad(
        kappa,
        m,
        xi,
        u,
        copula,
        *,
        basis_order=32,
        quad_order=None,
        theta_cap=None,
        transition_method="local_fixed",
        gh_order=5,
        correction="none",
        memory_budget_bytes=None):
    """Return fixed-grid sparse negative likelihood and exact gradient."""
    fail = 1e10, np.zeros(3, dtype=np.float64)
    if transition_method != "local_fixed" or correction != "none":
        raise ValueError(
            "sparse analytical gradient requires uncorrected "
            "transition_method='local_fixed'")
    u = np.asarray(u, dtype=np.float64)
    if u.ndim != 2 or u.shape[1] != 2 or len(u) < 1:
        return fail
    basis_order = _validate_jacobi_order(basis_order, "basis_order")
    if quad_order is None:
        from pyscarcopula.numerical.jacobi_tm import default_quad_order

        quad_order = default_quad_order(basis_order)
    quad_order = _validate_jacobi_order(quad_order, "quad_order")
    gh_order = _validate_jacobi_order(gh_order, "gh_order")
    _validate_sparse_filter_workspace(
        n_obs=len(u),
        quad_order=quad_order,
        gh_order=gh_order,
        gradient=True,
        correction="none",
        memory_budget_bytes=memory_budget_bytes,
    )
    try:
        tau, weights, transition, dprobabilities = (
            jacobi_sparse_fixed_grid_transition(
                kappa,
                m,
                xi,
                n_obs=len(u),
                quad_order=quad_order,
                gh_order=gh_order,
                return_grad=True,
                memory_budget_bytes=memory_budget_bytes,
            )
        )
        dweights = _fixed_tau_weight_derivatives(
            kappa, m, xi, tau, weights)
        fi_grid, _ = _emission_grid(
            u, copula, tau, theta_cap=theta_cap)
    except MemoryError:
        raise
    except Exception:
        return fail
    value, gradient = _sparse_neg_loglik_grad_kernel(
        weights,
        dweights,
        transition.indices,
        transition.probabilities,
        dprobabilities,
        transition.counts,
        fi_grid,
    )
    return float(value), np.asarray(gradient, dtype=np.float64)


def jacobi_sparse_matrix_forward_predictive_mean(
        kappa,
        m,
        xi,
        u,
        copula,
        *,
        basis_order=32,
        quad_order=None,
        theta_cap=None,
        transition_method="local",
        gh_order=5,
        correction="none",
        memory_budget_bytes=None):
    """Return sparse local-GH predictive copula-parameter means."""
    setup = _sparse_filter_setup(
        kappa,
        m,
        xi,
        u,
        copula,
        basis_order=basis_order,
        quad_order=quad_order,
        theta_cap=theta_cap,
        transition_method=transition_method,
        gh_order=gh_order,
        correction=correction,
        memory_budget_bytes=memory_budget_bytes,
    )
    if setup is None:
        raise ValueError("invalid Jacobi parameters or observations")
    _, _, weights, transition, fi_grid, theta = setup
    output = np.empty(len(fi_grid), dtype=np.float64)
    for observation, predicted, _, _ in _iter_sparse_filter(
            weights, transition, fi_grid):
        output[observation] = np.sum(predicted * theta)
    return output


def jacobi_sparse_matrix_forward_mixture_h(
        kappa,
        m,
        xi,
        u,
        copula,
        *,
        basis_order=32,
        quad_order=None,
        theta_cap=None,
        transition_method="local",
        gh_order=5,
        correction="none",
        memory_budget_bytes=None):
    """Return sparse local-GH predictive h-function mixtures."""
    setup = _sparse_filter_setup(
        kappa,
        m,
        xi,
        u,
        copula,
        basis_order=basis_order,
        quad_order=quad_order,
        theta_cap=theta_cap,
        transition_method=transition_method,
        gh_order=gh_order,
        correction=correction,
        memory_budget_bytes=memory_budget_bytes,
    )
    if setup is None:
        raise ValueError("invalid Jacobi parameters or observations")
    u, _, weights, transition, fi_grid, theta = setup
    h_grid = _h_grid_on_theta(copula, u, theta)
    output = np.empty(len(fi_grid), dtype=np.float64)
    for observation, predicted, _, _ in _iter_sparse_filter(
            weights, transition, fi_grid):
        output[observation] = np.sum(
            predicted * h_grid[observation])
    return clip_h_function_values(output)


def jacobi_sparse_matrix_forward_mixture_h_pair(
        kappa,
        m,
        xi,
        u,
        copula,
        *,
        basis_order=32,
        quad_order=None,
        theta_cap=None,
        transition_method="local",
        gh_order=5,
        correction="none",
        memory_budget_bytes=None):
    """Return both sparse local-GH predictive h-function directions."""
    setup = _sparse_filter_setup(
        kappa,
        m,
        xi,
        u,
        copula,
        basis_order=basis_order,
        quad_order=quad_order,
        theta_cap=theta_cap,
        transition_method=transition_method,
        gh_order=gh_order,
        correction=correction,
        memory_budget_bytes=memory_budget_bytes,
    )
    if setup is None:
        raise ValueError("invalid Jacobi parameters or observations")
    u, _, weights, transition, fi_grid, theta = setup
    first_grid, second_grid = _h_pair_grids_on_theta(copula, u, theta)
    first = np.empty(len(fi_grid), dtype=np.float64)
    second = np.empty(len(fi_grid), dtype=np.float64)
    for observation, predicted, _, _ in _iter_sparse_filter(
            weights, transition, fi_grid):
        first[observation] = np.sum(
            predicted * first_grid[observation])
        second[observation] = np.sum(
            predicted * second_grid[observation])
    return (
        clip_h_function_values(first),
        clip_h_function_values(second),
    )


def jacobi_sparse_matrix_state_distribution(
        kappa,
        m,
        xi,
        u,
        copula,
        *,
        basis_order=32,
        quad_order=None,
        theta_cap=None,
        transition_method="local",
        gh_order=5,
        correction="none",
        memory_budget_bytes=None,
        horizon="current"):
    """Return a sparse local-GH current or next state distribution."""
    horizon = str(horizon).lower()
    if horizon not in {"current", "next"}:
        raise ValueError("horizon must be 'current' or 'next'")
    setup = _sparse_filter_setup(
        kappa,
        m,
        xi,
        u,
        copula,
        basis_order=basis_order,
        quad_order=quad_order,
        theta_cap=theta_cap,
        transition_method=transition_method,
        gh_order=gh_order,
        correction=correction,
        memory_budget_bytes=memory_budget_bytes,
    )
    if setup is None:
        raise ValueError("invalid Jacobi parameters or observations")
    _, tau, weights, transition, fi_grid, _ = setup
    posterior = None
    for _, _, posterior, _ in _iter_sparse_filter(
            weights, transition, fi_grid):
        pass
    probability = posterior
    if horizon == "next":
        probability = transition.left_multiply(probability)
        probability /= np.sum(probability)
    return tau.copy(), probability


def sample_sparse_jacobi_trajectory(
        tau, weights, transition, n, *, rng=None,
        memory_budget_bytes=None):
    """Sample one path from a prepared sparse Jacobi transition."""
    n = _validate_nonnegative_int(n, "n")
    validate_float64_allocation(
        (n,),
        name="sparse Jacobi trajectory",
        memory_budget_bytes=memory_budget_bytes,
    )
    if n == 0:
        return np.empty(0, dtype=np.float64)
    if rng is None:
        rng = np.random.default_rng()
    uniforms = np.asarray(rng.random(n), dtype=np.float64)
    stationary_cdf = np.cumsum(weights)
    stationary_cdf[-1] = 1.0
    initial_index = int(np.searchsorted(
        stationary_cdf, uniforms[0], side="right"))
    return _sample_sparse_path_kernel(
        np.asarray(tau, dtype=np.float64),
        transition.indices,
        transition.probabilities,
        transition.counts,
        uniforms,
        initial_index,
    )


__all__ = [
    "SparseJacobiTransition",
    "compare_sparse_jacobi_corrections",
    "jacobi_sparse_fixed_grid_transition",
    "jacobi_sparse_matrix_forward_mixture_h",
    "jacobi_sparse_matrix_forward_mixture_h_pair",
    "jacobi_sparse_matrix_forward_predictive_mean",
    "jacobi_sparse_matrix_loglik",
    "jacobi_sparse_matrix_neg_loglik_with_grad",
    "jacobi_sparse_matrix_state_distribution",
    "jacobi_sparse_local_transition",
    "sample_sparse_jacobi_trajectory",
    "select_sparse_jacobi_order",
    "sparse_jacobi_full_horizon_diagnostics",
]
