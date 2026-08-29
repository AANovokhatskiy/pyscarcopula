"""Native dynamic multivariate copula operations."""

from __future__ import annotations

import numpy as np

from pyscarcopula._native.threads import validate_n_threads
from pyscarcopula._native import _descriptors, _extension
from pyscarcopula._native.errors import (
    NativeError,
    raise_for_status,
)


def _values(value) -> np.ndarray:
    return np.ascontiguousarray(
        np.atleast_1d(np.asarray(value, dtype=np.float64)).ravel())


def _rows(copula, u) -> np.ndarray:
    values = np.ascontiguousarray(np.asarray(u, dtype=np.float64))
    expected_d = int(copula.d)
    if values.ndim != 2 or values.shape[1] != expected_d:
        raise ValueError(
            f"u must have shape (T, {expected_d}), got {values.shape}")
    if len(values) == 0:
        raise ValueError("u must contain at least one observation")
    if not np.all(np.isfinite(values)):
        raise ValueError("u must contain only finite values")
    return values


def _kernel_diagnostics(result) -> dict:
    keys = (
        "student_ppf_cache_values",
        "student_ppf_exact_values",
        "student_ppf_asymptotic_values",
        "student_workspace_growth_events",
        "student_workspace_peak_bytes",
        "n_threads_requested",
        "student_parallel_blocks",
        "equicorr_parallel_blocks",
        "row_parallel_blocks",
    )
    diagnostics = {key: int(result.get(key, 0)) for key in keys}
    diagnostics["student_ppf_total_values"] = sum(
        diagnostics[key]
        for key in (
            "student_ppf_cache_values",
            "student_ppf_exact_values",
            "student_ppf_asymptotic_values",
        )
    )
    return diagnostics


def _validated_n_threads(n_threads) -> int:
    return validate_n_threads(n_threads)


def gaussian_score_correlation(u) -> np.ndarray:
    """Estimate the Gaussian-score correlation in the native static kernel."""
    observations = np.ascontiguousarray(np.asarray(u, dtype=np.float64))
    if observations.ndim != 2:
        raise ValueError("u must have shape (n,d)")
    module = _extension.load()
    result = dict(module.gaussian_score_correlation(observations))
    if result["status"] != module.SCAR_OK:
        coordinate = int(result.get("failure_coordinate", -1))
        if result["status"] in {
                module.SCAR_INVALID_SIZE, module.SCAR_INVALID_PARAMETER}:
            raise ValueError(
                "Gaussian score correlation requires non-constant, "
                f"positive-definite columns; failure_coordinate={coordinate}")
        raise NativeError(
            "C++ Gaussian score correlation failed with "
            f"status={result['status']}, "
            f"failure_index={result.get('failure_index', -1)}, "
            f"failure_coordinate={coordinate}")
    return np.asarray(result["correlation"], dtype=np.float64)


def _static_correlation_result(result, module, operation):
    result = dict(result)
    raise_for_status(
        result,
        operation,
        failure_fields={
            "failure_index": "index",
            "failure_coordinate": "coordinate",
        },
        numerical_exception=FloatingPointError,
    )
    return result


def correlation_logistic(values) -> np.ndarray:
    """Apply the native stable logistic correlation transform."""
    array = np.asarray(values, dtype=np.float64)
    module = _extension.load()
    result = _static_correlation_result(
        module.static_correlation_logistic(
            np.ascontiguousarray(array)),
        module,
        "static correlation logistic transform",
    )
    return np.asarray(result["values"], dtype=np.float64).reshape(array.shape)


def correlation_logit(values) -> np.ndarray:
    """Apply the native clipped inverse-logistic correlation transform."""
    array = np.asarray(values, dtype=np.float64)
    module = _extension.load()
    result = _static_correlation_result(
        module.static_correlation_logit(np.ascontiguousarray(array)),
        module,
        "static correlation logit transform",
    )
    return np.asarray(result["values"], dtype=np.float64).reshape(array.shape)


def preprocess_correlation(correlation, *, eigenvalue_floor=1e-8):
    """Project a matrix to SPD correlation form in the native kernel."""
    matrix = np.ascontiguousarray(correlation, dtype=np.float64)
    module = _extension.load()
    result = _static_correlation_result(
        module.static_preprocess_correlation(matrix, float(eigenvalue_floor)),
        module,
        "static correlation preprocessing",
    )
    return np.asarray(result["correlation"], dtype=np.float64), {
        "input_correlation": np.asarray(
            result["input_correlation"], dtype=np.float64),
        "min_eigenvalue_before": float(result["min_eigenvalue_before"]),
        "min_eigenvalue_after": float(result["min_eigenvalue_after"]),
        "projection_applied": bool(result["projection_applied"]),
        "nonfinite_kendall_pairs": tuple(
            tuple(int(value) for value in pair)
            for pair in result["nonfinite_kendall_pairs"]
        ),
    }


def prepare_dense_correlation(correlation):
    """Build inverse-Cholesky and log-determinant state in C++."""
    matrix = np.ascontiguousarray(correlation, dtype=np.float64)
    module = _extension.load()
    result = _static_correlation_result(
        module.static_prepare_dense_correlation(matrix),
        module,
        "static dense correlation preparation",
    )
    dimension = int(result["dimension"])
    inverse = np.asarray(
        result["inverse_cholesky"], dtype=np.float64).reshape(
            dimension, dimension)
    return np.ascontiguousarray(inverse), float(result["log_determinant"])


def prepare_student_ppf_table(observations, **options):
    """Prepare clipped observations, df nodes and quantiles in C++."""
    module = _extension.load()
    config = module.StudentPpfTableConfig()
    for name, value in options.items():
        if value is not None:
            setattr(config, name, value)
    values = np.ascontiguousarray(observations, dtype=np.float64)
    result = dict(module.student_prepare_ppf_table(values, config))
    raise_for_status(result, "Student PPF table preparation")
    nodes = np.asarray(result["nodes"], dtype=np.float64)
    table = (
        np.asarray(result["table"], dtype=np.float64).reshape((len(nodes),) + values.shape)
        if result["has_table"] else None
    )
    return np.asarray(result["observations"], dtype=np.float64).reshape(values.shape), nodes, table


def evaluate_student_ppf_table(observations, nodes, table, df, start=0, stop=None):
    """Evaluate an observation row block; native policy selects exact or cached PPF."""
    values = np.ascontiguousarray(observations, dtype=np.float64)
    if start == 0 and stop is None:
        offset, count, shape = 0, values.size, values.shape
    else:
        start, stop, _ = slice(start, stop).indices(len(values))
        row_width = values.strides[0] // values.itemsize
        offset = start * row_width
        count = values[start:stop].size
        shape = values[start:stop].shape
    result = _extension.load().student_evaluate_ppf_table(
        values, np.ascontiguousarray(nodes, dtype=np.float64),
        np.empty(0, dtype=np.float64) if table is None else table,
        float(df), offset, count)
    raise_for_status(result, "Student PPF table evaluation")
    return np.asarray(result["values"], dtype=np.float64).reshape(shape)


def interpolate_student_ppf_table(nodes, table, df):
    values = np.ascontiguousarray(table, dtype=np.float64)
    width = values.reshape(len(nodes), -1).shape[1]
    result = _extension.load().student_interpolate_ppf_table(
        np.ascontiguousarray(nodes, dtype=np.float64), values, float(df), width)
    raise_for_status(result, "Student PPF table interpolation")
    return np.asarray(result["values"], dtype=np.float64).reshape(values.shape[1:])


def estimate_kendall_correlation(observations, *, eigenvalue_floor=1e-8):
    """Estimate and project the Kendall correlation in C++."""
    values = np.ascontiguousarray(observations, dtype=np.float64)
    module = _extension.load()
    result = _static_correlation_result(
        module.static_estimate_kendall_correlation(
            values, float(eigenvalue_floor)),
        module,
        "static Kendall correlation initialization",
    )
    return np.asarray(result["correlation"], dtype=np.float64), {
        "input_correlation": np.asarray(
            result["input_correlation"], dtype=np.float64),
        "min_eigenvalue_before": float(result["min_eigenvalue_before"]),
        "min_eigenvalue_after": float(result["min_eigenvalue_after"]),
        "projection_applied": bool(result["projection_applied"]),
        "nonfinite_kendall_pairs": tuple(
            tuple(int(value) for value in pair)
            for pair in result["nonfinite_kendall_pairs"]
        ),
    }


def validate_correlation(correlation, *, tolerance=1e-8) -> None:
    """Validate a finite SPD correlation matrix in C++."""
    matrix = np.ascontiguousarray(correlation, dtype=np.float64)
    module = _extension.load()
    _static_correlation_result(
        module.static_validate_correlation(matrix, float(tolerance)),
        module,
        "static correlation validation",
    )


def make_shrinkage_correlation(raw_parameter, base) -> np.ndarray:
    """Build the dense shrinkage trial correlation natively."""
    matrix = np.ascontiguousarray(base, dtype=np.float64)
    module = _extension.load()
    result = _static_correlation_result(
        module.static_make_shrinkage_correlation(
            float(raw_parameter), matrix),
        module,
        "static shrinkage correlation transform",
    )
    dimension = int(result["dimension"])
    return np.asarray(
        result["values"], dtype=np.float64).reshape(dimension, dimension)


def pack_cholesky_correlation(
        correlation, *, eigenvalue_floor=1e-8) -> np.ndarray:
    """Pack native unit-diagonal Cholesky optimizer coordinates."""
    matrix = np.ascontiguousarray(correlation, dtype=np.float64)
    module = _extension.load()
    result = _static_correlation_result(
        module.static_pack_cholesky_correlation(
            matrix, float(eigenvalue_floor)),
        module,
        "static Cholesky correlation packing",
    )
    return np.asarray(result["values"], dtype=np.float64)


def unpack_cholesky_correlation(parameters, dimension) -> np.ndarray:
    """Build a correlation matrix from native Cholesky coordinates."""
    values = np.ascontiguousarray(parameters, dtype=np.float64)
    module = _extension.load()
    result = _static_correlation_result(
        module.static_unpack_cholesky_correlation(values, int(dimension)),
        module,
        "static Cholesky correlation unpacking",
    )
    return np.asarray(result["values"], dtype=np.float64).reshape(
        int(result["dimension"]), int(result["dimension"]))


def correlation_gradient_to_raw(
        mode, parameters, correlation, correlation_gradient,
        base=None) -> np.ndarray:
    """Pull a native correlation score back to optimizer coordinates."""
    modes = {"shrinkage": 0, "cholesky": 1}
    if mode not in modes:
        raise ValueError("mode must be 'shrinkage' or 'cholesky'")
    module = _extension.load()
    base_values = (
        np.empty(0, dtype=np.float64)
        if base is None
        else np.ascontiguousarray(base, dtype=np.float64)
    )
    result = _static_correlation_result(
        module.static_correlation_gradient_to_raw(
            modes[mode],
            np.ascontiguousarray(parameters, dtype=np.float64),
            np.ascontiguousarray(correlation, dtype=np.float64),
            np.ascontiguousarray(correlation_gradient, dtype=np.float64),
            base_values,
        ),
        module,
        "static correlation gradient pullback",
    )
    return np.asarray(result["values"], dtype=np.float64)


def shrinkage_raw_correlation_direction(parameters, base) -> np.ndarray:
    """Return the native lower-triangle shrinkage direction."""
    module = _extension.load()
    result = _static_correlation_result(
        module.static_shrinkage_raw_direction(
            np.ascontiguousarray(parameters, dtype=np.float64),
            np.ascontiguousarray(base, dtype=np.float64),
        ),
        module,
        "static shrinkage correlation direction",
    )
    return np.asarray(result["values"], dtype=np.float64)


def factor_correlation_from_loadings(loadings, uniqueness_min):
    """Validate factor loadings and derive uniqueness in C++."""
    module = _extension.load()
    result = _static_correlation_result(
        module.static_factor_correlation_from_loadings(
            np.ascontiguousarray(loadings, dtype=np.float64),
            float(uniqueness_min),
        ),
        module,
        "static factor-correlation construction",
    )
    return (
        np.asarray(result["loadings"], dtype=np.float64),
        np.asarray(result["uniqueness"], dtype=np.float64),
    )


def factor_correlation_from_unconstrained(values, uniqueness_min):
    """Map unconstrained rows to factor loadings in C++."""
    module = _extension.load()
    result = _static_correlation_result(
        module.static_factor_correlation_from_unconstrained(
            np.ascontiguousarray(values, dtype=np.float64),
            float(uniqueness_min),
        ),
        module,
        "static unconstrained factor transform",
    )
    return np.asarray(result["loadings"], dtype=np.float64)


def factor_correlation_to_dense(loadings, uniqueness) -> np.ndarray:
    """Materialize a diagnostic dense factor correlation in C++."""
    module = _extension.load()
    result = _static_correlation_result(
        module.static_factor_correlation_to_dense(
            np.ascontiguousarray(loadings, dtype=np.float64),
            np.ascontiguousarray(uniqueness, dtype=np.float64),
        ),
        module,
        "static dense factor materialization",
    )
    dimension = int(result["dimension"])
    return np.asarray(
        result["values"], dtype=np.float64).reshape(dimension, dimension)


def factor_parameterization_from_loadings(loadings, uniqueness_min) -> dict:
    """Build identifiable factor optimizer coordinates in C++."""
    module = _extension.load()
    result = _static_correlation_result(
        module.static_factor_parameterization_from_loadings(
            np.ascontiguousarray(loadings, dtype=np.float64),
            float(uniqueness_min),
        ),
        module,
        "static factor parameterization initialization",
    )
    return {
        "dimension": int(result["dimension"]),
        "rank": int(result["rank"]),
        "anchors": np.asarray(result["anchors"], dtype=np.int64),
        "free_rows": np.asarray(result["free_rows"], dtype=np.int64),
        "free_columns": np.asarray(result["free_columns"], dtype=np.int64),
        "diagonal_entries": np.asarray(
            result["diagonal_entries"], dtype=np.bool_),
        "max_norm": float(result["max_norm"]),
        "parameters": np.asarray(result["parameters"], dtype=np.float64),
    }


def factor_parameterization_loadings(
        parameters, *, free_rows, free_columns, diagonal_entries,
        dimension, rank, max_norm) -> np.ndarray:
    """Map factor optimizer coordinates to loadings in C++."""
    module = _extension.load()
    result = _static_correlation_result(
        module.static_factor_parameterization_loadings(
            np.ascontiguousarray(parameters, dtype=np.float64),
            np.ascontiguousarray(free_rows, dtype=np.float64),
            np.ascontiguousarray(free_columns, dtype=np.float64),
            np.ascontiguousarray(diagonal_entries, dtype=np.float64),
            int(dimension),
            int(rank),
            float(max_norm),
        ),
        module,
        "static factor loading transform",
    )
    return np.asarray(result["values"], dtype=np.float64).reshape(
        int(result["dimension"]), int(result["rank"]))


def factor_parameterization_pullback(
        parameters, loading_gradient, *, free_rows, free_columns,
        diagonal_entries, dimension, rank, max_norm) -> np.ndarray:
    """Pull loading gradients back to factor optimizer coordinates in C++."""
    module = _extension.load()
    result = _static_correlation_result(
        module.static_factor_parameterization_pullback(
            np.ascontiguousarray(parameters, dtype=np.float64),
            np.ascontiguousarray(loading_gradient, dtype=np.float64),
            np.ascontiguousarray(free_rows, dtype=np.float64),
            np.ascontiguousarray(free_columns, dtype=np.float64),
            np.ascontiguousarray(diagonal_entries, dtype=np.float64),
            int(dimension),
            int(rank),
            float(max_norm),
        ),
        module,
        "static factor gradient pullback",
    )
    return np.asarray(result["values"], dtype=np.float64)


def estimate_factor_loadings_from_projection(
        observations, rank, *, uniqueness_min, dimension_tile,
        random_projection):
    """Run tiled factor initialization from Python-owned random draws."""
    module = _extension.load()
    result = _static_correlation_result(
        module.static_estimate_factor_loadings(
            np.ascontiguousarray(observations, dtype=np.float64),
            int(rank),
            float(uniqueness_min),
            int(dimension_tile),
            np.ascontiguousarray(random_projection, dtype=np.float64),
        ),
        module,
        "static factor loading initialization",
    )
    return np.asarray(result["loadings"], dtype=np.float64), {
        "leading_eigenvalues": np.asarray(
            result["leading_eigenvalues"], dtype=np.float64),
        "subspace_size": int(result["subspace_size"]),
        "score_tile": int(result["score_tile"]),
    }


def prepare_equicorr_statistics(
        u, *, dimension_tile=16384, n_threads=1):
    """Compute equicorrelation sufficient statistics for one dense block."""
    if isinstance(dimension_tile, (bool, np.bool_)) or not isinstance(
            dimension_tile, (int, np.integer)):
        raise ValueError("dimension_tile must be a positive integer")
    dimension_tile = int(dimension_tile)
    if dimension_tile < 1:
        raise ValueError("dimension_tile must be a positive integer")
    observations = np.ascontiguousarray(np.asarray(u, dtype=np.float64))
    if observations.ndim != 2 or observations.shape[1] < 2:
        raise ValueError(
            "u must be a 2D array with shape (T, d), d >= 2")
    if len(observations) == 0:
        raise ValueError("u must contain at least one observation")

    module = _extension.load()
    result = dict(module.prepare_equicorr_sufficient_statistics(
        observations,
        dimension_tile,
        _validated_n_threads(n_threads),
    ))
    if result["status"] != module.SCAR_OK:
        if int(result.get("nonfinite_values", 0)):
            raise ValueError(
                "u must contain only finite values; first invalid flat "
                f"index={result['failure_index']}")
        raise NativeError(
            "C++ equicorrelation preparation failed with "
            f"status={result['status']}, "
            f"failure_index={result['failure_index']}")
    axis = {0: "sequential", 1: "rows", 2: "dimension_tiles"}.get(
        int(result["parallel_axis"]), "unknown")
    return (
        np.asarray(result["sum_z"], dtype=np.float64),
        np.asarray(result["sum_z2"], dtype=np.float64),
        {
            "n_threads_requested": int(result["n_threads_requested"]),
            "parallel_blocks": int(result["parallel_blocks"]),
            "parallel_axis": axis,
            "dimension_tiles": int(result["dimension_tiles"]),
            "temporary_values": int(result["temporary_values"]),
            "clipping_events": int(result["clipping_events"]),
            "nonfinite_values": int(result["nonfinite_values"]),
        },
    )


def _student_cache_block(copula, u, cache, t_index, *, prepare):
    from pyscarcopula.copula.multivariate.stochastic_student import (
        StochasticStudentCopula,
    )

    if not isinstance(copula, StochasticStudentCopula):
        return cache, 0
    if cache is None and prepare:
        cache = copula.prepare_emission_cache(u)
        return cache, 0
    row_offset = 0 if t_index is None else int(t_index)
    if cache is not None:
        cache.block(len(u), row_offset, expected_d=copula.d)
    return cache, row_offset


def transform(copula, x) -> np.ndarray:
    module = _extension.load()
    spec = _descriptors.make_multivariate_transform_spec(module, copula)
    return np.asarray(
        module.copula_transform(spec, _values(x)), dtype=np.float64)


def inverse_transform(copula, r) -> np.ndarray:
    module = _extension.load()
    spec = _descriptors.make_multivariate_transform_spec(module, copula)
    return np.asarray(
        module.copula_inverse_transform(spec, _values(r)),
        dtype=np.float64,
    )


def dtransform(copula, x) -> np.ndarray:
    module = _extension.load()
    spec = _descriptors.make_multivariate_transform_spec(module, copula)
    return np.asarray(
        module.copula_dtransform(spec, _values(x)), dtype=np.float64)


def _log_pdf_and_dlog_rows_result(
        copula, u, r, *, t_index=None, cache=None, n_threads=1):
    module = _extension.load()
    from pyscarcopula.copula.multivariate.equicorr_prepared import (
        EquicorrPreparedData,
    )
    if isinstance(u, EquicorrPreparedData):
        if int(getattr(copula, "d", -1)) != u.dimension:
            raise ValueError(
                "prepared dimension does not match copula dimension")
        spec = _descriptors.make_multivariate_spec(module, copula)
        result = dict(module.equicorr_log_pdf_and_grad_from_stats(
            spec,
            u.sum_z,
            u.sum_z2,
            _values(r),
            _validated_n_threads(n_threads),
        ))
        if result["status"] != module.SCAR_OK:
            raise NativeError(
                "C++ prepared Equicorr row evaluation failed with "
                f"status={result['status']}, "
                f"failure_index={result['failure_index']}")
        return result, (
            np.asarray(result["log_pdf"], dtype=np.float64),
            np.asarray(result["dlog_dr"], dtype=np.float64),
        )
    observations = _rows(copula, u)
    cache, row_offset = _student_cache_block(
        copula, observations, cache, t_index, prepare=False)
    spec = _descriptors.make_multivariate_spec(
        module, copula, cache=cache)
    result = dict(module.multivariate_log_pdf_and_grad(
        spec,
        observations,
        _values(r),
        row_offset,
        _validated_n_threads(n_threads),
    ))
    if result["status"] != module.SCAR_OK:
        raise NativeError(
            "C++ multivariate row evaluation failed with "
            f"status={result['status']}, "
            f"failure_index={result['failure_index']}")
    return result, (
        np.asarray(result["log_pdf"], dtype=np.float64),
        np.asarray(result["dlog_dr"], dtype=np.float64),
    )


def log_pdf_and_dlog_rows(
        copula, u, r, *, t_index=None, cache=None, n_threads=1):
    """Return row values without instrumentation metadata."""
    _, values = _log_pdf_and_dlog_rows_result(
        copula, u, r, t_index=t_index, cache=cache, n_threads=n_threads)
    return values


def log_pdf_and_dlog_rows_info(
        copula, u, r, *, t_index=None, cache=None, n_threads=1):
    """Return row values and read-only native kernel diagnostics."""
    result, values = _log_pdf_and_dlog_rows_result(
        copula, u, r, t_index=t_index, cache=cache, n_threads=n_threads)
    return (*values, _kernel_diagnostics(result))


def _pdf_and_grad_grid_result(
        copula, u, x_grid, *, t_index=0, cache=None, n_threads=1):
    module = _extension.load()
    from pyscarcopula.copula.multivariate.equicorr_prepared import (
        EquicorrPreparedData,
    )
    if isinstance(u, EquicorrPreparedData):
        if int(getattr(copula, "d", -1)) != u.dimension:
            raise ValueError(
                "prepared dimension does not match copula dimension")
        grid = _values(x_grid)
        if len(grid) == 0:
            raise ValueError("x_grid must contain at least one value")
        spec = _descriptors.make_multivariate_spec(module, copula)
        result = dict(module.equicorr_pdf_and_grad_grid_from_stats(
            spec,
            u.sum_z,
            u.sum_z2,
            grid,
            _validated_n_threads(n_threads),
        ))
        if result["status"] != module.SCAR_OK:
            raise NativeError(
                "C++ prepared Equicorr grid evaluation failed with "
                f"status={result['status']}, "
                f"failure_index={result['failure_index']}")
        return result, (
            np.asarray(result["pdf"], dtype=np.float64),
            np.asarray(result["d_pdf_dx"], dtype=np.float64),
        )
    observations = _rows(copula, u)
    grid = _values(x_grid)
    if len(grid) == 0:
        raise ValueError("x_grid must contain at least one value")
    cache, row_offset = _student_cache_block(
        copula, observations, cache, t_index, prepare=True)
    spec = _descriptors.make_multivariate_spec(
        module, copula, cache=cache)
    result = dict(module.multivariate_pdf_and_grad_grid(
        spec,
        observations,
        grid,
        row_offset,
        _validated_n_threads(n_threads),
    ))
    if result["status"] != module.SCAR_OK:
        raise NativeError(
            "C++ multivariate grid evaluation failed with "
            f"status={result['status']}, "
            f"failure_index={result['failure_index']}")
    return result, (
        np.asarray(result["pdf"], dtype=np.float64),
        np.asarray(result["d_pdf_dx"], dtype=np.float64),
    )


def pdf_and_grad_grid(
        copula, u, x_grid, *, t_index=0, cache=None, n_threads=1):
    """Return grid values without instrumentation metadata."""
    _, values = _pdf_and_grad_grid_result(
        copula,
        u,
        x_grid,
        t_index=t_index,
        cache=cache,
        n_threads=n_threads,
    )
    return values


def pdf_and_grad_grid_info(
        copula, u, x_grid, *, t_index=0, cache=None, n_threads=1):
    """Return grid values and read-only native kernel diagnostics."""
    result, values = _pdf_and_grad_grid_result(
        copula,
        u,
        x_grid,
        t_index=t_index,
        cache=cache,
        n_threads=n_threads,
    )
    return (*values, _kernel_diagnostics(result))


def _gaussian_conditional_latent_result(
        correlations, given_indices, given_latent, normal_draws, *,
        n_threads=1):
    """Evaluate native Gaussian conditional latent samples."""
    module = _extension.load()
    result = dict(module.multivariate_gaussian_conditional(
        np.ascontiguousarray(correlations, dtype=np.float64),
        np.ascontiguousarray(given_indices, dtype=np.int32),
        np.ascontiguousarray(given_latent, dtype=np.float64),
        np.ascontiguousarray(normal_draws, dtype=np.float64),
        _validated_n_threads(n_threads),
    ))
    if result["status"] != module.SCAR_OK:
        raise NativeError(
            "C++ Gaussian conditional sampling failed with "
            f"status={result['status']}, "
            f"failure_index={result['failure_index']}")
    return result, np.asarray(result["values"], dtype=np.float64)


def gaussian_conditional_latent(
        correlations, given_indices, given_latent, normal_draws, *,
        n_threads=1):
    """Evaluate native Gaussian conditional latent samples."""
    return _gaussian_conditional_latent_result(
        correlations,
        given_indices,
        given_latent,
        normal_draws,
        n_threads=n_threads,
    )[1]


def gaussian_conditional_latent_info(
        correlations, given_indices, given_latent, normal_draws, *,
        n_threads=1):
    """Return Gaussian latent samples and parallel diagnostics."""
    result, values = _gaussian_conditional_latent_result(
        correlations,
        given_indices,
        given_latent,
        normal_draws,
        n_threads=n_threads,
    )
    return values, {
        key: int(result.get(key, 0))
        for key in (
            "n_threads_requested",
            "parallel_blocks",
            "correlation_factorizations",
        )
    }


def _student_conditional_latent_result(
        correlations, given_indices, given_latent, df,
        normal_draws, chi_square_draws, *, n_threads=1):
    """Evaluate native Student conditional latent samples."""
    module = _extension.load()
    result = dict(module.multivariate_student_conditional(
        np.ascontiguousarray(correlations, dtype=np.float64),
        np.ascontiguousarray(given_indices, dtype=np.int32),
        np.ascontiguousarray(given_latent, dtype=np.float64),
        np.ascontiguousarray(df, dtype=np.float64),
        np.ascontiguousarray(normal_draws, dtype=np.float64),
        np.ascontiguousarray(chi_square_draws, dtype=np.float64),
        _validated_n_threads(n_threads),
    ))
    if result["status"] != module.SCAR_OK:
        raise NativeError(
            "C++ Student conditional sampling failed with "
            f"status={result['status']}, "
            f"failure_index={result['failure_index']}")
    return result, np.asarray(result["values"], dtype=np.float64)


def student_conditional_latent(
        correlations, given_indices, given_latent, df,
        normal_draws, chi_square_draws, *, n_threads=1):
    """Evaluate native Student conditional latent samples."""
    return _student_conditional_latent_result(
        correlations,
        given_indices,
        given_latent,
        df,
        normal_draws,
        chi_square_draws,
        n_threads=n_threads,
    )[1]


def student_conditional_latent_info(
        correlations, given_indices, given_latent, df,
        normal_draws, chi_square_draws, *, n_threads=1):
    """Return Student latent samples and parallel diagnostics."""
    result, values = _student_conditional_latent_result(
        correlations,
        given_indices,
        given_latent,
        df,
        normal_draws,
        chi_square_draws,
        n_threads=n_threads,
    )
    return values, {
        key: int(result.get(key, 0))
        for key in (
            "n_threads_requested",
            "parallel_blocks",
            "correlation_factorizations",
        )
    }


def _sampling_values(result, module, operation):
    result = dict(result)
    if result["status"] != module.SCAR_OK:
        raise NativeError(
            f"C++ {operation} failed with status={result['status']}, "
            f"failure_index={result['failure_index']}")
    return np.asarray(result["values"], dtype=np.float64)


def gaussian_sample_from_normals(
        correlation, normal_draws, *, n_threads=1):
    """Transform fixed standard-normal draws into dense Gaussian samples."""
    module = _extension.load()
    return _sampling_values(
        module.multivariate_gaussian_sample_from_normals(
            np.ascontiguousarray(correlation, dtype=np.float64),
            np.ascontiguousarray(normal_draws, dtype=np.float64),
            _validated_n_threads(n_threads),
        ),
        module,
        "dense Gaussian sampling",
    )


def equicorr_gaussian_sample_from_normals(
        rho, dimension, normal_draws, common_draws, *, n_threads=1):
    """Transform fixed draws with the equicorrelation Gaussian kernel."""
    module = _extension.load()
    return _sampling_values(
        module.equicorr_gaussian_sample_from_normals(
            _values(rho),
            int(dimension),
            np.ascontiguousarray(normal_draws, dtype=np.float64),
            _values(common_draws),
            _validated_n_threads(n_threads),
        ),
        module,
        "equicorrelation Gaussian sampling",
    )


def equicorr_gaussian_common_draw_count(rho, dimension, n_rows):
    """Return the native-required number of raw common-factor normals."""
    module = _extension.load()
    result = dict(module.equicorr_gaussian_common_draw_count(
        _values(rho), int(dimension), int(n_rows)))
    if result["status"] != module.SCAR_OK:
        raise NativeError(
            "C++ equicorrelation common-draw planning failed with "
            f"status={result['status']}, "
            f"failure_index={result['failure_index']}")
    return int(result["count"])


def validate_equicorrelation_path(rho, dimension, n_rows, *, name="rho"):
    """Validate a scalar-or-row equicorrelation path in native C++."""
    module = _extension.load()
    status = module.validate_equicorrelation_path(
        _values(rho), int(dimension), int(n_rows))
    if status != module.SCAR_OK:
        raise ValueError(
            f"{name} must be finite and inside the equicorrelation domain")


def student_sample_from_draws(
        correlation, df, normal_draws, chi_square_draws, *, n_threads=1):
    """Transform fixed normal/radial draws into dense Student samples."""
    module = _extension.load()
    return _sampling_values(
        module.multivariate_student_sample_from_draws(
            np.ascontiguousarray(correlation, dtype=np.float64),
            _values(df),
            np.ascontiguousarray(normal_draws, dtype=np.float64),
            _values(chi_square_draws),
            _validated_n_threads(n_threads),
        ),
        module,
        "dense Student sampling",
    )


def student_sample_from_normal_uniforms(
        correlation, df, normal_draws, chi_square_uniforms, *, n_threads=1):
    """Transform raw normal/uniform draws into dense Student samples."""
    module = _extension.load()
    return _sampling_values(
        module.multivariate_student_sample_from_normal_uniforms(
            np.ascontiguousarray(correlation, dtype=np.float64),
            _values(df),
            np.ascontiguousarray(normal_draws, dtype=np.float64),
            _values(chi_square_uniforms),
            _validated_n_threads(n_threads),
        ),
        module,
        "dense Student fixed-uniform sampling",
    )


def factor_gaussian_sample_from_normals(
        correlation, factor_draws, residual_draws, *, n_threads=1):
    """Transform fixed factor/residual normals into Gaussian samples."""
    module = _extension.load()
    native = getattr(correlation, "_native", correlation)
    return _sampling_values(
        module.factor_gaussian_sample_from_normals(
            native,
            np.ascontiguousarray(factor_draws, dtype=np.float64),
            np.ascontiguousarray(residual_draws, dtype=np.float64),
            _validated_n_threads(n_threads),
        ),
        module,
        "factor Gaussian sampling",
    )


def factor_student_sample_from_draws(
        correlation, df, factor_draws, residual_draws,
        chi_square_draws, *, n_threads=1):
    """Transform fixed factor/residual/radial draws into Student samples."""
    module = _extension.load()
    native = getattr(correlation, "_native", correlation)
    return _sampling_values(
        module.factor_student_sample_from_draws(
            native,
            _values(df),
            np.ascontiguousarray(factor_draws, dtype=np.float64),
            np.ascontiguousarray(residual_draws, dtype=np.float64),
            _values(chi_square_draws),
            _validated_n_threads(n_threads),
        ),
        module,
        "factor Student sampling",
    )


def factor_student_sample_from_normal_uniforms(
        correlation, df, factor_draws, residual_draws,
        chi_square_uniforms, *, n_threads=1):
    """Transform raw normal/uniform draws into factor Student samples."""
    module = _extension.load()
    native = getattr(correlation, "_native", correlation)
    return _sampling_values(
        module.factor_student_sample_from_normal_uniforms(
            native,
            _values(df),
            np.ascontiguousarray(factor_draws, dtype=np.float64),
            np.ascontiguousarray(residual_draws, dtype=np.float64),
            _values(chi_square_uniforms),
            _validated_n_threads(n_threads),
        ),
        module,
        "factor Student fixed-uniform sampling",
    )


def gaussian_conditional_from_uniforms(
        correlations, given_indices, given_uniforms, normal_draws, *,
        n_threads=1):
    """Evaluate the complete dense Gaussian conditional transform."""
    module = _extension.load()
    return _sampling_values(
        module.multivariate_gaussian_conditional_from_uniforms(
            np.ascontiguousarray(correlations, dtype=np.float64),
            np.ascontiguousarray(given_indices, dtype=np.int32),
            _values(given_uniforms),
            np.ascontiguousarray(normal_draws, dtype=np.float64),
            _validated_n_threads(n_threads),
        ),
        module,
        "dense Gaussian conditional sampling",
    )


def equicorr_gaussian_conditional_from_uniforms(
        rho, dimension, given_indices, given_uniforms, normal_draws, *,
        n_threads=1):
    """Evaluate the equicorrelation Gaussian conditional transform."""
    module = _extension.load()
    return _sampling_values(
        module.equicorr_gaussian_conditional_from_uniforms(
            _values(rho),
            int(dimension),
            np.ascontiguousarray(given_indices, dtype=np.int32),
            _values(given_uniforms),
            np.ascontiguousarray(normal_draws, dtype=np.float64),
            _validated_n_threads(n_threads),
        ),
        module,
        "equicorrelation Gaussian conditional sampling",
    )


def student_conditional_from_uniforms(
        correlations, given_indices, given_uniforms, df,
        normal_draws, chi_square_draws, *, n_threads=1):
    """Evaluate the complete dense Student conditional transform."""
    module = _extension.load()
    return _sampling_values(
        module.multivariate_student_conditional_from_uniforms(
            np.ascontiguousarray(correlations, dtype=np.float64),
            np.ascontiguousarray(given_indices, dtype=np.int32),
            _values(given_uniforms),
            _values(df),
            np.ascontiguousarray(normal_draws, dtype=np.float64),
            _values(chi_square_draws),
            _validated_n_threads(n_threads),
        ),
        module,
        "dense Student conditional sampling",
    )


def student_conditional_from_normal_uniforms(
        correlations, given_indices, given_uniforms, df,
        normal_draws, chi_square_uniforms, *, n_threads=1):
    """Evaluate dense Student conditioning from raw normal/uniform draws."""
    module = _extension.load()
    return _sampling_values(
        module.multivariate_student_conditional_from_normal_uniforms(
            np.ascontiguousarray(correlations, dtype=np.float64),
            np.ascontiguousarray(given_indices, dtype=np.int32),
            _values(given_uniforms),
            _values(df),
            np.ascontiguousarray(normal_draws, dtype=np.float64),
            _values(chi_square_uniforms),
            _validated_n_threads(n_threads),
        ),
        module,
        "dense Student conditional fixed-uniform sampling",
    )


def factor_gaussian_conditional_from_uniforms(
        correlation, given_indices, given_uniforms,
        factor_draws, residual_draws, *, n_threads=1):
    """Evaluate the complete factor Gaussian conditional transform."""
    module = _extension.load()
    native = getattr(correlation, "_native", correlation)
    return _sampling_values(
        module.factor_gaussian_conditional_from_uniforms(
            native,
            np.ascontiguousarray(given_indices, dtype=np.int32),
            _values(given_uniforms),
            np.ascontiguousarray(factor_draws, dtype=np.float64),
            np.ascontiguousarray(residual_draws, dtype=np.float64),
            _validated_n_threads(n_threads),
        ),
        module,
        "factor Gaussian conditional sampling",
    )


def factor_student_conditional_from_uniforms(
        correlation, given_indices, given_uniforms, df,
        factor_draws, residual_draws, chi_square_draws, *, n_threads=1):
    """Evaluate the complete factor Student conditional transform."""
    module = _extension.load()
    native = getattr(correlation, "_native", correlation)
    return _sampling_values(
        module.factor_student_conditional_from_uniforms(
            native,
            np.ascontiguousarray(given_indices, dtype=np.int32),
            _values(given_uniforms),
            _values(df),
            np.ascontiguousarray(factor_draws, dtype=np.float64),
            np.ascontiguousarray(residual_draws, dtype=np.float64),
            _values(chi_square_draws),
            _validated_n_threads(n_threads),
        ),
        module,
        "factor Student conditional sampling",
    )


def factor_student_conditional_from_normal_uniforms(
        correlation, given_indices, given_uniforms, df,
        factor_draws, residual_draws, chi_square_uniforms, *, n_threads=1):
    """Evaluate factor Student conditioning from raw normal/uniform draws."""
    module = _extension.load()
    native = getattr(correlation, "_native", correlation)
    return _sampling_values(
        module.factor_student_conditional_from_normal_uniforms(
            native,
            np.ascontiguousarray(given_indices, dtype=np.int32),
            _values(given_uniforms),
            _values(df),
            np.ascontiguousarray(factor_draws, dtype=np.float64),
            np.ascontiguousarray(residual_draws, dtype=np.float64),
            _values(chi_square_uniforms),
            _validated_n_threads(n_threads),
        ),
        module,
        "factor Student conditional fixed-uniform sampling",
    )


def _rosenblatt_values(result, module, operation):
    result = dict(result)
    if result["status"] != module.SCAR_OK:
        raise_for_status(
            result,
            operation,
            failure_fields={
                "failure_index": "row",
                "failure_coordinate": "coordinate",
            },
            numerical_exception=FloatingPointError,
        )
    return np.ascontiguousarray(
        np.asarray(result["residuals"], dtype=np.float64))


def gaussian_rosenblatt(correlation, u, *, n_threads=1):
    """Evaluate the dense Gaussian Rosenblatt transform natively."""
    module = _extension.load()
    return _rosenblatt_values(
        module.dense_gaussian_rosenblatt_transform(
            np.ascontiguousarray(correlation, dtype=np.float64),
            np.ascontiguousarray(u, dtype=np.float64),
            _validated_n_threads(n_threads),
        ),
        module,
        "dense Gaussian Rosenblatt transform",
    )


def equicorr_gaussian_rosenblatt(rho, u, *, n_threads=1):
    """Evaluate scalar-or-row equicorrelation Rosenblatt transforms."""
    module = _extension.load()
    return _rosenblatt_values(
        module.equicorr_gaussian_rosenblatt_transform(
            _values(rho),
            np.ascontiguousarray(u, dtype=np.float64),
            _validated_n_threads(n_threads),
        ),
        module,
        "equicorrelation Gaussian Rosenblatt transform",
    )


def factor_gaussian_rosenblatt(correlation, u, *, n_threads=1):
    """Evaluate the compact factor Gaussian Rosenblatt transform natively."""
    module = _extension.load()
    native = getattr(correlation, "_native", correlation)
    return _rosenblatt_values(
        module.factor_gaussian_rosenblatt_transform(
            native,
            np.ascontiguousarray(u, dtype=np.float64),
            _validated_n_threads(n_threads),
        ),
        module,
        "factor Gaussian Rosenblatt transform",
    )


def factor_student_rosenblatt(correlation, df, u, *, n_threads=1):
    """Evaluate the compact factor Student Rosenblatt transform natively."""
    module = _extension.load()
    native = getattr(correlation, "_native", correlation)
    return _rosenblatt_values(
        module.factor_student_rosenblatt_transform(
            native,
            np.ascontiguousarray(u, dtype=np.float64),
            _values(df),
            _validated_n_threads(n_threads),
        ),
        module,
        "factor Student Rosenblatt transform",
    )


def radial_uniform_summary(residuals, *, n_threads=1):
    """Reduce multivariate residuals to radial uniform summaries natively."""
    module = _extension.load()
    result = dict(module.radial_uniform_summary(
        np.ascontiguousarray(residuals, dtype=np.float64),
        _validated_n_threads(n_threads),
    ))
    if result["status"] != module.SCAR_OK:
        raise_for_status(
            result,
            "radial uniform summary",
            failure_fields={
                "failure_index": "row",
                "failure_coordinate": "coordinate",
            },
            numerical_exception=FloatingPointError,
        )
    return np.asarray(result["values"], dtype=np.float64)


def _dense_student_rosenblatt_arrays(correlation, df, u):
    """Validate and own dense Student Rosenblatt input arrays."""
    raw_correlation = np.asarray(correlation)
    raw_observations = np.asarray(u)
    raw_df = np.asarray(df)
    if np.iscomplexobj(raw_correlation):
        raise TypeError("correlation must contain real values")
    if np.iscomplexobj(raw_observations):
        raise TypeError("u must contain real values")
    if np.iscomplexobj(raw_df):
        raise TypeError("df must contain real values")

    observations = np.ascontiguousarray(
        np.asarray(raw_observations, dtype=np.float64))
    if observations.ndim != 2:
        raise ValueError("u must be a 2D array")
    rows, dimension = observations.shape
    correlation_array = np.ascontiguousarray(
        np.asarray(raw_correlation, dtype=np.float64))
    if correlation_array.shape != (dimension, dimension) or dimension < 1:
        raise ValueError(
            f"correlation must have shape ({dimension}, {dimension})")
    if not np.all(np.isfinite(correlation_array)):
        raise ValueError("correlation must contain only finite values")
    if np.any(np.isnan(observations)):
        raise ValueError("u must not contain NaN values")

    df_array = np.ascontiguousarray(
        np.atleast_1d(np.asarray(raw_df, dtype=np.float64)).ravel())
    if df_array.size not in ({0, 1} if rows == 0 else {1, rows}):
        raise ValueError("df must be scalar or have one value per row")
    if not np.all(np.isfinite(df_array)) or np.any(df_array <= 0.0):
        raise ValueError("df must contain finite positive values")
    return correlation_array, df_array, observations


def _dense_student_rosenblatt_prepared(
        correlation_array, df_array, observations, *, n_threads, module):
    """Execute the native dense Student transform on validated arrays."""
    if module is None:
        module = _extension.load()
    result = dict(module.dense_student_rosenblatt_transform(
        correlation_array,
        observations,
        df_array,
        _validated_n_threads(n_threads),
    ))
    status = int(result["status"])
    if status != int(module.SCAR_OK):
        failure_index = int(result.get("failure_index", -1))
        raise_for_status(
            result,
            "dense Student Rosenblatt transform",
            failure_fields={
                "failure_index": "row",
                "failure_coordinate": "coordinate",
            },
            numerical_exception=(
                np.linalg.LinAlgError
                if failure_index < 0
                else FloatingPointError
            ),
        )

    rows, dimension = observations.shape
    if (int(result["n_rows"]) != rows
            or int(result["dimension"]) != dimension):
        raise NativeError(
            "C++ dense Student Rosenblatt transform returned "
            "inconsistent dimensions")
    residuals = np.asarray(result["residuals"], dtype=np.float64)
    if residuals.size != rows * dimension:
        raise NativeError(
            "C++ dense Student Rosenblatt transform returned an invalid "
            "buffer")
    residuals = residuals.reshape(rows, dimension)
    if not np.all(np.isfinite(residuals)):
        raise NativeError(
            "C++ dense Student Rosenblatt transform returned invalid values")
    return np.ascontiguousarray(residuals)


def dense_student_rosenblatt(
        correlation, df, u, *, n_threads=1, module=None):
    """Evaluate the mandatory dense Student transform natively."""
    correlation_array, df_array, observations = (
        _dense_student_rosenblatt_arrays(correlation, df, u))
    return _dense_student_rosenblatt_prepared(
        correlation_array,
        df_array,
        observations,
        n_threads=_validated_n_threads(n_threads),
        module=module,
    )
