"""Native dynamic multivariate copula operations."""

from __future__ import annotations

import numpy as np

from pyscarcopula.numerical import _cpp_copula, _cpp_extension
from pyscarcopula.numerical._cpp_extension import (
    CppError,
    CppUnsupported,
    cpp_status_name,
)


_DENSE_STUDENT_NATIVE_MIN_DF = 0.1
_DENSE_STUDENT_NATIVE_MAX_CONDITION = 1.0e4
_DENSE_STUDENT_CORRELATION_TOLERANCE = 1.0e-12


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
    if isinstance(n_threads, (bool, np.bool_)) or not isinstance(
            n_threads, (int, np.integer)):
        raise ValueError("n_threads must be an integer in [1, 256]")
    value = int(n_threads)
    if value < 1 or value > 256:
        raise ValueError(f"n_threads must be in [1, 256], got {value}")
    return value


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

    module = _cpp_extension.load()
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
        raise CppError(
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
    module = _cpp_extension.load()
    spec = _cpp_copula.make_multivariate_transform_spec(module, copula)
    return np.asarray(
        module.copula_transform(spec, _values(x)), dtype=np.float64)


def inverse_transform(copula, r) -> np.ndarray:
    module = _cpp_extension.load()
    spec = _cpp_copula.make_multivariate_transform_spec(module, copula)
    return np.asarray(
        module.copula_inverse_transform(spec, _values(r)),
        dtype=np.float64,
    )


def dtransform(copula, x) -> np.ndarray:
    module = _cpp_extension.load()
    spec = _cpp_copula.make_multivariate_transform_spec(module, copula)
    return np.asarray(
        module.copula_dtransform(spec, _values(x)), dtype=np.float64)


def _log_pdf_and_dlog_rows_result(
        copula, u, r, *, t_index=None, cache=None, n_threads=1):
    module = _cpp_extension.load()
    from pyscarcopula.copula.multivariate.equicorr_prepared import (
        EquicorrPreparedData,
    )
    if isinstance(u, EquicorrPreparedData):
        if int(getattr(copula, "d", -1)) != u.dimension:
            raise ValueError(
                "prepared dimension does not match copula dimension")
        spec = _cpp_copula.make_multivariate_spec(module, copula)
        result = dict(module.equicorr_log_pdf_and_grad_from_stats(
            spec,
            u.sum_z,
            u.sum_z2,
            _values(r),
            _validated_n_threads(n_threads),
        ))
        if result["status"] != module.SCAR_OK:
            raise CppError(
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
    spec = _cpp_copula.make_multivariate_spec(
        module, copula, cache=cache)
    result = dict(module.multivariate_log_pdf_and_grad(
        spec,
        observations,
        _values(r),
        row_offset,
        _validated_n_threads(n_threads),
    ))
    if result["status"] != module.SCAR_OK:
        raise CppError(
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
    module = _cpp_extension.load()
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
        spec = _cpp_copula.make_multivariate_spec(module, copula)
        result = dict(module.equicorr_pdf_and_grad_grid_from_stats(
            spec,
            u.sum_z,
            u.sum_z2,
            grid,
            _validated_n_threads(n_threads),
        ))
        if result["status"] != module.SCAR_OK:
            raise CppError(
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
    spec = _cpp_copula.make_multivariate_spec(
        module, copula, cache=cache)
    result = dict(module.multivariate_pdf_and_grad_grid(
        spec,
        observations,
        grid,
        row_offset,
        _validated_n_threads(n_threads),
    ))
    if result["status"] != module.SCAR_OK:
        raise CppError(
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
    module = _cpp_extension.load()
    result = dict(module.multivariate_gaussian_conditional(
        np.ascontiguousarray(correlations, dtype=np.float64),
        np.ascontiguousarray(given_indices, dtype=np.int32),
        np.ascontiguousarray(given_latent, dtype=np.float64),
        np.ascontiguousarray(normal_draws, dtype=np.float64),
        _validated_n_threads(n_threads),
    ))
    if result["status"] != module.SCAR_OK:
        raise CppError(
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
    module = _cpp_extension.load()
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
        raise CppError(
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


def _dense_student_rosenblatt_arrays_supported(correlation, df):
    """Return whether strict native/SciPy parity is established for inputs."""
    if df.size and np.any(df < _DENSE_STUDENT_NATIVE_MIN_DF):
        return False
    symmetry_scale = np.maximum(
        1.0, np.maximum(np.abs(correlation), np.abs(correlation.T)))
    if np.any(
            np.abs(correlation - correlation.T)
            > _DENSE_STUDENT_CORRELATION_TOLERANCE * symmetry_scale):
        return False
    if not np.allclose(
            np.diag(correlation),
            1.0,
            rtol=0.0,
            atol=_DENSE_STUDENT_CORRELATION_TOLERANCE):
        return False
    try:
        eigenvalues = np.linalg.eigvalsh(correlation)
    except np.linalg.LinAlgError:
        return False
    if (not np.all(np.isfinite(eigenvalues))
            or eigenvalues[0] <= 0.0):
        return False
    condition = float(eigenvalues[-1] / eigenvalues[0])
    return condition <= _DENSE_STUDENT_NATIVE_MAX_CONDITION


def _dense_student_rosenblatt_prepared(
        correlation_array, df_array, observations, *, n_threads, module):
    """Execute the native dense Student transform on validated arrays."""
    if module is None:
        module = _cpp_extension.load()
    result = dict(module.dense_student_rosenblatt_transform(
        correlation_array,
        observations,
        df_array,
        _validated_n_threads(n_threads),
    ))
    status = int(result["status"])
    if status != int(module.SCAR_OK):
        failure_index = int(result.get("failure_index", -1))
        failure_coordinate = int(result.get("failure_coordinate", -1))
        message = (
            "C++ dense Student Rosenblatt transform failed: "
            f"status={status} ({cpp_status_name(status)})"
        )
        if failure_index >= 0:
            message += f", row={failure_index}"
        if failure_coordinate >= 0:
            message += f", coordinate={failure_coordinate}"
        if status in (int(module.SCAR_INVALID_SIZE),
                      int(module.SCAR_INVALID_PARAMETER)):
            raise ValueError(message)
        if status == int(module.SCAR_NUMERICAL_FAILURE):
            if failure_index < 0:
                raise np.linalg.LinAlgError(message)
            raise FloatingPointError(message)
        raise CppError(message)

    rows, dimension = observations.shape
    if (int(result["n_rows"]) != rows
            or int(result["dimension"]) != dimension):
        raise CppError(
            "C++ dense Student Rosenblatt transform returned "
            "inconsistent dimensions")
    residuals = np.asarray(result["residuals"], dtype=np.float64)
    if residuals.size != rows * dimension:
        raise CppError(
            "C++ dense Student Rosenblatt transform returned an invalid "
            "buffer")
    residuals = residuals.reshape(rows, dimension)
    if not np.all(np.isfinite(residuals)):
        raise CppError(
            "C++ dense Student Rosenblatt transform returned invalid values")
    return np.ascontiguousarray(residuals)


def _dense_student_rosenblatt_if_supported(
        correlation, df, u, *, n_threads=1, module=None):
    """Return native residuals, or ``None`` for a legacy-oracle case."""
    try:
        correlation_array, df_array, observations = (
            _dense_student_rosenblatt_arrays(correlation, df, u))
    except (TypeError, ValueError, OverflowError, np.linalg.LinAlgError):
        return None
    if not _dense_student_rosenblatt_arrays_supported(
            correlation_array, df_array):
        return None
    return _dense_student_rosenblatt_prepared(
        correlation_array,
        df_array,
        observations,
        n_threads=_validated_n_threads(n_threads),
        module=module,
    )


def dense_student_rosenblatt(
        correlation, df, u, *, n_threads=1, module=None):
    """Evaluate a parity-validated dense Student transform natively."""
    correlation_array, df_array, observations = (
        _dense_student_rosenblatt_arrays(correlation, df, u))
    if not _dense_student_rosenblatt_arrays_supported(
            correlation_array, df_array):
        raise CppUnsupported(
            "native dense Student Rosenblatt requires df >= "
            f"{_DENSE_STUDENT_NATIVE_MIN_DF:g}, a symmetric unit-diagonal "
            "positive-definite correlation matrix, and correlation "
            f"condition number <= {_DENSE_STUDENT_NATIVE_MAX_CONDITION:g}"
        )
    return _dense_student_rosenblatt_prepared(
        correlation_array,
        df_array,
        observations,
        n_threads=_validated_n_threads(n_threads),
        module=module,
    )
