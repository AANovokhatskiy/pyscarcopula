"""Native dynamic multivariate copula operations."""

from __future__ import annotations

import numpy as np

from pyscarcopula.numerical import _cpp_copula, _cpp_extension
from pyscarcopula.numerical._cpp_extension import CppError


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
