"""Shared helpers for independent process-level work.

This module centralizes resource planning, random-stream creation, and model
reconstruction for rolling risk metrics, independent fits, and bootstrap
calibration. Worker tasks must own their model state and must never share one
mutable random generator.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import multiprocessing as mp
import os
from typing import Any, Mapping, Sequence

import numpy as np

from pyscarcopula._types import NumericalConfig


def validate_n_jobs(n_jobs: int, n_tasks: int) -> tuple[int, int]:
    """Validate ``n_jobs`` and resolve the worker count for ``n_tasks``."""
    if isinstance(n_jobs, (bool, np.bool_)) or not isinstance(
            n_jobs, (int, np.integer)):
        raise ValueError("n_jobs must be -1 or a positive integer")
    if isinstance(n_tasks, (bool, np.bool_)) or not isinstance(
            n_tasks, (int, np.integer)):
        raise TypeError("n_tasks must be a non-negative integer")

    requested = int(n_jobs)
    task_count = int(n_tasks)
    if task_count < 0:
        raise ValueError("n_tasks must be non-negative")
    if requested == -1:
        resolved = max(os.cpu_count() or 1, 1)
    elif requested < 1:
        raise ValueError("n_jobs must be -1 or a positive integer")
    else:
        resolved = requested
    return requested, min(resolved, max(task_count, 1))


def resolve_parallelism(
        n_jobs: int,
        n_tasks: int,
        n_threads: int | None,
        fit_kwargs: Sequence[Mapping[str, Any]] = (),
        mp_start_method: str | None = None,
) -> tuple[int, dict[str, Any]]:
    """Resolve process/native-thread counts and return diagnostics."""
    requested_jobs, resolved_jobs = validate_n_jobs(n_jobs, n_tasks)
    requested_threads = n_threads

    if n_threads is None:
        configured = NumericalConfig().n_threads
        if resolved_jobs == 1:
            for kwargs in fit_kwargs:
                config = kwargs.get("config")
                if isinstance(config, NumericalConfig):
                    configured = config.n_threads
                    break
        resolved_threads = int(configured)
    else:
        resolved_threads = NumericalConfig(n_threads=n_threads).n_threads

    if resolved_jobs > 1 and requested_threads is None:
        resolved_threads = 1

    context = mp.get_context(mp_start_method)
    diagnostics = {
        "n_tasks": int(n_tasks),
        "n_jobs_requested": requested_jobs,
        "n_jobs": resolved_jobs,
        "n_threads_requested": requested_threads,
        "n_threads": resolved_threads,
        "multiprocessing_start_method": context.get_start_method(),
        "nested_parallelism": bool(
            resolved_jobs > 1 and resolved_threads > 1),
        "worker_model_ownership": "per_task",
        "prepared_evaluator_sharing": False,
    }
    return resolved_threads, diagnostics


def with_n_threads(
        kwargs: Mapping[str, Any], n_threads: int) -> dict[str, Any]:
    """Return fit kwargs with a validated native-thread configuration."""
    out = dict(kwargs)
    config = out.get("config")
    if config is None:
        out["config"] = NumericalConfig(n_threads=n_threads)
    elif isinstance(config, NumericalConfig):
        out["config"] = replace(config, n_threads=n_threads)
    else:
        raise TypeError("fit keyword 'config' must be NumericalConfig")
    return out


def coerce_seed_sequence(rng=None) -> np.random.SeedSequence:
    """Create a root seed sequence without sharing a mutable generator."""
    if isinstance(rng, np.random.SeedSequence):
        return rng
    if isinstance(rng, np.random.Generator):
        return np.random.SeedSequence(rng.bit_generator.random_raw(4))
    return np.random.SeedSequence(rng)


def spawn_seed_sequences(rng, n_tasks: int) -> tuple[np.random.SeedSequence, ...]:
    """Create one independent seed sequence per task in submission order."""
    if isinstance(n_tasks, (bool, np.bool_)) or not isinstance(
            n_tasks, (int, np.integer)):
        raise TypeError("n_tasks must be a non-negative integer")
    n_tasks = int(n_tasks)
    if n_tasks < 0:
        raise ValueError("n_tasks must be non-negative")
    return tuple(coerce_seed_sequence(rng).spawn(n_tasks))


def get_copula_constructor(copula):
    """Extract a model class and constructor-only worker state."""
    from pyscarcopula.copula.multivariate import (
        EquicorrGaussianCopula,
        GaussianCopula,
        StochasticStudentCopula,
        StudentCopula,
    )
    from pyscarcopula.vine.cvine import CVineCopula
    from pyscarcopula.vine.vine import VineCopula

    if isinstance(copula, CVineCopula):
        return (
            CVineCopula,
            dict(
                candidates=copula.candidates,
                allow_rotations=copula.allow_rotations,
                criterion=copula.criterion,
            ),
        )
    if isinstance(copula, VineCopula):
        structure = (
            copula.structure
            if copula.structure_source == "fixed"
            else None
        )
        return (
            VineCopula,
            dict(
                candidates=copula.candidates,
                allow_rotations=copula.allow_rotations,
                criterion=copula.criterion,
                truncation_level=copula.truncation_level,
                truncation_fill=copula.truncation_fill,
                threshold=copula.threshold,
                min_edge_logL=copula.min_edge_logL,
                transform_type=copula.transform_type,
                structure=structure,
                vine_type=copula.vine_type,
            ),
        )
    if isinstance(copula, StochasticStudentCopula):
        constructor_R = getattr(copula, "_constructor_R", None)
        constructor_corr_base = getattr(
            copula, "_constructor_corr_base", None)
        if not hasattr(copula, "_constructor_R"):
            if copula.corr_mode == "fixed" or copula.fit_result is None:
                constructor_R = copula.R
        if not hasattr(copula, "_constructor_corr_base"):
            preprocessing = getattr(
                copula, "_corr_base_preprocessing", None)
            if (
                    copula.fit_result is None
                    or getattr(preprocessing, "source", None) == "corr_base"):
                constructor_corr_base = getattr(copula, "_corr_base", None)
        kwargs = dict(
            d=copula.d,
            R=(
                None if constructor_R is None
                else np.array(constructor_R, copy=True)),
            corr_mode=copula.corr_mode,
            corr_base=(
                None if constructor_corr_base is None
                else np.array(constructor_corr_base, copy=True)),
            corr_shrinkage_init=copula._corr_shrinkage_init,
            cholesky_d_max=copula._cholesky_d_max,
            allow_large_cholesky=copula._allow_large_cholesky,
        )
        if copula.corr_mode == "factor":
            constructor_loadings = getattr(
                copula, "_constructor_factor_loadings", None)
            if (
                    constructor_loadings is None
                    and copula.fit_result is None):
                constructor_loadings = copula.factor_loadings_
            kwargs.pop("R", None)
            kwargs.pop("corr_base", None)
            kwargs.update(
                factor_rank=copula.factor_rank,
                factor_loadings=(
                    None
                    if constructor_loadings is None
                    else np.array(constructor_loadings, copy=True)),
                factor_estimation=copula.factor_estimation,
                factor_tile_size=copula.factor_tile_size,
                factor_uniqueness_min=copula.factor_uniqueness_min,
                factor_joint_max_params=copula.factor_joint_max_params,
                factor_joint_penalty=copula.factor_joint_penalty,
                factor_joint_condition_max=
                    copula.factor_joint_condition_max,
                factor_seed=copula._factor_seed,
                factor_oversampling=copula._factor_oversampling,
            )
        return StochasticStudentCopula, kwargs
    if isinstance(copula, EquicorrGaussianCopula):
        return EquicorrGaussianCopula, dict(d=copula.d)
    if isinstance(copula, GaussianCopula):
        constructor_R = getattr(copula, "_constructor_R", None)
        constructor_corr_base = getattr(
            copula, "_constructor_corr_base", None)
        if getattr(copula, "corr_mode", "dense") == "factor":
            constructor_loadings = getattr(
                copula, "_constructor_factor_loadings", None)
            if (
                    constructor_loadings is None
                    and copula.fit_result is None):
                constructor_loadings = copula.factor_loadings_
            return (
                GaussianCopula,
                {
                    "d": copula.d,
                    "corr_mode": "factor",
                    "factor_rank": copula.factor_rank,
                    "factor_estimation": copula.factor_estimation,
                    "factor_loadings": (
                        None
                        if constructor_loadings is None
                        else np.array(
                            constructor_loadings, copy=True)),
                    "factor_tile_size": copula.factor_tile_size,
                    "factor_uniqueness_min":
                        copula._factor_uniqueness_min,
                    "factor_seed": copula._factor_seed,
                    "factor_oversampling":
                        copula._factor_oversampling,
                },
            )
        return (
            GaussianCopula,
            {
                "d": copula.d,
                "R": (
                    None if constructor_R is None
                    else np.array(constructor_R, copy=True)),
                "corr_mode": copula.corr_mode,
                "corr_base": (
                    None if constructor_corr_base is None
                    else np.array(constructor_corr_base, copy=True)),
                "corr_shrinkage_init": copula._corr_shrinkage_init,
                "cholesky_d_max": copula._cholesky_d_max,
                "allow_large_cholesky": copula._allow_large_cholesky,
            },
        )
    if isinstance(copula, StudentCopula):
        constructor_R = getattr(copula, "_constructor_R", None)
        constructor_corr_base = getattr(
            copula, "_constructor_corr_base", None)
        if not hasattr(copula, "_constructor_R"):
            constructor_R = getattr(copula, "_supplied_correlation", None)
        if not hasattr(copula, "_constructor_corr_base"):
            constructor_corr_base = getattr(copula, "_corr_base", None)
        if copula.corr_mode == "factor":
            constructor_loadings = getattr(
                copula, "_constructor_factor_loadings", None)
            if (
                    constructor_loadings is None
                    and copula.fit_result is None):
                constructor_loadings = copula.factor_loadings_
            return StudentCopula, {
                "d": copula.d,
                "corr_mode": "factor",
                "factor_rank": copula.factor_rank,
                "factor_loadings": (
                    None if constructor_loadings is None
                    else np.array(constructor_loadings, copy=True)),
                "factor_estimation": copula.factor_estimation,
                "factor_tile_size": copula._factor_tile_size,
                "factor_uniqueness_min": copula._factor_uniqueness_min,
                "factor_joint_max_params": copula._factor_joint_max_params,
                "factor_joint_penalty": copula._factor_joint_penalty,
                "factor_joint_condition_max": (
                    copula._factor_joint_condition_max),
                "factor_seed": copula._factor_seed,
                "factor_oversampling": copula._factor_oversampling,
            }
        return StudentCopula, {
            "d": copula.d,
            "R": (
                None if constructor_R is None
                else np.array(constructor_R, copy=True)),
            "corr_mode": copula.corr_mode,
            "corr_base": (
                None if constructor_corr_base is None
                else np.array(constructor_corr_base, copy=True)),
            "corr_shrinkage_init": copula._corr_shrinkage_init,
            "cholesky_d_max": copula._cholesky_d_max,
            "allow_large_cholesky": copula._allow_large_cholesky,
        }

    kwargs = dict(rotate=copula._rotate)
    transform_type = getattr(copula, "_transform_type", None)
    if transform_type is not None:
        kwargs["transform_type"] = transform_type
    return type(copula), kwargs


def create_worker_model(copula_or_class, constructor_kwargs=None):
    """Create a fresh model containing no transient fitted worker state.

    Pass either a model prototype or the class/constructor mapping returned by
    :func:`get_copula_constructor`.
    """
    if constructor_kwargs is None:
        copula_class, constructor_kwargs = get_copula_constructor(
            copula_or_class)
    else:
        copula_class = copula_or_class
    return copula_class(**deepcopy(constructor_kwargs))
