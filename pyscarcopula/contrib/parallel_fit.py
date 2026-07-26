"""Process-level parallelism for independent copula fits.

The helpers in this module deliberately reconstruct a model per task.  A
worker therefore never receives another task's mutable fit state or prepared
native evaluator.
"""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass, replace
import multiprocessing as mp
import os
from typing import Any, Mapping, Sequence

import numpy as np

from pyscarcopula._types import NumericalConfig


@dataclass(frozen=True)
class IndependentFit:
    """One independently owned fitted model and its fit result."""

    model: Any
    result: Any


@dataclass(frozen=True)
class IndependentFitBatch:
    """Results and execution diagnostics from :func:`fit_independent`."""

    fits: tuple[IndependentFit, ...]
    diagnostics: Mapping[str, Any]

    @property
    def models(self) -> tuple[Any, ...]:
        """Return the fitted model from each task in submission order."""
        return tuple(item.model for item in self.fits)

    @property
    def results(self) -> tuple[Any, ...]:
        """Return the fit result from each task in submission order."""
        return tuple(item.result for item in self.fits)


def _validate_n_jobs(n_jobs: int, n_tasks: int) -> tuple[int, int]:
    if isinstance(n_jobs, (bool, np.bool_)) or not isinstance(
            n_jobs, (int, np.integer)):
        raise ValueError("n_jobs must be -1 or a positive integer")
    requested = int(n_jobs)
    if requested == -1:
        resolved = max(os.cpu_count() or 1, 1)
    elif requested < 1:
        raise ValueError("n_jobs must be -1 or a positive integer")
    else:
        resolved = requested
    return requested, min(resolved, max(n_tasks, 1))


def _resolve_parallelism(
        n_jobs: int,
        n_tasks: int,
        n_threads: int | None,
        fit_kwargs: Sequence[Mapping[str, Any]] = (),
        mp_start_method: str | None = None,
) -> tuple[int, dict[str, Any]]:
    requested_jobs, resolved_jobs = _validate_n_jobs(n_jobs, n_tasks)
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


def _with_n_threads(
        kwargs: Mapping[str, Any], n_threads: int) -> dict[str, Any]:
    out = dict(kwargs)
    config = out.get("config")
    if config is None:
        out["config"] = NumericalConfig(n_threads=n_threads)
    elif isinstance(config, NumericalConfig):
        out["config"] = replace(config, n_threads=n_threads)
    else:
        raise TypeError("fit keyword 'config' must be NumericalConfig")
    return out


def _fit_worker(args) -> IndependentFit:
    copula_class, constructor_kwargs, data, method, fit_kwargs = args
    model = copula_class(**deepcopy(constructor_kwargs))
    result = model.fit(data, method=method, **fit_kwargs)
    return IndependentFit(model=model, result=result)


def _as_task_values(value, n_tasks: int, name: str):
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = list(value)
        if len(values) != n_tasks:
            raise ValueError(
                f"{name} must contain exactly {n_tasks} items, got "
                f"{len(values)}")
        return values
    return [value] * n_tasks


def fit_independent(
        copulas: Any | Sequence[Any],
        datasets: Iterable[Any],
        *,
        method: str = "mle",
        fit_kwargs: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
        n_jobs: int = 1,
        n_threads: int | None = None,
        mp_start_method: str | None = None,
) -> IndependentFitBatch:
    """Fit independent datasets/models, optionally in worker processes.

    ``copulas`` may be one unfitted prototype, which is broadcast to every
    dataset, or one prototype per dataset.  ``fit_kwargs`` follows the same
    rule and can therefore carry task-specific initial points or optimizer
    settings.  Only constructor-level structural model parameters are copied;
    transient caches and previous fit state are intentionally excluded.

    When ``n_jobs > 1``, omitted ``n_threads`` resolves to ``1``.  Passing an
    explicit larger value opts into nested process/thread parallelism and is
    recorded in ``batch.diagnostics``.
    """
    data_values = [np.asarray(data, dtype=np.float64) for data in datasets]
    n_tasks = len(data_values)
    if n_tasks == 0:
        raise ValueError("datasets must contain at least one task")

    copula_values = _as_task_values(copulas, n_tasks, "copulas")
    if fit_kwargs is None or isinstance(fit_kwargs, Mapping):
        kwargs_values = [dict(fit_kwargs or {}) for _ in range(n_tasks)]
    else:
        kwargs_values = _as_task_values(
            fit_kwargs, n_tasks, "fit_kwargs")
        kwargs_values = [dict(item) for item in kwargs_values]

    resolved_threads, diagnostics = _resolve_parallelism(
        n_jobs, n_tasks, n_threads, kwargs_values, mp_start_method)
    kwargs_values = [
        _with_n_threads(item, resolved_threads) for item in kwargs_values]

    # Imported lazily to keep the contrib modules import-independent.
    from pyscarcopula.contrib.risk_metrics import _get_copula_constructor

    tasks = []
    for copula, data, kwargs in zip(
            copula_values, data_values, kwargs_values):
        copula_class, constructor_kwargs = _get_copula_constructor(copula)
        tasks.append((
            copula_class, constructor_kwargs, data, method, kwargs))

    if diagnostics["n_jobs"] == 1:
        fits = tuple(_fit_worker(task) for task in tasks)
    else:
        context = mp.get_context(diagnostics["multiprocessing_start_method"])
        with context.Pool(diagnostics["n_jobs"]) as pool:
            fits = tuple(pool.map(_fit_worker, tasks))

    return IndependentFitBatch(fits=fits, diagnostics=diagnostics)
