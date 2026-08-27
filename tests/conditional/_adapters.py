"""Test-only adapter for common conditional-sampling invariants.

The adapter deliberately stays explicit about runtime classes.  It is not a
second implementation of production dispatch and it is never used to test
the individual public signatures themselves.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from pyscarcopula import (
    EquicorrGaussianCopula,
    GaussianCopula,
    StochasticStudentCopula,
    StudentCopula,
    VineCopula,
)
from pyscarcopula.copula.base import BivariateCopula


def _positive_n_draws(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)):
        raise TypeError("n_draws must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError("n_draws must be positive")
    return result


def _reject_unsupported_controls(
        model: object,
        *,
        n_threads: int,
        return_diagnostics: bool) -> None:
    if n_threads != 1 and isinstance(
            model, (BivariateCopula, VineCopula)):
        raise TypeError(
            f"{type(model).__name__} direct predict does not expose "
            "n_threads"
        )
    if return_diagnostics and not isinstance(model, VineCopula):
        raise TypeError(
            f"{type(model).__name__} does not expose conditional diagnostics"
        )


def draw_conditional(
    model: object,
    n_draws: int,
    given: Mapping[int, float] | None,
    *,
    u_train: np.ndarray | None = None,
    parameter_path: float | np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    n_threads: int = 1,
    return_diagnostics: bool = False,
):
    """Draw through the model's direct public API with common test syntax."""

    n_draws = _positive_n_draws(n_draws)
    _reject_unsupported_controls(
        model,
        n_threads=n_threads,
        return_diagnostics=return_diagnostics,
    )
    if rng is None:
        rng = np.random.default_rng()

    if isinstance(model, GaussianCopula):
        if given is not None:
            return model.sample_conditional(
                n_draws, given=given, rng=rng, n_threads=n_threads
            )
        return model.sample(n_draws, rng=rng, n_threads=n_threads)

    if isinstance(model, StudentCopula):
        if given is not None:
            return model.sample_conditional(
                n_draws, given=given, rng=rng, n_threads=n_threads
            )
        if n_threads != 1:
            raise TypeError(
                "StudentCopula.sample does not expose n_threads"
            )
        return model.sample(n_draws, rng=rng)

    if isinstance(model, EquicorrGaussianCopula):
        return model.sample_conditional(
            n_draws,
            r=parameter_path,
            given=given,
            rng=rng,
            n_threads=n_threads,
        )

    if isinstance(model, StochasticStudentCopula):
        return model.sample_conditional(
            n_draws,
            r=parameter_path,
            given=given,
            rng=rng,
            n_threads=n_threads,
        )

    if isinstance(model, VineCopula):
        return model.predict(
            n_draws,
            u=u_train,
            given=given,
            rng=rng,
            return_diagnostics=return_diagnostics,
        )

    if isinstance(model, BivariateCopula):
        return model.predict(
            n_draws, u=u_train, given=given, rng=rng
        )

    raise TypeError(
        f"unsupported conditional runtime {type(model).__name__}"
    )
