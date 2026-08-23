"""Balanced relative timing helpers for repository benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
import statistics
import time
from typing import Callable, Generic, Hashable, Mapping, TypeVar


Key = TypeVar("Key", bound=Hashable)


@dataclass(frozen=True)
class InterleavedTimings(Generic[Key]):
    """Timing samples collected in alternating, interleaved rounds."""

    samples: Mapping[Key, tuple[float, ...]]
    results: Mapping[Key, object]

    @property
    def medians(self) -> dict[Key, float]:
        return {
            key: float(statistics.median(values))
            for key, values in self.samples.items()
        }

    def median_ratio(self, numerator: Key, denominator: Key) -> float:
        """Return the median within-round ``numerator / denominator``."""
        numerator_samples = self.samples[numerator]
        denominator_samples = self.samples[denominator]
        if len(numerator_samples) != len(denominator_samples):
            raise ValueError("paired timing series must have equal lengths")
        return float(statistics.median(
            left / right
            for left, right in zip(
                numerator_samples, denominator_samples, strict=True)
        ))


def interleaved_timings(
        calls: Mapping[Key, Callable[[], object]],
        *,
        repeats: int = 5) -> InterleavedTimings[Key]:
    """Measure alternatives in balanced rounds under OS CPU scheduling.

    Even rounds use insertion order and odd rounds reverse it. Ratios can then
    be computed within each round, preventing complete baseline and candidate
    phases from being systematically scheduled on different CPU-core classes.
    """
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if not calls:
        raise ValueError("calls must not be empty")

    keys = tuple(calls)
    samples: dict[Key, list[float]] = {key: [] for key in keys}
    results: dict[Key, object] = {}
    for repeat in range(repeats):
        order = keys if repeat % 2 == 0 else tuple(reversed(keys))
        for key in order:
            started = time.perf_counter()
            results[key] = calls[key]()
            samples[key].append(time.perf_counter() - started)

    return InterleavedTimings(
        samples={key: tuple(values) for key, values in samples.items()},
        results=results,
    )
