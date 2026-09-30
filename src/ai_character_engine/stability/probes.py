from __future__ import annotations

import asyncio
import gc
import tracemalloc
from collections.abc import Callable, Mapping
from typing import Protocol


class StabilityProbe(Protocol):
    def sample(self) -> Mapping[str, float]: ...


class AsyncioTaskProbe:
    """Counts other unfinished asyncio Tasks in the current loop.

    The current harness task is excluded, making a quiescent baseline/final sample
    useful for detecting leaked heartbeat/polling/background tasks.
    """

    def sample(self) -> Mapping[str, float]:
        current = asyncio.current_task()
        pending = sum(1 for task in asyncio.all_tasks() if task is not current and not task.done())
        return {"asyncio.pending_tasks": float(pending)}


class PythonRuntimeProbe:
    """Optional CPython process probe for stabilization tests.

    tracemalloc is opt-in because it is process-global and carries overhead. The
    probe never decides pass/fail; callers choose explicit MetricInvariant budgets.
    """

    def __init__(self, *, trace_memory: bool = False) -> None:
        self.trace_memory = bool(trace_memory)
        self._owns_tracemalloc = False
        if self.trace_memory and not tracemalloc.is_tracing():
            tracemalloc.start()
            self._owns_tracemalloc = True

    def sample(self) -> Mapping[str, float]:
        metrics: dict[str, float] = {"python.gc_objects": float(len(gc.get_objects()))}
        if self.trace_memory:
            current, peak = tracemalloc.get_traced_memory()
            metrics["python.traced_current_bytes"] = float(current)
            metrics["python.traced_peak_bytes"] = float(peak)
        return metrics

    def close(self) -> None:
        if self._owns_tracemalloc and tracemalloc.is_tracing():
            tracemalloc.stop()
            self._owns_tracemalloc = False


class CompositeProbe:
    def __init__(self, *probes: StabilityProbe) -> None:
        self._probes = tuple(probes)

    def sample(self) -> Mapping[str, float]:
        metrics: dict[str, float] = {}
        for probe in self._probes:
            for key, value in probe.sample().items():
                if key in metrics:
                    raise ValueError(f"duplicate stability metric: {key}")
                metrics[str(key)] = float(value)
        return metrics


class CallableProbe:
    def __init__(self, sampler: Callable[[], Mapping[str, float]]) -> None:
        self._sampler = sampler

    def sample(self) -> Mapping[str, float]:
        return {str(key): float(value) for key, value in self._sampler().items()}
