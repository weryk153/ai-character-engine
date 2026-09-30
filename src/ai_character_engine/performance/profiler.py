from __future__ import annotations

import asyncio
import inspect
import math
import os
import platform
import statistics
import time
import tracemalloc
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from .models import (
    BenchmarkConfig,
    BenchmarkEnvironment,
    BenchmarkMetrics,
    BenchmarkResult,
    BenchmarkSample,
)

BenchmarkOperation = Callable[[int], Any | Awaitable[Any]]
ClockNs = Callable[[], int]


def capture_benchmark_environment(
    profile_id: str = "local",
    *,
    metadata: Mapping[str, Any] | None = None,
) -> BenchmarkEnvironment:
    return BenchmarkEnvironment(
        profile_id=profile_id,
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        platform_system=platform.system(),
        machine=platform.machine(),
        cpu_count=os.cpu_count(),
        metadata=dict(metadata or {}),
    )


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


class BenchmarkRunner:
    """Bounded benchmark runner for environment-scoped performance evidence.

    Performance results are not correctness assertions. The runner owns no engine
    authority and receives only a caller-supplied operation.
    """

    def __init__(
        self,
        *,
        config: BenchmarkConfig | None = None,
        environment: BenchmarkEnvironment | None = None,
        clock_ns: ClockNs = time.perf_counter_ns,
        measure_peak_memory: bool = False,
    ) -> None:
        self.config = config or BenchmarkConfig()
        self.environment = environment or capture_benchmark_environment()
        self.clock_ns = clock_ns
        self.measure_peak_memory = measure_peak_memory

    async def run(
        self,
        name: str,
        operation: BenchmarkOperation,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> BenchmarkResult:
        if not name.strip():
            raise ValueError("benchmark name must not be empty")

        for index in range(self.config.warmup_iterations):
            await self._invoke(operation, -(index + 1))

        if self.measure_peak_memory:
            tracemalloc.start()
        start_total = self.clock_ns()
        samples: list[BenchmarkSample] = []
        failures = 0

        try:
            for batch_start in range(0, self.config.iterations, self.config.concurrency):
                batch = range(batch_start, min(self.config.iterations, batch_start + self.config.concurrency))
                tasks = [asyncio.create_task(self._measure(operation, index)) for index in batch]
                results = await asyncio.gather(*tasks)
                for sample in results:
                    samples.append(sample)
                    if not sample.success:
                        failures += 1
        finally:
            end_total = self.clock_ns()
            peak_memory = None
            if self.measure_peak_memory:
                _, peak_memory = tracemalloc.get_traced_memory()
                tracemalloc.stop()

        successful = [sample.duration_ns / 1_000_000 for sample in samples if sample.success]
        elapsed_ms = max(0.0, (end_total - start_total) / 1_000_000)
        metrics = self._metrics(successful, elapsed_ms, peak_memory)
        retained = tuple(samples) if self.config.record_samples else ()
        return BenchmarkResult(
            name=name,
            environment=self.environment,
            config=self.config,
            metrics=metrics,
            failures=failures,
            samples=retained,
            metadata=dict(metadata or {}),
        )

    async def _invoke(self, operation: BenchmarkOperation, index: int) -> Any:
        result = operation(index)
        if inspect.isawaitable(result):
            return await result
        return result

    async def _measure(self, operation: BenchmarkOperation, index: int) -> BenchmarkSample:
        start = self.clock_ns()
        try:
            await self._invoke(operation, index)
        except asyncio.CancelledError:
            end = self.clock_ns()
            return BenchmarkSample(index, max(0, end - start), False, "CancelledError")
        except BaseException as exc:
            end = self.clock_ns()
            return BenchmarkSample(index, max(0, end - start), False, type(exc).__name__)
        end = self.clock_ns()
        return BenchmarkSample(index, max(0, end - start), True)

    def _metrics(self, durations_ms: list[float], elapsed_ms: float, peak_memory: int | None) -> BenchmarkMetrics:
        if durations_ms:
            mean = statistics.fmean(durations_ms)
            median = statistics.median(durations_ms)
            minimum = min(durations_ms)
            maximum = max(durations_ms)
            p95 = _percentile(durations_ms, 0.95)
            p99 = _percentile(durations_ms, 0.99)
        else:
            mean = median = minimum = maximum = p95 = p99 = 0.0
        throughput = 0.0 if elapsed_ms <= 0 else len(durations_ms) / (elapsed_ms / 1000.0)
        return BenchmarkMetrics(
            count=len(durations_ms),
            mean_ms=mean,
            median_ms=median,
            p95_ms=p95,
            p99_ms=p99,
            min_ms=minimum,
            max_ms=maximum,
            throughput_ops_s=throughput,
            elapsed_ms=elapsed_ms,
            peak_memory_bytes=peak_memory,
        )
