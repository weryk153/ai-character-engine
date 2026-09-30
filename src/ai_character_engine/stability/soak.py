from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Iterable, Mapping
from typing import Any

from .models import (
    MetricInvariant,
    SoakConfig,
    SoakFailure,
    SoakReport,
    SoakSample,
    SoakStatus,
    SoakViolation,
)
from .probes import AsyncioTaskProbe, StabilityProbe

SoakOperation = Callable[[int], Any | Awaitable[Any]]


class SoakHarness:
    """Bounded offline soak runner with explicit failure/resource assertions.

    Iterations are executed in bounded batches of `concurrency`; the harness never
    creates one Task per requested iteration. It observes runtime health but owns no
    cognition, world, provider, broker, or host authority.
    """

    def __init__(
        self,
        *,
        config: SoakConfig | None = None,
        probe: StabilityProbe | None = None,
        invariants: Iterable[MetricInvariant] = (),
    ) -> None:
        self.config = config or SoakConfig()
        self.probe = probe or AsyncioTaskProbe()
        self.invariants = tuple(invariants)

    async def run(
        self,
        operation: SoakOperation,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> SoakReport:
        succeeded = failed = cancelled = completed = 0
        failures: list[SoakFailure] = []
        samples: list[SoakSample] = [self._sample(completed, succeeded, failed, cancelled)]
        next_sample = self.config.sample_every
        stopped_early = False

        for batch_start in range(0, self.config.iterations, self.config.concurrency):
            batch = range(batch_start, min(self.config.iterations, batch_start + self.config.concurrency))
            tasks = [asyncio.create_task(self._invoke(operation, index)) for index in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for index, result in zip(batch, results, strict=True):
                completed += 1
                if isinstance(result, asyncio.CancelledError):
                    cancelled += 1
                    self._append_failure(failures, SoakFailure(index, "CancelledError", str(result), True))
                elif isinstance(result, BaseException):
                    failed += 1
                    self._append_failure(
                        failures,
                        SoakFailure(index, type(result).__name__, str(result), False),
                    )
                else:
                    succeeded += 1

            if completed >= next_sample or completed == self.config.iterations:
                await asyncio.sleep(0)
                samples.append(self._sample(completed, succeeded, failed, cancelled))
                while next_sample <= completed:
                    next_sample += self.config.sample_every

            if self.config.fail_fast and failed + cancelled > self.config.max_failures:
                stopped_early = True
                break

        await asyncio.sleep(0)
        if samples[-1].completed != completed:
            samples.append(self._sample(completed, succeeded, failed, cancelled))
        else:
            # Re-sample at quiescence even if counters did not change. This catches
            # leaked heartbeat/background tasks that survive the last operation.
            samples.append(self._sample(completed, succeeded, failed, cancelled))

        violations = self._evaluate(samples, failed + cancelled)
        if stopped_early:
            violations.append(
                SoakViolation(
                    "fail_fast",
                    f"soak stopped after {completed}/{self.config.iterations} iterations",
                )
            )
        status = SoakStatus.PASSED if not violations else SoakStatus.FAILED
        return SoakReport(
            status=status,
            iterations=completed,
            succeeded=succeeded,
            failed=failed,
            cancelled=cancelled,
            samples=tuple(samples),
            failures=tuple(failures),
            violations=tuple(violations),
            metadata=dict(metadata or {}),
        )

    async def _invoke(self, operation: SoakOperation, iteration: int) -> Any:
        value = operation(iteration)
        if inspect.isawaitable(value):
            return await value
        return value

    def _sample(self, completed: int, succeeded: int, failed: int, cancelled: int) -> SoakSample:
        return SoakSample(completed, succeeded, failed, cancelled, self.probe.sample())

    def _append_failure(self, failures: list[SoakFailure], failure: SoakFailure) -> None:
        if len(failures) < self.config.max_recorded_failures:
            failures.append(failure)

    def _evaluate(self, samples: list[SoakSample], failures: int) -> list[SoakViolation]:
        violations: list[SoakViolation] = []
        if failures > self.config.max_failures:
            violations.append(
                SoakViolation(
                    "failure_budget_exceeded",
                    f"observed {failures} failures/cancellations; limit is {self.config.max_failures}",
                    observed=float(failures),
                    limit=float(self.config.max_failures),
                )
            )
        first = samples[0].metrics
        last = samples[-1].metrics
        for invariant in self.invariants:
            if invariant.metric not in first or invariant.metric not in last:
                violations.append(
                    SoakViolation(
                        "metric_missing",
                        f"metric {invariant.metric!r} was not present in baseline/final samples",
                        metric=invariant.metric,
                    )
                )
                continue
            baseline = float(first[invariant.metric])
            final = float(last[invariant.metric])
            if invariant.max_growth is not None and final - baseline > invariant.max_growth:
                violations.append(
                    SoakViolation(
                        "metric_growth_exceeded",
                        f"{invariant.metric} grew by {final - baseline:.3f}; limit is {invariant.max_growth:.3f}",
                        metric=invariant.metric,
                        observed=final - baseline,
                        limit=invariant.max_growth,
                    )
                )
            if invariant.max_final is not None and final > invariant.max_final:
                violations.append(
                    SoakViolation(
                        "metric_final_exceeded",
                        f"{invariant.metric} ended at {final:.3f}; limit is {invariant.max_final:.3f}",
                        metric=invariant.metric,
                        observed=final,
                        limit=invariant.max_final,
                    )
                )
        return violations
