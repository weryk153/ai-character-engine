from __future__ import annotations

import asyncio

from ai_character_engine.stability import (
    FailureInjector,
    FailureRule,
    MetricInvariant,
    SoakConfig,
    SoakHarness,
    injected_failure,
)


async def main() -> None:
    injector = FailureInjector(FailureRule((11, 37), injected_failure("synthetic provider outage")))

    async def operation(iteration: int) -> int:
        injector.checkpoint()
        await asyncio.sleep(0)
        return iteration

    harness = SoakHarness(
        config=SoakConfig(
            iterations=50,
            concurrency=4,
            sample_every=10,
            max_failures=2,
        ),
        invariants=(MetricInvariant("asyncio.pending_tasks", max_growth=0),),
    )
    report = await harness.run(operation, metadata={"scenario": "offline_failure_injection"})
    print(f"status={report.status.value} succeeded={report.succeeded} failed={report.failed}")
    print(f"samples={len(report.samples)} violations={len(report.violations)}")
    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
