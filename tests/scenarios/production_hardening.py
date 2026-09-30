"""Offline production hardening demonstration."""
from __future__ import annotations

import asyncio

from ai_character_engine.production import (
    CircuitBreakerPolicy,
    DegradedMode,
    Idempotency,
    ProductionHardeningConfig,
    ProductionHardeningRuntime,
    ResourceLimits,
    RetryPolicy,
)


async def main() -> None:
    attempts = 0

    async def flaky_read() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ConnectionError("synthetic provider outage")
        return "provider recovered"

    runtime = ProductionHardeningRuntime(
        config=ProductionHardeningConfig(
            retry=RetryPolicy(max_attempts=3, base_backoff_s=0.01, max_backoff_s=0.02),
            circuit_breaker=CircuitBreakerPolicy(failure_threshold=3, recovery_timeout_s=5),
            resources=ResourceLimits(max_concurrent_operations=2, max_pending_operations=4),
        )
    )

    recovered = await runtime.run(
        "read-provider",
        flaky_read,
        dependency="demo-provider",
        idempotency=Idempotency.SAFE,
    )
    print(recovered.status.value, recovered.attempts, recovered.value)

    runtime.set_degraded_mode(DegradedMode.READ_ONLY)
    rejected = await runtime.run(
        "authoritative-commit",
        lambda: "must not execute",
        allow_in_read_only=False,
        idempotency=Idempotency.UNSAFE,
    )
    print(rejected.status.value, rejected.error_message)
    print(runtime.health_snapshot())


if __name__ == "__main__":
    asyncio.run(main())
