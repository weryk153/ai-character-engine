from __future__ import annotations

import asyncio
import json

from ai_character_engine.character.profile import CharacterProfile
from ai_character_engine.context.builder import ContextBuilder
from ai_character_engine.performance import (
    BenchmarkConfig,
    BenchmarkRunner,
    PerformanceBudget,
    capture_benchmark_environment,
    evaluate_performance_budget,
)


async def main() -> None:
    character = CharacterProfile(id="perf-demo", name="PerfDemo", description="offline benchmark character")
    builder = ContextBuilder()
    environment = capture_benchmark_environment("offline-demo")

    def context_hot_path(_: int) -> None:
        builder.build(character=character, history=(), user_message="hello")

    baseline = await BenchmarkRunner(
        config=BenchmarkConfig(iterations=40, warmup_iterations=5),
        environment=environment,
    ).run("context_builder", context_hot_path, metadata={"role": "baseline"})

    current = await BenchmarkRunner(
        config=BenchmarkConfig(iterations=40, warmup_iterations=5),
        environment=environment,
    ).run("context_builder", context_hot_path, metadata={"role": "current"})

    evaluation = evaluate_performance_budget(
        current,
        baseline,
        budget=PerformanceBudget(
            max_p95_ratio=1.50,
            max_mean_ratio=1.50,
            min_throughput_ratio=0.50,
            max_peak_memory_ratio=None,
        ),
    )

    print(f"baseline p95={baseline.metrics.p95_ms:.4f} ms")
    print(f"current  p95={current.metrics.p95_ms:.4f} ms")
    print(f"current throughput={current.metrics.throughput_ops_s:.1f} ops/s")
    print(f"budget status={evaluation.status.value}")
    print("note: this is environment-scoped performance evidence, not a correctness test")
    print(json.dumps({"baseline": baseline.to_dict(), "current": current.to_dict(),
                      "evaluation": evaluation.to_dict()}, sort_keys=True))
    if not evaluation.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
