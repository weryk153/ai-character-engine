from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .budgets import evaluate_performance_budget
from .io import load_benchmark_result, save_benchmark_result
from .models import BenchmarkConfig, PerformanceBudget
from .profiler import BenchmarkRunner, capture_benchmark_environment


async def _run_noop(iterations: int, warmup: int, concurrency: int, profile_id: str, memory: bool):
    runner = BenchmarkRunner(
        config=BenchmarkConfig(iterations=iterations, warmup_iterations=warmup, concurrency=concurrency),
        environment=capture_benchmark_environment(profile_id),
        measure_peak_memory=memory,
    )

    async def operation(_: int) -> None:
        await asyncio.sleep(0)

    return await runner.run("offline_noop", operation, metadata={"portable_correctness_gate": False})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AI Character Engine performance benchmark/profile utility")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the offline benchmark smoke profile")
    run.add_argument("--iterations", type=int, default=200)
    run.add_argument("--warmup", type=int, default=20)
    run.add_argument("--concurrency", type=int, default=1)
    run.add_argument("--profile-id", default="local")
    run.add_argument("--measure-memory", action="store_true")
    run.add_argument("--output", type=Path)

    compare = sub.add_parser("compare", help="compare two reports from the same environment profile")
    compare.add_argument("baseline", type=Path)
    compare.add_argument("current", type=Path)
    compare.add_argument("--max-p95-ratio", type=float, default=1.20)
    compare.add_argument("--max-mean-ratio", type=float, default=1.20)
    compare.add_argument("--min-throughput-ratio", type=float, default=0.80)
    compare.add_argument("--max-peak-memory-ratio", type=float, default=1.25)
    compare.add_argument("--fail-on-regression", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "run":
        result = asyncio.run(_run_noop(args.iterations, args.warmup, args.concurrency, args.profile_id, args.measure_memory))
        if args.output:
            save_benchmark_result(result, args.output)
        else:
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result.passed else 1

    baseline = load_benchmark_result(args.baseline)
    current = load_benchmark_result(args.current)
    evaluation = evaluate_performance_budget(
        current,
        baseline,
        budget=PerformanceBudget(
            max_p95_ratio=args.max_p95_ratio,
            max_mean_ratio=args.max_mean_ratio,
            min_throughput_ratio=args.min_throughput_ratio,
            max_peak_memory_ratio=args.max_peak_memory_ratio,
        ),
    )
    print(json.dumps(evaluation.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if args.fail_on_regression and not evaluation.passed else 0


if __name__ == "__main__":
    raise SystemExit(main())
