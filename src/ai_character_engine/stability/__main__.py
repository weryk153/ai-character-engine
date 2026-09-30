from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .models import MetricInvariant, SoakConfig
from .soak import SoakHarness


async def _run(iterations: int, concurrency: int):
    harness = SoakHarness(
        config=SoakConfig(iterations=iterations, concurrency=concurrency, sample_every=max(1, iterations // 10)),
        invariants=(MetricInvariant("asyncio.pending_tasks", max_growth=0),),
    )

    async def operation(index: int) -> int:
        await asyncio.sleep(0)
        return index

    return await harness.run(operation, metadata={"scenario": "offline_noop"})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the offline AI Character Engine stability smoke soak")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fail-on-violations", action="store_true")
    args = parser.parse_args(argv)
    report = asyncio.run(_run(args.iterations, args.concurrency))
    payload = report.to_dict()
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 1 if args.fail_on_violations and not report.passed else 0


if __name__ == "__main__":
    raise SystemExit(main())
