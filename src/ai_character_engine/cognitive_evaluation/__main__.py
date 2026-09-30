"""Offline cognition-wide regression runner. Semantic judge use remains explicit Python opt-in."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from . import CognitiveEvalDataset, evaluate_cognitive_dataset


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Offline AI Character Engine cognitive evaluation")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, help="Write a full JSON report including trace")
    parser.add_argument("--fail-on-violations", action="store_true", help="Exit 1 for violations or unassessed cases")
    args = parser.parse_args(argv)
    if args.output is not None and args.output.resolve() == args.dataset.resolve():
        parser.error("output must not overwrite the input dataset")
    try:
        report = asyncio.run(evaluate_cognitive_dataset(CognitiveEvalDataset.from_jsonl(args.dataset)))
        if args.output:
            report.to_json(args.output)
        print(json.dumps(report.to_dict()["metrics"], ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, OSError, TypeError) as exc:
        print(f"Cognitive evaluation failed: {exc}", file=sys.stderr)
        return 2
    return int(
        args.fail_on_violations
        and (report.metrics.failed_cases > 0 or report.metrics.unassessed_cases > 0 or report.metrics.total_cases == 0)
    )


if __name__ == "__main__":
    raise SystemExit(main())
