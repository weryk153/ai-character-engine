"""Offline JSONL runner. No LLM flag: provider use is an explicit Python opt-in."""
import argparse
import asyncio
import json
import sys
from pathlib import Path

from . import CharacterEvalDataset, evaluate_dataset


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Offline persona consistency evaluation (no LLM calls)")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, help="Write full JSON report, including trace")
    parser.add_argument("--fail-on-violations", action="store_true", help="Exit 1 for violations or unassessed cases")
    args = parser.parse_args(argv)
    if args.output is not None and args.output.resolve() == args.dataset.resolve():
        parser.error("output must not overwrite the input dataset")
    try:
        report = asyncio.run(evaluate_dataset(CharacterEvalDataset.from_jsonl(args.dataset)))
        if args.output:
            report.to_json(args.output)
        print(json.dumps(report.to_dict()["metrics"], ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, OSError, TypeError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 2
    return int(args.fail_on_violations and (report.metrics.failed_cases > 0 or report.metrics.unassessed_cases > 0 or report.metrics.total_cases == 0))


if __name__ == "__main__":
    raise SystemExit(main())
