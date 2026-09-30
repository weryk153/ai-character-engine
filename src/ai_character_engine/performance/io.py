from __future__ import annotations

import json
from pathlib import Path

from .models import BenchmarkResult


def save_benchmark_result(result: BenchmarkResult, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_benchmark_result(path: str | Path) -> BenchmarkResult:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("benchmark result must be a JSON object")
    return BenchmarkResult.from_dict(value)
