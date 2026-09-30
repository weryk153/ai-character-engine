from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

from .evaluator import CognitiveEvaluator
from .models import CognitiveEvalCase, CognitiveEvalDimension, CognitiveEvalResult
from ai_character_engine.evaluation.models import Severity


@dataclass(frozen=True, slots=True)
class CognitiveEvalDataset:
    cases: tuple[CognitiveEvalCase, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "cases", tuple(self.cases))
        ids = [case.case_id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("cognitive eval dataset case IDs must be unique")

    @classmethod
    def from_jsonl(cls, path: str | Path) -> CognitiveEvalDataset:
        cases: list[CognitiveEvalCase] = []
        seen: set[str] = set()
        with Path(path).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                    if not isinstance(raw, dict):
                        raise ValueError("each row must be a JSON object")
                    case = CognitiveEvalCase.from_dict(raw)
                    if case.case_id in seen:
                        raise ValueError(f"duplicate case ID: {case.case_id}")
                    seen.add(case.case_id)
                    cases.append(case)
                except (ValueError, TypeError, KeyError, AttributeError) as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
        return cls(tuple(cases))

    def to_jsonl(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as handle:
            for case in self.cases:
                handle.write(json.dumps(case.to_dict(), ensure_ascii=False, allow_nan=False) + "\n")


@dataclass(frozen=True, slots=True)
class CognitiveAggregateMetrics:
    total_cases: int
    evaluated_cases: int
    passed_cases: int
    failed_cases: int
    unassessed_cases: int
    pass_rate: float | None
    per_dimension_failure_rate: dict[str, float | None]
    per_dimension_coverage: dict[str, float]
    severity_counts: dict[str, int]
    mean_quality_score: float | None
    labeled_cases: int = 0
    label_accuracy: float | None = None


def aggregate_cognitive_metrics(results: Iterable[CognitiveEvalResult]) -> CognitiveAggregateMetrics:
    results = tuple(results)
    assessed = tuple(x for x in results if x.passed is not None)
    passed = sum(x.passed is True for x in assessed)
    rates: dict[str, float | None] = {}
    coverage: dict[str, float] = {}
    for dimension in CognitiveEvalDimension:
        applicable = tuple(x for x in results if dimension in x.evaluated_dimensions)
        failed = sum(any(v.dimension is dimension for v in x.violations) for x in applicable)
        rates[dimension.value] = failed / len(applicable) if applicable else None
        coverage[dimension.value] = len(applicable) / len(results) if results else 0.0
    severity_counts = {severity.value: sum(x.severity is severity for x in assessed) for severity in Severity}
    severity_counts["none"] = passed
    scores = [x.quality_score for x in assessed if x.quality_score is not None]
    return CognitiveAggregateMetrics(
        total_cases=len(results), evaluated_cases=len(assessed), passed_cases=passed,
        failed_cases=len(assessed) - passed, unassessed_cases=len(results) - len(assessed),
        pass_rate=passed / len(assessed) if assessed else None,
        per_dimension_failure_rate=rates, per_dimension_coverage=coverage,
        severity_counts=severity_counts,
        mean_quality_score=sum(scores) / len(scores) if scores else None,
    )


@dataclass(frozen=True, slots=True)
class CognitiveEvalReport:
    results: tuple[CognitiveEvalResult, ...]
    metrics: CognitiveAggregateMetrics

    def to_dict(self) -> dict:
        return {"schema_version": "0.37", "metrics": asdict(self.metrics), "results": [x.to_dict() for x in self.results]}

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


async def evaluate_cognitive_dataset(dataset: CognitiveEvalDataset, evaluator: CognitiveEvaluator | None = None) -> CognitiveEvalReport:
    evaluator = evaluator or CognitiveEvaluator()
    results = tuple([await evaluator.evaluate(case) for case in dataset.cases])
    if any(result.case_id != case.case_id for case, result in zip(dataset.cases, results)):
        raise ValueError("cognitive evaluator returned a mismatched case ID")
    metrics = aggregate_cognitive_metrics(results)
    labeled = [(case, result) for case, result in zip(dataset.cases, results) if case.expected_failures is not None]
    correct = sum(
        set(case.expected_failures or ()) == {v.dimension for v in result.violations}
        for case, result in labeled
    )
    metrics = replace(metrics, labeled_cases=len(labeled), label_accuracy=correct / len(labeled) if labeled else None)
    return CognitiveEvalReport(results, metrics)
