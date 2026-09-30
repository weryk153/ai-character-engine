from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .evaluator import CharacterEvaluator, PersonaEvaluator
from .models import EvalCase, EvalDimension, EvalResult, Severity


@dataclass(frozen=True, slots=True)
class CharacterEvalDataset:
    cases: tuple[EvalCase, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "cases", tuple(self.cases))
        ids = [case.case_id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("dataset case IDs must be unique")

    @classmethod
    def from_jsonl(cls, path: str | Path) -> CharacterEvalDataset:
        cases = []
        seen = set()
        with Path(path).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    if not isinstance(data, dict):
                        raise ValueError("each row must be a JSON object")
                    case = EvalCase.from_dict(data)
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
class AggregateMetrics:
    total_cases: int
    evaluated_cases: int
    passed_cases: int
    failed_cases: int
    unassessed_cases: int
    pass_rate: float | None
    per_dimension_failure_rate: dict[str, float | None]
    per_dimension_coverage: dict[str, float]
    severity_counts: dict[str, int]
    mean_severity: float | None
    consistency_score: float | None
    labeled_cases: int = 0
    label_accuracy: float | None = None


def aggregate_metrics(results: Iterable[EvalResult]) -> AggregateMetrics:
    results = tuple(results)
    assessed = tuple(r for r in results if r.passed is not None)
    passed = sum(r.passed is True for r in assessed)
    dimension_rates = {}
    coverage = {}
    for dimension in EvalDimension:
        applicable = tuple(r for r in results if dimension in r.evaluated_dimensions)
        failed = sum(any(t.dimension == dimension for t in r.violations) for r in applicable)
        dimension_rates[dimension.value] = failed / len(applicable) if applicable else None
        coverage[dimension.value] = len(applicable) / len(results) if results else 0.0
    # Each case contributes its highest severity exactly once, even if both
    # evaluators find the same violation or several constraints fail.
    counts = {s.value: sum(r.severity == s for r in assessed) for s in Severity}
    counts["none"] = passed
    return AggregateMetrics(
        total_cases=len(results), evaluated_cases=len(assessed), passed_cases=passed,
        failed_cases=len(assessed) - passed, unassessed_cases=len(results) - len(assessed),
        pass_rate=passed / len(assessed) if assessed else None,
        per_dimension_failure_rate=dimension_rates, per_dimension_coverage=coverage,
        severity_counts=counts,
        mean_severity=sum(r.severity.penalty if r.severity else 0 for r in assessed) / len(assessed) if assessed else None,
        consistency_score=sum(r.consistency_score for r in assessed) / len(assessed) if assessed else None,
    )


@dataclass(frozen=True, slots=True)
class CharacterEvalReport:
    results: tuple[EvalResult, ...]
    metrics: AggregateMetrics

    def to_dict(self) -> dict:
        return {"schema_version": "0.12", "metrics": asdict(self.metrics), "results": [r.to_dict() for r in self.results]}

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


async def evaluate_dataset(
    dataset: CharacterEvalDataset, evaluator: CharacterEvaluator | None = None,
) -> CharacterEvalReport:
    from dataclasses import replace

    evaluator = evaluator if evaluator is not None else PersonaEvaluator()
    results = tuple([await evaluator.evaluate(case) for case in dataset.cases])
    if any(result.case_id != case.case_id for case, result in zip(dataset.cases, results)):
        raise ValueError("evaluator returned a mismatched case ID")
    metrics = aggregate_metrics(results)
    labeled = [(case, result) for case, result in zip(dataset.cases, results) if case.expected_failures is not None]
    # Unassessed cases cannot count as correctly evaluated negative labels.
    correct = sum(result.passed is not None and set(case.expected_failures) == {v.dimension for v in result.violations}
                  for case, result in labeled)
    metrics = replace(metrics, labeled_cases=len(labeled), label_accuracy=correct / len(labeled) if labeled else None)
    return CharacterEvalReport(results, metrics)
