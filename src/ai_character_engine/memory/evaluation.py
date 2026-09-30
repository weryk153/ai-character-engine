from __future__ import annotations

import asyncio
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from .retriever import retrieve_with_trace_async


@dataclass(frozen=True, slots=True)
class RetrievalEvalCase:
    character_id: str
    query: str
    relevant_memory_ids: tuple[str, ...]
    case_id: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "RetrievalEvalCase":
        return cls(
            character_id=str(data["character_id"]),
            query=str(data["query"]),
            relevant_memory_ids=tuple(str(x) for x in data["relevant_memory_ids"]),
            case_id=str(data["case_id"]) if data.get("case_id") is not None else None,
        )




@dataclass(frozen=True, slots=True)
class RetrievalEvalDataset:
    cases: tuple[RetrievalEvalCase, ...]

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "RetrievalEvalDataset":
        return cls(load_eval_jsonl(path))


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    k: int
    recall_at_k: float
    precision_at_k: float
    mrr: float
    ndcg_at_k: float
    hit_rate_at_k: float


@dataclass(frozen=True, slots=True)
class RetrievalEvalCaseResult:
    case: RetrievalEvalCase
    retrieved_memory_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RetrievalEvalReport:
    metrics: RetrievalMetrics
    cases: tuple[RetrievalEvalCaseResult, ...] = field(default_factory=tuple)


def _case_metrics(retrieved: Sequence[str], relevant: set[str], k: int) -> tuple[float, float, float, float, float]:
    top = list(retrieved[:k])
    hits = [1 if memory_id in relevant else 0 for memory_id in top]
    hit_count = sum(hits)
    recall = hit_count / len(relevant) if relevant else 0.0
    precision = hit_count / k if k > 0 else 0.0
    reciprocal_rank = 0.0
    for rank, memory_id in enumerate(retrieved, 1):
        if memory_id in relevant:
            reciprocal_rank = 1.0 / rank
            break
    dcg = sum(hit / math.log2(rank + 1) for rank, hit in enumerate(hits, 1))
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    ndcg = dcg / idcg if idcg else 0.0
    hit_rate = 1.0 if hit_count else 0.0
    return recall, precision, reciprocal_rank, ndcg, hit_rate


def compute_metrics(
    results: Sequence[RetrievalEvalCaseResult], *, k: int
) -> RetrievalMetrics:
    if k < 1:
        raise ValueError("k must be positive")
    if not results:
        return RetrievalMetrics(k, 0.0, 0.0, 0.0, 0.0, 0.0)
    totals = [0.0] * 5
    for result in results:
        values = _case_metrics(
            result.retrieved_memory_ids,
            set(result.case.relevant_memory_ids),
            k,
        )
        totals = [a + b for a, b in zip(totals, values)]
    count = len(results)
    return RetrievalMetrics(k, *(value / count for value in totals))


def load_eval_jsonl(path: str | Path) -> tuple[RetrievalEvalCase, ...]:
    cases = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append(RetrievalEvalCase.from_dict(json.loads(line)))
    return tuple(cases)


class RetrievalEvaluator:
    def __init__(self, retriever, *, k: int = 5) -> None:
        if k < 1:
            raise ValueError("k must be positive")
        self.retriever = retriever
        self.k = k

    async def evaluate(self, cases: Iterable[RetrievalEvalCase]) -> RetrievalEvalReport:
        results = []
        for case in cases:
            result = await retrieve_with_trace_async(
                self.retriever,
                character_id=case.character_id,
                query=case.query,
                limit=self.k,
            )
            results.append(
                RetrievalEvalCaseResult(
                    case=case,
                    retrieved_memory_ids=tuple(item.record.id for item in result.memories),
                )
            )
        case_results = tuple(results)
        return RetrievalEvalReport(
            metrics=compute_metrics(case_results, k=self.k),
            cases=case_results,
        )

    def evaluate_sync(self, cases: Iterable[RetrievalEvalCase]) -> RetrievalEvalReport:
        return asyncio.run(self.evaluate(cases))


@dataclass(frozen=True, slots=True)
class RetrievalComparisonReport:
    reports: dict[str, RetrievalEvalReport]


async def compare_retrievers(
    retrievers: dict[str, object],
    cases: Iterable[RetrievalEvalCase],
    *,
    k: int = 5,
) -> RetrievalComparisonReport:
    frozen_cases = tuple(cases)
    reports: dict[str, RetrievalEvalReport] = {}
    for name, retriever in retrievers.items():
        reports[name] = await RetrievalEvaluator(retriever, k=k).evaluate(frozen_cases)
    return RetrievalComparisonReport(reports)
