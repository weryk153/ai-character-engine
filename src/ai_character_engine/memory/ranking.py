from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from .models import RetrievedMemory


@dataclass(frozen=True, slots=True)
class ReciprocalRankFusion:
    """Weighted RRF avoids comparing lexical scores directly with cosine scores."""

    k: float = 60.0
    lexical_weight: float = 1.0
    vector_weight: float = 1.0

    def __post_init__(self) -> None:
        values = (self.k, self.lexical_weight, self.vector_weight)
        if not all(math.isfinite(v) and v >= 0 for v in values):
            raise ValueError("RRF parameters must be finite and nonnegative")
        if self.lexical_weight + self.vector_weight <= 0:
            raise ValueError("at least one RRF weight must be positive")

    @property
    def max_score(self) -> float:
        return (self.lexical_weight + self.vector_weight) / (self.k + 1)

    def fuse(
        self, lexical: Sequence[RetrievedMemory], vector: Sequence[RetrievedMemory]
    ) -> list[RetrievedMemory]:
        scores: dict[str, float] = {}
        records = {}
        for items, weight in (
            (lexical, self.lexical_weight),
            (vector, self.vector_weight),
        ):
            if weight == 0:
                continue
            seen: set[str] = set()
            for rank, item in enumerate(items, 1):
                memory_id = item.record.id
                if memory_id in seen:
                    continue
                seen.add(memory_id)
                records[memory_id] = item.record
                scores[memory_id] = scores.get(memory_id, 0.0) + weight / (
                    self.k + rank
                )
        return sorted(
            (RetrievedMemory(records[key], score) for key, score in scores.items()),
            key=lambda item: (-item.score, item.record.id),
        )


class MemoryReranker(Protocol):
    def rerank(
        self,
        candidates: Sequence[RetrievedMemory],
        *,
        query: str,
        now: datetime | None = None,
    ) -> list[RetrievedMemory]: ...


@dataclass(frozen=True, slots=True)
class ImportanceRecencyReranker:
    """Rerank normalized relevance with small, bounded memory-priority signals."""

    relevance_weight: float = 0.85
    importance_weight: float = 0.10
    recency_weight: float = 0.05

    def __post_init__(self) -> None:
        weights = (self.relevance_weight, self.importance_weight, self.recency_weight)
        if not all(math.isfinite(w) and w >= 0 for w in weights) or sum(weights) <= 0:
            raise ValueError(
                "reranker weights must be finite, nonnegative and not all zero"
            )

    def rerank(
        self,
        candidates: Sequence[RetrievedMemory],
        *,
        query: str,
        now: datetime | None = None,
    ) -> list[RetrievedMemory]:
        current_time = now or datetime.now(UTC)
        total = self.relevance_weight + self.importance_weight + self.recency_weight
        ranked = []
        for item in candidates:
            age_days = max(
                0.0, (current_time - item.record.created_at).total_seconds() / 86400
            )
            score = (
                self.relevance_weight * max(0.0, min(1.0, item.score))
                + self.importance_weight * item.record.importance
                + self.recency_weight / (1.0 + age_days / 30.0)
            ) / total
            ranked.append(RetrievedMemory(item.record, score))
        return sorted(ranked, key=lambda item: (-item.score, item.record.id))


class AsyncMemoryReranker(Protocol):
    async def rerank_async(
        self,
        candidates: Sequence[RetrievedMemory],
        *,
        query: str,
        now: datetime | None = None,
    ) -> list[RetrievedMemory]: ...


@dataclass(frozen=True, slots=True)
class CallableAsyncMemoryReranker:
    """Provider-neutral adapter for learned/cross-encoder rerankers."""

    rerank_fn: object

    async def rerank_async(
        self,
        candidates: Sequence[RetrievedMemory],
        *,
        query: str,
        now: datetime | None = None,
    ) -> list[RetrievedMemory]:
        import inspect
        result = self.rerank_fn(tuple(candidates), query=query, now=now)
        if inspect.isawaitable(result):
            result = await result
        return list(result)


async def rerank_async(
    reranker: MemoryReranker | AsyncMemoryReranker | None,
    candidates: Sequence[RetrievedMemory],
    *,
    query: str,
    now: datetime | None = None,
) -> list[RetrievedMemory]:
    if reranker is None:
        return list(candidates)
    async_method = getattr(reranker, "rerank_async", None)
    if callable(async_method):
        return list(await async_method(candidates, query=query, now=now))
    return list(reranker.rerank(candidates, query=query, now=now))
