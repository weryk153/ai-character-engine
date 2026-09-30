from __future__ import annotations

import math
import re
from datetime import UTC, datetime
from time import perf_counter
from typing import Protocol

from .models import RetrievedMemory
from .evidence import classify_user_text, evidence_weight
from .store import MemoryStore
from .trace import RetrievalCandidateTrace, RetrievalResult, RetrievalTrace

_CJK = re.compile(r"[\u3400-\u9fff]")
_WORD = re.compile(r"[a-zA-Z0-9_]+")


def _terms(text: str) -> set[str]:
    lowered = text.lower()
    terms = set(_WORD.findall(lowered))
    cjk_chars = _CJK.findall(lowered)
    terms.update(cjk_chars)
    terms.update(
        "".join(cjk_chars[index : index + 2])
        for index in range(max(0, len(cjk_chars) - 1))
    )
    return {term for term in terms if term}


class MemoryRetriever:
    """Retrieval interface with the v0.8 lexical implementation as default.

    MemoryRetriever(store) and retrieve() remain compatible. New strategies
    override retrieve_with_trace(); legacy custom retrieve() implementations
    are supported by retrieve_with_trace() below (the adapter function).
    """

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def retrieve(
        self,
        *,
        character_id: str,
        query: str,
        limit: int = 5,
        now: datetime | None = None,
    ) -> list[RetrievedMemory]:
        return list(
            self.retrieve_with_trace(
                character_id=character_id,
                query=query,
                limit=limit,
                now=now,
            ).memories
        )

    def retrieve_with_trace(
        self,
        *,
        character_id: str,
        query: str,
        limit: int = 5,
        now: datetime | None = None,
    ) -> RetrievalResult:
        start = perf_counter()
        records = self.store.list_for_character(character_id) if limit > 0 else []
        active = [r for r in records if r.character_id == character_id and r.is_active]
        query_terms = _terms(query)
        current_time = now or datetime.now(UTC)
        scored: list[RetrievedMemory] = []
        for record in active:
            record_terms = _terms(record.summary + " " + " ".join(record.tags))
            overlap = (
                len(query_terms & record_terms)
                / math.sqrt(len(query_terms) * len(record_terms))
                if query_terms and record_terms
                else 0.0
            )
            age_days = max(
                0.0, (current_time - record.created_at).total_seconds() / 86400.0
            )
            recency = 1.0 / (1.0 + age_days / 30.0)
            score = overlap * 0.65 + record.importance * 0.25 + recency * 0.10
            # v0.29.3 provenance-aware ranking. Old stores may not have an
            # evidence_type field, so infer it conservatively from source text.
            source_content = record.metadata.get("source_content", "")
            evidence_type = record.evidence_type
            if evidence_type == "unknown" and record.source_event_type == "user_message" and isinstance(source_content, str):
                evidence_type = classify_user_text(source_content)
            score *= evidence_weight(evidence_type)
            if overlap == 0 and record.importance < 0.8:
                continue
            scored.append(RetrievedMemory(record=record, score=score))
        # Preserve the original v0.8 ordering, including stable ties.
        scored.sort(key=lambda item: (item.score, item.record.created_at), reverse=True)
        selected = scored[: max(0, limit)]
        selected_ids = tuple(item.record.id for item in selected)
        return RetrievalResult(
            tuple(selected),
            RetrievalTrace(
                strategy="lexical",
                character_id=character_id,
                query=query,
                requested_limit=limit,
                total_records=len(records),
                active_records=len(active),
                filtered_inactive=sum(not r.is_active for r in records),
                candidates=tuple(
                    RetrievalCandidateTrace(
                        memory_id=item.record.id,
                        lexical_score=item.score,
                        lexical_rank=rank,
                        final_score=item.score,
                        selected=item.record.id in selected_ids,
                    )
                    for rank, item in enumerate(scored, 1)
                ),
                selected_memory_ids=selected_ids,
                elapsed_ms=(perf_counter() - start) * 1000,
            ),
        )


class LexicalMemoryRetriever(MemoryRetriever):
    """Explicit name for the backward-compatible lexical/importance/recency ranker."""


class AsyncMemoryRetriever(Protocol):
    async def retrieve_with_trace_async(
        self,
        *,
        character_id: str,
        query: str,
        limit: int = 5,
        now: datetime | None = None,
        rewrite_context: tuple[str, ...] = (),
    ) -> RetrievalResult: ...


def retrieve_with_trace(
    retriever: MemoryRetriever,
    *,
    character_id: str,
    query: str,
    limit: int = 5,
    now: datetime | None = None,
) -> RetrievalResult:
    """Adapt old retrievers that expose only retrieve(), without changing their output."""
    traced = getattr(retriever, "retrieve_with_trace", None)
    legacy_override = (
        isinstance(retriever, MemoryRetriever)
        and type(retriever).retrieve is not MemoryRetriever.retrieve
        and type(retriever).retrieve_with_trace is MemoryRetriever.retrieve_with_trace
    )
    if callable(traced) and not legacy_override:
        return traced(character_id=character_id, query=query, limit=limit, now=now)
    start = perf_counter()
    kwargs = dict(character_id=character_id, query=query, limit=limit)
    if now is not None:
        kwargs["now"] = now
    memories = tuple(retriever.retrieve(**kwargs))
    return RetrievalResult(
        memories,
        RetrievalTrace(
            strategy="custom",
            character_id=character_id,
            query=query,
            requested_limit=limit,
            candidates=tuple(
                RetrievalCandidateTrace(
                    memory_id=item.record.id,
                    final_score=item.score,
                    selected=True,
                )
                for item in memories
            ),
            selected_memory_ids=tuple(item.record.id for item in memories),
            elapsed_ms=(perf_counter() - start) * 1000,
        ),
    )


async def retrieve_with_trace_async(
    retriever,
    *,
    character_id: str,
    query: str,
    limit: int = 5,
    now: datetime | None = None,
    rewrite_context: tuple[str, ...] = (),
) -> RetrievalResult:
    """Async adapter for production retrievers while preserving v0.9 sync ones."""
    async_method = getattr(retriever, "retrieve_with_trace_async", None)
    if callable(async_method):
        import inspect
        kwargs = dict(character_id=character_id, query=query, limit=limit, now=now)
        try:
            parameters = inspect.signature(async_method).parameters
        except (TypeError, ValueError):
            parameters = {}
        if "rewrite_context" in parameters:
            kwargs["rewrite_context"] = rewrite_context
        return await async_method(**kwargs)
    return retrieve_with_trace(
        retriever, character_id=character_id, query=query, limit=limit, now=now
    )
