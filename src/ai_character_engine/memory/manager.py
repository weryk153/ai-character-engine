from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime

from ai_character_engine.events.models import CharacterEvent
from ai_character_engine.llm.models import LLMResponse
from ai_character_engine.state.models import CharacterStateSnapshot

from .consolidation import MemoryConsolidationResult, MemoryConsolidator
from .ledger import EventLedger, EventLedgerEntry, InMemoryEventLedger
from .models import MemoryRecord, RetrievedMemory
from .evidence import classify_memory_evidence
from .policy import DefaultMemoryWritePolicy, MemoryWritePolicy
from .retriever import (
    AsyncMemoryRetriever,
    MemoryRetriever,
    retrieve_with_trace,
    retrieve_with_trace_async,
)
from .revision import (
    HeuristicMemoryRevisionPolicy,
    MemoryRevisionPlan,
    MemoryRevisionPolicy,
    MemoryRevisionResult,
    apply_revision,
)
from .store import InMemoryMemoryStore, MemoryStore
from .summarizer import HeuristicMemorySummarizer, MemorySummarizer
from .trace import RetrievalResult, RetrievalTrace


class MemoryManager:
    """Coordinates ledger, memory write, consolidation, storage, and retrieval."""

    def __init__(
        self,
        *,
        store: MemoryStore | None = None,
        ledger: EventLedger | None = None,
        retriever: MemoryRetriever | AsyncMemoryRetriever | None = None,
        write_policy: MemoryWritePolicy | None = None,
        summarizer: MemorySummarizer | None = None,
        consolidator: MemoryConsolidator | None = None,
        revision_policy: MemoryRevisionPolicy | None = None,
        retrieval_limit: int = 5,
        auto_consolidate_threshold: int | None = 100,
    ) -> None:
        self.store = store or InMemoryMemoryStore()
        self.ledger = ledger or InMemoryEventLedger()
        self.retriever = retriever or MemoryRetriever(self.store)
        self.write_policy = write_policy or DefaultMemoryWritePolicy()
        self.summarizer = summarizer or HeuristicMemorySummarizer()
        self.consolidator = consolidator or MemoryConsolidator(self.store)
        self.revision_policy = revision_policy or HeuristicMemoryRevisionPolicy()
        self.retrieval_limit = retrieval_limit
        self.auto_consolidate_threshold = auto_consolidate_threshold
        # What memories are dated by, and the "now" they are ranked against. A
        # host with its own clock (CharacterCompanion, a game) sets it.
        self.clock: Callable[[], float] = time.time

    def _now(self) -> datetime:
        return datetime.fromtimestamp(self.clock(), UTC)
        self.last_ledger_entry: EventLedgerEntry | None = None
        self.last_consolidation_result: MemoryConsolidationResult | None = None
        self.last_revision_result: MemoryRevisionResult | None = None
        self.last_retrieval_trace: RetrievalTrace | None = None

    def retrieve_for_event(
        self,
        *,
        character_id: str,
        event: CharacterEvent,
    ) -> list[RetrievedMemory]:
        return list(self.retrieve_for_event_with_trace(
            character_id=character_id, event=event,
        ).memories)

    def retrieve_for_event_with_trace(
        self, *, character_id: str, event: CharacterEvent,
    ) -> RetrievalResult:
        self.last_retrieval_trace = None
        result = retrieve_with_trace(
            self.retriever, character_id=character_id,
            query=event_query(event), limit=self.retrieval_limit, now=self._now(),
        )
        self.last_retrieval_trace = result.trace
        return result

    async def retrieve_for_event_async(
        self, *, character_id: str, event: CharacterEvent,
    ) -> list[RetrievedMemory]:
        return list((await self.retrieve_for_event_with_trace_async(
            character_id=character_id, event=event,
        )).memories)

    async def retrieve_for_event_with_trace_async(
        self, *, character_id: str, event: CharacterEvent,
        rewrite_context: tuple[str, ...] = (),
    ) -> RetrievalResult:
        self.last_retrieval_trace = None
        result = await retrieve_with_trace_async(
            self.retriever, character_id=character_id,
            query=event_query(event), limit=self.retrieval_limit,
            rewrite_context=rewrite_context, now=self._now(),
        )
        self.last_retrieval_trace = result.trace
        return result

    def preview_revision(
        self, *, character_id: str, event: CharacterEvent
    ) -> MemoryRevisionPlan:
        """Plan a deterministic revision before inference without mutating storage."""

        existing = self.store.list_for_character(character_id)
        return self.revision_policy.plan(event=event, existing=existing)

    def record_interaction(
        self,
        *,
        character_id: str,
        event: CharacterEvent,
        response: LLMResponse,
        state_before: CharacterStateSnapshot,
        state_after: CharacterStateSnapshot,
        revision_plan: MemoryRevisionPlan | None = None,
    ) -> MemoryRecord | None:
        # The ledger is a complete append-only record, even when the event is
        # not important enough to become a retrievable long-term memory.
        ledger_entry = EventLedgerEntry.from_interaction(
            character_id=character_id,
            event=event,
            response=response,
            state_before=state_before,
            state_after=state_after,
        )
        self.ledger.append(ledger_entry)
        self.last_ledger_entry = ledger_entry
        self.last_consolidation_result = None
        self.last_revision_result = None

        if revision_plan is None:
            revision_plan = self.preview_revision(character_id=character_id, event=event)

        importance = self.write_policy.importance(
            event=event,
            response=response,
            state_before=state_before,
            state_after=state_after,
        )
        if importance is None and revision_plan.action != "forget":
            if revision_plan.requested_action != "none":
                _, revision_result = apply_revision(
                    store=self.store,
                    character_id=character_id,
                    plan=revision_plan,
                    new_record=None,
                    now=self._now(),
                )
                self.last_revision_result = revision_result
            return None

        # Forgetting is an operation on existing working memory. The raw event
        # is already safely preserved in the append-only ledger, so we do not
        # need a retrievable memory saying "the user asked us to forget".
        if revision_plan.action == "forget":
            _, revision_result = apply_revision(
                store=self.store,
                character_id=character_id,
                plan=revision_plan,
                new_record=None,
                now=self._now(),
            )
            self.last_revision_result = revision_result
            return None

        summary = self.summarizer.summarize(
            event=event,
            response=response,
            state_before=state_before,
            state_after=state_after,
        )
        tags = tuple(
            dict.fromkeys(
                (
                    event.type,
                    event.source,
                    *[str(key) for key in event.payload.keys()],
                )
            )
        )
        record = MemoryRecord(
            character_id=character_id,
            summary=summary,
            importance=importance,
            tags=tags,
            source_event_id=event.id,
            source_event_type=event.type,
            metadata={
                "source": event.source,
                "source_content": event.content,
                "evidence_type": classify_memory_evidence(event),
            },
            created_at=self._now(),
        )
        written, revision_result = apply_revision(
            store=self.store,
            character_id=character_id,
            plan=revision_plan,
            new_record=record,
            now=record.created_at,
        )
        self.last_revision_result = revision_result
        self._maybe_auto_consolidate(character_id)
        return written

    def append_background_candidate(
        self,
        *,
        character_id: str,
        summary: str,
        kind: str = "event",
        importance: float = 0.5,
        source_event_id: str | None = None,
        source_event_type: str | None = None,
        metadata: dict | None = None,
        tags: tuple[str, ...] = (),
    ) -> MemoryRecord:
        """Commit one coordinator-approved background memory candidate.

        This bypasses the interaction summarizer because the background worker already
        produced a typed summary. Callers are expected to perform confidence, freshness
        and provenance validation before invoking this authoritative write.
        """

        cleaned = summary.strip()
        if not cleaned:
            raise ValueError("background memory summary must not be empty")
        record = MemoryRecord(
            character_id=character_id,
            summary=cleaned,
            importance=importance,
            kind=kind,
            tags=tags,
            source_event_id=source_event_id,
            source_event_type=source_event_type,
            metadata=dict(metadata or {}),
            created_at=self._now(),
        )
        self.store.add(record)
        self._maybe_auto_consolidate(character_id)
        return record

    def consolidate(self, *, character_id: str) -> MemoryConsolidationResult:
        result = self.consolidator.consolidate(character_id=character_id)
        self.last_consolidation_result = result
        return result

    def _maybe_auto_consolidate(self, character_id: str) -> None:
        threshold = self.auto_consolidate_threshold
        if threshold is None or threshold <= 0:
            return
        if len(self.store.list_for_character(character_id)) >= threshold:
            self.consolidate(character_id=character_id)


def event_query(event: CharacterEvent) -> str:
    query = event.content
    if event.payload:
        query += " " + " ".join(f"{key} {value}" for key, value in event.payload.items())
    return query
