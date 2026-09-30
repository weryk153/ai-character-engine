from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Literal, Protocol

from ai_character_engine.events.models import CharacterEvent

from .models import MemoryRecord
from .retriever import _terms
from .store import MemoryStore

MemoryRevisionAction = Literal["none", "supersede", "forget"]


@dataclass(slots=True, frozen=True)
class MemoryRevisionPlan:
    action: MemoryRevisionAction = "none"
    target_ids: tuple[str, ...] = ()
    reason: str = ""
    requested_action: MemoryRevisionAction = "none"


@dataclass(slots=True, frozen=True)
class MemoryRevisionResult:
    action: MemoryRevisionAction = "none"
    target_ids: tuple[str, ...] = ()
    created_memory_id: str | None = None
    reason: str = ""
    requested_action: MemoryRevisionAction = "none"

    @property
    def changed(self) -> bool:
        return self.action != "none" and bool(self.target_ids)


class MemoryRevisionPolicy(Protocol):
    def plan(
        self,
        *,
        event: CharacterEvent,
        existing: list[MemoryRecord],
    ) -> MemoryRevisionPlan:
        ...


class HeuristicMemoryRevisionPolicy:
    """Conservative deterministic baseline for corrections and forgetting.

    Explicit host directives in event.payload take priority. Natural-language
    detection only acts on clear correction/forgetting signals and requires a
    lexically related active memory. This is intentionally conservative: a
    false "keep both" is safer than silently deleting the wrong memory.
    """

    _FORGET_SIGNALS = (
        "忘記",
        "別記",
        "不要記得",
        "不要再記",
        "forget",
        "don't remember",
        "do not remember",
    )
    _CORRECTION_SIGNALS = (
        "其實",
        "更正",
        "記錯",
        "不再",
        "已經不",
        "不是",
        "改成",
        "actually",
        "correction",
        "no longer",
        "used to",
        "instead",
    )
    _NOISE = (
        "忘記",
        "別記",
        "不要記得",
        "不要再記",
        "其實",
        "更正",
        "記錯",
        "不再",
        "已經不",
        "不是",
        "改成",
        "forget",
        "don't remember",
        "do not remember",
        "actually",
        "correction",
        "no longer",
        "used to",
        "instead",
    )

    def __init__(self, *, similarity_threshold: float = 0.16) -> None:
        self.similarity_threshold = similarity_threshold

    def plan(
        self,
        *,
        event: CharacterEvent,
        existing: list[MemoryRecord],
    ) -> MemoryRevisionPlan:
        active = [record for record in existing if record.is_active]

        explicit_action = str(event.payload.get("memory_action", "")).strip().lower()
        explicit_ids = tuple(
            str(item)
            for item in event.payload.get("memory_target_ids", ())
            if str(item)
        )
        active_ids = {record.id for record in active}
        explicit_ids = tuple(item for item in explicit_ids if item in active_ids)

        text = event.content.strip().lower()
        requested_from_text: MemoryRevisionAction = "none"
        if any(signal in text for signal in self._FORGET_SIGNALS):
            requested_from_text = "forget"
        elif any(signal in text for signal in self._CORRECTION_SIGNALS):
            requested_from_text = "supersede"

        if not active:
            if explicit_action in {"forget", "supersede", "correct"}:
                requested: MemoryRevisionAction = (
                    "forget" if explicit_action == "forget" else "supersede"
                )
                return MemoryRevisionPlan(
                    reason="no_matching_memory", requested_action=requested
                )
            if requested_from_text != "none":
                return MemoryRevisionPlan(
                    reason="no_matching_memory", requested_action=requested_from_text
                )
            return MemoryRevisionPlan()

        if explicit_action in {"forget", "supersede", "correct"}:
            action: MemoryRevisionAction = (
                "forget" if explicit_action == "forget" else "supersede"
            )
            if explicit_ids:
                return MemoryRevisionPlan(
                    action=action,
                    target_ids=explicit_ids,
                    reason="explicit_target_ids",
                    requested_action=action,
                )
            query = str(event.payload.get("memory_query") or event.content)
            targets = self._match(query, active, limit=3 if action == "forget" else 1)
            return MemoryRevisionPlan(
                action=action if targets else "none",
                target_ids=targets,
                reason="explicit_query" if targets else "no_matching_memory",
                requested_action=action,
            )

        if any(signal in text for signal in self._FORGET_SIGNALS):
            targets = self._match(event.content, active, limit=3)
            return MemoryRevisionPlan(
                action="forget" if targets else "none",
                target_ids=targets,
                reason="natural_language_forget" if targets else "no_matching_memory",
                requested_action="forget",
            )

        if any(signal in text for signal in self._CORRECTION_SIGNALS):
            targets = self._match(event.content, active, limit=1)
            return MemoryRevisionPlan(
                action="supersede" if targets else "none",
                target_ids=targets,
                reason="natural_language_correction" if targets else "no_matching_memory",
                requested_action="supersede",
            )

        return MemoryRevisionPlan()

    def _match(
        self,
        query: str,
        records: list[MemoryRecord],
        *,
        limit: int,
    ) -> tuple[str, ...]:
        cleaned = query.lower()
        for noise in self._NOISE:
            cleaned = cleaned.replace(noise, " ")
        query_terms = _terms(cleaned)
        if not query_terms:
            return ()

        scored: list[tuple[float, MemoryRecord]] = []
        for record in records:
            source_content = str(record.metadata.get("source_content", ""))
            record_terms = _terms(record.summary + " " + source_content)
            if not record_terms:
                continue
            score = len(query_terms & record_terms) / len(query_terms | record_terms)
            if score >= self.similarity_threshold:
                scored.append((score, record))
        scored.sort(key=lambda item: (item[0], item[1].created_at), reverse=True)
        return tuple(record.id for _, record in scored[:limit])


def guard_revision_response_text(text: str, plan: MemoryRevisionPlan) -> tuple[str, bool]:
    """Keep user-visible memory-operation acknowledgements consistent with engine state.

    Forget is the dangerous case: a model can say "I remembered it" while the
    deterministic memory layer has already selected the item for forgetting.
    For a forget request we therefore emit a small authoritative receipt. This
    is intentionally narrow; ordinary conversation and corrections remain
    model-authored. If storage later fails, CharacterRuntime fails the whole
    turn and the guarded text is not returned to the host.
    """

    if plan.requested_action != "forget":
        return text, False
    if plan.action == "forget" and plan.target_ids:
        return (
            "好的，已完成忘記這項長期記憶；之後不會再把它作為可檢索的記憶使用。",
            True,
        )
    return (
        "我沒有找到符合的已儲存長期記憶，因此沒有刪除任何記憶。",
        True,
    )


def revision_context(plan: MemoryRevisionPlan) -> str | None:
    """Authoritative pre-commit guidance for the final model response.

    The plan is deterministic and is reused during record_interaction. If the
    eventual storage commit fails, the whole turn fails rather than returning
    a response that claims a mutation succeeded.
    """

    requested = plan.requested_action
    if requested == "forget":
        if plan.action == "forget" and plan.target_ids:
            return (
                "Authoritative memory operation: this turn matched stored memory for forgetting. "
                "If the turn completes, those records will be marked forgotten and excluded from "
                "working-memory retrieval. Acknowledge the forget request; do not say you saved, "
                "remembered, or will keep the forgotten information."
            )
        return (
            "Authoritative memory operation: the user asked to forget something, but no matching "
            "active long-term memory was found. Do not claim that a stored memory was removed; "
            "briefly acknowledge that there was no matching stored item to forget."
        )
    if requested == "supersede":
        if plan.action == "supersede" and plan.target_ids:
            return (
                "Authoritative memory operation: this turn is a correction that will supersede "
                "matching older memory if the turn completes. Treat the user's current assertion "
                "as the current fact and do not present the superseded value as still current."
            )
        return (
            "Authoritative memory operation: the user supplied a correction, but no matching "
            "active memory was found to supersede. Treat the current direct assertion as the new "
            "current fact without claiming that an old stored record was changed."
        )
    return None


def apply_revision(
    *,
    store: MemoryStore,
    character_id: str,
    plan: MemoryRevisionPlan,
    new_record: MemoryRecord | None,
    now: datetime | None = None,
) -> tuple[MemoryRecord | None, MemoryRevisionResult]:
    records = store.list_for_character(character_id)
    if plan.action == "none" or not plan.target_ids:
        if new_record is not None:
            store.add(new_record)
        return new_record, MemoryRevisionResult(
            reason=plan.reason, requested_action=plan.requested_action
        )

    target_ids = set(plan.target_ids)
    timestamp = now or datetime.now(UTC)
    replacement: list[MemoryRecord] = []

    if plan.action == "forget":
        for record in records:
            if record.id in target_ids and record.is_active:
                replacement.append(
                    replace(record, status="forgotten", forgotten_at=timestamp)
                )
            else:
                replacement.append(record)
        store.replace_for_character(character_id, replacement)
        return None, MemoryRevisionResult(
            action="forget",
            target_ids=plan.target_ids,
            reason=plan.reason,
            requested_action=plan.requested_action,
        )

    if new_record is None:
        return None, MemoryRevisionResult(
            reason="supersede_without_new_memory", requested_action=plan.requested_action
        )

    revised_new = replace(new_record, supersedes=plan.target_ids)
    for record in records:
        if record.id in target_ids and record.is_active:
            replacement.append(
                replace(record, status="superseded", superseded_by=revised_new.id)
            )
        else:
            replacement.append(record)
    replacement.append(revised_new)
    store.replace_for_character(character_id, replacement)
    return revised_new, MemoryRevisionResult(
        action="supersede",
        target_ids=plan.target_ids,
        created_memory_id=revised_new.id,
        reason=plan.reason,
        requested_action=plan.requested_action,
    )
