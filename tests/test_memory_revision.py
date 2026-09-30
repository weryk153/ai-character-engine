from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ai_character_engine.character.profile import CharacterProfile
from ai_character_engine.events.models import CharacterEvent
from ai_character_engine.llm.models import LLMResponse
from ai_character_engine.memory import (
    HeuristicMemoryRevisionPolicy,
    InMemoryEventLedger,
    InMemoryMemoryStore,
    JsonlMemoryStore,
    MemoryManager,
    MemoryRecord,
    MemoryRetriever,
)
from ai_character_engine.runtime.character_runtime import CharacterRuntime
from ai_character_engine.state.models import CharacterState
from tests.fakes import ScriptedLLMClient


def _record(manager: MemoryManager, text: str) -> MemoryRecord | None:
    state = CharacterState().snapshot()
    return manager.record_interaction(
        character_id="test",
        event=CharacterEvent.user_message(text),
        response=LLMResponse(text="ok"),
        state_before=state,
        state_after=state,
    )


def test_natural_language_correction_supersedes_related_active_memory() -> None:
    store = InMemoryMemoryStore()
    manager = MemoryManager(
        store=store,
        auto_consolidate_threshold=None,
        revision_policy=HeuristicMemoryRevisionPolicy(similarity_threshold=0.10),
    )
    old = _record(manager, "我最喜歡七武士")
    assert old is not None

    new = _record(manager, "其實我現在不再那麼喜歡七武士了")
    assert new is not None

    records = store.list_for_character("test")
    old_after = next(record for record in records if record.id == old.id)
    new_after = next(record for record in records if record.id == new.id)
    assert old_after.status == "superseded"
    assert old_after.superseded_by == new.id
    assert new_after.status == "active"
    assert old.id in new_after.supersedes
    assert manager.last_revision_result is not None
    assert manager.last_revision_result.action == "supersede"


def test_retriever_ignores_superseded_memory() -> None:
    old = MemoryRecord(
        id="old",
        character_id="test",
        summary="User said: I love coffee",
        status="superseded",
        superseded_by="new",
    )
    new = MemoryRecord(
        id="new",
        character_id="test",
        summary="User said: I no longer like coffee",
        supersedes=("old",),
    )
    retriever = MemoryRetriever(InMemoryMemoryStore([old, new]))

    results = retriever.retrieve(character_id="test", query="coffee")

    assert results
    assert all(item.record.id != "old" for item in results)
    assert results[0].record.id == "new"


def test_forget_request_marks_related_memory_forgotten_without_new_working_memory() -> None:
    store = InMemoryMemoryStore()
    ledger = InMemoryEventLedger()
    manager = MemoryManager(
        store=store,
        ledger=ledger,
        auto_consolidate_threshold=None,
        revision_policy=HeuristicMemoryRevisionPolicy(similarity_threshold=0.10),
    )
    old = _record(manager, "我喜歡咖啡")
    assert old is not None

    forgotten_write = _record(manager, "忘記我喜歡咖啡這件事")

    assert forgotten_write is None
    records = store.list_for_character("test")
    assert len(records) == 1
    assert records[0].id == old.id
    assert records[0].status == "forgotten"
    assert records[0].forgotten_at is not None
    assert manager.last_revision_result is not None
    assert manager.last_revision_result.action == "forget"
    # Both the original interaction and the forget request remain auditable.
    assert len(ledger.list_for_character("test")) == 2


def test_explicit_revision_target_is_deterministic() -> None:
    first = MemoryRecord(character_id="test", summary="User likes coffee")
    second = MemoryRecord(character_id="test", summary="User likes tea")
    store = InMemoryMemoryStore([first, second])
    manager = MemoryManager(store=store, auto_consolidate_threshold=None)
    state = CharacterState().snapshot()
    event = CharacterEvent(
        type="user_message",
        source="user",
        content="Please correct that preference.",
        payload={
            "memory_action": "correct",
            "memory_target_ids": [first.id],
        },
    )

    new = manager.record_interaction(
        character_id="test",
        event=event,
        response=LLMResponse(text="ok"),
        state_before=state,
        state_after=state,
    )

    assert new is not None
    records = {record.id: record for record in store.list_for_character("test")}
    assert records[first.id].status == "superseded"
    assert records[second.id].status == "active"
    assert new.supersedes == (first.id,)


def test_normal_new_information_does_not_revision_old_memory() -> None:
    store = InMemoryMemoryStore()
    manager = MemoryManager(store=store, auto_consolidate_threshold=None)
    first = _record(manager, "我喜歡咖啡")
    second = _record(manager, "我也喜歡拉麵")

    assert first is not None and second is not None
    records = store.list_for_character("test")
    assert {record.status for record in records} == {"active"}


def test_jsonl_memory_store_persists_revision_fields(tmp_path) -> None:
    path = tmp_path / "memory.jsonl"
    timestamp = datetime.now(UTC)
    store = JsonlMemoryStore(path)
    store.add(
        MemoryRecord(
            id="old",
            character_id="test",
            summary="old fact",
            status="forgotten",
            forgotten_at=timestamp,
            supersedes=("earlier",),
        )
    )

    reloaded = JsonlMemoryStore(path)
    record = reloaded.list_for_character("test")[0]
    assert record.status == "forgotten"
    assert record.forgotten_at == timestamp
    assert record.supersedes == ("earlier",)


@pytest.mark.asyncio
async def test_runtime_exposes_memory_revision_trace() -> None:
    character = CharacterProfile(id="test", name="Test", description="Test")
    store = InMemoryMemoryStore(
        [
            MemoryRecord(
                character_id="test",
                summary="User said: 我最喜歡七武士",
                metadata={"source_content": "我最喜歡七武士"},
            )
        ]
    )
    memory = MemoryManager(
        store=store,
        auto_consolidate_threshold=None,
        revision_policy=HeuristicMemoryRevisionPolicy(similarity_threshold=0.10),
    )
    runtime = CharacterRuntime(
        character=character,
        llm=ScriptedLLMClient([LLMResponse(text="知道了")]),
        memory_manager=memory,
        max_history_messages=0,
    )

    result = await runtime.process_event(
        CharacterEvent.user_message("其實我不再那麼喜歡七武士了")
    )

    assert result.memory_revision is not None
    assert result.memory_revision.action == "supersede"
    assert result.memory_revision.changed
