from datetime import UTC, datetime, timedelta

import pytest

from ai_character_engine.character.profile import CharacterProfile
from ai_character_engine.events.models import CharacterEvent
from ai_character_engine.llm.models import LLMResponse
from ai_character_engine.memory import (
    DefaultMemoryWritePolicy,
    InMemoryEventLedger,
    InMemoryMemoryStore,
    JsonlEventLedger,
    MemoryConsolidator,
    MemoryManager,
    MemoryRecord,
)
from ai_character_engine.runtime.character_runtime import CharacterRuntime
from tests.fakes import ScriptedLLMClient


def test_event_ledger_is_append_only_even_when_memory_is_skipped() -> None:
    ledger = InMemoryEventLedger()
    policy = DefaultMemoryWritePolicy(
        minimum_importance=0.9,
        default_importance=0.2,
    )
    manager = MemoryManager(
        ledger=ledger,
        write_policy=policy,
        auto_consolidate_threshold=None,
    )
    event = CharacterEvent.user_message("routine hello")
    from ai_character_engine.state.models import CharacterState

    state = CharacterState().snapshot()
    record = manager.record_interaction(
        character_id="test",
        event=event,
        response=LLMResponse(text="hello"),
        state_before=state,
        state_after=state,
    )

    assert record is None
    entries = ledger.list_for_character("test")
    assert len(entries) == 1
    assert entries[0].event_id == event.id
    assert entries[0].response_text == "hello"


def test_jsonl_event_ledger_persists(tmp_path) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger = JsonlEventLedger(path)
    manager = MemoryManager(
        ledger=ledger,
        auto_consolidate_threshold=None,
    )
    event = CharacterEvent.user_message("I like Seven Samurai")
    from ai_character_engine.state.models import CharacterState

    state = CharacterState().snapshot()
    manager.record_interaction(
        character_id="test",
        event=event,
        response=LLMResponse(text="Noted."),
        state_before=state,
        state_after=state,
    )

    reloaded = JsonlEventLedger(path)
    entries = reloaded.list_for_character("test")
    assert len(entries) == 1
    assert entries[0].content == "I like Seven Samurai"


def test_consolidator_merges_repeated_memories_but_preserves_unrelated() -> None:
    now = datetime.now(UTC)
    repeated_a = MemoryRecord(
        character_id="test",
        summary="User likes 黑澤明 and 七武士.",
        importance=0.7,
        tags=("user_message", "user"),
        source_event_type="user_message",
        created_at=now - timedelta(days=10),
    )
    repeated_b = MemoryRecord(
        character_id="test",
        summary="User likes 黑澤明 and especially 七武士.",
        importance=0.75,
        tags=("user_message", "user"),
        source_event_type="user_message",
        created_at=now - timedelta(days=2),
    )
    unrelated = MemoryRecord(
        character_id="test",
        summary="User bought coffee yesterday.",
        importance=0.5,
        tags=("user_message", "user"),
        source_event_type="user_message",
        created_at=now - timedelta(days=1),
    )
    store = InMemoryMemoryStore([repeated_a, repeated_b, unrelated])
    consolidator = MemoryConsolidator(
        store,
        similarity_threshold=0.35,
    )

    result = consolidator.consolidate(character_id="test")
    records = store.list_for_character("test")

    assert result.changed
    assert result.before_count == 3
    assert result.after_count == 2
    assert len(result.created) == 1
    consolidated = result.created[0]
    assert consolidated.kind == "consolidated"
    assert consolidated.metadata["occurrences"] == 2
    assert "黑澤明" in consolidated.summary
    assert any("coffee" in record.summary for record in records)


def test_consolidation_never_deletes_event_ledger() -> None:
    ledger = InMemoryEventLedger()
    store = InMemoryMemoryStore()
    manager = MemoryManager(
        ledger=ledger,
        store=store,
        consolidator=MemoryConsolidator(store, similarity_threshold=0.25),
        auto_consolidate_threshold=2,
    )
    from ai_character_engine.state.models import CharacterState

    state = CharacterState().snapshot()
    for text in ("我喜歡七武士", "我真的很喜歡七武士"):
        manager.record_interaction(
            character_id="test",
            event=CharacterEvent.user_message(text),
            response=LLMResponse(text="知道了"),
            state_before=state,
            state_after=state,
        )

    assert len(ledger.list_for_character("test")) == 2
    records = store.list_for_character("test")
    assert len(records) == 1
    assert records[0].kind == "consolidated"


@pytest.mark.asyncio
async def test_runtime_exposes_ledger_and_auto_consolidation() -> None:
    character = CharacterProfile(id="test", name="Test", description="Test")
    store = InMemoryMemoryStore()
    memory = MemoryManager(
        store=store,
        consolidator=MemoryConsolidator(store, similarity_threshold=0.25),
        auto_consolidate_threshold=2,
    )
    runtime = CharacterRuntime(
        character=character,
        llm=ScriptedLLMClient(
            [LLMResponse(text="ok1"), LLMResponse(text="ok2")]
        ),
        memory_manager=memory,
        max_history_messages=0,
    )

    first = await runtime.process_event(
        CharacterEvent.user_message("我喜歡七武士")
    )
    second = await runtime.process_event(
        CharacterEvent.user_message("我真的很喜歡七武士")
    )

    assert first.text == "ok1"
    assert second.text == "ok2"
    assert memory.last_ledger_entry is not None
    assert second.memory_consolidation is not None
    assert second.memory_consolidation.changed
    assert second.ledger_entry is not None
    assert len(memory.ledger.list_for_character("test")) == 2


def test_new_evidence_reinforces_existing_consolidated_memory() -> None:
    store = InMemoryMemoryStore()
    consolidator = MemoryConsolidator(store, similarity_threshold=0.20)
    first = MemoryRecord(
        character_id="test",
        summary="User likes 七武士.",
        importance=0.7,
    )
    second = MemoryRecord(
        character_id="test",
        summary="User really likes 七武士.",
        importance=0.7,
    )
    store.add(first)
    store.add(second)
    first_result = consolidator.consolidate(character_id="test")
    assert first_result.changed
    assert store.list_for_character("test")[0].metadata["occurrences"] == 2

    store.add(
        MemoryRecord(
            character_id="test",
            summary="User said 七武士 is a favorite movie.",
            importance=0.75,
        )
    )
    second_result = consolidator.consolidate(character_id="test")
    records = store.list_for_character("test")

    assert second_result.changed
    assert len(records) == 1
    assert records[0].kind == "consolidated"
    assert records[0].metadata["occurrences"] == 3
