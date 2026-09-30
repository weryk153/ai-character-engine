from datetime import UTC, datetime, timedelta

import pytest

from ai_character_engine.character.profile import CharacterProfile
from ai_character_engine.events.models import CharacterEvent
from ai_character_engine.llm.models import LLMResponse
from ai_character_engine.memory import (
    InMemoryMemoryStore,
    JsonlMemoryStore,
    MemoryManager,
    MemoryRecord,
    MemoryRetriever,
)
from ai_character_engine.runtime.character_runtime import CharacterRuntime
from tests.fakes import ScriptedLLMClient, system_context


@pytest.fixture
def character() -> CharacterProfile:
    return CharacterProfile(id="test", name="Test", description="Test character")


def test_memory_retriever_prefers_relevant_memory() -> None:
    now = datetime.now(UTC)
    store = InMemoryMemoryStore(
        [
            MemoryRecord(
                character_id="test",
                summary="User likes director 黑澤明 and the film 七武士.",
                importance=0.7,
                created_at=now - timedelta(days=5),
            ),
            MemoryRecord(
                character_id="test",
                summary="User bought coffee yesterday.",
                importance=0.4,
                created_at=now - timedelta(days=1),
            ),
        ]
    )
    retriever = MemoryRetriever(store)

    results = retriever.retrieve(
        character_id="test",
        query="我是不是說過我喜歡黑澤明？",
        now=now,
    )

    assert results
    assert "黑澤明" in results[0].record.summary


def test_jsonl_memory_store_persists_records(tmp_path) -> None:
    path = tmp_path / "memory.jsonl"
    store = JsonlMemoryStore(path)
    record = MemoryRecord(
        character_id="test",
        summary="User likes old movies.",
        importance=0.8,
        tags=("user_message",),
    )
    store.add(record)

    reloaded = JsonlMemoryStore(path)
    records = reloaded.list_for_character("test")

    assert len(records) == 1
    assert records[0].id == record.id
    assert records[0].summary == record.summary


@pytest.mark.asyncio
async def test_runtime_writes_memory(character: CharacterProfile) -> None:
    memory = MemoryManager()
    llm = ScriptedLLMClient([LLMResponse(text="I remember that.")])
    runtime = CharacterRuntime(
        character=character,
        llm=llm,
        memory_manager=memory,
        max_history_messages=0,
    )

    result = await runtime.process_event(
        CharacterEvent.user_message("我喜歡黑澤明的電影")
    )

    assert result.memory_written is not None
    assert "黑澤明" in result.memory_written.summary
    assert memory.store.list_for_character("test")


@pytest.mark.asyncio
async def test_runtime_retrieves_memory_into_context(character: CharacterProfile) -> None:
    store = InMemoryMemoryStore(
        [
            MemoryRecord(
                character_id="test",
                summary="User said they like 黑澤明 and 七武士.",
                importance=0.8,
            )
        ]
    )
    memory = MemoryManager(store=store)
    llm = ScriptedLLMClient([LLMResponse(text="Yes, you mentioned him before.")])
    runtime = CharacterRuntime(
        character=character,
        llm=llm,
        memory_manager=memory,
        max_history_messages=0,
    )

    result = await runtime.process_event(
        CharacterEvent.user_message("我之前有說過喜歡黑澤明嗎？")
    )

    assert result.retrieved_memories
    context = system_context(llm.calls[0])
    assert "- memory [" in context
    assert "黑澤明" in context


@pytest.mark.asyncio
async def test_memory_survives_new_runtime_with_jsonl_store(
    character: CharacterProfile,
    tmp_path,
) -> None:
    path = tmp_path / "memory.jsonl"
    first_memory = MemoryManager(store=JsonlMemoryStore(path))
    first_runtime = CharacterRuntime(
        character=character,
        llm=ScriptedLLMClient([LLMResponse(text="知道了。")]),
        memory_manager=first_memory,
        max_history_messages=0,
    )
    await first_runtime.run_turn("我最喜歡黑澤明的七武士")

    second_memory = MemoryManager(store=JsonlMemoryStore(path))
    second_llm = ScriptedLLMClient([LLMResponse(text="你提過七武士。")])
    second_runtime = CharacterRuntime(
        character=character,
        llm=second_llm,
        memory_manager=second_memory,
        max_history_messages=0,
    )
    result = await second_runtime.process_event(
        CharacterEvent.user_message("我喜歡哪一部黑澤明電影？")
    )

    assert result.retrieved_memories
    assert "七武士" in result.retrieved_memories[0].record.summary


def test_the_memory_file_is_never_half_written(tmp_path, monkeypatch):
    """rewrite_memories and a host's memory page rewrite the whole file. A
    crash in the middle would lose every memory of the character, and a torn
    last line then stops the store from loading at all."""
    import os
    from pathlib import Path

    from ai_character_engine.memory.store import JsonlMemoryStore

    path = tmp_path / "memory.jsonl"
    store = JsonlMemoryStore(path)
    store.add(MemoryRecord(character_id="c", summary="one", importance=0.5))
    written = []
    real_replace = os.replace

    def replace(source, destination):
        written.append((Path(source).name, Path(destination).name))
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", replace)

    store.replace_for_character(
        "c", [MemoryRecord(character_id="c", summary="two", importance=0.5)]
    )

    assert [record.summary for record in JsonlMemoryStore(path).list_for_character("c")] == ["two"]
    assert written and written[-1][1] == "memory.jsonl"
    assert not list(tmp_path.glob("*.tmp"))

