"""Memories that keep coming up stay fresh; the user's boundaries never fade."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta

from ai_character_engine.events.models import CharacterEvent
from ai_character_engine.memory import InMemoryMemoryStore, MemoryManager, MemoryRecord
from ai_character_engine.memory.retriever import retention


def test_recalled_memories_fade_slower_and_boundaries_never_fade():
    now = datetime(2026, 10, 15, tzinfo=UTC)
    fresh = MemoryRecord("mei", "likes tea", importance=0.5, created_at=now - timedelta(days=10))
    recalled = replace(
        fresh,
        metadata={"recalled": 4, "last_recalled_at": (now - timedelta(days=10)).isoformat()},
    )
    boundary = replace(fresh, importance=0.9)
    assert retention(fresh, now) < retention(recalled, now) < 1.0
    assert retention(boundary, now) == 1.0
    # A recall today restores it fully, however old the memory is.
    today = replace(fresh, metadata={"recalled": 1, "last_recalled_at": now.isoformat()})
    assert retention(today, now) == 1.0


def test_retrieval_marks_the_records_it_returned():
    store = InMemoryMemoryStore(
        [MemoryRecord("mei", "The user likes strong black tea.", importance=0.5)]
    )
    manager = MemoryManager(store=store)
    event = CharacterEvent.user_message("Want some tea?")
    (first,) = manager.retrieve_for_event(character_id="mei", event=event)
    assert first.record.metadata.get("recalled") in (None, 0)
    (second,) = manager.retrieve_for_event(character_id="mei", event=event)
    (stored,) = store.list_for_character("mei")
    assert stored.metadata["recalled"] == 2
    assert stored.metadata["last_recalled_at"]
    assert second.record.metadata["recalled"] == 1


def test_a_memory_recalled_often_outranks_its_unrecalled_twin():
    now = datetime(2026, 10, 15, tzinfo=UTC)
    old = now - timedelta(days=60)
    quiet = MemoryRecord("mei", "The user keeps a cat named Mochi.", importance=0.5, created_at=old)
    lively = MemoryRecord(
        "mei",
        "The user keeps a cat named Mochi.",
        importance=0.5,
        created_at=old,
        metadata={"recalled": 6, "last_recalled_at": (now - timedelta(days=2)).isoformat()},
    )
    manager = MemoryManager(store=InMemoryMemoryStore([quiet, lively]))
    manager.clock = lambda: now.timestamp()
    ranked = manager.retrieve_for_event(
        character_id="mei", event=CharacterEvent.user_message("How is Mochi the cat?")
    )
    assert [item.record.id for item in ranked] == [lively.id, quiet.id]


def test_recall_counts_survive_a_restart(tmp_path):
    from ai_character_engine.memory import JsonlMemoryStore

    path = tmp_path / "memory.jsonl"
    store = JsonlMemoryStore(path)
    store.add(MemoryRecord("mei", "The user likes strong black tea.", importance=0.5))
    MemoryManager(store=store).retrieve_for_event(
        character_id="mei", event=CharacterEvent.user_message("Tea?")
    )
    (reloaded,) = JsonlMemoryStore(path).list_for_character("mei")
    assert reloaded.metadata["recalled"] == 1
