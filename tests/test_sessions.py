from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ai_character_engine.character.profile import CharacterProfile
from ai_character_engine.memory.manager import MemoryManager
from ai_character_engine.session import (
    CharacterRuntimeFactory,
    InMemoryRelationshipStore,
    InMemorySessionStore,
    JsonFileRelationshipStore,
    JsonFileSessionStore,
    SessionManager,
    SessionUnavailableError,
)
from tests.fakes import FakeLLMClient


@pytest.fixture
def character() -> CharacterProfile:
    return CharacterProfile(
        id="mei",
        name="Mei",
        description="A test character",
    )


def build_factory(
    character: CharacterProfile,
    *,
    manager: SessionManager | None = None,
    memory_manager: MemoryManager | None = None,
    relationship_store=None,
) -> CharacterRuntimeFactory:
    return CharacterRuntimeFactory(
        characters={character.id: character},
        llm_factory=lambda record: FakeLLMClient("ok"),
        session_manager=manager or SessionManager(default_ttl_seconds=None),
        memory_manager=memory_manager,
        relationship_store=relationship_store,
    )


@pytest.mark.asyncio
async def test_multi_user_history_and_state_are_isolated(character: CharacterProfile) -> None:
    factory = build_factory(character)

    alice = factory.create(user_id="alice", character_id="mei", session_id="a")
    bob = factory.create(user_id="bob", character_id="mei", session_id="b")

    alice.runtime.state.trust = 88
    await alice.run_turn("hello from alice")
    await bob.run_turn("hello from bob")

    assert alice.runtime.state.trust == 88
    assert bob.runtime.state.trust == 50
    assert [message.content for message in alice.runtime.history] != [
        message.content for message in bob.runtime.history
    ]
    assert alice.runtime.memory_scope_id != bob.runtime.memory_scope_id


@pytest.mark.asyncio
async def test_same_user_multi_session_shares_relationship_but_not_history(
    character: CharacterProfile,
) -> None:
    relationships = InMemoryRelationshipStore()
    factory = build_factory(character, relationship_store=relationships)

    first = factory.create(user_id="alice", character_id="mei", session_id="s1")
    first.runtime.state.trust = 77
    first.runtime.state.favorability = 66
    first.runtime.state.relationship_stage = "friend"
    await first.run_turn("session one")

    second = factory.create(user_id="alice", character_id="mei", session_id="s2")

    assert second.runtime.state.trust == 77
    assert second.runtime.state.favorability == 66
    assert second.runtime.state.relationship_stage == "friend"
    assert second.runtime.history == []
    assert first.runtime.memory_scope_id == second.runtime.memory_scope_id


@pytest.mark.asyncio
async def test_memory_scope_is_user_character_not_session(character: CharacterProfile) -> None:
    memory = MemoryManager(auto_consolidate_threshold=None)
    factory = build_factory(character, memory_manager=memory)

    alice_one = factory.create(user_id="alice", character_id="mei", session_id="a1")
    alice_two = factory.create(user_id="alice", character_id="mei", session_id="a2")
    bob = factory.create(user_id="bob", character_id="mei", session_id="b1")

    await alice_one.run_turn("I like coffee")
    await bob.run_turn("I like tea")

    alice_records = memory.store.list_for_character(alice_one.runtime.memory_scope_id)
    bob_records = memory.store.list_for_character(bob.runtime.memory_scope_id)

    assert any("coffee" in record.summary for record in alice_records)
    assert not any("tea" in record.summary for record in alice_records)
    assert any("tea" in record.summary for record in bob_records)
    assert alice_one.runtime.memory_scope_id == alice_two.runtime.memory_scope_id


@pytest.mark.asyncio
async def test_json_session_store_restores_history_and_state_after_restart(
    character: CharacterProfile,
    tmp_path: Path,
) -> None:
    path = tmp_path / "sessions.json"
    store = JsonFileSessionStore(path)
    manager = SessionManager(store, default_ttl_seconds=None)
    factory = build_factory(character, manager=manager)

    session = factory.create(user_id="alice", character_id="mei", session_id="resume-me")
    session.runtime.state.emotion = "happy"
    session.runtime.state.energy = 42
    await session.run_turn("persist this")

    # Simulate a process restart by reconstructing store/manager/factory.
    store2 = JsonFileSessionStore(path)
    manager2 = SessionManager(store2, default_ttl_seconds=None)
    factory2 = build_factory(character, manager=manager2)
    restored = factory2.restore("resume-me")

    assert restored.runtime.state.emotion == "happy"
    assert restored.runtime.state.energy == 42
    assert [message.content for message in restored.runtime.history] == [
        "persist this",
        "ok",
    ]


@pytest.mark.asyncio
async def test_json_relationship_store_persists_user_character_relationship(
    character: CharacterProfile,
    tmp_path: Path,
) -> None:
    relationship_path = tmp_path / "relationships.json"
    relationships = JsonFileRelationshipStore(relationship_path)
    factory = build_factory(character, relationship_store=relationships)

    first = factory.create(user_id="alice", character_id="mei", session_id="s1")
    first.runtime.state.trust = 91
    first.runtime.state.relationship_stage = "close_friend"
    await first.run_turn("save relation")

    relationships2 = JsonFileRelationshipStore(relationship_path)
    factory2 = build_factory(character, relationship_store=relationships2)
    second = factory2.create(user_id="alice", character_id="mei", session_id="s2")

    assert second.runtime.state.trust == 91
    assert second.runtime.state.relationship_stage == "close_friend"


def test_session_ttl_expiration_and_cleanup(character: CharacterProfile) -> None:
    now = datetime(2026, 9, 18, 0, 0, tzinfo=UTC)
    current = [now]
    manager = SessionManager(
        InMemorySessionStore(),
        default_ttl_seconds=10,
        clock=lambda: current[0],
    )
    manager.create(user_id="alice", character_id=character.id, session_id="s1")

    current[0] = now + timedelta(seconds=11)
    expired = manager.cleanup_expired()

    assert expired == ("s1",)
    with pytest.raises(SessionUnavailableError):
        manager.require("s1")
    inactive = manager.require("s1", require_active=False)
    assert inactive.status == "expired"


def test_touch_extends_ttl_and_records_audit(character: CharacterProfile) -> None:
    now = datetime(2026, 9, 18, 0, 0, tzinfo=UTC)
    current = [now]
    manager = SessionManager(default_ttl_seconds=10, clock=lambda: current[0])
    record = manager.create(user_id="alice", character_id=character.id, session_id="s1")
    original_expiry = record.expires_at

    current[0] = now + timedelta(seconds=5)
    touched = manager.touch("s1")

    assert touched.expires_at is not None
    assert original_expiry is not None
    assert touched.expires_at > original_expiry
    assert [entry.action for entry in manager.audit_log] == ["created", "touched"]


def test_close_session_blocks_future_use(character: CharacterProfile) -> None:
    manager = SessionManager(default_ttl_seconds=None)
    manager.create(user_id="alice", character_id=character.id, session_id="s1")
    closed = manager.close("s1")

    assert closed.status == "closed"
    assert closed.closed_at is not None
    with pytest.raises(SessionUnavailableError):
        manager.require("s1")


def test_session_scopes_are_explicit(character: CharacterProfile) -> None:
    manager = SessionManager(default_ttl_seconds=None)
    record = manager.create(user_id="alice", character_id=character.id, session_id="abc")

    assert record.scopes.character_scope == "character:mei"
    assert record.scopes.user_scope == "user:alice"
    assert record.scopes.relationship_scope == "user:alice:character:mei"
    assert record.scopes.session_scope == "session:abc"
    assert record.scopes.memory_scope_id == record.scopes.relationship_scope


def test_backward_compatible_runtime_memory_scope_defaults_to_character(
    character: CharacterProfile,
) -> None:
    from ai_character_engine.runtime.character_runtime import CharacterRuntime

    runtime = CharacterRuntime(character=character, llm=FakeLLMClient())
    assert runtime.memory_scope_id == character.id
