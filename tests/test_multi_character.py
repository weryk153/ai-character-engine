from __future__ import annotations

from ai_character_engine._version import VERSION
import asyncio
from pathlib import Path

import pytest

from ai_character_engine.character.profile import CharacterProfile
from ai_character_engine.long_term_cognition import LongTermCognitionManager
from ai_character_engine.multi_character import (
    CharacterExchange,
    CharacterIsolationError,
    CharacterSchedulerOverloadedError,
    ExchangeStatus,
    FairCharacterScheduler,
    FairSchedulerConfig,
    InMemoryCharacterExchangeStore,
    InMemorySharedCognitionStore,
    JsonlCharacterExchangeStore,
    JsonlSharedCognitionStore,
    KnowledgeVisibility,
    MultiCharacterRuntime,
    SharedCognitionAccessError,
    SharedCognitionRecord,
    UnknownCharacterError,
)
from ai_character_engine.runtime import CharacterRuntime
from ai_character_engine.state import CharacterState
from tests.fakes import FakeLLMClient


def runtime(character_id: str, *, llm=None, scope: str | None = None, state=None, cognition=None):
    return CharacterRuntime(
        character=CharacterProfile(id=character_id, name=character_id.title(), description="test"),
        llm=llm or FakeLLMClient(f"reply-{character_id}"),
        state=state,
        memory_scope_id=scope or f"memory:{character_id}",
        cognition_scope_id=scope or f"cognition:{character_id}",
        goal_scope_id=scope or f"goal:{character_id}",
        long_term_cognition=cognition,
    )


def ensemble(*ids: str, **kwargs) -> MultiCharacterRuntime:
    return MultiCharacterRuntime({cid: runtime(cid) for cid in ids}, **kwargs)


def test_visibility_enum_is_small_and_explicit():
    assert [item.value for item in KnowledgeVisibility] == ["private", "direct", "public"]


def test_direct_shared_record_requires_audience():
    with pytest.raises(ValueError, match="audience"):
        SharedCognitionRecord("a", "observation", "x", KnowledgeVisibility.DIRECT)


def test_private_and_public_shared_record_reject_audience():
    for visibility in (KnowledgeVisibility.PRIVATE, KnowledgeVisibility.PUBLIC):
        with pytest.raises(ValueError, match="audience"):
            SharedCognitionRecord("a", "observation", "x", visibility, ("b",))


def test_shared_record_rejects_owner_inside_direct_audience():
    with pytest.raises(ValueError, match="owner"):
        SharedCognitionRecord("a", "observation", "x", KnowledgeVisibility.DIRECT, ("a",))


def test_shared_record_visibility_rules():
    private = SharedCognitionRecord("a", "note", "secret")
    direct = SharedCognitionRecord("a", "note", "for b", KnowledgeVisibility.DIRECT, ("b",))
    public = SharedCognitionRecord("a", "note", "for all", KnowledgeVisibility.PUBLIC)
    assert private.visible_to("a") is True
    assert private.visible_to("b") is False
    assert direct.visible_to("a") is True
    assert direct.visible_to("b") is True
    assert direct.visible_to("c") is False
    assert public.visible_to("b") is True


def test_shared_store_rejects_duplicate_ids():
    record = SharedCognitionRecord("a", "note", "x", id="same")
    store = InMemorySharedCognitionStore((record,))
    with pytest.raises(ValueError, match="duplicate"):
        store.add(record)


def test_shared_jsonl_roundtrip(tmp_path: Path):
    path = tmp_path / "shared.jsonl"
    store = JsonlSharedCognitionStore(path)
    record = SharedCognitionRecord(
        "a", "belief_summary", "Tea may matter", KnowledgeVisibility.DIRECT, ("b",),
        source_type="belief", source_id="belief-1", metadata={"confidence": 0.8},
    )
    store.add(record)
    loaded = JsonlSharedCognitionStore(path).get(record.id)
    assert loaded == record


def test_exchange_rejects_self_delivery():
    with pytest.raises(ValueError, match="different"):
        CharacterExchange("a", "a", "hello")


def test_exchange_jsonl_persists_status_update(tmp_path: Path):
    path = tmp_path / "exchanges.jsonl"
    store = JsonlCharacterExchangeStore(path)
    exchange = CharacterExchange("a", "b", "hello", id="e1")
    store.add(exchange)
    store.replace(CharacterExchange("a", "b", "hello", id="e1", status=ExchangeStatus.DELIVERED))
    loaded = JsonlCharacterExchangeStore(path).get("e1")
    assert loaded is not None and loaded.status is ExchangeStatus.DELIVERED


def test_multi_character_requires_at_least_one_runtime():
    with pytest.raises(ValueError):
        MultiCharacterRuntime({})


def test_registry_key_must_match_runtime_character_id():
    with pytest.raises(CharacterIsolationError, match="does not match"):
        MultiCharacterRuntime({"wrong": runtime("a")})


def test_different_characters_cannot_share_memory_cognition_goal_scope_by_default():
    a = runtime("a", scope="shared")
    b = runtime("b", scope="shared")
    with pytest.raises(CharacterIsolationError, match="share .* scope"):
        MultiCharacterRuntime({"a": a, "b": b})


def test_scope_isolation_can_be_explicitly_disabled_for_host_managed_namespaces():
    a = runtime("a", scope="shared")
    b = runtime("b", scope="shared")
    multi = MultiCharacterRuntime({"a": a, "b": b}, enforce_scope_isolation=False)
    assert multi.character_ids == ("a", "b")


def test_different_characters_cannot_share_mutable_state_object():
    state = CharacterState()
    with pytest.raises(CharacterIsolationError, match="mutable state"):
        MultiCharacterRuntime({"a": runtime("a", state=state), "b": runtime("b", state=state)})


def test_unknown_character_is_explicit_error():
    multi = ensemble("a")
    with pytest.raises(UnknownCharacterError):
        multi.runtime_for("missing")


@pytest.mark.asyncio
async def test_process_event_updates_only_selected_character_history():
    multi = ensemble("a", "b")
    await multi.run_turn("a", "hello")
    assert [m.content for m in multi.runtime_for("a").history] == ["hello", "reply-a"]
    assert multi.runtime_for("b").history == []


@pytest.mark.asyncio
async def test_send_message_is_explicit_environment_event_not_user_message():
    a_llm = FakeLLMClient("a")
    b_llm = FakeLLMClient("b")
    multi = MultiCharacterRuntime({"a": runtime("a", llm=a_llm), "b": runtime("b", llm=b_llm)})
    result = await multi.send_message(sender_character_id="a", recipient_character_id="b", content="I saw rain")
    assert result.exchange.status is ExchangeStatus.DELIVERED
    assert result.recipient_result.event.type == "character_message"
    assert result.recipient_result.event.source == "character:a"
    assert result.recipient_result.event.payload["authoritative"] is False
    assert result.recipient_result.event.payload["memory_evidence_type"] == "event_observation"
    assert a_llm.calls == []
    assert b_llm.calls[-1][-1].role == "event"


@pytest.mark.asyncio
async def test_send_message_does_not_copy_recipient_history_into_sender():
    multi = ensemble("a", "b")
    await multi.send_message(sender_character_id="a", recipient_character_id="b", content="hello b")
    assert multi.runtime_for("a").history == []
    assert any("hello b" in message.content for message in multi.runtime_for("b").history)


class FailingLLM:
    async def generate(self, messages, *, tools=None):
        raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_failed_exchange_is_audited_without_marking_delivered():
    store = InMemoryCharacterExchangeStore()
    multi = MultiCharacterRuntime({"a": runtime("a"), "b": runtime("b", llm=FailingLLM())}, exchange_store=store)
    with pytest.raises(RuntimeError, match="boom"):
        await multi.send_message(sender_character_id="a", recipient_character_id="b", content="hello")
    exchanges = store.list_for_character("b")
    assert len(exchanges) == 1
    assert exchanges[0].status is ExchangeStatus.FAILED
    assert "RuntimeError" in (exchanges[0].error or "")


def test_publish_rejects_unknown_direct_audience():
    multi = ensemble("a", "b")
    with pytest.raises(UnknownCharacterError):
        multi.publish_shared_cognition(
            owner_character_id="a", kind="note", content="x",
            visibility=KnowledgeVisibility.DIRECT, audience_character_ids=("missing",),
        )


def test_visible_listing_does_not_deliver_or_mutate_recipient_runtime():
    multi = ensemble("a", "b")
    multi.publish_shared_cognition(
        owner_character_id="a", kind="note", content="visible",
        visibility=KnowledgeVisibility.PUBLIC,
    )
    assert [r.content for r in multi.visible_shared_cognition("b")] == ["visible"]
    assert multi.runtime_for("b").history == []


@pytest.mark.asyncio
async def test_private_shared_cognition_cannot_cross_owner_boundary():
    multi = ensemble("a", "b")
    record = multi.publish_shared_cognition(owner_character_id="a", kind="note", content="secret")
    with pytest.raises(SharedCognitionAccessError):
        await multi.deliver_shared_cognition(record.id, recipient_character_id="b")


@pytest.mark.asyncio
async def test_direct_shared_cognition_only_delivers_to_audience():
    multi = ensemble("a", "b", "c")
    record = multi.publish_shared_cognition(
        owner_character_id="a", kind="belief_summary", content="maybe relevant",
        visibility=KnowledgeVisibility.DIRECT, audience_character_ids=("b",),
        source_type="belief", source_id="belief-1",
    )
    result = await multi.deliver_shared_cognition(record.id, recipient_character_id="b")
    assert result.event.type == "shared_cognition"
    assert result.event.payload["authoritative"] is False
    assert result.event.payload["source_type"] == "belief"
    with pytest.raises(SharedCognitionAccessError):
        await multi.deliver_shared_cognition(record.id, recipient_character_id="c")


@pytest.mark.asyncio
async def test_public_shared_cognition_can_be_delivered_to_any_registered_character():
    multi = ensemble("a", "b", "c")
    record = multi.publish_shared_cognition(
        owner_character_id="a", kind="observation", content="door opened",
        visibility=KnowledgeVisibility.PUBLIC,
    )
    b = await multi.deliver_shared_cognition(record.id, recipient_character_id="b")
    c = await multi.deliver_shared_cognition(record.id, recipient_character_id="c")
    assert b.event.payload["shared_cognition_id"] == record.id
    assert c.event.payload["shared_cognition_id"] == record.id


@pytest.mark.asyncio
async def test_shared_belief_summary_does_not_auto_write_recipient_belief_store():
    cognition = LongTermCognitionManager()
    multi = MultiCharacterRuntime({
        "a": runtime("a"),
        "b": runtime("b", cognition=cognition),
    })
    record = multi.publish_shared_cognition(
        owner_character_id="a", kind="belief_summary", content="User may prefer tea",
        visibility=KnowledgeVisibility.DIRECT, audience_character_ids=("b",),
        source_type="belief", source_id="belief-a",
    )
    await multi.deliver_shared_cognition(record.id, recipient_character_id="b")
    assert cognition.beliefs(character_id=multi.runtime_for("b").cognition_scope_id) == ()


@pytest.mark.asyncio
async def test_broadcast_defaults_to_all_other_characters():
    multi = ensemble("a", "b", "c")
    results = await multi.broadcast_message(sender_character_id="a", content="hello everyone")
    assert {item.exchange.recipient_character_id for item in results} == {"b", "c"}
    assert multi.runtime_for("a").history == []


@pytest.mark.asyncio
async def test_broadcast_rejects_duplicate_recipients_and_sender():
    multi = ensemble("a", "b")
    with pytest.raises(ValueError, match="unique"):
        await multi.broadcast_message(sender_character_id="a", content="x", recipient_character_ids=("b", "b"))
    with pytest.raises(ValueError, match="sender"):
        await multi.broadcast_message(sender_character_id="a", content="x", recipient_character_ids=("a",))


def test_scheduler_config_rejects_invalid_limits():
    with pytest.raises(ValueError):
        FairSchedulerConfig(max_concurrent_characters=0)
    with pytest.raises(ValueError):
        FairSchedulerConfig(max_pending_per_character=-1)
    with pytest.raises(ValueError):
        FairSchedulerConfig(max_total_pending=-1)


@pytest.mark.asyncio
async def test_scheduler_allows_different_characters_to_run_in_parallel():
    scheduler = FairCharacterScheduler(FairSchedulerConfig(max_concurrent_characters=2))
    gate = asyncio.Event()
    started = {"a": asyncio.Event(), "b": asyncio.Event()}

    async def work(cid):
        started[cid].set()
        await gate.wait()
        return cid

    ta = asyncio.create_task(scheduler.run("a", lambda: work("a")))
    tb = asyncio.create_task(scheduler.run("b", lambda: work("b")))
    await asyncio.wait_for(asyncio.gather(started["a"].wait(), started["b"].wait()), 1)
    gate.set()
    assert set(await asyncio.gather(ta, tb)) == {"a", "b"}


@pytest.mark.asyncio
async def test_scheduler_serializes_same_character():
    scheduler = FairCharacterScheduler(FairSchedulerConfig(max_concurrent_characters=2))
    gate = asyncio.Event(); first_started = asyncio.Event(); second_started = asyncio.Event()

    async def first():
        first_started.set(); await gate.wait(); return 1

    async def second():
        second_started.set(); return 2

    t1 = asyncio.create_task(scheduler.run("a", first))
    await first_started.wait()
    t2 = asyncio.create_task(scheduler.run("a", second))
    await asyncio.sleep(0.02)
    assert not second_started.is_set()
    gate.set()
    assert await t1 == 1
    assert await t2 == 2


@pytest.mark.asyncio
async def test_scheduler_round_robin_prevents_one_character_from_taking_next_slot():
    scheduler = FairCharacterScheduler(FairSchedulerConfig(max_concurrent_characters=1, max_pending_per_character=5))
    gate = asyncio.Event(); order = []

    async def blocked():
        order.append("a1"); await gate.wait()

    async def mark(name):
        order.append(name)

    a1 = asyncio.create_task(scheduler.run("a", blocked))
    await asyncio.sleep(0)
    a2 = asyncio.create_task(scheduler.run("a", lambda: mark("a2")))
    b1 = asyncio.create_task(scheduler.run("b", lambda: mark("b1")))
    await asyncio.sleep(0)
    gate.set()
    await asyncio.gather(a1, a2, b1)
    assert order == ["a1", "b1", "a2"]


@pytest.mark.asyncio
async def test_scheduler_rejects_pending_over_character_limit():
    scheduler = FairCharacterScheduler(FairSchedulerConfig(max_concurrent_characters=1, max_pending_per_character=1, max_total_pending=5))
    gate = asyncio.Event(); started = asyncio.Event()
    async def blocked(): started.set(); await gate.wait()
    t1 = asyncio.create_task(scheduler.run("a", blocked)); await started.wait()
    t2 = asyncio.create_task(scheduler.run("a", lambda: asyncio.sleep(0)))
    await asyncio.sleep(0)
    with pytest.raises(CharacterSchedulerOverloadedError):
        await scheduler.run("a", lambda: asyncio.sleep(0))
    gate.set(); await asyncio.gather(t1, t2)


@pytest.mark.asyncio
async def test_scheduler_rejects_global_pending_limit():
    scheduler = FairCharacterScheduler(FairSchedulerConfig(max_concurrent_characters=1, max_pending_per_character=4, max_total_pending=1))
    gate = asyncio.Event(); started = asyncio.Event()
    async def blocked(): started.set(); await gate.wait()
    t1 = asyncio.create_task(scheduler.run("a", blocked)); await started.wait()
    t2 = asyncio.create_task(scheduler.run("b", lambda: asyncio.sleep(0)))
    await asyncio.sleep(0)
    with pytest.raises(CharacterSchedulerOverloadedError):
        await scheduler.run("c", lambda: asyncio.sleep(0))
    gate.set(); await asyncio.gather(t1, t2)


@pytest.mark.asyncio
async def test_zero_pending_allows_immediate_capacity_but_rejects_waiting_work():
    scheduler = FairCharacterScheduler(FairSchedulerConfig(max_concurrent_characters=1, max_pending_per_character=0, max_total_pending=0))
    gate = asyncio.Event(); started = asyncio.Event()
    async def blocked(): started.set(); await gate.wait(); return 1
    t1 = asyncio.create_task(scheduler.run("a", blocked)); await started.wait()
    with pytest.raises(CharacterSchedulerOverloadedError):
        await scheduler.run("b", lambda: asyncio.sleep(0))
    gate.set(); assert await t1 == 1


@pytest.mark.asyncio
async def test_cancelled_waiter_is_removed_and_does_not_block_later_work():
    scheduler = FairCharacterScheduler(FairSchedulerConfig(max_concurrent_characters=1, max_pending_per_character=2))
    gate = asyncio.Event(); started = asyncio.Event()
    async def blocked(): started.set(); await gate.wait()
    first = asyncio.create_task(scheduler.run("a", blocked)); await started.wait()
    cancelled = asyncio.create_task(scheduler.run("b", lambda: asyncio.sleep(0)))
    await asyncio.sleep(0); cancelled.cancel()
    with pytest.raises(asyncio.CancelledError): await cancelled
    gate.set(); await first
    assert await scheduler.run("c", lambda: asyncio.sleep(0, result="ok")) == "ok"


@pytest.mark.asyncio
async def test_scheduler_snapshot_reports_running_and_pending_by_character():
    scheduler = FairCharacterScheduler(FairSchedulerConfig(max_concurrent_characters=1, max_pending_per_character=2))
    gate = asyncio.Event(); started = asyncio.Event()
    async def blocked(): started.set(); await gate.wait()
    t1 = asyncio.create_task(scheduler.run("a", blocked)); await started.wait()
    t2 = asyncio.create_task(scheduler.run("b", lambda: asyncio.sleep(0)))
    await asyncio.sleep(0)
    snap = scheduler.snapshot()
    assert snap.running_character_ids == ("a",)
    assert snap.pending_by_character == {"b": 1}
    assert snap.total_pending == 1
    gate.set(); await asyncio.gather(t1, t2)


def test_multi_character_package_does_not_import_cognition_authority_modules_directly():
    root = Path(__file__).resolve().parents[1] / "src" / "ai_character_engine" / "multi_character"
    source = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))
    assert "ai_character_engine.memory" not in source
    assert "ai_character_engine.long_term_cognition" not in source
    assert "ai_character_engine.goals" not in source
    assert "ai_character_engine.commit" not in source


def test_public_exports_and_version():
    import ai_character_engine as ace
    assert ace.__version__ == VERSION
    assert ace.MultiCharacterRuntime is MultiCharacterRuntime
    assert ace.FairCharacterScheduler is FairCharacterScheduler
    assert ace.KnowledgeVisibility.PUBLIC.value == "public"


def test_docs_describe_authority_boundaries():
    root = Path(__file__).resolve().parents[1]
    guide = (root / "docs" / "architecture.md").read_text(encoding="utf-8")
    assert "A second character is another authority boundary" in guide
    assert "does not auto-promote" in guide


def test_offline_example_runs_without_provider():
    import os
    import subprocess
    import sys
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "src")
    proc = subprocess.run(
        [sys.executable, str(root / "tests/scenarios/multi_character_runtime.py")],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert proc.returncode == 0, proc.stderr
    assert "exchange: delivered" in proc.stdout
    assert "shared observation authoritative: False" in proc.stdout
