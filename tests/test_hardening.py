import asyncio
from datetime import UTC, datetime, time, timedelta, timezone

import pytest

from ai_character_engine.autonomy import AutonomyController, AutonomyPolicy, AutonomyScheduler, ProactiveCandidate, QuietHours
from ai_character_engine.events import CharacterEvent
from ai_character_engine.host import CharacterHostBridge, HostBridgeError
from ai_character_engine.llm.models import LLMResponse
from ai_character_engine.memory.manager import MemoryManager
from ai_character_engine.memory.ledger import InMemoryEventLedger
from ai_character_engine.runtime import RuntimeBusyError
from ai_character_engine.tools.models import ToolCall, ToolDefinition

from .test_regressions import NOW, Gate, TrustPolicy, candidate, make_scheduler, runtime


@pytest.mark.parametrize("kwargs", [
    {"content": "x" * 8193}, {"source": "x" * 257}, {"id": ""},
    {"dedupe_key": ""}, {"cooldown_key": "x" * 257},
    {"payload": {"raw": b"image"}}, {"payload": {"score": float("nan")}},
    {"payload": {1: "bad key"}}, {"payload": {"big": "漢" * 6000}},
    {"payload": {"nested": list(range(4097))}},
    {"payload": {"memory_importance": True}}, {"payload": {"memory_importance": 2}},
    {"not_before": NOW + timedelta(seconds=2), "expires_at": NOW + timedelta(seconds=1)},
])
def test_candidate_bounds(kwargs):
    with pytest.raises(ValueError):
        candidate(**kwargs)


@pytest.mark.parametrize("kwargs", [
    {"max_pending": True}, {"max_attempts": 0}, {"max_cooldown_keys": 0},
    {"min_priority": 1.5}, {"retry_backoff": timedelta(seconds=-1)},
    {"max_retry_backoff": timedelta(0)}, {"allowed_event_types": "event"},
])
def test_policy_bounds(kwargs):
    with pytest.raises(ValueError):
        AutonomyPolicy(**kwargs)


def test_payload_is_a_deeply_immutable_snapshot():
    original = {"scene": {"objects": ["cup"]}}
    item = candidate(payload=original)
    original["scene"]["objects"].append("secret")
    assert item.payload["scene"]["objects"] == ("cup",)
    with pytest.raises(TypeError):
        item.payload["scene"]["objects"] = ()
    event_copy = item.event_payload()
    event_copy["scene"]["objects"].append("new")
    assert item.event_payload()["scene"]["objects"] == ["cup"]


def test_local_time_validation():
    with pytest.raises(ValueError):
        QuietHours(time(22, tzinfo=UTC), time(7))
    with pytest.raises(ValueError):
        QuietHours(time(22), time(7)).contains(NOW.replace(tzinfo=None))


@pytest.mark.parametrize("hour,minute,quiet", [(13, 59, False), (14, 0, True), (22, 59, True), (23, 0, False)])
def test_taipei_boundaries(hour, minute, quiet):
    window = QuietHours(time(22), time(7), timezone(timedelta(hours=8)))
    assert window.contains(NOW.replace(hour=hour, minute=minute)) is quiet


def test_inflight_is_not_expired_or_discarded_and_claim_is_identity_checked():
    clock = [NOW]
    s = AutonomyScheduler(AutonomyPolicy(max_pending=1), clock=lambda: clock[0])
    item = candidate(expires_at=NOW + timedelta(seconds=1), dedupe_key="x")
    s.submit(item)
    claim = s.claim_next()
    clock[0] += timedelta(seconds=2)
    assert s.submit(candidate("new", priority=99)).reason == "queue_full"
    assert s.in_flight is item and s.pending == ()
    with pytest.raises(ValueError):
        s.discard(item)
    with pytest.raises(ValueError):
        make_scheduler().ack(claim)
    with pytest.raises(ValueError):
        s.ack(claim, now=NOW.replace(tzinfo=None))
    assert s.in_flight is item
    s.ack(claim)
    with pytest.raises(ValueError):
        s.ack(claim)
    assert s.in_flight is None


def test_hold_is_not_replaced_expired_or_lost_under_pressure():
    s = make_scheduler(AutonomyPolicy(max_pending=1))
    item = candidate(dedupe_key="same", expires_at=NOW + timedelta(seconds=1))
    s.submit(item)
    claim = s.claim_next()
    s.release(claim, requires_review=True)
    assert s.submit(candidate("new", dedupe_key="same", priority=99)).reason == "duplicate_held"
    assert s.submit(candidate("other", priority=99)).reason == "queue_full"
    assert s.next_ready(NOW + timedelta(days=1)) is None
    assert s.held == (item,)
    s.discard(item)
    assert s.held == ()


async def test_retry_backoff_cap_and_attempt_limit_allow_other_candidates():
    class Bad:
        async def process_event(self, event):
            raise RuntimeError("secret")

    clock = [NOW]
    s = AutonomyScheduler(AutonomyPolicy(global_cooldown=timedelta(0), max_retry_backoff=timedelta(seconds=2)), clock=lambda: clock[0])
    item, other = candidate(priority=90), candidate("other", priority=10)
    s.submit(item)
    c = AutonomyController(Bad(), s)
    assert (await c.run_once()).reason == "runtime_failed"
    s.submit(other)
    assert s.next_ready() is other
    s.discard(other)
    assert (await c.run_once()).status == "empty"
    clock[0] += timedelta(seconds=1)
    assert (await c.run_once()).status == "failed"
    assert s.retry_state(item.id).retry_at == NOW + timedelta(seconds=3)
    clock[0] += timedelta(seconds=2)
    result = await c.run_once()
    assert result.reason == "attempts_exhausted" and result.requires_review
    assert s.retry_state(item.id).attempts == 3
    assert (await c.run_once()).status == "empty"
    s.resume(item.id)
    assert s.retry_state(item.id).attempts == 0


def test_cooldown_keys_prune_and_capacity_never_weakens_existing_cooldown():
    clock = [NOW]
    s = AutonomyScheduler(AutonomyPolicy(global_cooldown=timedelta(0), max_cooldown_keys=1, per_key_cooldown=timedelta(seconds=5)), clock=lambda: clock[0])
    first = candidate(cooldown_key="one")
    s.submit(first)
    s.ack(s.claim_next())
    second = candidate("other", cooldown_key="two")
    s.submit(second)
    assert s.next_ready() is None  # Do not evict an unexpired key to make room.
    clock[0] += timedelta(seconds=5)
    assert s.next_ready() is second
    s.ack(s.claim_next())
    for i in range(100):
        clock[0] += timedelta(seconds=5)
        s.submit(candidate(str(i), cooldown_key=str(i)))
        s.ack(s.claim_next())
        assert len(s._last_by_key) == 1


async def test_runtime_direct_entry_shares_guard_and_rejects_resets():
    gate = Gate()
    engine = runtime(gate)
    task = asyncio.create_task(engine.run_turn("first"))
    await gate.started.wait()
    with pytest.raises(RuntimeBusyError):
        await engine.run_turn("second")
    with pytest.raises(RuntimeBusyError):
        engine.reset_history()
    with pytest.raises(RuntimeBusyError):
        engine.reset_state()
    gate.release.set()
    await task
    assert not engine.turns.busy


async def test_bridge_first_blocks_autonomy_and_other_bridge():
    gate = Gate()
    engine = runtime(gate, memory_manager=MemoryManager())
    original_manager = engine.memory_manager
    bridge = CharacterHostBridge(engine)
    task = asyncio.create_task(bridge.process("user", skip_memory=True))
    await gate.started.wait()
    s = make_scheduler()
    s.submit(candidate())
    assert (await AutonomyController(engine, s).run_once()).reason == "runtime_busy"
    with pytest.raises(HostBridgeError):
        await CharacterHostBridge(engine).process("other")
    with pytest.raises(HostBridgeError):
        bridge.restore_history([])
    gate.release.set()
    await task
    assert engine.memory_manager is original_manager and engine.history == []


async def test_event_id_is_stable_across_safe_retry():
    events = []

    class Flaky:
        async def process_event(self, event):
            events.append(event)
            if len(events) == 1:
                raise RuntimeError("fail")
            return object()

    s = make_scheduler(AutonomyPolicy(retry_backoff=timedelta(0)))
    item = candidate()
    s.submit(item)
    c = AutonomyController(Flaky(), s)
    await c.run_once()
    await c.run_once()
    assert [e.id for e in events] == [item.id, item.id]


def effect_runtime(mode):
    started = asyncio.Event()
    effects = []

    class Model:
        calls = 0

        async def generate(self, messages, *, tools=None):
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(tool_calls=(ToolCall("one", "counter", {}),))
            started.set()
            if mode == "failure":
                raise RuntimeError("synthetic failure after tool")
            await asyncio.Event().wait()

    engine = runtime(Model(), state_policy=TrustPolicy())
    engine.tool_registry.register(ToolDefinition("counter", "Offline counter", {"type": "object", "properties": {}}), lambda: effects.append("executed") or "ok")
    return engine, started, effects


@pytest.mark.parametrize("mode", ["failure", "cancel", "timeout"])
async def test_external_effect_failure_never_retries_without_host_review(mode):
    engine, started, effects = effect_runtime(mode)
    s = make_scheduler()
    item = candidate()
    s.submit(item)
    controller = AutonomyController(engine, s, turn_timeout_seconds=0.02 if mode == "timeout" else 10)
    task = asyncio.create_task(controller.run_once())
    await started.wait()
    if mode == "cancel":
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        result = await task
        assert result.requires_review and result.reason == "partial_execution"
    assert effects == ["executed"]
    assert engine.history == [] and engine.state.trust == 50
    assert s.held == (item,)
    assert (await controller.run_once(now=NOW + timedelta(days=1))).status == "empty"
    assert effects == ["executed"]


async def test_ledger_append_then_error_is_held_without_duplicate_append():
    class AmbiguousLedger(InMemoryEventLedger):
        def append(self, entry):
            super().append(entry)
            raise OSError("write acknowledgement failed")

    manager = MemoryManager(ledger=AmbiguousLedger())
    engine = runtime(memory_manager=manager)
    s = make_scheduler()
    s.submit(candidate())
    c = AutonomyController(engine, s)
    assert (await c.run_once()).requires_review
    assert (await c.run_once()).status == "empty"
    assert engine.history == []
    assert len(manager.ledger.list_for_character("review")) == 1


async def test_timeout_before_effects_restores_state_and_releases_guard():
    gate = Gate()
    engine = runtime(gate, state_policy=TrustPolicy())
    s = make_scheduler()
    item = candidate()
    s.submit(item)
    result = await AutonomyController(engine, s, turn_timeout_seconds=0.01).run_once()
    assert result.reason == "runtime_timeout" and not result.requires_review
    assert engine.state.trust == 50 and not engine.turns.busy
    assert s.pending == (item,) and not s.busy


@pytest.mark.parametrize("value", [True, 0, -1, float("inf"), float("nan")])
def test_controller_timeout_validation(value):
    with pytest.raises(ValueError):
        AutonomyController(runtime(), make_scheduler(), turn_timeout_seconds=value)


@pytest.mark.parametrize("mode", ["failure", "cancel", "timeout"])
async def test_bridge_preserves_partial_execution_warning(mode):
    from ai_character_engine.host import HostBridgeConfig

    engine, started, effects = effect_runtime(mode)
    bridge = CharacterHostBridge(engine, config=HostBridgeConfig(turn_timeout_seconds=0.02 if mode == "timeout" else 10))
    task = asyncio.create_task(bridge.process("normal turn"))
    await started.wait()
    if mode == "cancel":
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
    else:
        with pytest.raises(HostBridgeError) as caught:
            await task
    assert caught.value.requires_review
    assert effects == ["executed"] and not bridge.busy


def test_submit_prunes_entries_that_expired_while_waiting():
    clock = [NOW]
    s = AutonomyScheduler(AutonomyPolicy(max_pending=1), clock=lambda: clock[0])
    old = candidate(priority=99, dedupe_key="same", expires_at=NOW + timedelta(seconds=1))
    assert s.submit(old).status == "accepted"
    clock[0] += timedelta(seconds=2)
    fresh = candidate("fresh", priority=1, dedupe_key="same", created_at=clock[0])
    assert s.submit(fresh).status == "accepted"
    assert s.pending == (fresh,)
