"""v0.23.1 regressions derived from the v0.23 independent quality review."""
import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from ai_character_engine import CharacterProfile, CharacterRuntime
from ai_character_engine.autonomy import (
    AdmissionStatus, AutonomyController, AutonomyPolicy, AutonomyScheduler,
    DispatchStatus, ProactiveCandidate,
)
from ai_character_engine.host import CharacterHostBridge
from ai_character_engine.llm.models import LLMResponse
from ai_character_engine.memory.manager import MemoryManager
from ai_character_engine.memory.ledger import InMemoryEventLedger
from ai_character_engine.state.models import StatePatch
from ai_character_engine.state.policy import NoopStatePolicy

NOW = datetime(2026, 9, 21, 12, tzinfo=UTC)


def make_scheduler(policy=None, **kwargs):
    kwargs.setdefault('clock', lambda: NOW)
    return AutonomyScheduler(policy, **kwargs)


def candidate(content="observation", **kwargs):
    return ProactiveCandidate(content=content, source=kwargs.pop("source", "review"), created_at=kwargs.pop("created_at", NOW), **kwargs)


class Gate:
    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0

    async def process_event(self, event):
        self.calls += 1
        self.started.set()
        await self.release.wait()
        return object()

    async def generate(self, messages, *, tools=None):
        await self.process_event(None)
        return LLMResponse(text="reply")


class GoodLLM:
    async def generate(self, messages, *, tools=None):
        return LLMResponse(text="reply")


def runtime(llm=None, **kwargs):
    return CharacterRuntime(character=CharacterProfile("review", "Review", "Careful"), llm=llm or GoodLLM(), **kwargs)


@pytest.mark.parametrize("mode", ["dedupe", "queue_pressure"])
async def test_inflight_candidate_must_not_be_evicted(mode):
    gate = Gate()
    scheduler = make_scheduler(AutonomyPolicy(max_pending=1))
    key = "same" if mode == "dedupe" else None
    scheduler.submit(candidate(priority=10, dedupe_key=key))
    controller = AutonomyController(gate, scheduler)
    task = asyncio.create_task(controller.run_once(now=NOW))
    await gate.started.wait()
    scheduler.submit(candidate("new", priority=90, dedupe_key=key))
    gate.release.set()
    result = (await asyncio.gather(task, return_exceptions=True))[0]
    assert not isinstance(result, Exception), repr(result)


async def test_shared_scheduler_must_not_dispatch_same_candidate_twice():
    gate = Gate()
    scheduler = make_scheduler()
    scheduler.submit(candidate())
    a, b = AutonomyController(gate, scheduler), AutonomyController(gate, scheduler)
    first = asyncio.create_task(a.run_once(now=NOW))
    await gate.started.wait()
    second = asyncio.create_task(b.run_once(now=NOW))
    await asyncio.sleep(0)  # Yield the event loop; no wall-clock delay.
    gate.release.set()
    results = await asyncio.gather(first, second, return_exceptions=True)
    assert gate.calls == 1, f"runtime calls={gate.calls}, results={results!r}"


async def test_bridge_and_autonomy_must_share_runtime_turn_guard():
    gate = Gate()
    engine = runtime(gate)
    bridge = CharacterHostBridge(engine)
    scheduler = make_scheduler()
    scheduler.submit(candidate())
    proactive = asyncio.create_task(AutonomyController(engine, scheduler).run_once(now=NOW))
    await gate.started.wait()
    normal = asyncio.create_task(bridge.process("normal user turn"))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    concurrent_calls = gate.calls
    gate.release.set()
    await asyncio.gather(proactive, normal, return_exceptions=True)
    assert concurrent_calls == 1, f"simultaneous LLM calls={concurrent_calls}"


class TrustPolicy(NoopStatePolicy):
    def on_event(self, event, state):
        return StatePatch(trust_delta=5)


async def test_failed_retry_must_not_apply_state_transition_twice():
    class FailOnce:
        calls = 0

        async def generate(self, messages, *, tools=None):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("synthetic failure")
            return LLMResponse(text="reply")

    engine = runtime(FailOnce(), state_policy=TrustPolicy())
    scheduler = make_scheduler()
    scheduler.submit(candidate())
    controller = AutonomyController(engine, scheduler)
    assert (await controller.run_once(now=NOW)).status == DispatchStatus.FAILED
    assert (await controller.run_once(now=NOW + timedelta(seconds=1))).status == DispatchStatus.DELIVERED
    assert engine.state.trust == 55, f"trust={engine.state.trust}, expected one +5 transition"


async def test_cancellation_must_restore_local_state():
    gate = Gate()
    engine = runtime(gate, state_policy=TrustPolicy())
    scheduler = make_scheduler()
    scheduler.submit(candidate())
    task = asyncio.create_task(AutonomyController(engine, scheduler).run_once(now=NOW))
    await gate.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert engine.state.trust == 50, f"trust after cancellation={engine.state.trust}"


async def test_memory_failure_retry_must_not_duplicate_history():
    class FailOnceLedger(InMemoryEventLedger):
        calls = 0

        def append(self, entry):
            self.calls += 1
            if self.calls == 1:
                raise OSError("synthetic storage failure")
            super().append(entry)

    engine = runtime(memory_manager=MemoryManager(ledger=FailOnceLedger()))
    scheduler = make_scheduler()
    scheduler.submit(candidate())
    controller = AutonomyController(engine, scheduler)
    failed = await controller.run_once(now=NOW)
    assert failed.status == DispatchStatus.FAILED and failed.requires_review
    assert engine.history == []
    assert (await controller.run_once(now=NOW)).status == DispatchStatus.EMPTY
    scheduler.resume(failed.candidate.id)  # Host reconciled the deliberately failing test ledger.
    assert (await controller.run_once(now=NOW)).status == DispatchStatus.DELIVERED
    assert len(engine.history) == 2, f"history messages={len(engine.history)}"


def test_expired_entry_must_not_reject_a_fresh_submission():
    scheduler = make_scheduler(AutonomyPolicy(max_pending=1), clock=lambda: NOW)
    scheduler.submit(candidate(priority=100, created_at=NOW - timedelta(seconds=2), expires_at=NOW - timedelta(seconds=1)))
    result = scheduler.submit(candidate(priority=50))
    assert result.status == AdmissionStatus.ACCEPTED, result


def test_equal_priority_fifo_means_submission_order():
    scheduler = make_scheduler()
    first = candidate("first submitted", created_at=NOW)
    second = candidate("second submitted", created_at=NOW - timedelta(seconds=10))
    scheduler.submit(first)
    scheduler.submit(second)
    assert scheduler.next_ready(NOW) is first


async def test_explicit_now_must_not_skip_cooldown_after_slow_turn():
    clock = [NOW]

    class SlowRuntime:
        async def process_event(self, event):
            clock[0] += timedelta(seconds=60)
            return object()

    scheduler = make_scheduler(clock=lambda: clock[0])
    scheduler.submit(candidate())
    await AutonomyController(SlowRuntime(), scheduler).run_once(now=NOW)
    scheduler.submit(candidate("next", created_at=clock[0]))
    assert scheduler.next_ready(clock[0]) is None


def test_naive_mark_dispatched_must_reject_before_mutation():
    scheduler = make_scheduler()
    item = candidate()
    scheduler.submit(item)
    with pytest.raises(ValueError, match="timezone-aware"):
        scheduler.mark_dispatched(item, NOW.replace(tzinfo=None))
    assert scheduler.pending == (item,)


@pytest.mark.parametrize("priority", [True, 12.5])
def test_priority_must_be_integer_not_bool(priority):
    with pytest.raises((ValueError, TypeError)):
        candidate(priority=priority)


def test_canonical_event_identity_is_preserved():
    scheduler = make_scheduler()
    item = candidate(not_before=NOW + timedelta(minutes=1))
    scheduler.submit(item)
    assert scheduler.next_ready(NOW) is None
    assert scheduler.next_ready(NOW + timedelta(minutes=1)) is item


async def test_memory_default_skips_retrievable_store_but_keeps_ledger_history():
    manager = MemoryManager()
    engine = runtime(memory_manager=manager)
    scheduler = make_scheduler()
    scheduler.submit(candidate())
    result = await AutonomyController(engine, scheduler).run_once(now=NOW)
    assert result.status == DispatchStatus.DELIVERED
    assert manager.store.list_for_character("review") == []
    assert len(manager.ledger.list_for_character("review")) == 1
    assert len(engine.history) == 2


async def test_existing_single_controller_overlap_guard_works():
    gate = Gate()
    scheduler = make_scheduler()
    scheduler.submit(candidate())
    controller = AutonomyController(gate, scheduler)
    task = asyncio.create_task(controller.run_once(now=NOW))
    await gate.started.wait()
    result = await controller.run_once(now=NOW)
    gate.release.set()
    await task
    assert result.status == DispatchStatus.BLOCKED
