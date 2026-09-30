import asyncio
from datetime import UTC, datetime, time, timedelta, timezone

import pytest

from ai_character_engine.autonomy import (
    AdmissionStatus,
    AutonomyController,
    AutonomyPolicy,
    AutonomyScheduler,
    DispatchStatus,
    ProactiveCandidate,
    QuietHours,
)
from ai_character_engine.llm.models import LLMResponse
from ai_character_engine.runtime.models import CharacterRunResult


NOW = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)


def make_scheduler(policy=None):
    return AutonomyScheduler(policy, clock=lambda: NOW)


def candidate(content="notice", **kwargs):
    return ProactiveCandidate(
        content=content,
        source="test_host",
        created_at=kwargs.pop("created_at", NOW),
        **kwargs,
    )


def test_candidate_validation():
    with pytest.raises(ValueError, match="content"):
        candidate(" ")
    with pytest.raises(ValueError, match="priority"):
        candidate(priority=101)
    with pytest.raises(ValueError, match="timezone-aware"):
        candidate(created_at=NOW.replace(tzinfo=None))
    with pytest.raises(ValueError, match="later"):
        candidate(expires_at=NOW)


def test_policy_validation():
    with pytest.raises(ValueError, match="cooldowns"):
        AutonomyPolicy(global_cooldown=timedelta(seconds=-1))
    with pytest.raises(ValueError, match="max_pending"):
        AutonomyPolicy(max_pending=0)
    with pytest.raises(ValueError, match="allowed_event_types"):
        AutonomyPolicy(allowed_event_types=frozenset())


def test_admission_rejects_low_priority_and_unlisted_type():
    scheduler = make_scheduler(AutonomyPolicy(min_priority=20))
    assert scheduler.submit(candidate(priority=19)).reason == "below_min_priority"
    result = scheduler.submit(candidate(priority=50, event_type="host_secret"))
    assert result.status is AdmissionStatus.REJECTED
    assert result.reason == "event_type_not_allowed"


def test_pending_duplicate_is_rejected_or_replaced_by_higher_priority():
    scheduler = make_scheduler()
    first = candidate("first", priority=20, dedupe_key="weather")
    assert scheduler.submit(first).status is AdmissionStatus.ACCEPTED
    assert scheduler.submit(candidate("same", priority=20, dedupe_key="weather")).reason == "duplicate_pending"
    better = candidate("urgent", priority=80, dedupe_key="weather")
    result = scheduler.submit(better)
    assert result.status is AdmissionStatus.REPLACED
    assert result.replaced_candidate_id == first.id
    assert scheduler.pending == (better,)


def test_queue_pressure_keeps_more_important_candidate():
    scheduler = make_scheduler(AutonomyPolicy(max_pending=2))
    low = candidate("low", priority=10)
    medium = candidate("medium", priority=20)
    scheduler.submit(low)
    scheduler.submit(medium)
    assert scheduler.submit(candidate("also low", priority=10)).reason == "queue_full"
    high = candidate("high", priority=90)
    result = scheduler.submit(high)
    assert result.status is AdmissionStatus.REPLACED
    assert low not in scheduler.pending
    assert scheduler.pending == (medium, high)


def test_priority_then_fifo_dispatch_order():
    policy = AutonomyPolicy(global_cooldown=timedelta(0))
    scheduler = make_scheduler(policy)
    older = candidate("older", priority=50)
    newer = candidate("newer", priority=50, created_at=NOW + timedelta(seconds=1))
    urgent = candidate("urgent", priority=90, created_at=NOW + timedelta(seconds=2))
    for item in (older, newer, urgent):
        scheduler.submit(item)
    assert scheduler.next_ready(NOW + timedelta(seconds=3)) is urgent
    scheduler.mark_dispatched(urgent, NOW + timedelta(seconds=3))
    assert scheduler.next_ready(NOW + timedelta(seconds=3)) is older


def test_not_before_and_expiry_are_enforced():
    scheduler = make_scheduler()
    future = candidate(not_before=NOW + timedelta(minutes=1))
    expired = candidate(expires_at=NOW + timedelta(seconds=1))
    scheduler.submit(future)
    scheduler.submit(expired)
    assert scheduler.next_ready(NOW + timedelta(seconds=2)) is None
    assert expired not in scheduler.pending
    assert scheduler.next_ready(NOW + timedelta(minutes=1)) is future


def test_global_and_key_cooldowns():
    policy = AutonomyPolicy(
        global_cooldown=timedelta(seconds=10),
        per_key_cooldown=timedelta(minutes=1),
    )
    scheduler = make_scheduler(policy)
    first = candidate(cooldown_key="ambient")
    scheduler.submit(first)
    scheduler.mark_dispatched(first, NOW)
    same = candidate("same", cooldown_key="ambient", created_at=NOW + timedelta(seconds=1))
    other = candidate("other", cooldown_key="system", created_at=NOW + timedelta(seconds=1))
    scheduler.submit(same)
    scheduler.submit(other)
    assert scheduler.next_ready(NOW + timedelta(seconds=9)) is None
    assert scheduler.next_ready(NOW + timedelta(seconds=10)) is other
    scheduler.mark_dispatched(other, NOW + timedelta(seconds=10))
    assert scheduler.next_ready(NOW + timedelta(seconds=60)) is same


@pytest.mark.parametrize(
    ("hour", "quiet"), [(22, True), (23, True), (0, True), (6, True), (7, False), (12, False)]
)
def test_overnight_quiet_hours(hour, quiet):
    window = QuietHours(time(22), time(7), timezone.utc)
    assert window.contains(NOW.replace(hour=hour)) is quiet


def test_quiet_hours_use_configured_timezone():
    taipei = timezone(timedelta(hours=8))
    scheduler = make_scheduler(
        AutonomyPolicy(
            quiet_hours=QuietHours(time(22), time(7), taipei),
            global_cooldown=timedelta(0),
        )
    )
    item = candidate()
    scheduler.submit(item)
    assert scheduler.next_ready(NOW.replace(hour=15)) is None  # 23:00 Taipei
    assert scheduler.next_ready(NOW.replace(hour=1)) is item  # 09:00 Taipei


def test_equal_quiet_hour_endpoints_block_all_day():
    assert QuietHours(time(0), time(0)).contains(NOW)


def test_naive_dispatch_time_is_rejected():
    scheduler = make_scheduler()
    scheduler.submit(candidate())
    with pytest.raises(ValueError, match="timezone-aware"):
        scheduler.next_ready(NOW.replace(tzinfo=None))


class FakeRuntime:
    def __init__(self):
        self.events = []
        self.fail = False
        self.wait = False
        self.started = asyncio.Event()

    async def process_event(self, event):
        self.events.append(event)
        self.started.set()
        if self.wait:
            await asyncio.Event().wait()
        if self.fail:
            raise RuntimeError("provider secret")
        return CharacterRunResult(event=event, response=LLMResponse(text="noticed"))


async def test_controller_delivers_one_bounded_event():
    runtime = FakeRuntime()
    scheduler = make_scheduler(AutonomyPolicy(global_cooldown=timedelta(0)))
    item = candidate(priority=75, dedupe_key="scene", payload={"scene": "desk"})
    scheduler.submit(item)
    result = await AutonomyController(runtime, scheduler).run_once(now=NOW)
    assert result.status is DispatchStatus.DELIVERED
    assert result.run_result.text == "noticed"
    assert scheduler.pending == ()
    event = runtime.events[0]
    assert event.type == "proactive_observation"
    assert event.payload["scene"] == "desk"
    assert event.payload["memory_importance"] == 0.0
    assert event.payload["autonomy"]["candidate_id"] == item.id


async def test_explicit_memory_importance_is_preserved():
    runtime = FakeRuntime()
    scheduler = make_scheduler(AutonomyPolicy(global_cooldown=timedelta(0)))
    scheduler.submit(candidate(payload={"memory_importance": 0.8}))
    await AutonomyController(runtime, scheduler).run_once(now=NOW)
    assert runtime.events[0].payload["memory_importance"] == 0.8


async def test_controller_failure_is_sanitized_and_retriable():
    runtime = FakeRuntime()
    runtime.fail = True
    scheduler = make_scheduler()
    item = candidate()
    scheduler.submit(item)
    controller = AutonomyController(runtime, scheduler)
    failed = await controller.run_once(now=NOW)
    assert failed.status is DispatchStatus.FAILED
    assert failed.reason == "runtime_failed"
    assert scheduler.pending == (item,)
    runtime.fail = False
    delivered = await controller.run_once(now=NOW + timedelta(seconds=1))
    assert delivered.status is DispatchStatus.DELIVERED


async def test_controller_rejects_overlap_without_consuming_queue():
    runtime = FakeRuntime()
    runtime.wait = True
    scheduler = make_scheduler(AutonomyPolicy(global_cooldown=timedelta(0)))
    first, second = candidate("first", priority=80), candidate("second", priority=70)
    scheduler.submit(first)
    scheduler.submit(second)
    controller = AutonomyController(runtime, scheduler)
    task = asyncio.create_task(controller.run_once(now=NOW))
    await runtime.started.wait()
    blocked = await controller.run_once(now=NOW)
    assert blocked.status is DispatchStatus.BLOCKED
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert scheduler.pending == (first, second)


async def test_empty_controller_tick_is_explicit():
    result = await AutonomyController(FakeRuntime(), make_scheduler()).run_once(now=NOW)
    assert result.status is DispatchStatus.EMPTY
