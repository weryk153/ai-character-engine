from __future__ import annotations

from ai_character_engine._version import VERSION
import asyncio

import pytest

from ai_character_engine.character.profile import CharacterProfile
from ai_character_engine.events.models import CharacterEvent
from ai_character_engine.llm.models import LLMResponse
from ai_character_engine.memory import MemoryManager
from ai_character_engine.runtime import CharacterRuntime
from ai_character_engine.state.models import CharacterState
from ai_character_engine.tasks import (
    MultiTaskRuntime,
    MultiTaskRuntimeConfig,
    TaskOutput,
    TaskPriority,
    TaskQueueFullError,
    TaskRuntimeClosedError,
    TaskStatus,
    UnknownTaskError,
    UnknownTaskTypeError,
)
from tests.fakes import FakeLLMClient


def make_runtime(*, llm=None, memory_manager=None, state=None) -> CharacterRuntime:
    return CharacterRuntime(
        character=CharacterProfile(id="c", name="燈", description="test"),
        llm=llm or FakeLLMClient("foreground reply"),
        memory_manager=memory_manager,
        state=state,
    )


@pytest.mark.asyncio
async def test_background_task_runs_and_reports_lifecycle():
    engine = MultiTaskRuntime(make_runtime(), config=MultiTaskRuntimeConfig(worker_count=1))
    engine.register("summary", lambda ctx: {"seen": ctx.request.payload["text"]})
    async with engine:
        handle = await engine.submit_background("summary", {"text": "hello"})
        result = await handle.wait()
    assert result.status is TaskStatus.SUCCEEDED
    assert result.output is not None
    assert result.output.value == {"seen": "hello"}
    assert [event.status for event in engine.lifecycle_events(task_id=handle.task_id)] == [
        TaskStatus.QUEUED,
        TaskStatus.RUNNING,
        TaskStatus.SUCCEEDED,
    ]


@pytest.mark.asyncio
async def test_foreground_turn_bypasses_running_background_task():
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_background(ctx):
        started.set()
        await release.wait()
        return "done"

    engine = MultiTaskRuntime(make_runtime(), config=MultiTaskRuntimeConfig(worker_count=1))
    engine.register("slow", slow_background)
    async with engine:
        handle = await engine.submit_background("slow")
        await started.wait()
        foreground = await asyncio.wait_for(engine.run_turn("hello"), timeout=0.25)
        assert foreground.text == "foreground reply"
        assert handle.status is TaskStatus.RUNNING
        release.set()
        assert (await handle.wait()).status is TaskStatus.SUCCEEDED


class GateLLM:
    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def generate(self, messages, *, tools=None):
        self.started.set()
        await self.release.wait()
        return LLMResponse(text="ok")


@pytest.mark.asyncio
async def test_new_background_work_waits_while_foreground_is_active():
    llm = GateLLM()
    ran = asyncio.Event()
    engine = MultiTaskRuntime(make_runtime(llm=llm), config=MultiTaskRuntimeConfig(worker_count=1))
    engine.register("bg", lambda ctx: ran.set())
    async with engine:
        foreground = asyncio.create_task(engine.run_turn("hello"))
        await llm.started.wait()
        handle = await engine.submit_background("bg")
        await asyncio.sleep(0)
        assert handle.status is TaskStatus.QUEUED
        assert not ran.is_set()
        llm.release.set()
        await foreground
        assert (await handle.wait()).status is TaskStatus.SUCCEEDED
        assert ran.is_set()


@pytest.mark.asyncio
async def test_priority_queue_selects_high_before_low_after_foreground_release():
    llm = GateLLM()
    order: list[str] = []
    engine = MultiTaskRuntime(make_runtime(llm=llm), config=MultiTaskRuntimeConfig(worker_count=1))
    engine.register("record", lambda ctx: order.append(ctx.request.payload["name"]))
    async with engine:
        foreground = asyncio.create_task(engine.run_turn("hello"))
        await llm.started.wait()
        low = await engine.submit_background("record", {"name": "low"}, priority=TaskPriority.LOW)
        high = await engine.submit_background("record", {"name": "high"}, priority=TaskPriority.HIGH)
        llm.release.set()
        await foreground
        await asyncio.gather(low.wait(), high.wait())
    assert order == ["high", "low"]


@pytest.mark.asyncio
async def test_background_snapshot_is_submission_time_and_cannot_mutate_runtime_state():
    state = CharacterState(custom={"nested": {"value": 1}})
    seen = {}

    async def inspect(ctx):
        seen["revision"] = ctx.snapshot.revision
        seen["trust"] = ctx.snapshot.state.trust
        seen["history"] = tuple(message.content for message in ctx.snapshot.history)
        # Nested values are copied. Mutating this copy must not mutate CharacterRuntime.
        ctx.snapshot.state.custom["nested"]["value"] = 999
        return "ok"

    runtime = make_runtime(state=state)
    engine = MultiTaskRuntime(runtime, config=MultiTaskRuntimeConfig(worker_count=1))
    engine.register("inspect", inspect)
    async with engine:
        await engine.run_turn("first")
        handle = await engine.submit_background("inspect")
        await engine.run_turn("second")
        result = await handle.wait()
    assert result.snapshot_revision == 1
    assert seen["revision"] == 1
    assert seen["history"][-2:] == ("first", "foreground reply")
    assert runtime.state.custom["nested"]["value"] == 1
    assert engine.revision == 2


@pytest.mark.asyncio
async def test_background_proposals_are_non_authoritative_and_not_committed():
    memory = MemoryManager()
    runtime = make_runtime(memory_manager=memory)

    def worker(ctx):
        return TaskOutput(
            value="analysis",
            proposals=(
                ctx.proposal("state", {"trust_delta": 50}),
                ctx.proposal("memory", {"summary": "invented background memory"}),
            ),
        )

    engine = MultiTaskRuntime(runtime)
    engine.register("analysis", worker)
    async with engine:
        handle = await engine.submit_background("analysis")
        result = await handle.wait()
    assert result.status is TaskStatus.SUCCEEDED
    assert result.output is not None and len(result.output.proposals) == 2
    assert all(p.base_revision == 0 for p in result.output.proposals)
    assert runtime.state.trust == 50
    assert memory.store.list_for_character("c") == []
    assert runtime.history == []


@pytest.mark.asyncio
async def test_timeout_is_terminal_result_not_uncaught_exception():
    async def never(ctx):
        await asyncio.Event().wait()

    engine = MultiTaskRuntime(
        make_runtime(),
        config=MultiTaskRuntimeConfig(worker_count=1, default_timeout_s=0.02),
    )
    engine.register("never", never)
    async with engine:
        handle = await engine.submit_background("never")
        result = await handle.wait()
    assert result.status is TaskStatus.TIMED_OUT
    assert "timed out" in (result.error or "")


@pytest.mark.asyncio
async def test_running_task_can_be_cancelled():
    started = asyncio.Event()

    async def block(ctx):
        started.set()
        await asyncio.Event().wait()

    engine = MultiTaskRuntime(make_runtime(), config=MultiTaskRuntimeConfig(worker_count=1))
    engine.register("block", block)
    async with engine:
        handle = await engine.submit_background("block")
        await started.wait()
        assert handle.cancel() is True
        result = await handle.wait()
    assert result.status is TaskStatus.CANCELLED
    assert handle.cancel() is False


@pytest.mark.asyncio
async def test_queued_task_can_be_cancelled_without_running_handler():
    llm = GateLLM()
    calls = 0

    def handler(ctx):
        nonlocal calls
        calls += 1

    engine = MultiTaskRuntime(make_runtime(llm=llm), config=MultiTaskRuntimeConfig(worker_count=1))
    engine.register("queued", handler)
    async with engine:
        foreground = asyncio.create_task(engine.run_turn("hello"))
        await llm.started.wait()
        handle = await engine.submit_background("queued")
        assert handle.cancel() is True
        llm.release.set()
        await foreground
        result = await handle.wait()
        await engine.join_background()
    assert result.status is TaskStatus.CANCELLED
    assert calls == 0


@pytest.mark.asyncio
async def test_worker_exception_becomes_failed_result_and_other_tasks_continue():
    def fail(ctx):
        raise ValueError("boom")

    engine = MultiTaskRuntime(make_runtime(), config=MultiTaskRuntimeConfig(worker_count=1))
    engine.register("fail", fail)
    engine.register("ok", lambda ctx: "ok")
    async with engine:
        bad = await engine.submit_background("fail")
        good = await engine.submit_background("ok")
        bad_result, good_result = await asyncio.gather(bad.wait(), good.wait())
    assert bad_result.status is TaskStatus.FAILED
    assert "ValueError: boom" == bad_result.error
    assert good_result.status is TaskStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_bounded_queue_rejects_excess_pending_work():
    llm = GateLLM()
    engine = MultiTaskRuntime(
        make_runtime(llm=llm),
        config=MultiTaskRuntimeConfig(worker_count=1, max_pending_tasks=1),
    )
    engine.register("bg", lambda ctx: "ok")
    async with engine:
        foreground = asyncio.create_task(engine.run_turn("hello"))
        await llm.started.wait()
        first = await engine.submit_background("bg")
        with pytest.raises(TaskQueueFullError):
            await engine.submit_background("bg")
        llm.release.set()
        await foreground
        await first.wait()


@pytest.mark.asyncio
async def test_unknown_task_type_is_rejected_before_queueing():
    engine = MultiTaskRuntime(make_runtime())
    async with engine:
        with pytest.raises(UnknownTaskTypeError):
            await engine.submit_background("missing")


@pytest.mark.asyncio
async def test_unknown_task_id_has_clear_error():
    engine = MultiTaskRuntime(make_runtime())
    async with engine:
        with pytest.raises(UnknownTaskError):
            engine.status("missing")


@pytest.mark.asyncio
async def test_close_cancels_queued_and_running_tasks_and_rejects_new_work():
    started = asyncio.Event()

    async def block(ctx):
        started.set()
        await asyncio.Event().wait()

    engine = MultiTaskRuntime(make_runtime(), config=MultiTaskRuntimeConfig(worker_count=1))
    engine.register("block", block)
    running = await engine.submit_background("block")
    await started.wait()
    queued = await engine.submit_background("block")
    await engine.close()
    assert (await running.wait()).status is TaskStatus.CANCELLED
    assert (await queued.wait()).status is TaskStatus.CANCELLED
    with pytest.raises(TaskRuntimeClosedError):
        await engine.submit_background("block")


@pytest.mark.asyncio
async def test_multiple_workers_run_background_tasks_concurrently():
    started = 0
    both = asyncio.Event()
    release = asyncio.Event()

    async def worker(ctx):
        nonlocal started
        started += 1
        if started == 2:
            both.set()
        await release.wait()
        return ctx.request.payload["id"]

    engine = MultiTaskRuntime(make_runtime(), config=MultiTaskRuntimeConfig(worker_count=2))
    engine.register("parallel", worker)
    async with engine:
        a = await engine.submit_background("parallel", {"id": "a"})
        b = await engine.submit_background("parallel", {"id": "b"})
        await asyncio.wait_for(both.wait(), timeout=0.25)
        assert engine.running_background == 2
        release.set()
        ra, rb = await asyncio.gather(a.wait(), b.wait())
    assert {ra.output.value, rb.output.value} == {"a", "b"}


@pytest.mark.asyncio
async def test_optional_foreground_preemption_cancels_running_background():
    started = asyncio.Event()

    async def block(ctx):
        started.set()
        await asyncio.Event().wait()

    engine = MultiTaskRuntime(
        make_runtime(),
        config=MultiTaskRuntimeConfig(
            worker_count=1,
            cancel_running_background_on_foreground=True,
        ),
    )
    engine.register("block", block)
    async with engine:
        handle = await engine.submit_background("block")
        await started.wait()
        result = await engine.run_turn("urgent")
        cancelled = await handle.wait()
    assert result.text == "foreground reply"
    assert cancelled.status is TaskStatus.CANCELLED


def test_task_context_does_not_expose_runtime_or_memory_manager():
    from ai_character_engine.tasks.models import TaskContext

    fields = TaskContext.__dataclass_fields__
    assert set(fields) == {"request", "snapshot"}
    assert "runtime" not in fields
    assert "memory_manager" not in fields


def test_multitask_runtime_config_validates_bounds():
    with pytest.raises(ValueError):
        MultiTaskRuntimeConfig(worker_count=0)
    with pytest.raises(ValueError):
        MultiTaskRuntimeConfig(max_pending_tasks=0)
    with pytest.raises(ValueError):
        MultiTaskRuntimeConfig(default_timeout_s=0)

@pytest.mark.asyncio
async def test_foreground_failure_does_not_advance_revision():
    class FailLLM:
        async def generate(self, messages, *, tools=None):
            raise RuntimeError("nope")

    engine = MultiTaskRuntime(make_runtime(llm=FailLLM()))
    async with engine:
        with pytest.raises(RuntimeError, match="nope"):
            await engine.run_turn("hello")
    assert engine.revision == 0


@pytest.mark.asyncio
async def test_close_can_gracefully_drain_running_background_task():
    started = asyncio.Event()
    release = asyncio.Event()

    async def worker(ctx):
        started.set()
        await release.wait()
        return "finished"

    engine = MultiTaskRuntime(make_runtime(), config=MultiTaskRuntimeConfig(worker_count=1))
    engine.register("slow", worker)
    handle = await engine.submit_background("slow")
    await started.wait()
    closing = asyncio.create_task(engine.close(cancel_running=False))
    await asyncio.sleep(0)
    assert not closing.done()
    release.set()
    await closing
    result = await handle.wait()
    assert result.status is TaskStatus.SUCCEEDED
    assert result.run_ms is not None
    assert result.total_ms >= result.run_ms


def test_task_request_payload_is_copied_from_host_input():
    from ai_character_engine.tasks import TaskRequest

    payload = {"nested": {"value": 1}}
    request = TaskRequest(task_type="x", payload=payload)
    payload["nested"]["value"] = 99
    assert request.payload["nested"]["value"] == 1

def test_public_api_exports_multitask_contracts():
    import ai_character_engine as ace

    assert ace.__version__ == VERSION
    assert ace.MultiTaskRuntime is MultiTaskRuntime
    assert ace.TaskPriority.NORMAL is TaskPriority.NORMAL
    assert ace.TaskStatus.SUCCEEDED is TaskStatus.SUCCEEDED


def test_event_loop_shutdown_with_a_running_background_task_does_not_hang():
    """asyncio.run() cancels every task at once when the loop closes.

    A host that is torn down that way (server shutdown, Ctrl-C) never gets to
    call close(); the worker must still stop instead of waiting for more work.
    """
    import threading

    async def scenario():
        runtime = MultiTaskRuntime(
            CharacterRuntime(
                character=CharacterProfile(id="c", name="C", description="test"),
                llm=FakeLLMClient(),
            )
        )
        started = asyncio.Event()

        async def never_finishes(context):
            started.set()
            await asyncio.Event().wait()

        runtime.register("slow", never_finishes)
        await runtime.submit_background("slow")
        await started.wait()

    thread = threading.Thread(target=lambda: asyncio.run(scenario()), daemon=True)
    thread.start()
    thread.join(5)
    assert not thread.is_alive()


@pytest.mark.asyncio
async def test_cancelling_one_running_task_keeps_the_worker_alive_for_the_next():
    runtime = MultiTaskRuntime(
        CharacterRuntime(
            character=CharacterProfile(id="c", name="C", description="test"),
            llm=FakeLLMClient(),
        ),
        config=MultiTaskRuntimeConfig(worker_count=1),
    )
    started = asyncio.Event()

    async def never_finishes(context):
        started.set()
        await asyncio.Event().wait()

    runtime.register("slow", never_finishes)
    runtime.register("quick", lambda context: "done")
    async with runtime:
        slow = await runtime.submit_background("slow")
        await started.wait()
        assert slow.cancel() is True
        assert (await slow.wait()).status is TaskStatus.CANCELLED
        quick = await runtime.submit_background("quick")
        result = await asyncio.wait_for(quick.wait(), 2)
    assert result.status is TaskStatus.SUCCEEDED


def test_event_loop_shutdown_does_not_hang_when_a_handler_turns_cancellation_into_an_error():
    """Found in a real host: a handler's cleanup raised ValueError while it was
    being cancelled. The task finished as FAILED, the worker's own cancellation
    had been spent on the handler, and the worker went back to wait for work."""
    import threading

    async def scenario():
        runtime = MultiTaskRuntime(
            CharacterRuntime(
                character=CharacterProfile(id="c", name="C", description="test"),
                llm=FakeLLMClient(),
            )
        )
        started = asyncio.Event()

        async def breaks_while_cancelled(context):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                raise ValueError("cleanup failed")

        runtime.register("slow", breaks_while_cancelled)
        await runtime.submit_background("slow")
        await started.wait()

    thread = threading.Thread(target=lambda: asyncio.run(scenario()), daemon=True)
    thread.start()
    thread.join(5)
    assert not thread.is_alive()


@pytest.mark.asyncio
async def test_a_host_can_run_the_foreground_turn_its_own_way():
    """The host bridge streams text, enforces a timeout and handles
    interruption, but went around the task runtime, so its turns never advanced
    the revision that background work and commits are measured against."""
    engine = MultiTaskRuntime(make_runtime(), config=MultiTaskRuntimeConfig(worker_count=1))
    seen_during_turn = []

    async def turn():
        seen_during_turn.append(engine.foreground_active)
        return await engine.runtime.process_event(CharacterEvent.user_message("hello"))

    async with engine:
        result = await engine.run_foreground_turn(turn)

    assert result.text == "foreground reply"
    assert engine.revision == 1
    assert seen_during_turn == [True]
    assert engine.foreground_active is False


@pytest.mark.asyncio
async def test_a_failed_foreground_turn_does_not_advance_the_revision():
    engine = MultiTaskRuntime(make_runtime(), config=MultiTaskRuntimeConfig(worker_count=1))

    async def turn():
        raise RuntimeError("provider is down")

    async with engine:
        with pytest.raises(RuntimeError):
            await engine.run_foreground_turn(turn)

    assert engine.revision == 0
    assert engine.foreground_active is False


@pytest.mark.asyncio
async def test_finished_tasks_are_kept_for_a_while_not_for_ever():
    """A host that runs for days submits a few tasks on every turn. Each was
    kept with a copy of the whole conversation: 28 KiB a turn in an 800-turn
    run. What a finished task leaves is its result, and only the newest."""
    engine = MultiTaskRuntime(
        make_runtime(), config=MultiTaskRuntimeConfig(worker_count=1, lifecycle_history=4)
    )
    engine.register("echo", lambda ctx: ctx.request.payload["n"])
    async with engine:
        handles = [await engine.submit_background("echo", {"n": n}) for n in range(10)]
        results = [await handle.wait() for handle in handles]

        assert [result.output.value for result in results] == list(range(10))
        assert engine.tracked_tasks == 4
        # The newest can still be asked about, as before.
        assert handles[-1].status is TaskStatus.SUCCEEDED
        assert (await handles[-1].wait()).output.value == 9
        assert handles[-1].cancel() is False
        with pytest.raises(UnknownTaskError):
            handles[0].status


@pytest.mark.asyncio
async def test_a_task_that_has_not_finished_is_never_let_go():
    gate = asyncio.Event()

    async def held(ctx):
        await gate.wait()
        return "done"

    engine = MultiTaskRuntime(
        make_runtime(), config=MultiTaskRuntimeConfig(worker_count=1, lifecycle_history=1)
    )
    engine.register("held", held)
    engine.register("echo", lambda ctx: "echo")
    async with engine:
        first = await engine.submit_background("held")
        others = [await engine.submit_background("echo") for _ in range(3)]
        await asyncio.sleep(0)
        assert first.status in {TaskStatus.QUEUED, TaskStatus.RUNNING}
        gate.set()
        assert (await first.wait()).output.value == "done"
        for other in others:
            await other.wait()
        assert engine.tracked_tasks == 1

