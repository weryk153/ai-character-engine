from __future__ import annotations

import asyncio
import copy
import inspect
import itertools
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any, Protocol

from ai_character_engine.events.models import CharacterEvent
from ai_character_engine.runtime.character_runtime import CharacterRuntime
from ai_character_engine.runtime.models import CharacterRunResult

from .errors import (
    TaskQueueFullError,
    TaskRuntimeClosedError,
    UnknownTaskError,
    UnknownTaskTypeError,
)
from .models import (
    TaskContext,
    TaskLifecycleEvent,
    TaskOutput,
    TaskPriority,
    TaskRequest,
    TaskResult,
    TaskSnapshot,
    TaskStateSnapshot,
    TaskStatus,
)


class TaskHandler(Protocol):
    def __call__(self, context: TaskContext) -> TaskOutput | Any | Awaitable[TaskOutput | Any]: ...


LifecycleSink = Callable[[TaskLifecycleEvent], None]


@dataclass(slots=True, frozen=True)
class MultiTaskRuntimeConfig:
    worker_count: int = 2
    max_pending_tasks: int = 64
    default_timeout_s: float = 30.0
    pause_new_background_during_foreground: bool = True
    cancel_running_background_on_foreground: bool = False
    lifecycle_history: int = 512

    def __post_init__(self) -> None:
        if self.worker_count < 1:
            raise ValueError("worker_count must be >= 1")
        if self.max_pending_tasks < 1:
            raise ValueError("max_pending_tasks must be >= 1")
        if self.default_timeout_s <= 0:
            raise ValueError("default_timeout_s must be > 0")
        if self.lifecycle_history < 1:
            raise ValueError("lifecycle_history must be >= 1")


@dataclass(slots=True)
class _TaskRecord:
    request: TaskRequest
    snapshot: TaskSnapshot
    status: TaskStatus
    queued_at: datetime
    future: asyncio.Future[TaskResult]
    started_at: datetime | None = None
    running: asyncio.Task[Any] | None = None


class TaskHandle:
    """Lightweight host handle for one submitted background task."""

    __slots__ = ("_runtime", "task_id")

    def __init__(self, runtime: "MultiTaskRuntime", task_id: str) -> None:
        self._runtime = runtime
        self.task_id = task_id

    @property
    def status(self) -> TaskStatus:
        return self._runtime.status(self.task_id)

    def cancel(self) -> bool:
        return self._runtime.cancel(self.task_id)

    async def wait(self) -> TaskResult:
        return await self._runtime.wait(self.task_id)


class MultiTaskRuntime:
    """Foreground character turns plus bounded non-authoritative background work.

    Foreground turns bypass the background queue and continue to use the existing
    CharacterRuntime/TurnCoordinator authoritative path. Background workers only
    receive immutable snapshots and return TaskOutput/TaskProposal values. v0.30
    intentionally does *not* apply proposals to CharacterState or MemoryManager.
    """

    def __init__(
        self,
        runtime: CharacterRuntime,
        *,
        config: MultiTaskRuntimeConfig | None = None,
        on_lifecycle: LifecycleSink | None = None,
    ) -> None:
        self.runtime = runtime
        self.config = config or MultiTaskRuntimeConfig()
        self._on_lifecycle = on_lifecycle
        self._handlers: dict[str, TaskHandler] = {}
        self._queue: asyncio.PriorityQueue[tuple[int, int, str]] = asyncio.PriorityQueue(
            maxsize=self.config.max_pending_tasks
        )
        self._sequence = itertools.count()
        # Tasks that are queued or running. A record holds a copy of the
        # conversation as it was when the task was submitted.
        self._records: dict[str, _TaskRecord] = {}
        # What finished tasks leave behind: their result, the newest
        # lifecycle_history of them. A host that runs for days must not keep
        # every task it ever ran.
        self._finished: dict[str, TaskResult] = {}
        self._workers: list[asyncio.Task[None]] = []
        self._closed = False
        self._started = False
        self._revision = 0
        self._foreground_count = 0
        self._authority_lock = asyncio.Lock()
        self._background_gate = asyncio.Event()
        self._background_gate.set()
        self._lifecycle: deque[TaskLifecycleEvent] = deque(
            maxlen=self.config.lifecycle_history
        )

    @property
    def revision(self) -> int:
        """Monotonic revision of successful authoritative foreground turns."""

        return self._revision

    @property
    def pending_background(self) -> int:
        return sum(1 for record in self._records.values() if record.status is TaskStatus.QUEUED)

    @property
    def running_background(self) -> int:
        return sum(1 for record in self._records.values() if record.status is TaskStatus.RUNNING)

    @property
    def foreground_active(self) -> bool:
        return self._foreground_count > 0

    @property
    def closed(self) -> bool:
        return self._closed

    def register(self, task_type: str, handler: TaskHandler) -> None:
        cleaned = task_type.strip()
        if not cleaned:
            raise ValueError("task_type must not be empty")
        if self._closed:
            raise TaskRuntimeClosedError("task runtime is closed")
        self._handlers[cleaned] = handler

    def unregister(self, task_type: str) -> None:
        self._handlers.pop(task_type, None)

    def lifecycle_events(self, *, task_id: str | None = None) -> tuple[TaskLifecycleEvent, ...]:
        if task_id is None:
            return tuple(self._lifecycle)
        return tuple(event for event in self._lifecycle if event.task_id == task_id)

    @property
    def tracked_tasks(self) -> int:
        """How many tasks the runtime can still answer for."""

        return len(self._records) + len(self._finished)

    def status(self, task_id: str) -> TaskStatus:
        finished = self._finished.get(task_id)
        if finished is not None:
            return finished.status
        return self._record(task_id).status

    async def start(self) -> None:
        if self._closed:
            raise TaskRuntimeClosedError("task runtime is closed")
        if self._started:
            return
        self._started = True
        self._workers = [
            asyncio.create_task(self._worker_loop(index), name=f"character-task-worker-{index}")
            for index in range(self.config.worker_count)
        ]

    async def __aenter__(self) -> "MultiTaskRuntime":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def submit_background(
        self,
        task_type: str,
        payload: dict[str, Any] | None = None,
        *,
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout_s: float | None = None,
        source: str = "host",
    ) -> TaskHandle:
        if self._closed:
            raise TaskRuntimeClosedError("task runtime is closed")
        cleaned = task_type.strip()
        if cleaned not in self._handlers:
            raise UnknownTaskTypeError(f"no worker registered for task type {cleaned!r}")
        if not self._started:
            await self.start()

        request = TaskRequest(
            task_type=cleaned,
            payload=payload or {},
            priority=priority,
            timeout_s=timeout_s,
            source=source,
        )
        loop = asyncio.get_running_loop()
        queued_at = datetime.now(UTC)
        record = _TaskRecord(
            request=request,
            snapshot=self.capture_snapshot(),
            status=TaskStatus.QUEUED,
            queued_at=queued_at,
            future=loop.create_future(),
        )
        try:
            self._queue.put_nowait((int(priority), next(self._sequence), request.id))
        except asyncio.QueueFull as exc:
            raise TaskQueueFullError(
                f"background queue is full (max={self.config.max_pending_tasks})"
            ) from exc
        self._records[request.id] = record
        self._emit(record, TaskStatus.QUEUED)
        return TaskHandle(self, request.id)

    async def wait(self, task_id: str) -> TaskResult:
        finished = self._finished.get(task_id)
        if finished is not None:
            return finished
        return await asyncio.shield(self._record(task_id).future)

    def cancel(self, task_id: str) -> bool:
        if task_id in self._finished:
            return False
        record = self._record(task_id)
        if record.status.terminal:
            return False
        if record.status is TaskStatus.QUEUED:
            self._finish(record, TaskStatus.CANCELLED, error="cancelled before execution")
            return True
        if record.running is not None:
            record.running.cancel()
            return True
        return False

    @asynccontextmanager
    async def authority_guard(self) -> AsyncIterator[None]:
        """Serialize authoritative foreground turns and background commits.

        Background inference remains concurrent. Only the short authoritative mutation
        boundary shares this lock so a proposal cannot pass a freshness check and then
        race with a foreground turn before it writes State or Memory.
        """

        async with self._authority_lock:
            yield

    async def run_foreground(self, event: CharacterEvent) -> CharacterRunResult:
        """Run one authoritative character turn without entering the background queue."""

        return await self.run_foreground_turn(lambda: self.runtime.process_event(event))

    async def run_foreground_turn(
        self, turn: Callable[[], Awaitable[CharacterRunResult]]
    ) -> CharacterRunResult:
        """Run a foreground turn the host performs itself, as the authoritative turn.

        ``turn`` must drive this runtime's CharacterRuntime, for example through
        CharacterHostBridge.process so the host keeps streaming, its timeout and
        interruption. The revision advances only when the turn succeeds.
        """

        if self._closed:
            raise TaskRuntimeClosedError("task runtime is closed")
        async with self.authority_guard():
            self._begin_foreground()
            try:
                if self.config.cancel_running_background_on_foreground:
                    for record in tuple(self._records.values()):
                        if record.status is TaskStatus.RUNNING and record.running is not None:
                            record.running.cancel()
                result = await turn()
            except BaseException:
                raise
            else:
                self._revision += 1
                return result
            finally:
                self._end_foreground()

    async def run_turn(self, user_message: str) -> CharacterRunResult:
        return await self.run_foreground(CharacterEvent.user_message(user_message))

    async def join_background(self) -> None:
        """Wait until all currently queued background entries have been consumed."""

        await self._queue.join()

    async def close(self, *, cancel_running: bool = True) -> None:
        if self._closed:
            return
        self._closed = True

        running_futures: list[asyncio.Future[TaskResult]] = []
        for record in tuple(self._records.values()):
            if record.status is TaskStatus.QUEUED:
                self._finish(record, TaskStatus.CANCELLED, error="task runtime closed")
            elif record.status is TaskStatus.RUNNING:
                running_futures.append(record.future)
                if cancel_running and record.running is not None:
                    record.running.cancel()

        # Let workers observe cancellation and terminalize their TaskResult before
        # cancelling the worker loops themselves. Cancelling both layers at once
        # can consume the worker cancellation inside _execute_record and leave the
        # loop waiting forever for another queue item.
        if running_futures:
            # cancel_running=False is a graceful drain: no new work is accepted,
            # queued work is cancelled, but already-running handlers may finish.
            await asyncio.gather(
                *(asyncio.shield(future) for future in running_futures),
                return_exceptions=True,
            )

        for worker in self._workers:
            worker.cancel()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

    def capture_snapshot(self) -> TaskSnapshot:
        """Capture the same immutable task snapshot used by local workers.

        This is intentionally read-only and exists so deployment adapters can
        serialize work without receiving CharacterRuntime mutation authority.
        """
        return self._capture_snapshot()

    def _capture_snapshot(self) -> TaskSnapshot:
        state = self.runtime.state.snapshot()
        return TaskSnapshot(
            revision=self._revision,
            captured_at=datetime.now(UTC),
            character_id=self.runtime.character.id,
            character_name=self.runtime.character.name,
            state=TaskStateSnapshot(
                emotion=state.emotion,
                energy=state.energy,
                trust=state.trust,
                favorability=state.favorability,
                relationship_stage=state.relationship_stage,
                custom=copy.deepcopy(state.custom),
            ),
            history=tuple(copy.deepcopy(self.runtime.history)),
            memory_scope_id=self.runtime.memory_scope_id,
        )

    def _begin_foreground(self) -> None:
        self._foreground_count += 1
        if self.config.pause_new_background_during_foreground:
            self._background_gate.clear()

    def _end_foreground(self) -> None:
        self._foreground_count = max(0, self._foreground_count - 1)
        if self._foreground_count == 0:
            self._background_gate.set()

    async def _worker_loop(self, worker_index: int) -> None:
        while True:
            try:
                if self.config.pause_new_background_during_foreground:
                    await self._background_gate.wait()
                item = await self._queue.get()
            except asyncio.CancelledError:
                return
            try:
                if self.config.pause_new_background_during_foreground and self.foreground_active:
                    # Put the item back before waiting. This avoids holding a low-priority
                    # task while foreground work is active, so the priority queue can choose
                    # the best candidate once foreground releases.
                    self._queue.put_nowait(item)
                    continue
                _, _, task_id = item
                record = self._records.get(task_id)
                if record is None or record.status is not TaskStatus.QUEUED:
                    continue
                await self._execute_record(record, worker_index)
                # A handler can spend the worker's cancellation: awaiting it hands
                # the cancel to the handler, which may finish with a result or an
                # ordinary error instead. The request is still on record.
                current = asyncio.current_task()
                if current is not None and current.cancelling():
                    return
            finally:
                self._queue.task_done()

    async def _execute_record(self, record: _TaskRecord, worker_index: int) -> None:
        handler = self._handlers.get(record.request.task_type)
        if handler is None:
            self._finish(
                record,
                TaskStatus.FAILED,
                error=f"worker unregistered for task type {record.request.task_type!r}",
            )
            return

        record.status = TaskStatus.RUNNING
        record.started_at = datetime.now(UTC)
        self._emit(record, TaskStatus.RUNNING, detail=f"worker={worker_index}")
        context = TaskContext(record.request, record.snapshot)

        async def invoke() -> TaskOutput:
            value = handler(context)
            if inspect.isawaitable(value):
                value = await value
            if isinstance(value, TaskOutput):
                return value
            return TaskOutput(value=value)

        running = asyncio.create_task(invoke(), name=f"character-task-{record.request.id}")
        record.running = running
        timeout_s = record.request.timeout_s or self.config.default_timeout_s
        try:
            output = await asyncio.wait_for(running, timeout=timeout_s)
        except asyncio.TimeoutError:
            self._finish(
                record,
                TaskStatus.TIMED_OUT,
                error=f"background task timed out after {timeout_s:.3f}s",
            )
        except asyncio.CancelledError:
            self._finish(record, TaskStatus.CANCELLED, error="background task cancelled")
            # Swallow only a cancellation aimed at this one task. When the worker
            # itself is cancelled (the event loop is shutting down and the host
            # never reached close()), it must stop rather than wait for more work.
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                running.cancel()
                raise
        except Exception as exc:
            self._finish(record, TaskStatus.FAILED, error=f"{type(exc).__name__}: {exc}")
        else:
            self._finish(record, TaskStatus.SUCCEEDED, output=output)
        finally:
            record.running = None

    def _finish(
        self,
        record: _TaskRecord,
        status: TaskStatus,
        *,
        output: TaskOutput | None = None,
        error: str | None = None,
    ) -> None:
        if record.status.terminal and record.future.done():
            return
        record.status = status
        result = TaskResult(
            task_id=record.request.id,
            task_type=record.request.task_type,
            status=status,
            output=output,
            snapshot_revision=record.snapshot.revision,
            queued_at=record.queued_at,
            started_at=record.started_at,
            completed_at=datetime.now(UTC),
            error=error,
        )
        self._emit(record, status, detail=error)
        if not record.future.done():
            record.future.set_result(result)
        if self._records.pop(record.request.id, None) is not None:
            self._finished[record.request.id] = result
            while len(self._finished) > self.config.lifecycle_history:
                del self._finished[next(iter(self._finished))]

    def _emit(
        self,
        record: _TaskRecord,
        status: TaskStatus,
        *,
        detail: str | None = None,
    ) -> None:
        event = TaskLifecycleEvent(
            task_id=record.request.id,
            task_type=record.request.task_type,
            status=status,
            occurred_at=datetime.now(UTC),
            detail=detail,
        )
        self._lifecycle.append(event)
        if self._on_lifecycle is not None:
            self._on_lifecycle(event)

    def _record(self, task_id: str) -> _TaskRecord:
        try:
            return self._records[task_id]
        except KeyError as exc:
            raise UnknownTaskError(f"unknown task id {task_id!r}") from exc
