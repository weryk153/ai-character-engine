from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ai_character_engine.production.models import Idempotency
from ai_character_engine.production.runtime import default_failure_classifier
from ai_character_engine.tasks.models import TaskContext, TaskOutput, TaskRequest, TaskSnapshot

from .broker import DistributedTaskBroker
from .models import DistributedCompletion, DistributedLease, DistributedTaskEnvelope

DistributedTaskHandler = Callable[[TaskContext], TaskOutput | Any | Awaitable[TaskOutput | Any]]


@dataclass(frozen=True, slots=True)
class DistributedWorkerConfig:
    lease_heartbeat_interval_s: float = 5.0
    default_task_timeout_s: float = 30.0

    def __post_init__(self) -> None:
        if self.lease_heartbeat_interval_s <= 0:
            raise ValueError("lease_heartbeat_interval_s must be > 0")
        if self.default_task_timeout_s <= 0:
            raise ValueError("default_task_timeout_s must be > 0")


class DistributedTaskCoordinator:
    """Transport-neutral host facade. It owns delivery, never cognition authority."""

    def __init__(self, broker: DistributedTaskBroker) -> None:
        self.broker = broker

    async def submit(
        self,
        request: TaskRequest,
        snapshot: TaskSnapshot,
        *,
        idempotency: Idempotency,
        metadata: Mapping[str, Any] | None = None,
    ) -> DistributedTaskEnvelope:
        envelope = DistributedTaskEnvelope(
            request=request,
            snapshot=snapshot,
            idempotency=idempotency,
            metadata=dict(metadata or {}),
        )
        await self.broker.enqueue(envelope)
        return envelope

    async def cancel(self, task_id: str) -> bool:
        return await self.broker.cancel(task_id)

    async def reap_expired(self) -> int:
        return await self.broker.reap_expired()

    async def wait(self, task_id: str, *, poll_interval_s: float = 0.05, timeout_s: float | None = None):
        if poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be > 0")
        if timeout_s is not None and timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")

        async def poll():
            while True:
                state = await self.broker.snapshot(task_id)
                if state.status.terminal:
                    return state
                await asyncio.sleep(poll_interval_s)

        if timeout_s is None:
            return await poll()
        return await asyncio.wait_for(poll(), timeout=timeout_s)


class DistributedWorker:
    """Executes leased non-authoritative TaskContext values on a remote-capable boundary."""

    def __init__(
        self,
        broker: DistributedTaskBroker,
        *,
        worker_id: str,
        handlers: Mapping[str, DistributedTaskHandler],
        config: DistributedWorkerConfig | None = None,
        failure_classifier=default_failure_classifier,
    ) -> None:
        if not worker_id.strip():
            raise ValueError("worker_id must not be empty")
        cleaned = {key.strip(): value for key, value in handlers.items() if key.strip()}
        if not cleaned:
            raise ValueError("handlers must contain at least one task type")
        self.broker = broker
        self.worker_id = worker_id
        self.handlers = cleaned
        self.config = config or DistributedWorkerConfig()
        self.failure_classifier = failure_classifier

    async def run_once(self) -> bool:
        lease = await self.broker.lease(self.worker_id, tuple(self.handlers))
        if lease is None:
            return False
        await self._execute(lease)
        return True

    async def run_forever(self, *, poll_interval_s: float = 0.25, stop_event: asyncio.Event | None = None) -> None:
        if poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be > 0")
        while stop_event is None or not stop_event.is_set():
            handled = await self.run_once()
            if not handled:
                if stop_event is None:
                    await asyncio.sleep(poll_interval_s)
                else:
                    try:
                        await asyncio.wait_for(stop_event.wait(), timeout=poll_interval_s)
                    except asyncio.TimeoutError:
                        pass

    async def _execute(self, lease: DistributedLease) -> None:
        handler = self.handlers.get(lease.envelope.request.task_type)
        if handler is None:
            return
        stop = asyncio.Event()
        heartbeat_task = asyncio.create_task(self._heartbeat_loop(lease, stop))
        context = TaskContext(lease.envelope.request, lease.envelope.snapshot)
        async def invoke() -> TaskOutput:
            value = handler(context)
            if inspect.isawaitable(value):
                value = await value
            return value if isinstance(value, TaskOutput) else TaskOutput(value=value)

        try:
            timeout_s = lease.envelope.request.timeout_s or self.config.default_task_timeout_s
            output = await asyncio.wait_for(invoke(), timeout=timeout_s)
            completion = DistributedCompletion(
                task_id=lease.task_id,
                lease_id=lease.lease_id,
                worker_id=self.worker_id,
                fencing_token=lease.fencing_token,
                output=output,
            )
            await self.broker.complete(completion)
        except asyncio.CancelledError:
            # Process/worker loss is modeled by abandoning the lease. The broker's
            # fencing and expiry policy decides whether re-execution is safe.
            raise
        except Exception as exc:
            completion = DistributedCompletion(
                task_id=lease.task_id,
                lease_id=lease.lease_id,
                worker_id=self.worker_id,
                fencing_token=lease.fencing_token,
                error=f"{type(exc).__name__}: {exc}",
                failure_class=self.failure_classifier(exc),
            )
            await self.broker.complete(completion)
        finally:
            stop.set()
            heartbeat_task.cancel()
            await asyncio.gather(heartbeat_task, return_exceptions=True)

    async def _heartbeat_loop(self, lease: DistributedLease, stop: asyncio.Event) -> None:
        current = lease
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.config.lease_heartbeat_interval_s)
                return
            except asyncio.TimeoutError:
                refreshed = await self.broker.heartbeat(current)
                if refreshed is None:
                    return
                current = refreshed
