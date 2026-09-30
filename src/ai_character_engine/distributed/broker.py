from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol

from ai_character_engine.production.models import FailureClass, Idempotency
from ai_character_engine.tasks.models import TaskOutput

from .errors import DistributedQueueFullError, DistributedTaskNotFoundError, DuplicateDistributedTaskError
from .models import (
    CompletionDisposition,
    CompletionReceipt,
    DistributedCompletion,
    DistributedLease,
    DistributedLifecycleEvent,
    DistributedTaskEnvelope,
    DistributedTaskSnapshot,
    DistributedTaskStatus,
    clone_task_output,
    new_lease_id,
)


@dataclass(frozen=True, slots=True)
class DistributedBrokerConfig:
    lease_ttl_s: float = 30.0
    max_attempts: int = 3
    max_pending_tasks: int = 1024
    lifecycle_history: int = 2048

    def __post_init__(self) -> None:
        if self.lease_ttl_s <= 0:
            raise ValueError("lease_ttl_s must be > 0")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.max_pending_tasks < 1:
            raise ValueError("max_pending_tasks must be >= 1")
        if self.lifecycle_history < 1:
            raise ValueError("lifecycle_history must be >= 1")


class DistributedTaskBroker(Protocol):
    async def enqueue(self, envelope: DistributedTaskEnvelope) -> bool: ...
    async def lease(self, worker_id: str, task_types: tuple[str, ...]) -> DistributedLease | None: ...
    async def heartbeat(self, lease: DistributedLease) -> DistributedLease | None: ...
    async def complete(self, completion: DistributedCompletion) -> CompletionReceipt: ...
    async def reap_expired(self) -> int: ...
    async def cancel(self, task_id: str) -> bool: ...
    async def snapshot(self, task_id: str) -> DistributedTaskSnapshot: ...
    async def lifecycle_events(self, *, task_id: str | None = None) -> tuple[DistributedLifecycleEvent, ...]: ...


@dataclass(slots=True)
class _Record:
    envelope: DistributedTaskEnvelope
    sequence: int
    status: DistributedTaskStatus = DistributedTaskStatus.QUEUED
    attempts: int = 0
    fencing_token: int = 0
    lease: DistributedLease | None = None
    result: TaskOutput | None = None
    error: str | None = None
    last_completion_key: tuple[str, int] | None = None


class InMemoryDistributedTaskBroker:
    """Reference broker with at-least-once execution and fenced result acceptance.

    It is intentionally an in-process reference implementation, not a pretend
    distributed database. Hosts may implement ``DistributedTaskBroker`` with
    their queue/storage technology while preserving these semantics.
    """

    def __init__(
        self,
        *,
        config: DistributedBrokerConfig | None = None,
        clock=None,
    ) -> None:
        self.config = config or DistributedBrokerConfig()
        self._clock = clock or (lambda: datetime.now(UTC))
        self._records: dict[str, _Record] = {}
        self._sequence = 0
        self._lock = asyncio.Lock()
        self._events: deque[DistributedLifecycleEvent] = deque(maxlen=self.config.lifecycle_history)

    async def enqueue(self, envelope: DistributedTaskEnvelope) -> bool:
        envelope.to_dict()  # fail before queueing non-serializable cross-process payloads
        async with self._lock:
            existing = self._records.get(envelope.task_id)
            if existing is not None:
                if _same_envelope(existing.envelope, envelope):
                    return False
                raise DuplicateDistributedTaskError(
                    f"task id {envelope.task_id!r} already exists with different content"
                )
            pending = sum(1 for record in self._records.values() if not record.status.terminal)
            if pending >= self.config.max_pending_tasks:
                raise DistributedQueueFullError(
                    f"distributed queue is full (max={self.config.max_pending_tasks})"
                )
            self._sequence += 1
            self._records[envelope.task_id] = _Record(envelope=envelope, sequence=self._sequence)
            self._emit(self._records[envelope.task_id], "queued")
            return True

    async def lease(self, worker_id: str, task_types: tuple[str, ...]) -> DistributedLease | None:
        if not worker_id.strip():
            raise ValueError("worker_id must not be empty")
        cleaned = tuple(dict.fromkeys(x.strip() for x in task_types if x.strip()))
        if not cleaned:
            raise ValueError("task_types must contain at least one task type")
        async with self._lock:
            self._reap_expired_locked()
            candidates = [
                record for record in self._records.values()
                if record.status is DistributedTaskStatus.QUEUED
                and record.envelope.request.task_type in cleaned
            ]
            if not candidates:
                return None
            record = min(candidates, key=lambda r: (int(r.envelope.request.priority), r.sequence))
            record.attempts += 1
            record.fencing_token += 1
            now = self._clock()
            lease = DistributedLease(
                envelope=record.envelope,
                lease_id=new_lease_id(),
                worker_id=worker_id,
                attempt=record.attempts,
                fencing_token=record.fencing_token,
                leased_at=now,
                expires_at=now + timedelta(seconds=self.config.lease_ttl_s),
            )
            record.lease = lease
            record.status = DistributedTaskStatus.LEASED
            self._emit(record, "leased", worker_id=worker_id, lease_id=lease.lease_id)
            return lease

    async def heartbeat(self, lease: DistributedLease) -> DistributedLease | None:
        async with self._lock:
            self._reap_expired_locked()
            record = self._records.get(lease.task_id)
            if record is None or not self._matches_active_lease(record, lease.lease_id, lease.worker_id, lease.fencing_token):
                return None
            now = self._clock()
            refreshed = DistributedLease(
                envelope=record.envelope,
                lease_id=lease.lease_id,
                worker_id=lease.worker_id,
                attempt=lease.attempt,
                fencing_token=lease.fencing_token,
                leased_at=lease.leased_at,
                expires_at=now + timedelta(seconds=self.config.lease_ttl_s),
            )
            record.lease = refreshed
            self._emit(record, "heartbeat", worker_id=lease.worker_id, lease_id=lease.lease_id)
            return refreshed

    async def complete(self, completion: DistributedCompletion) -> CompletionReceipt:
        completion.to_dict()  # remote results must satisfy the versioned wire contract
        async with self._lock:
            self._reap_expired_locked()
            record = self._records.get(completion.task_id)
            if record is None:
                raise DistributedTaskNotFoundError(f"unknown distributed task {completion.task_id!r}")
            key = (completion.lease_id, completion.fencing_token)
            if record.last_completion_key == key:
                return CompletionReceipt(
                    completion.task_id, CompletionDisposition.DUPLICATE,
                    record.status, record.attempts, "completion already accepted",
                )
            if record.status.terminal:
                return CompletionReceipt(
                    completion.task_id, CompletionDisposition.REJECTED,
                    record.status, record.attempts, "task already terminal",
                )
            if not self._matches_active_lease(
                record, completion.lease_id, completion.worker_id, completion.fencing_token
            ):
                return CompletionReceipt(
                    completion.task_id, CompletionDisposition.STALE_LEASE,
                    record.status, record.attempts, "lease is no longer current",
                )

            record.last_completion_key = key
            record.lease = None
            if completion.succeeded:
                record.result = completion.output
                record.error = None
                record.status = DistributedTaskStatus.SUCCEEDED
                self._emit(record, "succeeded", worker_id=completion.worker_id, lease_id=completion.lease_id)
                return CompletionReceipt(
                    completion.task_id, CompletionDisposition.ACCEPTED,
                    record.status, record.attempts,
                )

            record.error = completion.error
            failure = completion.failure_class or FailureClass.PERMANENT
            if failure is FailureClass.AMBIGUOUS:
                record.status = DistributedTaskStatus.REVIEW_REQUIRED
                self._emit(record, "review_required", worker_id=completion.worker_id, lease_id=completion.lease_id, detail=record.error)
                return CompletionReceipt(completion.task_id, CompletionDisposition.ACCEPTED, record.status, record.attempts)

            can_retry = (
                failure is FailureClass.TRANSIENT
                and record.envelope.idempotency is Idempotency.SAFE
                and record.attempts < self.config.max_attempts
            )
            if can_retry:
                record.status = DistributedTaskStatus.QUEUED
                self._emit(record, "requeued", worker_id=completion.worker_id, lease_id=completion.lease_id, detail=record.error)
                return CompletionReceipt(completion.task_id, CompletionDisposition.REQUEUED, record.status, record.attempts)

            if failure is FailureClass.TRANSIENT and record.envelope.idempotency is not Idempotency.SAFE:
                record.status = DistributedTaskStatus.REVIEW_REQUIRED
            elif failure is FailureClass.TRANSIENT and record.attempts >= self.config.max_attempts:
                record.status = DistributedTaskStatus.DEAD_LETTER
            else:
                record.status = DistributedTaskStatus.FAILED
            self._emit(record, record.status.value, worker_id=completion.worker_id, lease_id=completion.lease_id, detail=record.error)
            return CompletionReceipt(completion.task_id, CompletionDisposition.ACCEPTED, record.status, record.attempts)

    async def reap_expired(self) -> int:
        async with self._lock:
            return self._reap_expired_locked()

    async def cancel(self, task_id: str) -> bool:
        async with self._lock:
            record = self._record(task_id)
            if record.status.terminal:
                return False
            record.status = DistributedTaskStatus.CANCELLED
            record.lease = None
            self._emit(record, "cancelled")
            return True

    async def snapshot(self, task_id: str) -> DistributedTaskSnapshot:
        async with self._lock:
            record = self._record(task_id)
            lease = record.lease
            return DistributedTaskSnapshot(
                task_id=record.envelope.task_id,
                task_type=record.envelope.request.task_type,
                status=record.status,
                attempts=record.attempts,
                fencing_token=record.fencing_token,
                active_lease_id=lease.lease_id if lease else None,
                active_worker_id=lease.worker_id if lease else None,
                result=clone_task_output(record.result),
                error=record.error,
            )

    async def lifecycle_events(self, *, task_id: str | None = None) -> tuple[DistributedLifecycleEvent, ...]:
        async with self._lock:
            if task_id is None:
                return tuple(self._events)
            return tuple(event for event in self._events if event.task_id == task_id)

    def _reap_expired_locked(self) -> int:
        now = self._clock()
        reaped = 0
        for record in self._records.values():
            lease = record.lease
            if record.status is not DistributedTaskStatus.LEASED or lease is None or lease.expires_at > now:
                continue
            reaped += 1
            record.lease = None
            if record.envelope.idempotency is not Idempotency.SAFE:
                record.status = DistributedTaskStatus.REVIEW_REQUIRED
                self._emit(record, "review_required", worker_id=lease.worker_id, lease_id=lease.lease_id, detail="lease expired with non-safe idempotency")
            elif record.attempts >= self.config.max_attempts:
                record.status = DistributedTaskStatus.DEAD_LETTER
                self._emit(record, "dead_letter", worker_id=lease.worker_id, lease_id=lease.lease_id, detail="lease expired after max attempts")
            else:
                record.status = DistributedTaskStatus.QUEUED
                self._emit(record, "lease_expired_requeue", worker_id=lease.worker_id, lease_id=lease.lease_id)
        return reaped

    def _matches_active_lease(self, record: _Record, lease_id: str, worker_id: str, fencing_token: int) -> bool:
        lease = record.lease
        return (
            record.status is DistributedTaskStatus.LEASED
            and lease is not None
            and lease.lease_id == lease_id
            and lease.worker_id == worker_id
            and lease.fencing_token == fencing_token
        )

    def _record(self, task_id: str) -> _Record:
        try:
            return self._records[task_id]
        except KeyError as exc:
            raise DistributedTaskNotFoundError(f"unknown distributed task {task_id!r}") from exc

    def _emit(self, record: _Record, status: str, *, worker_id: str | None = None, lease_id: str | None = None, detail: str | None = None) -> None:
        self._events.append(
            DistributedLifecycleEvent(
                task_id=record.envelope.task_id,
                task_type=record.envelope.request.task_type,
                status=status,
                at=self._clock(),
                attempt=record.attempts,
                worker_id=worker_id,
                lease_id=lease_id,
                fencing_token=record.fencing_token or None,
                detail=detail,
            )
        )


def _same_envelope(left: DistributedTaskEnvelope, right: DistributedTaskEnvelope) -> bool:
    return (
        left.request == right.request
        and left.snapshot == right.snapshot
        and left.idempotency == right.idempotency
        and dict(left.metadata) == dict(right.metadata)
        and left.protocol_version == right.protocol_version
    )
