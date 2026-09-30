from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum, IntEnum
from types import MappingProxyType
from typing import Any, Mapping
from uuid import uuid4

from ai_character_engine.llm.models import Message


class TaskPriority(IntEnum):
    """Background scheduling priority. Lower values run first."""

    CRITICAL = 0
    HIGH = 20
    NORMAL = 50
    LOW = 80
    IDLE = 100


class TaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"

    @property
    def terminal(self) -> bool:
        return self in {
            TaskStatus.SUCCEEDED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
            TaskStatus.TIMED_OUT,
        }


@dataclass(slots=True, frozen=True)
class TaskRequest:
    """One background task request.

    ``payload`` is deliberately provider-neutral. v0.31 may add model-role routing,
    but v0.30 only owns task lifecycle and scheduling.
    """

    task_type: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    timeout_s: float | None = None
    source: str = "host"
    id: str = field(default_factory=lambda: uuid4().hex)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if not self.task_type.strip():
            raise ValueError("task_type must not be empty")
        if not self.source.strip():
            raise ValueError("source must not be empty")
        if self.timeout_s is not None and self.timeout_s <= 0:
            raise ValueError("timeout_s must be > 0 when provided")
        object.__setattr__(self, "payload", MappingProxyType(copy.deepcopy(dict(self.payload))))


@dataclass(slots=True, frozen=True)
class TaskStateSnapshot:
    """Immutable, renderer/model-neutral copy of authoritative character state."""

    emotion: str
    energy: float
    trust: float
    favorability: float
    relationship_stage: str
    custom: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "custom", MappingProxyType(copy.deepcopy(dict(self.custom))))


@dataclass(slots=True, frozen=True)
class TaskSnapshot:
    """Read-only character snapshot captured when a background task is submitted.

    No CharacterRuntime, MemoryManager, store or mutable CharacterState reference is
    exposed to workers. Background work therefore produces data/proposals; it does
    not become an alternative authoritative commit path.
    """

    revision: int
    captured_at: datetime
    character_id: str
    character_name: str
    state: TaskStateSnapshot
    history: tuple[Message, ...]
    memory_scope_id: str


@dataclass(slots=True, frozen=True)
class TaskProposal:
    """Non-authoritative proposal emitted by a background worker.

    A proposal carries enough provenance for a future commit coordinator to
    decide whether the result is trustworthy and still fresh.  v0.32 still
    never applies proposals directly to CharacterState or MemoryManager.
    """

    target: str
    payload: Mapping[str, Any]
    base_revision: int
    source_task_id: str | None = None
    confidence: float | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    id: str = field(default_factory=lambda: uuid4().hex)

    def __post_init__(self) -> None:
        if not self.target.strip():
            raise ValueError("proposal target must not be empty")
        if self.base_revision < 0:
            raise ValueError("base_revision must be >= 0")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("proposal confidence must be between 0 and 1")
        object.__setattr__(self, "payload", MappingProxyType(copy.deepcopy(dict(self.payload))))
        object.__setattr__(self, "provenance", MappingProxyType(copy.deepcopy(dict(self.provenance))))

    def is_stale(self, current_revision: int) -> bool:
        if current_revision < 0:
            raise ValueError("current_revision must be >= 0")
        return current_revision != self.base_revision


@dataclass(slots=True, frozen=True)
class TaskOutput:
    """Worker output. ``proposals`` remain non-authoritative in v0.30."""

    value: Any = None
    proposals: tuple[TaskProposal, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(copy.deepcopy(dict(self.metadata))))


@dataclass(slots=True, frozen=True)
class TaskResult:
    task_id: str
    task_type: str
    status: TaskStatus
    output: TaskOutput | None
    snapshot_revision: int
    queued_at: datetime
    started_at: datetime | None
    completed_at: datetime
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.status.terminal:
            raise ValueError("TaskResult requires a terminal status")

    @property
    def queue_wait_ms(self) -> float | None:
        if self.started_at is None:
            return None
        return max(0.0, (self.started_at - self.queued_at).total_seconds() * 1000.0)

    @property
    def run_ms(self) -> float | None:
        if self.started_at is None:
            return None
        return max(0.0, (self.completed_at - self.started_at).total_seconds() * 1000.0)

    @property
    def total_ms(self) -> float:
        return max(0.0, (self.completed_at - self.queued_at).total_seconds() * 1000.0)


@dataclass(slots=True, frozen=True)
class TaskLifecycleEvent:
    task_id: str
    task_type: str
    status: TaskStatus
    occurred_at: datetime
    detail: str | None = None


@dataclass(slots=True, frozen=True)
class TaskContext:
    request: TaskRequest
    snapshot: TaskSnapshot

    def proposal(
        self,
        target: str,
        payload: Mapping[str, Any],
        *,
        confidence: float | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> TaskProposal:
        return TaskProposal(
            target=target,
            payload=payload,
            base_revision=self.snapshot.revision,
            source_task_id=self.request.id,
            confidence=confidence,
            provenance=provenance or {},
        )
