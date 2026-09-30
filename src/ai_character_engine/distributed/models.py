from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping
from uuid import uuid4

from ai_character_engine.llm.models import Message
from ai_character_engine.production.models import FailureClass, Idempotency
from ai_character_engine.tasks.models import TaskOutput, TaskProposal, TaskRequest, TaskSnapshot, TaskStateSnapshot
from ai_character_engine.tools.models import ToolCall, ToolResult

DISTRIBUTED_PROTOCOL_VERSION = 1


class DistributedTaskStatus(str, Enum):
    QUEUED = "queued"
    LEASED = "leased"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REVIEW_REQUIRED = "review_required"
    CANCELLED = "cancelled"
    DEAD_LETTER = "dead_letter"

    @property
    def terminal(self) -> bool:
        return self in {
            DistributedTaskStatus.SUCCEEDED,
            DistributedTaskStatus.FAILED,
            DistributedTaskStatus.REVIEW_REQUIRED,
            DistributedTaskStatus.CANCELLED,
            DistributedTaskStatus.DEAD_LETTER,
        }


class CompletionDisposition(str, Enum):
    ACCEPTED = "accepted"
    REQUEUED = "requeued"
    DUPLICATE = "duplicate"
    STALE_LEASE = "stale_lease"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class DistributedTaskEnvelope:
    request: TaskRequest
    snapshot: TaskSnapshot
    idempotency: Idempotency = Idempotency.UNKNOWN
    metadata: Mapping[str, Any] = field(default_factory=dict)
    protocol_version: int = DISTRIBUTED_PROTOCOL_VERSION
    submitted_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if self.protocol_version != DISTRIBUTED_PROTOCOL_VERSION:
            raise ValueError(
                f"unsupported distributed protocol version {self.protocol_version}; "
                f"expected {DISTRIBUTED_PROTOCOL_VERSION}"
            )
        object.__setattr__(self, "metadata", MappingProxyType(copy.deepcopy(dict(self.metadata))))

    @property
    def task_id(self) -> str:
        return self.request.id

    def to_dict(self) -> dict[str, Any]:
        data = {
            "protocol_version": self.protocol_version,
            "submitted_at": self.submitted_at.isoformat(),
            "idempotency": self.idempotency.value,
            "metadata": copy.deepcopy(dict(self.metadata)),
            "request": _request_to_dict(self.request),
            "snapshot": _snapshot_to_dict(self.snapshot),
        }
        _ensure_json_safe(data)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DistributedTaskEnvelope":
        version = int(data.get("protocol_version", 0))
        if version != DISTRIBUTED_PROTOCOL_VERSION:
            raise ValueError(
                f"unsupported distributed protocol version {version}; expected {DISTRIBUTED_PROTOCOL_VERSION}"
            )
        return cls(
            request=_request_from_dict(_mapping(data, "request")),
            snapshot=_snapshot_from_dict(_mapping(data, "snapshot")),
            idempotency=Idempotency(str(data.get("idempotency", Idempotency.UNKNOWN.value))),
            metadata=dict(_mapping(data, "metadata", default={})),
            protocol_version=version,
            submitted_at=_parse_dt(data.get("submitted_at")),
        )


@dataclass(frozen=True, slots=True)
class DistributedLease:
    envelope: DistributedTaskEnvelope
    lease_id: str
    worker_id: str
    attempt: int
    fencing_token: int
    leased_at: datetime
    expires_at: datetime

    def __post_init__(self) -> None:
        if not self.lease_id.strip() or not self.worker_id.strip():
            raise ValueError("lease_id and worker_id must not be empty")
        if self.attempt < 1 or self.fencing_token < 1:
            raise ValueError("attempt and fencing_token must be >= 1")
        if self.expires_at <= self.leased_at:
            raise ValueError("expires_at must be after leased_at")

    @property
    def task_id(self) -> str:
        return self.envelope.task_id

    def expired(self, now: datetime | None = None) -> bool:
        return (now or datetime.now(UTC)) >= self.expires_at


@dataclass(frozen=True, slots=True)
class DistributedCompletion:
    task_id: str
    lease_id: str
    worker_id: str
    fencing_token: int
    output: TaskOutput | None = None
    error: str | None = None
    failure_class: FailureClass | None = None
    completed_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if not self.task_id.strip() or not self.lease_id.strip() or not self.worker_id.strip():
            raise ValueError("task_id, lease_id, and worker_id must not be empty")
        if self.fencing_token < 1:
            raise ValueError("fencing_token must be >= 1")
        if self.output is not None and self.error is not None:
            raise ValueError("completion cannot contain both output and error")
        if self.output is None and self.error is None:
            raise ValueError("completion requires output or error")
        if self.output is not None and self.failure_class is not None:
            raise ValueError("successful completion must not include failure_class")

    @property
    def succeeded(self) -> bool:
        return self.output is not None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "protocol_version": DISTRIBUTED_PROTOCOL_VERSION,
            "task_id": self.task_id,
            "lease_id": self.lease_id,
            "worker_id": self.worker_id,
            "fencing_token": self.fencing_token,
            "completed_at": self.completed_at.isoformat(),
            "output": _output_to_dict(self.output) if self.output is not None else None,
            "error": self.error,
            "failure_class": self.failure_class.value if self.failure_class else None,
        }
        _ensure_json_safe(data)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DistributedCompletion":
        version = int(data.get("protocol_version", 0))
        if version != DISTRIBUTED_PROTOCOL_VERSION:
            raise ValueError(f"unsupported distributed protocol version {version}")
        raw_output = data.get("output")
        failure = data.get("failure_class")
        return cls(
            task_id=str(data["task_id"]),
            lease_id=str(data["lease_id"]),
            worker_id=str(data["worker_id"]),
            fencing_token=int(data["fencing_token"]),
            completed_at=_parse_dt(data.get("completed_at")),
            output=_output_from_dict(raw_output) if isinstance(raw_output, Mapping) else None,
            error=str(data["error"]) if data.get("error") is not None else None,
            failure_class=FailureClass(str(failure)) if failure is not None else None,
        )


@dataclass(frozen=True, slots=True)
class CompletionReceipt:
    task_id: str
    disposition: CompletionDisposition
    status: DistributedTaskStatus
    attempt: int
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class DistributedTaskSnapshot:
    task_id: str
    task_type: str
    status: DistributedTaskStatus
    attempts: int
    fencing_token: int
    active_lease_id: str | None
    active_worker_id: str | None
    result: TaskOutput | None
    error: str | None


@dataclass(frozen=True, slots=True)
class DistributedLifecycleEvent:
    task_id: str
    task_type: str
    status: str
    at: datetime
    attempt: int
    worker_id: str | None = None
    lease_id: str | None = None
    fencing_token: int | None = None
    detail: str | None = None


def clone_task_output(output: TaskOutput | None) -> TaskOutput | None:
    if output is None:
        return None
    return _output_from_dict(_output_to_dict(output))


def new_lease_id() -> str:
    return uuid4().hex


def _ensure_json_safe(value: Any) -> None:
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"distributed wire payload must be JSON-safe: {exc}") from exc


def _mapping(data: Mapping[str, Any], key: str, *, default: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    value = data.get(key, default)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value


def _parse_dt(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("datetime field must be an ISO-8601 string")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _request_to_dict(request: TaskRequest) -> dict[str, Any]:
    return {
        "task_type": request.task_type,
        "payload": copy.deepcopy(dict(request.payload)),
        "priority": int(request.priority),
        "timeout_s": request.timeout_s,
        "source": request.source,
        "id": request.id,
        "created_at": request.created_at.isoformat(),
    }


def _request_from_dict(data: Mapping[str, Any]) -> TaskRequest:
    from ai_character_engine.tasks.models import TaskPriority
    return TaskRequest(
        task_type=str(data["task_type"]),
        payload=dict(_mapping(data, "payload", default={})),
        priority=TaskPriority(int(data.get("priority", 50))),
        timeout_s=float(data["timeout_s"]) if data.get("timeout_s") is not None else None,
        source=str(data.get("source", "host")),
        id=str(data["id"]),
        created_at=_parse_dt(data.get("created_at")),
    )


def _message_to_dict(message: Message) -> dict[str, Any]:
    return {
        "role": message.role,
        "content": message.content,
        "tool_calls": [
            {"call_id": call.call_id, "name": call.name, "arguments": copy.deepcopy(call.arguments)}
            for call in message.tool_calls
        ],
        "tool_result": None if message.tool_result is None else {
            "call_id": message.tool_result.call_id,
            "name": message.tool_result.name,
            "output": message.tool_result.output,
            "is_error": message.tool_result.is_error,
        },
    }


def _message_from_dict(data: Mapping[str, Any]) -> Message:
    raw_calls = data.get("tool_calls", [])
    if not isinstance(raw_calls, list):
        raise ValueError("tool_calls must be a list")
    calls = tuple(
        ToolCall(str(item["call_id"]), str(item["name"]), dict(_mapping(item, "arguments", default={})))
        for item in raw_calls
        if isinstance(item, Mapping)
    )
    raw_result = data.get("tool_result")
    result = None
    if isinstance(raw_result, Mapping):
        result = ToolResult(
            call_id=str(raw_result["call_id"]),
            name=str(raw_result["name"]),
            output=str(raw_result["output"]),
            is_error=bool(raw_result.get("is_error", False)),
        )
    return Message(role=str(data["role"]), content=str(data.get("content", "")), tool_calls=calls, tool_result=result)  # type: ignore[arg-type]


def _snapshot_to_dict(snapshot: TaskSnapshot) -> dict[str, Any]:
    return {
        "revision": snapshot.revision,
        "captured_at": snapshot.captured_at.isoformat(),
        "character_id": snapshot.character_id,
        "character_name": snapshot.character_name,
        "state": {
            "emotion": snapshot.state.emotion,
            "energy": snapshot.state.energy,
            "trust": snapshot.state.trust,
            "favorability": snapshot.state.favorability,
            "relationship_stage": snapshot.state.relationship_stage,
            "custom": copy.deepcopy(dict(snapshot.state.custom)),
        },
        "history": [_message_to_dict(message) for message in snapshot.history],
        "memory_scope_id": snapshot.memory_scope_id,
    }


def _snapshot_from_dict(data: Mapping[str, Any]) -> TaskSnapshot:
    state = _mapping(data, "state")
    raw_history = data.get("history", [])
    if not isinstance(raw_history, list):
        raise ValueError("history must be a list")
    return TaskSnapshot(
        revision=int(data["revision"]),
        captured_at=_parse_dt(data.get("captured_at")),
        character_id=str(data["character_id"]),
        character_name=str(data["character_name"]),
        state=TaskStateSnapshot(
            emotion=str(state["emotion"]),
            energy=float(state["energy"]),
            trust=float(state["trust"]),
            favorability=float(state["favorability"]),
            relationship_stage=str(state["relationship_stage"]),
            custom=dict(_mapping(state, "custom", default={})),
        ),
        history=tuple(_message_from_dict(item) for item in raw_history if isinstance(item, Mapping)),
        memory_scope_id=str(data["memory_scope_id"]),
    )


def _proposal_to_dict(proposal: TaskProposal) -> dict[str, Any]:
    return {
        "target": proposal.target,
        "payload": copy.deepcopy(dict(proposal.payload)),
        "base_revision": proposal.base_revision,
        "source_task_id": proposal.source_task_id,
        "confidence": proposal.confidence,
        "provenance": copy.deepcopy(dict(proposal.provenance)),
        "created_at": proposal.created_at.isoformat(),
        "id": proposal.id,
    }


def _proposal_from_dict(data: Mapping[str, Any]) -> TaskProposal:
    return TaskProposal(
        target=str(data["target"]),
        payload=dict(_mapping(data, "payload", default={})),
        base_revision=int(data["base_revision"]),
        source_task_id=str(data["source_task_id"]) if data.get("source_task_id") is not None else None,
        confidence=float(data["confidence"]) if data.get("confidence") is not None else None,
        provenance=dict(_mapping(data, "provenance", default={})),
        created_at=_parse_dt(data.get("created_at")),
        id=str(data["id"]),
    )


def _output_to_dict(output: TaskOutput) -> dict[str, Any]:
    return {
        "value": copy.deepcopy(output.value),
        "proposals": [_proposal_to_dict(proposal) for proposal in output.proposals],
        "metadata": copy.deepcopy(dict(output.metadata)),
    }


def _output_from_dict(data: Mapping[str, Any]) -> TaskOutput:
    raw_proposals = data.get("proposals", [])
    if not isinstance(raw_proposals, list):
        raise ValueError("proposals must be a list")
    return TaskOutput(
        value=copy.deepcopy(data.get("value")),
        proposals=tuple(_proposal_from_dict(item) for item in raw_proposals if isinstance(item, Mapping)),
        metadata=dict(_mapping(data, "metadata", default={})),
    )
