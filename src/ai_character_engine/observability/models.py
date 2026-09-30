from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4


SpanStatus = Literal["ok", "error"]


@dataclass(frozen=True, slots=True)
class TraceContext:
    """Correlation identifiers propagated across service/runtime boundaries."""

    trace_id: str
    request_id: str | None = None
    session_id: str | None = None
    user_id: str | None = None
    character_id: str | None = None
    parent_span_id: str | None = None

    def __post_init__(self) -> None:
        if not self.trace_id.strip():
            raise ValueError("trace_id must not be empty")

    @classmethod
    def create(cls, **kwargs: Any) -> "TraceContext":
        return cls(trace_id=uuid4().hex, **kwargs)

    def child(self, *, parent_span_id: str | None = None, **updates: Any) -> "TraceContext":
        values = {
            "trace_id": self.trace_id,
            "request_id": self.request_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "character_id": self.character_id,
            "parent_span_id": parent_span_id if parent_span_id is not None else self.parent_span_id,
        }
        values.update(updates)
        return TraceContext(**values)


@dataclass(frozen=True, slots=True)
class SpanRecord:
    trace_id: str
    span_id: str
    name: str
    started_at: datetime
    ended_at: datetime
    duration_ms: float
    status: SpanStatus = "ok"
    parent_span_id: str | None = None
    request_id: str | None = None
    session_id: str | None = None
    user_id: str | None = None
    character_id: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    error_type: str | None = None
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class TurnTelemetry:
    trace_id: str
    event_id: str
    character_id: str
    duration_ms: float
    request_id: str | None = None
    session_id: str | None = None
    user_id: str | None = None
    model: str | None = None
    rounds: int = 1
    tool_count: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    model_latency_ms: float | None = None
    estimated_context_tokens: int | None = None
    retrieved_memory_count: int = 0
    memory_written: bool = False
    retrieval_strategy: str | None = None
    queue_wait_ms: float | None = None
    ttft_ms: float | None = None
    decode_tokens_per_second: float | None = None
    estimated_cost: float | None = None
    inference_profile: str | None = None


@dataclass(frozen=True, slots=True)
class ProductionEvalTelemetry:
    trace_id: str
    case_id: str
    character_id: str
    passed: bool | None
    consistency_score: float | None
    violation_count: int
    severity: str | None = None
    sampled: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
