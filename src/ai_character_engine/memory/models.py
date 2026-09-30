from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

_MEMORY_STATUSES = {"active", "superseded", "forgotten"}


def _clamp_importance(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(slots=True, frozen=True)
class MemoryRecord:
    """One persistent character memory.

    Working memory is revision-aware. A record may remain stored for audit and
    history while no longer being eligible for retrieval.
    """

    character_id: str
    summary: str
    importance: float = 0.5
    kind: str = "event"
    tags: tuple[str, ...] = field(default_factory=tuple)
    source_event_id: str | None = None
    source_event_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    status: str = "active"
    supersedes: tuple[str, ...] = field(default_factory=tuple)
    superseded_by: str | None = None
    forgotten_at: datetime | None = None
    id: str = field(default_factory=lambda: uuid4().hex)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Appended last to keep positional construction stable. Metadata is provider-neutral.
    embedding: tuple[float, ...] | None = None
    embedding_metadata: dict[str, Any] = field(default_factory=dict)
    vector_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.character_id.strip():
            raise ValueError("character_id must not be empty")
        if not self.summary.strip():
            raise ValueError("memory summary must not be empty")
        if not self.kind.strip():
            raise ValueError("memory kind must not be empty")
        if self.status not in _MEMORY_STATUSES:
            raise ValueError(f"invalid memory status: {self.status}")
        object.__setattr__(self, "importance", _clamp_importance(self.importance))
        if self.embedding is not None:
            import math
            vector = tuple(float(value) for value in self.embedding)
            if not vector or not all(math.isfinite(value) for value in vector):
                raise ValueError("embedding must be a non-empty finite vector")
            object.__setattr__(self, "embedding", vector)

    @property
    def is_active(self) -> bool:
        return self.status == "active"

    @property
    def evidence_type(self) -> str:
        """Provider-neutral provenance class stored in metadata.

        Records persisted before provenance was recorded default to
        ``unknown``; retrieval can still infer a conservative class from their
        original source text.
        """

        value = self.metadata.get("evidence_type", "unknown")
        return str(value) if value else "unknown"


@dataclass(slots=True, frozen=True)
class RetrievedMemory:
    """A memory selected for the current inference context."""

    record: MemoryRecord
    score: float
