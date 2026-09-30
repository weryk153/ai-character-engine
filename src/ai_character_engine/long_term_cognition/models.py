from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping
from uuid import uuid4


SAFE_BELIEF_EVIDENCE_TYPES = frozenset(
    {
        "asserted_fact",
        "event_observation",
        "conversation_summary",
    }
)


class ReflectionStatus(str, Enum):
    PROVISIONAL = "provisional"
    RETIRED = "retired"


class BeliefStatus(str, Enum):
    ACTIVE = "active"
    CONTESTED = "contested"
    RETIRED = "retired"


def _clean_required(value: str, *, field_name: str) -> str:
    cleaned = str(value).strip()
    if not cleaned:
        raise ValueError(f"{field_name} must not be empty")
    return cleaned


def _confidence(value: float, *, field_name: str) -> float:
    numeric = float(value)
    if not 0 <= numeric <= 1:
        raise ValueError(f"{field_name} must be between 0 and 1")
    return numeric


def _normalize(value: str) -> str:
    return " ".join(value.casefold().split())


@dataclass(frozen=True, slots=True)
class CognitionEvidenceRef:
    """A traceable reference to evidence used by one reflection.

    The record stores a compact excerpt plus provenance; it does not duplicate an
    entire conversation transcript.  Independence is defined by
    ``(source_type, source_id)`` so multiple excerpts from one event still count as
    one source during belief consolidation.
    """

    source_type: str
    source_id: str
    evidence_type: str
    excerpt: str
    confidence: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_type", _clean_required(self.source_type, field_name="source_type"))
        object.__setattr__(self, "source_id", _clean_required(self.source_id, field_name="source_id"))
        object.__setattr__(self, "evidence_type", _clean_required(self.evidence_type, field_name="evidence_type"))
        object.__setattr__(self, "excerpt", _clean_required(self.excerpt, field_name="evidence excerpt"))
        if self.confidence is not None:
            object.__setattr__(
                self,
                "confidence",
                _confidence(self.confidence, field_name="evidence confidence"),
            )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def independence_key(self) -> tuple[str, str]:
        return (_normalize(self.source_type), _normalize(self.source_id))

    @property
    def safe_for_belief(self) -> bool:
        return _normalize(self.evidence_type) in SAFE_BELIEF_EVIDENCE_TYPES


@dataclass(frozen=True, slots=True)
class BeliefClaim:
    """Structured claim used for deterministic grouping and conflict detection."""

    subject: str
    predicate: str
    object: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "subject", _clean_required(self.subject, field_name="belief subject"))
        object.__setattr__(self, "predicate", _clean_required(self.predicate, field_name="belief predicate"))
        object.__setattr__(self, "object", _clean_required(self.object, field_name="belief object"))

    @property
    def key(self) -> tuple[str, str]:
        return (_normalize(self.subject), _normalize(self.predicate))

    @property
    def value_key(self) -> tuple[str, str, str]:
        return (*self.key, _normalize(self.object))


@dataclass(frozen=True, slots=True)
class ReflectionRecord:
    """Durable model interpretation that remains explicitly provisional."""

    character_id: str
    insight: str
    confidence: float
    evidence: tuple[CognitionEvidenceRef, ...] = field(default_factory=tuple)
    claim: BeliefClaim | None = None
    base_revision: int = 0
    source_proposal_id: str | None = None
    source_task_id: str | None = None
    status: ReflectionStatus = ReflectionStatus.PROVISIONAL
    metadata: Mapping[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid4().hex)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        object.__setattr__(self, "character_id", _clean_required(self.character_id, field_name="character_id"))
        object.__setattr__(self, "insight", _clean_required(self.insight, field_name="reflection insight"))
        object.__setattr__(self, "confidence", _confidence(self.confidence, field_name="reflection confidence"))
        if self.base_revision < 0:
            raise ValueError("base_revision must be >= 0")
        object.__setattr__(self, "evidence", tuple(self.evidence))
        if not isinstance(self.status, ReflectionStatus):
            object.__setattr__(self, "status", ReflectionStatus(str(self.status)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def is_provisional(self) -> bool:
        return self.status is ReflectionStatus.PROVISIONAL


@dataclass(frozen=True, slots=True)
class BeliefRecord:
    """A consolidated, revisable hypothesis backed by independent evidence."""

    character_id: str
    claim: BeliefClaim
    confidence: float
    evidence: tuple[CognitionEvidenceRef, ...]
    source_reflection_ids: tuple[str, ...]
    support_count: int
    status: BeliefStatus = BeliefStatus.ACTIVE
    metadata: Mapping[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid4().hex)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        object.__setattr__(self, "character_id", _clean_required(self.character_id, field_name="character_id"))
        object.__setattr__(self, "confidence", _confidence(self.confidence, field_name="belief confidence"))
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "source_reflection_ids", tuple(dict.fromkeys(self.source_reflection_ids)))
        if self.support_count < 0:
            raise ValueError("support_count must be >= 0")
        independent = {item.independence_key for item in self.evidence}
        if self.support_count != len(independent):
            raise ValueError("support_count must equal independent evidence source count")
        if not isinstance(self.status, BeliefStatus):
            object.__setattr__(self, "status", BeliefStatus(str(self.status)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def is_active(self) -> bool:
        return self.status is BeliefStatus.ACTIVE
