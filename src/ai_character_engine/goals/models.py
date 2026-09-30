from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping
from uuid import uuid4


def _required(value: str, *, field_name: str) -> str:
    cleaned = " ".join(str(value).strip().split())
    if not cleaned:
        raise ValueError(f"{field_name} must not be empty")
    return cleaned


def _unit(value: float, *, field_name: str) -> float:
    numeric = float(value)
    if not 0 <= numeric <= 1:
        raise ValueError(f"{field_name} must be between 0 and 1")
    return numeric


def _normalize(value: str) -> str:
    return " ".join(str(value).casefold().split())


class GoalHorizon(str, Enum):
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"


class GoalStatus(str, Enum):
    ACTIVE = "active"
    BLOCKED = "blocked"
    PAUSED = "paused"
    COMPLETED = "completed"
    RETIRED = "retired"


class MotivationKind(str, Enum):
    """Why an action direction currently matters.

    Motivation is action policy, not a source of world truth. Each signal still
    points back to one authoritative source so it can be audited or invalidated.
    """

    EXPLICIT_REQUEST = "explicit_request"
    UNFINISHED_INTENT = "unfinished_intent"
    COMMITMENT = "commitment"
    BELIEF_ALIGNMENT = "belief_alignment"
    STATE_PRESSURE = "state_pressure"


# One table for the worker prompt and the commit check, so the model is told
# exactly what the coordinator will accept.
MOTIVATION_SOURCE_TYPES: Mapping[MotivationKind, frozenset[str]] = MappingProxyType(
    {
        MotivationKind.EXPLICIT_REQUEST: frozenset({"event", "memory"}),
        MotivationKind.UNFINISHED_INTENT: frozenset({"event", "memory"}),
        MotivationKind.COMMITMENT: frozenset({"event", "memory"}),
        MotivationKind.BELIEF_ALIGNMENT: frozenset({"belief"}),
        MotivationKind.STATE_PRESSURE: frozenset({"state"}),
    }
)


@dataclass(frozen=True, slots=True)
class GoalEvidenceRef:
    """Canonical authoritative input used to justify a goal/motivation signal."""

    source_type: str
    source_id: str
    excerpt: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_type", _required(self.source_type, field_name="goal evidence source_type"))
        object.__setattr__(self, "source_id", _required(self.source_id, field_name="goal evidence source_id"))
        object.__setattr__(self, "excerpt", _required(self.excerpt, field_name="goal evidence excerpt"))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def independence_key(self) -> tuple[str, str]:
        return (_normalize(self.source_type), _normalize(self.source_id))


@dataclass(frozen=True, slots=True)
class MotivationSignal:
    """One bounded, traceable reason that increases current goal motivation."""

    kind: MotivationKind
    strength: float
    evidence: GoalEvidenceRef
    rationale: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, MotivationKind):
            object.__setattr__(self, "kind", MotivationKind(str(self.kind)))
        object.__setattr__(self, "strength", _unit(self.strength, field_name="motivation strength"))
        object.__setattr__(self, "rationale", _required(self.rationale, field_name="motivation rationale"))

    @property
    def key(self) -> tuple[str, tuple[str, str]]:
        return (self.kind.value, self.evidence.independence_key)


@dataclass(frozen=True, slots=True)
class GoalRecord:
    """A durable action intention derived from authoritative evidence.

    A GoalRecord is intentionally *not* a fact. It may influence what the
    character tries to do, but it must never be used as evidence that the world
    or user is a certain way.
    """

    character_id: str
    objective: str
    horizon: GoalHorizon
    urgency: float
    confidence: float
    motivation_signals: tuple[MotivationSignal, ...]
    status: GoalStatus = GoalStatus.ACTIVE
    conflict_key: str | None = None
    source_proposal_ids: tuple[str, ...] = field(default_factory=tuple)
    source_task_ids: tuple[str, ...] = field(default_factory=tuple)
    base_revisions: tuple[int, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid4().hex)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        object.__setattr__(self, "character_id", _required(self.character_id, field_name="character_id"))
        object.__setattr__(self, "objective", _required(self.objective, field_name="goal objective"))
        if not isinstance(self.horizon, GoalHorizon):
            object.__setattr__(self, "horizon", GoalHorizon(str(self.horizon)))
        object.__setattr__(self, "urgency", _unit(self.urgency, field_name="goal urgency"))
        object.__setattr__(self, "confidence", _unit(self.confidence, field_name="goal confidence"))
        signals = tuple(self.motivation_signals)
        if not signals:
            raise ValueError("goal requires at least one motivation signal")
        if len(signals) > 16:
            raise ValueError("goal supports at most 16 motivation signals")
        object.__setattr__(self, "motivation_signals", signals)
        if not isinstance(self.status, GoalStatus):
            object.__setattr__(self, "status", GoalStatus(str(self.status)))
        conflict_key = None if self.conflict_key is None else " ".join(str(self.conflict_key).strip().split())
        object.__setattr__(self, "conflict_key", conflict_key or None)
        object.__setattr__(self, "source_proposal_ids", tuple(dict.fromkeys(str(x) for x in self.source_proposal_ids if str(x).strip())))
        object.__setattr__(self, "source_task_ids", tuple(dict.fromkeys(str(x) for x in self.source_task_ids if str(x).strip())))
        revisions = tuple(dict.fromkeys(int(x) for x in self.base_revisions))
        if any(value < 0 for value in revisions):
            raise ValueError("base revisions must be >= 0")
        object.__setattr__(self, "base_revisions", revisions)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def semantic_key(self) -> tuple[str, str, str]:
        return (
            self.horizon.value,
            _normalize(self.objective),
            _normalize(self.conflict_key or ""),
        )

    @property
    def support_count(self) -> int:
        return len({signal.evidence.independence_key for signal in self.motivation_signals})

    @property
    def motivation_score(self) -> float:
        """Deterministically combine independent source strengths.

        Repeated signals from one source cannot inflate motivation. The strongest
        signal per authoritative source wins, then independent sources reinforce
        with a bounded noisy-OR composition.
        """

        strongest: dict[tuple[str, str], float] = {}
        for signal in self.motivation_signals:
            key = signal.evidence.independence_key
            strongest[key] = max(strongest.get(key, 0.0), signal.strength)
        remaining = 1.0
        for strength in strongest.values():
            remaining *= 1.0 - strength
        return max(0.0, min(1.0, 1.0 - remaining))

    @property
    def rank_score(self) -> float:
        return max(0.0, min(1.0, 0.65 * self.motivation_score + 0.35 * self.urgency))

    @property
    def is_active(self) -> bool:
        return self.status is GoalStatus.ACTIVE
