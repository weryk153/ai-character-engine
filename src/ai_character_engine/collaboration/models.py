from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping
from uuid import uuid4

from ai_character_engine.cognition.models import CognitiveRole, CognitiveRouteRequirements


class SpecialistKind(str, Enum):
    """Built-in cognitive specialist classes.

    These are deliberately bounded character-cognition roles, not arbitrary agents.
    Hosts may register additional specialists, but the collaboration runtime still
    enforces the same one-plan / one-fanout / one-verifier topology.
    """

    MEMORY = "memory"
    VISION = "vision"
    TOOL = "tool"


class SpecialistRunStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


class VerificationDecision(str, Enum):
    ACCEPT = "accept"
    PARTIAL = "partial"
    REJECT = "reject"
    UNVERIFIED = "unverified"


@dataclass(frozen=True, slots=True)
class CollaborationSource:
    id: str
    source_type: str
    content: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in (("id", self.id), ("source_type", self.source_type), ("content", self.content)):
            if not str(value).strip():
                raise ValueError(f"collaboration source {name} must not be empty")
        object.__setattr__(self, "metadata", MappingProxyType(copy.deepcopy(dict(self.metadata))))

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_type": self.source_type,
            "content": self.content,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class SpecialistSpec:
    specialist_id: str
    kind: SpecialistKind
    role: CognitiveRole
    purpose: str
    system_prompt: str
    requirements: CognitiveRouteRequirements | None = None

    def __post_init__(self) -> None:
        if not self.specialist_id.strip():
            raise ValueError("specialist_id must not be empty")
        if not self.purpose.strip():
            raise ValueError("specialist purpose must not be empty")
        if not self.system_prompt.strip():
            raise ValueError("specialist system_prompt must not be empty")


@dataclass(frozen=True, slots=True)
class CognitiveWorkItem:
    item_id: str
    specialist_id: str
    instruction: str
    source_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.item_id.strip():
            raise ValueError("work item id must not be empty")
        if not self.specialist_id.strip():
            raise ValueError("work item specialist_id must not be empty")
        if not self.instruction.strip():
            raise ValueError("work item instruction must not be empty")
        if len(set(self.source_ids)) != len(self.source_ids):
            raise ValueError("work item source_ids must be unique")
        if any(not source_id.strip() for source_id in self.source_ids):
            raise ValueError("work item source_ids must not contain empty ids")


@dataclass(frozen=True, slots=True)
class CognitiveWorkPlan:
    objective: str
    rationale: str
    work_items: tuple[CognitiveWorkItem, ...]
    base_revision: int
    planner_model: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.objective.strip():
            raise ValueError("collaboration objective must not be empty")
        if not self.rationale.strip():
            raise ValueError("collaboration rationale must not be empty")
        if self.base_revision < 0:
            raise ValueError("base_revision must be >= 0")
        ids = [item.item_id for item in self.work_items]
        if len(set(ids)) != len(ids):
            raise ValueError("work item ids must be unique")
        object.__setattr__(self, "metadata", MappingProxyType(copy.deepcopy(dict(self.metadata))))


@dataclass(frozen=True, slots=True)
class SpecialistFinding:
    work_item_id: str
    specialist_id: str
    kind: SpecialistKind
    status: SpecialistRunStatus
    summary: str | None = None
    confidence: float | None = None
    evidence_source_ids: tuple[str, ...] = ()
    recommendations: tuple[str, ...] = ()
    model: str | None = None
    error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.work_item_id.strip() or not self.specialist_id.strip():
            raise ValueError("specialist finding ids must not be empty")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("specialist confidence must be between 0 and 1")
        if self.status is SpecialistRunStatus.SUCCEEDED and not (self.summary or "").strip():
            raise ValueError("successful specialist finding requires a summary")
        if self.status is not SpecialistRunStatus.SUCCEEDED and not (self.error or "").strip():
            raise ValueError("failed/timed-out specialist finding requires an error")
        object.__setattr__(self, "metadata", MappingProxyType(copy.deepcopy(dict(self.metadata))))


@dataclass(frozen=True, slots=True)
class VerificationIssue:
    code: str
    message: str
    work_item_id: str | None = None

    def __post_init__(self) -> None:
        if not self.code.strip() or not self.message.strip():
            raise ValueError("verification issue code/message must not be empty")


@dataclass(frozen=True, slots=True)
class VerificationReport:
    decision: VerificationDecision
    accepted_work_item_ids: tuple[str, ...]
    issues: tuple[VerificationIssue, ...]
    summary: str
    confidence: float | None = None
    model: str | None = None

    def __post_init__(self) -> None:
        if not self.summary.strip():
            raise ValueError("verification summary must not be empty")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("verification confidence must be between 0 and 1")
        if len(set(self.accepted_work_item_ids)) != len(self.accepted_work_item_ids):
            raise ValueError("accepted work item ids must be unique")
        if self.decision in {VerificationDecision.REJECT, VerificationDecision.UNVERIFIED} and self.accepted_work_item_ids:
            raise ValueError("rejected/unverified collaboration cannot accept work items")


@dataclass(frozen=True, slots=True)
class CollaborationResult:
    objective: str
    plan: CognitiveWorkPlan
    findings: tuple[SpecialistFinding, ...]
    verification: VerificationReport
    base_revision: int
    collaboration_id: str = field(default_factory=lambda: uuid4().hex)

    def __post_init__(self) -> None:
        if not self.objective.strip() or not self.collaboration_id.strip():
            raise ValueError("collaboration result ids/objective must not be empty")
        if self.base_revision < 0:
            raise ValueError("base_revision must be >= 0")

    @property
    def accepted_findings(self) -> tuple[SpecialistFinding, ...]:
        accepted = set(self.verification.accepted_work_item_ids)
        return tuple(
            finding
            for finding in self.findings
            if finding.status is SpecialistRunStatus.SUCCEEDED and finding.work_item_id in accepted
        )

    def is_stale(self, current_revision: int) -> bool:
        if current_revision < 0:
            raise ValueError("current_revision must be >= 0")
        return current_revision != self.base_revision


@dataclass(frozen=True, slots=True)
class SpecialistCollaborationConfig:
    max_work_items: int = 4
    max_parallel_specialists: int = 4
    planner_timeout_s: float = 20.0
    specialist_timeout_s: float = 25.0
    verifier_timeout_s: float = 20.0
    max_sources: int = 32
    max_source_chars: int = 2_000
    history_sources: int = 8
    require_verifier: bool = True

    def __post_init__(self) -> None:
        if not 1 <= self.max_work_items <= 12:
            raise ValueError("max_work_items must be between 1 and 12")
        if not 1 <= self.max_parallel_specialists <= 12:
            raise ValueError("max_parallel_specialists must be between 1 and 12")
        for name, value in (
            ("planner_timeout_s", self.planner_timeout_s),
            ("specialist_timeout_s", self.specialist_timeout_s),
            ("verifier_timeout_s", self.verifier_timeout_s),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be > 0")
        if self.max_sources < 1:
            raise ValueError("max_sources must be >= 1")
        if self.max_source_chars < 64:
            raise ValueError("max_source_chars must be >= 64")
        if self.history_sources < 0:
            raise ValueError("history_sources must be >= 0")
