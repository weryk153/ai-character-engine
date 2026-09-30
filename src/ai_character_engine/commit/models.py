from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping


class CommitStatus(str, Enum):
    COMMITTED = "committed"
    REJECTED = "rejected"
    STALE = "stale"
    CONFLICT = "conflict"
    DUPLICATE = "duplicate"
    RETRYABLE_ERROR = "retryable_error"
    REVIEW_REQUIRED = "review_required"


class CommitNextAction(str, Enum):
    NONE = "none"
    REBASE = "rebase"
    RERUN = "rerun"
    RETRY = "retry"
    MANUAL_REVIEW = "manual_review"


class StalePolicy(str, Enum):
    REJECT = "reject"
    ALLOW_MANUAL_REBASE = "allow_manual_rebase"
    RERUN = "rerun"


@dataclass(slots=True, frozen=True)
class CommitTargetPolicy:
    min_confidence: float = 0.0
    stale_policy: StalePolicy = StalePolicy.REJECT
    expected_worker_kind: str | None = None
    max_age_s: float | None = 300.0
    require_source_task_id: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.min_confidence <= 1:
            raise ValueError("min_confidence must be between 0 and 1")
        if self.max_age_s is not None and self.max_age_s <= 0:
            raise ValueError("max_age_s must be > 0 when provided")


@dataclass(slots=True, frozen=True)
class CommitResult:
    proposal_id: str
    target: str
    status: CommitStatus
    reason: str
    base_revision: int
    current_revision: int
    next_action: CommitNextAction = CommitNextAction.NONE
    commit_sequence: int | None = None
    applied_record_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    decided_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def committed(self) -> bool:
        return self.status is CommitStatus.COMMITTED


@dataclass(slots=True, frozen=True)
class CommitLifecycleEvent:
    proposal_id: str
    target: str
    status: CommitStatus
    reason: str
    base_revision: int
    current_revision: int
    occurred_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    commit_sequence: int | None = None
