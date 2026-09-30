from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any


class OperationStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REJECTED = "rejected"
    REVIEW_REQUIRED = "review_required"
    DEGRADED = "degraded"


class FailureClass(str, Enum):
    TRANSIENT = "transient"
    PERMANENT = "permanent"
    AMBIGUOUS = "ambiguous"
    OVERLOADED = "overloaded"


class Idempotency(str, Enum):
    SAFE = "safe"
    UNSAFE = "unsafe"
    UNKNOWN = "unknown"


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class DegradedMode(str, Enum):
    NORMAL = "normal"
    CONSERVATIVE = "conservative"
    READ_ONLY = "read_only"


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 3
    base_backoff_s: float = 0.25
    max_backoff_s: float = 5.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.base_backoff_s < 0:
            raise ValueError("base_backoff_s must be >= 0")
        if self.max_backoff_s <= 0:
            raise ValueError("max_backoff_s must be > 0")
        if self.base_backoff_s > self.max_backoff_s:
            raise ValueError("base_backoff_s must be <= max_backoff_s")

    def backoff_s(self, failed_attempt: int) -> float:
        if failed_attempt < 1:
            raise ValueError("failed_attempt must be >= 1")
        return min(self.max_backoff_s, self.base_backoff_s * (2 ** (failed_attempt - 1)))


@dataclass(frozen=True, slots=True)
class CircuitBreakerPolicy:
    failure_threshold: int = 5
    recovery_timeout_s: float = 30.0

    def __post_init__(self) -> None:
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if self.recovery_timeout_s <= 0:
            raise ValueError("recovery_timeout_s must be > 0")


@dataclass(frozen=True, slots=True)
class ResourceLimits:
    max_concurrent_operations: int = 8
    max_pending_operations: int = 64
    default_timeout_s: float = 30.0

    def __post_init__(self) -> None:
        if self.max_concurrent_operations < 1:
            raise ValueError("max_concurrent_operations must be >= 1")
        if self.max_pending_operations < 0:
            raise ValueError("max_pending_operations must be >= 0")
        if self.default_timeout_s <= 0:
            raise ValueError("default_timeout_s must be > 0")


@dataclass(frozen=True, slots=True)
class ProductionHardeningConfig:
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    circuit_breaker: CircuitBreakerPolicy = field(default_factory=CircuitBreakerPolicy)
    resources: ResourceLimits = field(default_factory=ResourceLimits)
    lifecycle_history: int = 1024

    def __post_init__(self) -> None:
        if self.lifecycle_history < 1:
            raise ValueError("lifecycle_history must be >= 1")


@dataclass(frozen=True, slots=True)
class OperationLifecycleEvent:
    operation_id: str
    operation_name: str
    status: str
    attempt: int
    at: datetime = field(default_factory=lambda: datetime.now(UTC))
    failure_class: FailureClass | None = None
    error_type: str | None = None
    dependency: str | None = None
    degraded_mode: DegradedMode = DegradedMode.NORMAL
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.operation_id.strip() or not self.operation_name.strip():
            raise ValueError("operation id/name must not be empty")
        if self.attempt < 0:
            raise ValueError("attempt must be >= 0")
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class OperationResult:
    operation_id: str
    operation_name: str
    status: OperationStatus
    attempts: int
    value: Any = None
    failure_class: FailureClass | None = None
    error_type: str | None = None
    error_message: str | None = None
    retryable: bool = False
    requires_review: bool = False
    degraded_mode: DegradedMode = DegradedMode.NORMAL
    dependency: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.status is OperationStatus.SUCCEEDED


@dataclass(frozen=True, slots=True)
class ProductionHealthSnapshot:
    degraded_mode: DegradedMode
    in_flight: int
    waiting: int
    completed: int
    failed: int
    rejected: int
    review_required: int
    open_circuits: tuple[str, ...]
    lifecycle_events: int


@dataclass(frozen=True, slots=True)
class EvaluationGatePolicy:
    min_quality_score: float = 0.75
    severe_violation_mode: DegradedMode = DegradedMode.READ_ONLY
    low_quality_mode: DegradedMode = DegradedMode.CONSERVATIVE

    def __post_init__(self) -> None:
        if not 0 <= self.min_quality_score <= 1:
            raise ValueError("min_quality_score must be within [0, 1]")


@dataclass(frozen=True, slots=True)
class EvaluationGateDecision:
    mode: DegradedMode
    reason: str
    quality_score: float | None
    violation_count: int
    should_alert: bool
