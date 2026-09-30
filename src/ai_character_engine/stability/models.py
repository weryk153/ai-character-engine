from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping


class SoakStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class SoakConfig:
    iterations: int = 1000
    concurrency: int = 1
    sample_every: int = 100
    max_failures: int = 0
    max_recorded_failures: int = 100
    fail_fast: bool = False

    def __post_init__(self) -> None:
        if self.iterations < 1:
            raise ValueError("iterations must be >= 1")
        if self.concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        if self.sample_every < 1:
            raise ValueError("sample_every must be >= 1")
        if self.max_failures < 0:
            raise ValueError("max_failures must be >= 0")
        if self.max_recorded_failures < 0:
            raise ValueError("max_recorded_failures must be >= 0")


@dataclass(frozen=True, slots=True)
class SoakSample:
    completed: int
    succeeded: int
    failed: int
    cancelled: int
    metrics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if min(self.completed, self.succeeded, self.failed, self.cancelled) < 0:
            raise ValueError("sample counters must be >= 0")
        if self.succeeded + self.failed + self.cancelled > self.completed:
            raise ValueError("sample outcome counts cannot exceed completed")
        cleaned: dict[str, float] = {}
        for key, value in self.metrics.items():
            name = str(key).strip()
            if not name:
                raise ValueError("metric names must not be empty")
            cleaned[name] = float(value)
        object.__setattr__(self, "metrics", MappingProxyType(cleaned))


@dataclass(frozen=True, slots=True)
class SoakFailure:
    iteration: int
    error_type: str
    message: str
    cancelled: bool = False

    def __post_init__(self) -> None:
        if self.iteration < 0:
            raise ValueError("iteration must be >= 0")
        if not self.error_type.strip():
            raise ValueError("error_type must not be empty")


@dataclass(frozen=True, slots=True)
class SoakViolation:
    code: str
    message: str
    metric: str | None = None
    observed: float | None = None
    limit: float | None = None

    def __post_init__(self) -> None:
        if not self.code.strip() or not self.message.strip():
            raise ValueError("violation code/message must not be empty")


@dataclass(frozen=True, slots=True)
class MetricInvariant:
    metric: str
    max_growth: float | None = None
    max_final: float | None = None

    def __post_init__(self) -> None:
        if not self.metric.strip():
            raise ValueError("metric must not be empty")
        if self.max_growth is None and self.max_final is None:
            raise ValueError("at least one invariant limit is required")
        if self.max_growth is not None and self.max_growth < 0:
            raise ValueError("max_growth must be >= 0")
        if self.max_final is not None and self.max_final < 0:
            raise ValueError("max_final must be >= 0")


@dataclass(frozen=True, slots=True)
class SoakReport:
    status: SoakStatus
    iterations: int
    succeeded: int
    failed: int
    cancelled: int
    samples: tuple[SoakSample, ...]
    failures: tuple[SoakFailure, ...]
    violations: tuple[SoakViolation, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def passed(self) -> bool:
        return self.status is SoakStatus.PASSED

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "iterations": self.iterations,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "cancelled": self.cancelled,
            "samples": [
                {
                    "completed": sample.completed,
                    "succeeded": sample.succeeded,
                    "failed": sample.failed,
                    "cancelled": sample.cancelled,
                    "metrics": dict(sample.metrics),
                }
                for sample in self.samples
            ],
            "failures": [
                {
                    "iteration": failure.iteration,
                    "error_type": failure.error_type,
                    "message": failure.message,
                    "cancelled": failure.cancelled,
                }
                for failure in self.failures
            ],
            "violations": [
                {
                    "code": item.code,
                    "message": item.message,
                    "metric": item.metric,
                    "observed": item.observed,
                    "limit": item.limit,
                }
                for item in self.violations
            ],
            "metadata": dict(self.metadata),
        }
