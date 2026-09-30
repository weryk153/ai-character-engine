from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping


PERFORMANCE_REPORT_SCHEMA_VERSION = 1


class PerformanceStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    NOT_COMPARABLE = "not_comparable"


@dataclass(frozen=True, slots=True)
class BenchmarkEnvironment:
    profile_id: str
    python_version: str
    python_implementation: str
    platform_system: str
    machine: str
    cpu_count: int | None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.profile_id.strip():
            raise ValueError("profile_id must not be empty")
        if self.cpu_count is not None and self.cpu_count < 1:
            raise ValueError("cpu_count must be >= 1 when provided")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "python_version": self.python_version,
            "python_implementation": self.python_implementation,
            "platform_system": self.platform_system,
            "machine": self.machine,
            "cpu_count": self.cpu_count,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BenchmarkEnvironment":
        return cls(
            profile_id=str(value["profile_id"]),
            python_version=str(value.get("python_version", "")),
            python_implementation=str(value.get("python_implementation", "")),
            platform_system=str(value.get("platform_system", "")),
            machine=str(value.get("machine", "")),
            cpu_count=None if value.get("cpu_count") is None else int(value["cpu_count"]),
            metadata=dict(value.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    iterations: int = 100
    warmup_iterations: int = 10
    concurrency: int = 1
    max_failures: int = 0
    record_samples: bool = False

    def __post_init__(self) -> None:
        if self.iterations < 1:
            raise ValueError("iterations must be >= 1")
        if self.warmup_iterations < 0:
            raise ValueError("warmup_iterations must be >= 0")
        if self.concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        if self.max_failures < 0:
            raise ValueError("max_failures must be >= 0")


@dataclass(frozen=True, slots=True)
class BenchmarkSample:
    iteration: int
    duration_ns: int
    success: bool = True
    error_type: str | None = None

    def __post_init__(self) -> None:
        if self.iteration < 0:
            raise ValueError("iteration must be >= 0")
        if self.duration_ns < 0:
            raise ValueError("duration_ns must be >= 0")
        if self.success and self.error_type is not None:
            raise ValueError("successful sample cannot have error_type")
        if not self.success and not (self.error_type or "").strip():
            raise ValueError("failed sample requires error_type")


@dataclass(frozen=True, slots=True)
class BenchmarkMetrics:
    count: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    throughput_ops_s: float
    elapsed_ms: float
    peak_memory_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.count < 0:
            raise ValueError("count must be >= 0")
        for name in ("mean_ms", "median_ms", "p95_ms", "p99_ms", "min_ms", "max_ms", "throughput_ops_s", "elapsed_ms"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.peak_memory_bytes is not None and self.peak_memory_bytes < 0:
            raise ValueError("peak_memory_bytes must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "mean_ms": self.mean_ms,
            "median_ms": self.median_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "throughput_ops_s": self.throughput_ops_s,
            "elapsed_ms": self.elapsed_ms,
            "peak_memory_bytes": self.peak_memory_bytes,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BenchmarkMetrics":
        return cls(
            count=int(value["count"]),
            mean_ms=float(value["mean_ms"]),
            median_ms=float(value["median_ms"]),
            p95_ms=float(value["p95_ms"]),
            p99_ms=float(value["p99_ms"]),
            min_ms=float(value["min_ms"]),
            max_ms=float(value["max_ms"]),
            throughput_ops_s=float(value["throughput_ops_s"]),
            elapsed_ms=float(value["elapsed_ms"]),
            peak_memory_bytes=None if value.get("peak_memory_bytes") is None else int(value["peak_memory_bytes"]),
        )


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    name: str
    environment: BenchmarkEnvironment
    config: BenchmarkConfig
    metrics: BenchmarkMetrics
    failures: int = 0
    samples: tuple[BenchmarkSample, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = PERFORMANCE_REPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("benchmark name must not be empty")
        if self.failures < 0:
            raise ValueError("failures must be >= 0")
        if self.schema_version != PERFORMANCE_REPORT_SCHEMA_VERSION:
            raise ValueError("unsupported performance report schema version")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def passed(self) -> bool:
        return self.failures <= self.config.max_failures and self.metrics.count > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "environment": self.environment.to_dict(),
            "config": {
                "iterations": self.config.iterations,
                "warmup_iterations": self.config.warmup_iterations,
                "concurrency": self.config.concurrency,
                "max_failures": self.config.max_failures,
                "record_samples": self.config.record_samples,
            },
            "metrics": self.metrics.to_dict(),
            "failures": self.failures,
            "samples": [
                {
                    "iteration": item.iteration,
                    "duration_ns": item.duration_ns,
                    "success": item.success,
                    "error_type": item.error_type,
                }
                for item in self.samples
            ],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BenchmarkResult":
        config = value["config"]
        return cls(
            schema_version=int(value.get("schema_version", PERFORMANCE_REPORT_SCHEMA_VERSION)),
            name=str(value["name"]),
            environment=BenchmarkEnvironment.from_dict(value["environment"]),
            config=BenchmarkConfig(
                iterations=int(config["iterations"]),
                warmup_iterations=int(config.get("warmup_iterations", 0)),
                concurrency=int(config.get("concurrency", 1)),
                max_failures=int(config.get("max_failures", 0)),
                record_samples=bool(config.get("record_samples", False)),
            ),
            metrics=BenchmarkMetrics.from_dict(value["metrics"]),
            failures=int(value.get("failures", 0)),
            samples=tuple(
                BenchmarkSample(
                    iteration=int(item["iteration"]),
                    duration_ns=int(item["duration_ns"]),
                    success=bool(item.get("success", True)),
                    error_type=item.get("error_type"),
                )
                for item in value.get("samples", ())
            ),
            metadata=dict(value.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class PerformanceBudget:
    max_p95_ratio: float | None = 1.20
    max_mean_ratio: float | None = 1.20
    min_throughput_ratio: float | None = 0.80
    max_peak_memory_ratio: float | None = 1.25
    absolute_max_p95_ms: float | None = None
    absolute_min_throughput_ops_s: float | None = None
    environment_profile_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("max_p95_ratio", "max_mean_ratio", "min_throughput_ratio", "max_peak_memory_ratio"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be > 0")
        if self.absolute_max_p95_ms is not None and self.absolute_max_p95_ms < 0:
            raise ValueError("absolute_max_p95_ms must be >= 0")
        if self.absolute_min_throughput_ops_s is not None and self.absolute_min_throughput_ops_s < 0:
            raise ValueError("absolute_min_throughput_ops_s must be >= 0")
        if (self.absolute_max_p95_ms is not None or self.absolute_min_throughput_ops_s is not None) and not self.environment_profile_id:
            raise ValueError("absolute performance thresholds require environment_profile_id")


@dataclass(frozen=True, slots=True)
class PerformanceViolation:
    code: str
    message: str
    observed: float | None = None
    limit: float | None = None

    def __post_init__(self) -> None:
        if not self.code.strip() or not self.message.strip():
            raise ValueError("violation code/message must not be empty")


@dataclass(frozen=True, slots=True)
class PerformanceEvaluation:
    status: PerformanceStatus
    current_name: str
    baseline_name: str
    violations: tuple[PerformanceViolation, ...] = ()

    @property
    def passed(self) -> bool:
        return self.status is PerformanceStatus.PASSED

    @property
    def comparable(self) -> bool:
        return self.status is not PerformanceStatus.NOT_COMPARABLE

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "passed": self.passed,
            "comparable": self.comparable,
            "current_name": self.current_name,
            "baseline_name": self.baseline_name,
            "violations": [
                {"code": item.code, "message": item.message, "observed": item.observed, "limit": item.limit}
                for item in self.violations
            ],
        }
