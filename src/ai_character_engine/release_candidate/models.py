from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from ai_character_engine.release import ReleaseTarget

RC_EVIDENCE_SCHEMA_VERSION = 1


class MatrixEvidenceStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"


class ReleaseCandidateGateStatus(str, Enum):
    PASS = "pass"
    BLOCKED = "blocked"
    FAIL = "fail"


@dataclass(frozen=True, slots=True)
class MatrixEvidence:
    platform: str
    python_version: str
    engine_version: str
    status: MatrixEvidenceStatus
    pytest_passed: bool
    compileall_passed: bool
    matrix_self_check_passed: bool
    test_summary: str = ""
    source: str = "ci"
    schema_version: int = RC_EVIDENCE_SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != RC_EVIDENCE_SCHEMA_VERSION:
            raise ValueError(f"unsupported RC evidence schema: {self.schema_version}")
        if any(type(v) is not bool for v in (self.pytest_passed, self.compileall_passed, self.matrix_self_check_passed)):
            raise ValueError("check results must be JSON booleans")
        target = ReleaseTarget(self.platform, self.python_version)
        object.__setattr__(self, "platform", target.platform)
        object.__setattr__(self, "python_version", target.python_version)
        if not self.engine_version.strip():
            raise ValueError("engine_version is required")
        if not self.source.strip():
            raise ValueError("source is required")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def target(self) -> ReleaseTarget:
        return ReleaseTarget(self.platform, self.python_version)

    @property
    def passed(self) -> bool:
        return (
            self.status is MatrixEvidenceStatus.PASS
            and self.pytest_passed
            and self.compileall_passed
            and self.matrix_self_check_passed
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "platform": self.platform,
            "python_version": self.python_version,
            "engine_version": self.engine_version,
            "status": self.status.value,
            "pytest_passed": self.pytest_passed,
            "compileall_passed": self.compileall_passed,
            "matrix_self_check_passed": self.matrix_self_check_passed,
            "test_summary": self.test_summary,
            "source": self.source,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MatrixEvidence":
        metadata = value.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("matrix evidence metadata must be an object")
        return cls(
            platform=str(value["platform"]),
            python_version=str(value["python_version"]),
            engine_version=str(value["engine_version"]),
            status=MatrixEvidenceStatus(str(value["status"])),
            pytest_passed=value["pytest_passed"],
            compileall_passed=value["compileall_passed"],
            matrix_self_check_passed=value["matrix_self_check_passed"],
            test_summary=str(value.get("test_summary", "")),
            source=str(value.get("source", "ci")),
            schema_version=int(value.get("schema_version", 0)),
            metadata=dict(metadata),
        )


@dataclass(frozen=True, slots=True)
class ReleaseCandidateGate:
    name: str
    status: ReleaseCandidateGateStatus
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "status": self.status.value, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class ReleaseCandidateReport:
    engine_version: str
    gates: tuple[ReleaseCandidateGate, ...]
    matrix_evidence: tuple[MatrixEvidence, ...] = ()
    schema_version: int = RC_EVIDENCE_SCHEMA_VERSION
    candidate_sha256: str = ""
    acceptance_evidence: tuple[Mapping[str, Any], ...] = ()

    @property
    def ready_for_v1(self) -> bool:
        return all(gate.status is ReleaseCandidateGateStatus.PASS for gate in self.gates)

    @property
    def blockers(self) -> tuple[ReleaseCandidateGate, ...]:
        return tuple(gate for gate in self.gates if gate.status is not ReleaseCandidateGateStatus.PASS)

    @property
    def failures(self) -> tuple[ReleaseCandidateGate, ...]:
        return tuple(gate for gate in self.gates if gate.status is ReleaseCandidateGateStatus.FAIL)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "engine_version": self.engine_version,
            "ready_for_v1": self.ready_for_v1,
            "candidate_sha256": self.candidate_sha256,
            "acceptance_evidence": [dict(item) for item in self.acceptance_evidence],
            "gates": [gate.to_dict() for gate in self.gates],
            "matrix_evidence": [item.to_dict() for item in self.matrix_evidence],
        }
