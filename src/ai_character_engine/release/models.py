from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

RELEASE_MANIFEST_SCHEMA_VERSION = 1


class DistributionArtifactKind(str, Enum):
    WHEEL = "wheel"
    SDIST = "sdist"


class ReleaseCheckStatus(str, Enum):
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"


@dataclass(frozen=True, slots=True)
class ReleaseTarget:
    platform: str
    python_version: str

    def __post_init__(self) -> None:
        platform = self.platform.strip().lower()
        python_version = self.python_version.strip()
        if platform not in {"linux", "macos", "windows"}:
            raise ValueError(f"unsupported release platform: {self.platform!r}")
        parts = python_version.split(".")
        if len(parts) != 2 or not all(part.isdigit() for part in parts):
            raise ValueError("python_version must use major.minor form")
        object.__setattr__(self, "platform", platform)
        object.__setattr__(self, "python_version", python_version)

    def to_dict(self) -> dict[str, str]:
        return {"platform": self.platform, "python_version": self.python_version}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReleaseTarget":
        return cls(str(value["platform"]), str(value["python_version"]))


@dataclass(frozen=True, slots=True)
class ReleaseMatrix:
    targets: tuple[ReleaseTarget, ...]

    def __post_init__(self) -> None:
        if not self.targets:
            raise ValueError("release matrix requires at least one target")
        keys = [(target.platform, target.python_version) for target in self.targets]
        if len(keys) != len(set(keys)):
            raise ValueError("release matrix targets must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {"targets": [target.to_dict() for target in self.targets]}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReleaseMatrix":
        raw = value.get("targets")
        if not isinstance(raw, list):
            raise ValueError("release matrix targets must be a list")
        return cls(tuple(ReleaseTarget.from_dict(item) for item in raw))


@dataclass(frozen=True, slots=True)
class DistributionArtifact:
    path: str
    filename: str
    kind: DistributionArtifactKind
    distribution: str
    version: str
    sha256: str
    size_bytes: int
    requires_python: str | None = None
    wheel_tags: tuple[str, ...] = ()
    root_is_purelib: bool | None = None
    members: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.filename:
            raise ValueError("artifact filename is required")
        if not self.distribution:
            raise ValueError("artifact distribution is required")
        if not self.version:
            raise ValueError("artifact version is required")
        if len(self.sha256) != 64:
            raise ValueError("artifact sha256 must contain 64 hex characters")
        try:
            int(self.sha256, 16)
        except ValueError as exc:
            raise ValueError("artifact sha256 must be hexadecimal") from exc
        if self.size_bytes <= 0:
            raise ValueError("artifact size must be positive")

    @property
    def portable_pure_python_wheel(self) -> bool:
        if self.kind is not DistributionArtifactKind.WHEEL:
            return False
        return bool(self.root_is_purelib) and bool(self.wheel_tags) and all(
            tag.endswith("-none-any") for tag in self.wheel_tags
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "filename": self.filename,
            "kind": self.kind.value,
            "distribution": self.distribution,
            "version": self.version,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "requires_python": self.requires_python,
            "wheel_tags": list(self.wheel_tags),
            "root_is_purelib": self.root_is_purelib,
            "members": list(self.members),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DistributionArtifact":
        return cls(
            path=str(value["path"]),
            filename=str(value["filename"]),
            kind=DistributionArtifactKind(str(value["kind"])),
            distribution=str(value["distribution"]),
            version=str(value["version"]),
            sha256=str(value["sha256"]),
            size_bytes=int(value["size_bytes"]),
            requires_python=(None if value.get("requires_python") is None else str(value["requires_python"])),
            wheel_tags=tuple(str(item) for item in value.get("wheel_tags", ())),
            root_is_purelib=(None if value.get("root_is_purelib") is None else bool(value["root_is_purelib"])),
            members=tuple(str(item) for item in value.get("members", ())),
        )


@dataclass(frozen=True, slots=True)
class ReleaseCheck:
    name: str
    status: ReleaseCheckStatus
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "status": self.status.value, "detail": self.detail}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReleaseCheck":
        return cls(str(value["name"]), ReleaseCheckStatus(str(value["status"])), str(value["detail"]))


@dataclass(frozen=True, slots=True)
class ReleaseReadinessReport:
    checks: tuple[ReleaseCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.status is not ReleaseCheckStatus.FAIL for check in self.checks)

    @property
    def warnings(self) -> tuple[ReleaseCheck, ...]:
        return tuple(check for check in self.checks if check.status is ReleaseCheckStatus.WARNING)

    @property
    def failures(self) -> tuple[ReleaseCheck, ...]:
        return tuple(check for check in self.checks if check.status is ReleaseCheckStatus.FAIL)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": [check.to_dict() for check in self.checks],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReleaseReadinessReport":
        raw = value.get("checks")
        if not isinstance(raw, list):
            raise ValueError("release report checks must be a list")
        return cls(tuple(ReleaseCheck.from_dict(item) for item in raw))


@dataclass(frozen=True, slots=True)
class ReleaseManifest:
    engine_version: str
    python_requires: str
    matrix: ReleaseMatrix
    artifacts: tuple[DistributionArtifact, ...]
    contract_versions: Mapping[str, int]
    schema_version: int = RELEASE_MANIFEST_SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != RELEASE_MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"unsupported release manifest schema: {self.schema_version}")
        if not self.engine_version:
            raise ValueError("engine version is required")
        if not self.python_requires:
            raise ValueError("python requires is required")
        object.__setattr__(self, "contract_versions", MappingProxyType(dict(self.contract_versions)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "engine_version": self.engine_version,
            "python_requires": self.python_requires,
            "matrix": self.matrix.to_dict(),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "contract_versions": dict(self.contract_versions),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReleaseManifest":
        raw_artifacts = value.get("artifacts")
        if not isinstance(raw_artifacts, list):
            raise ValueError("release manifest artifacts must be a list")
        contracts = value.get("contract_versions")
        if not isinstance(contracts, Mapping):
            raise ValueError("release manifest contract_versions must be an object")
        metadata = value.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("release manifest metadata must be an object")
        return cls(
            engine_version=str(value["engine_version"]),
            python_requires=str(value["python_requires"]),
            matrix=ReleaseMatrix.from_dict(value["matrix"]),
            artifacts=tuple(DistributionArtifact.from_dict(item) for item in raw_artifacts),
            contract_versions={str(key): int(version) for key, version in contracts.items()},
            schema_version=int(value.get("schema_version", 0)),
            metadata=dict(metadata),
        )
