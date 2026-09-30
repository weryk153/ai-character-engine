from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping


PERSISTENCE_SCHEMA_CONTRACT_VERSION = 1
PERSISTENCE_FIXTURE_SCHEMA_VERSION = 1
SCHEMA_VERSION_FIELD = "_schema_version"
LEGACY_UNVERSIONED_SCHEMA_VERSION = 0


class PersistenceSurface(str, Enum):
    MEMORY = "memory"
    REFLECTION = "reflection"
    BELIEF = "belief"
    GOAL = "goal"
    WORLD_SNAPSHOT = "world_snapshot"
    WORLD_EVENT = "world_event"
    WORLD_OBSERVATION = "world_observation"
    SHARED_COGNITION = "shared_cognition"
    CHARACTER_EXCHANGE = "character_exchange"
    SESSION = "session"
    RELATIONSHIP = "relationship"


class PersistenceMigrationError(ValueError):
    """Base failure for deterministic persistence migration."""


class UnsupportedPersistenceSchemaError(PersistenceMigrationError):
    """Raised when a record is newer than this engine or a downgrade is requested."""


class PersistenceMigrationPathError(PersistenceMigrationError):
    """Raised when no complete deterministic migration chain exists."""


class PersistenceFixtureError(PersistenceMigrationError):
    """Raised when a replay fixture is malformed or cannot be verified."""


@dataclass(frozen=True, slots=True)
class PersistenceMigrationStep:
    surface: PersistenceSurface
    from_version: int
    to_version: int

    def __post_init__(self) -> None:
        if self.from_version < 0 or self.to_version < 0:
            raise ValueError("schema versions must be >= 0")
        if self.to_version <= self.from_version:
            raise ValueError("migration step must move forward")


@dataclass(frozen=True, slots=True)
class PersistenceMigrationResult:
    surface: PersistenceSurface
    source_version: int
    target_version: int
    payload: Mapping[str, Any]
    steps: tuple[PersistenceMigrationStep, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.source_version < 0 or self.target_version < 0:
            raise ValueError("schema versions must be >= 0")
        object.__setattr__(self, "payload", MappingProxyType(copy.deepcopy(dict(self.payload))))
        object.__setattr__(self, "steps", tuple(self.steps))

    @property
    def migrated(self) -> bool:
        return self.source_version != self.target_version

    def to_dict(self) -> dict[str, Any]:
        return {
            "surface": self.surface.value,
            "source_version": self.source_version,
            "target_version": self.target_version,
            "migrated": self.migrated,
            "steps": [
                {
                    "surface": step.surface.value,
                    "from_version": step.from_version,
                    "to_version": step.to_version,
                }
                for step in self.steps
            ],
            "payload": copy.deepcopy(dict(self.payload)),
        }


@dataclass(frozen=True, slots=True)
class PersistenceReplayFixture:
    fixture_id: str
    engine_version: str
    surface: PersistenceSurface
    source_schema_version: int
    payload: Mapping[str, Any]
    expected_payload: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    fixture_schema_version: int = PERSISTENCE_FIXTURE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        fixture_id = str(self.fixture_id).strip()
        engine_version = str(self.engine_version).strip()
        if not fixture_id:
            raise ValueError("fixture_id must not be empty")
        if not engine_version:
            raise ValueError("engine_version must not be empty")
        if self.source_schema_version < 0:
            raise ValueError("source_schema_version must be >= 0")
        if self.fixture_schema_version != PERSISTENCE_FIXTURE_SCHEMA_VERSION:
            raise PersistenceFixtureError(
                f"unsupported fixture schema version: {self.fixture_schema_version}"
            )
        object.__setattr__(self, "fixture_id", fixture_id)
        object.__setattr__(self, "engine_version", engine_version)
        object.__setattr__(self, "payload", MappingProxyType(copy.deepcopy(dict(self.payload))))
        object.__setattr__(
            self,
            "expected_payload",
            MappingProxyType(copy.deepcopy(dict(self.expected_payload))),
        )
        object.__setattr__(self, "metadata", MappingProxyType(copy.deepcopy(dict(self.metadata))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture_schema_version": self.fixture_schema_version,
            "fixture_id": self.fixture_id,
            "engine_version": self.engine_version,
            "surface": self.surface.value,
            "source_schema_version": self.source_schema_version,
            "payload": copy.deepcopy(dict(self.payload)),
            "expected_payload": copy.deepcopy(dict(self.expected_payload)),
            "metadata": copy.deepcopy(dict(self.metadata)),
        }


@dataclass(frozen=True, slots=True)
class PersistenceReplayResult:
    fixture_id: str
    surface: PersistenceSurface
    passed: bool
    expected_fingerprint: str
    actual_fingerprint: str
    issues: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "issues", tuple(str(item) for item in self.issues))

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture_id": self.fixture_id,
            "surface": self.surface.value,
            "passed": self.passed,
            "expected_fingerprint": self.expected_fingerprint,
            "actual_fingerprint": self.actual_fingerprint,
            "issues": list(self.issues),
        }
