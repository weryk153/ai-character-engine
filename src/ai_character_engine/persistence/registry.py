from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable, Mapping
from typing import Any

from .models import (
    LEGACY_UNVERSIONED_SCHEMA_VERSION,
    SCHEMA_VERSION_FIELD,
    PersistenceMigrationError,
    PersistenceMigrationPathError,
    PersistenceMigrationResult,
    PersistenceMigrationStep,
    PersistenceSurface,
    UnsupportedPersistenceSchemaError,
)

MigrationFunction = Callable[[dict[str, Any]], dict[str, Any]]


CURRENT_PERSISTENCE_SCHEMA_VERSIONS: Mapping[PersistenceSurface, int] = {
    surface: 1 for surface in PersistenceSurface
}

_RECORD_TYPES: Mapping[PersistenceSurface, str | None] = {
    PersistenceSurface.MEMORY: None,
    PersistenceSurface.REFLECTION: "reflection",
    PersistenceSurface.BELIEF: "belief",
    PersistenceSurface.GOAL: "goal",
    PersistenceSurface.WORLD_SNAPSHOT: "world_snapshot",
    PersistenceSurface.WORLD_EVENT: "world_event",
    PersistenceSurface.WORLD_OBSERVATION: "world_observation",
    PersistenceSurface.SHARED_COGNITION: "shared_cognition",
    PersistenceSurface.CHARACTER_EXCHANGE: "character_exchange",
    PersistenceSurface.SESSION: None,
    PersistenceSurface.RELATIONSHIP: None,
}

_REQUIRED_FIELDS: Mapping[PersistenceSurface, tuple[str, ...]] = {
    PersistenceSurface.MEMORY: ("id", "character_id", "summary", "created_at"),
    PersistenceSurface.REFLECTION: ("id", "character_id", "insight", "confidence", "created_at"),
    PersistenceSurface.BELIEF: (
        "id",
        "character_id",
        "claim",
        "confidence",
        "support_count",
        "created_at",
        "updated_at",
    ),
    PersistenceSurface.GOAL: (
        "id",
        "character_id",
        "objective",
        "horizon",
        "urgency",
        "confidence",
        "motivation_signals",
        "created_at",
        "updated_at",
    ),
    PersistenceSurface.WORLD_SNAPSHOT: ("revision", "values"),
    PersistenceSurface.WORLD_EVENT: (
        "id",
        "type",
        "base_revision",
        "revision",
        "created_at",
    ),
    PersistenceSurface.WORLD_OBSERVATION: (
        "id",
        "character_id",
        "world_event_id",
        "world_revision",
        "content",
        "created_at",
    ),
    PersistenceSurface.SHARED_COGNITION: (
        "id",
        "owner_character_id",
        "kind",
        "content",
        "visibility",
        "created_at",
    ),
    PersistenceSurface.CHARACTER_EXCHANGE: (
        "id",
        "sender_character_id",
        "recipient_character_id",
        "content",
        "created_at",
        "updated_at",
    ),
    PersistenceSurface.SESSION: (
        "id",
        "user_id",
        "character_id",
        "created_at",
        "last_activity",
    ),
    PersistenceSurface.RELATIONSHIP: (
        "user_id",
        "character_id",
        "updated_at",
    ),
}

_LEGACY_DEFAULTS: Mapping[PersistenceSurface, Mapping[str, Any]] = {
    PersistenceSurface.MEMORY: {
        "importance": 0.5,
        "kind": "event",
        "tags": [],
        "source_event_id": None,
        "source_event_type": None,
        "metadata": {},
        "embedding": None,
        "embedding_metadata": {},
        "vector_metadata": {},
        "status": "active",
        "supersedes": [],
        "superseded_by": None,
        "forgotten_at": None,
    },
    PersistenceSurface.REFLECTION: {
        "evidence": [],
        "claim": None,
        "base_revision": 0,
        "source_proposal_id": None,
        "source_task_id": None,
        "status": "provisional",
        "metadata": {},
    },
    PersistenceSurface.BELIEF: {
        "evidence": [],
        "source_reflection_ids": [],
        "status": "active",
        "metadata": {},
    },
    PersistenceSurface.GOAL: {
        "status": "active",
        "conflict_key": None,
        "source_proposal_ids": [],
        "source_task_ids": [],
        "base_revisions": [],
        "metadata": {},
    },
    PersistenceSurface.WORLD_SNAPSHOT: {},
    PersistenceSurface.WORLD_EVENT: {
        "content": "",
        "changes": [],
        "source_type": "host",
        "source_id": None,
        "perception_scope": "hidden",
        "observer_character_ids": [],
        "observable_keys": [],
        "metadata": {},
    },
    PersistenceSurface.WORLD_OBSERVATION: {
        "changes": [],
        "perception_policy": "explicit",
        "metadata": {},
    },
    PersistenceSurface.SHARED_COGNITION: {
        "audience_character_ids": [],
        "source_type": "host",
        "source_id": None,
        "metadata": {},
    },
    PersistenceSurface.CHARACTER_EXCHANGE: {
        "status": "pending",
        "recipient_event_id": None,
        "error": None,
        "metadata": {},
    },
    PersistenceSurface.SESSION: {
        "status": "active",
        "expires_at": None,
        "closed_at": None,
        "metadata": {},
        "runtime_snapshot": None,
        "version": 1,
    },
    PersistenceSurface.RELATIONSHIP: {
        "trust": 50.0,
        "favorability": 50.0,
        "relationship_stage": "stranger",
    },
}


def _coerce_surface(surface: PersistenceSurface | str) -> PersistenceSurface:
    if isinstance(surface, PersistenceSurface):
        return surface
    try:
        return PersistenceSurface(str(surface))
    except ValueError as exc:
        raise PersistenceMigrationError(f"unknown persistence surface: {surface!r}") from exc


def _schema_version(payload: Mapping[str, Any]) -> int:
    raw = payload.get(SCHEMA_VERSION_FIELD, LEGACY_UNVERSIONED_SCHEMA_VERSION)
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        raise PersistenceMigrationError(
            f"{SCHEMA_VERSION_FIELD} must be a non-negative integer"
        )
    return raw


def _validate_payload(surface: PersistenceSurface, payload: Mapping[str, Any]) -> None:
    expected_record_type = _RECORD_TYPES[surface]
    actual_record_type = payload.get("record_type")
    if expected_record_type is not None and actual_record_type != expected_record_type:
        raise PersistenceMigrationError(
            f"{surface.value} payload requires record_type={expected_record_type!r}; "
            f"got {actual_record_type!r}"
        )
    missing = [field for field in _REQUIRED_FIELDS[surface] if field not in payload]
    if missing:
        raise PersistenceMigrationError(
            f"{surface.value} payload is missing required fields: {', '.join(missing)}"
        )


def _legacy_to_v1(surface: PersistenceSurface, payload: dict[str, Any]) -> dict[str, Any]:
    migrated = copy.deepcopy(payload)
    expected_record_type = _RECORD_TYPES[surface]
    if expected_record_type is not None:
        actual = migrated.get("record_type")
        if actual is None:
            migrated["record_type"] = expected_record_type
        elif actual != expected_record_type:
            raise PersistenceMigrationError(
                f"{surface.value} legacy payload has record_type={actual!r}; "
                f"expected {expected_record_type!r}"
            )
    for key, value in _LEGACY_DEFAULTS[surface].items():
        if key not in migrated:
            migrated[key] = copy.deepcopy(value)
    migrated[SCHEMA_VERSION_FIELD] = 1
    _validate_payload(surface, migrated)
    return migrated


class PersistenceMigrationRegistry:
    """Deterministic, forward-only persistence schema migration registry.

    The registry never interprets application meaning beyond explicitly registered
    migrations. Unknown future versions and missing migration steps fail closed.
    """

    def __init__(
        self,
        *,
        current_versions: Mapping[PersistenceSurface | str, int] | None = None,
    ) -> None:
        source = current_versions or CURRENT_PERSISTENCE_SCHEMA_VERSIONS
        normalized: dict[PersistenceSurface, int] = {}
        for surface, version in source.items():
            normalized[_coerce_surface(surface)] = int(version)
        if set(normalized) != set(PersistenceSurface):
            missing = sorted(surface.value for surface in set(PersistenceSurface) - set(normalized))
            raise ValueError(f"current_versions must define every persistence surface; missing={missing!r}")
        if any(version < 0 for version in normalized.values()):
            raise ValueError("current schema versions must be >= 0")
        self._current_versions = normalized
        self._migrations: dict[tuple[PersistenceSurface, int], tuple[int, MigrationFunction]] = {}

    def current_version(self, surface: PersistenceSurface | str) -> int:
        return self._current_versions[_coerce_surface(surface)]

    def register(
        self,
        surface: PersistenceSurface | str,
        from_version: int,
        to_version: int,
        migrate: MigrationFunction,
    ) -> None:
        surface = _coerce_surface(surface)
        if from_version < 0 or to_version <= from_version:
            raise ValueError("migration versions must move forward from >= 0")
        key = (surface, from_version)
        if key in self._migrations:
            raise ValueError(
                f"migration already registered for {surface.value} v{from_version}"
            )
        self._migrations[key] = (to_version, migrate)

    def migrate(
        self,
        surface: PersistenceSurface | str,
        payload: Mapping[str, Any],
        *,
        source_version: int | None = None,
        target_version: int | None = None,
    ) -> PersistenceMigrationResult:
        surface = _coerce_surface(surface)
        if not isinstance(payload, Mapping):
            raise PersistenceMigrationError("persistence payload must be a mapping")
        current = self.current_version(surface)
        source = _schema_version(payload) if source_version is None else int(source_version)
        target = current if target_version is None else int(target_version)
        if source < 0 or target < 0:
            raise PersistenceMigrationError("schema versions must be >= 0")
        if source > current:
            raise UnsupportedPersistenceSchemaError(
                f"{surface.value} schema v{source} is newer than supported v{current}"
            )
        if target > current:
            raise UnsupportedPersistenceSchemaError(
                f"{surface.value} target schema v{target} is newer than supported v{current}"
            )
        if target < source:
            raise UnsupportedPersistenceSchemaError(
                f"persistence downgrades are not supported: {surface.value} v{source} -> v{target}"
            )
        working = copy.deepcopy(dict(payload))
        if source_version is not None:
            embedded = working.get(SCHEMA_VERSION_FIELD)
            if embedded is not None and embedded != source:
                raise PersistenceMigrationError(
                    f"declared source_version v{source} disagrees with payload {SCHEMA_VERSION_FIELD}={embedded!r}"
                )
        steps: list[PersistenceMigrationStep] = []
        version = source
        while version < target:
            entry = self._migrations.get((surface, version))
            if entry is None:
                raise PersistenceMigrationPathError(
                    f"no migration registered for {surface.value} v{version} toward v{target}"
                )
            next_version, migrate = entry
            if next_version <= version or next_version > target:
                raise PersistenceMigrationPathError(
                    f"invalid migration edge for {surface.value}: v{version} -> v{next_version}"
                )
            candidate = migrate(copy.deepcopy(working))
            if not isinstance(candidate, dict):
                raise PersistenceMigrationError(
                    f"migration for {surface.value} v{version} did not return a dict"
                )
            # The callback receives a deep copy, so it cannot mutate the caller's payload.
            working = copy.deepcopy(candidate)
            working[SCHEMA_VERSION_FIELD] = next_version
            steps.append(PersistenceMigrationStep(surface, version, next_version))
            version = next_version
        working[SCHEMA_VERSION_FIELD] = target
        _validate_payload(surface, working)
        return PersistenceMigrationResult(surface, source, target, working, tuple(steps))


def build_default_persistence_migration_registry() -> PersistenceMigrationRegistry:
    registry = PersistenceMigrationRegistry()
    for surface in PersistenceSurface:
        registry.register(
            surface,
            LEGACY_UNVERSIONED_SCHEMA_VERSION,
            1,
            lambda payload, surface=surface: _legacy_to_v1(surface, payload),
        )
    return registry


DEFAULT_PERSISTENCE_MIGRATIONS = build_default_persistence_migration_registry()


def current_persistence_schema_version(surface: PersistenceSurface | str) -> int:
    return DEFAULT_PERSISTENCE_MIGRATIONS.current_version(surface)


def migrate_persistence_payload(
    surface: PersistenceSurface | str,
    payload: Mapping[str, Any],
    *,
    source_version: int | None = None,
    target_version: int | None = None,
) -> PersistenceMigrationResult:
    return DEFAULT_PERSISTENCE_MIGRATIONS.migrate(
        surface,
        payload,
        source_version=source_version,
        target_version=target_version,
    )


def stamp_current_schema(
    surface: PersistenceSurface | str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    surface = _coerce_surface(surface)
    stamped = copy.deepcopy(dict(payload))
    stamped[SCHEMA_VERSION_FIELD] = current_persistence_schema_version(surface)
    _validate_payload(surface, stamped)
    return stamped


def infer_persistence_surface(payload: Mapping[str, Any]) -> PersistenceSurface:
    if not isinstance(payload, Mapping):
        raise PersistenceMigrationError("persistence payload must be a mapping")
    record_type = payload.get("record_type")
    matches = [surface for surface, expected in _RECORD_TYPES.items() if expected == record_type and expected is not None]
    if len(matches) == 1:
        return matches[0]
    raise PersistenceMigrationError(
        "cannot infer persistence surface; pass an explicit surface for untyped records"
    )


def canonical_persistence_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        copy.deepcopy(dict(payload)),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def persistence_payload_fingerprint(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_persistence_json(payload).encode("utf-8")).hexdigest()
