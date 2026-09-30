from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import (
    PERSISTENCE_FIXTURE_SCHEMA_VERSION,
    PersistenceFixtureError,
    PersistenceReplayFixture,
    PersistenceReplayResult,
    PersistenceSurface,
)
from .registry import migrate_persistence_payload, persistence_payload_fingerprint


def load_persistence_replay_fixture(path: str | Path) -> PersistenceReplayFixture:
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PersistenceFixtureError(f"failed to load persistence fixture {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PersistenceFixtureError(f"{path}: fixture root must be an object")
    try:
        return PersistenceReplayFixture(
            fixture_schema_version=int(
                payload.get("fixture_schema_version", PERSISTENCE_FIXTURE_SCHEMA_VERSION)
            ),
            fixture_id=payload["fixture_id"],
            engine_version=payload["engine_version"],
            surface=PersistenceSurface(payload["surface"]),
            source_schema_version=int(payload["source_schema_version"]),
            payload=dict(payload["payload"]),
            expected_payload=dict(payload["expected_payload"]),
            metadata=dict(payload.get("metadata", {})),
        )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, PersistenceFixtureError):
            raise
        raise PersistenceFixtureError(f"{path}: malformed fixture: {exc}") from exc


def save_persistence_replay_fixture(
    fixture: PersistenceReplayFixture,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(fixture.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def verify_persistence_replay_fixture(
    fixture: PersistenceReplayFixture,
) -> PersistenceReplayResult:
    expected = dict(fixture.expected_payload)
    expected_fingerprint = persistence_payload_fingerprint(expected)
    issues: list[str] = []
    try:
        migrated = migrate_persistence_payload(
            fixture.surface,
            fixture.payload,
            source_version=fixture.source_schema_version,
        )
        actual = dict(migrated.payload)
        actual_fingerprint = persistence_payload_fingerprint(actual)
        if actual != expected:
            issues.append("migrated_payload_mismatch")
    except Exception as exc:  # verification should return evidence instead of hiding fixture failures
        actual_fingerprint = ""
        issues.append(f"migration_failed:{type(exc).__name__}:{exc}")
    return PersistenceReplayResult(
        fixture_id=fixture.fixture_id,
        surface=fixture.surface,
        passed=not issues,
        expected_fingerprint=expected_fingerprint,
        actual_fingerprint=actual_fingerprint,
        issues=tuple(issues),
    )


def verify_persistence_fixture_directory(path: str | Path) -> tuple[PersistenceReplayResult, ...]:
    path = Path(path)
    if not path.exists() or not path.is_dir():
        raise PersistenceFixtureError(f"fixture directory does not exist: {path}")
    fixture_paths = sorted(item for item in path.glob("*.json") if item.is_file())
    if not fixture_paths:
        raise PersistenceFixtureError(f"fixture directory contains no .json fixtures: {path}")
    return tuple(
        verify_persistence_replay_fixture(load_persistence_replay_fixture(item))
        for item in fixture_paths
    )
