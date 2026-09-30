"""Offline persistence migration + replay verification example."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_character_engine.persistence import (
    PersistenceFileFormat,
    PersistenceSurface,
    migrate_persistence_file,
    migrate_persistence_payload,
    verify_persistence_fixture_directory,
)


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "persistence" / "v0.45"


def main() -> None:
    legacy_memory = {
        "id": "memory-example",
        "character_id": "character-example",
        "summary": "The user asked for concise answers.",
        "created_at": "2026-09-01T12:00:00+00:00",
    }
    migrated = migrate_persistence_payload(PersistenceSurface.MEMORY, legacy_memory)
    print(
        json.dumps(
            {
                "surface": migrated.surface.value,
                "source_version": migrated.source_version,
                "target_version": migrated.target_version,
                "summary": migrated.payload["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )

    replay = verify_persistence_fixture_directory(FIXTURES)
    print(
        json.dumps(
            {
                "fixture_count": len(replay),
                "all_replay_fixtures_passed": all(item.passed for item in replay),
            },
            sort_keys=True,
        )
    )

    with TemporaryDirectory() as directory:
        old_path = Path(directory) / "legacy-memory.jsonl"
        new_path = Path(directory) / "memory-v1.jsonl"
        old_path.write_text(json.dumps(legacy_memory) + "\n", encoding="utf-8")
        migrated_count = migrate_persistence_file(
            old_path,
            new_path,
            file_format=PersistenceFileFormat.JSONL,
            surface=PersistenceSurface.MEMORY,
        )
        print(
            json.dumps(
                {
                    "explicit_file_migration_count": migrated_count,
                    "source_unchanged": "_schema_version" not in old_path.read_text(encoding="utf-8"),
                    "output_schema_version": json.loads(new_path.read_text(encoding="utf-8"))["_schema_version"],
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
