from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from .models import PersistenceMigrationError, PersistenceSurface
from .registry import infer_persistence_surface, migrate_persistence_payload


class PersistenceFileFormat(str, Enum):
    JSON = "json"
    JSON_ARRAY = "json-array"
    JSONL = "jsonl"


def _resolve_surface(
    configured: PersistenceSurface | str | None,
    payload: dict[str, Any],
) -> PersistenceSurface:
    if configured is None or configured == "auto":
        return infer_persistence_surface(payload)
    if isinstance(configured, PersistenceSurface):
        return configured
    return PersistenceSurface(str(configured))


def _migrate_object(
    payload: Any,
    *,
    surface: PersistenceSurface | str | None,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise PersistenceMigrationError("persistence record must be a JSON object")
    resolved = _resolve_surface(surface, payload)
    return dict(migrate_persistence_payload(resolved, payload).payload)


def migrate_persistence_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    file_format: PersistenceFileFormat | str,
    surface: PersistenceSurface | str | None = None,
) -> int:
    """Explicitly migrate a JSON/JSONL artifact without mutating the source file.

    The caller must choose a different output path. This prevents a migration CLI
    from silently destroying the only copy of a legacy artifact.
    """

    source = Path(input_path)
    destination = Path(output_path)
    if source.resolve() == destination.resolve():
        raise PersistenceMigrationError("input and output paths must be different")
    try:
        fmt = file_format if isinstance(file_format, PersistenceFileFormat) else PersistenceFileFormat(str(file_format))
    except ValueError as exc:
        raise PersistenceMigrationError(f"unsupported persistence file format: {file_format!r}") from exc
    if not source.exists():
        raise PersistenceMigrationError(f"input file does not exist: {source}")

    count = 0
    if fmt is PersistenceFileFormat.JSON:
        raw = json.loads(source.read_text(encoding="utf-8"))
        migrated: Any = _migrate_object(raw, surface=surface)
        count = 1
        encoded = json.dumps(migrated, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    elif fmt is PersistenceFileFormat.JSON_ARRAY:
        raw = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise PersistenceMigrationError("json-array input must contain a JSON array")
        migrated = [_migrate_object(item, surface=surface) for item in raw]
        count = len(migrated)
        encoded = json.dumps(migrated, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    else:
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                rows.append(_migrate_object(raw, surface=surface))
            except Exception as exc:
                raise PersistenceMigrationError(f"{source}:{line_number}: {exc}") from exc
        count = len(rows)
        encoded = "".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in rows
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    tmp.write_text(encoded, encoding="utf-8")
    tmp.replace(destination)
    return count
