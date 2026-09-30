from __future__ import annotations

import json
import os
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Protocol

from ai_character_engine.persistence import PersistenceSurface, migrate_persistence_payload, stamp_current_schema

from .models import MemoryRecord


class MemoryStore(Protocol):
    def add(self, record: MemoryRecord) -> None:
        ...

    def list_for_character(self, character_id: str) -> list[MemoryRecord]:
        ...

    def replace_for_character(
        self,
        character_id: str,
        records: Iterable[MemoryRecord],
    ) -> None:
        ...


class InMemoryMemoryStore:
    def __init__(self, records: Iterable[MemoryRecord] = ()) -> None:
        self._records = list(records)

    def add(self, record: MemoryRecord) -> None:
        self._records.append(record)

    def list_for_character(self, character_id: str) -> list[MemoryRecord]:
        return [record for record in self._records if record.character_id == character_id]

    def replace_for_character(
        self,
        character_id: str,
        records: Iterable[MemoryRecord],
    ) -> None:
        kept = [record for record in self._records if record.character_id != character_id]
        self._records = [*kept, *list(records)]


class JsonlMemoryStore(InMemoryMemoryStore):
    """Tiny persistent store for development and single-process examples.

    This is deliberately not a production database. It gives v0.5 real
    persistence without coupling the engine to Redis/PostgreSQL/Qdrant yet.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        records: list[MemoryRecord] = []
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                payload = json.loads(line)
                payload = dict(migrate_persistence_payload(PersistenceSurface.MEMORY, payload).payload)
                records.append(self._from_dict(payload))
        super().__init__(records)

    def add(self, record: MemoryRecord) -> None:
        super().add(record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(self._to_dict(record), ensure_ascii=False) + "\n")

    def replace_for_character(
        self,
        character_id: str,
        records: Iterable[MemoryRecord],
    ) -> None:
        super().replace_for_character(character_id, records)
        self._rewrite()

    def _rewrite(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = "".join(
            json.dumps(self._to_dict(record), ensure_ascii=False) + "\n"
            for record in self._records
        )
        # Written beside and moved into place: a crash in the middle of the
        # file would lose every memory of every character in it.
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, self.path)

    @staticmethod
    def _to_dict(record: MemoryRecord) -> dict:
        return stamp_current_schema(PersistenceSurface.MEMORY, {
            "id": record.id,
            "character_id": record.character_id,
            "summary": record.summary,
            "importance": record.importance,
            "kind": record.kind,
            "tags": list(record.tags),
            "source_event_id": record.source_event_id,
            "source_event_type": record.source_event_type,
            "metadata": record.metadata,
            "embedding": list(record.embedding) if record.embedding is not None else None,
            "embedding_metadata": record.embedding_metadata,
            "vector_metadata": record.vector_metadata,
            "status": record.status,
            "supersedes": list(record.supersedes),
            "superseded_by": record.superseded_by,
            "forgotten_at": (record.forgotten_at.isoformat() if record.forgotten_at else None),
            "created_at": record.created_at.isoformat(),
        })

    @staticmethod
    def _from_dict(payload: dict) -> MemoryRecord:
        return MemoryRecord(
            id=payload["id"],
            character_id=payload["character_id"],
            summary=payload["summary"],
            importance=payload.get("importance", 0.5),
            kind=payload.get("kind", "event"),
            tags=tuple(payload.get("tags", ())),
            source_event_id=payload.get("source_event_id"),
            source_event_type=payload.get("source_event_type"),
            metadata=dict(payload.get("metadata", {})),
            embedding=(tuple(payload["embedding"]) if payload.get("embedding") is not None else None),
            embedding_metadata=dict(payload.get("embedding_metadata", {})),
            vector_metadata=dict(payload.get("vector_metadata", {})),
            status=payload.get("status", "active"),
            supersedes=tuple(payload.get("supersedes", ())),
            superseded_by=payload.get("superseded_by"),
            forgotten_at=(
                datetime.fromisoformat(payload["forgotten_at"])
                if payload.get("forgotten_at")
                else None
            ),
            created_at=datetime.fromisoformat(payload["created_at"]),
        )
