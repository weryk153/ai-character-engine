from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Protocol

from ai_character_engine.persistence import PersistenceSurface, migrate_persistence_payload, stamp_current_schema

from .models import RelationshipSnapshot, SessionRecord
from .serialization import session_from_dict, session_to_dict


class SessionStore(Protocol):
    """Persistence boundary for session metadata and runtime snapshots."""

    def get(self, session_id: str) -> SessionRecord | None:
        ...

    def put(self, record: SessionRecord) -> None:
        ...

    def list(self) -> list[SessionRecord]:
        ...

    def delete(self, session_id: str) -> None:
        ...


# Semantic alias used in the public API/documentation. Redis/PostgreSQL adapters
# can implement this protocol without changing SessionManager.
PersistentSessionStore = SessionStore


class InMemorySessionStore:
    def __init__(self, records: Iterable[SessionRecord] = ()) -> None:
        self._records = {record.id: record for record in records}

    def get(self, session_id: str) -> SessionRecord | None:
        return self._records.get(session_id)

    def put(self, record: SessionRecord) -> None:
        self._records[record.id] = record

    def list(self) -> list[SessionRecord]:
        return list(self._records.values())

    def delete(self, session_id: str) -> None:
        self._records.pop(session_id, None)


class JsonFileSessionStore(InMemorySessionStore):
    """Small local persistent store used to demonstrate process restore.

    Production services should provide a Redis/PostgreSQL/etc. adapter that
    implements SessionStore rather than coupling SessionManager to this class.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        records: list[SessionRecord] = []
        if self.path.exists() and self.path.read_text(encoding="utf-8").strip():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            records = [session_from_dict(item) for item in payload]
        super().__init__(records)

    def put(self, record: SessionRecord) -> None:
        super().put(record)
        self._rewrite()

    def delete(self, session_id: str) -> None:
        super().delete(session_id)
        self._rewrite()

    def _rewrite(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                [session_to_dict(record) for record in self.list()],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


class RelationshipStore(Protocol):
    def get(self, *, user_id: str, character_id: str) -> RelationshipSnapshot | None:
        ...

    def put(self, snapshot: RelationshipSnapshot) -> None:
        ...


class InMemoryRelationshipStore:
    def __init__(self, snapshots: Iterable[RelationshipSnapshot] = ()) -> None:
        self._snapshots = {
            (snapshot.user_id, snapshot.character_id): snapshot
            for snapshot in snapshots
        }

    def get(self, *, user_id: str, character_id: str) -> RelationshipSnapshot | None:
        return self._snapshots.get((user_id, character_id))

    def put(self, snapshot: RelationshipSnapshot) -> None:
        self._snapshots[(snapshot.user_id, snapshot.character_id)] = snapshot

class JsonFileRelationshipStore(InMemoryRelationshipStore):
    """Small local persistent relationship store for development/examples."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        snapshots: list[RelationshipSnapshot] = []
        if self.path.exists() and self.path.read_text(encoding="utf-8").strip():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            migrated = [
                dict(migrate_persistence_payload(PersistenceSurface.RELATIONSHIP, item).payload)
                for item in payload
            ]
            snapshots = [
                RelationshipSnapshot(
                    user_id=item["user_id"],
                    character_id=item["character_id"],
                    trust=float(item.get("trust", 50.0)),
                    favorability=float(item.get("favorability", 50.0)),
                    relationship_stage=item.get("relationship_stage", "stranger"),
                    updated_at=datetime.fromisoformat(item["updated_at"]),
                )
                for item in migrated
            ]
        super().__init__(snapshots)

    def put(self, snapshot: RelationshipSnapshot) -> None:
        super().put(snapshot)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            stamp_current_schema(PersistenceSurface.RELATIONSHIP, {
                "user_id": item.user_id,
                "character_id": item.character_id,
                "trust": item.trust,
                "favorability": item.favorability,
                "relationship_stage": item.relationship_stage,
                "updated_at": item.updated_at.isoformat(),
            })
            for item in self._snapshots.values()
        ]
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
