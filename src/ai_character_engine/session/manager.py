from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from .models import SessionAuditEntry, SessionRecord, SessionRuntimeSnapshot
from .store import InMemorySessionStore, SessionStore


class SessionNotFoundError(KeyError):
    pass


class SessionUnavailableError(RuntimeError):
    pass


class SessionManager:
    """Creates, restores, touches, closes, and expires conversation sessions."""

    def __init__(
        self,
        store: SessionStore | None = None,
        *,
        default_ttl_seconds: float | None = 3600.0,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if default_ttl_seconds is not None and default_ttl_seconds <= 0:
            raise ValueError("default_ttl_seconds must be > 0 or None")
        self.store = store or InMemorySessionStore()
        self.default_ttl_seconds = default_ttl_seconds
        self.clock = clock or (lambda: datetime.now(UTC))
        self.audit_log: list[SessionAuditEntry] = []

    def create(
        self,
        *,
        user_id: str,
        character_id: str,
        session_id: str | None = None,
        metadata: dict | None = None,
        ttl_seconds: float | None = None,
        snapshot: SessionRuntimeSnapshot | None = None,
    ) -> SessionRecord:
        now = self.clock()
        effective_ttl = self.default_ttl_seconds if ttl_seconds is None else ttl_seconds
        if effective_ttl is not None and effective_ttl <= 0:
            raise ValueError("ttl_seconds must be > 0 or None")
        expires_at = (
            now + timedelta(seconds=effective_ttl)
            if effective_ttl is not None
            else None
        )
        record = SessionRecord(
            id=session_id or uuid4().hex,
            user_id=user_id,
            character_id=character_id,
            created_at=now,
            last_activity=now,
            expires_at=expires_at,
            metadata=dict(metadata or {}),
            runtime_snapshot=snapshot,
        )
        if self.store.get(record.id) is not None:
            raise ValueError(f"session already exists: {record.id}")
        self.store.put(record)
        self._audit("created", record)
        return record

    def get(self, session_id: str, *, require_active: bool = True) -> SessionRecord | None:
        record = self.store.get(session_id)
        if record is None:
            return None
        self._expire_if_needed(record)
        if require_active and not record.is_active:
            raise SessionUnavailableError(
                f"session {session_id} is {record.status}"
            )
        return record

    def require(self, session_id: str, *, require_active: bool = True) -> SessionRecord:
        record = self.get(session_id, require_active=require_active)
        if record is None:
            raise SessionNotFoundError(session_id)
        return record

    def touch(
        self,
        session_id: str,
        *,
        snapshot: SessionRuntimeSnapshot | None = None,
        metadata_updates: dict | None = None,
        ttl_seconds: float | None = None,
    ) -> SessionRecord:
        record = self.require(session_id)
        now = self.clock()
        effective_ttl = self.default_ttl_seconds if ttl_seconds is None else ttl_seconds
        if effective_ttl is not None and effective_ttl <= 0:
            raise ValueError("ttl_seconds must be > 0 or None")
        record.last_activity = now
        if effective_ttl is not None:
            record.expires_at = now + timedelta(seconds=effective_ttl)
        elif ttl_seconds is None and self.default_ttl_seconds is None:
            record.expires_at = None
        if snapshot is not None:
            record.runtime_snapshot = snapshot
        if metadata_updates:
            record.metadata.update(metadata_updates)
        record.version += 1
        self.store.put(record)
        self._audit("touched", record)
        return record

    def close(self, session_id: str) -> SessionRecord:
        record = self.require(session_id)
        now = self.clock()
        record.status = "closed"
        record.closed_at = now
        record.last_activity = now
        record.version += 1
        self.store.put(record)
        self._audit("closed", record)
        return record

    def cleanup_expired(self) -> tuple[str, ...]:
        expired: list[str] = []
        for record in self.store.list():
            if self._expire_if_needed(record):
                expired.append(record.id)
        return tuple(expired)

    def list_for_user_character(
        self,
        *,
        user_id: str,
        character_id: str,
        include_inactive: bool = False,
    ) -> list[SessionRecord]:
        records = [
            record
            for record in self.store.list()
            if record.user_id == user_id and record.character_id == character_id
        ]
        for record in records:
            self._expire_if_needed(record)
        if not include_inactive:
            records = [record for record in records if record.is_active]
        return sorted(records, key=lambda record: record.last_activity, reverse=True)

    def _expire_if_needed(self, record: SessionRecord) -> bool:
        if not record.is_active or record.expires_at is None:
            return False
        if self.clock() < record.expires_at:
            return False
        record.status = "expired"
        record.version += 1
        self.store.put(record)
        self._audit("expired", record)
        return True

    def _audit(self, action: str, record: SessionRecord) -> None:
        self.audit_log.append(
            SessionAuditEntry(
                action=action,
                session_id=record.id,
                user_id=record.user_id,
                character_id=record.character_id,
                at=self.clock(),
                details={"status": record.status, "version": record.version},
            )
        )
