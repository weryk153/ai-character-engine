from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from ai_character_engine.persistence import PersistenceSurface, migrate_persistence_payload, stamp_current_schema

from .models import CharacterExchange, ExchangeStatus, KnowledgeVisibility, SharedCognitionRecord


class SharedCognitionStore(Protocol):
    def add(self, record: SharedCognitionRecord) -> None: ...
    def get(self, record_id: str) -> SharedCognitionRecord | None: ...
    def list_for_owner(self, owner_character_id: str) -> list[SharedCognitionRecord]: ...
    def list_visible_to(self, character_id: str) -> list[SharedCognitionRecord]: ...


class CharacterExchangeStore(Protocol):
    def add(self, exchange: CharacterExchange) -> None: ...
    def get(self, exchange_id: str) -> CharacterExchange | None: ...
    def replace(self, exchange: CharacterExchange) -> None: ...
    def list_for_character(self, character_id: str) -> list[CharacterExchange]: ...


class InMemorySharedCognitionStore:
    def __init__(self, records: Iterable[SharedCognitionRecord] = ()) -> None:
        self._records = list(records)

    def add(self, record: SharedCognitionRecord) -> None:
        if self.get(record.id) is not None:
            raise ValueError(f"duplicate shared cognition id: {record.id}")
        self._records.append(record)

    def get(self, record_id: str) -> SharedCognitionRecord | None:
        return next((record for record in self._records if record.id == record_id), None)

    def list_for_owner(self, owner_character_id: str) -> list[SharedCognitionRecord]:
        return [record for record in self._records if record.owner_character_id == owner_character_id]

    def list_visible_to(self, character_id: str) -> list[SharedCognitionRecord]:
        return [record for record in self._records if record.visible_to(character_id)]


class InMemoryCharacterExchangeStore:
    def __init__(self, exchanges: Iterable[CharacterExchange] = ()) -> None:
        self._exchanges = list(exchanges)

    def add(self, exchange: CharacterExchange) -> None:
        if self.get(exchange.id) is not None:
            raise ValueError(f"duplicate character exchange id: {exchange.id}")
        self._exchanges.append(exchange)

    def get(self, exchange_id: str) -> CharacterExchange | None:
        return next((item for item in self._exchanges if item.id == exchange_id), None)

    def replace(self, exchange: CharacterExchange) -> None:
        found = False
        items: list[CharacterExchange] = []
        for current in self._exchanges:
            if current.id == exchange.id:
                items.append(exchange)
                found = True
            else:
                items.append(current)
        if not found:
            raise KeyError(exchange.id)
        self._exchanges = items

    def list_for_character(self, character_id: str) -> list[CharacterExchange]:
        return [
            item
            for item in self._exchanges
            if item.sender_character_id == character_id or item.recipient_character_id == character_id
        ]


class JsonlSharedCognitionStore(InMemorySharedCognitionStore):
    """Single-process local/dev persistence for explicit shared cognition exports."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        records: list[SharedCognitionRecord] = []
        if self.path.exists():
            for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if payload.get("record_type") != "shared_cognition":
                    raise ValueError(f"{self.path}:{line_number}: unknown record_type")
                payload = dict(migrate_persistence_payload(PersistenceSurface.SHARED_COGNITION, payload).payload)
                records.append(_shared_from_dict(payload))
        super().__init__(records)

    def add(self, record: SharedCognitionRecord) -> None:
        super().add(record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_shared_to_dict(record), ensure_ascii=False, sort_keys=True) + "\n")


class JsonlCharacterExchangeStore(InMemoryCharacterExchangeStore):
    """Single-process local/dev persistence for cross-character exchange audit."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        exchanges: list[CharacterExchange] = []
        if self.path.exists():
            for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if payload.get("record_type") != "character_exchange":
                    raise ValueError(f"{self.path}:{line_number}: unknown record_type")
                payload = dict(migrate_persistence_payload(PersistenceSurface.CHARACTER_EXCHANGE, payload).payload)
                exchanges.append(_exchange_from_dict(payload))
        super().__init__(exchanges)

    def add(self, exchange: CharacterExchange) -> None:
        super().add(exchange)
        self._rewrite()

    def replace(self, exchange: CharacterExchange) -> None:
        super().replace(exchange)
        self._rewrite()

    def _rewrite(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = "".join(
            json.dumps(_exchange_to_dict(item), ensure_ascii=False, sort_keys=True) + "\n"
            for item in self._exchanges
        )
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(self.path)


def _shared_to_dict(record: SharedCognitionRecord) -> dict[str, Any]:
    return stamp_current_schema(PersistenceSurface.SHARED_COGNITION, {
        "record_type": "shared_cognition",
        "id": record.id,
        "owner_character_id": record.owner_character_id,
        "kind": record.kind,
        "content": record.content,
        "visibility": record.visibility.value,
        "audience_character_ids": list(record.audience_character_ids),
        "source_type": record.source_type,
        "source_id": record.source_id,
        "metadata": _jsonable(record.metadata),
        "created_at": record.created_at.isoformat(),
    })


def _shared_from_dict(payload: dict[str, Any]) -> SharedCognitionRecord:
    return SharedCognitionRecord(
        id=payload["id"],
        owner_character_id=payload["owner_character_id"],
        kind=payload["kind"],
        content=payload["content"],
        visibility=KnowledgeVisibility(payload["visibility"]),
        audience_character_ids=tuple(payload.get("audience_character_ids", ())),
        source_type=payload.get("source_type", "host"),
        source_id=payload.get("source_id"),
        metadata=dict(payload.get("metadata", {})),
        created_at=datetime.fromisoformat(payload["created_at"]),
    )


def _exchange_to_dict(exchange: CharacterExchange) -> dict[str, Any]:
    return stamp_current_schema(PersistenceSurface.CHARACTER_EXCHANGE, {
        "record_type": "character_exchange",
        "id": exchange.id,
        "sender_character_id": exchange.sender_character_id,
        "recipient_character_id": exchange.recipient_character_id,
        "content": exchange.content,
        "status": exchange.status.value,
        "recipient_event_id": exchange.recipient_event_id,
        "error": exchange.error,
        "metadata": _jsonable(exchange.metadata),
        "created_at": exchange.created_at.isoformat(),
        "updated_at": exchange.updated_at.isoformat(),
    })


def _exchange_from_dict(payload: dict[str, Any]) -> CharacterExchange:
    return CharacterExchange(
        id=payload["id"],
        sender_character_id=payload["sender_character_id"],
        recipient_character_id=payload["recipient_character_id"],
        content=payload["content"],
        status=ExchangeStatus(payload.get("status", ExchangeStatus.PENDING.value)),
        recipient_event_id=payload.get("recipient_event_id"),
        error=payload.get("error"),
        metadata=dict(payload.get("metadata", {})),
        created_at=datetime.fromisoformat(payload["created_at"]),
        updated_at=datetime.fromisoformat(payload["updated_at"]),
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict) or hasattr(value, "items"):
        return {str(key): _jsonable(item) for key, item in dict(value).items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)
