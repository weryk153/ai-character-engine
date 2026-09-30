from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from ai_character_engine.events.models import CharacterEvent
from ai_character_engine.llm.models import LLMResponse
from ai_character_engine.state.models import CharacterStateSnapshot


@dataclass(slots=True, frozen=True)
class EventLedgerEntry:
    """Append-only record of what actually happened in the character world.

    Ledger entries are audit/history data. They are deliberately separate from
    retrievable long-term memories, which may later be merged or discarded.
    """

    character_id: str
    event_id: str
    event_type: str
    source: str
    content: str
    response_text: str
    payload: dict[str, Any] = field(default_factory=dict)
    state_before: dict[str, Any] = field(default_factory=dict)
    state_after: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid4().hex)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def from_interaction(
        cls,
        *,
        character_id: str,
        event: CharacterEvent,
        response: LLMResponse,
        state_before: CharacterStateSnapshot,
        state_after: CharacterStateSnapshot,
    ) -> "EventLedgerEntry":
        return cls(
            character_id=character_id,
            event_id=event.id,
            event_type=event.type,
            source=event.source,
            content=event.content,
            response_text=response.text,
            payload=dict(event.payload),
            state_before=asdict(state_before),
            state_after=asdict(state_after),
        )


class EventLedger(Protocol):
    def append(self, entry: EventLedgerEntry) -> None:
        ...

    def list_for_character(self, character_id: str) -> list[EventLedgerEntry]:
        ...


class InMemoryEventLedger:
    def __init__(self, entries: Iterable[EventLedgerEntry] = ()) -> None:
        self._entries = list(entries)

    def append(self, entry: EventLedgerEntry) -> None:
        self._entries.append(entry)

    def list_for_character(self, character_id: str) -> list[EventLedgerEntry]:
        return [entry for entry in self._entries if entry.character_id == character_id]


class JsonlEventLedger(InMemoryEventLedger):
    """Simple append-only JSONL ledger for local development."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        entries: list[EventLedgerEntry] = []
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                entries.append(self._from_dict(json.loads(line)))
        super().__init__(entries)

    def append(self, entry: EventLedgerEntry) -> None:
        super().append(entry)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(self._to_dict(entry), ensure_ascii=False) + "\n")

    @staticmethod
    def _to_dict(entry: EventLedgerEntry) -> dict[str, Any]:
        return {
            "id": entry.id,
            "character_id": entry.character_id,
            "event_id": entry.event_id,
            "event_type": entry.event_type,
            "source": entry.source,
            "content": entry.content,
            "response_text": entry.response_text,
            "payload": entry.payload,
            "state_before": entry.state_before,
            "state_after": entry.state_after,
            "created_at": entry.created_at.isoformat(),
        }

    @staticmethod
    def _from_dict(payload: dict[str, Any]) -> EventLedgerEntry:
        return EventLedgerEntry(
            id=payload["id"],
            character_id=payload["character_id"],
            event_id=payload["event_id"],
            event_type=payload["event_type"],
            source=payload["source"],
            content=payload.get("content", ""),
            response_text=payload.get("response_text", ""),
            payload=dict(payload.get("payload", {})),
            state_before=dict(payload.get("state_before", {})),
            state_after=dict(payload.get("state_after", {})),
            created_at=datetime.fromisoformat(payload["created_at"]),
        )
