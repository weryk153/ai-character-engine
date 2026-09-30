from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4


class KnowledgeVisibility(str, Enum):
    """Visibility of an explicitly exported cross-character cognition item."""

    PRIVATE = "private"
    DIRECT = "direct"
    PUBLIC = "public"


class ExchangeStatus(str, Enum):
    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"


def _clean(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must not be empty")
    return cleaned


@dataclass(slots=True, frozen=True)
class SharedCognitionRecord:
    """Explicit, non-authoritative export from one character boundary.

    Records are advisory evidence only. Publishing one never mutates another
    character's Memory, Reflection, Belief, Goal or State. A host must deliver
    the record explicitly before another CharacterRuntime can observe it.
    """

    owner_character_id: str
    kind: str
    content: str
    visibility: KnowledgeVisibility = KnowledgeVisibility.PRIVATE
    audience_character_ids: tuple[str, ...] = field(default_factory=tuple)
    source_type: str = "host"
    source_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid4().hex)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        owner = _clean(self.owner_character_id, field_name="owner_character_id")
        kind = _clean(self.kind, field_name="kind")
        content = _clean(self.content, field_name="content")
        source_type = _clean(self.source_type, field_name="source_type")
        record_id = _clean(self.id, field_name="id")
        audience = tuple(dict.fromkeys(_clean(x, field_name="audience_character_id") for x in self.audience_character_ids))
        if owner in audience:
            raise ValueError("owner must not appear in direct audience")
        if self.visibility is KnowledgeVisibility.DIRECT and not audience:
            raise ValueError("direct visibility requires at least one audience character")
        if self.visibility in {KnowledgeVisibility.PRIVATE, KnowledgeVisibility.PUBLIC} and audience:
            raise ValueError(f"{self.visibility.value} visibility must not define an audience")
        object.__setattr__(self, "owner_character_id", owner)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "content", content)
        object.__setattr__(self, "source_type", source_type)
        object.__setattr__(self, "id", record_id)
        object.__setattr__(self, "audience_character_ids", audience)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def visible_to(self, character_id: str) -> bool:
        character_id = _clean(character_id, field_name="character_id")
        if character_id == self.owner_character_id:
            return True
        if self.visibility is KnowledgeVisibility.PUBLIC:
            return True
        if self.visibility is KnowledgeVisibility.DIRECT:
            return character_id in self.audience_character_ids
        return False


@dataclass(slots=True, frozen=True)
class CharacterExchange:
    """Audit record for one explicitly mediated character-to-character message."""

    sender_character_id: str
    recipient_character_id: str
    content: str
    id: str = field(default_factory=lambda: uuid4().hex)
    status: ExchangeStatus = ExchangeStatus.PENDING
    recipient_event_id: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        sender = _clean(self.sender_character_id, field_name="sender_character_id")
        recipient = _clean(self.recipient_character_id, field_name="recipient_character_id")
        content = _clean(self.content, field_name="content")
        exchange_id = _clean(self.id, field_name="id")
        if sender == recipient:
            raise ValueError("sender and recipient must be different characters")
        object.__setattr__(self, "sender_character_id", sender)
        object.__setattr__(self, "recipient_character_id", recipient)
        object.__setattr__(self, "content", content)
        object.__setattr__(self, "id", exchange_id)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(slots=True, frozen=True)
class FairSchedulerConfig:
    max_concurrent_characters: int = 4
    max_pending_per_character: int = 8
    max_total_pending: int = 64

    def __post_init__(self) -> None:
        if self.max_concurrent_characters < 1:
            raise ValueError("max_concurrent_characters must be >= 1")
        if self.max_pending_per_character < 0:
            raise ValueError("max_pending_per_character must be >= 0")
        if self.max_total_pending < 0:
            raise ValueError("max_total_pending must be >= 0")


@dataclass(slots=True, frozen=True)
class FairSchedulerSnapshot:
    running_character_ids: tuple[str, ...]
    pending_by_character: dict[str, int]

    @property
    def total_pending(self) -> int:
        return sum(self.pending_by_character.values())
