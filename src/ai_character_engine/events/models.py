from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4


@dataclass(slots=True, frozen=True)
class CharacterEvent:
    """An observation delivered to a character runtime.

    Events are provider-neutral. A user message is one event type, while host
    applications may also publish events such as super chats, game state
    changes, timers, vision observations, or stream events.
    """

    type: str
    source: str
    content: str
    payload: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid4().hex)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if not self.type.strip():
            raise ValueError("event type must not be empty")
        if not self.source.strip():
            raise ValueError("event source must not be empty")
        if not self.content.strip() and not self.payload:
            raise ValueError("event must contain content or payload")

    @classmethod
    def user_message(cls, content: str, *, source: str = "user") -> "CharacterEvent":
        if not content.strip():
            raise ValueError("user message must not be empty")
        return cls(type="user_message", source=source, content=content)
