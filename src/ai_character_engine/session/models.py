from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from ai_character_engine.llm.models import Message
from ai_character_engine.state.models import CharacterStateSnapshot

SessionStatus = Literal["active", "closed", "expired"]


def _clean(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must not be empty")
    return cleaned


@dataclass(slots=True, frozen=True)
class SessionScopes:
    """Stable identity scopes used by a multi-user character application.

    Character scope is shared by everybody using the same character profile.
    User scope belongs to one application user. Relationship scope is shared
    across that user's sessions with the same character. Session scope is only
    the current conversation.
    """

    user_id: str
    character_id: str
    session_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "user_id", _clean(self.user_id, field_name="user_id"))
        object.__setattr__(
            self,
            "character_id",
            _clean(self.character_id, field_name="character_id"),
        )
        object.__setattr__(
            self,
            "session_id",
            _clean(self.session_id, field_name="session_id"),
        )

    @property
    def character_scope(self) -> str:
        return f"character:{self.character_id}"

    @property
    def user_scope(self) -> str:
        return f"user:{self.user_id}"

    @property
    def relationship_scope(self) -> str:
        return f"user:{self.user_id}:character:{self.character_id}"

    @property
    def session_scope(self) -> str:
        return f"session:{self.session_id}"

    @property
    def memory_scope_id(self) -> str:
        # Long-term memory follows the user × character relationship rather
        # than one transient conversation session.
        return self.relationship_scope


@dataclass(slots=True, frozen=True)
class SessionRuntimeSnapshot:
    """Serializable runtime state required to resume one conversation."""

    state: CharacterStateSnapshot
    history: tuple[Message, ...] = field(default_factory=tuple)


@dataclass(slots=True)
class SessionRecord:
    """Persistent metadata and runtime snapshot for one conversation session."""

    user_id: str
    character_id: str
    id: str = field(default_factory=lambda: uuid4().hex)
    status: SessionStatus = "active"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_activity: datetime = field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime | None = None
    closed_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    runtime_snapshot: SessionRuntimeSnapshot | None = None
    version: int = 1

    def __post_init__(self) -> None:
        self.user_id = _clean(self.user_id, field_name="user_id")
        self.character_id = _clean(self.character_id, field_name="character_id")
        self.id = _clean(self.id, field_name="session_id")
        if self.status not in {"active", "closed", "expired"}:
            raise ValueError(f"invalid session status: {self.status}")
        if self.version < 1:
            raise ValueError("session version must be >= 1")

    @property
    def scopes(self) -> SessionScopes:
        return SessionScopes(
            user_id=self.user_id,
            character_id=self.character_id,
            session_id=self.id,
        )

    @property
    def is_active(self) -> bool:
        return self.status == "active"


@dataclass(slots=True, frozen=True)
class RelationshipSnapshot:
    """User × character relationship data shared across conversation sessions."""

    user_id: str
    character_id: str
    trust: float = 50.0
    favorability: float = 50.0
    relationship_stage: str = "stranger"
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def from_state(
        cls,
        *,
        user_id: str,
        character_id: str,
        state: CharacterStateSnapshot,
        updated_at: datetime | None = None,
    ) -> "RelationshipSnapshot":
        return cls(
            user_id=user_id,
            character_id=character_id,
            trust=state.trust,
            favorability=state.favorability,
            relationship_stage=state.relationship_stage,
            updated_at=updated_at or datetime.now(UTC),
        )


@dataclass(slots=True, frozen=True)
class SessionAuditEntry:
    action: str
    session_id: str
    user_id: str
    character_id: str
    at: datetime = field(default_factory=lambda: datetime.now(UTC))
    details: dict[str, Any] = field(default_factory=dict)
