from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4


class WorldChangeKind(str, Enum):
    SET = "set"
    DELETE = "delete"


class WorldPerceptionScope(str, Enum):
    """Who is eligible to perceive an event when observation is explicitly requested.

    Eligibility is not delivery. PUBLIC still does not inject the event into any
    character automatically.
    """

    HIDDEN = "hidden"
    DIRECT = "direct"
    PUBLIC = "public"


def _clean(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must not be empty")
    return cleaned


def _clean_keys(values: tuple[str, ...] | list[str], *, field_name: str) -> tuple[str, ...]:
    cleaned = tuple(dict.fromkeys(_clean(value, field_name=field_name) for value in values))
    return cleaned


@dataclass(slots=True, frozen=True)
class WorldPatch:
    """Host-authorized mutation request against a canonical world revision."""

    set_values: dict[str, Any] = field(default_factory=dict)
    delete_keys: tuple[str, ...] = field(default_factory=tuple)
    expected_revision: int | None = None

    def __post_init__(self) -> None:
        if self.expected_revision is not None and self.expected_revision < 0:
            raise ValueError("expected_revision must be >= 0")
        cleaned_set: dict[str, Any] = {}
        for key, value in self.set_values.items():
            cleaned_set[_clean(str(key), field_name="world key")] = copy.deepcopy(value)
        cleaned_delete = _clean_keys(list(self.delete_keys), field_name="world key")
        overlap = set(cleaned_set) & set(cleaned_delete)
        if overlap:
            raise ValueError(f"world patch cannot set and delete the same key: {sorted(overlap)!r}")
        object.__setattr__(self, "set_values", cleaned_set)
        object.__setattr__(self, "delete_keys", cleaned_delete)


@dataclass(slots=True, frozen=True)
class WorldFactChange:
    key: str
    kind: WorldChangeKind
    before_value: Any = None
    after_value: Any = None
    existed_before: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", _clean(self.key, field_name="world change key"))
        object.__setattr__(self, "before_value", copy.deepcopy(self.before_value))
        object.__setattr__(self, "after_value", copy.deepcopy(self.after_value))


@dataclass(slots=True, frozen=True)
class WorldStateSnapshot:
    revision: int = 0
    values: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.revision < 0:
            raise ValueError("world revision must be >= 0")
        object.__setattr__(self, "values", copy.deepcopy(dict(self.values)))


@dataclass(slots=True, frozen=True)
class WorldEvent:
    """Canonical audit record for one world occurrence and optional state change."""

    type: str
    content: str
    base_revision: int
    revision: int
    changes: tuple[WorldFactChange, ...] = field(default_factory=tuple)
    source_type: str = "host"
    source_id: str | None = None
    perception_scope: WorldPerceptionScope = WorldPerceptionScope.HIDDEN
    observer_character_ids: tuple[str, ...] = field(default_factory=tuple)
    observable_keys: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid4().hex)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        event_type = _clean(self.type, field_name="world event type")
        content = self.content.strip()
        if not content and not self.changes:
            raise ValueError("world event must contain content or changes")
        source_type = _clean(self.source_type, field_name="source_type")
        event_id = _clean(self.id, field_name="world event id")
        if self.base_revision < 0 or self.revision < 1:
            raise ValueError("world event revisions are invalid")
        if self.revision != self.base_revision + 1:
            raise ValueError("world event revision must equal base_revision + 1")
        observers = _clean_keys(list(self.observer_character_ids), field_name="observer_character_id")
        if self.perception_scope is WorldPerceptionScope.DIRECT and not observers:
            raise ValueError("direct world perception requires observer_character_ids")
        if self.perception_scope is not WorldPerceptionScope.DIRECT and observers:
            raise ValueError(f"{self.perception_scope.value} world perception must not define observers")
        change_keys = tuple(change.key for change in self.changes)
        if len(set(change_keys)) != len(change_keys):
            raise ValueError("world event changes must use unique keys")
        observable = _clean_keys(list(self.observable_keys), field_name="observable world key")
        unknown_observable = set(observable) - set(change_keys)
        if unknown_observable:
            raise ValueError(
                "observable_keys must refer to keys changed by this event: "
                f"{sorted(unknown_observable)!r}"
            )
        object.__setattr__(self, "type", event_type)
        object.__setattr__(self, "content", content)
        object.__setattr__(self, "source_type", source_type)
        object.__setattr__(self, "id", event_id)
        object.__setattr__(self, "observer_character_ids", observers)
        object.__setattr__(self, "observable_keys", observable)
        object.__setattr__(self, "metadata", copy.deepcopy(dict(self.metadata)))

    @property
    def changed_keys(self) -> tuple[str, ...]:
        return tuple(change.key for change in self.changes)

    @property
    def effective_observable_keys(self) -> tuple[str, ...]:
        return self.observable_keys or self.changed_keys


@dataclass(slots=True, frozen=True)
class WorldPerceptionProjection:
    """Policy output. Runtime still owns canonical values and validates fact keys."""

    content: str | None = None
    fact_keys: tuple[str, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.content is not None and not self.content.strip():
            raise ValueError("projection content must be non-empty when provided")
        if self.fact_keys is not None:
            object.__setattr__(
                self,
                "fact_keys",
                _clean_keys(list(self.fact_keys), field_name="projected world key"),
            )
        object.__setattr__(self, "metadata", copy.deepcopy(dict(self.metadata)))


@dataclass(slots=True, frozen=True)
class WorldObservation:
    """Character-scoped observation derived from canonical world provenance.

    The observation is evidence for a character, not authoritative Character
    State/Belief/Goal data and not permission to mutate the world.
    """

    character_id: str
    world_event_id: str
    world_revision: int
    content: str
    changes: tuple[WorldFactChange, ...] = field(default_factory=tuple)
    perception_policy: str = "explicit"
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid4().hex)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        object.__setattr__(self, "character_id", _clean(self.character_id, field_name="character_id"))
        object.__setattr__(self, "world_event_id", _clean(self.world_event_id, field_name="world_event_id"))
        object.__setattr__(self, "content", _clean(self.content, field_name="world observation content"))
        object.__setattr__(self, "perception_policy", _clean(self.perception_policy, field_name="perception_policy"))
        object.__setattr__(self, "id", _clean(self.id, field_name="world observation id"))
        if self.world_revision < 1:
            raise ValueError("world observation revision must be >= 1")
        object.__setattr__(self, "metadata", copy.deepcopy(dict(self.metadata)))
