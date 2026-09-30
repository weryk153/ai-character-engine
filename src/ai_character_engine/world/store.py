from __future__ import annotations

import copy
import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from ai_character_engine.persistence import PersistenceSurface, migrate_persistence_payload, stamp_current_schema

from .errors import WorldRevisionConflictError
from .models import (
    WorldChangeKind,
    WorldEvent,
    WorldFactChange,
    WorldObservation,
    WorldPerceptionScope,
    WorldStateSnapshot,
)


class WorldStore(Protocol):
    def snapshot(self) -> WorldStateSnapshot: ...
    def get_event(self, event_id: str) -> WorldEvent | None: ...
    def list_events(self) -> list[WorldEvent]: ...
    def commit(self, snapshot: WorldStateSnapshot, event: WorldEvent) -> None: ...


class WorldObservationStore(Protocol):
    def add(self, observation: WorldObservation) -> None: ...
    def get(self, observation_id: str) -> WorldObservation | None: ...
    def list_for_character(self, character_id: str) -> list[WorldObservation]: ...


class InMemoryWorldStore:
    def __init__(
        self,
        *,
        snapshot: WorldStateSnapshot | None = None,
        events: Iterable[WorldEvent] = (),
    ) -> None:
        self._snapshot = copy.deepcopy(snapshot or WorldStateSnapshot())
        self._events = list(copy.deepcopy(tuple(events)))
        self._validate_loaded_state()

    def snapshot(self) -> WorldStateSnapshot:
        return copy.deepcopy(self._snapshot)

    def get_event(self, event_id: str) -> WorldEvent | None:
        event = next((item for item in self._events if item.id == event_id), None)
        return copy.deepcopy(event)

    def list_events(self) -> list[WorldEvent]:
        return list(copy.deepcopy(tuple(self._events)))

    def commit(self, snapshot: WorldStateSnapshot, event: WorldEvent) -> None:
        if self.get_event(event.id) is not None:
            raise ValueError(f"duplicate world event id: {event.id}")
        if self._snapshot.revision != event.base_revision:
            raise WorldRevisionConflictError(
                f"world store revision changed: expected {event.base_revision}, current {self._snapshot.revision}"
            )
        if snapshot.revision != event.revision:
            raise ValueError("snapshot revision must match event revision")
        self._snapshot = copy.deepcopy(snapshot)
        self._events.append(copy.deepcopy(event))

    def _validate_loaded_state(self) -> None:
        if not self._events:
            if self._snapshot.revision != 0:
                raise ValueError("non-zero world snapshot requires event history")
            return
        previous = 0
        ids: set[str] = set()
        for event in self._events:
            if event.id in ids:
                raise ValueError(f"duplicate world event id: {event.id}")
            ids.add(event.id)
            if event.base_revision != previous or event.revision != previous + 1:
                raise ValueError("world event history is not revision-contiguous")
            previous = event.revision
        if self._snapshot.revision != previous:
            raise ValueError("world snapshot revision does not match event history")


class InMemoryWorldObservationStore:
    def __init__(self, observations: Iterable[WorldObservation] = ()) -> None:
        self._observations = list(copy.deepcopy(tuple(observations)))

    def add(self, observation: WorldObservation) -> None:
        if self.get(observation.id) is not None:
            raise ValueError(f"duplicate world observation id: {observation.id}")
        self._observations.append(copy.deepcopy(observation))

    def get(self, observation_id: str) -> WorldObservation | None:
        found = next((item for item in self._observations if item.id == observation_id), None)
        return copy.deepcopy(found)

    def list_for_character(self, character_id: str) -> list[WorldObservation]:
        return list(
            copy.deepcopy(
                tuple(item for item in self._observations if item.character_id == character_id)
            )
        )


class JsonlWorldStore(InMemoryWorldStore):
    """Single-process local/dev persistence for canonical world state + event audit."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        snapshot = WorldStateSnapshot()
        events: list[WorldEvent] = []
        if self.path.exists():
            lines = [line for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if lines:
                header = json.loads(lines[0])
                if header.get("record_type") != "world_snapshot":
                    raise ValueError(f"{self.path}:1: expected world_snapshot")
                header = dict(migrate_persistence_payload(PersistenceSurface.WORLD_SNAPSHOT, header).payload)
                snapshot = _snapshot_from_dict(header)
                for line_number, line in enumerate(lines[1:], start=2):
                    payload = json.loads(line)
                    if payload.get("record_type") != "world_event":
                        raise ValueError(f"{self.path}:{line_number}: unknown record_type")
                    payload = dict(migrate_persistence_payload(PersistenceSurface.WORLD_EVENT, payload).payload)
                    events.append(_event_from_dict(payload))
        super().__init__(snapshot=snapshot, events=events)

    def commit(self, snapshot: WorldStateSnapshot, event: WorldEvent) -> None:
        super().commit(snapshot, event)
        self._rewrite()

    def _rewrite(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(_snapshot_to_dict(self._snapshot), ensure_ascii=False, sort_keys=True)]
        lines.extend(
            json.dumps(_event_to_dict(item), ensure_ascii=False, sort_keys=True)
            for item in self._events
        )
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        tmp.replace(self.path)


class JsonlWorldObservationStore(InMemoryWorldObservationStore):
    """Single-process local/dev observation audit; not a character Memory store."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        observations: list[WorldObservation] = []
        if self.path.exists():
            for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if payload.get("record_type") != "world_observation":
                    raise ValueError(f"{self.path}:{line_number}: unknown record_type")
                payload = dict(migrate_persistence_payload(PersistenceSurface.WORLD_OBSERVATION, payload).payload)
                observations.append(_observation_from_dict(payload))
        super().__init__(observations)

    def add(self, observation: WorldObservation) -> None:
        super().add(observation)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_observation_to_dict(observation), ensure_ascii=False, sort_keys=True) + "\n")


def _snapshot_to_dict(snapshot: WorldStateSnapshot) -> dict[str, Any]:
    return stamp_current_schema(PersistenceSurface.WORLD_SNAPSHOT, {
        "record_type": "world_snapshot",
        "revision": snapshot.revision,
        "values": _jsonable(snapshot.values),
    })


def _snapshot_from_dict(payload: dict[str, Any]) -> WorldStateSnapshot:
    return WorldStateSnapshot(revision=int(payload["revision"]), values=dict(payload.get("values", {})))


def _change_to_dict(change: WorldFactChange) -> dict[str, Any]:
    return {
        "key": change.key,
        "kind": change.kind.value,
        "before_value": _jsonable(change.before_value),
        "after_value": _jsonable(change.after_value),
        "existed_before": change.existed_before,
    }


def _change_from_dict(payload: dict[str, Any]) -> WorldFactChange:
    return WorldFactChange(
        key=payload["key"],
        kind=WorldChangeKind(payload["kind"]),
        before_value=payload.get("before_value"),
        after_value=payload.get("after_value"),
        existed_before=bool(payload.get("existed_before", False)),
    )


def _event_to_dict(event: WorldEvent) -> dict[str, Any]:
    return stamp_current_schema(PersistenceSurface.WORLD_EVENT, {
        "record_type": "world_event",
        "id": event.id,
        "type": event.type,
        "content": event.content,
        "base_revision": event.base_revision,
        "revision": event.revision,
        "changes": [_change_to_dict(change) for change in event.changes],
        "source_type": event.source_type,
        "source_id": event.source_id,
        "perception_scope": event.perception_scope.value,
        "observer_character_ids": list(event.observer_character_ids),
        "observable_keys": list(event.observable_keys),
        "metadata": _jsonable(event.metadata),
        "created_at": event.created_at.isoformat(),
    })


def _event_from_dict(payload: dict[str, Any]) -> WorldEvent:
    return WorldEvent(
        id=payload["id"],
        type=payload["type"],
        content=payload.get("content", ""),
        base_revision=int(payload["base_revision"]),
        revision=int(payload["revision"]),
        changes=tuple(_change_from_dict(item) for item in payload.get("changes", ())),
        source_type=payload.get("source_type", "host"),
        source_id=payload.get("source_id"),
        perception_scope=WorldPerceptionScope(payload.get("perception_scope", "hidden")),
        observer_character_ids=tuple(payload.get("observer_character_ids", ())),
        observable_keys=tuple(payload.get("observable_keys", ())),
        metadata=dict(payload.get("metadata", {})),
        created_at=datetime.fromisoformat(payload["created_at"]),
    )


def _observation_to_dict(observation: WorldObservation) -> dict[str, Any]:
    return stamp_current_schema(PersistenceSurface.WORLD_OBSERVATION, {
        "record_type": "world_observation",
        "id": observation.id,
        "character_id": observation.character_id,
        "world_event_id": observation.world_event_id,
        "world_revision": observation.world_revision,
        "content": observation.content,
        "changes": [_change_to_dict(change) for change in observation.changes],
        "perception_policy": observation.perception_policy,
        "metadata": _jsonable(observation.metadata),
        "created_at": observation.created_at.isoformat(),
    })


def _observation_from_dict(payload: dict[str, Any]) -> WorldObservation:
    return WorldObservation(
        id=payload["id"],
        character_id=payload["character_id"],
        world_event_id=payload["world_event_id"],
        world_revision=int(payload["world_revision"]),
        content=payload["content"],
        changes=tuple(_change_from_dict(item) for item in payload.get("changes", ())),
        perception_policy=payload.get("perception_policy", "explicit"),
        metadata=dict(payload.get("metadata", {})),
        created_at=datetime.fromisoformat(payload["created_at"]),
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
