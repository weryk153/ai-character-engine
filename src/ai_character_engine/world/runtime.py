from __future__ import annotations

import copy
import json
import threading
from collections.abc import Mapping, Sequence
from typing import Any

from ai_character_engine.events.models import CharacterEvent

from .errors import (
    WorldPerceptionDeniedError,
    WorldPerceptionProjectionError,
    WorldRevisionConflictError,
)
from .models import (
    WorldChangeKind,
    WorldEvent,
    WorldFactChange,
    WorldObservation,
    WorldPatch,
    WorldPerceptionScope,
    WorldStateSnapshot,
)
from .perception import ExplicitWorldPerceptionPolicy, WorldPerceptionPolicy
from .store import (
    InMemoryWorldObservationStore,
    InMemoryWorldStore,
    WorldObservationStore,
    WorldStore,
)


class WorldRuntime:
    """Canonical shared world state with explicit character perception boundaries.

    World mutation is an explicit host-side authority call. Character cognition
    never writes this runtime automatically. Perception produces non-authoritative
    CharacterEvent evidence and is never auto-delivered.
    """

    def __init__(
        self,
        *,
        store: WorldStore | None = None,
        observation_store: WorldObservationStore | None = None,
        perception_policy: WorldPerceptionPolicy | None = None,
    ) -> None:
        self.store = store or InMemoryWorldStore()
        self.observation_store = observation_store or InMemoryWorldObservationStore()
        self.perception_policy = perception_policy or ExplicitWorldPerceptionPolicy()
        self._lock = threading.RLock()

    def snapshot(self) -> WorldStateSnapshot:
        return self.store.snapshot()

    def event(self, event_id: str) -> WorldEvent:
        event = self.store.get_event(event_id)
        if event is None:
            raise KeyError(event_id)
        return event

    def events(self) -> tuple[WorldEvent, ...]:
        return tuple(self.store.list_events())

    def apply_event(
        self,
        *,
        type: str,
        content: str = "",
        patch: WorldPatch | None = None,
        source_type: str = "host",
        source_id: str | None = None,
        perception_scope: WorldPerceptionScope = WorldPerceptionScope.HIDDEN,
        observer_character_ids: Sequence[str] = (),
        observable_keys: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
        event_id: str | None = None,
    ) -> WorldEvent:
        """Atomically record a canonical world event and optional key/value patch."""

        patch = patch or WorldPatch()
        with self._lock:
            before = self.store.snapshot()
            if patch.expected_revision is not None and patch.expected_revision != before.revision:
                raise WorldRevisionConflictError(
                    f"stale world revision: expected {patch.expected_revision}, current {before.revision}"
                )
            values = copy.deepcopy(before.values)
            changes: list[WorldFactChange] = []
            for key, value in patch.set_values.items():
                existed = key in values
                previous = copy.deepcopy(values.get(key))
                values[key] = copy.deepcopy(value)
                changes.append(
                    WorldFactChange(
                        key=key,
                        kind=WorldChangeKind.SET,
                        before_value=previous,
                        after_value=value,
                        existed_before=existed,
                    )
                )
            for key in patch.delete_keys:
                existed = key in values
                previous = copy.deepcopy(values.get(key))
                values.pop(key, None)
                changes.append(
                    WorldFactChange(
                        key=key,
                        kind=WorldChangeKind.DELETE,
                        before_value=previous,
                        after_value=None,
                        existed_before=existed,
                    )
                )
            revision = before.revision + 1
            event_kwargs = dict(
                type=type,
                content=content,
                base_revision=before.revision,
                revision=revision,
                changes=tuple(changes),
                source_type=source_type,
                source_id=source_id,
                perception_scope=perception_scope,
                observer_character_ids=tuple(observer_character_ids),
                observable_keys=tuple(observable_keys),
                metadata=dict(metadata or {}),
            )
            if event_id is not None:
                event_kwargs["id"] = event_id
            event = WorldEvent(**event_kwargs)
            snapshot = WorldStateSnapshot(revision=revision, values=values)
            self.store.commit(snapshot, event)
            return event

    def observe_event(
        self,
        event_id: str,
        *,
        character_id: str,
        policy: WorldPerceptionPolicy | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> WorldObservation:
        """Explicitly project one canonical event into one character observation."""

        event = self.event(event_id)
        active_policy = policy or self.perception_policy
        projection = active_policy.project(character_id=character_id, event=event)
        if projection is None:
            raise WorldPerceptionDeniedError(
                f"character {character_id!r} may not perceive world event {event_id!r}"
            )
        allowed_keys = set(event.effective_observable_keys)
        requested_keys = (
            tuple(projection.fact_keys)
            if projection.fact_keys is not None
            else event.effective_observable_keys
        )
        unknown = set(requested_keys) - allowed_keys
        if unknown:
            raise WorldPerceptionProjectionError(
                f"perception policy attempted to expose non-observable keys: {sorted(unknown)!r}"
            )
        change_by_key = {change.key: change for change in event.changes}
        changes = tuple(copy.deepcopy(change_by_key[key]) for key in requested_keys if key in change_by_key)
        observation = WorldObservation(
            character_id=character_id,
            world_event_id=event.id,
            world_revision=event.revision,
            content=projection.content or event.content or _fallback_observation_content(changes),
            changes=changes,
            perception_policy=getattr(active_policy, "name", type(active_policy).__name__),
            metadata={**dict(projection.metadata), **dict(metadata or {})},
        )
        self.observation_store.add(observation)
        return observation

    def observations_for(self, character_id: str) -> tuple[WorldObservation, ...]:
        items = self.observation_store.list_for_character(character_id)
        items.sort(key=lambda item: (item.created_at, item.id))
        return tuple(items)

    def character_event(self, observation: WorldObservation) -> CharacterEvent:
        """Convert an observation into provider-neutral evidence for CharacterRuntime."""

        facts = []
        for change in observation.changes:
            facts.append(
                {
                    "key": change.key,
                    "operation": change.kind.value,
                    "value": copy.deepcopy(change.after_value),
                }
            )
        content = observation.content
        if facts:
            rendered = []
            for fact in facts:
                if fact["operation"] == WorldChangeKind.DELETE.value:
                    rendered.append(f"- {fact['key']}: <deleted>")
                else:
                    rendered.append(
                        f"- {fact['key']}: {json.dumps(fact['value'], ensure_ascii=False, sort_keys=True)}"
                    )
            content += "\nObserved world facts:\n" + "\n".join(rendered)
        return CharacterEvent(
            type="world_observation",
            source="world",
            content=content,
            payload={
                "world_observation_id": observation.id,
                "world_event_id": observation.world_event_id,
                "world_revision": observation.world_revision,
                "perception_policy": observation.perception_policy,
                "facts": facts,
                "memory_evidence_type": "event_observation",
                "authoritative": False,
                "canonical_world_provenance": True,
            },
        )


def _fallback_observation_content(changes: Sequence[WorldFactChange]) -> str:
    if changes:
        return "A world state change was observed."
    return "A world event was observed."
