from __future__ import annotations

from typing import Protocol

from .models import WorldEvent, WorldPerceptionProjection, WorldPerceptionScope


class WorldPerceptionPolicy(Protocol):
    """Provider/host-neutral projection policy for one character and world event."""

    name: str

    def project(self, *, character_id: str, event: WorldEvent) -> WorldPerceptionProjection | None: ...


class ExplicitWorldPerceptionPolicy:
    """Small default policy based only on an event's explicit perception scope."""

    name = "explicit"

    def project(self, *, character_id: str, event: WorldEvent) -> WorldPerceptionProjection | None:
        if event.perception_scope is WorldPerceptionScope.HIDDEN:
            return None
        if event.perception_scope is WorldPerceptionScope.DIRECT:
            if character_id not in event.observer_character_ids:
                return None
        return WorldPerceptionProjection(
            content=event.content or None,
            fact_keys=event.effective_observable_keys,
        )
