from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ai_character_engine.events.models import CharacterEvent
from ai_character_engine.state.models import CharacterStateSnapshot, StatePatch
from ai_character_engine.tools.models import ToolResult

EventPredicate = Callable[[CharacterEvent], bool]
ToolPredicate = Callable[[ToolResult], bool]


@dataclass(slots=True, frozen=True)
class EventStateRule:
    event_type: str
    patch: StatePatch
    source: str | None = None
    predicate: EventPredicate | None = None

    def matches(self, event: CharacterEvent) -> bool:
        if event.type != self.event_type:
            return False
        if self.source is not None and event.source != self.source:
            return False
        return self.predicate(event) if self.predicate is not None else True


@dataclass(slots=True, frozen=True)
class ToolStateRule:
    tool_name: str
    patch: StatePatch
    predicate: ToolPredicate | None = None

    def matches(self, result: ToolResult) -> bool:
        if result.name != self.tool_name:
            return False
        return self.predicate(result) if self.predicate is not None else True


class RuleBasedStatePolicy:
    """Small deterministic rule engine for character state transitions.

    Rules are host-defined. The engine intentionally does not decide that a
    donation, compliment, purchase, game result, etc. *must* change a character
    in a particular way.
    """

    def __init__(
        self,
        *,
        event_rules: list[EventStateRule] | None = None,
        tool_rules: list[ToolStateRule] | None = None,
    ) -> None:
        self.event_rules = list(event_rules or [])
        self.tool_rules = list(tool_rules or [])

    def on_event(
        self,
        event: CharacterEvent,
        state: CharacterStateSnapshot,
    ) -> StatePatch | None:
        return self._combine(rule.patch for rule in self.event_rules if rule.matches(event))

    def on_tool_result(
        self,
        result: ToolResult,
        state: CharacterStateSnapshot,
    ) -> StatePatch | None:
        return self._combine(rule.patch for rule in self.tool_rules if rule.matches(result))

    @staticmethod
    def _combine(patches) -> StatePatch | None:
        selected = list(patches)
        if not selected:
            return None

        emotion = next((p.emotion for p in reversed(selected) if p.emotion is not None), None)
        relationship_stage = next(
            (p.relationship_stage for p in reversed(selected) if p.relationship_stage is not None),
            None,
        )
        custom_updates = {}
        reasons: list[str] = []
        for patch in selected:
            custom_updates.update(patch.custom_updates)
            if patch.reason:
                reasons.append(patch.reason)

        return StatePatch(
            emotion=emotion,
            energy_delta=sum(p.energy_delta for p in selected),
            trust_delta=sum(p.trust_delta for p in selected),
            favorability_delta=sum(p.favorability_delta for p in selected),
            relationship_stage=relationship_stage,
            custom_updates=custom_updates,
            reason="; ".join(reasons) or None,
        )
