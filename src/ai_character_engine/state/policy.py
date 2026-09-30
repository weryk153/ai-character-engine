from __future__ import annotations

from typing import Protocol

from ai_character_engine.events.models import CharacterEvent
from ai_character_engine.state.models import CharacterStateSnapshot, StatePatch
from ai_character_engine.tools.models import ToolResult


class CharacterStatePolicy(Protocol):
    """Decides deterministic state transitions outside the LLM."""

    def on_event(
        self,
        event: CharacterEvent,
        state: CharacterStateSnapshot,
    ) -> StatePatch | None:
        ...

    def on_tool_result(
        self,
        result: ToolResult,
        state: CharacterStateSnapshot,
    ) -> StatePatch | None:
        ...


class NoopStatePolicy:
    def on_event(
        self,
        event: CharacterEvent,
        state: CharacterStateSnapshot,
    ) -> StatePatch | None:
        return None

    def on_tool_result(
        self,
        result: ToolResult,
        state: CharacterStateSnapshot,
    ) -> StatePatch | None:
        return None
