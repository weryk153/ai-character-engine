from __future__ import annotations

from typing import Protocol

from ai_character_engine.events.models import CharacterEvent
from ai_character_engine.llm.models import LLMResponse
from ai_character_engine.state.models import CharacterStateSnapshot


class MemorySummarizer(Protocol):
    def summarize(
        self,
        *,
        event: CharacterEvent,
        response: LLMResponse,
        state_before: CharacterStateSnapshot,
        state_after: CharacterStateSnapshot,
    ) -> str:
        ...


class HeuristicMemorySummarizer:
    """No-extra-LLM-call summarizer used by the reference implementation."""

    def __init__(self, *, max_chars: int = 500) -> None:
        self.max_chars = max_chars

    def summarize(
        self,
        *,
        event: CharacterEvent,
        response: LLMResponse,
        state_before: CharacterStateSnapshot,
        state_after: CharacterStateSnapshot,
    ) -> str:
        if event.type == "user_message":
            base = f"User said: {event.content.strip()}"
        else:
            base = f"Event {event.type} from {event.source}: {event.content.strip()}"

        # A generated answer is not independent evidence about the user. Saving
        # it into another user memory can resurrect a corrected/forgotten fact
        # through a later recall question, or persist a model hallucination.
        # Full assistant replies remain in history and the append-only ledger.
        if event.type != "user_message" and response.text.strip():
            base += f" Character replied: {response.text.strip()}"

        changes: list[str] = []
        if state_before.emotion != state_after.emotion:
            changes.append(f"emotion {state_before.emotion}->{state_after.emotion}")
        if state_before.trust != state_after.trust:
            changes.append(f"trust {state_before.trust:g}->{state_after.trust:g}")
        if state_before.favorability != state_after.favorability:
            changes.append(
                f"favorability {state_before.favorability:g}->{state_after.favorability:g}"
            )
        if state_before.relationship_stage != state_after.relationship_stage:
            changes.append(
                "relationship_stage "
                f"{state_before.relationship_stage}->{state_after.relationship_stage}"
            )
        if changes:
            base += " State changed: " + ", ".join(changes) + "."

        return self._truncate(base)

    def _truncate(self, text: str) -> str:
        if len(text) <= self.max_chars:
            return text
        return text[: self.max_chars - 1].rstrip() + "…"
