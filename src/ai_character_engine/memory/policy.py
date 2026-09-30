from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ai_character_engine.events.models import CharacterEvent
from ai_character_engine.llm.models import LLMResponse
from ai_character_engine.state.models import CharacterStateSnapshot

from .evidence import classify_memory_evidence


class MemoryWritePolicy(Protocol):
    def importance(
        self,
        *,
        event: CharacterEvent,
        response: LLMResponse,
        state_before: CharacterStateSnapshot,
        state_after: CharacterStateSnapshot,
    ) -> float | None:
        """Return 0..1 to write a memory, or None to skip it."""
        ...


@dataclass(slots=True)
class DefaultMemoryWritePolicy:
    """Simple deterministic write policy suitable for learning and tests.

    Host applications can replace this policy with domain rules or an LLM
    classifier later. An event payload may provide `memory_importance` when the
    host already knows that an event is especially important.
    """

    minimum_importance: float = 0.35
    default_importance: float = 0.45
    importance_by_event_type: dict[str, float] = field(
        default_factory=lambda: {
            "superchat_received": 0.8,
            "gift_received": 0.75,
            "relationship_milestone": 0.95,
            "quest_completed": 0.75,
        }
    )

    def importance(
        self,
        *,
        event: CharacterEvent,
        response: LLMResponse,
        state_before: CharacterStateSnapshot,
        state_after: CharacterStateSnapshot,
    ) -> float | None:
        explicit = event.payload.get("memory_importance")
        if explicit is not None:
            score = max(0.0, min(1.0, float(explicit)))
        else:
            evidence_type = classify_memory_evidence(event)
            # Questions, quoted/reference text and one-off instructions are
            # valuable audit/history data but are not user facts by default.
            # Hosts may opt them in explicitly with memory_importance.
            if event.type == "user_message" and evidence_type in {
                "user_question", "quoted_reference", "user_instruction", "memory_operation"
            }:
                return None
            score = self.importance_by_event_type.get(
                event.type,
                self.default_importance,
            )

        # State transitions make an interaction more memorable without allowing
        # the LLM to write arbitrary relationship values.
        relationship_change = (
            abs(state_after.trust - state_before.trust)
            + abs(state_after.favorability - state_before.favorability)
        )
        if relationship_change >= 5:
            score = min(1.0, score + 0.15)

        if score < self.minimum_importance:
            return None
        return score
