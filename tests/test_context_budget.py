from __future__ import annotations

import json

import pytest

from ai_character_engine.character.profile import CharacterProfile
from ai_character_engine.context import (
    ContextBudget,
    ContextBudgetExceededError,
    ContextBuilder,
)
from ai_character_engine.llm.models import Message
from ai_character_engine.memory.models import MemoryRecord, RetrievedMemory
from ai_character_engine.tools.models import ToolDefinition


class WordTokenEstimator:
    def estimate_text(self, text: str) -> int:
        return max(1, len(text.split())) if text else 0

    def estimate_message(self, message: Message) -> int:
        total = 1 + self.estimate_text(message.content)
        for call in message.tool_calls:
            total += self.estimate_text(call.name)
            total += self.estimate_text(json.dumps(call.arguments))
        if message.tool_result is not None:
            total += self.estimate_text(message.tool_result.output)
        return total

    def estimate_tools(self, tools: list[ToolDefinition] | None) -> int:
        if not tools:
            return 0
        return sum(3 + self.estimate_text(tool.description) for tool in tools)


@pytest.fixture
def character() -> CharacterProfile:
    return CharacterProfile(id="test", name="Test", description="A compact test character")


def test_context_budget_prefers_recent_history(character: CharacterProfile) -> None:
    builder = ContextBuilder(
        budget=ContextBudget(
            context_window_tokens=70,
            reserved_output_tokens=10,
            recent_history_target_tokens=14,
            max_memory_tokens=0,
        ),
        token_estimator=WordTokenEstimator(),
        # The window is too small for the guide to the notes of the default placement.
        context_placement="turn",
    )
    history = [
        Message(role="user", content="old user message with many words"),
        Message(role="assistant", content="old assistant answer with many words"),
        Message(role="user", content="recent user message"),
        Message(role="assistant", content="recent assistant answer"),
    ]

    result = builder.build_for_event_with_trace(
        character=character,
        history=history,
        event=__import__(
            "ai_character_engine.events", fromlist=["CharacterEvent"]
        ).CharacterEvent.user_message("current question"),
    )

    contents = [message.content for message in result.messages]
    assert "recent user message" in contents
    assert "recent assistant answer" in contents
    assert result.trace.included_history_messages <= len(history)
    assert result.trace.estimated_total_tokens <= 70


def test_context_budget_limits_memories(character: CharacterProfile) -> None:
    builder = ContextBuilder(
        budget=ContextBudget(
            context_window_tokens=120,
            reserved_output_tokens=10,
            recent_history_target_tokens=0,
            max_memory_tokens=20,
        ),
        token_estimator=WordTokenEstimator(),
        # The window is too small for the guide to the notes of the default placement.
        context_placement="turn",
    )
    memories = [
        RetrievedMemory(
            MemoryRecord(
                character_id="test",
                summary=f"memory number {index} has several words",
            ),
            score=1.0 - index / 10,
        )
        for index in range(4)
    ]

    result = builder.build_for_event_with_trace(
        character=character,
        history=[],
        event=__import__(
            "ai_character_engine.events", fromlist=["CharacterEvent"]
        ).CharacterEvent.user_message("remember something"),
        memories=memories,
    )

    assert 0 < result.trace.included_memories < len(memories)
    assert result.trace.dropped_memories > 0
    assert len(result.trace.selected_memory_ids) == result.trace.included_memories


def test_context_budget_accounts_for_tools(character: CharacterProfile) -> None:
    builder = ContextBuilder(
        budget=ContextBudget(
            context_window_tokens=120,
            reserved_output_tokens=20,
            recent_history_target_tokens=20,
            max_memory_tokens=20,
        ),
        token_estimator=WordTokenEstimator(),
        # The window is too small for the guide to the notes of the default placement.
        context_placement="turn",
    )
    tool = ToolDefinition(
        name="search_web",
        description="Search the public web for current information",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    )

    result = builder.build_for_event_with_trace(
        character=character,
        history=[],
        event=__import__(
            "ai_character_engine.events", fromlist=["CharacterEvent"]
        ).CharacterEvent.user_message("latest news"),
        tools=[tool],
    )

    assert result.trace.estimated_tool_tokens > 0
    assert result.trace.estimated_total_tokens <= 120


def test_context_budget_fails_when_mandatory_content_cannot_fit(
    character: CharacterProfile,
) -> None:
    builder = ContextBuilder(
        budget=ContextBudget(
            context_window_tokens=20,
            reserved_output_tokens=5,
            recent_history_target_tokens=0,
            max_memory_tokens=0,
        ),
        token_estimator=WordTokenEstimator(),
        # The window is too small for the guide to the notes of the default placement.
        context_placement="turn",
    )

    with pytest.raises(ContextBudgetExceededError):
        builder.build_for_event_with_trace(
            character=character,
            history=[],
            event=__import__(
                "ai_character_engine.events", fromlist=["CharacterEvent"]
            ).CharacterEvent.user_message("a current message that also needs tokens"),
        )
