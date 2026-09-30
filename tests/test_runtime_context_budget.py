import pytest

from ai_character_engine.character.profile import CharacterProfile
from ai_character_engine.context import ContextBudget, ContextBuilder
from ai_character_engine.runtime.character_runtime import CharacterRuntime
from tests.fakes import FakeLLMClient


@pytest.mark.asyncio
async def test_runtime_exposes_context_trace_and_records_current_turn() -> None:
    runtime = CharacterRuntime(
        character=CharacterProfile(
            id="test",
            name="Test",
            description="A test character",
        ),
        llm=FakeLLMClient("ok"),
        context_builder=ContextBuilder(
            budget=ContextBudget(
                context_window_tokens=220,
                reserved_output_tokens=30,
                recent_history_target_tokens=25,
                max_memory_tokens=0,
            ),
            # The window is too small for the guide to the notes of the default placement.
            context_placement="turn",
        ),
        max_history_messages=100,
    )

    for index in range(6):
        await runtime.run_turn(f"older message {index} " + "word " * 12)

    result = await runtime.process_event(
        __import__(
            "ai_character_engine.events", fromlist=["CharacterEvent"]
        ).CharacterEvent.user_message("current message")
    )

    assert result.context_trace is not None
    assert result.context_trace.dropped_history_messages > 0
    # Context selection must not break persistent session history recording.
    assert runtime.history[-2].content == "current message"
    assert runtime.history[-1].content == "ok"
