import pytest

from ai_character_engine.character.profile import CharacterProfile
from ai_character_engine.context.builder import ContextBuilder
from ai_character_engine.events import CharacterEvent
from ai_character_engine.llm.models import LLMResponse
from ai_character_engine.runtime import CharacterEventLoop, CharacterRuntime
from ai_character_engine.tools import ToolCall, ToolDefinition, ToolRegistry
from tests.fakes import FakeLLMClient, ScriptedLLMClient


@pytest.fixture
def character() -> CharacterProfile:
    return CharacterProfile(id="test", name="Test", description="Test character")


def test_context_builder_keeps_environment_event_distinct(character: CharacterProfile) -> None:
    event = CharacterEvent(
        type="superchat_received",
        source="youtube",
        content="A viewer sent a donation.",
        payload={"amount": 300},
        id="event-1",
    )

    messages = ContextBuilder().build_for_event(
        character=character,
        history=[],
        event=event,
    )

    assert messages[-1].role == "event"
    assert "superchat_received" in messages[-1].content
    assert '"amount": 300' in messages[-1].content


@pytest.mark.asyncio
async def test_runtime_processes_external_event(character: CharacterProfile) -> None:
    fake = FakeLLMClient("Thanks for the support!")
    runtime = CharacterRuntime(character=character, llm=fake)
    event = CharacterEvent(
        type="superchat_received",
        source="youtube",
        content="A viewer sent a donation.",
        id="event-1",
    )

    result = await runtime.process_event(event)

    assert result.event.id == "event-1"
    assert result.text == "Thanks for the support!"
    assert result.rounds == 1
    assert [message.role for message in runtime.history] == ["event", "assistant"]


@pytest.mark.asyncio
async def test_event_can_trigger_tool_action_then_speech(character: CharacterProfile) -> None:
    expression_state = {"value": "neutral"}

    def set_expression(expression: str) -> str:
        expression_state["value"] = expression
        return f"expression:{expression}"

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="set_expression",
            description="Set the visible facial expression.",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {"type": "string"},
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
        ),
        set_expression,
    )
    fake = ScriptedLLMClient(
        [
            LLMResponse(
                tool_calls=(
                    ToolCall(
                        call_id="call-1",
                        name="set_expression",
                        arguments={"expression": "surprised"},
                    ),
                )
            ),
            LLMResponse(text="Wait, really? Thank you!"),
        ]
    )
    runtime = CharacterRuntime(
        character=character,
        llm=fake,
        tool_registry=registry,
    )
    event = CharacterEvent(
        type="superchat_received",
        source="youtube",
        content="A viewer sent a large donation.",
    )

    result = await runtime.process_event(event)

    assert expression_state["value"] == "surprised"
    assert result.text == "Wait, really? Thank you!"
    assert result.rounds == 2
    assert len(result.tool_results) == 1
    assert result.tool_results[0].name == "set_expression"
    assert [message.role for message in runtime.history] == [
        "event",
        "assistant",
        "tool",
        "assistant",
    ]


@pytest.mark.asyncio
async def test_event_loop_processes_events_fifo(character: CharacterProfile) -> None:
    fake = ScriptedLLMClient(
        [LLMResponse(text="first-response"), LLMResponse(text="second-response")]
    )
    runtime = CharacterRuntime(character=character, llm=fake)
    loop = CharacterEventLoop(runtime)

    first = CharacterEvent(type="timer", source="clock", content="First event", id="1")
    second = CharacterEvent(type="timer", source="clock", content="Second event", id="2")
    await loop.publish(first)
    await loop.publish(second)

    first_result = await loop.run_once()
    second_result = await loop.run_once()

    assert first_result.event.id == "1"
    assert first_result.text == "first-response"
    assert second_result.event.id == "2"
    assert second_result.text == "second-response"
    assert loop.pending_count == 0


@pytest.mark.asyncio
async def test_run_turn_still_behaves_as_user_chat(character: CharacterProfile) -> None:
    fake = FakeLLMClient("hello")
    runtime = CharacterRuntime(character=character, llm=fake)

    response = await runtime.run_turn("Hi")

    assert response.text == "hello"
    assert runtime.history[0].role == "user"
    assert runtime.history[0].content == "Hi"
