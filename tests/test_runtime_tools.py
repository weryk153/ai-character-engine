import pytest

from ai_character_engine.character.profile import CharacterProfile
from ai_character_engine.llm.models import LLMResponse
from ai_character_engine.runtime.character_runtime import CharacterRuntime
from ai_character_engine.tools.models import ToolCall, ToolDefinition
from ai_character_engine.tools.registry import ToolRegistry
from tests.fakes import ScriptedLLMClient


@pytest.fixture
def character() -> CharacterProfile:
    return CharacterProfile(id="test", name="Test", description="Test character")


@pytest.mark.asyncio
async def test_runtime_executes_tool_and_returns_final_answer(character: CharacterProfile) -> None:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="get_current_time",
            description="Return the current local time.",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
        ),
        lambda: "14:32",
    )
    fake = ScriptedLLMClient(
        [
            LLMResponse(
                tool_calls=(
                    ToolCall(call_id="call-1", name="get_current_time", arguments={}),
                ),
                model="fake",
                input_tokens=10,
                output_tokens=1,
            ),
            LLMResponse(
                text="It is 14:32.",
                model="fake",
                input_tokens=15,
                output_tokens=5,
            ),
        ]
    )
    runtime = CharacterRuntime(character=character, llm=fake, tool_registry=registry)

    response = await runtime.run_turn("What time is it?")

    assert response.text == "It is 14:32."
    assert response.input_tokens == 25
    assert response.output_tokens == 6
    assert len(fake.calls) == 2
    second_call = fake.calls[1]
    assert any(message.role == "tool" and message.content == "14:32" for message in second_call)
    assert [message.role for message in runtime.history] == ["user", "assistant", "tool", "assistant"]


@pytest.mark.asyncio
async def test_runtime_passes_registered_tools_to_model(character: CharacterProfile) -> None:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="ping",
            description="Return pong.",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
        ),
        lambda: "pong",
    )
    fake = ScriptedLLMClient([LLMResponse(text="done")])
    runtime = CharacterRuntime(character=character, llm=fake, tool_registry=registry)

    await runtime.run_turn("hello")

    assert fake.tools_seen[0] is not None
    assert fake.tools_seen[0][0].name == "ping"


@pytest.mark.asyncio
async def test_runtime_stops_infinite_tool_loop(character: CharacterProfile) -> None:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="ping",
            description="Return pong.",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
        ),
        lambda: "pong",
    )
    repeated = LLMResponse(
        tool_calls=(ToolCall(call_id="call-1", name="ping", arguments={}),)
    )
    fake = ScriptedLLMClient([repeated, repeated])
    runtime = CharacterRuntime(
        character=character,
        llm=fake,
        tool_registry=registry,
        max_tool_rounds=2,
    )

    with pytest.raises(RuntimeError, match="max_tool_rounds"):
        await runtime.run_turn("loop forever")
