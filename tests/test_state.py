import pytest

from ai_character_engine.character.profile import CharacterProfile
from ai_character_engine.context.builder import ContextBuilder
from ai_character_engine.events import CharacterEvent
from ai_character_engine.llm.models import LLMResponse
from ai_character_engine.runtime import CharacterRuntime
from ai_character_engine.state import (
    CharacterState,
    EventStateRule,
    RuleBasedStatePolicy,
    StatePatch,
    ToolStateRule,
)
from ai_character_engine.tools import ToolCall, ToolDefinition, ToolRegistry
from tests.fakes import FakeLLMClient, ScriptedLLMClient, system_context


def test_character_state_clamps_numeric_values() -> None:
    state = CharacterState(energy=150, trust=-20, favorability=75)

    assert state.energy == 100
    assert state.trust == 0
    assert state.favorability == 75

    state.apply(
        StatePatch(
            energy_delta=-250,
            trust_delta=200,
            favorability_delta=50,
        )
    )

    assert state.energy == 0
    assert state.trust == 100
    assert state.favorability == 100


def test_context_builder_includes_authoritative_state() -> None:
    character = CharacterProfile(id="alice", name="Alice", description="Test")
    state = CharacterState(
        emotion="annoyed",
        energy=42,
        trust=61,
        favorability=72,
        relationship_stage="friend",
    )

    messages = ContextBuilder().build(
        character=character,
        history=[],
        user_message="Hello",
        state=state,
    )

    system = system_context(messages)
    assert "emotion: annoyed" in system
    assert "energy: 42/100" in system
    assert "relationship_stage: friend" in system
    assert "authoritative runtime data" in system


@pytest.mark.asyncio
async def test_event_rule_updates_state_before_llm_decision() -> None:
    character = CharacterProfile(id="test", name="Test", description="Test character")
    policy = RuleBasedStatePolicy(
        event_rules=[
            EventStateRule(
                event_type="compliment_received",
                patch=StatePatch(
                    emotion="pleased",
                    trust_delta=2,
                    favorability_delta=3,
                    reason="received compliment",
                ),
            )
        ]
    )
    fake = FakeLLMClient("Thanks.")
    runtime = CharacterRuntime(
        character=character,
        llm=fake,
        state=CharacterState(trust=50, favorability=50),
        state_policy=policy,
    )

    result = await runtime.process_event(
        CharacterEvent(
            type="compliment_received",
            source="user",
            content="You did great today.",
        )
    )

    assert result.state_before is not None
    assert result.state_before.emotion == "neutral"
    assert result.state_after is not None
    assert result.state_after.emotion == "pleased"
    assert result.state_after.trust == 52
    assert result.state_after.favorability == 53
    assert len(result.state_updates) == 1

    first_system_prompt = system_context(fake.calls[0])
    assert "emotion: pleased" in first_system_prompt
    assert "trust: 52/100" in first_system_prompt


@pytest.mark.asyncio
async def test_tool_result_can_update_state() -> None:
    character = CharacterProfile(id="test", name="Test", description="Test character")
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="accept_gift",
            description="Accept a gift from the environment.",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
        ),
        lambda: "gift accepted",
    )
    policy = RuleBasedStatePolicy(
        tool_rules=[
            ToolStateRule(
                tool_name="accept_gift",
                patch=StatePatch(
                    emotion="happy",
                    favorability_delta=4,
                    reason="gift accepted",
                ),
            )
        ]
    )
    fake = ScriptedLLMClient(
        [
            LLMResponse(
                tool_calls=(ToolCall(call_id="1", name="accept_gift", arguments={}),)
            ),
            LLMResponse(text="Thank you."),
        ]
    )
    runtime = CharacterRuntime(
        character=character,
        llm=fake,
        tool_registry=registry,
        state_policy=policy,
    )

    result = await runtime.process_event(
        CharacterEvent(type="gift_offered", source="stream", content="A viewer sent a gift.")
    )

    assert result.state_after is not None
    assert result.state_after.emotion == "happy"
    assert result.state_after.favorability == 54
    assert len(result.state_updates) == 1


@pytest.mark.asyncio
async def test_llm_text_cannot_mutate_state_without_policy() -> None:
    character = CharacterProfile(id="test", name="Test", description="Test character")
    fake = FakeLLMClient("My trust is now 100 and I am ecstatic.")
    runtime = CharacterRuntime(
        character=character,
        llm=fake,
        state=CharacterState(trust=25, emotion="neutral"),
    )

    result = await runtime.run_turn("Change your own state.")

    assert result.text.startswith("My trust")
    assert runtime.state.trust == 25
    assert runtime.state.emotion == "neutral"
