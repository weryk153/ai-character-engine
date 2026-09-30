from __future__ import annotations

import asyncio

import pytest

from ai_character_engine.character.profile import CharacterProfile
from ai_character_engine.cognition import (
    CognitiveModelRouter,
    CognitiveModelRuntime,
    CognitiveOptimization,
    CognitiveRole,
    CognitiveRolePolicy,
    CognitiveRouteRequirements,
    CognitiveTaskHandler,
    MissingCognitiveRolePolicyError,
    NoEligibleCognitiveModelError,
)
from ai_character_engine.llm import LLMError, LLMResponse, LLMStreamChunk, Message, ModelEndpoint
from ai_character_engine.runtime import CharacterRuntime
from ai_character_engine.tasks import MultiTaskRuntime, TaskOutput, TaskStatus
from ai_character_engine.tools import ToolDefinition
from tests.fakes import FakeLLMClient


class NamedClient:
    def __init__(self, name: str, *, fail: bool = False) -> None:
        self.name = name
        self.fail = fail
        self.calls = 0

    async def generate(self, messages, *, tools=None):
        self.calls += 1
        if self.fail:
            raise LLMError(f"{self.name} failed")
        return LLMResponse(text=f"reply:{self.name}", model=self.name)


class StreamingClient(NamedClient):
    def __init__(self, name: str, *, fail_before=False, fail_after=False) -> None:
        super().__init__(name)
        self.fail_before = fail_before
        self.fail_after = fail_after

    async def stream_generate(self, messages, *, tools=None):
        self.calls += 1
        if self.fail_before:
            raise LLMError(f"{self.name} pre-stream failure")
        yield LLMStreamChunk(text=f"{self.name}:hello")
        if self.fail_after:
            raise LLMError(f"{self.name} mid-stream failure")
        yield LLMStreamChunk(final=True, response=LLMResponse(text=f"{self.name}:hello", model=self.name))


def endpoint(name: str, client=None, **kwargs) -> ModelEndpoint:
    return ModelEndpoint(name, client or NamedClient(name), **kwargs)


def runtime_for(*endpoints: ModelEndpoint, policies, default_policy=None) -> CognitiveModelRuntime:
    return CognitiveModelRuntime(
        endpoints=endpoints,
        router=CognitiveModelRouter(policies=policies, default_policy=default_policy),
    )


def user(text="hello") -> list[Message]:
    return [Message(role="user", content=text)]


def tool() -> ToolDefinition:
    return ToolDefinition(
        name="clock",
        description="read clock",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
    )


@pytest.mark.asyncio
async def test_explicit_role_primary_endpoint_is_selected():
    fast = endpoint("fast")
    deep = endpoint("deep")
    models = runtime_for(
        fast,
        deep,
        policies={CognitiveRole.DIALOGUE: CognitiveRolePolicy(primary_endpoint_ids=("fast",), fallback_endpoint_ids=("deep",))},
    )
    response = await models.generate(CognitiveRole.DIALOGUE, user())
    assert response.text == "reply:fast"
    assert response.metadata["cognitive"]["role"] == "dialogue"
    assert response.metadata["cognitive"]["selected_endpoint_id"] == "fast"
    assert fast.client.calls == 1
    assert deep.client.calls == 0


@pytest.mark.asyncio
async def test_role_fallback_chain_uses_second_endpoint_on_failure():
    bad_client = NamedClient("bad", fail=True)
    good_client = NamedClient("good")
    models = runtime_for(
        endpoint("bad", bad_client),
        endpoint("good", good_client),
        policies={CognitiveRole.SUMMARY: CognitiveRolePolicy(primary_endpoint_ids=("bad",), fallback_endpoint_ids=("good",))},
    )
    response = await models.generate(CognitiveRole.SUMMARY, user())
    assert response.text == "reply:good"
    assert response.metadata["cognitive"]["fallback_used"] is True
    assert response.metadata["cognitive"]["selected_endpoint_id"] == "good"


def test_explicit_chain_filters_missing_capabilities_before_gateway_call():
    no_vision = endpoint("chat", capabilities=frozenset({"chat"}))
    vision = endpoint("vision", capabilities=frozenset({"chat", "vision"}))
    models = runtime_for(
        no_vision,
        vision,
        policies={
            CognitiveRole.VISION: CognitiveRolePolicy(
                primary_endpoint_ids=("chat",),
                fallback_endpoint_ids=("vision",),
                required_capabilities=frozenset({"chat", "vision"}),
            )
        },
    )
    decision = models.plan(CognitiveRole.VISION, user())
    assert decision.endpoint_ids == ("vision",)
    assert decision.rejected["chat"].startswith("missing_capabilities")


def test_tools_requirement_filters_endpoint_without_tools():
    no_tools = endpoint("small", capabilities=frozenset({"chat"}))
    with_tools = endpoint("main", capabilities=frozenset({"chat", "tools"}))
    models = runtime_for(
        no_tools,
        with_tools,
        policies={CognitiveRole.DIALOGUE: CognitiveRolePolicy(primary_endpoint_ids=("small", "main"))},
    )
    decision = models.plan(CognitiveRole.DIALOGUE, user(), tools=[tool()])
    assert decision.endpoint_ids == ("main",)
    assert decision.rejected["small"] == "missing_capabilities:tools"


def test_high_quality_policy_ranks_quality_tier_when_no_explicit_chain():
    cheap = endpoint("cheap", quality_tier=1, cost_tier=0)
    smart = endpoint("smart", quality_tier=5, cost_tier=5)
    models = runtime_for(
        cheap,
        smart,
        policies={CognitiveRole.REFLECTION: CognitiveRolePolicy(optimization=CognitiveOptimization.HIGH_QUALITY)},
    )
    decision = models.plan(CognitiveRole.REFLECTION, user())
    assert decision.endpoint_ids[0] == "smart"


def test_request_can_override_role_optimization_for_low_cost():
    cheap = endpoint("cheap", cost_tier=0, latency_tier=3, quality_tier=1)
    fast = endpoint("fast", cost_tier=4, latency_tier=0, quality_tier=4)
    models = runtime_for(
        cheap,
        fast,
        policies={CognitiveRole.MEMORY: CognitiveRolePolicy(optimization=CognitiveOptimization.HIGH_QUALITY)},
    )
    decision = models.plan(
        CognitiveRole.MEMORY,
        user(),
        requirements=CognitiveRouteRequirements(optimization=CognitiveOptimization.LOW_COST),
    )
    assert decision.endpoint_ids[0] == "cheap"


def test_preferred_tags_are_soft_priority_not_hard_requirement():
    generic = endpoint("generic", tags=frozenset(), latency_tier=0)
    zh = endpoint("zh", tags=frozenset({"zh-tw"}), latency_tier=5)
    models = runtime_for(
        generic,
        zh,
        policies={CognitiveRole.DIALOGUE: CognitiveRolePolicy(preferred_tags=frozenset({"zh-tw"}))},
    )
    decision = models.plan(CognitiveRole.DIALOGUE, user())
    assert decision.endpoint_ids[0] == "zh"


def test_request_tier_constraints_reject_expensive_or_low_quality_models():
    cheap_low = endpoint("cheap-low", cost_tier=0, quality_tier=1)
    expensive_good = endpoint("expensive-good", cost_tier=5, quality_tier=5)
    balanced = endpoint("balanced", cost_tier=2, quality_tier=3)
    models = runtime_for(
        cheap_low,
        expensive_good,
        balanced,
        policies={CognitiveRole.SUMMARY: CognitiveRolePolicy()},
    )
    decision = models.plan(
        CognitiveRole.SUMMARY,
        user(),
        requirements=CognitiveRouteRequirements(max_cost_tier=2, min_quality_tier=2),
    )
    assert decision.endpoint_ids == ("balanced",)
    assert decision.rejected["cheap-low"] == "quality_tier_too_low"
    assert decision.rejected["expensive-good"] == "cost_tier_exceeded"


def test_missing_role_policy_fails_clearly():
    models = runtime_for(endpoint("x"), policies={})
    with pytest.raises(MissingCognitiveRolePolicyError):
        models.plan(CognitiveRole.EMOTION, user())


def test_default_role_policy_can_cover_unspecified_roles():
    models = runtime_for(endpoint("x"), policies={}, default_policy=CognitiveRolePolicy())
    decision = models.plan(CognitiveRole.EMOTION, user())
    assert decision.endpoint_ids == ("x",)


def test_no_eligible_model_is_a_routing_error_before_provider_execution():
    client = NamedClient("text")
    models = runtime_for(
        endpoint("text", client, capabilities=frozenset({"chat"})),
        policies={CognitiveRole.VISION: CognitiveRolePolicy(required_capabilities=frozenset({"chat", "vision"}))},
    )
    with pytest.raises(NoEligibleCognitiveModelError):
        models.plan(CognitiveRole.VISION, user())
    assert client.calls == 0


@pytest.mark.asyncio
async def test_role_client_implements_normal_llm_client_interface():
    models = runtime_for(
        endpoint("dialogue"),
        policies={CognitiveRole.DIALOGUE: CognitiveRolePolicy(primary_endpoint_ids=("dialogue",))},
    )
    client = models.client(CognitiveRole.DIALOGUE)
    response = await client.generate(user())
    assert response.text == "reply:dialogue"
    assert response.metadata["cognitive"]["role"] == "dialogue"


@pytest.mark.asyncio
async def test_streaming_falls_back_before_any_visible_delta():
    bad = StreamingClient("bad", fail_before=True)
    good = StreamingClient("good")
    models = runtime_for(
        endpoint("bad", bad, capabilities=frozenset({"chat", "streaming"})),
        endpoint("good", good, capabilities=frozenset({"chat", "streaming"})),
        policies={CognitiveRole.DIALOGUE: CognitiveRolePolicy(primary_endpoint_ids=("bad",), fallback_endpoint_ids=("good",))},
    )
    chunks = [chunk async for chunk in models.stream_generate(CognitiveRole.DIALOGUE, user())]
    assert chunks[0].text == "good:hello"
    assert chunks[-1].response.metadata["cognitive"]["selected_endpoint_id"] == "good"
    assert chunks[-1].response.metadata["cognitive"]["fallback_used"] is True


@pytest.mark.asyncio
async def test_streaming_does_not_fallback_after_visible_delta():
    bad = StreamingClient("bad", fail_after=True)
    good = StreamingClient("good")
    models = runtime_for(
        endpoint("bad", bad, capabilities=frozenset({"chat", "streaming"})),
        endpoint("good", good, capabilities=frozenset({"chat", "streaming"})),
        policies={CognitiveRole.DIALOGUE: CognitiveRolePolicy(primary_endpoint_ids=("bad",), fallback_endpoint_ids=("good",))},
    )
    seen = []
    with pytest.raises(LLMError, match="fallback suppressed"):
        async for chunk in models.stream_generate(CognitiveRole.DIALOGUE, user()):
            seen.append(chunk.text)
    assert seen == ["bad:hello"]
    assert good.calls == 0


def test_native_streaming_requirement_filters_generate_only_endpoint():
    generated = endpoint("generated", capabilities=frozenset({"chat"}))
    streamed = endpoint("streamed", StreamingClient("streamed"), capabilities=frozenset({"chat", "streaming"}))
    models = runtime_for(
        generated,
        streamed,
        policies={CognitiveRole.DIALOGUE: CognitiveRolePolicy()},
    )
    decision = models.plan(
        CognitiveRole.DIALOGUE,
        user(),
        requirements=CognitiveRouteRequirements(require_native_streaming=True),
    )
    assert decision.endpoint_ids == ("streamed",)


@pytest.mark.asyncio
async def test_cognitive_task_handler_lets_background_task_request_role_not_provider():
    summary = NamedClient("summary-model")
    models = runtime_for(
        endpoint("summary-model", summary),
        policies={CognitiveRole.SUMMARY: CognitiveRolePolicy(primary_endpoint_ids=("summary-model",))},
    )
    character = CharacterRuntime(
        character=CharacterProfile(id="c", name="C", description="test"),
        llm=FakeLLMClient("foreground"),
    )
    tasks = MultiTaskRuntime(character)
    tasks.register(
        "summarize",
        CognitiveTaskHandler(
            models=models,
            role=CognitiveRole.SUMMARY,
            prompt_builder=lambda ctx: [Message(role="user", content=ctx.request.payload["text"])],
        ),
    )
    async with tasks:
        handle = await tasks.submit_background("summarize", {"text": "conversation"})
        result = await handle.wait()
    assert result.status is TaskStatus.SUCCEEDED
    assert result.output.value == "reply:summary-model"
    assert result.output.metadata["cognitive"]["role"] == "summary"
    assert "endpoint" not in handle.task_id.lower()


@pytest.mark.asyncio
async def test_cognitive_task_proposal_remains_non_authoritative():
    models = runtime_for(
        endpoint("memory"),
        policies={CognitiveRole.MEMORY: CognitiveRolePolicy(primary_endpoint_ids=("memory",))},
    )
    character = CharacterRuntime(
        character=CharacterProfile(id="c", name="C", description="test"),
        llm=FakeLLMClient("foreground"),
    )
    tasks = MultiTaskRuntime(character)

    def result_builder(ctx, response):
        return TaskOutput(
            value=response.text,
            proposals=(ctx.proposal("state", {"trust_delta": 40}),),
        )

    tasks.register(
        "memory-analyze",
        CognitiveTaskHandler(
            models=models,
            role=CognitiveRole.MEMORY,
            prompt_builder=lambda ctx: user("analyze"),
            result_builder=result_builder,
        ),
    )
    async with tasks:
        result = await (await tasks.submit_background("memory-analyze")).wait()
    assert result.output.proposals[0].base_revision == 0
    assert character.state.trust == 50
    assert character.history == []


def test_quality_tier_validation_is_backward_compatible_extension():
    with pytest.raises(ValueError):
        endpoint("bad-quality", quality_tier=-1)
    assert endpoint("default-quality").quality_tier == 1


def test_role_policy_rejects_duplicate_ids_across_primary_and_fallback():
    with pytest.raises(ValueError):
        CognitiveRolePolicy(primary_endpoint_ids=("x",), fallback_endpoint_ids=("x",))


def test_route_metadata_is_serializable_plain_data():
    models = runtime_for(
        endpoint("x", tags=frozenset({"local"})),
        policies={CognitiveRole.EMOTION: CognitiveRolePolicy(preferred_tags=frozenset({"local"}))},
    )
    metadata = models.plan(CognitiveRole.EMOTION, user()).to_metadata()
    assert metadata["role"] == "emotion"
    assert metadata["endpoint_ids"] == ["x"]
    assert metadata["preferred_tags"] == ["local"]

@pytest.mark.asyncio
async def test_dialogue_role_client_can_power_character_runtime_without_provider_id_in_runtime():
    models = runtime_for(
        endpoint("chat-main", NamedClient("chat-main")),
        policies={CognitiveRole.DIALOGUE: CognitiveRolePolicy(primary_endpoint_ids=("chat-main",))},
    )
    character = CharacterRuntime(
        character=CharacterProfile(id="c", name="C", description="test"),
        llm=models.client(CognitiveRole.DIALOGUE),
    )
    result = await character.process_event(__import__("ai_character_engine").CharacterEvent.user_message("hi"))
    assert result.text == "reply:chat-main"
    assert models.last_decision is not None
    assert models.last_decision.role is CognitiveRole.DIALOGUE


def test_public_api_exports_cognitive_contracts():
    import ai_character_engine as ace

    assert ace.__version__ == "1.0.0"
    assert ace.CognitiveRole.SUMMARY.value == "summary"
    assert ace.CognitiveOptimization.HIGH_QUALITY.value == "high_quality"
    assert ace.CognitiveModelRuntime is CognitiveModelRuntime
    assert ace.CognitiveTaskHandler is CognitiveTaskHandler
