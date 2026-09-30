from __future__ import annotations

import pytest

from ai_character_engine.llm import (
    GatewayCircuitBreaker,
    LLMError,
    LLMResponse,
    Message,
    ModelEndpoint,
    ModelGatewayClient,
    RoutingRule,
    RuleBasedModelRouter,
    StaticModelRouter,
)
from ai_character_engine.tools.models import ToolDefinition


class StubClient:
    def __init__(self, *, text: str = "ok", model: str = "stub", error: Exception | None = None):
        self.text = text
        self.model = model
        self.error = error
        self.calls = 0

    async def generate(self, messages, *, tools=None):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return LLMResponse(
            text=self.text,
            model=self.model,
            input_tokens=10,
            output_tokens=2,
            latency_ms=1,
            metadata={"provider": "stub"},
        )


def tool() -> ToolDefinition:
    return ToolDefinition(
        name="clock",
        description="read time",
        parameters={"type": "object", "properties": {}, "additionalProperties": False}
    )


@pytest.mark.asyncio
async def test_gateway_primary_success() -> None:
    primary = StubClient(text="primary", model="fast")
    gateway = ModelGatewayClient(
        endpoints=(ModelEndpoint("fast", primary),),
        router=StaticModelRouter("fast", route_name="chat"),
    )
    result = await gateway.generate([Message(role="user", content="hi")])
    assert result.text == "primary"
    assert result.metadata["provider"] == "stub"
    trace = result.metadata["gateway"]
    assert trace["route_name"] == "chat"
    assert trace["selected_endpoint_id"] == "fast"
    assert trace["fallback_used"] is False


@pytest.mark.asyncio
async def test_gateway_falls_back_on_llm_error() -> None:
    bad = StubClient(error=LLMError("down"))
    good = StubClient(text="fallback", model="main")
    gateway = ModelGatewayClient(
        endpoints=(ModelEndpoint("bad", bad), ModelEndpoint("good", good)),
        router=StaticModelRouter("bad", "good", route_name="main-with-fallback"),
    )
    result = await gateway.generate([Message(role="user", content="hi")])
    assert result.text == "fallback"
    assert bad.calls == good.calls == 1
    trace = result.metadata["gateway"]
    assert trace["fallback_used"] is True
    assert [x["success"] for x in trace["attempts"]] == [False, True]


@pytest.mark.asyncio
async def test_gateway_does_not_fallback_on_programming_error() -> None:
    bad = StubClient(error=ValueError("bug"))
    good = StubClient(text="should-not-run")
    gateway = ModelGatewayClient(
        endpoints=(ModelEndpoint("bad", bad), ModelEndpoint("good", good)),
        router=StaticModelRouter("bad", "good"),
    )
    with pytest.raises(ValueError):
        await gateway.generate([Message(role="user", content="hi")])
    assert good.calls == 0


@pytest.mark.asyncio
async def test_gateway_skips_endpoint_without_tool_capability() -> None:
    no_tools = StubClient(text="wrong")
    tools = StubClient(text="right")
    gateway = ModelGatewayClient(
        endpoints=(
            ModelEndpoint("fast", no_tools, capabilities=frozenset({"chat"})),
            ModelEndpoint("main", tools, capabilities=frozenset({"chat", "tools"})),
        ),
        router=StaticModelRouter("fast", "main"),
    )
    result = await gateway.generate(
        [Message(role="user", content="what time")], tools=[tool()]
    )
    assert result.text == "right"
    assert no_tools.calls == 0
    attempts = result.metadata["gateway"]["attempts"]
    assert attempts[0]["skipped_reason"] == "missing_tools_capability"


@pytest.mark.asyncio
async def test_rule_router_can_route_long_prompts_to_main_model() -> None:
    fast = StubClient(text="fast")
    main = StubClient(text="main")
    router = RuleBasedModelRouter(
        rules=(RoutingRule("long-context", ("main",), min_input_chars=100),),
        default_endpoint_ids=("fast", "main"),
    )
    gateway = ModelGatewayClient(
        endpoints=(ModelEndpoint("fast", fast), ModelEndpoint("main", main)),
        router=router,
    )
    short = await gateway.generate([Message(role="user", content="hi")])
    long = await gateway.generate([Message(role="user", content="x" * 120)])
    assert short.text == "fast"
    assert long.text == "main"
    assert long.metadata["gateway"]["route_name"] == "long-context"


@pytest.mark.asyncio
async def test_circuit_breaker_skips_repeatedly_failing_endpoint() -> None:
    clock = [0.0]
    breaker = GatewayCircuitBreaker(
        failure_threshold=1, recovery_seconds=30, clock=lambda: clock[0]
    )
    bad = StubClient(error=LLMError("down"))
    good = StubClient(text="good")
    gateway = ModelGatewayClient(
        endpoints=(ModelEndpoint("bad", bad), ModelEndpoint("good", good)),
        router=StaticModelRouter("bad", "good"),
        circuit_breaker=breaker,
    )
    await gateway.generate([Message(role="user", content="one")])
    await gateway.generate([Message(role="user", content="two")])
    assert bad.calls == 1
    assert gateway.last_trace is not None
    assert gateway.last_trace.attempts[0].skipped_reason == "circuit_open"

@pytest.mark.asyncio
async def test_gateway_timeout_falls_back() -> None:
    import asyncio

    class SlowClient:
        async def generate(self, messages, *, tools=None):
            await asyncio.sleep(0.05)
            return LLMResponse(text="too late")

    good = StubClient(text="fallback")
    gateway = ModelGatewayClient(
        endpoints=(
            ModelEndpoint("slow", SlowClient(), timeout_seconds=0.001),
            ModelEndpoint("good", good),
        ),
        router=StaticModelRouter("slow", "good"),
    )
    result = await gateway.generate([Message(role="user", content="hi")])
    assert result.text == "fallback"
    assert result.metadata["gateway"]["fallback_used"] is True
    assert result.metadata["gateway"]["attempts"][0]["error_type"] == "TimeoutError"


@pytest.mark.asyncio
async def test_skipped_primary_counts_as_fallback() -> None:
    no_tools = StubClient(text="wrong")
    tools = StubClient(text="right")
    gateway = ModelGatewayClient(
        endpoints=(
            ModelEndpoint("fast", no_tools, capabilities=frozenset({"chat"})),
            ModelEndpoint("main", tools),
        ),
        router=StaticModelRouter("fast", "main"),
    )
    result = await gateway.generate([Message(role="user", content="time")], tools=[tool()])
    assert result.metadata["gateway"]["fallback_used"] is True
