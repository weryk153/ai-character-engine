from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from ai_character_engine.llm.base import LLMClient
from ai_character_engine.llm.errors import LLMError
from ai_character_engine.llm.gateway import (
    GatewayCircuitBreaker,
    GatewayRequest,
    ModelEndpoint,
    ModelGatewayClient,
    StaticModelRouter,
)
from ai_character_engine.llm.models import LLMResponse, LLMStreamChunk, Message
from ai_character_engine.tools.models import ToolDefinition

from .models import CognitiveRole, CognitiveRouteDecision, CognitiveRouteRequirements
from .router import CognitiveModelRouter


class CognitiveModelRuntime:
    """Role-aware inference layer over the existing provider-neutral gateway.

    A task asks for a semantic role; this runtime chooses concrete endpoints and
    delegates provider execution/fallback to ``ModelGatewayClient``. The routing
    layer never writes CharacterState or MemoryManager.
    """

    def __init__(
        self,
        *,
        endpoints: tuple[ModelEndpoint, ...],
        router: CognitiveModelRouter,
        circuit_breaker: GatewayCircuitBreaker | None = None,
        fallback_on: tuple[type[BaseException], ...] = (LLMError, TimeoutError),
    ) -> None:
        if not endpoints:
            raise ValueError("at least one model endpoint is required")
        self.endpoints = {endpoint.endpoint_id: endpoint for endpoint in endpoints}
        if len(self.endpoints) != len(endpoints):
            raise ValueError("endpoint_id values must be unique")
        self._endpoint_tuple = endpoints
        self.router = router
        self.circuit_breaker = circuit_breaker or GatewayCircuitBreaker()
        self.fallback_on = fallback_on
        self.last_decision: CognitiveRouteDecision | None = None

    def client(
        self,
        role: CognitiveRole,
        *,
        requirements: CognitiveRouteRequirements | None = None,
    ) -> "CognitiveRoleClient":
        return CognitiveRoleClient(self, role, requirements=requirements)

    def plan(
        self,
        role: CognitiveRole,
        messages: list[Message],
        *,
        tools: list[ToolDefinition] | None = None,
        requirements: CognitiveRouteRequirements | None = None,
    ) -> CognitiveRouteDecision:
        decision = self.router.route(
            role,
            GatewayRequest(tuple(messages), tuple(tools or ())),
            self.endpoints,
            requirements=requirements,
        )
        self.last_decision = decision
        return decision

    async def generate(
        self,
        role: CognitiveRole,
        messages: list[Message],
        *,
        tools: list[ToolDefinition] | None = None,
        requirements: CognitiveRouteRequirements | None = None,
    ) -> LLMResponse:
        decision = self.plan(role, messages, tools=tools, requirements=requirements)
        gateway = self._gateway(decision)
        response = await gateway.generate(messages, tools=tools)
        return _with_cognitive_metadata(response, decision)

    async def stream_generate(
        self,
        role: CognitiveRole,
        messages: list[Message],
        *,
        tools: list[ToolDefinition] | None = None,
        requirements: CognitiveRouteRequirements | None = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        decision = self.plan(role, messages, tools=tools, requirements=requirements)
        gateway = self._gateway(decision)
        async for chunk in gateway.stream_generate(messages, tools=tools):
            if chunk.final and chunk.response is not None:
                yield LLMStreamChunk(
                    final=True,
                    response=_with_cognitive_metadata(chunk.response, decision),
                )
            else:
                yield chunk

    def _gateway(self, decision: CognitiveRouteDecision) -> ModelGatewayClient:
        return ModelGatewayClient(
            endpoints=self._endpoint_tuple,
            router=StaticModelRouter(*decision.endpoint_ids, route_name=decision.route_name),
            circuit_breaker=self.circuit_breaker,
            fallback_on=self.fallback_on,
        )


class CognitiveRoleClient:
    """Normal LLMClient/StreamingLLMClient facade bound to one cognitive role."""

    def __init__(
        self,
        runtime: CognitiveModelRuntime,
        role: CognitiveRole,
        *,
        requirements: CognitiveRouteRequirements | None = None,
    ) -> None:
        self.runtime = runtime
        self.role = role
        self.requirements = requirements

    async def generate(
        self,
        messages: list[Message],
        *,
        tools: list[ToolDefinition] | None = None,
    ) -> LLMResponse:
        return await self.runtime.generate(
            self.role,
            messages,
            tools=tools,
            requirements=self.requirements,
        )

    def stream_generate(
        self,
        messages: list[Message],
        *,
        tools: list[ToolDefinition] | None = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        return self.runtime.stream_generate(
            self.role,
            messages,
            tools=tools,
            requirements=self.requirements,
        )


def _with_cognitive_metadata(
    response: LLMResponse,
    decision: CognitiveRouteDecision,
) -> LLMResponse:
    metadata: dict[str, Any] = dict(response.metadata)
    cognitive = decision.to_metadata()
    gateway = metadata.get("gateway")
    if isinstance(gateway, dict):
        cognitive["selected_endpoint_id"] = gateway.get("selected_endpoint_id")
        cognitive["fallback_used"] = gateway.get("fallback_used")
    metadata["cognitive"] = cognitive
    return LLMResponse(
        text=response.text,
        tool_calls=response.tool_calls,
        model=response.model,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        latency_ms=response.latency_ms,
        metadata=metadata,
    )
