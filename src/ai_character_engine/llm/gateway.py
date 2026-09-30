from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Protocol

from ai_character_engine.llm.base import LLMClient
from ai_character_engine.llm.errors import LLMError
from ai_character_engine.llm.inference import InferenceProfileName
from ai_character_engine.llm.models import LLMResponse, LLMStreamChunk, Message
from ai_character_engine.tools.models import ToolDefinition


@dataclass(frozen=True, slots=True)
class ModelEndpoint:
    """One addressable model/client behind the gateway."""

    endpoint_id: str
    client: LLMClient
    capabilities: frozenset[str] = field(default_factory=lambda: frozenset({"chat", "tools"}))
    tags: frozenset[str] = field(default_factory=frozenset)
    timeout_seconds: float | None = None
    enabled: bool = True
    cost_tier: int = 1
    priority: int = 100
    latency_tier: int = 1
    throughput_tier: int = 1
    quality_tier: int = 1
    inference_profiles: frozenset[InferenceProfileName] = field(
        default_factory=lambda: frozenset({"low_latency", "balanced", "throughput", "low_cost"})
    )

    def __post_init__(self) -> None:
        if not self.endpoint_id.strip():
            raise ValueError("endpoint_id must not be empty")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0 when set")
        if self.cost_tier < 0:
            raise ValueError("cost_tier must be >= 0")
        if self.latency_tier < 0:
            raise ValueError("latency_tier must be >= 0")
        if self.throughput_tier < 0:
            raise ValueError("throughput_tier must be >= 0")
        if self.quality_tier < 0:
            raise ValueError("quality_tier must be >= 0")
        if not self.inference_profiles:
            raise ValueError("inference_profiles must not be empty")


@dataclass(frozen=True, slots=True)
class GatewayRequest:
    messages: tuple[Message, ...]
    tools: tuple[ToolDefinition, ...] = ()

    @property
    def requires_tools(self) -> bool:
        return bool(self.tools)

    @property
    def input_chars(self) -> int:
        return sum(len(message.content) for message in self.messages)


@dataclass(frozen=True, slots=True)
class RoutePlan:
    route_name: str
    endpoint_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.route_name.strip():
            raise ValueError("route_name must not be empty")
        if not self.endpoint_ids:
            raise ValueError("endpoint_ids must not be empty")


class ModelRouter(Protocol):
    def route(
        self,
        request: GatewayRequest,
        endpoints: dict[str, ModelEndpoint],
    ) -> RoutePlan: ...


@dataclass(frozen=True, slots=True)
class RoutingRule:
    name: str
    endpoint_ids: tuple[str, ...]
    require_tools: bool | None = None
    min_input_chars: int | None = None
    max_input_chars: int | None = None
    required_tags: frozenset[str] = field(default_factory=frozenset)

    def matches(self, request: GatewayRequest, endpoints: dict[str, ModelEndpoint]) -> bool:
        if self.require_tools is not None and request.requires_tools != self.require_tools:
            return False
        if self.min_input_chars is not None and request.input_chars < self.min_input_chars:
            return False
        if self.max_input_chars is not None and request.input_chars > self.max_input_chars:
            return False
        if self.required_tags:
            if not any(
                endpoint_id in endpoints
                and self.required_tags.issubset(endpoints[endpoint_id].tags)
                for endpoint_id in self.endpoint_ids
            ):
                return False
        return True




class ProfileAwareModelRouter:
    """Ranks endpoints using coarse latency/cost/throughput hints for one profile.

    Tiers are deployment hints, not measured truth. Lower latency/cost tiers are
    preferred; higher throughput tiers are preferred. Observability/benchmarks
    should be used to calibrate them in production.
    """

    def __init__(self, profile: InferenceProfileName = "balanced", *, route_name: str | None = None) -> None:
        self.profile = profile
        self.route_name = route_name or f"profile:{profile}"

    def route(self, request: GatewayRequest, endpoints: dict[str, ModelEndpoint]) -> RoutePlan:
        eligible = [
            endpoint
            for endpoint in endpoints.values()
            if endpoint.enabled and self.profile in endpoint.inference_profiles
        ]
        if request.requires_tools:
            eligible = [endpoint for endpoint in eligible if "tools" in endpoint.capabilities]
        eligible = [endpoint for endpoint in eligible if "chat" in endpoint.capabilities]
        if not eligible:
            # Preserve the gateway's normal skipped-attempt trace by returning all
            # endpoints in deterministic priority order when no profile match exists.
            eligible = sorted(endpoints.values(), key=lambda endpoint: (endpoint.priority, endpoint.endpoint_id))

        def score(endpoint: ModelEndpoint) -> tuple[float, int, str]:
            if self.profile == "low_latency":
                primary = float(endpoint.latency_tier)
            elif self.profile == "low_cost":
                primary = float(endpoint.cost_tier)
            elif self.profile == "throughput":
                primary = float(-endpoint.throughput_tier)
            else:
                primary = (endpoint.latency_tier + endpoint.cost_tier - endpoint.throughput_tier) / 3
            return (primary, endpoint.priority, endpoint.endpoint_id)

        ranked = tuple(endpoint.endpoint_id for endpoint in sorted(eligible, key=score))
        return RoutePlan(self.route_name, ranked)


class StaticModelRouter:
    def __init__(self, *endpoint_ids: str, route_name: str = "default") -> None:
        if not endpoint_ids:
            raise ValueError("at least one endpoint_id is required")
        self.plan = RoutePlan(route_name=route_name, endpoint_ids=tuple(endpoint_ids))

    def route(self, request: GatewayRequest, endpoints: dict[str, ModelEndpoint]) -> RoutePlan:
        return self.plan


class RuleBasedModelRouter:
    """Simple deterministic routing for model classes such as fast/main/reasoning."""

    def __init__(
        self,
        *,
        rules: tuple[RoutingRule, ...] = (),
        default_endpoint_ids: tuple[str, ...],
        default_route_name: str = "default",
    ) -> None:
        if not default_endpoint_ids:
            raise ValueError("default_endpoint_ids must not be empty")
        self.rules = rules
        self.default_endpoint_ids = default_endpoint_ids
        self.default_route_name = default_route_name

    def route(self, request: GatewayRequest, endpoints: dict[str, ModelEndpoint]) -> RoutePlan:
        for rule in self.rules:
            if rule.matches(request, endpoints):
                return RoutePlan(rule.name, rule.endpoint_ids)
        return RoutePlan(self.default_route_name, self.default_endpoint_ids)


@dataclass(frozen=True, slots=True)
class GatewayAttempt:
    endpoint_id: str
    success: bool
    latency_ms: float
    model: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    skipped_reason: str | None = None


@dataclass(frozen=True, slots=True)
class GatewayTrace:
    route_name: str
    selected_endpoint_id: str | None
    fallback_used: bool
    attempts: tuple[GatewayAttempt, ...]
    elapsed_ms: float

    def to_metadata(self) -> dict[str, object]:
        return {
            "route_name": self.route_name,
            "selected_endpoint_id": self.selected_endpoint_id,
            "fallback_used": self.fallback_used,
            "elapsed_ms": self.elapsed_ms,
            "attempts": [
                {
                    "endpoint_id": attempt.endpoint_id,
                    "success": attempt.success,
                    "latency_ms": attempt.latency_ms,
                    "model": attempt.model,
                    "error_type": attempt.error_type,
                    "error_message": attempt.error_message,
                    "skipped_reason": attempt.skipped_reason,
                }
                for attempt in self.attempts
            ],
        }


@dataclass(slots=True)
class EndpointHealth:
    consecutive_failures: int = 0
    opened_until: float = 0.0


class GatewayCircuitBreaker:
    """Small local circuit breaker; distributed services should use a shared health system."""

    def __init__(
        self,
        *,
        failure_threshold: int = 3,
        recovery_seconds: float = 30.0,
        clock=time.monotonic,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if recovery_seconds <= 0:
            raise ValueError("recovery_seconds must be > 0")
        self.failure_threshold = failure_threshold
        self.recovery_seconds = recovery_seconds
        self.clock = clock
        self._health: dict[str, EndpointHealth] = {}

    def allow(self, endpoint_id: str) -> bool:
        state = self._health.setdefault(endpoint_id, EndpointHealth())
        if state.opened_until <= self.clock():
            if state.opened_until:
                state.opened_until = 0.0
                state.consecutive_failures = 0
            return True
        return False

    def success(self, endpoint_id: str) -> None:
        self._health[endpoint_id] = EndpointHealth()

    def failure(self, endpoint_id: str) -> None:
        state = self._health.setdefault(endpoint_id, EndpointHealth())
        state.consecutive_failures += 1
        if state.consecutive_failures >= self.failure_threshold:
            state.opened_until = self.clock() + self.recovery_seconds


class ModelGatewayClient:
    """Provider-neutral routing/fallback layer implementing the normal LLMClient protocol."""

    def __init__(
        self,
        *,
        endpoints: tuple[ModelEndpoint, ...],
        router: ModelRouter,
        circuit_breaker: GatewayCircuitBreaker | None = None,
        fallback_on: tuple[type[BaseException], ...] = (LLMError, TimeoutError),
    ) -> None:
        if not endpoints:
            raise ValueError("at least one model endpoint is required")
        self.endpoints = {endpoint.endpoint_id: endpoint for endpoint in endpoints}
        if len(self.endpoints) != len(endpoints):
            raise ValueError("endpoint_id values must be unique")
        self.router = router
        self.circuit_breaker = circuit_breaker or GatewayCircuitBreaker()
        self.fallback_on = fallback_on
        self.last_trace: GatewayTrace | None = None

    async def stream_generate(
        self,
        messages: list[Message],
        *,
        tools: list[ToolDefinition] | None = None,
    ):
        """Route a streamed request without replaying already-emitted text.

        Gateway fallback is allowed only before an endpoint exposes user-visible
        text. Once any delta is yielded, retrying on another endpoint could make
        the host speak duplicate/conflicting prefixes, so later failures are
        surfaced immediately.
        """
        request = GatewayRequest(tuple(messages), tuple(tools or ()))
        plan = self.router.route(request, self.endpoints)
        attempts: list[GatewayAttempt] = []
        started = time.perf_counter()
        last_error: BaseException | None = None

        for route_index, endpoint_id in enumerate(plan.endpoint_ids):
            endpoint = self.endpoints.get(endpoint_id)
            if endpoint is None:
                attempts.append(GatewayAttempt(endpoint_id, False, 0.0, skipped_reason="unknown_endpoint"))
                continue
            if not endpoint.enabled:
                attempts.append(GatewayAttempt(endpoint_id, False, 0.0, skipped_reason="disabled"))
                continue
            if request.requires_tools and "tools" not in endpoint.capabilities:
                attempts.append(GatewayAttempt(endpoint_id, False, 0.0, skipped_reason="missing_tools_capability"))
                continue
            if "chat" not in endpoint.capabilities:
                attempts.append(GatewayAttempt(endpoint_id, False, 0.0, skipped_reason="missing_chat_capability"))
                continue
            if not self.circuit_breaker.allow(endpoint_id):
                attempts.append(GatewayAttempt(endpoint_id, False, 0.0, skipped_reason="circuit_open"))
                continue

            call_started = time.perf_counter()
            emitted = False
            try:
                stream_method = getattr(endpoint.client, "stream_generate", None)

                async def updates():
                    if callable(stream_method):
                        async for update in stream_method(messages, tools=tools):
                            yield update
                    else:
                        response = await endpoint.client.generate(messages, tools=tools)
                        if response.text:
                            yield LLMStreamChunk(text=response.text)
                        yield LLMStreamChunk(final=True, response=response)

                final: LLMResponse | None = None
                if endpoint.timeout_seconds is None:
                    iterator = updates()
                    async for update in iterator:
                        if update.text:
                            emitted = True
                            yield update
                        if update.final:
                            final = update.response
                else:
                    async with asyncio.timeout(endpoint.timeout_seconds):
                        iterator = updates()
                        async for update in iterator:
                            if update.text:
                                emitted = True
                                yield update
                            if update.final:
                                final = update.response

                if final is None:
                    raise LLMError(f"endpoint {endpoint_id!r} stream ended without a final response")

                latency_ms = (time.perf_counter() - call_started) * 1000
                self.circuit_breaker.success(endpoint_id)
                attempts.append(
                    GatewayAttempt(
                        endpoint_id=endpoint_id,
                        success=True,
                        latency_ms=latency_ms,
                        model=final.model,
                    )
                )
                elapsed_ms = (time.perf_counter() - started) * 1000
                trace = GatewayTrace(
                    route_name=plan.route_name,
                    selected_endpoint_id=endpoint_id,
                    fallback_used=route_index > 0,
                    attempts=tuple(attempts),
                    elapsed_ms=elapsed_ms,
                )
                self.last_trace = trace
                metadata = dict(final.metadata)
                metadata["gateway"] = trace.to_metadata()
                yield LLMStreamChunk(
                    final=True,
                    response=LLMResponse(
                        text=final.text,
                        tool_calls=final.tool_calls,
                        model=final.model,
                        input_tokens=final.input_tokens,
                        output_tokens=final.output_tokens,
                        latency_ms=elapsed_ms,
                        metadata=metadata,
                    ),
                )
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                latency_ms = (time.perf_counter() - call_started) * 1000
                attempts.append(
                    GatewayAttempt(
                        endpoint_id=endpoint_id,
                        success=False,
                        latency_ms=latency_ms,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                )
                self.circuit_breaker.failure(endpoint_id)
                last_error = exc
                if emitted:
                    elapsed_ms = (time.perf_counter() - started) * 1000
                    self.last_trace = GatewayTrace(
                        route_name=plan.route_name,
                        selected_endpoint_id=endpoint_id,
                        fallback_used=route_index > 0,
                        attempts=tuple(attempts),
                        elapsed_ms=elapsed_ms,
                    )
                    raise LLMError(
                        f"streaming endpoint {endpoint_id!r} failed after emitting text; fallback suppressed"
                    ) from exc
                if not isinstance(exc, self.fallback_on):
                    raise
                continue

        elapsed_ms = (time.perf_counter() - started) * 1000
        self.last_trace = GatewayTrace(
            route_name=plan.route_name,
            selected_endpoint_id=None,
            fallback_used=False,
            attempts=tuple(attempts),
            elapsed_ms=elapsed_ms,
        )
        if last_error is not None:
            raise LLMError(
                f"all eligible model endpoints failed for route {plan.route_name!r}: {last_error}"
            ) from last_error
        raise LLMError(f"no eligible model endpoint for route {plan.route_name!r}")

    async def generate(
        self,
        messages: list[Message],
        *,
        tools: list[ToolDefinition] | None = None,
    ) -> LLMResponse:
        request = GatewayRequest(tuple(messages), tuple(tools or ()))
        plan = self.router.route(request, self.endpoints)
        attempts: list[GatewayAttempt] = []
        started = time.perf_counter()
        last_error: BaseException | None = None

        for route_index, endpoint_id in enumerate(plan.endpoint_ids):
            endpoint = self.endpoints.get(endpoint_id)
            if endpoint is None:
                attempts.append(GatewayAttempt(endpoint_id, False, 0.0, skipped_reason="unknown_endpoint"))
                continue
            if not endpoint.enabled:
                attempts.append(GatewayAttempt(endpoint_id, False, 0.0, skipped_reason="disabled"))
                continue
            if request.requires_tools and "tools" not in endpoint.capabilities:
                attempts.append(GatewayAttempt(endpoint_id, False, 0.0, skipped_reason="missing_tools_capability"))
                continue
            if "chat" not in endpoint.capabilities:
                attempts.append(GatewayAttempt(endpoint_id, False, 0.0, skipped_reason="missing_chat_capability"))
                continue
            if not self.circuit_breaker.allow(endpoint_id):
                attempts.append(GatewayAttempt(endpoint_id, False, 0.0, skipped_reason="circuit_open"))
                continue

            call_started = time.perf_counter()
            try:
                call = endpoint.client.generate(messages, tools=tools)
                if endpoint.timeout_seconds is not None:
                    response = await asyncio.wait_for(call, timeout=endpoint.timeout_seconds)
                else:
                    response = await call
            except Exception as exc:
                latency_ms = (time.perf_counter() - call_started) * 1000
                attempts.append(
                    GatewayAttempt(
                        endpoint_id=endpoint_id,
                        success=False,
                        latency_ms=latency_ms,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                )
                self.circuit_breaker.failure(endpoint_id)
                last_error = exc
                if not isinstance(exc, self.fallback_on):
                    raise
                continue

            latency_ms = (time.perf_counter() - call_started) * 1000
            self.circuit_breaker.success(endpoint_id)
            attempts.append(
                GatewayAttempt(
                    endpoint_id=endpoint_id,
                    success=True,
                    latency_ms=latency_ms,
                    model=response.model,
                )
            )
            elapsed_ms = (time.perf_counter() - started) * 1000
            trace = GatewayTrace(
                route_name=plan.route_name,
                selected_endpoint_id=endpoint_id,
                fallback_used=route_index > 0,
                attempts=tuple(attempts),
                elapsed_ms=elapsed_ms,
            )
            self.last_trace = trace
            metadata = dict(response.metadata)
            metadata["gateway"] = trace.to_metadata()
            return LLMResponse(
                text=response.text,
                tool_calls=response.tool_calls,
                model=response.model,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                latency_ms=elapsed_ms,
                metadata=metadata,
            )

        elapsed_ms = (time.perf_counter() - started) * 1000
        self.last_trace = GatewayTrace(
            route_name=plan.route_name,
            selected_endpoint_id=None,
            fallback_used=False,
            attempts=tuple(attempts),
            elapsed_ms=elapsed_ms,
        )
        if last_error is not None:
            raise LLMError(
                f"all eligible model endpoints failed for route {plan.route_name!r}: {last_error}"
            ) from last_error
        raise LLMError(f"no eligible model endpoint for route {plan.route_name!r}")
