from __future__ import annotations

import asyncio

import pytest

from ai_character_engine import CharacterEvent
from ai_character_engine.character.profile import CharacterProfile
from ai_character_engine.llm import (
    BenchmarkRequest,
    BenchmarkRunner,
    InferenceBackpressureError,
    InferenceCapabilities,
    InferenceOptimizedClient,
    InferencePolicy,
    InferenceTuningHints,
    LLMResponse,
    Message,
    ModelEndpoint,
    ModelGatewayClient,
    ProfileAwareModelRouter,
    TokenCostModel,
)
from ai_character_engine.observability import InMemoryObservabilitySink, Tracer
from ai_character_engine.runtime import CharacterRuntime


class FixedClient:
    def __init__(self, *, delay: float = 0.0, fail: bool = False) -> None:
        self.delay = delay
        self.fail = fail
        self.calls = 0

    async def generate(self, messages, *, tools=None):
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError("boom")
        return LLMResponse(
            text="ok",
            model="fixed",
            input_tokens=1_000,
            output_tokens=100,
            latency_ms=self.delay * 1000 if self.delay else 1.0,
            metadata={"ttft_ms": 10.0},
        )


def test_token_cost_model_cloud_and_local() -> None:
    cloud = TokenCostModel.cloud(input_per_million=2.0, output_per_million=10.0)
    assert cloud.estimate(input_tokens=1_000_000, output_tokens=500_000) == pytest.approx(7.0)
    local = TokenCostModel.local_amortized(flat_request_cost=0.002)
    assert local.estimate(input_tokens=999, output_tokens=999) == pytest.approx(0.002)


@pytest.mark.asyncio
async def test_inference_wrapper_records_cost_profile_and_unsupported_hints() -> None:
    policy = InferencePolicy(
        profile="throughput",
        hints=InferenceTuningHints(
            prefer_continuous_batching=True,
            prefer_speculative_decoding=True,
        ),
    )
    capabilities = InferenceCapabilities(
        continuous_batching=True,
        speculative_decoding=False,
    )
    client = InferenceOptimizedClient(
        FixedClient(),
        policy=policy,
        capabilities=capabilities,
        cost_model=TokenCostModel.cloud(
            input_per_million=1.0,
            output_per_million=2.0,
            label="test-cloud",
        ),
    )
    result = await client.generate([Message(role="user", content="hi")])
    meta = result.metadata["inference"]
    assert meta["profile"] == "throughput"
    assert meta["estimated_cost"] == pytest.approx(0.0012)
    assert meta["cost_label"] == "test-cloud"
    assert meta["ttft_ms"] == 10.0
    assert meta["decode_tokens_per_second"] is not None
    assert meta["unsupported_hints"] == ["speculative_decoding"]


@pytest.mark.asyncio
async def test_backpressure_rejects_when_active_and_queue_full() -> None:
    blocker = asyncio.Event()

    class BlockingClient:
        async def generate(self, messages, *, tools=None):
            await blocker.wait()
            return LLMResponse(text="done", input_tokens=1, output_tokens=1)

    client = InferenceOptimizedClient(
        BlockingClient(),
        policy=InferencePolicy(max_concurrency=1, max_queue_size=0, request_timeout_seconds=None),
    )
    first = asyncio.create_task(client.generate([Message(role="user", content="a")]))
    await asyncio.sleep(0)
    with pytest.raises(InferenceBackpressureError):
        await client.generate([Message(role="user", content="b")])
    blocker.set()
    await first


@pytest.mark.asyncio
async def test_benchmark_runner_reports_success_error_and_throughput() -> None:
    client = InferenceOptimizedClient(
        FixedClient(delay=0.002),
        policy=InferencePolicy(max_concurrency=2, max_queue_size=10),
        cost_model=TokenCostModel.local_amortized(flat_request_cost=0.01),
    )
    runner = BenchmarkRunner(client, concurrency=2)
    report = await runner.run(
        [
            BenchmarkRequest(str(i), (Message(role="user", content="hi"),))
            for i in range(4)
        ]
    )
    assert report.success_rate == 1.0
    assert report.error_rate == 0.0
    assert report.p50_latency_ms is not None
    assert report.p95_latency_ms is not None
    assert report.throughput_requests_per_second > 0
    assert report.total_estimated_cost == pytest.approx(0.04)


@pytest.mark.asyncio
async def test_benchmark_runner_records_errors() -> None:
    runner = BenchmarkRunner(FixedClient(fail=True), concurrency=1)
    report = await runner.run(
        [BenchmarkRequest("x", (Message(role="user", content="hi"),))]
    )
    assert report.success_rate == 0.0
    assert report.error_rate == 1.0
    assert report.samples[0].error_type == "RuntimeError"


@pytest.mark.asyncio
async def test_profile_router_prefers_expected_tiers() -> None:
    fast = FixedClient()
    cheap = FixedClient()
    bulk = FixedClient()
    endpoints = (
        ModelEndpoint("fast", fast, latency_tier=0, cost_tier=3, throughput_tier=1),
        ModelEndpoint("cheap", cheap, latency_tier=2, cost_tier=0, throughput_tier=1),
        ModelEndpoint("bulk", bulk, latency_tier=2, cost_tier=2, throughput_tier=5),
    )

    low_latency = ModelGatewayClient(
        endpoints=endpoints,
        router=ProfileAwareModelRouter("low_latency"),
    )
    low_cost = ModelGatewayClient(
        endpoints=endpoints,
        router=ProfileAwareModelRouter("low_cost"),
    )
    throughput = ModelGatewayClient(
        endpoints=endpoints,
        router=ProfileAwareModelRouter("throughput"),
    )

    a = await low_latency.generate([Message(role="user", content="hi")])
    b = await low_cost.generate([Message(role="user", content="hi")])
    c = await throughput.generate([Message(role="user", content="hi")])
    assert a.metadata["gateway"]["selected_endpoint_id"] == "fast"
    assert b.metadata["gateway"]["selected_endpoint_id"] == "cheap"
    assert c.metadata["gateway"]["selected_endpoint_id"] == "bulk"


@pytest.mark.asyncio
async def test_runtime_observability_records_inference_metrics() -> None:
    sink = InMemoryObservabilitySink()
    optimized = InferenceOptimizedClient(
        FixedClient(),
        policy=InferencePolicy(profile="low_cost"),
        cost_model=TokenCostModel.local_amortized(flat_request_cost=0.005),
    )
    runtime = CharacterRuntime(
        character=CharacterProfile(id="c", name="C", description="test"),
        llm=optimized,
        tracer=Tracer(sink),
    )
    await runtime.process_event(CharacterEvent.user_message("hi"))
    turn = sink.turns[-1]
    assert turn.inference_profile == "low_cost"
    assert turn.estimated_cost == pytest.approx(0.005)
    assert turn.queue_wait_ms is not None
    assert turn.ttft_ms == 10.0
    assert turn.decode_tokens_per_second is not None
    summary = sink.summary()
    assert summary.total_estimated_cost == pytest.approx(0.005)
    assert summary.avg_queue_wait_ms is not None


def test_inference_policy_validation() -> None:
    with pytest.raises(ValueError):
        InferencePolicy(max_concurrency=0)
    with pytest.raises(ValueError):
        InferencePolicy(max_queue_size=-1)
