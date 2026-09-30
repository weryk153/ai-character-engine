from __future__ import annotations

import asyncio

from ai_character_engine.llm import (
    BenchmarkRequest,
    BenchmarkRunner,
    InferenceCapabilities,
    InferenceOptimizedClient,
    InferencePolicy,
    LLMResponse,
    Message,
    TokenCostModel,
)


class DemoClient:
    async def generate(self, messages, *, tools=None):
        await asyncio.sleep(0.01)
        return LLMResponse(
            text="demo",
            model="demo-model",
            input_tokens=800,
            output_tokens=80,
            latency_ms=10.0,
            metadata={"ttft_ms": 3.0},
        )


async def main() -> None:
    client = InferenceOptimizedClient(
        DemoClient(),
        policy=InferencePolicy.low_latency(max_queue_size=8),
        capabilities=InferenceCapabilities(
            kv_cache=True,
            prefix_caching=True,
            continuous_batching=True,
        ),
        cost_model=TokenCostModel.cloud(
            input_per_million=1.0,
            output_per_million=4.0,
            label="demo-cloud",
        ),
    )
    response = await client.generate([Message(role="user", content="hello")])
    print("inference metadata:", response.metadata["inference"])

    runner = BenchmarkRunner(client, concurrency=4)
    report = await runner.run(
        [
            BenchmarkRequest(
                request_id=f"req-{i}",
                messages=(Message(role="user", content=f"hello {i}"),),
            )
            for i in range(8)
        ]
    )
    print("success_rate:", report.success_rate)
    print("throughput_rps:", round(report.throughput_requests_per_second, 2))
    print("p50_ms:", round(report.p50_latency_ms or 0, 2))
    print("p95_ms:", round(report.p95_latency_ms or 0, 2))
    print("estimated_cost:", round(report.total_estimated_cost, 6))


if __name__ == "__main__":
    asyncio.run(main())
