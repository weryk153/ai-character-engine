from __future__ import annotations

import asyncio

from ai_character_engine.llm import (
    LLMResponse,
    Message,
    ModelEndpoint,
    ModelGatewayClient,
    RoutingRule,
    RuleBasedModelRouter,
)


class DemoClient:
    def __init__(self, name: str, *, fail: bool = False) -> None:
        self.name = name
        self.fail = fail

    async def generate(self, messages, *, tools=None):
        if self.fail:
            from ai_character_engine.llm import LLMError
            raise LLMError(f"{self.name} unavailable")
        return LLMResponse(
            text=f"response from {self.name}",
            model=self.name,
            input_tokens=42,
            output_tokens=8,
        )


async def main() -> None:
    gateway = ModelGatewayClient(
        endpoints=(
            ModelEndpoint("fast", DemoClient("fast-model"), tags=frozenset({"fast"}), cost_tier=0),
            ModelEndpoint("main", DemoClient("main-model"), tags=frozenset({"main"})),
            ModelEndpoint("backup", DemoClient("backup-model"), tags=frozenset({"backup"})),
        ),
        router=RuleBasedModelRouter(
            rules=(
                RoutingRule("long-context", ("main", "backup"), min_input_chars=100),
            ),
            default_endpoint_ids=("fast", "main", "backup"),
        ),
    )

    short = await gateway.generate([Message(role="user", content="hello")])
    long = await gateway.generate([Message(role="user", content="x" * 120)])
    print(short.text)
    print(short.metadata["gateway"])
    print(long.text)
    print(long.metadata["gateway"])


if __name__ == "__main__":
    asyncio.run(main())
