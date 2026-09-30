"""Offline demo: semantic cognitive roles routed to different fake models.

Run:
    PYTHONPATH=src python tests/scenarios/cognitive_routing.py
"""
from __future__ import annotations

import asyncio

from ai_character_engine import (
    CharacterProfile,
    CharacterRuntime,
    CognitiveModelRouter,
    CognitiveModelRuntime,
    CognitiveOptimization,
    CognitiveRole,
    CognitiveRolePolicy,
    CognitiveTaskHandler,
    ModelEndpoint,
    MultiTaskRuntime,
)
from ai_character_engine.llm import LLMResponse, Message


class DemoModel:
    def __init__(self, name: str) -> None:
        self.name = name

    async def generate(self, messages, *, tools=None):
        prompt = messages[-1].content if messages else ""
        return LLMResponse(
            text=f"[{self.name}] {prompt[:72]}",
            model=self.name,
        )


async def main() -> None:
    endpoints = (
        ModelEndpoint(
            "fast-chat",
            DemoModel("fast-chat"),
            tags=frozenset({"local", "chat"}),
            latency_tier=0,
            cost_tier=1,
            quality_tier=3,
        ),
        ModelEndpoint(
            "cheap-worker",
            DemoModel("cheap-worker"),
            tags=frozenset({"local", "worker"}),
            latency_tier=1,
            cost_tier=0,
            quality_tier=2,
        ),
        ModelEndpoint(
            "deep-reflection",
            DemoModel("deep-reflection"),
            tags=frozenset({"reasoning"}),
            latency_tier=3,
            cost_tier=3,
            quality_tier=5,
        ),
    )

    router = CognitiveModelRouter(
        policies={
            CognitiveRole.DIALOGUE: CognitiveRolePolicy(
                primary_endpoint_ids=("fast-chat",),
                fallback_endpoint_ids=("deep-reflection",),
                optimization=CognitiveOptimization.LOW_LATENCY,
            ),
            CognitiveRole.SUMMARY: CognitiveRolePolicy(
                primary_endpoint_ids=("cheap-worker",),
                fallback_endpoint_ids=("fast-chat",),
                optimization=CognitiveOptimization.LOW_COST,
            ),
            CognitiveRole.REFLECTION: CognitiveRolePolicy(
                primary_endpoint_ids=("deep-reflection",),
                fallback_endpoint_ids=("fast-chat",),
                optimization=CognitiveOptimization.HIGH_QUALITY,
            ),
        }
    )
    cognition = CognitiveModelRuntime(endpoints=endpoints, router=router)

    character = CharacterRuntime(
        character=CharacterProfile(id="demo", name="燈", description="routing demo"),
        llm=cognition.client(CognitiveRole.DIALOGUE),
    )
    tasks = MultiTaskRuntime(character)
    tasks.register(
        "summary",
        CognitiveTaskHandler(
            models=cognition,
            role=CognitiveRole.SUMMARY,
            prompt_builder=lambda ctx: [
                Message(role="user", content=f"Summarize: {ctx.request.payload['text']}")
            ],
        ),
    )
    tasks.register(
        "reflection",
        CognitiveTaskHandler(
            models=cognition,
            role=CognitiveRole.REFLECTION,
            prompt_builder=lambda ctx: [
                Message(role="user", content=f"Reflect on: {ctx.request.payload['text']}")
            ],
        ),
    )

    async with tasks:
        foreground = await tasks.run_turn("你好，今天先做角色對話。")
        summary = await (await tasks.submit_background("summary", {"text": "今天完成模型路由。"})).wait()
        reflection = await (
            await tasks.submit_background("reflection", {"text": "模型路由的風險是什麼？"})
        ).wait()

    print("foreground:", foreground.text)
    print("summary:", summary.output.value)
    print("summary model:", summary.output.metadata["cognitive"]["selected_endpoint_id"])
    print("reflection:", reflection.output.value)
    print("reflection model:", reflection.output.metadata["cognitive"]["selected_endpoint_id"])
    print("authoritative revision:", tasks.revision)


if __name__ == "__main__":
    asyncio.run(main())
