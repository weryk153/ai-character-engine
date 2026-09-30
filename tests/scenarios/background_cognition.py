"""Offline v0.32 demo: foreground dialogue + parallel background cognition.

No network/model is required. Deterministic fake clients demonstrate that the
foreground response returns first, while memory/emotion/summary/reflection
workers produce non-authoritative proposals in the background.
"""
from __future__ import annotations

import asyncio
import json

from ai_character_engine import (
    BackgroundCognitionConfig,
    BackgroundCognitionKind,
    BackgroundCognitionRuntime,
    BackgroundWorkerSpec,
    CharacterProfile,
    CharacterRuntime,
    CognitiveModelRouter,
    CognitiveModelRuntime,
    CognitiveRole,
    CognitiveRolePolicy,
    ModelEndpoint,
    MultiTaskRuntime,
)
from ai_character_engine.llm.models import LLMResponse


class Foreground:
    async def generate(self, messages, *, tools=None):
        return LLMResponse(text="先處理你的問題；背景分析不用等。", model="dialogue")


class JsonClient:
    def __init__(self, name: str, payload: dict):
        self.name = name
        self.payload = payload

    async def generate(self, messages, *, tools=None):
        await asyncio.sleep(0.02)
        return LLMResponse(
            text=json.dumps(self.payload, ensure_ascii=False),
            model=self.name,
        )


def cognition() -> CognitiveModelRuntime:
    clients = {
        CognitiveRole.MEMORY: JsonClient(
            "memory-small",
            {
                "items": [{
                    "summary": "使用者今天工作很煩。",
                    "kind": "event",
                    "importance": 0.6,
                    "confidence": 0.9,
                }],
                "confidence": 0.9,
                "evidence": ["今天工作很煩"],
            },
        ),
        CognitiveRole.EMOTION: JsonClient(
            "emotion-small",
            {
                "emotion": "frustrated",
                "intensity": 0.7,
                "confidence": 0.85,
                "evidence": ["工作很煩"],
            },
        ),
        CognitiveRole.SUMMARY: JsonClient(
            "summary-cheap",
            {
                "summary": "使用者正在談工作壓力。",
                "confidence": 0.8,
                "evidence": ["最近對話"],
            },
        ),
        CognitiveRole.REFLECTION: JsonClient(
            "reflection-deep",
            {
                "insight": "之後可以追問工作問題是否改善。",
                "confidence": 0.75,
                "evidence": ["未解決的工作壓力"],
            },
        ),
    }
    endpoints = tuple(
        ModelEndpoint(endpoint_id=f"{role.value}-endpoint", client=client)
        for role, client in clients.items()
    )
    policies = {
        role: CognitiveRolePolicy(primary_endpoint_ids=(f"{role.value}-endpoint",))
        for role in clients
    }
    return CognitiveModelRuntime(
        endpoints=endpoints,
        router=CognitiveModelRouter(policies=policies),
    )


async def main() -> None:
    character = CharacterRuntime(
        character=CharacterProfile(id="demo", name="Demo", description="v0.32 demo"),
        llm=Foreground(),
    )
    tasks = MultiTaskRuntime(character)
    background = BackgroundCognitionRuntime(
        tasks,
        cognition(),
        config=BackgroundCognitionConfig(
            worker_specs=(
                BackgroundWorkerSpec(BackgroundCognitionKind.MEMORY_EXTRACTION),
                BackgroundWorkerSpec(BackgroundCognitionKind.EMOTION_ANALYSIS),
                BackgroundWorkerSpec(BackgroundCognitionKind.CONVERSATION_SUMMARY),
                BackgroundWorkerSpec(BackgroundCognitionKind.REFLECTION),
            )
        ),
    )

    async with tasks:
        foreground = await background.run_turn("我今天工作很煩。")
        print("FOREGROUND:", foreground.text)
        print("REVISION AFTER FOREGROUND:", background.revision)
        print("BACKGROUND HANDLES:", len(background.handles()))

        for result in await background.collect_all():
            print("\nTASK:", result.task_type, result.status.value)
            if result.output is None:
                continue
            typed = result.output.value
            print("MODEL:", typed.model)
            print("VALUE:", typed.value)
            for proposal in result.output.proposals:
                print(
                    "PROPOSAL:", proposal.target,
                    "base_revision=", proposal.base_revision,
                    "confidence=", proposal.confidence,
                    "stale=", background.proposal_is_stale(proposal),
                )

    print("\nNo background proposal was committed automatically.")


if __name__ == "__main__":
    asyncio.run(main())
