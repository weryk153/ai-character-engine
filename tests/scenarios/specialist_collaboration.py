"""Offline v0.36 Specialist / Multi-Agent Runtime example.

No network, provider account, renderer, microphone, or host application is required.
The example demonstrates the bounded topology:
planner -> parallel specialists -> verifier.
"""

from __future__ import annotations

import asyncio
import json

from ai_character_engine.character.profile import CharacterProfile
from ai_character_engine.cognition import (
    CognitiveModelRouter,
    CognitiveModelRuntime,
    CognitiveRole,
    CognitiveRolePolicy,
)
from ai_character_engine.collaboration import SpecialistCollaborationRuntime
from ai_character_engine.llm import LLMResponse, ModelEndpoint
from ai_character_engine.runtime import CharacterRuntime
from ai_character_engine.tasks import MultiTaskRuntime
from ai_character_engine.tools import ToolDefinition, ToolRegistry


class FixedClient:
    def __init__(self, payload, *, model: str):
        self.payload = payload
        self.model = model

    async def generate(self, messages, *, tools=None):
        return LLMResponse(text=json.dumps(self.payload), model=self.model)


class ForegroundClient:
    async def generate(self, messages, *, tools=None):
        return LLMResponse(text="I can inspect the context before deciding what to do next.", model="foreground-demo")


def cognitive_models() -> CognitiveModelRuntime:
    clients = {
        CognitiveRole.PLANNER: FixedClient(
            {
                "rationale": "The request benefits from memory and tool analysis.",
                "work_items": [
                    {
                        "id": "memory-1",
                        "specialist": "memory",
                        "instruction": "Check the latest event for durable user preference evidence.",
                        "source_ids": [],
                    },
                    {
                        "id": "tool-1",
                        "specialist": "tool",
                        "instruction": "Check whether a registered tool could help later.",
                        "source_ids": [],
                    },
                ],
            },
            model="planner-demo",
        ),
        CognitiveRole.MEMORY: FixedClient(
            {
                "summary": "No durable preference is established by the selected evidence slice.",
                "confidence": 0.82,
                "evidence_source_ids": [],
                "recommendations": ["Keep the finding advisory."],
            },
            model="memory-demo",
        ),
        CognitiveRole.TOOL: FixedClient(
            {
                "summary": "The clock tool is registered and may help only if current time becomes relevant.",
                "confidence": 0.9,
                "evidence_source_ids": [],
                "recommendations": ["tool:clock"],
            },
            model="tool-demo",
        ),
        CognitiveRole.VERIFIER: FixedClient(
            {
                "decision": "accept",
                "accepted_work_item_ids": ["memory-1", "tool-1"],
                "issues": [],
                "summary": "Both findings are bounded advisory cognition and do not claim an authoritative write.",
                "confidence": 0.91,
            },
            model="verifier-demo",
        ),
    }
    endpoints = []
    policies = {}
    for role, client in clients.items():
        endpoint_id = f"demo-{role.value}"
        endpoints.append(ModelEndpoint(endpoint_id, client))
        policies[role] = CognitiveRolePolicy(primary_endpoint_ids=(endpoint_id,))
    return CognitiveModelRuntime(
        endpoints=tuple(endpoints),
        router=CognitiveModelRouter(policies=policies),
    )


async def main() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="clock",
            description="Read the host clock.",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
        ),
        lambda: "12:00",
    )
    runtime = CharacterRuntime(
        character=CharacterProfile(
            id="demo",
            name="Aki",
            description="A renderer-neutral AI character.",
        ),
        llm=ForegroundClient(),
        tool_registry=registry,
    )
    tasks = MultiTaskRuntime(runtime)
    collaboration = SpecialistCollaborationRuntime(tasks, cognitive_models())

    async with tasks:
        foreground = await tasks.run_turn("Before answering complex requests, inspect the relevant context.")
        handle = await collaboration.submit_after_foreground(
            foreground,
            objective="Identify useful memory and tool context for a future character decision.",
        )
        result = await handle.wait()

    value = result.output.value
    print("task_status:", result.status.value)
    print("base_revision:", value.base_revision)
    print("verification:", value.verification.decision.value)
    for finding in value.accepted_findings:
        print(f"- {finding.specialist_id}: {finding.summary}")
    print("authoritative proposals:", len(result.output.proposals))


if __name__ == "__main__":
    asyncio.run(main())
