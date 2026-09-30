"""Offline demo: foreground dialogue plus safe background tasks."""
from __future__ import annotations

import asyncio

from ai_character_engine import (
    CharacterProfile,
    CharacterRuntime,
    MultiTaskRuntime,
    MultiTaskRuntimeConfig,
    TaskOutput,
    TaskPriority,
)
from ai_character_engine.llm.models import LLMResponse


class DemoLLM:
    async def generate(self, messages, *, tools=None):
        return LLMResponse(text="前景回覆完成。", model="demo")


async def main() -> None:
    character = CharacterProfile(id="demo", name="Hikari", description="demo character")
    runtime = CharacterRuntime(character=character, llm=DemoLLM())
    tasks = MultiTaskRuntime(
        runtime,
        config=MultiTaskRuntimeConfig(worker_count=2, max_pending_tasks=8),
    )

    async def summarize(context):
        await asyncio.sleep(0.02)
        return TaskOutput(
            value=f"summary:{context.request.payload['text']}",
            proposals=(
                context.proposal(
                    "memory",
                    {"summary": "candidate memory - not committed in v0.30"},
                ),
            ),
        )

    async def classify(context):
        await asyncio.sleep(0.01)
        return {"emotion": "neutral", "source_revision": context.snapshot.revision}

    tasks.register("summary", summarize)
    tasks.register("emotion", classify)

    async with tasks:
        summary = await tasks.submit_background(
            "summary", {"text": "今天完成了 multi-task runtime"}, priority=TaskPriority.NORMAL
        )
        emotion = await tasks.submit_background(
            "emotion", {"text": "今天完成了 multi-task runtime"}, priority=TaskPriority.HIGH
        )

        foreground = await tasks.run_turn("今天進度如何？")
        print("foreground:", foreground.text)

        for handle in (emotion, summary):
            result = await handle.wait()
            print(handle.task_id[:8], result.task_type, result.status.value, result.output.value)
            for proposal in result.output.proposals:
                print("  proposal:", proposal.target, dict(proposal.payload), "base_revision=", proposal.base_revision)

        print("authoritative revision:", tasks.revision)
        print("history messages:", len(runtime.history))


if __name__ == "__main__":
    asyncio.run(main())
