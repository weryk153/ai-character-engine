"""Offline proactive-character example with deterministic fake inference."""

import asyncio
from datetime import UTC, datetime, timedelta

from ai_character_engine import CharacterProfile, CharacterRuntime
from ai_character_engine.autonomy import (
    AutonomyController,
    AutonomyPolicy,
    AutonomyScheduler,
    ProactiveCandidate,
)
from ai_character_engine.llm.models import LLMResponse


class DemoLLM:
    async def generate(self, messages, *, tools=None):
        observation = messages[-1].content.split("Event: ")[-1]
        return LLMResponse(text=f"I noticed: {observation}")


async def main() -> None:
    now = datetime.now(UTC)
    runtime = CharacterRuntime(
        character=CharacterProfile("demo", "Demo", "A careful observer"),
        llm=DemoLLM(),
    )
    scheduler = AutonomyScheduler(
        AutonomyPolicy(global_cooldown=timedelta(0))
    )
    scheduler.submit(
        ProactiveCandidate(
            content="The user has been quiet for a while.",
            source="demo_host",
            priority=40,
            dedupe_key="idle_check_in",
            cooldown_key="idle_check_in",
            created_at=now,
        )
    )
    dispatch = await AutonomyController(runtime, scheduler).run_once(now=now)
    print(dispatch.status, dispatch.run_result.text)


if __name__ == "__main__":
    asyncio.run(main())
