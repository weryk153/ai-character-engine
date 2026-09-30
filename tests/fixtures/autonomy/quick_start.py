import asyncio
from datetime import UTC, datetime
from ai_character_engine import CharacterProfile, CharacterRuntime
from ai_character_engine.autonomy import (
    AutonomyController, AutonomyScheduler, ProactiveCandidate,
)
from ai_character_engine.llm.models import LLMResponse

class OfflineLLM:
    async def generate(self, messages, *, tools=None):
        return LLMResponse(text="Would you like to take a break?")

async def main():
    now = datetime(2026, 9, 21, 12, tzinfo=UTC)
    runtime = CharacterRuntime(
        character=CharacterProfile("demo", "Demo", "A careful observer"),
        llm=OfflineLLM(),
    )
    scheduler = AutonomyScheduler(clock=lambda: now)
    item = ProactiveCandidate(
        content="The user has been quiet for ten minutes.",
        source="host_idle", priority=30,
        dedupe_key="idle_check_in", cooldown_key="idle_check_in",
        created_at=now,
    )
    admission = scheduler.submit(item)
    assert admission.status == "accepted"
    controller = AutonomyController(runtime, scheduler)
    result = await controller.run_once()
    assert result.status == "delivered"
    assert result.run_result.event.id == item.id
    assert len(runtime.history) == 2
    print(result.status)
    print(result.run_result.text)

if __name__ == "__main__":
    asyncio.run(main())
