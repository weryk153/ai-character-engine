import asyncio
from ai_character_engine import CharacterProfile, CharacterRuntime
from ai_character_engine.autonomy import (
    AutonomyController, AutonomyScheduler, ProactiveCandidate,
)
from ai_character_engine.llm.models import LLMResponse
from ai_character_engine.memory.manager import MemoryManager

class Model:
    async def generate(self, messages, *, tools=None):
        return LLMResponse(text="Noted")

async def main():
    memory = MemoryManager()
    runtime = CharacterRuntime(
        character=CharacterProfile("demo", "Demo", "Observer"),
        llm=Model(), memory_manager=memory,
    )
    scheduler = AutonomyScheduler()
    scheduler.submit(ProactiveCandidate(content="Desk is visible", source="host"))
    result = await AutonomyController(runtime, scheduler).run_once()
    assert result.status == "delivered"
    assert memory.store.list_for_character("demo") == []
    assert len(memory.ledger.list_for_character("demo")) == 1
    assert len(runtime.history) == 2

if __name__ == "__main__":
    asyncio.run(main())
