from __future__ import annotations

import asyncio
from collections.abc import Iterable
from pathlib import Path

from ai_character_engine.character.profile import CharacterProfile
from ai_character_engine.events.models import CharacterEvent
from ai_character_engine.llm.models import LLMResponse, Message
from ai_character_engine.memory import JsonlMemoryStore, MemoryManager
from ai_character_engine.runtime.character_runtime import CharacterRuntime
from ai_character_engine.tools.models import ToolDefinition


class DemoLLM:
    def __init__(self, responses: Iterable[str]) -> None:
        self.responses = iter(responses)

    async def generate(
        self,
        messages: list[Message],
        *,
        tools: list[ToolDefinition] | None = None,
    ) -> LLMResponse:
        return LLMResponse(text=next(self.responses), model="demo")


async def main() -> None:
    character = CharacterProfile(
        id="demo-character",
        name="Demo",
        description="A concise demo character.",
    )
    memory_path = Path(".data/demo-memory.jsonl")

    first_runtime = CharacterRuntime(
        character=character,
        llm=DemoLLM(["記住了。"]),
        memory_manager=MemoryManager(store=JsonlMemoryStore(memory_path)),
        max_history_messages=0,
    )
    await first_runtime.run_turn("我最喜歡黑澤明的《七武士》。")

    # New runtime, empty conversation history, same persistent memory file.
    second_runtime = CharacterRuntime(
        character=character,
        llm=DemoLLM(["你之前提過《七武士》。"]),
        memory_manager=MemoryManager(store=JsonlMemoryStore(memory_path)),
        max_history_messages=0,
    )
    result = await second_runtime.process_event(
        CharacterEvent.user_message("我之前說過最喜歡哪一部黑澤明電影？")
    )

    print(result.text)
    print("Retrieved memories:")
    for memory in result.retrieved_memories:
        print(f"- score={memory.score:.3f}: {memory.record.summary}")


if __name__ == "__main__":
    asyncio.run(main())
