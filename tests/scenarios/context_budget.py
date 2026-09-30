from __future__ import annotations

import asyncio

from ai_character_engine.character.profile import CharacterProfile
from ai_character_engine.context import ContextBudget, ContextBuilder
from ai_character_engine.llm.models import LLMResponse
from ai_character_engine.runtime.character_runtime import CharacterRuntime


class DemoLLMClient:
    async def generate(self, messages, *, tools=None) -> LLMResponse:
        return LLMResponse(
            text="ok",
            model="demo",
            input_tokens=10,
            output_tokens=2,
        )


async def main() -> None:
    character = CharacterProfile(
        id="demo",
        name="Mira",
        description="A concise AI character used to demonstrate context budgeting.",
        personality=["observant", "direct"],
    )
    builder = ContextBuilder(
        budget=ContextBudget(
            context_window_tokens=512,
            reserved_output_tokens=96,
            recent_history_target_tokens=160,
            max_memory_tokens=96,
        )
    )
    runtime = CharacterRuntime(
        character=character,
        llm=DemoLLMClient(),
        context_builder=builder,
        max_history_messages=100,
    )

    for index in range(8):
        await runtime.run_turn(
            f"conversation turn {index}: " + "some repeated context " * 8
        )

    result = await runtime.process_event(
        __import__(
            "ai_character_engine.events", fromlist=["CharacterEvent"]
        ).CharacterEvent.user_message("What did we just talk about?")
    )

    print(result.text)
    print(result.context_trace)


if __name__ == "__main__":
    asyncio.run(main())
