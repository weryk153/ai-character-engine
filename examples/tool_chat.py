import asyncio
import logging
import os
from datetime import datetime

from dotenv import load_dotenv

from ai_character_engine.character.profile import CharacterProfile
from ai_character_engine.llm.provider import OpenAIResponsesClient
from ai_character_engine.runtime.character_runtime import CharacterRuntime
from ai_character_engine.tools.models import ToolDefinition
from ai_character_engine.tools.registry import ToolRegistry


def build_tools() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="get_current_time",
            description="Get the current local date and time from the host running the character engine.",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
        ),
        lambda: datetime.now().isoformat(timespec="seconds"),
    )
    registry.register(
        ToolDefinition(
            name="add_numbers",
            description="Add two numbers exactly. Use this tool when exact addition is requested.",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number."},
                    "b": {"type": "number", "description": "Second number."},
                },
                "required": ["a", "b"],
                "additionalProperties": False,
            },
        ),
        lambda a, b: a + b,
    )
    return registry


async def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    model = os.getenv("AI_CHARACTER_MODEL", "").strip()
    if not model:
        raise SystemExit("Set AI_CHARACTER_MODEL in .env or the environment before starting chat.")
    character = CharacterProfile(
        id="example-researcher",
        name="Rin",
        description="A rational, curious researcher with a dry sense of humor.",
        personality=["rational", "curious", "occasionally sarcastic"],
        speaking_style=["natural conversation", "concise but substantive"],
        rules=[
            "Use an available tool when it is needed for current or exact external information.",
            "Do not invent tool results.",
        ],
    )
    runtime = CharacterRuntime(
        character=character,
        llm=OpenAIResponsesClient(model=model),
        tool_registry=build_tools(),
    )

    print("Try: 'What time is it?' or 'What is 123.4 + 567.8?' Type /exit to quit.")
    while True:
        user_input = input("You > ").strip()
        if user_input == "/exit":
            break
        response = await runtime.run_turn(user_input)
        print(f"{character.name} > {response.text}")


if __name__ == "__main__":
    asyncio.run(main())
