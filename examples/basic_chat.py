import asyncio
import logging
import os

from dotenv import load_dotenv

from ai_character_engine.character.profile import CharacterProfile
from ai_character_engine.llm.provider import OpenAIResponsesClient
from ai_character_engine.runtime.character_runtime import CharacterRuntime


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
        rules=["Do not pretend to know real-time information you do not have."],
    )
    runtime = CharacterRuntime(
        character=character,
        llm=OpenAIResponsesClient(model=model),
    )

    print("Type /exit to quit.")
    while True:
        user_input = input("You > ").strip()
        if user_input == "/exit":
            break
        response = await runtime.run_turn(user_input)
        print(f"{character.name} > {response.text}")


if __name__ == "__main__":
    asyncio.run(main())
