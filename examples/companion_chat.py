"""Chat with one character in the terminal.

The character remembers what you tell it, keeps goals and thoughts of its own,
and its mood and trust move with the conversation. Everything is stored under
``companion-data/``; run the script again and the character still knows you.

    python examples/companion_chat.py --base-url http://127.0.0.1:1234/v1 --model qwen/qwen3.5-9b
"""

from __future__ import annotations

import argparse
import asyncio

from ai_character_engine import CharacterProfile
from ai_character_engine.companion import CharacterCompanion
from ai_character_engine.llm.local import OpenAICompatibleChatClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--storage", default="companion-data/guide")
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="let a reasoning model think before it answers (off by default: the "
        "thought alone can use up the output limit)",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    def model(temperature: float) -> OpenAICompatibleChatClient:
        options = {"temperature": temperature, "max_tokens": 400}
        if not args.thinking:
            # LM Studio takes this switch; Ollama takes {"think": False}, vLLM
            # {"chat_template_kwargs": {"enable_thinking": False}}.
            options["extra_body"] = {"reasoning_effort": "none"}
        return OpenAICompatibleChatClient(
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            request_options=options,
        )

    companion = CharacterCompanion(
        character=CharacterProfile(
            id="guide",
            name="Guide",
            description="A patient, curious guide who likes to find out how things work.",
        ),
        llm=model(0.7),
        background_llm=model(0.2),
        storage_dir=args.storage,
    )
    print("Type a message. /state shows the character, /quit ends.")
    try:
        while True:
            text = (await asyncio.to_thread(input, "> ")).strip()
            if text == "/quit":
                # Closing cancels background work; what she was still taking
                # from the last turns would be lost.
                await companion.settle()
                break
            if text == "/state":
                # What she took from the conversation is worked out in the
                # background; wait for it before showing it.
                await companion.settle()
                print(companion.snapshot())
                print(companion.memories("terminal"))
                continue
            if not text:
                continue
            await companion.reply(
                text,
                conversation_id="terminal",
                on_text_delta=lambda delta: print(delta, end="", flush=True),
            )
            print()
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        await companion.close()


if __name__ == "__main__":
    asyncio.run(main())
