from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from ai_character_engine import CharacterProfile
from ai_character_engine.llm.models import LLMResponse, Message
from ai_character_engine.session import (
    CharacterRuntimeFactory,
    JsonFileRelationshipStore,
    JsonFileSessionStore,
    SessionManager,
)


class ExampleLLM:
    async def generate(self, messages: list[Message], *, tools=None) -> LLMResponse:
        last = next((message.content for message in reversed(messages) if message.role == "user"), "")
        return LLMResponse(text=f"I heard: {last}", model="example")


async def main() -> None:
    character = CharacterProfile(
        id="guide",
        name="Guide",
        description="A concise demonstration character.",
    )

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        session_path = root / "sessions.json"
        relationship_path = root / "relationships.json"

        manager = SessionManager(
            JsonFileSessionStore(session_path),
            default_ttl_seconds=None,
        )
        factory = CharacterRuntimeFactory(
            characters={character.id: character},
            llm_factory=lambda record: ExampleLLM(),
            session_manager=manager,
            relationship_store=JsonFileRelationshipStore(relationship_path),
        )

        session = factory.create(
            user_id="user-1",
            character_id="guide",
            session_id="demo-session",
        )
        session.runtime.state.trust = 72
        await session.run_turn("Remember this conversation")

        # Simulate a process restart by rebuilding stores and the factory.
        manager_after_restart = SessionManager(
            JsonFileSessionStore(session_path),
            default_ttl_seconds=None,
        )
        factory_after_restart = CharacterRuntimeFactory(
            characters={character.id: character},
            llm_factory=lambda record: ExampleLLM(),
            session_manager=manager_after_restart,
            relationship_store=JsonFileRelationshipStore(relationship_path),
        )
        restored = factory_after_restart.restore("demo-session")

        print("restored history:", [message.content for message in restored.runtime.history])
        print("restored trust:", restored.runtime.state.trust)
        print("memory scope:", restored.runtime.memory_scope_id)


if __name__ == "__main__":
    asyncio.run(main())
