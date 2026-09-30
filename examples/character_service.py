"""Offline FastAPI reference host for character integration.

Run locally after installing the optional service extra:
    pip install -e '.[service]'
    uvicorn examples.character_service:app --reload
"""

from ai_character_engine.character.profile import CharacterProfile
from ai_character_engine.llm.models import LLMResponse
from ai_character_engine.service import (
    BufferedCharacterStreamSource,
    CharacterService,
    CharacterServiceConfig,
    create_app,
)
from ai_character_engine.session import CharacterRuntimeFactory, SessionManager


class DemoLLM:
    async def generate(self, messages, *, tools=None):
        last = next((m.content for m in reversed(messages) if m.role in {"user", "event"}), "")
        return LLMResponse(
            text=f"Character heard: {last}",
            model="demo",
            input_tokens=12,
            output_tokens=6,
            latency_ms=1.0,
        )


character = CharacterProfile(
    id="demo",
    name="Demo Character",
    description="A small local service example.",
)
manager = SessionManager(default_ttl_seconds=3600)
factory = CharacterRuntimeFactory(
    characters={character.id: character},
    llm_factory=lambda record: DemoLLM(),
    session_manager=manager,
)
config = CharacterServiceConfig(
    cors_origins=("http://localhost:3000",),
    stream_chunk_chars=12,
)
service = CharacterService(
    runtime_factory=factory,
    session_manager=manager,
    stream_source=BufferedCharacterStreamSource(chunk_chars=config.stream_chunk_chars),
    default_timeout_seconds=config.default_timeout_seconds,
)
app = create_app(service=service, config=config)


if __name__ == "__main__":
    print("FastAPI app created. Routes:")
    for route in app.routes:
        path = getattr(route, "path", None)
        if path:
            print(" -", path)
