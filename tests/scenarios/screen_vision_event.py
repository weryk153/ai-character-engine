"""Offline screenshot -> vision observation -> CharacterRuntime example."""
from __future__ import annotations

import asyncio
import base64

from ai_character_engine.character.profile import CharacterProfile
from ai_character_engine.llm.models import LLMResponse
from ai_character_engine.runtime import CharacterRuntime
from ai_character_engine.vision import ImageInput, VisionAnalysis, VisionFrame, VisionPipeline, FrameGate

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9Z0iUAAAAASUVORK5CYII="
)


class ScreenVision:
    async def analyze(self, image, *, prompt=None):
        return VisionAnalysis(
            text="The build console shows a successful test run and no visible error banner.",
            provider="demo-screen-vlm",
            model="fake-vlm",
        )


class EventAwareLLM:
    async def generate(self, messages, *, tools=None):
        observation = next((m.content for m in reversed(messages) if m.role == "event"), "")
        return LLMResponse(text=f"I noticed the screen update: {observation}", model="fake-chat")


async def main() -> None:
    pipeline = VisionPipeline(
        provider=ScreenVision(),
        frame_gate=FrameGate(min_interval_seconds=0),
    )
    frame = VisionFrame(
        image=ImageInput.from_bytes(PNG, mime_type="image/png"),
        source_type="screenshot",
    )
    event = await pipeline.to_character_event(frame, prompt="Did the tests pass?")
    assert event is not None

    runtime = CharacterRuntime(
        character=CharacterProfile(id="mei", name="Mei", description="A research assistant."),
        llm=EventAwareLLM(),
    )
    result = await runtime.process_event(event)
    print(result.text)


if __name__ == "__main__":
    asyncio.run(main())
