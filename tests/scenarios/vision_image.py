"""Offline single-image example using a fake VLM provider."""
from __future__ import annotations

import asyncio
import base64

from ai_character_engine.vision import ImageInput, VisionAnalysis, VisionPipeline, FrameGate

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9Z0iUAAAAASUVORK5CYII="
)


class DemoVisionProvider:
    async def analyze(self, image, *, prompt=None):
        return VisionAnalysis(
            text="A tiny demo image is visible; the important point is the provider-neutral pipeline.",
            provider="demo-vlm",
            model="fake-local-vlm",
        )


async def main() -> None:
    pipeline = VisionPipeline(
        provider=DemoVisionProvider(),
        frame_gate=FrameGate(min_interval_seconds=0),
    )
    event = await pipeline.image_event(
        ImageInput.from_bytes(PNG, mime_type="image/png"),
        prompt="What is on screen?",
        source_type="upload",
    )
    assert event is not None
    print(event.type)
    print(event.content)
    print(event.payload["vision"])


if __name__ == "__main__":
    asyncio.run(main())
