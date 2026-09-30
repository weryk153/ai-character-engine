"""Offline game-frame sampling/dedup example."""
from __future__ import annotations

import asyncio
import base64

from ai_character_engine.vision import FrameGate, ImageInput, VisionAnalysis, VisionFrame, VisionPipeline

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9Z0iUAAAAASUVORK5CYII="
)


class GameVision:
    async def analyze(self, image, *, prompt=None):
        return VisionAnalysis(
            text="The player's HP is low and an enemy is approaching.",
            provider="demo-game-vlm",
            model="fake-game-vlm",
            tags=("low_hp", "enemy_nearby"),
        )


async def main() -> None:
    pipeline = VisionPipeline(
        provider=GameVision(),
        frame_gate=FrameGate(min_interval_seconds=0, deduplicate=True),
    )
    frame1 = VisionFrame(ImageInput.from_bytes(PNG, mime_type="image/png"), source_type="game")
    frame2 = VisionFrame(ImageInput.from_bytes(PNG, mime_type="image/png"), source_type="game")
    first = await pipeline.to_character_event(frame1)
    duplicate = await pipeline.to_character_event(frame2)
    assert first is not None
    print(first.content)
    print("duplicate skipped:", duplicate is None)
    print("memory mode:", first.payload["vision_memory"])


if __name__ == "__main__":
    asyncio.run(main())
