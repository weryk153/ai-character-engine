from __future__ import annotations

from typing import Protocol

from .models import ImageInput, VisionAnalysis


class VisionProvider(Protocol):
    async def analyze(self, image: ImageInput, *, prompt: str | None = None) -> VisionAnalysis:
        ...


class MultimodalModelClient(Protocol):
    """Minimal capability contract for a model that can consume image + text."""

    async def analyze(self, image: ImageInput, *, prompt: str | None = None) -> VisionAnalysis:
        ...
