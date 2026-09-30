from __future__ import annotations

import base64
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit
from uuid import uuid4

ImageSource = Literal["bytes", "path", "url"]
VisionSourceType = Literal["upload", "screenshot", "camera", "game", "other"]


@dataclass(slots=True, frozen=True)
class ImageInput:
    """Provider-neutral image reference.

    Raw bytes live only at the vision boundary. Character events contain a text
    observation plus safe metadata, never the image bytes themselves.
    """

    source: ImageSource
    mime_type: str
    data: bytes | None = None
    path: str | None = None
    url: str | None = None
    width: int | None = None
    height: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.mime_type.strip():
            raise ValueError("mime_type must not be empty")
        provided = sum(value is not None for value in (self.data, self.path, self.url))
        if provided != 1:
            raise ValueError("exactly one of data, path, or url must be provided")
        if self.source == "bytes" and self.data is None:
            raise ValueError("bytes source requires data")
        if self.source == "path" and self.path is None:
            raise ValueError("path source requires path")
        if self.source == "url" and self.url is None:
            raise ValueError("url source requires url")
        if self.width is not None and self.width <= 0:
            raise ValueError("width must be > 0")
        if self.height is not None and self.height <= 0:
            raise ValueError("height must be > 0")

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        *,
        mime_type: str,
        width: int | None = None,
        height: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "ImageInput":
        return cls(
            source="bytes", mime_type=mime_type, data=bytes(data), width=width,
            height=height, metadata=dict(metadata or {}),
        )

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        mime_type: str,
        width: int | None = None,
        height: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "ImageInput":
        return cls(
            source="path", mime_type=mime_type, path=str(path), width=width,
            height=height, metadata=dict(metadata or {}),
        )

    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        mime_type: str,
        width: int | None = None,
        height: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "ImageInput":
        parts = urlsplit(url)
        if parts.scheme not in {"http", "https"} or not parts.netloc:
            raise ValueError("image URL must be http(s)")
        return cls(
            source="url", mime_type=mime_type, url=url, width=width,
            height=height, metadata=dict(metadata or {}),
        )

    def read_bytes(self, *, max_bytes: int | None = None) -> bytes:
        if self.data is not None:
            payload = self.data
        elif self.path is not None:
            path = Path(self.path)
            if max_bytes is not None and path.stat().st_size > max_bytes:
                raise ValueError("image exceeds max_image_bytes")
            payload = path.read_bytes()
        else:
            raise ValueError("URL image does not have local bytes")
        if max_bytes is not None and len(payload) > max_bytes:
            raise ValueError("image exceeds max_image_bytes")
        return payload

    def to_data_url(self, *, max_bytes: int | None = None) -> str:
        payload = self.read_bytes(max_bytes=max_bytes)
        encoded = base64.b64encode(payload).decode("ascii")
        return f"data:{self.mime_type};base64,{encoded}"


@dataclass(slots=True, frozen=True)
class VisionFrame:
    image: ImageInput
    source_type: VisionSourceType = "other"
    id: str = field(default_factory=lambda: uuid4().hex)
    captured_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class VisionAnalysis:
    text: str
    provider: str
    model: str | None = None
    tags: tuple[str, ...] = ()
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("vision analysis text must not be empty")
        if not self.provider.strip():
            raise ValueError("vision provider must not be empty")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")


@dataclass(slots=True, frozen=True)
class VisionPipelineResult:
    frame: VisionFrame
    analysis: VisionAnalysis
    event_content: str
    event_payload: dict[str, Any]
    persistent_memory: bool


@dataclass(slots=True, frozen=True)
class MultimodalEvent:
    """Normalized text+vision observation before entering CharacterRuntime."""

    text: str
    frame: VisionFrame
    analysis: VisionAnalysis
    persistent_memory: bool = False
