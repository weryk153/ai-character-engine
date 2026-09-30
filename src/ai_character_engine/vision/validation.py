from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

from .errors import VisionInputError
from .models import ImageInput


def sniff_image_mime(data: bytes) -> str | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def image_dimensions(data: bytes, mime_type: str) -> tuple[int, int] | None:
    if mime_type == "image/png" and len(data) >= 24 and data.startswith(b"\x89PNG"):
        return struct.unpack(">II", data[16:24])
    if mime_type == "image/gif" and len(data) >= 10:
        return struct.unpack("<HH", data[6:10])
    if mime_type == "image/jpeg" and len(data) >= 4:
        i = 2
        while i + 9 < len(data):
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            i += 2
            if marker in {0xD8, 0xD9}:
                continue
            if i + 2 > len(data):
                break
            length = int.from_bytes(data[i:i+2], "big")
            if length < 2 or i + length > len(data):
                break
            if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
                if i + 7 <= len(data):
                    height = int.from_bytes(data[i+3:i+5], "big")
                    width = int.from_bytes(data[i+5:i+7], "big")
                    return width, height
                break
            i += length
    return None


@dataclass(slots=True, frozen=True)
class VisionInputPolicy:
    max_image_bytes: int = 8 * 1024 * 1024
    max_pixels: int = 20_000_000
    allowed_mime_types: frozenset[str] = frozenset(
        {"image/png", "image/jpeg", "image/webp", "image/gif"}
    )
    require_sniff_match: bool = True

    def validate(self, image: ImageInput) -> tuple[int | None, int | None, int | None]:
        mime = image.mime_type.lower().strip()
        if mime not in self.allowed_mime_types:
            raise VisionInputError(f"unsupported image MIME type: {image.mime_type}")

        width = image.width
        height = image.height
        size: int | None = None
        if image.data is not None or image.path is not None:
            try:
                if image.path is not None:
                    size = Path(image.path).stat().st_size
                    if size > self.max_image_bytes:
                        raise VisionInputError("image exceeds max_image_bytes")
                data = image.read_bytes(max_bytes=self.max_image_bytes)
            except (OSError, ValueError) as exc:
                raise VisionInputError(str(exc)) from exc
            size = len(data)
            sniffed = sniff_image_mime(data)
            if sniffed is None:
                raise VisionInputError("unrecognized image format")
            if self.require_sniff_match and sniffed != mime:
                raise VisionInputError(
                    f"declared MIME {mime} does not match image bytes {sniffed}"
                )
            detected = image_dimensions(data, sniffed)
            if detected is not None:
                width = width or detected[0]
                height = height or detected[1]

        if width is not None and height is not None:
            if width * height > self.max_pixels:
                raise VisionInputError("image exceeds max_pixels")
        return width, height, size
