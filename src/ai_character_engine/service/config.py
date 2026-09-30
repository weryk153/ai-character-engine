from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True, frozen=True)
class CharacterServiceConfig:
    title: str = "AI Character Engine Service"
    debug: bool = False
    cors_origins: tuple[str, ...] = field(default_factory=tuple)
    cors_allow_credentials: bool = False
    default_timeout_seconds: float = 30.0
    stream_chunk_chars: int = 24
    websocket_max_message_chars: int = 20_000

    def __post_init__(self) -> None:
        if self.default_timeout_seconds <= 0:
            raise ValueError("default_timeout_seconds must be > 0")
        if self.stream_chunk_chars <= 0:
            raise ValueError("stream_chunk_chars must be > 0")
        if self.websocket_max_message_chars <= 0:
            raise ValueError("websocket_max_message_chars must be > 0")
