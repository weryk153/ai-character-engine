from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

AudioEncoding = Literal["pcm_s16le", "wav"]
VoiceEventType = Literal[
    "speech_start",
    "speech_end",
    "stt_partial",
    "stt_final",
    "character_delta",
    "tts_audio",
    "trace",
    "final",
    "error",
]


@dataclass(frozen=True, slots=True)
class AudioFormat:
    sample_rate_hz: int = 16_000
    channels: int = 1
    sample_width_bytes: int = 2
    encoding: AudioEncoding = "pcm_s16le"

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be > 0")
        if self.channels <= 0:
            raise ValueError("channels must be > 0")
        if self.sample_width_bytes <= 0:
            raise ValueError("sample_width_bytes must be > 0")


@dataclass(frozen=True, slots=True)
class AudioChunk:
    data: bytes
    format: AudioFormat = field(default_factory=AudioFormat)
    sequence: int = 0
    timestamp_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("sequence must be >= 0")


@dataclass(frozen=True, slots=True)
class TranscriptResult:
    text: str
    language: str | None = None
    confidence: float | None = None
    latency_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class SynthesizedAudio:
    data: bytes
    format: AudioFormat
    duration_ms: float | None = None
    latency_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TextDelta:
    text: str
    final: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class VoicePipelineEvent:
    type: VoiceEventType
    text: str | None = None
    audio: AudioChunk | None = None
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class VoiceTurnResult:
    transcript: TranscriptResult
    response_text: str
    audio: tuple[SynthesizedAudio, ...]
    trace_id: str
    stt_latency_ms: float | None = None
    character_latency_ms: float | None = None
    tts_latency_ms: float | None = None
    total_latency_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
