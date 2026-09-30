from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from uuid import uuid4

from ai_character_engine.runtime.models import CharacterRunResult
from ai_character_engine.voice.models import AudioChunk, AudioFormat
from ai_character_engine.vision.models import VisionFrame


class LiveInputKind(StrEnum):
    TEXT = "text"
    AUDIO = "audio"


class LiveEventType(StrEnum):
    TURN_STARTED = "turn_started"
    TURN_INTERRUPTED = "turn_interrupted"
    SPEECH_STARTED = "speech_started"
    SPEECH_ENDED = "speech_ended"
    BARGE_IN_DETECTED = "barge_in_detected"
    STT_FINAL = "stt_final"
    CHARACTER_DELTA = "character_delta"
    REPLY = "reply"
    TTS_AUDIO = "tts_audio"
    MOUTH_CUE = "mouth_cue"
    AVATAR_CUE = "avatar_cue"
    AVATAR_RESET = "avatar_reset"
    AVATAR_BEHAVIOR_CUE = "avatar_behavior_cue"
    AVATAR_BEHAVIOR_RESET = "avatar_behavior_reset"
    STREAM_METRICS = "stream_metrics"
    PLAYBACK_STARTED = "playback_started"
    PLAYBACK_INTERRUPTED = "playback_interrupted"
    PLAYBACK_FINISHED = "playback_finished"
    AUTONOMY_DELIVERED = "autonomy_delivered"
    AUTONOMY_FAILED = "autonomy_failed"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class LiveRuntimeConfig:
    poll_interval_seconds: float = 0.10
    max_pending_inputs: int = 32
    max_context_frames: int = 2
    context_frame_ttl_seconds: float = 5.0
    max_audio_bytes: int = 16 * 1024 * 1024
    stt_timeout_seconds: float = 30.0
    tts_timeout_seconds: float = 30.0
    tts_voice: str | None = None
    streaming_output: bool = False
    prefer_streaming_tts: bool = True
    emit_mouth_cues: bool = True
    stream_queue_size: int = 8

    def __post_init__(self) -> None:
        for name in (
            "poll_interval_seconds",
            "context_frame_ttl_seconds",
            "stt_timeout_seconds",
            "tts_timeout_seconds",
        ):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("max_pending_inputs", "max_context_frames", "max_audio_bytes", "stream_queue_size"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class LiveTurnInput:
    kind: LiveInputKind
    text: str = ""
    audio: bytes = b""
    audio_format: AudioFormat = field(default_factory=AudioFormat)
    frames: tuple[VisionFrame, ...] = ()
    id: str = field(default_factory=lambda: uuid4().hex)

    def __post_init__(self) -> None:
        if not isinstance(self.kind, LiveInputKind):
            raise ValueError("kind must be a LiveInputKind")
        if not isinstance(self.frames, tuple):
            object.__setattr__(self, "frames", tuple(self.frames))
        if self.kind is LiveInputKind.TEXT:
            if not isinstance(self.text, str) or not self.text.strip():
                raise ValueError("text input must contain non-whitespace text")
            if self.audio:
                raise ValueError("text input must not contain audio bytes")
        elif self.kind is LiveInputKind.AUDIO:
            if not isinstance(self.audio, bytes) or not self.audio:
                raise ValueError("audio input must contain bytes")
            if self.text:
                raise ValueError("audio input must not contain text")


@dataclass(frozen=True, slots=True)
class LiveRuntimeEvent:
    type: LiveEventType
    input_id: str | None = None
    text: str | None = None
    audio: AudioChunk | None = None
    run_result: CharacterRunResult | None = None
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MouthActivityCue:
    """Provider-neutral timing hint for avatar mouth activity.

    It intentionally does not name phonemes or visemes. A renderer host
    can map the active window to jaw-open amplitude, audio-envelope analysis, or
    a provider-specific viseme system without coupling that policy to the engine.
    """

    segment_sequence: int
    chunk_index: int
    start_offset_ms: float = 0.0
    duration_ms: float | None = None
    activity: float = 1.0
    timing_source: str = "audio_duration"

    def __post_init__(self) -> None:
        if self.segment_sequence < 0 or self.chunk_index < 0:
            raise ValueError("cue sequence values must be >= 0")
        if self.start_offset_ms < 0:
            raise ValueError("start_offset_ms must be >= 0")
        if self.duration_ms is not None and self.duration_ms < 0:
            raise ValueError("duration_ms must be >= 0 when supplied")
        if not 0 <= self.activity <= 1:
            raise ValueError("activity must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_sequence": self.segment_sequence,
            "chunk_index": self.chunk_index,
            "start_offset_ms": self.start_offset_ms,
            "duration_ms": self.duration_ms,
            "activity": self.activity,
            "timing_source": self.timing_source,
            "align_to": "tts_audio",
        }
