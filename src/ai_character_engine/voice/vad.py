from __future__ import annotations

import math
from array import array
from dataclasses import dataclass
from typing import Protocol

from .models import AudioChunk, AudioFormat


class VoiceActivityDetector(Protocol):
    def is_speech(self, chunk: AudioChunk) -> bool: ...


class EnergyVoiceActivityDetector:
    """Simple PCM16 energy detector for tests and local baselines.

    This is not a neural VAD. It intentionally provides a deterministic,
    dependency-free baseline that applications may replace with Silero/WebRTC
    or another production detector.
    """

    def __init__(self, *, rms_threshold: float = 500.0) -> None:
        if rms_threshold < 0:
            raise ValueError("rms_threshold must be >= 0")
        self.rms_threshold = rms_threshold

    def is_speech(self, chunk: AudioChunk) -> bool:
        fmt = chunk.format
        if fmt.encoding != "pcm_s16le" or fmt.sample_width_bytes != 2:
            raise ValueError("EnergyVoiceActivityDetector requires pcm_s16le 16-bit audio")
        if not chunk.data:
            return False
        samples = array("h")
        samples.frombytes(chunk.data)
        if not samples:
            return False
        mean_square = sum(sample * sample for sample in samples) / len(samples)
        return math.sqrt(mean_square) >= self.rms_threshold


@dataclass(frozen=True, slots=True)
class UtteranceSegmenterConfig:
    min_speech_chunks: int = 1
    end_silence_chunks: int = 3
    max_chunks: int = 500

    def __post_init__(self) -> None:
        if self.min_speech_chunks < 1:
            raise ValueError("min_speech_chunks must be >= 1")
        if self.end_silence_chunks < 1:
            raise ValueError("end_silence_chunks must be >= 1")
        if self.max_chunks < 1:
            raise ValueError("max_chunks must be >= 1")


class UtteranceSegmenter:
    """Groups audio chunks into an utterance using a VAD decision stream."""

    def __init__(
        self,
        detector: VoiceActivityDetector,
        *,
        config: UtteranceSegmenterConfig | None = None,
    ) -> None:
        self.detector = detector
        self.config = config or UtteranceSegmenterConfig()
        self.reset()

    def reset(self) -> None:
        self._chunks: list[AudioChunk] = []
        self._speech_chunks = 0
        self._trailing_silence = 0
        self._started = False

    @property
    def started(self) -> bool:
        return self._started

    @property
    def speech_chunks(self) -> int:
        return self._speech_chunks

    @property
    def trailing_silence_chunks(self) -> int:
        return self._trailing_silence

    def push(self, chunk: AudioChunk) -> tuple[AudioChunk, ...] | None:
        speech = self.detector.is_speech(chunk)
        if speech:
            self._started = True
            self._speech_chunks += 1
            self._trailing_silence = 0
            self._chunks.append(chunk)
        elif self._started:
            self._trailing_silence += 1
            self._chunks.append(chunk)

        if len(self._chunks) >= self.config.max_chunks:
            return self._finish_if_valid()

        if self._started and self._trailing_silence >= self.config.end_silence_chunks:
            return self._finish_if_valid()
        return None

    def flush(self) -> tuple[AudioChunk, ...] | None:
        return self._finish_if_valid()

    def _finish_if_valid(self) -> tuple[AudioChunk, ...] | None:
        chunks = tuple(self._chunks)
        valid = self._speech_chunks >= self.config.min_speech_chunks
        self.reset()
        return chunks if valid else None
