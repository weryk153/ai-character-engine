from __future__ import annotations

import math
import sys
from array import array
from dataclasses import dataclass

from ai_character_engine.voice.models import AudioChunk

from .models import AudioEnvelopePoint


@dataclass(frozen=True, slots=True)
class AudioEnvelopeConfig:
    window_ms: float = 20.0
    gain: float = 2.5
    noise_floor: float = 0.015
    attack: float = 0.70
    release: float = 0.30

    def __post_init__(self) -> None:
        if not math.isfinite(self.window_ms) or self.window_ms <= 0:
            raise ValueError("window_ms must be finite and > 0")
        if not math.isfinite(self.gain) or self.gain <= 0:
            raise ValueError("gain must be finite and > 0")
        if not 0 <= self.noise_floor < 1:
            raise ValueError("noise_floor must be in [0, 1)")
        if not 0 < self.attack <= 1 or not 0 < self.release <= 1:
            raise ValueError("attack and release must be in (0, 1]")


class Pcm16EnvelopeAnalyzer:
    """Convert PCM16 audio into bounded mouth-open envelope points.

    The analyzer is intentionally signal-only: it does not guess phonemes. It
    uses RMS energy, a configurable noise floor, gain, and asymmetric smoothing.
    Non-PCM16 chunks return no points instead of pretending to understand a
    compressed format.
    """

    def __init__(self, config: AudioEnvelopeConfig | None = None) -> None:
        self.config = config or AudioEnvelopeConfig()
        self._smoothed = 0.0

    def reset(self) -> None:
        self._smoothed = 0.0

    def analyze(self, chunk: AudioChunk, *, start_offset_ms: float = 0.0) -> tuple[AudioEnvelopePoint, ...]:
        fmt = chunk.format
        if fmt.encoding != "pcm_s16le" or fmt.sample_width_bytes != 2 or not chunk.data:
            return ()
        samples = array("h")
        usable = len(chunk.data) - (len(chunk.data) % 2)
        samples.frombytes(chunk.data[:usable])
        if sys.byteorder != "little":
            samples.byteswap()
        if not samples:
            return ()

        samples_per_window = max(1, round(fmt.sample_rate_hz * self.config.window_ms / 1000.0) * fmt.channels)
        points: list[AudioEnvelopePoint] = []
        cursor = 0
        while cursor < len(samples):
            block = samples[cursor : cursor + samples_per_window]
            if not block:
                break
            rms = math.sqrt(sum(float(v) * float(v) for v in block) / len(block)) / 32768.0
            normalized = max(0.0, (rms - self.config.noise_floor) / (1.0 - self.config.noise_floor))
            target = min(1.0, normalized * self.config.gain)
            factor = self.config.attack if target >= self._smoothed else self.config.release
            self._smoothed += (target - self._smoothed) * factor
            frame_count = len(block) / max(1, fmt.channels)
            duration_ms = frame_count * 1000.0 / fmt.sample_rate_hz
            frame_offset = (cursor / max(1, fmt.channels)) * 1000.0 / fmt.sample_rate_hz
            points.append(
                AudioEnvelopePoint(
                    start_offset_ms=start_offset_ms + frame_offset,
                    duration_ms=duration_ms,
                    amplitude=max(0.0, min(1.0, self._smoothed)),
                )
            )
            cursor += samples_per_window
        return tuple(points)
