from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from enum import StrEnum

from .base import AudioOutputSink


class DuplexVoiceError(RuntimeError):
    """Raised when duplex/barge-in safety contracts cannot be satisfied."""


class LiveVoicePhase(StrEnum):
    IDLE = "idle"
    GENERATING = "generating"
    PLAYBACK = "playback"


@dataclass(frozen=True, slots=True)
class DuplexVoiceConfig:
    """Host policy for continuous microphone input and barge-in.

    Automatic acoustic barge-in is conservative by default. Enabling it requires
    the host to explicitly confirm that its capture path has echo cancellation or
    another equivalent mechanism preventing the character's own speaker output
    from being treated as user speech.
    """

    automatic_barge_in: bool = False
    echo_cancellation_confirmed: bool = False
    barge_in_min_speech_chunks: int = 2
    vad_rms_threshold: float = 500.0
    min_speech_chunks: int = 1
    end_silence_chunks: int = 3
    max_utterance_chunks: int = 500
    sentence_min_chars: int = 4
    sentence_max_chars: int = 120

    def __post_init__(self) -> None:
        if self.automatic_barge_in and not self.echo_cancellation_confirmed:
            raise ValueError(
                "automatic_barge_in requires echo_cancellation_confirmed=True"
            )
        if type(self.barge_in_min_speech_chunks) is not int or self.barge_in_min_speech_chunks < 1:
            raise ValueError("barge_in_min_speech_chunks must be a positive integer")
        if type(self.min_speech_chunks) is not int or self.min_speech_chunks < 1:
            raise ValueError("min_speech_chunks must be a positive integer")
        if type(self.end_silence_chunks) is not int or self.end_silence_chunks < 1:
            raise ValueError("end_silence_chunks must be a positive integer")
        if type(self.max_utterance_chunks) is not int or self.max_utterance_chunks < 1:
            raise ValueError("max_utterance_chunks must be a positive integer")
        if type(self.sentence_min_chars) is not int or self.sentence_min_chars < 1:
            raise ValueError("sentence_min_chars must be a positive integer")
        if type(self.sentence_max_chars) is not int or self.sentence_max_chars < self.sentence_min_chars:
            raise ValueError("sentence_max_chars must be >= sentence_min_chars")
        if type(self.vad_rms_threshold) not in (int, float) or not math.isfinite(self.vad_rms_threshold) or self.vad_rms_threshold < 0:
            raise ValueError("vad_rms_threshold must be finite and >= 0")


def supports_interruptible_playback(sink: AudioOutputSink | None) -> bool:
    """Return whether a sink exposes an awaitable ``stop()`` hook.

    Protocol runtime checks are intentionally avoided; this is a small structural
    check suitable for host-provided sink implementations.
    """

    if sink is None:
        return False
    stop = getattr(sink, "stop", None)
    return callable(stop) and inspect.iscoroutinefunction(stop)
