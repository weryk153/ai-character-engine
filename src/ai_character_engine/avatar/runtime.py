from __future__ import annotations

from dataclasses import replace
from typing import Any

from ai_character_engine.voice.models import AudioChunk

from .envelope import Pcm16EnvelopeAnalyzer
from .models import AvatarCueBundle, ExpressionRequest, VisemeCue
from .scheduler import ExpressionScheduler
from .viseme import CompositeVisemeAdapter, VisemeAdapter


class AvatarRuntime:
    """Stateful, renderer-neutral expressive avatar timeline for one live turn."""

    def __init__(
        self,
        *,
        envelope: Pcm16EnvelopeAnalyzer | None = None,
        visemes: VisemeAdapter | None = None,
        expressions: ExpressionScheduler | None = None,
    ) -> None:
        self.envelope = envelope or Pcm16EnvelopeAnalyzer()
        self.visemes = visemes or CompositeVisemeAdapter()
        self.expressions = expressions or ExpressionScheduler()
        self._turn_id: str | None = None
        self._elapsed_ms = 0.0

    @property
    def active(self) -> bool:
        return self._turn_id is not None

    @property
    def turn_id(self) -> str | None:
        return self._turn_id

    @property
    def elapsed_ms(self) -> float:
        return self._elapsed_ms

    def begin_turn(self, turn_id: str) -> None:
        if not turn_id.strip():
            raise ValueError("turn_id must not be empty")
        self._turn_id = turn_id
        self._elapsed_ms = 0.0
        self.envelope.reset()
        self.expressions.reset()

    def schedule_expression(self, request: ExpressionRequest) -> str:
        if not self.active:
            raise RuntimeError("avatar turn is not active")
        return self.expressions.schedule(request, base_offset_ms=self._elapsed_ms)

    def feed_audio(
        self,
        *,
        text: str,
        chunk: AudioChunk,
        segment_sequence: int,
        chunk_index: int,
        duration_ms: float | None,
        emotion: str | None = None,
        metadata: dict[str, Any] | None = None,
        allow_text_visemes: bool = True,
    ) -> AvatarCueBundle:
        if self._turn_id is None:
            raise RuntimeError("avatar turn is not active")
        actual_duration = duration_ms
        if actual_duration is None and chunk.format.encoding == "pcm_s16le":
            bytes_per_second = chunk.format.sample_rate_hz * chunk.format.channels * chunk.format.sample_width_bytes
            if bytes_per_second > 0:
                actual_duration = len(chunk.data) * 1000.0 / bytes_per_second
        safe_duration = max(0.0, actual_duration or 0.0)
        start_ms = self._elapsed_ms
        envelope = self.envelope.analyze(chunk, start_offset_ms=start_ms)
        relative_visemes = self.visemes.adapt(
            text,
            duration_ms=safe_duration if actual_duration is not None else None,
            metadata=metadata or chunk.metadata,
            allow_text_fallback=allow_text_visemes,
        )
        visemes = tuple(
            replace(cue, start_offset_ms=start_ms + cue.start_offset_ms)
            for cue in relative_visemes
        )
        expressions = self.expressions.resolve(
            start_ms=start_ms,
            duration_ms=safe_duration,
            emotion=emotion,
        ) if safe_duration > 0 else ()
        base = AvatarCueBundle(
            self._turn_id,
            segment_sequence,
            chunk_index,
            start_ms,
            actual_duration,
            envelope=envelope,
            visemes=visemes,
            expressions=expressions,
        )
        self._elapsed_ms += safe_duration
        return base

    def end_turn(self) -> dict[str, Any]:
        turn_id = self._turn_id
        elapsed = self._elapsed_ms
        self._turn_id = None
        self._elapsed_ms = 0.0
        self.envelope.reset()
        self.expressions.reset()
        return {
            "turn_id": turn_id,
            "elapsed_ms": elapsed,
            "reset": ("mouth", "expressions"),
        }
