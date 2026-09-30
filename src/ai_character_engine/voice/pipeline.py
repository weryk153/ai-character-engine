from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from ai_character_engine.events.models import CharacterEvent
from ai_character_engine.observability import TraceContext, Tracer
from ai_character_engine.session.runtime import ManagedCharacterSession

from .base import CharacterTextStreamProvider, SpeechToTextProvider, TextToSpeechProvider
from .models import (
    AudioChunk,
    AudioFormat,
    SynthesizedAudio,
    VoicePipelineEvent,
    VoiceTurnResult,
)
from .streaming import BufferedCharacterTextStream
from .text import SentenceSegmenter


@dataclass(frozen=True, slots=True)
class VoicePipelineConfig:
    stt_timeout_seconds: float = 30.0
    tts_timeout_seconds: float = 30.0
    max_audio_bytes: int = 10 * 1024 * 1024
    tts_voice: str | None = None
    sentence_min_chars: int = 4
    sentence_max_chars: int = 120

    def __post_init__(self) -> None:
        if self.stt_timeout_seconds <= 0:
            raise ValueError("stt_timeout_seconds must be > 0")
        if self.tts_timeout_seconds <= 0:
            raise ValueError("tts_timeout_seconds must be > 0")
        if self.max_audio_bytes < 1:
            raise ValueError("max_audio_bytes must be >= 1")


class VoicePipeline:
    """Provider-neutral STT -> character -> TTS pipeline.

    Audio capture/playback are intentionally kept outside the core pipeline.
    Host applications can connect microphone, WebRTC, WebSocket or local
    sound-device adapters without changing the character runtime.
    """

    def __init__(
        self,
        *,
        stt: SpeechToTextProvider,
        tts: TextToSpeechProvider,
        text_stream: CharacterTextStreamProvider | None = None,
        config: VoicePipelineConfig | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self.stt = stt
        self.tts = tts
        self.text_stream = text_stream or BufferedCharacterTextStream()
        self.config = config or VoicePipelineConfig()
        self.tracer = tracer or Tracer()

    async def process_turn(
        self,
        session: ManagedCharacterSession,
        *,
        audio: bytes,
        audio_format: AudioFormat | None = None,
        trace_context: TraceContext | None = None,
    ) -> VoiceTurnResult:
        fmt = audio_format or AudioFormat()
        self._validate_audio(audio)
        context = self._trace_context(session, trace_context)
        started = time.perf_counter()

        with self.tracer.span("voice.turn", context=context) as root:
            child = context.child(parent_span_id=root.span_id)
            transcript, stt_ms = await self._transcribe(audio, fmt, child)
            if not transcript.text.strip():
                raise ValueError("STT returned an empty transcript")

            character_started = time.perf_counter()
            with self.tracer.span("voice.character", context=child) as span:
                character_result = await session.process_event(
                    CharacterEvent.user_message(transcript.text),
                    trace_context=child.child(parent_span_id=span.span_id),
                )
                response = character_result.response
                span.set_attribute("response_chars", len(response.text))
            character_ms = (time.perf_counter() - character_started) * 1000

            audio_result, tts_ms = await self._synthesize(response.text, child)
            total_ms = (time.perf_counter() - started) * 1000
            root.set_attribute("stt_latency_ms", stt_ms)
            root.set_attribute("character_latency_ms", character_ms)
            root.set_attribute("tts_latency_ms", tts_ms)

        return VoiceTurnResult(
            transcript=transcript,
            response_text=response.text,
            audio=(audio_result,),
            trace_id=context.trace_id,
            stt_latency_ms=stt_ms,
            character_latency_ms=character_ms,
            tts_latency_ms=tts_ms,
            total_latency_ms=total_ms,
            metadata={"session_id": session.record.id},
        )

    async def stream_turn(
        self,
        session: ManagedCharacterSession,
        *,
        audio: bytes,
        audio_format: AudioFormat | None = None,
        trace_context: TraceContext | None = None,
    ):
        fmt = audio_format or AudioFormat()
        self._validate_audio(audio)
        context = self._trace_context(session, trace_context)
        started = time.perf_counter()

        with self.tracer.span("voice.turn", context=context) as root:
            child = context.child(parent_span_id=root.span_id)
            transcript, stt_ms = await self._transcribe(audio, fmt, child)
            if not transcript.text.strip():
                raise ValueError("STT returned an empty transcript")
            yield VoicePipelineEvent(
                "stt_final",
                text=transcript.text,
                data={"language": transcript.language, "latency_ms": stt_ms, "trace_id": context.trace_id},
            )

            segmenter = SentenceSegmenter(
                min_chars=self.config.sentence_min_chars,
                max_chars=self.config.sentence_max_chars,
            )
            response_parts: list[str] = []
            audio_results: list[SynthesizedAudio] = []
            tts_total_ms = 0.0
            character_started = time.perf_counter()

            with self.tracer.span("voice.character", context=child) as character_span:
                stream_context = child.child(parent_span_id=character_span.span_id)
                async for delta in self.text_stream.stream_text(
                    session,
                    content=transcript.text,
                    trace_context=stream_context,
                ):
                    if delta.text:
                        response_parts.append(delta.text)
                        yield VoicePipelineEvent(
                            "character_delta",
                            text=delta.text,
                            data={"trace_id": context.trace_id},
                        )
                        for sentence in segmenter.push(delta.text):
                            audio_result, tts_ms = await self._synthesize(sentence, child)
                            tts_total_ms += tts_ms
                            audio_results.append(audio_result)
                            yield VoicePipelineEvent(
                                "tts_audio",
                                text=sentence,
                                audio=AudioChunk(
                                    audio_result.data,
                                    format=audio_result.format,
                                    sequence=len(audio_results) - 1,
                                ),
                                data={"latency_ms": tts_ms, "trace_id": context.trace_id},
                            )
                character_span.set_attribute("response_chars", sum(map(len, response_parts)))

            tail = segmenter.flush()
            if tail:
                audio_result, tts_ms = await self._synthesize(tail, child)
                tts_total_ms += tts_ms
                audio_results.append(audio_result)
                yield VoicePipelineEvent(
                    "tts_audio",
                    text=tail,
                    audio=AudioChunk(
                        audio_result.data,
                        format=audio_result.format,
                        sequence=len(audio_results) - 1,
                    ),
                    data={"latency_ms": tts_ms, "trace_id": context.trace_id},
                )

            response_text = "".join(response_parts)
            character_ms = (time.perf_counter() - character_started) * 1000
            total_ms = (time.perf_counter() - started) * 1000
            root.set_attribute("stt_latency_ms", stt_ms)
            root.set_attribute("character_latency_ms", character_ms)
            root.set_attribute("tts_latency_ms", tts_total_ms)
            root.set_attribute("tts_segments", len(audio_results))
            yield VoicePipelineEvent(
                "trace",
                data={
                    "trace_id": context.trace_id,
                    "stt_latency_ms": stt_ms,
                    "character_latency_ms": character_ms,
                    "tts_latency_ms": tts_total_ms,
                    "total_latency_ms": total_ms,
                    "tts_segments": len(audio_results),
                },
            )
            yield VoicePipelineEvent(
                "final",
                text=response_text,
                data={
                    "trace_id": context.trace_id,
                    "session_id": session.record.id,
                    "tts_segments": len(audio_results),
                },
            )

    def _validate_audio(self, audio: bytes) -> None:
        if not audio:
            raise ValueError("audio must not be empty")
        if len(audio) > self.config.max_audio_bytes:
            raise ValueError(
                f"audio exceeds max_audio_bytes={self.config.max_audio_bytes}"
            )

    def _trace_context(
        self,
        session: ManagedCharacterSession,
        trace_context: TraceContext | None,
    ) -> TraceContext:
        base = trace_context or TraceContext.create()
        return base.child(
            session_id=session.record.id,
            user_id=session.record.user_id,
            character_id=session.record.character_id,
        )

    async def _transcribe(
        self,
        audio: bytes,
        fmt: AudioFormat,
        context: TraceContext,
    ):
        started = time.perf_counter()
        with self.tracer.span("voice.stt", context=context) as span:
            result = await asyncio.wait_for(
                self.stt.transcribe(audio, audio_format=fmt),
                timeout=self.config.stt_timeout_seconds,
            )
            elapsed = (time.perf_counter() - started) * 1000
            span.set_attribute("audio_bytes", len(audio))
            span.set_attribute("language", result.language)
            span.set_attribute("provider_latency_ms", result.latency_ms)
            span.set_attribute("elapsed_ms", elapsed)
        return result, elapsed

    async def _synthesize(self, text: str, context: TraceContext):
        started = time.perf_counter()
        with self.tracer.span("voice.tts", context=context) as span:
            result = await asyncio.wait_for(
                self.tts.synthesize(text, voice=self.config.tts_voice),
                timeout=self.config.tts_timeout_seconds,
            )
            elapsed = (time.perf_counter() - started) * 1000
            span.set_attribute("text_chars", len(text))
            span.set_attribute("provider_latency_ms", result.latency_ms)
            span.set_attribute("elapsed_ms", elapsed)
        return result, elapsed
