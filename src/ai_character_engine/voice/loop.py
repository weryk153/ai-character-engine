from __future__ import annotations

from collections.abc import AsyncIterator

from ai_character_engine.observability import TraceContext
from ai_character_engine.session.runtime import ManagedCharacterSession

from .base import AudioInputSource, AudioOutputSink
from .models import AudioChunk, VoicePipelineEvent
from .pipeline import VoicePipeline
from .vad import UtteranceSegmenter


class VoiceConversationLoop:
    """Connects an audio source/sink to VoicePipeline using VAD segmentation.

    The loop is deliberately transport/device-neutral. A host may provide a
    microphone source, WebRTC stream, WebSocket audio source, or prerecorded
    fixture, plus a local speaker or remote audio sink.
    """

    def __init__(
        self,
        *,
        pipeline: VoicePipeline,
        segmenter: UtteranceSegmenter,
    ) -> None:
        self.pipeline = pipeline
        self.segmenter = segmenter

    async def run(
        self,
        *,
        session: ManagedCharacterSession,
        source: AudioInputSource,
        sink: AudioOutputSink | None = None,
        trace_context: TraceContext | None = None,
    ) -> AsyncIterator[VoicePipelineEvent]:
        async for chunk in source.chunks():
            utterance = self.segmenter.push(chunk)
            if utterance is None:
                continue
            async for event in self._process_utterance(
                session=session,
                utterance=utterance,
                sink=sink,
                trace_context=trace_context,
            ):
                yield event

        tail = self.segmenter.flush()
        if tail is not None:
            async for event in self._process_utterance(
                session=session,
                utterance=tail,
                sink=sink,
                trace_context=trace_context,
            ):
                yield event

    async def _process_utterance(
        self,
        *,
        session: ManagedCharacterSession,
        utterance: tuple[AudioChunk, ...],
        sink: AudioOutputSink | None,
        trace_context: TraceContext | None,
    ) -> AsyncIterator[VoicePipelineEvent]:
        if not utterance:
            return
        audio_format = utterance[0].format
        if any(chunk.format != audio_format for chunk in utterance):
            raise ValueError("all chunks in an utterance must use the same AudioFormat")
        audio = b"".join(chunk.data for chunk in utterance)
        yield VoicePipelineEvent("speech_end", data={"chunks": len(utterance)})
        async for event in self.pipeline.stream_turn(
            session,
            audio=audio,
            audio_format=audio_format,
            trace_context=trace_context,
        ):
            if sink is not None and event.type == "tts_audio" and event.audio is not None:
                await sink.play(event.audio)
            yield event
