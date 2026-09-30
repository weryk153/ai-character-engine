from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import Protocol

from ai_character_engine.observability import TraceContext
from ai_character_engine.session.runtime import ManagedCharacterSession

from .models import AudioChunk, AudioFormat, SynthesizedAudio, TextDelta, TranscriptResult


class SpeechToTextProvider(Protocol):
    async def transcribe(
        self,
        audio: bytes,
        *,
        audio_format: AudioFormat,
    ) -> TranscriptResult: ...


class StreamingSpeechToTextProvider(Protocol):
    async def transcribe_stream(
        self,
        chunks: AsyncIterator[AudioChunk],
    ) -> AsyncIterator[TranscriptResult]: ...


class TextToSpeechProvider(Protocol):
    async def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
    ) -> SynthesizedAudio: ...


class StreamingTextToSpeechProvider(Protocol):
    async def synthesize_stream(
        self,
        text: str,
        *,
        voice: str | None = None,
    ) -> AsyncIterator[AudioChunk]: ...


class CharacterTextStreamProvider(Protocol):
    async def stream_text(
        self,
        session: ManagedCharacterSession,
        *,
        content: str,
        trace_context: TraceContext,
    ) -> AsyncIterator[TextDelta]: ...


class AudioInputSource(Protocol):
    def chunks(self) -> AsyncIterator[AudioChunk]: ...


class AudioOutputSink(Protocol):
    async def play(self, chunk: AudioChunk) -> None: ...


class InterruptibleAudioOutputSink(AudioOutputSink, Protocol):
    async def stop(self) -> None: ...


class CallableSpeechToTextProvider:
    def __init__(
        self,
        func: Callable[[bytes, AudioFormat], Awaitable[TranscriptResult]],
    ) -> None:
        self.func = func

    async def transcribe(
        self,
        audio: bytes,
        *,
        audio_format: AudioFormat,
    ) -> TranscriptResult:
        return await self.func(audio, audio_format)


class CallableTextToSpeechProvider:
    def __init__(
        self,
        func: Callable[[str, str | None], Awaitable[SynthesizedAudio]],
    ) -> None:
        self.func = func

    async def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
    ) -> SynthesizedAudio:
        return await self.func(text, voice)
