from __future__ import annotations

import struct

import pytest

from ai_character_engine.character.profile import CharacterProfile
from ai_character_engine.context.builder import ContextBuilder
from ai_character_engine.llm.models import LLMResponse
from ai_character_engine.observability import InMemoryObservabilitySink, Tracer
from ai_character_engine.runtime.character_runtime import CharacterRuntime
from ai_character_engine.session import CharacterRuntimeFactory, SessionManager
from ai_character_engine.session.store import InMemorySessionStore
from ai_character_engine.voice import (
    AudioChunk,
    AudioFormat,
    BufferedCharacterTextStream,
    EnergyVoiceActivityDetector,
    SentenceSegmenter,
    SynthesizedAudio,
    TranscriptResult,
    UtteranceSegmenter,
    UtteranceSegmenterConfig,
    VoicePipeline,
    VoicePipelineConfig,
)


class EchoLLM:
    async def generate(self, messages, *, tools=None):
        user = next((m.content for m in reversed(messages) if m.role in {"user", "event"}), "")
        return LLMResponse(text=f"收到：{user}。請放心！", model="fake")


class FakeSTT:
    def __init__(self, text: str = "你好") -> None:
        self.text = text
        self.calls = 0

    async def transcribe(self, audio: bytes, *, audio_format: AudioFormat) -> TranscriptResult:
        self.calls += 1
        return TranscriptResult(text=self.text, language="zh-TW", latency_ms=12.0)


class FakeTTS:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def synthesize(self, text: str, *, voice: str | None = None) -> SynthesizedAudio:
        self.calls.append(text)
        return SynthesizedAudio(
            data=(text + "|audio").encode(),
            format=AudioFormat(sample_rate_hz=24_000),
            duration_ms=float(len(text) * 20),
            latency_ms=7.0,
            metadata={"voice": voice},
        )


def make_session(*, tracer: Tracer | None = None):
    character = CharacterProfile(
        id="mei",
        name="Mei",
        description="理性的研究者",
        personality=["理性"],
        speaking_style=["自然口語"],
    )
    manager = SessionManager(InMemorySessionStore())
    factory = CharacterRuntimeFactory(
        characters={character.id: character},
        llm_factory=lambda _record: EchoLLM(),
        context_builder_factory=lambda: ContextBuilder(),
        session_manager=manager,
        tracer=tracer,
    )
    return factory.create(user_id="user-a", character_id="mei")


@pytest.mark.asyncio
async def test_voice_pipeline_process_turn_stt_character_tts():
    stt = FakeSTT("今天好累")
    tts = FakeTTS()
    pipeline = VoicePipeline(stt=stt, tts=tts, config=VoicePipelineConfig(tts_voice="alice"))
    result = await pipeline.process_turn(make_session(), audio=b"1234")

    assert result.transcript.text == "今天好累"
    assert "今天好累" in result.response_text
    assert len(result.audio) == 1
    assert tts.calls == [result.response_text]
    assert result.audio[0].metadata["voice"] == "alice"
    assert result.trace_id


@pytest.mark.asyncio
async def test_voice_stream_turn_emits_transcript_deltas_tts_and_final():
    stt = FakeSTT("測試語音")
    tts = FakeTTS()
    pipeline = VoicePipeline(
        stt=stt,
        tts=tts,
        text_stream=BufferedCharacterTextStream(chunk_chars=3),
        config=VoicePipelineConfig(sentence_min_chars=2, sentence_max_chars=20),
    )
    events = [event async for event in pipeline.stream_turn(make_session(), audio=b"abcd")]
    types = [event.type for event in events]
    assert types[0] == "stt_final"
    assert "character_delta" in types
    assert "tts_audio" in types
    assert types[-2:] == ["trace", "final"]
    assert events[-1].text == "收到：測試語音。請放心！"
    assert tts.calls == ["收到：測試語音。", "請放心！"]


def pcm16(*samples: int) -> bytes:
    return b"".join(struct.pack("<h", value) for value in samples)


def test_energy_vad_and_utterance_segmenter():
    fmt = AudioFormat()
    vad = EnergyVoiceActivityDetector(rms_threshold=500)
    silence = AudioChunk(pcm16(0, 0, 0, 0), format=fmt)
    speech = AudioChunk(pcm16(1200, -1200, 1000, -1000), format=fmt)
    assert vad.is_speech(silence) is False
    assert vad.is_speech(speech) is True

    segmenter = UtteranceSegmenter(
        vad,
        config=UtteranceSegmenterConfig(min_speech_chunks=1, end_silence_chunks=2),
    )
    assert segmenter.push(silence) is None
    assert segmenter.push(speech) is None
    assert segmenter.push(silence) is None
    utterance = segmenter.push(silence)
    assert utterance is not None
    assert len(utterance) == 3


def test_sentence_segmenter_handles_chinese_punctuation():
    segmenter = SentenceSegmenter(min_chars=2, max_chars=20)
    assert segmenter.push("今天好") == ()
    assert segmenter.push("累。可是") == ("今天好累。",)
    assert segmenter.push("還是要做完！") == ("可是還是要做完！",)
    assert segmenter.flush() is None


@pytest.mark.asyncio
async def test_voice_pipeline_rejects_empty_and_oversized_audio():
    pipeline = VoicePipeline(
        stt=FakeSTT(),
        tts=FakeTTS(),
        config=VoicePipelineConfig(max_audio_bytes=4),
    )
    session = make_session()
    with pytest.raises(ValueError, match="must not be empty"):
        await pipeline.process_turn(session, audio=b"")
    with pytest.raises(ValueError, match="exceeds"):
        await pipeline.process_turn(session, audio=b"12345")


@pytest.mark.asyncio
async def test_voice_spans_share_trace_and_include_stt_tts():
    sink = InMemoryObservabilitySink()
    tracer = Tracer(sink)
    session = make_session(tracer=tracer)
    pipeline = VoicePipeline(stt=FakeSTT(), tts=FakeTTS(), tracer=tracer)
    result = await pipeline.process_turn(session, audio=b"1234")

    spans = [span for span in sink.spans if span.trace_id == result.trace_id]
    names = {span.name for span in spans}
    assert {"voice.turn", "voice.stt", "voice.character", "voice.tts"} <= names
    assert "runtime.process_event" in names


@pytest.mark.asyncio
async def test_stt_timeout_is_enforced():
    class SlowSTT:
        async def transcribe(self, audio: bytes, *, audio_format: AudioFormat):
            import asyncio
            await asyncio.sleep(0.05)
            return TranscriptResult(text="late")

    pipeline = VoicePipeline(
        stt=SlowSTT(),
        tts=FakeTTS(),
        config=VoicePipelineConfig(stt_timeout_seconds=0.001),
    )
    with pytest.raises(TimeoutError):
        await pipeline.process_turn(make_session(), audio=b"1234")

@pytest.mark.asyncio
async def test_voice_conversation_loop_connects_source_and_sink():
    from ai_character_engine.voice import VoiceConversationLoop

    fmt = AudioFormat()
    silence = AudioChunk(pcm16(0, 0, 0, 0), format=fmt)
    speech = AudioChunk(pcm16(1400, -1400, 1300, -1300), format=fmt)

    class Source:
        async def _gen(self):
            for chunk in (speech, silence, silence):
                yield chunk

        def chunks(self):
            return self._gen()

    class Sink:
        def __init__(self):
            self.played = []

        async def play(self, chunk):
            self.played.append(chunk)

    sink = Sink()
    pipeline = VoicePipeline(
        stt=FakeSTT("嗨"),
        tts=FakeTTS(),
        config=VoicePipelineConfig(sentence_min_chars=2),
    )
    loop = VoiceConversationLoop(
        pipeline=pipeline,
        segmenter=UtteranceSegmenter(
            EnergyVoiceActivityDetector(rms_threshold=500),
            config=UtteranceSegmenterConfig(end_silence_chunks=2),
        ),
    )
    events = [
        event
        async for event in loop.run(
            session=make_session(),
            source=Source(),
            sink=sink,
        )
    ]
    assert events[0].type == "speech_end"
    assert events[-1].type == "final"
    assert len(sink.played) >= 1
