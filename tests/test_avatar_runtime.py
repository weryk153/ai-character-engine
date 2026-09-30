import asyncio
from array import array

import pytest

from ai_character_engine.avatar import (
    AvatarRuntime,
    CompositeVisemeAdapter,
    ExpressionRequest,
    ExpressionScheduler,
    MetadataVisemeAdapter,
    Pcm16EnvelopeAnalyzer,
    TextVowelVisemeAdapter,
    Viseme,
)
from ai_character_engine.character.profile import CharacterProfile
from ai_character_engine.host import CharacterHostBridge
from ai_character_engine.live import DuplexVoiceConfig, LiveCharacterOrchestrator, LiveEventType, LiveRuntimeConfig
from ai_character_engine.llm.models import LLMResponse, LLMStreamChunk
from ai_character_engine.runtime import CharacterRuntime
from ai_character_engine.state import NoopStatePolicy, StatePatch
from ai_character_engine.voice import AudioChunk, AudioFormat, SynthesizedAudio


def pcm(*values: int) -> bytes:
    return array("h", values).tobytes()


def repeated_pcm(value: int, samples: int = 320) -> bytes:
    return array("h", [value] * samples).tobytes()


class ImmediateStreamingLLM:
    async def generate(self, messages, *, tools=None):
        raise AssertionError("stream_generate expected")

    async def stream_generate(self, messages, *, tools=None):
        yield LLMStreamChunk(text="あいうえお。")
        yield LLMStreamChunk(final=True, response=LLMResponse(text="あいうえお。"))


class SlowStreamingLLM:
    def __init__(self):
        self.release = asyncio.Event()
        self.first = asyncio.Event()

    async def generate(self, messages, *, tools=None):
        raise AssertionError("stream_generate expected")

    async def stream_generate(self, messages, *, tools=None):
        yield LLMStreamChunk(text="あああ。")
        self.first.set()
        await self.release.wait()
        yield LLMStreamChunk(text="いいい。")
        yield LLMStreamChunk(final=True, response=LLMResponse(text="あああ。いいい。"))


class NonStreamingLLM:
    async def generate(self, messages, *, tools=None):
        return LLMResponse(text="あいうえお。")


class MetadataStreamingTTS:
    async def synthesize(self, text, *, voice=None):
        raise AssertionError("streaming TTS expected")

    async def synthesize_stream(self, text, *, voice=None):
        yield AudioChunk(
            repeated_pcm(8000),
            AudioFormat(),
            metadata={
                "visemes": [
                    {"viseme": "aa", "start_ms": 0, "duration_ms": 10, "weight": 0.9},
                    {"viseme": "ih", "start_ms": 10, "duration_ms": 10},
                ]
            },
        )


class PlainTTS:
    async def synthesize(self, text, *, voice=None):
        return SynthesizedAudio(
            repeated_pcm(7000),
            AudioFormat(),
            duration_ms=20,
            metadata={"provider": "fake"},
        )


class RecordingSink:
    def __init__(self):
        self.played = []

    async def play(self, chunk):
        self.played.append(chunk)


class FirstSink(RecordingSink):
    def __init__(self):
        super().__init__()
        self.first_done = asyncio.Event()

    async def play(self, chunk):
        await super().play(chunk)
        self.first_done.set()

    async def stop(self):
        pass


class HappyPolicy(NoopStatePolicy):
    def on_event(self, event, state):
        return StatePatch(emotion="happy")


def runtime(llm, *, state_policy=None):
    return CharacterRuntime(
        character=CharacterProfile("v027", "V027", "avatar test"),
        llm=llm,
        state_policy=state_policy,
    )


def stream_config():
    return LiveRuntimeConfig(streaming_output=True)


def test_pcm16_envelope_tracks_silence_and_signal():
    analyzer = Pcm16EnvelopeAnalyzer()
    silence = analyzer.analyze(AudioChunk(repeated_pcm(0), AudioFormat()))
    assert silence
    assert max(point.amplitude for point in silence) == 0
    analyzer.reset()
    signal = analyzer.analyze(AudioChunk(repeated_pcm(12000), AudioFormat()))
    assert max(point.amplitude for point in signal) > 0.5
    assert all(0 <= point.amplitude <= 1 for point in signal)


def test_metadata_viseme_adapter_accepts_provider_neutral_timing():
    cues = MetadataVisemeAdapter().adapt(
        "ignored",
        duration_ms=100,
        metadata={"visemes": [{"viseme": "aa", "start_ms": 10, "duration_ms": 20, "weight": 0.8}]},
    )
    assert len(cues) == 1
    assert cues[0].viseme is Viseme.A
    assert cues[0].source == "tts_metadata"
    assert cues[0].start_offset_ms == 10


def test_text_viseme_adapter_is_explicit_low_confidence_fallback():
    cues = TextVowelVisemeAdapter().adapt("あいうえお", duration_ms=100)
    assert [cue.viseme for cue in cues] == [Viseme.A, Viseme.I, Viseme.U, Viseme.E, Viseme.O]
    assert all(cue.source == "text_heuristic" for cue in cues)
    assert TextVowelVisemeAdapter().adapt("あいうえお", duration_ms=100, allow_text_fallback=False) == ()


def test_composite_prefers_tts_metadata_over_text_heuristic():
    cues = CompositeVisemeAdapter().adapt(
        "あいうえお",
        duration_ms=100,
        metadata={"visemes": [{"viseme": "oh", "start_ms": 0, "duration_ms": 100}]},
    )
    assert len(cues) == 1
    assert cues[0].viseme is Viseme.O
    assert cues[0].source == "tts_metadata"


def test_expression_scheduler_host_priority_overrides_state_expression():
    scheduler = ExpressionScheduler()
    scheduler.schedule(ExpressionRequest("wink", duration_ms=200, priority=100), base_offset_ms=0)
    cues = scheduler.resolve(start_ms=0, duration_ms=50, emotion="happy")
    assert len(cues) == 1
    assert cues[0].expression == "wink"
    assert cues[0].source == "host"


def test_avatar_runtime_uses_global_audio_timeline_and_resets():
    avatar = AvatarRuntime()
    avatar.begin_turn("turn")
    one = avatar.feed_audio(
        text="あ", chunk=AudioChunk(repeated_pcm(1000), AudioFormat()),
        segment_sequence=0, chunk_index=0, duration_ms=20,
    )
    two = avatar.feed_audio(
        text="い", chunk=AudioChunk(repeated_pcm(1000), AudioFormat()),
        segment_sequence=0, chunk_index=1, duration_ms=30,
    )
    assert one.start_offset_ms == 0
    assert two.start_offset_ms == 20
    assert avatar.elapsed_ms == 50
    reset = avatar.end_turn()
    assert reset["elapsed_ms"] == 50
    assert not avatar.active


def test_avatar_runtime_scheduled_expression_beats_character_emotion():
    avatar = AvatarRuntime()
    avatar.begin_turn("turn")
    avatar.schedule_expression(ExpressionRequest("blink_custom", duration_ms=100, priority=200))
    bundle = avatar.feed_audio(
        text="あ", chunk=AudioChunk(repeated_pcm(1000), AudioFormat()),
        segment_sequence=0, chunk_index=0, duration_ms=20, emotion="happy",
    )
    assert [cue.expression for cue in bundle.expressions] == ["blink_custom"]


async def test_live_streaming_emits_avatar_cue_and_reset_without_raw_audio_in_event_data():
    avatar = AvatarRuntime()
    live = LiveCharacterOrchestrator(
        CharacterHostBridge(runtime(ImmediateStreamingLLM(), state_policy=HappyPolicy())),
        tts=MetadataStreamingTTS(),
        audio_sink=RecordingSink(),
        avatar_runtime=avatar,
        config=stream_config(),
        duplex=DuplexVoiceConfig(),
    )
    live.submit_text("hello")
    events = await live.run_once()
    cue = next(e for e in events if e.type is LiveEventType.AVATAR_CUE)
    reset = next(e for e in events if e.type is LiveEventType.AVATAR_RESET)
    assert cue.data["visemes"][0]["source"] == "tts_metadata"
    assert any(item["expression"] == "happy" for item in cue.data["expressions"])
    assert all(not isinstance(value, bytes) for value in cue.data.values())
    assert reset.data["reason"] == "completed"
    assert not avatar.active


async def test_streaming_tts_audio_preserves_provider_metadata():
    live = LiveCharacterOrchestrator(
        CharacterHostBridge(runtime(ImmediateStreamingLLM())),
        tts=MetadataStreamingTTS(),
        config=stream_config(),
    )
    live.submit_text("hello")
    events = await live.run_once()
    audio = next(e for e in events if e.type is LiveEventType.TTS_AUDIO)
    assert audio.audio.metadata["visemes"][0]["viseme"] == "aa"


async def test_streaming_without_provider_visemes_does_not_repeat_whole_sentence_heuristic_per_chunk():
    class TwoChunkTTS:
        async def synthesize(self, text, *, voice=None):
            raise AssertionError

        async def synthesize_stream(self, text, *, voice=None):
            yield AudioChunk(repeated_pcm(6000), AudioFormat())
            yield AudioChunk(repeated_pcm(6000), AudioFormat())

    live = LiveCharacterOrchestrator(
        CharacterHostBridge(runtime(ImmediateStreamingLLM())),
        tts=TwoChunkTTS(),
        avatar_runtime=AvatarRuntime(),
        config=stream_config(),
    )
    live.submit_text("hello")
    events = await live.run_once()
    cues = [e for e in events if e.type is LiveEventType.AVATAR_CUE]
    assert len(cues) == 2
    assert all(e.data["visemes"] == [] for e in cues)
    assert all(e.data["envelope"] for e in cues)


async def test_nonstreaming_tts_can_use_text_viseme_fallback():
    live = LiveCharacterOrchestrator(
        CharacterHostBridge(runtime(NonStreamingLLM())),
        tts=PlainTTS(),
        avatar_runtime=AvatarRuntime(),
        config=LiveRuntimeConfig(streaming_output=False),
    )
    live.submit_text("hello")
    events = await live.run_once()
    cue = next(e for e in events if e.type is LiveEventType.AVATAR_CUE)
    assert [item["viseme"] for item in cue.data["visemes"]] == ["a", "i", "u", "e", "o"]
    assert next(e for e in events if e.type is LiveEventType.AVATAR_RESET).data["reason"] == "completed"


async def test_no_avatar_runtime_preserves_v026_event_contract():
    live = LiveCharacterOrchestrator(
        CharacterHostBridge(runtime(NonStreamingLLM())),
        tts=PlainTTS(),
        audio_sink=RecordingSink(),
        config=LiveRuntimeConfig(streaming_output=False),
        duplex=DuplexVoiceConfig(),
    )
    live.submit_text("hello")
    events = await live.run_once()
    assert [e.type for e in events] == [
        LiveEventType.TURN_STARTED,
        LiveEventType.REPLY,
        LiveEventType.PLAYBACK_STARTED,
        LiveEventType.TTS_AUDIO,
        LiveEventType.PLAYBACK_FINISHED,
    ]


async def test_generation_interruption_resets_avatar_runtime():
    llm = SlowStreamingLLM()
    sink = FirstSink()
    avatar = AvatarRuntime()
    live = LiveCharacterOrchestrator(
        CharacterHostBridge(runtime(llm)),
        tts=PlainTTS(),
        audio_sink=sink,
        avatar_runtime=avatar,
        config=stream_config(),
        duplex=DuplexVoiceConfig(),
    )
    live.submit_text("hello")
    task = asyncio.create_task(live.run_once())
    await asyncio.wait_for(sink.first_done.wait(), 1)
    await asyncio.sleep(0)
    interrupted = await live.interrupt_output(reason="test")
    assert interrupted is not None
    events = await asyncio.wait_for(task, 1)
    reset = next(e for e in events if e.type is LiveEventType.AVATAR_RESET)
    assert reset.data["reason"] == "interrupted"
    assert not avatar.active
