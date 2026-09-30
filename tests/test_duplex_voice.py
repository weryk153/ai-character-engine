import asyncio
from array import array

import pytest

from ai_character_engine import CharacterProfile, CharacterRuntime
from ai_character_engine.host import CharacterHostBridge
from ai_character_engine.live import (
    DuplexVoiceConfig,
    DuplexVoiceError,
    LiveCharacterOrchestrator,
    LiveEventType,
    LiveVoicePhase,
)
from ai_character_engine.llm.models import LLMResponse
from ai_character_engine.voice import AudioChunk, AudioFormat, SynthesizedAudio, TranscriptResult


class ReplyLLM:
    def __init__(self, text="第一句。第二句。"):
        self.text = text

    async def generate(self, messages, *, tools=None):
        return LLMResponse(text=self.text)


class BlockingLLM:
    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def generate(self, messages, *, tools=None):
        self.started.set()
        await self.release.wait()
        return LLMResponse(text="too late")


class FakeSTT:
    def __init__(self, text="新的問題"):
        self.text = text
        self.calls = []

    async def transcribe(self, audio, *, audio_format):
        self.calls.append((audio, audio_format))
        return TranscriptResult(self.text, language="zh", confidence=0.95)


class FakeTTS:
    def __init__(self):
        self.calls = []

    async def synthesize(self, text, *, voice=None):
        self.calls.append(text)
        return SynthesizedAudio(text.encode(), AudioFormat(), duration_ms=20, latency_ms=1)


class RecordingSink:
    def __init__(self):
        self.played = []

    async def play(self, chunk):
        self.played.append(chunk)


class InterruptibleSink:
    def __init__(self):
        self.played = []
        self.second_started = asyncio.Event()
        self.stop_calls = 0

    async def play(self, chunk):
        self.played.append(chunk)
        if len(self.played) == 2:
            self.second_started.set()
            await asyncio.Event().wait()

    async def stop(self):
        self.stop_calls += 1


def runtime(llm=None):
    return CharacterRuntime(
        character=CharacterProfile("duplex", "Duplex", "voice test"),
        llm=llm or ReplyLLM(),
    )


def pcm_chunk(value: int, sequence: int) -> AudioChunk:
    samples = array("h", [value] * 160)
    return AudioChunk(samples.tobytes(), AudioFormat(), sequence=sequence)


def speech(sequence: int) -> AudioChunk:
    return pcm_chunk(2000, sequence)


def silence(sequence: int) -> AudioChunk:
    return pcm_chunk(0, sequence)


def safe_duplex(**kwargs):
    return DuplexVoiceConfig(
        automatic_barge_in=True,
        echo_cancellation_confirmed=True,
        barge_in_min_speech_chunks=2,
        end_silence_chunks=2,
        **kwargs,
    )


def test_automatic_barge_in_requires_echo_cancellation_confirmation():
    with pytest.raises(ValueError, match="echo_cancellation_confirmed"):
        DuplexVoiceConfig(automatic_barge_in=True)


def test_automatic_barge_in_with_tts_rejects_non_interruptible_sink():
    with pytest.raises(DuplexVoiceError, match="audio_sink.stop"):
        LiveCharacterOrchestrator(
            CharacterHostBridge(runtime()),
            stt=FakeSTT(),
            tts=FakeTTS(),
            audio_sink=RecordingSink(),
            duplex=safe_duplex(),
        )


async def test_vad_continuous_input_emits_boundaries_and_queues_audio_turn():
    stt = FakeSTT()
    live = LiveCharacterOrchestrator(
        CharacterHostBridge(runtime(ReplyLLM("收到"))),
        stt=stt,
        duplex=DuplexVoiceConfig(end_silence_chunks=2),
    )
    start = await live.ingest_audio_chunk(speech(0))
    assert [e.type for e in start] == [LiveEventType.SPEECH_STARTED]
    assert await live.ingest_audio_chunk(speech(1)) == ()
    assert await live.ingest_audio_chunk(silence(2)) == ()
    end = await live.ingest_audio_chunk(silence(3))
    assert [e.type for e in end] == [LiveEventType.SPEECH_ENDED]
    assert live.pending_inputs == 1

    events = await live.run_once()
    assert LiveEventType.STT_FINAL in [e.type for e in events]
    assert events[-1].type == LiveEventType.REPLY
    assert stt.calls and stt.calls[0][0]


async def test_generation_can_be_cancelled_by_confirmed_vad_barge_in():
    llm = BlockingLLM()
    live = LiveCharacterOrchestrator(
        CharacterHostBridge(runtime(llm)),
        stt=FakeSTT(),
        duplex=safe_duplex(),
    )
    live.submit_text("old turn")
    task = asyncio.create_task(live.run_once())
    await llm.started.wait()
    assert live.voice_phase is LiveVoicePhase.GENERATING

    first = await live.ingest_audio_chunk(speech(0))
    second = await live.ingest_audio_chunk(speech(1))
    assert first[0].type == LiveEventType.SPEECH_STARTED
    assert [e.type for e in second] == [LiveEventType.BARGE_IN_DETECTED, LiveEventType.TURN_INTERRUPTED]

    old_events = await asyncio.wait_for(task, 1)
    assert old_events[-1].type == LiveEventType.TURN_INTERRUPTED
    assert live.bridge.runtime.history == []
    assert live.voice_phase is LiveVoicePhase.IDLE


async def test_sentence_tts_tracks_only_completed_playback_as_heard():
    sink = InterruptibleSink()
    tts = FakeTTS()
    bridge = CharacterHostBridge(runtime(ReplyLLM("第一句。第二句。")))
    live = LiveCharacterOrchestrator(
        bridge,
        stt=FakeSTT(),
        tts=tts,
        audio_sink=sink,
        duplex=safe_duplex(),
    )
    live.submit_text("speak")
    task = asyncio.create_task(live.run_once())
    await sink.second_started.wait()
    assert live.voice_phase is LiveVoicePhase.PLAYBACK
    assert live.played_text == "第一句。"

    await live.ingest_audio_chunk(speech(0))
    control = await live.ingest_audio_chunk(speech(1))
    assert [e.type for e in control] == [LiveEventType.BARGE_IN_DETECTED, LiveEventType.PLAYBACK_INTERRUPTED]
    assert control[-1].text == "第一句。"
    assert sink.stop_calls == 1

    events = await asyncio.wait_for(task, 1)
    types = [e.type for e in events]
    assert LiveEventType.PLAYBACK_STARTED in types
    assert LiveEventType.PLAYBACK_FINISHED not in types
    assert tts.calls == ["第一句。", "第二句。"]
    assert bridge.runtime.history[-1].content == "第一句。 [Interrupted by user]"


async def test_completed_sentence_playback_emits_finished_and_full_played_text():
    sink = RecordingSink()
    tts = FakeTTS()
    live = LiveCharacterOrchestrator(
        CharacterHostBridge(runtime()),
        tts=tts,
        audio_sink=sink,
        duplex=DuplexVoiceConfig(),
    )
    live.submit_text("hello")
    events = await live.run_once()
    types = [e.type for e in events]
    assert types == [
        LiveEventType.TURN_STARTED,
        LiveEventType.REPLY,
        LiveEventType.PLAYBACK_STARTED,
        LiveEventType.TTS_AUDIO,
        LiveEventType.TTS_AUDIO,
        LiveEventType.PLAYBACK_FINISHED,
    ]
    assert tts.calls == ["第一句。", "第二句。"]
    assert live.played_text == "第一句。第二句。"
    assert events[-1].data["segments"] == 2


async def test_tts_without_sink_is_generated_but_not_claimed_as_heard():
    tts = FakeTTS()
    live = LiveCharacterOrchestrator(
        CharacterHostBridge(runtime()), tts=tts, duplex=DuplexVoiceConfig()
    )
    live.submit_text("hello")
    events = await live.run_once()
    assert [e.type for e in events] == [
        LiveEventType.TURN_STARTED,
        LiveEventType.REPLY,
        LiveEventType.TTS_AUDIO,
        LiveEventType.TTS_AUDIO,
    ]
    assert all(e.data["played"] is False for e in events if e.type is LiveEventType.TTS_AUDIO)
    assert live.played_text == ""


async def test_manual_interrupt_does_not_require_echo_cancellation():
    sink = InterruptibleSink()
    bridge = CharacterHostBridge(runtime())
    live = LiveCharacterOrchestrator(
        bridge, tts=FakeTTS(), audio_sink=sink, duplex=DuplexVoiceConfig()
    )
    live.submit_text("hello")
    task = asyncio.create_task(live.run_once())
    await sink.second_started.wait()
    event = await live.interrupt_output(reason="push_to_talk")
    assert event is not None and event.type is LiveEventType.PLAYBACK_INTERRUPTED
    assert event.data["reason"] == "push_to_talk"
    await asyncio.wait_for(task, 1)
    assert bridge.runtime.history[-1].content == "第一句。 [Interrupted by user]"


async def test_barge_in_is_disabled_by_default_even_during_generation():
    llm = BlockingLLM()
    live = LiveCharacterOrchestrator(CharacterHostBridge(runtime(llm)), stt=FakeSTT())
    live.submit_text("old")
    task = asyncio.create_task(live.run_once())
    await llm.started.wait()
    await live.ingest_audio_chunk(speech(0))
    events = await live.ingest_audio_chunk(speech(1))
    assert all(e.type is not LiveEventType.BARGE_IN_DETECTED for e in events)
    assert not task.done()
    llm.release.set()
    result = await asyncio.wait_for(task, 1)
    assert result[-1].type == LiveEventType.REPLY


async def test_flush_audio_input_queues_trailing_speech_without_fake_silence():
    live = LiveCharacterOrchestrator(CharacterHostBridge(runtime()), stt=FakeSTT())
    await live.ingest_audio_chunk(speech(0))
    flushed = await live.flush_audio_input()
    assert flushed[0].type is LiveEventType.SPEECH_ENDED
    assert flushed[0].data["flushed"] is True
    assert live.pending_inputs == 1


async def test_interrupt_when_idle_is_a_noop():
    live = LiveCharacterOrchestrator(CharacterHostBridge(runtime()))
    assert await live.interrupt_output() is None

async def test_playback_suppresses_microphone_vad_without_echo_cancellation():
    sink = InterruptibleSink()
    live = LiveCharacterOrchestrator(
        CharacterHostBridge(runtime()),
        stt=FakeSTT(),
        tts=FakeTTS(),
        audio_sink=sink,
        duplex=DuplexVoiceConfig(),
    )
    live.submit_text("hello")
    task = asyncio.create_task(live.run_once())
    await sink.second_started.wait()
    assert live.voice_phase is LiveVoicePhase.PLAYBACK
    assert await live.ingest_audio_chunk(speech(0)) == ()
    assert live.pending_inputs == 0
    await live.interrupt_output(reason="finish-test")
    await asyncio.wait_for(task, 1)


def test_sounddevice_output_exposes_async_stop_contract():
    from ai_character_engine.voice import SoundDeviceOutput, supports_interruptible_playback

    assert supports_interruptible_playback(SoundDeviceOutput()) is True
