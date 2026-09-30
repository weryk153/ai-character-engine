import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from ai_character_engine import CharacterProfile, CharacterRuntime
from ai_character_engine.autonomy import AutonomyPolicy, AutonomyScheduler, ProactiveCandidate
from ai_character_engine.host import CharacterHostBridge
from ai_character_engine.live import (
    LiveCharacterOrchestrator,
    LiveEventType,
    LiveRuntimeBackpressureError,
    LiveRuntimeConfig,
    LiveRuntimeError,
)
from ai_character_engine.llm.models import LLMResponse
from ai_character_engine.vision import ImageInput, VisionAnalysis, VisionFrame, VisionPipeline
from ai_character_engine.voice.models import AudioFormat, SynthesizedAudio, TranscriptResult

NOW = datetime(2026, 9, 22, 8, tzinfo=UTC)


class EchoLLM:
    async def generate(self, messages, *, tools=None):
        return LLMResponse(text=f"reply:{messages[-1].content[:40]}")


class RecordingLLM:
    def __init__(self):
        self.messages = []

    async def generate(self, messages, *, tools=None):
        self.messages.append(messages)
        return LLMResponse(text="reply")


class FakeSTT:
    def __init__(self, text="hello by voice"):
        self.text = text
        self.calls = []

    async def transcribe(self, audio, *, audio_format):
        self.calls.append((audio, audio_format))
        return TranscriptResult(self.text, language="en", confidence=0.9)


class FakeTTS:
    def __init__(self):
        self.calls = []

    async def synthesize(self, text, *, voice=None):
        self.calls.append((text, voice))
        return SynthesizedAudio(b"pcm", AudioFormat(), duration_ms=10, latency_ms=2)


class Sink:
    def __init__(self):
        self.chunks = []

    async def play(self, chunk):
        self.chunks.append(chunk)


class Vision:
    def __init__(self):
        self.calls = []

    async def analyze(self, image, *, prompt=None):
        self.calls.append((image, prompt))
        return VisionAnalysis("a red cup on the desk", provider="fake-vlm", model="v1", tags=("cup",), confidence=0.8)


def engine(llm=None):
    return CharacterRuntime(
        character=CharacterProfile("live", "Live", "Responsive"),
        llm=llm or EchoLLM(),
    )


def png_frame(frame_id="frame-1"):
    # Validation only needs a PNG signature and IHDR dimensions for these fake-provider tests.
    data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8 + (1).to_bytes(4, "big") + (1).to_bytes(4, "big")
    return VisionFrame(ImageInput.from_bytes(data, mime_type="image/png"), source_type="camera", id=frame_id, captured_at=NOW)


async def test_text_turn_flows_through_bridge():
    runtime = engine()
    live = LiveCharacterOrchestrator(CharacterHostBridge(runtime))
    input_id = live.submit_text("hello")
    events = await live.run_once()
    assert [e.type for e in events] == [LiveEventType.TURN_STARTED, LiveEventType.REPLY]
    assert events[-1].input_id == input_id
    assert events[-1].text.startswith("reply:")
    assert len(runtime.history) == 2


async def test_audio_turn_uses_stt_bridge_tts_and_sink():
    stt, tts, sink = FakeSTT(), FakeTTS(), Sink()
    live = LiveCharacterOrchestrator(CharacterHostBridge(engine()), stt=stt, tts=tts, audio_sink=sink)
    live.submit_audio(b"voice")
    events = await live.run_once()
    assert [e.type for e in events] == [
        LiveEventType.TURN_STARTED, LiveEventType.STT_FINAL, LiveEventType.REPLY, LiveEventType.TTS_AUDIO
    ]
    assert stt.calls[0][0] == b"voice"
    assert tts.calls and sink.chunks and sink.chunks[0].data == b"pcm"


def test_audio_requires_stt_and_is_bounded():
    live = LiveCharacterOrchestrator(CharacterHostBridge(engine()), config=LiveRuntimeConfig(max_audio_bytes=4))
    with pytest.raises(LiveRuntimeError):
        live.submit_audio(b"x")
    live = LiveCharacterOrchestrator(CharacterHostBridge(engine()), stt=FakeSTT(), config=LiveRuntimeConfig(max_audio_bytes=4))
    with pytest.raises(ValueError):
        live.submit_audio(b"12345")


async def test_foreground_input_is_dispatched_before_autonomy():
    runtime = engine()
    scheduler = AutonomyScheduler(AutonomyPolicy(global_cooldown=timedelta(0)))
    scheduler.submit(ProactiveCandidate("idle thought", "test", created_at=NOW))
    live = LiveCharacterOrchestrator(CharacterHostBridge(runtime), scheduler=scheduler)
    live.submit_text("foreground")
    first = await live.run_once()
    second = await live.run_once()
    assert first[-1].type == LiveEventType.REPLY
    assert second[-1].type == LiveEventType.AUTONOMY_DELIVERED


async def test_autonomy_failure_is_exposed_without_throwing_from_run_once():
    class Bad:
        async def generate(self, messages, *, tools=None):
            raise RuntimeError("synthetic")

    runtime = engine(Bad())
    scheduler = AutonomyScheduler(AutonomyPolicy(global_cooldown=timedelta(0), retry_backoff=timedelta(0)))
    scheduler.submit(ProactiveCandidate("idle", "test", created_at=NOW))
    live = LiveCharacterOrchestrator(CharacterHostBridge(runtime), scheduler=scheduler)
    events = await live.run_once()
    assert events[0].type == LiveEventType.AUTONOMY_FAILED
    assert events[0].data["reason"] == "runtime_failed"


async def test_recent_frame_is_attached_to_next_turn_only_as_ephemeral_vision_context():
    llm = RecordingLLM()
    vision_provider = Vision()
    vision = VisionPipeline(provider=vision_provider)
    live = LiveCharacterOrchestrator(CharacterHostBridge(engine(llm), vision=vision))
    live.observe_frame(png_frame())
    live.submit_text("what do you see?")
    events = await live.run_once()
    assert vision_provider.calls
    assert events[-1].data["frames"] == 1
    prompt = llm.messages[-1][-1].content
    assert "Visual observation (camera): a red cup on the desk" in prompt
    assert "PNG" not in prompt


async def test_frame_ttl_prevents_stale_camera_context():
    clock = [100.0]
    vision_provider = Vision()
    live = LiveCharacterOrchestrator(
        CharacterHostBridge(engine(), vision=VisionPipeline(provider=vision_provider)),
        config=LiveRuntimeConfig(context_frame_ttl_seconds=1),
        monotonic=lambda: clock[0],
    )
    live.observe_frame(png_frame())
    clock[0] += 2
    live.submit_text("hello")
    events = await live.run_once()
    assert events[-1].data["frames"] == 0
    assert vision_provider.calls == []


async def test_visual_candidate_carries_metadata_not_raw_image_bytes():
    scheduler = AutonomyScheduler()
    live = LiveCharacterOrchestrator(
        CharacterHostBridge(engine(), vision=VisionPipeline(provider=Vision())),
        scheduler=scheduler,
    )
    admission = await live.submit_visual_candidate(png_frame(), dedupe_key="scene")
    assert admission is not None and admission.status == "accepted"
    item = scheduler.pending[0]
    assert item.payload["vision"]["frame_id"] == "frame-1"
    assert item.payload["vision"]["provider"] == "fake-vlm"
    assert b"PNG" not in repr(item.payload).encode()
    assert "data" not in item.payload["vision"]


def test_foreground_queue_has_explicit_backpressure():
    live = LiveCharacterOrchestrator(
        CharacterHostBridge(engine()), config=LiveRuntimeConfig(max_pending_inputs=1)
    )
    live.submit_text("one")
    with pytest.raises(LiveRuntimeBackpressureError):
        live.submit_text("two")


async def test_input_error_becomes_structured_event_and_loop_can_continue():
    class EmptySTT:
        async def transcribe(self, audio, *, audio_format):
            return TranscriptResult("   ")

    live = LiveCharacterOrchestrator(CharacterHostBridge(engine()), stt=EmptySTT())
    live.submit_audio(b"x")
    first = await live.run_once()
    assert first[-1].type == LiveEventType.ERROR
    live.submit_text("recover")
    second = await live.run_once()
    assert second[-1].type == LiveEventType.REPLY


async def test_run_generator_is_continuous_and_close_stops_it():
    live = LiveCharacterOrchestrator(
        CharacterHostBridge(engine()), config=LiveRuntimeConfig(poll_interval_seconds=0.01)
    )
    live.submit_text("hello")
    seen = []

    async def consume():
        async for event in live.run():
            seen.append(event.type)
            if event.type is LiveEventType.REPLY:
                await live.close()

    await asyncio.wait_for(consume(), 1)
    assert seen == [LiveEventType.TURN_STARTED, LiveEventType.REPLY]
    assert live.closed


async def test_only_one_run_iterator_can_be_active():
    live = LiveCharacterOrchestrator(
        CharacterHostBridge(engine()), config=LiveRuntimeConfig(poll_interval_seconds=0.05)
    )
    first = live.run()
    task = asyncio.create_task(anext(first))
    await asyncio.sleep(0)
    second = live.run()
    with pytest.raises(LiveRuntimeError):
        await anext(second)
    live.submit_text("wake")
    await task
    await first.aclose()
    await live.close()
