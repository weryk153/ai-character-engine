import asyncio
from array import array

import pytest

from ai_character_engine.character.profile import CharacterProfile
from ai_character_engine.host import CharacterHostBridge
from ai_character_engine.live import (
    DuplexVoiceConfig,
    LiveCharacterOrchestrator,
    LiveEventType,
    LiveRuntimeConfig,
    LiveVoicePhase,
    MouthActivityCue,
)
from ai_character_engine.llm.models import LLMResponse, LLMStreamChunk, Message
from ai_character_engine.runtime import CharacterRuntime
from ai_character_engine.state import NoopStatePolicy, StatePatch
from ai_character_engine.tools.models import ToolDefinition
from ai_character_engine.voice import AudioChunk, AudioFormat, SynthesizedAudio


class StreamingReplyLLM:
    def __init__(self, parts=("第一句。", "第二句。")):
        self.parts = parts
        self.first_delta = asyncio.Event()
        self.release = asyncio.Event()

    async def generate(self, messages, *, tools=None):
        raise AssertionError("stream_generate should be used")

    async def stream_generate(self, messages, *, tools=None):
        text = ""
        for index, part in enumerate(self.parts):
            text += part
            yield LLMStreamChunk(text=part)
            if index == 0:
                self.first_delta.set()
                await self.release.wait()
        yield LLMStreamChunk(final=True, response=LLMResponse(text=text))


class ImmediateStreamingLLM:
    async def generate(self, messages, *, tools=None):
        raise AssertionError("stream_generate should be used")

    async def stream_generate(self, messages, *, tools=None):
        yield LLMStreamChunk(text="第一句。")
        yield LLMStreamChunk(text="第二句。")
        yield LLMStreamChunk(final=True, response=LLMResponse(text="第一句。第二句。"))


class NonStreamingLLM:
    async def generate(self, messages, *, tools=None):
        return LLMResponse(text="完整回答。")


class FakeTTS:
    def __init__(self):
        self.calls = []
        self.first_call = asyncio.Event()

    async def synthesize(self, text, *, voice=None):
        self.calls.append(text)
        self.first_call.set()
        return SynthesizedAudio(
            text.encode("utf-8"), AudioFormat(), duration_ms=25, latency_ms=1
        )


class StreamingTTS:
    def __init__(self):
        self.calls = []

    async def synthesize(self, text, *, voice=None):
        raise AssertionError("synthesize_stream should be preferred")

    async def synthesize_stream(self, text, *, voice=None):
        self.calls.append(text)
        yield AudioChunk(b"\x00\x00" * 160, AudioFormat())
        yield AudioChunk(b"\x00\x00" * 160, AudioFormat())


class RecordingSink:
    def __init__(self):
        self.played = []

    async def play(self, chunk):
        self.played.append(chunk)


class FirstSentenceSink:
    def __init__(self):
        self.played = []
        self.first_done = asyncio.Event()
        self.stop_calls = 0

    async def play(self, chunk):
        self.played.append(chunk)
        if len(self.played) == 1:
            self.first_done.set()

    async def stop(self):
        self.stop_calls += 1


class TrustPolicy(NoopStatePolicy):
    def on_event(self, event, state):
        return StatePatch(trust_delta=5)


def runtime(llm, *, state_policy=None):
    return CharacterRuntime(
        character=CharacterProfile("v026", "V026", "streaming test"),
        llm=llm,
        state_policy=state_policy,
    )


def streaming_config(**kwargs):
    return LiveRuntimeConfig(streaming_output=True, **kwargs)


def test_mouth_activity_cue_is_provider_neutral_and_validated():
    cue = MouthActivityCue(1, 2, duration_ms=40)
    assert cue.to_dict()["align_to"] == "tts_audio"
    assert "viseme" not in cue.to_dict()
    with pytest.raises(ValueError):
        MouthActivityCue(-1, 0)


async def test_streaming_llm_starts_tts_before_generation_finishes():
    llm = StreamingReplyLLM()
    tts = FakeTTS()
    live = LiveCharacterOrchestrator(
        CharacterHostBridge(runtime(llm)),
        tts=tts,
        config=streaming_config(),
    )
    live.submit_text("hello")
    task = asyncio.create_task(live.run_once())
    await asyncio.wait_for(tts.first_call.wait(), 1)
    assert not task.done()
    assert tts.calls == ["第一句。"]
    llm.release.set()
    events = await asyncio.wait_for(task, 1)
    assert [e.text for e in events if e.type is LiveEventType.CHARACTER_DELTA] == [
        "第一句。",
        "第二句。",
    ]
    metrics = next(e for e in events if e.type is LiveEventType.STREAM_METRICS)
    assert metrics.data["llm_ttft_ms"] is not None
    assert metrics.data["ttfa_ms"] is not None


async def test_run_yields_character_delta_before_llm_final_response():
    llm = StreamingReplyLLM(parts=("先說。", "再說。"))
    live = LiveCharacterOrchestrator(
        CharacterHostBridge(runtime(llm)), config=streaming_config()
    )
    live.submit_text("hello")
    stream = live.run()
    first = await asyncio.wait_for(anext(stream), 1)
    second = await asyncio.wait_for(anext(stream), 1)
    assert first.type is LiveEventType.TURN_STARTED
    assert second.type is LiveEventType.CHARACTER_DELTA
    assert second.text == "先說。"
    assert not llm.release.is_set()
    llm.release.set()
    while True:
        event = await asyncio.wait_for(anext(stream), 1)
        if event.type is LiveEventType.STREAM_METRICS:
            break
    await live.close()
    await stream.aclose()


async def test_streaming_tts_yields_multiple_audio_chunks_and_mouth_cues():
    tts = StreamingTTS()
    sink = RecordingSink()
    live = LiveCharacterOrchestrator(
        CharacterHostBridge(runtime(ImmediateStreamingLLM())),
        tts=tts,
        audio_sink=sink,
        config=streaming_config(),
        duplex=DuplexVoiceConfig(),
    )
    live.submit_text("hello")
    events = await live.run_once()
    audio = [e for e in events if e.type is LiveEventType.TTS_AUDIO]
    cues = [e for e in events if e.type is LiveEventType.MOUTH_CUE]
    assert len(audio) == 4
    assert len(cues) == 4
    assert all(e.data["streaming_tts"] is True for e in audio)
    assert len(sink.played) == 4
    assert live.played_text == "第一句。第二句。"


async def test_nonstreaming_llm_falls_back_to_one_full_delta():
    tts = FakeTTS()
    live = LiveCharacterOrchestrator(
        CharacterHostBridge(runtime(NonStreamingLLM())),
        tts=tts,
        config=streaming_config(),
    )
    live.submit_text("hello")
    events = await live.run_once()
    deltas = [e.text for e in events if e.type is LiveEventType.CHARACTER_DELTA]
    assert deltas == ["完整回答。"]
    assert tts.calls == ["完整回答。"]


async def test_nonstreaming_tts_is_a_supported_fallback():
    live = LiveCharacterOrchestrator(
        CharacterHostBridge(runtime(ImmediateStreamingLLM())),
        tts=FakeTTS(),
        config=streaming_config(prefer_streaming_tts=True),
    )
    live.submit_text("hello")
    events = await live.run_once()
    audio = [e for e in events if e.type is LiveEventType.TTS_AUDIO]
    assert len(audio) == 2
    assert all(e.data["streaming_tts"] is False for e in audio)


async def test_tool_enabled_streaming_text_is_buffered_until_final():
    llm = StreamingReplyLLM(parts=("暫存一。", "暫存二。"))
    engine = runtime(llm)
    seen = []
    tool = ToolDefinition(
        name="noop",
        description="test",
        parameters={"type": "object", "properties": {}},
    )
    task = asyncio.create_task(
        engine._generate_response(
            [Message("user", "hello")],
            tools=[tool],
            on_text_delta=lambda text: seen.append(text),
        )
    )
    await llm.first_delta.wait()
    await asyncio.sleep(0)
    assert seen == []
    llm.release.set()
    response = await task
    assert response.text == "暫存一。暫存二。"
    assert seen == ["暫存一。", "暫存二。"]
    assert response.metadata["streaming"]["tool_safe_buffering"] is True


async def test_a_host_can_have_text_streamed_at_once_even_with_tools():
    """Measured on a local 9B model: with one tool registered the first words
    arrived when the whole reply was finished (4-5 s) instead of after 2 s,
    because a voice host cannot start speaking until then."""
    llm = StreamingReplyLLM(parts=("第一句。", "第二句。"))
    engine = runtime(llm)
    engine.stream_text_with_tools = True
    seen = []
    tool = ToolDefinition(
        name="noop",
        description="test",
        parameters={"type": "object", "properties": {}},
    )
    task = asyncio.create_task(
        engine._generate_response(
            [Message("user", "hello")],
            tools=[tool],
            on_text_delta=lambda text: seen.append(text),
        )
    )
    await llm.first_delta.wait()
    await asyncio.sleep(0)
    assert seen == ["第一句。"]
    llm.release.set()
    response = await task
    assert seen == ["第一句。", "第二句。"]
    assert response.text == "第一句。第二句。"
    assert response.metadata["streaming"]["tool_safe_buffering"] is False
    assert response.metadata["streaming"]["runtime_streamed_text"] is True


def test_text_is_held_back_with_tools_unless_the_host_asks():
    assert runtime(StreamingReplyLLM()).stream_text_with_tools is False


async def test_interrupt_after_streamed_sentence_rolls_back_state_but_keeps_heard_history():
    llm = StreamingReplyLLM(parts=("第一句。", "永遠等不到。"))
    sink = FirstSentenceSink()
    live = LiveCharacterOrchestrator(
        CharacterHostBridge(runtime(llm, state_policy=TrustPolicy())),
        tts=FakeTTS(),
        audio_sink=sink,
        config=streaming_config(),
        duplex=DuplexVoiceConfig(),
    )
    live.submit_text("原本問題")
    task = asyncio.create_task(live.run_once())
    await asyncio.wait_for(sink.first_done.wait(), 1)
    # after one full segment, generation is still waiting for the next delta
    await asyncio.sleep(0)
    assert live.played_text == "第一句。"
    assert live.voice_phase in (LiveVoicePhase.GENERATING, LiveVoicePhase.PLAYBACK)
    event = await live.interrupt_output(reason="push_to_talk")
    assert event is not None
    events = await asyncio.wait_for(task, 1)
    assert next(e for e in events if e.type is LiveEventType.STREAM_METRICS).data["interrupted"] is True
    assert live.bridge.runtime.state.trust == 50
    assert [m.role for m in live.bridge.runtime.history] == ["user", "assistant"]
    assert live.bridge.runtime.history[0].content == "原本問題"
    assert live.bridge.runtime.history[1].content == "第一句。 [Interrupted by user]"


async def test_successful_streaming_turn_commits_history_once():
    live = LiveCharacterOrchestrator(
        CharacterHostBridge(runtime(ImmediateStreamingLLM())),
        tts=FakeTTS(),
        audio_sink=RecordingSink(),
        config=streaming_config(),
    )
    live.submit_text("hello")
    await live.run_once()
    history = live.bridge.runtime.history
    assert [m.role for m in history] == ["user", "assistant"]
    assert history[1].content == "第一句。第二句。"


async def test_streaming_can_be_opted_out_without_changing_v025_event_contract():
    live = LiveCharacterOrchestrator(
        CharacterHostBridge(runtime(NonStreamingLLM())),
        tts=FakeTTS(),
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
    assert all(e.type is not LiveEventType.CHARACTER_DELTA for e in events)


async def test_metrics_report_first_playback_when_sink_is_present():
    live = LiveCharacterOrchestrator(
        CharacterHostBridge(runtime(ImmediateStreamingLLM())),
        tts=FakeTTS(),
        audio_sink=RecordingSink(),
        config=streaming_config(),
    )
    live.submit_text("hello")
    events = await live.run_once()
    metrics = next(e for e in events if e.type is LiveEventType.STREAM_METRICS)
    assert metrics.data["first_playback_ms"] is not None
    assert metrics.data["ttfa_ms"] <= metrics.data["first_playback_ms"] + 5
    assert metrics.data["audio_chunks"] == 2


def test_streaming_config_validates_bounded_queue():
    with pytest.raises(ValueError, match="stream_queue_size"):
        LiveRuntimeConfig(streaming_output=True, stream_queue_size=0)

class GatewayStreamingClient:
    def __init__(self, *, fail_before=False, fail_after=False, text="串流。"):
        self.fail_before = fail_before
        self.fail_after = fail_after
        self.text = text
        self.calls = 0

    async def generate(self, messages, *, tools=None):
        self.calls += 1
        return LLMResponse(text=self.text, model="fallback")

    async def stream_generate(self, messages, *, tools=None):
        from ai_character_engine.llm.errors import LLMError
        self.calls += 1
        if self.fail_before:
            raise LLMError("before")
        yield LLMStreamChunk(text=self.text)
        if self.fail_after:
            raise LLMError("after")
        yield LLMStreamChunk(final=True, response=LLMResponse(text=self.text, model="stream"))


async def test_gateway_preserves_streaming_capability_and_metadata():
    from ai_character_engine.llm import ModelEndpoint, ModelGatewayClient, StaticModelRouter

    client = GatewayStreamingClient(text="即時。")
    gateway = ModelGatewayClient(
        endpoints=(ModelEndpoint("local", client),),
        router=StaticModelRouter("local"),
    )
    updates = [u async for u in gateway.stream_generate([Message("user", "hi")])]
    assert updates[0].text == "即時。"
    final = updates[-1].response
    assert final is not None
    assert final.metadata["gateway"]["selected_endpoint_id"] == "local"


async def test_gateway_can_fallback_before_first_stream_delta():
    from ai_character_engine.llm import ModelEndpoint, ModelGatewayClient, StaticModelRouter

    bad = GatewayStreamingClient(fail_before=True)
    good = GatewayStreamingClient(text="後備。")
    gateway = ModelGatewayClient(
        endpoints=(ModelEndpoint("bad", bad), ModelEndpoint("good", good)),
        router=StaticModelRouter("bad", "good"),
    )
    updates = [u async for u in gateway.stream_generate([Message("user", "hi")])]
    assert [u.text for u in updates if u.text] == ["後備。"]
    assert updates[-1].response.metadata["gateway"]["fallback_used"] is True


async def test_gateway_suppresses_fallback_after_stream_text_is_visible():
    from ai_character_engine.llm import ModelEndpoint, ModelGatewayClient, StaticModelRouter
    from ai_character_engine.llm.errors import LLMError

    bad = GatewayStreamingClient(fail_after=True, text="已經播出。")
    good = GatewayStreamingClient(text="不應重播。")
    gateway = ModelGatewayClient(
        endpoints=(ModelEndpoint("bad", bad), ModelEndpoint("good", good)),
        router=StaticModelRouter("bad", "good"),
    )
    seen = []
    with pytest.raises(LLMError, match="fallback suppressed"):
        async for update in gateway.stream_generate([Message("user", "hi")]):
            if update.text:
                seen.append(update.text)
    assert seen == ["已經播出。"]
    assert good.calls == 0

async def test_openai_compatible_client_exposes_true_sse_text_deltas():
    from types import SimpleNamespace
    from ai_character_engine.llm import OpenAICompatibleChatClient

    class Stream:
        def __init__(self):
            self.closed = False
            self.items = [
                SimpleNamespace(
                    model="local",
                    usage=None,
                    choices=[SimpleNamespace(index=0, delta=SimpleNamespace(content="你", tool_calls=[]))],
                ),
                SimpleNamespace(
                    model="local",
                    usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2),
                    choices=[SimpleNamespace(index=0, delta=SimpleNamespace(content="好", tool_calls=[]))],
                ),
            ]

        def __aiter__(self):
            self._it = iter(self.items)
            return self

        async def __anext__(self):
            try:
                return next(self._it)
            except StopIteration:
                raise StopAsyncIteration

        async def close(self):
            self.closed = True

    class Completions:
        def __init__(self):
            self.requests = []
            self.stream = Stream()

        async def create(self, **kwargs):
            self.requests.append(kwargs)
            return self.stream

    completions = Completions()
    fake = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    client = OpenAICompatibleChatClient(
        model="m", base_url="http://localhost/v1", client=fake
    )
    updates = [u async for u in client.stream_generate([Message("user", "hi")])]
    assert [u.text for u in updates if u.text] == ["你", "好"]
    assert updates[-1].response.text == "你好"
    assert updates[-1].response.input_tokens == 3
    assert completions.requests[0]["stream"] is True
    assert completions.stream.closed is True

class BlockingTTS:
    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def synthesize(self, text, *, voice=None):
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        return SynthesizedAudio(text.encode(), AudioFormat(), duration_ms=20)


class BurstStreamingLLM:
    def __init__(self, count=5):
        self.count = count
        self.finished = asyncio.Event()

    async def generate(self, messages, *, tools=None):
        raise AssertionError

    async def stream_generate(self, messages, *, tools=None):
        text = ""
        for i in range(self.count):
            part = f"第{i}句。"
            text += part
            yield LLMStreamChunk(text=part)
        self.finished.set()
        yield LLMStreamChunk(final=True, response=LLMResponse(text=text))


async def test_interrupt_during_tts_synthesis_cancels_worker_after_generation_commit():
    tts = BlockingTTS()
    bridge = CharacterHostBridge(runtime(ImmediateStreamingLLM()))
    live = LiveCharacterOrchestrator(
        bridge,
        tts=tts,
        config=streaming_config(),
    )
    live.submit_text("hello")
    task = asyncio.create_task(live.run_once())
    await asyncio.wait_for(tts.started.wait(), 1)
    assert bridge.runtime.history[-1].content == "第一句。第二句。"
    event = await live.interrupt_output(reason="stop_tts")
    assert event is not None and event.type is LiveEventType.TURN_INTERRUPTED
    events = await asyncio.wait_for(task, 1)
    await asyncio.wait_for(tts.cancelled.wait(), 1)
    assert next(e for e in events if e.type is LiveEventType.STREAM_METRICS).data["interrupted"] is True
    # No audio completed: the generated answer must not survive in history.
    assert bridge.runtime.history[-1].content.strip() == "[Interrupted by user]"
    assert live.played_text == ""
    assert len(bridge.runtime.history) == 2
    assert await live.interrupt_output(reason="duplicate_stop") is None
    assert len(bridge.runtime.history) == 2


async def test_bounded_tts_queue_backpressures_llm_stream():
    llm = BurstStreamingLLM(count=6)
    tts = BlockingTTS()
    live = LiveCharacterOrchestrator(
        CharacterHostBridge(runtime(llm)),
        tts=tts,
        config=streaming_config(stream_queue_size=1),
    )
    live.submit_text("hello")
    task = asyncio.create_task(live.run_once())
    await asyncio.wait_for(tts.started.wait(), 1)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert not llm.finished.is_set(), "LLM producer should be backpressured by the bounded TTS queue"
    tts.release.set()
    await asyncio.wait_for(task, 1)
    assert llm.finished.is_set()

async def test_zero_playback_cancel_does_not_duplicate_state_or_memory_commit():
    from ai_character_engine.memory.manager import MemoryManager
    tts = BlockingTTS()
    rt = runtime(ImmediateStreamingLLM(), state_policy=TrustPolicy())
    rt.memory_manager = MemoryManager()
    live = LiveCharacterOrchestrator(CharacterHostBridge(rt), tts=tts, config=streaming_config())
    live.submit_text('我叫測試使用者。')
    task = asyncio.create_task(live.run_once())
    await asyncio.wait_for(tts.started.wait(), 1)
    assert rt.state.trust == 55
    assert len(rt.memory_manager.ledger.list_for_character(rt.memory_scope_id)) == 1
    await live.interrupt_output(reason='before_first_audio')
    await asyncio.wait_for(task, 1)
    await live.interrupt_output(reason='repeated_stop')
    assert rt.state.trust == 55
    assert len(rt.memory_manager.ledger.list_for_character(rt.memory_scope_id)) == 1
    assert len(rt.memory_manager.store.list_for_character(rt.memory_scope_id)) == 1
    assert live.played_text == ''
    assert rt.history[-1].content.strip() == '[Interrupted by user]'
    assert len(rt.history) == 2
