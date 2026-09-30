"""Half-duplex hardware orchestration with replaceable providers and benchmarks."""
from __future__ import annotations
import asyncio
from contextlib import aclosing
from dataclasses import dataclass
from datetime import datetime
import json
import math
import time
import sys
from pathlib import Path
from zoneinfo import ZoneInfo
from ai_character_engine.character.profile import CharacterProfile
from ai_character_engine.events.models import CharacterEvent
from ai_character_engine.observability import TraceContext, Tracer
from ai_character_engine.session import CharacterRuntimeFactory, SessionManager
from ai_character_engine.session.store import InMemorySessionStore
from ai_character_engine.tools.models import ToolDefinition
from ai_character_engine.tools.registry import ToolRegistry
from .models import AudioChunk, AudioFormat
from .vad import EnergyVoiceActivityDetector, UtteranceSegmenter, UtteranceSegmenterConfig
from .mac_providers import VoiceHardwareError


def time_tool_registry(timezone='Asia/Taipei'):
    zone = ZoneInfo(timezone)  # host-chosen zone; the model supplies no arguments
    registry = ToolRegistry()
    registry.register(ToolDefinition('get_current_time',
        'Read the current date and time. Call this whenever the user asks what time it is.',
        {'type': 'object', 'properties': {}, 'required': [], 'additionalProperties': False}),
        lambda: {'iso8601': datetime.now(zone).isoformat(timespec='seconds'), 'timezone': timezone})
    return registry


def make_session(llm, *, timezone='Asia/Taipei', tracer=None):
    character = CharacterProfile(id='live-character', name='燈',
        description='親切、直接的語音角色。以繁體中文自然回答，每次一至兩句。詢問現在時間必須呼叫 get_current_time，再依工具結果回答。',
        personality=['坦率', '體貼'], speaking_style=['簡短自然口語', '不使用 Markdown'],
        rules=['When asked for the current time, call get_current_time immediately. Do not say you will check later. Never guess the time.',
               'After receiving the tool result, answer the user directly in Traditional Chinese.'])
    manager = SessionManager(InMemorySessionStore())
    return CharacterRuntimeFactory(characters={character.id: character},
        llm_factory=lambda _: llm, session_manager=manager,
        tool_registry_factory=lambda: time_tool_registry(timezone), max_tool_rounds=4,
        tracer=tracer).create(user_id='local-user', character_id=character.id)


@dataclass(frozen=True)
class CapturedUtterance:
    audio: bytes
    format: AudioFormat
    speech_end_at: float
    endpoint_at: float


async def capture_utterance(source, *, threshold=500, end_silence_ms=600,
                            max_seconds=20, listen_timeout=60):
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError('VAD threshold must be finite and > 0')
    if not 20 <= end_silence_ms <= 5000 or not 1 <= max_seconds <= 30 or listen_timeout <= 0:
        raise ValueError('Invalid endpoint, utterance or listen timeout settings')
    detector = EnergyVoiceActivityDetector(rms_threshold=threshold)
    segmenter = UtteranceSegmenter(detector, config=UtteranceSegmenterConfig(
        min_speech_chunks=5, end_silence_chunks=math.ceil(end_silence_ms / 20),
        max_chunks=int(max_seconds * 50)))
    last_speech_at = None
    # Always close the source BEFORE STT/LLM/TTS, including early return/error.
    async with aclosing(source.chunks()) as chunks:
        async with asyncio.timeout(listen_timeout):
            async for chunk in chunks:
                if chunk.format != AudioFormat() or len(chunk.data) != 640:
                    raise VoiceHardwareError('Capture source must provide 20 ms / 16 kHz mono PCM16 chunks')
                if detector.is_speech(chunk):
                    last_speech_at = chunk.timestamp_ms / 1000 if chunk.timestamp_ms is not None else time.perf_counter()
                utterance = segmenter.push(chunk)
                if utterance:
                    return CapturedUtterance(b''.join(c.data for c in utterance), chunk.format,
                                             last_speech_at, time.perf_counter())
    raise VoiceHardwareError('Audio source ended before a complete utterance')


class LiveVoiceRunner:
    def __init__(self, *, session, llm, stt, tts, sink, voice=None,
                 stt_timeout=60, character_timeout=180, tts_timeout=30, tracer=None):
        self.session, self.llm, self.stt, self.tts, self.sink = session, llm, stt, tts, sink
        self.voice = voice
        self.stt_timeout, self.character_timeout, self.tts_timeout = stt_timeout, character_timeout, tts_timeout
        self.tracer = tracer or Tracer()
        self._task = None

    def cancel_current_turn(self):
        """Host hook. It cancels STT/LLM/TTS/playback; it is NOT automatic barge-in."""
        if self._task:
            self._task.cancel()
        if hasattr(self.sink, 'cancel'):
            self.sink.cancel()

    async def run_text(self, text, *, on_text=None):
        """Keyboard input bypasses capture/VAD/STT; missing metrics remain null."""
        if not text.strip():
            raise ValueError('Text must not be empty')
        return await self.run_turn(None, on_text=on_text, text=text)

    async def run_turn(self, utterance, *, on_text=None, text=None):
        if self._task is not None:
            raise RuntimeError('LiveVoiceRunner permits one turn at a time')
        self._task = asyncio.current_task()
        context = TraceContext.create().child(session_id=self.session.record.id,
            user_id=self.session.record.user_id, character_id=self.session.record.character_id)
        self.llm.reset()
        try:
            with self.tracer.span('voice.live_turn', context=context) as root:
                child = context.child(parent_span_id=root.span_id)
                started = time.perf_counter()
                submitted_at = utterance.speech_end_at if utterance is not None else started
                stt_ms, speech_end_to_stt_ms = None, None
                if text is None:
                    if utterance is None:
                        raise ValueError('Supply audio or text')
                    with self.tracer.span('voice.stt', context=child):
                        async with asyncio.timeout(self.stt_timeout):
                            transcript = await self.stt.transcribe(utterance.audio, audio_format=utterance.format)
                    stt_completed_at = time.perf_counter()
                    stt_ms = (stt_completed_at - started) * 1000
                    speech_end_to_stt_ms = (stt_completed_at - submitted_at) * 1000
                    text = transcript.text
                if not text.strip():
                    raise VoiceHardwareError('STT heard no speech; speak closer or adjust --vad-threshold')
                if on_text:
                    on_text('You', text)
                character_started = time.perf_counter()
                with self.tracer.span('voice.character', context=child) as span:
                    async with asyncio.timeout(self.character_timeout):
                        result = await self.session.process_event(CharacterEvent.user_message(text),
                            trace_context=child.child(parent_span_id=span.span_id))
                character_ms = (time.perf_counter() - character_started) * 1000
                if not result.response.text.strip():
                    raise VoiceHardwareError('Character returned empty text; check model tool/reasoning support')
                if on_text:
                    on_text('Character', result.response.text)
                tts_started = time.perf_counter()
                tts_ms, first_audio = None, None
                if self.tts is not None and self.sink is not None:
                    with self.tracer.span('voice.tts', context=child):
                        async with asyncio.timeout(self.tts_timeout):
                            audio = await self.tts.synthesize(result.response.text, voice=self.voice)
                    tts_ms = (time.perf_counter() - tts_started) * 1000
                    with self.tracer.span('voice.playback', context=child):
                        await self.sink.play(AudioChunk(audio.data, audio.format))
                    first_audio = getattr(self.sink, 'first_audio_at', None)
                metrics = dict(trace_id=context.trace_id,
                    stt_latency_ms=stt_ms, speech_end_to_stt_ms=speech_end_to_stt_ms, **self.llm.metrics(), character_latency_ms=character_ms,
                    tts_audio_ready_ms=tts_ms,
                    tts_provider_ttfa_ms=tts_ms,
                    first_playback_ms=(first_audio - submitted_at) * 1000 if first_audio is not None else None,
                    output_mode="buffered_half_duplex",
                    streaming_tts=False,
                    tts_ttfa_ms=(first_audio - tts_started) * 1000 if first_audio is not None else None,
                    end_to_end_latency_ms=(first_audio - submitted_at) * 1000 if first_audio is not None else None,
                    endpointing_ms=(utterance.endpoint_at - utterance.speech_end_at) * 1000 if utterance is not None else None,
                    turn_complete_ms=(time.perf_counter() - submitted_at) * 1000,
                    input_source='microphone' if utterance is not None else 'text',
                    response_ready_ms=(tts_started - submitted_at) * 1000,
                    tool_calls=sum(c['tool_calls'] for c in self.llm.calls),
                    tool_results_successful=sum(not t.is_error for t in result.tool_results),
                    tool_results_failed=sum(t.is_error for t in result.tool_results),
                    audio_timing='portaudio_first_buffer_dac_estimate' if first_audio is not None else None,
                    full_barge_in=False)
                return metrics
        finally:
            self._task = None


METRICS = ('stt_latency_ms', 'llm_ttft_ms', 'llm_total_latency_ms', 'tts_ttfa_ms', 'end_to_end_latency_ms')


class LatencyBenchmark:
    """Timing-only JSONL; no text, raw audio, API keys or provider error bodies."""
    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.rows = []

    def add(self, metrics):
        self.rows.append(metrics)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open('a', encoding='utf-8') as file:
                file.write(json.dumps(metrics, ensure_ascii=False, allow_nan=False) + '\n')

    def summary(self):
        output = {'turns': len(self.rows)}
        for key in METRICS:
            values = sorted(row[key] for row in self.rows if row.get(key) is not None)
            if not values:
                output[key] = {'n': 0, 'p50': None, 'p95': None}
            else:
                output[key] = {'n': len(values), 'p50': values[math.ceil(len(values)*.5)-1],
                               'p95': values[math.ceil(len(values)*.95)-1]}
        return output


async def read_console_line(prompt='You> '):
    """macOS/Unix console input without an uncancellable blocking executor thread."""
    print(prompt, end='', flush=True)
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    fd = sys.stdin.fileno()
    def ready():
        if not future.done():
            future.set_result(sys.stdin.readline())
    loop.add_reader(fd, ready)
    try:
        line = await future
        return None if line == '' else line.rstrip('\r\n')
    finally:
        loop.remove_reader(fd)
