from __future__ import annotations
import asyncio
import io
import json
import struct
import sys
import time
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock
import wave
import pytest

from ai_character_engine.llm import ModelEndpoint, ModelGatewayClient, StaticModelRouter
from ai_character_engine.llm.models import LLMResponse, Message
from ai_character_engine.llm.voice_stream import VoiceStreamingChatClient, MeasuredLLM
from ai_character_engine.llm.errors import LLMError
from ai_character_engine.tools import ToolExecutor
from ai_character_engine.tools.models import ToolCall
from ai_character_engine.voice.models import AudioChunk, AudioFormat, SynthesizedAudio, TranscriptResult
from ai_character_engine.voice.live import capture_utterance, make_session, time_tool_registry, LiveVoiceRunner, LatencyBenchmark
from ai_character_engine.voice.mac_providers import FasterWhisperSTT, MacOSSayTTS, VoiceHardwareError, decode_wav, stop_process

SPEECH = struct.pack('<h', 1800) * 320
SILENCE = b'\0' * 640


class FakeSource:
    def __init__(self, frames):
        self.frames = frames
        self.closed = False
    async def chunks(self):
        try:
            for pcm in self.frames:
                yield AudioChunk(pcm, timestamp_ms=time.perf_counter() * 1000)
                await asyncio.sleep(0)
        finally:
            self.closed = True


class FakeSTT:
    async def transcribe(self, audio, *, audio_format):
        assert SPEECH in audio
        return TranscriptResult('現在幾點？', language='zh')


class FakeTTS:
    async def synthesize(self, text, *, voice=None):
        assert 'Asia/Taipei' in text
        return SynthesizedAudio(SPEECH, AudioFormat())


class FakeSink:
    first_audio_at = None
    played = None
    async def play(self, chunk):
        self.first_audio_at = time.perf_counter()
        self.played = chunk
    def cancel(self):
        self.cancelled = True


class ToolLLM:
    async def generate(self, messages, *, tools=None):
        assert tools[0].name == 'get_current_time'
        if messages[-1].role == 'tool':
            result = messages[-1].tool_result
            assert not result.is_error
            data = json.loads(result.output)
            return LLMResponse(text=f"現在時間 {data['iso8601']} {data['timezone']}",
                               metadata={'voice_stream': {'ttft_ms': .1}})
        return LLMResponse(tool_calls=(ToolCall('clock1', 'get_current_time', {}),))


@pytest.mark.asyncio
async def test_fake_audio_full_runtime_gateway_tool_pipeline(tmp_path):
    source = FakeSource([SILENCE] * 4 + [SPEECH] * 8 + [SILENCE] * 30)
    utterance = await capture_utterance(source)
    assert source.closed
    gateway = ModelGatewayClient(endpoints=(ModelEndpoint('fake', ToolLLM()),), router=StaticModelRouter('fake'))
    llm = MeasuredLLM(gateway)
    session, sink = make_session(llm), FakeSink()
    runner = LiveVoiceRunner(session=session, llm=llm, stt=FakeSTT(), tts=FakeTTS(), sink=sink)
    metrics = await runner.run_turn(utterance)
    assert sink.played.data == SPEECH
    assert metrics['llm_rounds'] == 2 and metrics['tool_calls'] == 1
    assert metrics['tool_results_successful'] == 1 and metrics['tool_results_failed'] == 0
    assert metrics['llm_ttft_ms'] is not None
    assert metrics['end_to_end_latency_ms'] >= metrics['tts_ttfa_ms'] >= 0
    assert any(m.role == 'tool' for m in session.runtime.history)
    benchmark = LatencyBenchmark(tmp_path / 'latency.jsonl')
    benchmark.add(metrics)
    saved = (tmp_path / 'latency.jsonl').read_text()
    assert '現在' not in saved and 'pcm' not in saved
    assert benchmark.summary()['turns'] == 1


@pytest.mark.asyncio
async def test_reject_empty_stt_without_llm_or_tts():
    source = FakeSource([SPEECH] * 5 + [SILENCE] * 30)
    utterance = await capture_utterance(source)
    llm = MeasuredLLM(AsyncMock())
    stt = NS(transcribe=AsyncMock(return_value=TranscriptResult('  ')))
    tts = AsyncMock()
    runner = LiveVoiceRunner(session=make_session(llm), llm=llm, stt=stt, tts=tts, sink=FakeSink())
    with pytest.raises(VoiceHardwareError, match='no speech'):
        await runner.run_turn(utterance)
    tts.synthesize.assert_not_called()
    llm.client.generate.assert_not_called()


@pytest.mark.asyncio
async def test_capture_rejects_clicks_and_format_and_closes():
    source = FakeSource([SPEECH] * 2 + [SILENCE] * 30)
    with pytest.raises(VoiceHardwareError, match='complete utterance'):
        await capture_utterance(source)
    assert source.closed
    source = FakeSource([b'odd'])
    with pytest.raises(VoiceHardwareError, match='20 ms'):
        await capture_utterance(source)
    assert source.closed


@pytest.mark.asyncio
async def test_max_utterance_is_bounded():
    source = FakeSource([SPEECH] * 100)
    result = await capture_utterance(source, max_seconds=1)
    assert len(result.audio) == 50 * 640 and source.closed


@pytest.mark.asyncio
async def test_capture_timeout_closes_source():
    class Slow(FakeSource):
        async def chunks(self):
            try:
                await asyncio.sleep(10)
                yield AudioChunk(SILENCE)
            finally:
                self.closed = True
    source = Slow([])
    with pytest.raises(TimeoutError):
        await capture_utterance(source, listen_timeout=.01)
    assert source.closed


@pytest.mark.asyncio
async def test_cancel_hook_and_stt_timeout():
    source = FakeSource([SPEECH] * 5 + [SILENCE] * 30)
    utterance = await capture_utterance(source)
    entered = asyncio.Event()
    async def slow(*args, **kwargs):
        entered.set()
        await asyncio.sleep(10)
    llm, sink = MeasuredLLM(ToolLLM()), FakeSink()
    runner = LiveVoiceRunner(session=make_session(llm), llm=llm, stt=NS(transcribe=slow), tts=FakeTTS(), sink=sink)
    task = asyncio.create_task(runner.run_turn(utterance))
    await entered.wait()
    runner.cancel_current_turn()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert runner._task is None and sink.cancelled
    runner.stt_timeout = .01
    with pytest.raises(TimeoutError):
        await runner.run_turn(utterance)
    assert runner._task is None


@pytest.mark.asyncio
async def test_clock_tool_rejects_extra_arguments():
    executor = ToolExecutor(time_tool_registry())
    result = await executor.execute(ToolCall('x', 'get_current_time', {'command': 'anything'}))
    assert result.is_error
    result = await executor.execute(ToolCall('x', 'get_current_time', {}))
    assert not result.is_error and json.loads(result.output)['timezone'] == 'Asia/Taipei'


def chunk(content=None, calls=None, finish=None):
    return NS(model='fake', usage=None, choices=[NS(index=0, finish_reason=finish,
        delta=NS(content=content, tool_calls=calls))])


class Stream:
    def __init__(self, chunks, slow=False):
        self.chunks, self.slow, self.closed = chunks, slow, False
    def __aiter__(self):
        return self.iterate()
    async def iterate(self):
        for c in self.chunks:
            if self.slow:
                await asyncio.sleep(10)
            yield c
    async def close(self):
        self.closed = True


def client_for(stream, **kwargs):
    create = AsyncMock(return_value=stream)
    return VoiceStreamingChatClient(model='fake', base_url='http://localhost/v1',
        client=NS(chat=NS(completions=NS(create=create))), **kwargs), create


@pytest.mark.asyncio
async def test_real_delta_measurement_and_tool_fragment_assembly():
    stream = Stream([chunk(), chunk(calls=[NS(index=0, id='t1', function=NS(name='get_current_time', arguments='{'))]),
                     chunk(calls=[NS(index=0, id=None, function=NS(name=None, arguments='}'))]), chunk(finish='tool_calls')])
    client, create = client_for(stream)
    result = await client.generate([Message('user', 'time')], tools=time_tool_registry().definitions())
    assert result.tool_calls == (ToolCall('t1', 'get_current_time', {}),)
    assert result.metadata['voice_stream']['ttft_ms'] is None
    assert result.metadata['voice_stream']['first_event_ms'] is not None
    assert stream.closed and create.call_args.kwargs['stream'] is True
    stream2 = Stream([chunk(), chunk('現在'), chunk('三點'), chunk(finish='stop')])
    client, _ = client_for(stream2)
    result = await client.generate([Message('user', 'time')])
    assert result.text == '現在三點' and result.metadata['voice_stream']['ttft_ms'] >= 0


@pytest.mark.asyncio
@pytest.mark.parametrize('chunks', [[chunk('partial')], [chunk('partial', finish='length')],
    [chunk(finish='stop')], [chunk(calls=[NS(index=0, id='x', function=NS(name='get_current_time', arguments='bad'))], finish='tool_calls')]])
async def test_bad_streams_fail_and_close(chunks):
    stream = Stream(chunks)
    client, _ = client_for(stream)
    with pytest.raises(LLMError):
        await client.generate([])
    assert stream.closed


@pytest.mark.asyncio
async def test_stream_timeout_and_cancel_close_network():
    stream = Stream([chunk('hello')], slow=True)
    client, _ = client_for(stream, timeout_seconds=.01)
    with pytest.raises(LLMError):
        await client.generate([])
    assert stream.closed
    stream = Stream([chunk('hello')], slow=True)
    client, _ = client_for(stream)
    task = asyncio.create_task(client.generate([]))
    await asyncio.sleep(.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stream.closed


@pytest.mark.asyncio
async def test_stt_format_validation_no_worker_spawned():
    provider = FasterWhisperSTT()
    for pcm, fmt in [(b'x', AudioFormat()), (SPEECH, AudioFormat(sample_rate_hz=48000)),
                      (b'\0' * 960002, AudioFormat())]:
        with pytest.raises(VoiceHardwareError):
            await provider.transcribe(pcm, audio_format=fmt)
        assert provider.process is None


@pytest.mark.asyncio
async def test_stt_timeout_terminates_persistent_worker():
    provider = FasterWhisperSTT(timeout=.01)
    proc = await asyncio.create_subprocess_exec(sys.executable, '-c', 'import time; time.sleep(30)',
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE)
    provider.process = proc
    with pytest.raises(TimeoutError):
        await provider.transcribe(SPEECH, audio_format=AudioFormat())
    assert proc.returncode is not None and provider.process is None


@pytest.mark.asyncio
async def test_stt_worker_crash_closes():
    provider = FasterWhisperSTT(timeout=1)
    proc = await asyncio.create_subprocess_exec(sys.executable, '-S', '-c', 'pass',
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE)
    provider.process = proc
    started = time.perf_counter()
    with pytest.raises(VoiceHardwareError, match='worker exited'):
        await provider.transcribe(SPEECH, audio_format=AudioFormat())
    assert time.perf_counter() - started < .5
    assert provider.process is None


def test_wav_decode_and_benchmark_null_percentiles():
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(22050); wav.writeframes(SPEECH)
    pcm, fmt = decode_wav(buffer.getvalue())
    assert pcm == SPEECH and fmt.sample_rate_hz == 22050
    b = LatencyBenchmark()
    for value in [10, 40, 30, 20]:
        b.add({'stt_latency_ms': value, 'llm_ttft_ms': None})
    assert b.summary()['stt_latency_ms'] == {'n': 4, 'p50': 20, 'p95': 40}
    assert b.summary()['llm_ttft_ms']['p50'] is None

@pytest.mark.asyncio
async def test_macos_tts_uses_stdin_and_removes_temp_files(monkeypatch):
    from pathlib import Path
    import ai_character_engine.voice.mac_providers as providers
    captured = {}
    class Proc:
        returncode = 0
        async def communicate(self, data):
            captured['stdin'] = data
            with wave.open(str(captured['path']), 'wb') as wav:
                wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(22050); wav.writeframes(SPEECH)
            return b'', b''
    async def spawn(*args, **kwargs):
        captured['args'] = args
        captured['path'] = Path(args[args.index('-o') + 1])
        return Proc()
    monkeypatch.setattr(providers.platform, 'system', lambda: 'Darwin')
    monkeypatch.setattr(providers.shutil, 'which', lambda _: '/usr/bin/say')
    monkeypatch.setattr(providers.asyncio, 'create_subprocess_exec', spawn)
    text = '-v bad; $(do-not-execute) `literal`'
    result = await MacOSSayTTS().synthesize(text, voice='Meijia')
    assert text not in captured['args'] and captured['stdin'] == text.encode()
    assert result.data == SPEECH and not captured['path'].exists()


@pytest.mark.asyncio
async def test_macos_tts_timeout_kills_process(monkeypatch):
    import ai_character_engine.voice.mac_providers as providers
    class Proc:
        returncode = None
        killed = False
        async def communicate(self, data):
            await asyncio.sleep(10)
        def terminate(self):
            self.killed = True
            self.returncode = -15
        async def wait(self):
            return self.returncode
    proc = Proc()
    monkeypatch.setattr(providers.platform, 'system', lambda: 'Darwin')
    monkeypatch.setattr(providers.shutil, 'which', lambda _: '/usr/bin/say')
    monkeypatch.setattr(providers.asyncio, 'create_subprocess_exec', AsyncMock(return_value=proc))
    with pytest.raises(TimeoutError):
        await MacOSSayTTS(timeout=.01).synthesize('hello')
    assert proc.killed


@pytest.mark.asyncio
async def test_sound_output_callbacks_and_cancel(monkeypatch):
    from ai_character_engine.voice import devices
    class Stop(Exception): pass
    class Stream:
        aborted = closed = False
        def __init__(self, **kwargs):
            self.kwargs = kwargs
        def start(self):
            buffer = bytearray(1280)
            try:
                self.kwargs['callback'](buffer, 640, NS(currentTime=1, outputBufferDacTime=1.01), False)
            except Stop:
                pass
            assert buffer[:640] == SPEECH
            assert buffer[640:] == SILENCE
            self.kwargs['finished_callback']()
        def abort(self): self.aborted = True
        def close(self): self.closed = True
    streams = []
    def create(**kwargs):
        streams.append(Stream(**kwargs)); return streams[-1]
    monkeypatch.setattr(devices, 'sounddevice', lambda: NS(RawOutputStream=create, CallbackStop=Stop, PortAudioError=OSError))
    sink = devices.SoundDeviceOutput()
    await sink.play(AudioChunk(SPEECH))
    assert sink.first_audio_at is not None and streams[0].aborted and streams[0].closed
    assert sink.stream is None


@pytest.mark.asyncio
async def test_sound_input_overflow_closes_device(monkeypatch):
    from ai_character_engine.voice import devices
    state = {}
    class Stream:
        def __init__(self, **kwargs): self.callback = kwargs['callback']
        def __enter__(self):
            timing = NS(inputBufferAdcTime=1, currentTime=1)
            for _ in range(3): self.callback(SPEECH, 320, timing, False)
            return self
        def __exit__(self, *args): state['closed'] = True
    monkeypatch.setattr(devices, 'sounddevice', lambda: NS(RawInputStream=Stream, PortAudioError=OSError))
    with pytest.raises(VoiceHardwareError, match='overflow'):
        async for _ in devices.SoundDeviceInput(queue_chunks=1).chunks(): pass
    assert state['closed']


def test_sherpa_requires_existing_model(tmp_path):
    from ai_character_engine.voice.mac_providers import SherpaOnnxSTT
    with pytest.raises(VoiceHardwareError, match='model.int8.onnx'):
        SherpaOnnxSTT(str(tmp_path))


def test_faster_whisper_worker_consumes_lazy_segments(monkeypatch):
    from ai_character_engine.voice import whisper_worker as worker
    state = {}
    class Array:
        def astype(self, dtype): return self
        def __truediv__(self, divisor): return self
    class Model:
        def __init__(self, name, **kwargs):
            assert kwargs['device'] == 'cpu' and kwargs['compute_type'] == 'int8'
        def transcribe(self, audio, **kwargs):
            def segments():
                state['consumed'] = True
                yield NS(text='你好')
            return segments(), NS(language='zh')
    monkeypatch.setitem(sys.modules, 'numpy', NS(frombuffer=lambda *a, **k: Array(), float32='float32'))
    monkeypatch.setitem(sys.modules, 'faster_whisper', NS(WhisperModel=Model))
    request = json.dumps(dict(backend='faster-whisper', model='fake', threads=1, language='zh', local_files_only=True))
    stdin, stdout = io.StringIO(request + '\n' + '{"pcm":"AAA="}\n'), io.StringIO()
    monkeypatch.setattr(sys, 'stdin', stdin)
    monkeypatch.setattr(sys, 'stdout', stdout)
    assert worker.main() == 0
    assert state['consumed']
    rows = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert rows == [{'ready': True}, {'text': '你好', 'language': 'zh'}]

@pytest.mark.asyncio
async def test_clock_question_requires_real_tool_and_does_not_force_followup():
    stream = Stream([chunk('現在三點。', finish='stop')])
    client, create = client_for(stream, require_time_tool=True)
    with pytest.raises(LLMError, match='ignored required'):
        await client.generate([Message('user', '现在几点？')], tools=time_tool_registry().definitions())
    assert create.call_args.kwargs['tool_choice'] == 'required'
    from ai_character_engine.tools.models import ToolResult
    followup = Stream([chunk('現在三點。', finish='stop')])
    client, create = client_for(followup, require_time_tool=True)
    await client.generate([Message('tool', '', tool_result=ToolResult('t', 'get_current_time', '{}'))],
                          tools=time_tool_registry().definitions())
    assert 'tool_choice' not in create.call_args.kwargs

@pytest.mark.asyncio
async def test_text_only_bypasses_audio_and_preserves_agent():
    llm = MeasuredLLM(ToolLLM())
    stt = NS(transcribe=AsyncMock(side_effect=AssertionError('must not record/transcribe')))
    runner = LiveVoiceRunner(session=make_session(llm), llm=llm, stt=stt, tts=None, sink=None)
    result = await runner.run_text('現在幾點？')
    stt.transcribe.assert_not_called()
    assert result['input_source'] == 'text'
    assert result['stt_latency_ms'] is None and result['tts_ttfa_ms'] is None
    assert result['endpointing_ms'] is None and result['end_to_end_latency_ms'] is None
    assert result['response_ready_ms'] >= 0 and result['tool_results_successful'] == 1


@pytest.mark.asyncio
async def test_text_input_can_keep_spoken_reply():
    llm, sink = MeasuredLLM(ToolLLM()), FakeSink()
    runner = LiveVoiceRunner(session=make_session(llm), llm=llm, stt=None, tts=FakeTTS(), sink=sink)
    result = await runner.run_text('現在幾點？')
    assert result['stt_latency_ms'] is None and sink.played is not None
    assert result['tts_ttfa_ms'] >= 0 and result['end_to_end_latency_ms'] >= result['tts_ttfa_ms']


@pytest.mark.asyncio
async def test_async_console_eof_and_cancel_remove_reader(monkeypatch):
    from io import StringIO
    from ai_character_engine.voice import live

    # The console adapter explicitly targets macOS/Unix. Exercise its readiness
    # protocol on every OS without requiring a Windows loop to expose add_reader.
    class Console(StringIO):
        def fileno(self):
            return 42

    loop = asyncio.get_running_loop()
    readers = {}
    removed = []

    def remove_reader(fd):
        removed.append(fd)
        return readers.pop(fd, None) is not None

    reader_loop = NS(create_future=loop.create_future,
                     add_reader=lambda fd, callback: readers.__setitem__(fd, callback),
                     remove_reader=remove_reader)
    monkeypatch.setattr(live, 'asyncio', NS(get_running_loop=lambda: reader_loop))
    monkeypatch.setattr(sys, 'stdin', Console('hello\n'))

    task = asyncio.create_task(live.read_console_line(''))
    await asyncio.sleep(0)
    readers[42]()
    assert await task == 'hello'
    assert not readers

    task = asyncio.create_task(live.read_console_line(''))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not readers

    task = asyncio.create_task(live.read_console_line(''))
    await asyncio.sleep(0)
    readers[42]()
    assert await task is None
    assert not readers
    assert removed == [42, 42, 42]
