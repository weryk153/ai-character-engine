"""Optional, cancellable Mac adapters for the v0.19 provider protocols."""
from __future__ import annotations
import asyncio
import base64
import io
import json
import platform
import shutil
import sys
import tempfile
import time
import wave
from pathlib import Path
from .models import AudioFormat, SynthesizedAudio, TranscriptResult


class VoiceHardwareError(RuntimeError):
    """An actionable device/provider failure safe to display in the demo."""


async def stop_process(process):
    if process is None or process.returncode is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(process.wait(), 2)
    except TimeoutError:
        process.kill()
        await process.wait()


def decode_wav(data):
    with wave.open(io.BytesIO(data), 'rb') as wav:
        if wav.getsampwidth() != 2 or wav.getcomptype() != 'NONE':
            raise VoiceHardwareError('Expected uncompressed PCM16 WAV')
        fmt = AudioFormat(sample_rate_hz=wav.getframerate(), channels=wav.getnchannels())
        pcm = wav.readframes(wav.getnframes())
    if not pcm:
        raise VoiceHardwareError('Audio provider returned empty audio')
    return pcm, fmt


class FasterWhisperSTT:
    """Persistent isolated CPU worker; timeout/cancel terminates native inference.

    Call start() before recording to exclude model loading from warm STT latency.
    A failed worker is discarded; a subsequent call may start a fresh worker.
    """
    def __init__(self, model='base', *, language='zh', threads=4,
                 local_files_only=False, timeout=60.0, load_timeout=300.0):
        if timeout <= 0 or load_timeout <= 0 or threads < 1:
            raise ValueError('STT timeout, load timeout and threads must be positive')
        self.config = dict(backend='faster-whisper', model=model, language=language, threads=threads,
                           local_files_only=local_files_only)
        self.timeout, self.load_timeout = timeout, load_timeout
        self.process = None
        self.lock = asyncio.Lock()

    async def start(self):
        async with self.lock:
            await self._start()

    async def _read_worker_line(self, timeout, *, failure_message):
        """Read one worker response while also watching for process exit.

        Python 3.13 can leave ``StreamReader.readline()`` waiting until its
        timeout even after a short-lived child has already exited. Racing the
        stdout read against ``process.wait()`` turns that condition into an
        immediate, actionable hardware/provider error instead.
        """
        process = self.process
        if process is None or process.stdout is None:
            raise VoiceHardwareError(failure_message)
        read_task = asyncio.create_task(process.stdout.readline())
        wait_task = asyncio.create_task(process.wait())
        try:
            done, pending = await asyncio.wait(
                {read_task, wait_task}, timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED)
            if not done:
                raise TimeoutError
            if read_task in done:
                raw = read_task.result()
                if raw:
                    return raw
                raise VoiceHardwareError(failure_message)
            # The worker exited before producing a protocol response.  Give an
            # already-buffered stdout line one event-loop turn to win the race.
            await asyncio.sleep(0)
            if read_task.done():
                raw = read_task.result()
                if raw:
                    return raw
            raise VoiceHardwareError(failure_message)
        finally:
            for task in (read_task, wait_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(read_task, wait_task, return_exceptions=True)

    async def _start(self):
        if self.process and self.process.returncode is None:
            # ``returncode`` may lag a very short-lived child on some Python
            # versions; yield once so asyncio can reap it before reusing pipes.
            await asyncio.sleep(0)
            if self.process.returncode is None:
                return
        self.process = await asyncio.create_subprocess_exec(
            sys.executable, '-m', 'ai_character_engine.voice.whisper_worker',
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL, limit=1_000_000)
        try:
            self.process.stdin.write((json.dumps(self.config) + '\n').encode())
            await self.process.stdin.drain()
            result = await self._read_worker_line(
                self.load_timeout,
                failure_message='STT model worker exited during startup; check model path/cache, native dependencies and available memory')
            if not json.loads(result).get('ready'):
                raise VoiceHardwareError('STT model failed to load. Install .[voice-mac], check model path/cache or allow the initial download.')
        except BaseException:
            await self.aclose()
            raise

    async def transcribe(self, audio, *, audio_format):
        if audio_format != AudioFormat() or not audio or len(audio) % 2:
            raise VoiceHardwareError('STT requires nonempty 16 kHz mono PCM16 audio')
        if len(audio) > 16_000 * 2 * 30:
            raise VoiceHardwareError('Utterance exceeds 30 seconds')
        async with self.lock:
            await self._start()
            started = time.perf_counter()
            try:
                if self.process is None or self.process.stdin is None:
                    raise VoiceHardwareError('STT worker is unavailable; restart the transcription request')
                self.process.stdin.write((json.dumps({'pcm': base64.b64encode(audio).decode()}) + '\n').encode())
                await self.process.stdin.drain()
                raw = await self._read_worker_line(
                    self.timeout,
                    failure_message='STT worker exited during inference; check model, native dependencies and available memory')
                result = json.loads(raw)
                if 'error' in result:
                    raise VoiceHardwareError('STT inference failed; check model and available memory')
                return TranscriptResult(result['text'], language=result['language'],
                                        latency_ms=(time.perf_counter() - started) * 1000)
            except BaseException:
                await self.aclose()
                raise

    async def aclose(self):
        process, self.process = self.process, None
        await stop_process(process)


class MacOSSayTTS:
    """macOS say -> temporary WAV -> PCM; never pass model text to a shell."""
    def __init__(self, *, timeout=30.0, sample_rate=22050):
        if timeout <= 0 or sample_rate <= 0:
            raise ValueError('TTS timeout and sample rate must be positive')
        self.timeout, self.sample_rate = timeout, sample_rate

    async def synthesize(self, text, *, voice=None):
        if platform.system() != 'Darwin' or not shutil.which('say'):
            raise VoiceHardwareError('macOS system TTS requires macOS and /usr/bin/say')
        if not text.strip() or len(text) > 8000:
            raise VoiceHardwareError('TTS requires 1–8000 characters')
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix='character-tts-') as directory:
            path = Path(directory) / 'speech.wav'
            args = ['say', '-o', str(path), '--file-format=WAVE',
                    f'--data-format=LEI16@{self.sample_rate}', '-f', '-']
            if voice:
                args += ['-v', voice]
            process = await asyncio.create_subprocess_exec(*args, stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
            try:
                _, error = await asyncio.wait_for(process.communicate(text.encode()), self.timeout)
                if process.returncode:
                    raise VoiceHardwareError('macOS TTS failed; run say -v "?" and select an installed voice')
                pcm, fmt = decode_wav(path.read_bytes())
            finally:
                await stop_process(process)
        return SynthesizedAudio(pcm, fmt,
            duration_ms=len(pcm) / (fmt.sample_rate_hz * fmt.channels * 2) * 1000,
            latency_ms=(time.perf_counter() - started) * 1000,
            metadata={'provider': 'macos_say', 'buffered': True})


class SherpaOnnxSTT(FasterWhisperSTT):
    """Reuse an existing SenseVoice int8 model folder; no model downloads."""
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        folder = Path(model).expanduser()
        if not all((folder / name).is_file() for name in ('model.int8.onnx', 'tokens.txt')):
            raise VoiceHardwareError('SenseVoice folder must contain model.int8.onnx and tokens.txt')
        if self.config['language'] not in (None, 'zh', 'en', 'ja', 'ko', 'yue'):
            raise VoiceHardwareError('SenseVoice supports auto/zh/en/ja/ko/yue')
        self.config.update(backend='sherpa-onnx', model=str(folder))
