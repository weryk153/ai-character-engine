"""PortAudio adapters; audio callbacks never wait on an asyncio consumer."""
from __future__ import annotations
import asyncio
import queue
import threading
import time
from .models import AudioChunk, AudioFormat
from .mac_providers import VoiceHardwareError


def sounddevice():
    try:
        import sounddevice as sd
        return sd
    except (ImportError, OSError) as exc:
        raise VoiceHardwareError('Install .[voice-mac]; if PortAudio is missing, brew install portaudio') from exc


def device_report(input_device=None, output_device=None, *, check_input=True,
                  output_sample_rate_hz=22_050, check_output=True):
    """Return device capabilities for the formats the selected providers will use.

    ``output_sample_rate_hz=None`` deliberately skips the output-format probe.
    This is useful for providers such as GPT-SoVITS whose WAV sample rate is
    discovered from the actual synthesis response instead of being fixed by
    the engine. ``SoundDeviceOutput.play()`` still opens PortAudio with that
    real returned format, so an unsupported format fails at playback rather
    than being falsely preflighted as 22.05 kHz.
    """
    sd = sounddevice()
    try:
        devices = [dict(index=i, name=d['name'], inputs=d['max_input_channels'],
                        outputs=d['max_output_channels'], default_rate=d['default_samplerate'])
                   for i, d in enumerate(sd.query_devices())]
        if check_input:
            sd.check_input_settings(device=input_device, channels=1, dtype='int16', samplerate=16000)
        output_checked = bool(check_output and output_sample_rate_hz is not None)
        if output_checked:
            if int(output_sample_rate_hz) <= 0:
                raise ValueError('output_sample_rate_hz must be positive')
            sd.check_output_settings(device=output_device, channels=1, dtype='int16',
                                     samplerate=int(output_sample_rate_hz))
        return dict(devices=devices, default_devices=list(sd.default.device),
                    input_16khz_mono=True if check_input else None,
                    output_pcm16_mono=True if output_checked else None,
                    output_sample_rate_hz=int(output_sample_rate_hz) if output_checked else None)
    except Exception as exc:
        raise VoiceHardwareError('Audio device/format unavailable. Use --list-devices and choose --input-device / --output-device.') from exc


class SoundDeviceInput:
    """20 ms PCM16 chunks with a bounded queue; overflow aborts this capture."""
    def __init__(self, device=None, *, queue_chunks=100):
        self.device, self.queue_chunks = device, queue_chunks

    async def chunks(self):
        sd = sounddevice()
        pending = queue.Queue(maxsize=self.queue_chunks)
        failed = threading.Event()
        sequence = 0
        def callback(data, frames, timing, status):
            if status:
                failed.set()
                return
            # Convert ADC time to perf_counter time at the END of this buffer.
            end = time.perf_counter() + float(timing.inputBufferAdcTime - timing.currentTime) + frames / 16000
            try:
                pending.put_nowait((bytes(data), end))
            except queue.Full:
                failed.set()
        try:
            with sd.RawInputStream(device=self.device, samplerate=16000, channels=1,
                                   dtype='int16', blocksize=320, callback=callback):
                while True:
                    if failed.is_set():
                        raise VoiceHardwareError('Microphone overflow/status error; close busy audio apps and retry')
                    try:
                        pcm, end = pending.get_nowait()
                    except queue.Empty:
                        await asyncio.sleep(0.01)
                        continue
                    yield AudioChunk(pcm, sequence=sequence, timestamp_ms=end * 1000)
                    sequence += 1
        except sd.PortAudioError as exc:
            raise VoiceHardwareError('Cannot open microphone. Enable Microphone permission for your terminal in System Settings, then restart it.') from exc


class SoundDeviceOutput:
    """PCM sink with first-buffer DAC scheduling timestamp and immediate abort.

    Timestamp is a PortAudio estimate, not an acoustic loopback measurement.
    """
    def __init__(self, device=None):
        self.device = device
        self.stream = None
        self.first_audio_at = None

    async def play(self, chunk):
        sd = sounddevice()
        fmt, pcm = chunk.format, chunk.data
        if fmt.encoding != 'pcm_s16le' or fmt.sample_width_bytes != 2 or not pcm or len(pcm) % (fmt.channels * 2):
            raise VoiceHardwareError('Speaker requires nonempty, aligned PCM16 audio')
        done, failed = threading.Event(), threading.Event()
        offset = 0
        self.first_audio_at = None
        def callback(outdata, frames, timing, status):
            nonlocal offset
            if status:
                failed.set()
            count = min(len(outdata), len(pcm) - offset)
            outdata[:] = b'\0' * len(outdata)
            if count:
                outdata[:count] = pcm[offset:offset + count]
                if self.first_audio_at is None:
                    self.first_audio_at = time.perf_counter() + max(0.0, float(timing.outputBufferDacTime - timing.currentTime))
                offset += count
            if offset >= len(pcm):
                raise sd.CallbackStop
        try:
            self.stream = sd.RawOutputStream(device=self.device, samplerate=fmt.sample_rate_hz,
                channels=fmt.channels, dtype='int16', callback=callback, finished_callback=done.set)
            self.stream.start()
            duration = len(pcm) / (fmt.sample_rate_hz * fmt.channels * 2)
            async with asyncio.timeout(duration + 10):
                while not done.is_set():
                    await asyncio.sleep(0.01)
            if failed.is_set():
                raise VoiceHardwareError('Speaker underflow/status error; close busy audio apps and retry')
        except sd.PortAudioError as exc:
            raise VoiceHardwareError('Cannot open speaker; check --output-device and audio settings') from exc
        finally:
            self.cancel()

    async def stop(self):
        """Cooperative async stop hook used by the v0.25 duplex runtime."""
        self.cancel()

    def cancel(self):
        if self.stream is not None:
            stream, self.stream = self.stream, None
            try:
                stream.abort()
            finally:
                stream.close()
