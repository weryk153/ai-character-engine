"""GPT-SoVITS api_v2 adapter; buffered PCM output with cancellable HTTP I/O.

Cancelling closes this client's request. The independent server may finish its
ongoing inference; this adapter never stops a shared server or changes weights.
"""
from __future__ import annotations
import asyncio
import json
import math
import time
import wave
from pathlib import Path
from urllib.parse import urlsplit
try:
    import httpx
except ImportError:
    httpx = None
from .mac_providers import VoiceHardwareError, decode_wav
from .models import SynthesizedAudio


class GPTSoVITSTTS:
    def __init__(self, *, ref_audio_path, prompt_text, prompt_lang='ja', text_lang='zh',
                 api_url='http://127.0.0.1:9880/tts', timeout=120.0,
                 max_audio_bytes=20 * 1024 * 1024, transport=None):
        if httpx is None:
            raise VoiceHardwareError('GPT-SoVITS adapter needs httpx; install .[voice-audio]')
        url = urlsplit(api_url)
        if url.scheme not in ('http', 'https') or not url.hostname or url.username or url.password:
            raise ValueError('GPT-SoVITS requires an HTTP(S) URL without credentials')
        if not ref_audio_path or not prompt_text.strip():
            raise ValueError('GPT-SoVITS requires a reference audio path and its exact transcript')
        if not math.isfinite(timeout) or timeout <= 0 or max_audio_bytes < 1:
            raise ValueError('GPT-SoVITS timeout and audio limit must be positive')
        self.api_url, self.timeout, self.max_audio_bytes = api_url, timeout, max_audio_bytes
        self.transport = transport
        self.reference = dict(ref_audio_path=str(ref_audio_path), prompt_text=prompt_text,
                              prompt_lang=prompt_lang, text_lang=text_lang)

    @classmethod
    def from_config(cls, path, *, timeout=120.0):
        path = Path(path).expanduser()
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            raise VoiceHardwareError('Cannot read GPT-SoVITS JSON config; set --sovits-config PATH') from exc
        if not isinstance(data, dict):
            raise VoiceHardwareError('GPT-SoVITS config must be a JSON object')
        allowed = {'api_url', 'ref_audio_path', 'prompt_text', 'prompt_lang', 'text_lang'}
        if set(data) - allowed:
            raise VoiceHardwareError('Unknown GPT-SoVITS config fields')
        if not isinstance(data.get('ref_audio_path'), str) or not isinstance(data.get('prompt_text'), str):
            raise VoiceHardwareError('GPT-SoVITS config requires string ref_audio_path and prompt_text')
        if any(not isinstance(value, str) for value in data.values()):
            raise VoiceHardwareError('GPT-SoVITS config values must be strings')
        data['ref_audio_path'] = str(Path(data['ref_audio_path']).expanduser())
        return cls(**data, timeout=timeout)

    async def check(self):
        """Verify the server advertises api_v2 POST /tts; does not synthesize."""
        base = self.api_url.rsplit('/tts', 1)[0]
        try:
            async with httpx.AsyncClient(timeout=10, transport=self.transport) as client:
                response = await client.get(base + '/openapi.json')
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict) or 'post' not in data.get('paths', {}).get('/tts', {}):
                    raise VoiceHardwareError('Server does not advertise GPT-SoVITS POST /tts')
        except (httpx.HTTPError, ValueError) as exc:
            raise VoiceHardwareError('GPT-SoVITS unavailable; start its api_v2 server and check --sovits-config') from exc

    async def synthesize(self, text, *, voice=None):
        # Reference audio defines the voice. macOS --voice does not apply here.
        if not text.strip() or len(text) > 8000:
            raise VoiceHardwareError('TTS requires 1–8000 characters')
        payload = dict(self.reference, text=text, text_split_method='cut5', batch_size=1,
                       media_type='wav', streaming_mode=False)
        started = time.perf_counter()
        try:
            async with asyncio.timeout(self.timeout):
                async with httpx.AsyncClient(timeout=self.timeout, transport=self.transport) as client:
                    async with client.stream('POST', self.api_url, json=payload) as response:
                        if response.status_code != 200:
                            raise VoiceHardwareError(f'GPT-SoVITS returned HTTP {response.status_code}; check server log, model and reference transcript')
                        data = bytearray()
                        async for chunk in response.aiter_bytes():
                            if len(data) + len(chunk) > self.max_audio_bytes:
                                raise VoiceHardwareError('GPT-SoVITS audio exceeds configured byte limit')
                            data.extend(chunk)
            pcm, fmt = decode_wav(bytes(data))
        except (httpx.TimeoutException, TimeoutError) as exc:
            raise VoiceHardwareError('GPT-SoVITS synthesis timed out; server may still be computing. Wait before retrying or increase --tts-timeout.') from exc
        except httpx.HTTPError as exc:
            raise VoiceHardwareError('Cannot reach GPT-SoVITS; check local server and its logs') from exc
        except (wave.Error, EOFError) as exc:
            raise VoiceHardwareError('GPT-SoVITS returned invalid WAV audio') from exc
        return SynthesizedAudio(pcm, fmt,
            duration_ms=len(pcm) * 1000 / (fmt.sample_rate_hz * fmt.channels * 2),
            latency_ms=(time.perf_counter() - started) * 1000,
            metadata={'provider': 'gpt_sovits', 'buffered': True})
