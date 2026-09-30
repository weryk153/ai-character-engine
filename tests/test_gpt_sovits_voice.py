import asyncio
import io
import json
import wave
import pytest
import httpx
from ai_character_engine.voice.gpt_sovits import GPTSoVITSTTS
from ai_character_engine.voice.mac_providers import VoiceHardwareError


def wav_bytes():
    out = io.BytesIO()
    with wave.open(out, 'wb') as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(32000)
        wav.writeframes(b'\x01\x00' * 320)
    return out.getvalue()


def provider(handler, **kwargs):
    return GPTSoVITSTTS(ref_audio_path='/reference.wav', prompt_text='reference words',
                       transport=httpx.MockTransport(handler), **kwargs)


@pytest.mark.asyncio
async def test_post_types_language_and_pcm():
    def handler(request):
        assert request.method == 'POST' and request.url.path == '/tts'
        payload = json.loads(request.content)
        assert payload['streaming_mode'] is False and payload['batch_size'] == 1
        assert payload['text_lang'] == 'zh' and payload['prompt_lang'] == 'ja'
        assert payload['prompt_text'] == 'reference words'
        assert payload['text'] == '你好' and 'Meijia' not in request.content.decode()
        return httpx.Response(200, content=wav_bytes())
    audio = await provider(handler).synthesize('你好', voice='Meijia')
    assert audio.format.sample_rate_hz == 32000 and len(audio.data) == 640
    assert audio.metadata['provider'] == 'gpt_sovits'


@pytest.mark.asyncio
@pytest.mark.parametrize('status,body,limit,match', [
    (400,b'secret server error',1000,'HTTP 400'),
    (200,b'not audio',1000,'invalid WAV'),
    (200,wav_bytes(),10,'byte limit'),
])
async def test_bad_response_is_actionable(status,body,limit,match):
    with pytest.raises(VoiceHardwareError, match=match) as exc:
        await provider(lambda _: httpx.Response(status,content=body), max_audio_bytes=limit).synthesize('hi')
    assert 'secret server error' not in str(exc.value)


@pytest.mark.asyncio
async def test_network_timeout_and_cancel():
    async def slow(request):
        await asyncio.sleep(10)
    with pytest.raises(VoiceHardwareError, match='timed out'):
        await provider(slow, timeout=.01).synthesize('hello')
    task = asyncio.create_task(provider(slow).synthesize('hello'))
    await asyncio.sleep(.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_openapi_check():
    await provider(lambda _: httpx.Response(200,json={'paths':{'/tts':{'post':{}}}})).check()
    with pytest.raises(VoiceHardwareError, match='advertise'):
        await provider(lambda _: httpx.Response(200,json={'paths':{}})).check()


@pytest.mark.parametrize('data', [[], {}, {'ref_audio_path': 1, 'prompt_text':'x'},
    {'ref_audio_path':'x', 'prompt_text':'x', 'unexpected':'y'}])
def test_bad_config(tmp_path,data):
    path=tmp_path/'config.json';path.write_text(json.dumps(data))
    with pytest.raises(VoiceHardwareError):
        GPTSoVITSTTS.from_config(path)
