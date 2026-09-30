#!/usr/bin/env python3
"""Mac live voice demo. Install editable package first; --help needs no devices."""
from __future__ import annotations
import argparse
import asyncio
import importlib.util
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

from ai_character_engine.llm import ModelEndpoint, ModelGatewayClient, StaticModelRouter
from ai_character_engine.llm.errors import LLMError
from ai_character_engine.llm.voice_stream import MeasuredLLM, VoiceStreamingChatClient
from ai_character_engine.voice.devices import SoundDeviceInput, SoundDeviceOutput, device_report, sounddevice
from ai_character_engine.voice.live import LiveVoiceRunner, LatencyBenchmark, capture_utterance, make_session, read_console_line
from ai_character_engine.voice.mac_providers import FasterWhisperSTT, SherpaOnnxSTT, MacOSSayTTS, VoiceHardwareError
from ai_character_engine.voice.models import AudioChunk
from ai_character_engine.voice.gpt_sovits import GPTSoVITSTTS


def positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError('must be a finite positive number')
    return number


def device(value):
    return int(value) if value.isdecimal() else value


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--text', action='store_true', help='type messages instead of using microphone/STT; /quit or Ctrl-D exits')
    p.add_argument('--no-audio', action='store_true', help='with --text, text-only replies; no mic/speaker/STT/TTS dependencies')
    p.add_argument('--backend', choices=['lmstudio', 'ollama', 'cloud'], default=os.getenv('VOICE_BACKEND', 'lmstudio'))
    p.add_argument('--base-url', default=os.getenv('VOICE_BASE_URL'))
    p.add_argument('--model', default=os.getenv('VOICE_MODEL'))
    p.add_argument('--api-key-env', default='VOICE_API_KEY', help='environment variable name, never the key itself')
    p.add_argument('--reasoning', choices=['auto', 'none', 'server'], default='auto', help='auto disables reasoning for Qwen3.5 on LM Studio')
    p.add_argument('--max-tokens', type=int, default=512)
    p.add_argument('--stt-backend', choices=['faster-whisper', 'sherpa-onnx'], default=os.getenv('VOICE_STT_BACKEND', 'faster-whisper'))
    p.add_argument('--stt-model', default=os.getenv('VOICE_STT_MODEL', 'base'), help='faster-whisper size, repository or local directory')
    p.add_argument('--language', default='zh', help='Whisper language code or auto')
    p.add_argument('--local-files-only', action='store_true', help='disable STT model downloads')
    p.add_argument('--tts-provider', choices=['macos', 'gpt-sovits'], default=os.getenv('VOICE_TTS_PROVIDER', 'macos'))
    p.add_argument('--sovits-config', type=Path, default=Path(os.getenv('VOICE_SOVITS_CONFIG', 'config/gpt_sovits.local.json')))
    p.add_argument('--voice', default=os.getenv('VOICE_TTS_VOICE', 'Meijia'))
    p.add_argument('--output-sample-rate', type=int, default=(int(os.getenv('VOICE_OUTPUT_SAMPLE_RATE')) if os.getenv('VOICE_OUTPUT_SAMPLE_RATE') else None),
                   help='speaker preflight/TTS sample-rate hint; macOS say defaults to 22050, dynamic providers may omit')
    p.add_argument('--timezone', default='Asia/Taipei')
    p.add_argument('--input-device', type=device)
    p.add_argument('--output-device', type=device)
    p.add_argument('--vad-threshold', type=positive, default=500)
    p.add_argument('--end-silence-ms', type=int, default=600)
    p.add_argument('--max-utterance-seconds', type=positive, default=20)
    p.add_argument('--listen-timeout', type=positive, default=60)
    p.add_argument('--stt-timeout', type=positive, default=60)
    p.add_argument('--llm-timeout', type=positive, default=120)
    p.add_argument('--tts-timeout', type=positive, default=30)
    p.add_argument('--turns', type=int, default=0, help='0 runs until Ctrl-C')
    p.add_argument('--benchmark', type=Path, help='append timing-only JSONL; transcripts are console-only')
    p.add_argument('--list-devices', action='store_true')
    p.add_argument('--list-voices', action='store_true')
    p.add_argument('--test-agent', action='store_true', help='text-only current-time tool test, no mic/STT/TTS needed')
    p.add_argument('--check', action='store_true', help='check dependencies, formats, voice and server model; does not record/play audio or load STT')
    p.add_argument('--test-speaker', action='store_true', help='say a short test phrase on selected output')
    return p


def config(args):
    if args.no_audio and (not args.text or args.test_speaker):
        raise ValueError('--no-audio requires --text and cannot be combined with --test-speaker')
    if args.turns < 0 or args.max_tokens < 1:
        raise ValueError('--turns must be >= 0 and --max-tokens must be >= 1')
    if args.output_sample_rate is not None and args.output_sample_rate <= 0:
        raise ValueError('--output-sample-rate must be > 0')
    if not 20 <= args.end_silence_ms <= 5000 or not 1 <= args.max_utterance_seconds <= 30:
        raise ValueError('Use 20–5000 ms endpoint silence and 1–30 second utterances')
    ZoneInfo(args.timezone)
    defaults = {'lmstudio': ('http://127.0.0.1:1234/v1', 'qwen/qwen3.5-9b'),
                'ollama': ('http://127.0.0.1:11434/v1', 'qwen2.5:3b')}
    base, model = defaults.get(args.backend, (None, None))
    args.base_url, args.model = args.base_url or base, args.model or model
    key = os.getenv(args.api_key_env)
    if args.backend == 'cloud' and (not args.base_url or not args.model or not key):
        raise ValueError('Cloud needs --base-url, --model and a key in the --api-key-env variable')
    extra = {}
    if args.reasoning == 'none' or (args.reasoning == 'auto' and args.backend == 'lmstudio' and 'qwen3.5' in args.model.lower()):
        extra['reasoning_effort'] = 'none'
    return key, extra


async def installed_voices():
    if platform.system() != 'Darwin':
        raise VoiceHardwareError('The baseline TTS requires macOS')
    process = await asyncio.create_subprocess_exec('say', '-v', '?', stdout=asyncio.subprocess.PIPE)
    from ai_character_engine.voice.mac_providers import stop_process
    try:
        raw, _ = await asyncio.wait_for(process.communicate(), 10)
        if process.returncode:
            raise VoiceHardwareError('Cannot enumerate macOS voices')
        return raw.decode()
    finally:
        await stop_process(process)


async def main(args):
    if args.list_devices:
        print(sounddevice().query_devices())
        return 0
    if args.list_voices:
        print(await installed_voices())
        return 0
    key, extra = config(args)
    if not args.test_agent and not args.no_audio:
        # macOS say has a configured output rate. GPT-SoVITS (and other remote
        # TTS backends) may choose their own WAV rate, so do not claim a 22.05
        # kHz speaker preflight unless the host explicitly supplies a hint.
        output_rate = args.output_sample_rate
        if output_rate is None and args.tts_provider == 'macos':
            output_rate = 22_050
        report = device_report(
            args.input_device, args.output_device, check_input=not args.text,
            output_sample_rate_hz=output_rate, check_output=output_rate is not None)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if output_rate is None:
            print('Speaker sample-rate preflight skipped: provider output rate is dynamic; actual synthesized audio will be validated by PortAudio.', flush=True)
        if args.tts_provider == 'macos':
            voices = await installed_voices()
            if not any(line.startswith(args.voice + ' ') for line in voices.splitlines()):
                raise VoiceHardwareError('Selected voice is not installed. Run --list-voices, then --voice NAME')
    sink = None if args.no_audio else SoundDeviceOutput(args.output_device)
    tts = None
    if not args.no_audio and not args.test_agent:
        tts = (GPTSoVITSTTS.from_config(args.sovits_config, timeout=args.tts_timeout)
               if args.tts_provider == 'gpt-sovits' else MacOSSayTTS(timeout=args.tts_timeout, sample_rate=args.output_sample_rate or 22_050))
        if args.tts_provider == 'gpt-sovits':
            await tts.check()
    if args.test_speaker:
        audio = await tts.synthesize('你好，這是角色語音測試。', voice=args.voice)
        await sink.play(AudioChunk(audio.data, audio.format))
        print('Speaker playback completed. Confirm audibility yourself.')
        return 0
    module = args.stt_backend.replace('-', '_')
    if not args.test_agent and not args.text and importlib.util.find_spec(module) is None:
        raise VoiceHardwareError(f'Missing {module}; install .[voice-mac] or .[voice-sherpa] as appropriate')
    stt_type = SherpaOnnxSTT if args.stt_backend == 'sherpa-onnx' else FasterWhisperSTT
    stt = None if args.test_agent or args.text else stt_type(args.stt_model, language=None if args.language == 'auto' else args.language,
        local_files_only=args.local_files_only, timeout=args.stt_timeout)
    provider = VoiceStreamingChatClient(model=args.model, base_url=args.base_url, api_key=key,
        backend=args.backend, timeout_seconds=args.llm_timeout, max_concurrency=1,
        max_tokens=args.max_tokens, extra_body=extra, require_time_tool=True)
    benchmark = LatencyBenchmark(args.benchmark)
    try:
        try:
            models = await asyncio.wait_for(provider.client.models.list(), 10)
        except Exception as exc:
            raise VoiceHardwareError('Cannot reach model server. Start LM Studio Local Server / ollama serve; check URL and authentication.') from exc
        names = [m.id for m in models.data]
        if args.model not in names:
            raise VoiceHardwareError(f'Model {args.model!r} unavailable. Server lists: {names}')
        print(f'Model available: {args.model} ({args.backend}); input={"keyboard" if args.text else "microphone"}; audio={"off" if args.no_audio or args.test_agent else args.tts_provider}')
        if args.check:
            print('Preflight passed. Text mode bypasses microphone and STT.' if args.text else 'Preflight passed. Mic permission, actual playback, STT model load and tool behavior still require a live test.')
            return 0
        gateway = ModelGatewayClient(endpoints=(ModelEndpoint('voice', provider, timeout_seconds=args.llm_timeout),),
                                     router=StaticModelRouter('voice'))
        llm = MeasuredLLM(gateway)
        session = make_session(llm, timezone=args.timezone)
        if args.test_agent:
            from ai_character_engine.events.models import CharacterEvent
            result = await session.process_event(CharacterEvent.user_message('現在幾點？請查詢目前時間，簡短回答。'))
            tool_results = [m.tool_result for m in session.runtime.history if m.role == 'tool']
            print(result.response.text)
            print(json.dumps(llm.metrics(), ensure_ascii=False, indent=2))
            if not any(t.name == 'get_current_time' and not t.is_error for t in tool_results):
                raise VoiceHardwareError('Agent test failed: model did not execute get_current_time. Check model tool support and reasoning settings.')
            print('Agent test passed: real tool call and response completed.')
            return 0
        runner = LiveVoiceRunner(session=session, llm=llm, stt=stt, tts=tts, sink=sink,
            voice=args.voice, stt_timeout=args.stt_timeout, character_timeout=args.llm_timeout * 4 + 10,
            tts_timeout=args.tts_timeout)
        if args.text:
            print('Text input: /quit or Ctrl-D exits. Ctrl-C cancels and closes resources.', flush=True)
        else:
            print('Loading STT (initial model download may take several minutes)…', flush=True)
            await stt.start()
            print('Half-duplex: wait for Listening before speaking. Ctrl-C stops and closes audio.', flush=True)
        completed = 0
        while args.turns == 0 or completed < args.turns:
            on_text = lambda role, text: print(f'{role}: {text}', flush=True)
            if args.text:
                content = await read_console_line()
                if content is None or content.strip().lower() == '/quit':
                    break
                if not content.strip():
                    continue
                metrics = await runner.run_text(content, on_text=on_text)
            else:
                print('Listening…', flush=True)
                try:
                    utterance = await capture_utterance(SoundDeviceInput(args.input_device),
                        threshold=args.vad_threshold, end_silence_ms=args.end_silence_ms,
                        max_seconds=args.max_utterance_seconds, listen_timeout=args.listen_timeout)
                except TimeoutError:
                    print('No complete speech before listen timeout; listening again.', flush=True)
                    continue
                metrics = await runner.run_turn(utterance, on_text=on_text)
            benchmark.add(metrics)
            completed += 1
            print(json.dumps(metrics, ensure_ascii=False), flush=True)
        return 0
    finally:
        if sink is not None:
            sink.cancel()
        if stt is not None:
            await stt.aclose()
        await provider.client.close()
        if benchmark.rows:
            print('Benchmark (nearest-rank p50/p95, milliseconds):')
            print(json.dumps(benchmark.summary(), ensure_ascii=False, indent=2))


def cli():
    try:
        return asyncio.run(main(parser().parse_args()))
    except KeyboardInterrupt:
        print('\nStopped. Active resources closed.')
        return 130
    except (VoiceHardwareError, ValueError, KeyError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 2
    except TimeoutError:
        print('Error: provider timeout. Check server load or increase the relevant timeout; resources closed.', file=sys.stderr)
        return 2
    except LLMError:
        print('Error: LLM request failed. Check model tools/streaming support, token limit and reasoning mode; see docs/mac_live_voice.md.', file=sys.stderr)
        return 2
    except OSError:
        print('Error: device/process/file unavailable. Check dependencies, audio settings and benchmark file permissions.', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(cli())
