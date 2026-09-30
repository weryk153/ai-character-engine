# GPT-SoVITS integration

The optional `GPTSoVITSTTS` adapter connects to an existing GPT-SoVITS `api_v2.py`
service. The application owner supplies model files and reference audio. This
repository contains no personal voice recording, model checkpoint or preconfigured
machine-specific path.

## Configure the service

Copy `config/gpt_sovits.example.json` to `config/gpt_sovits.local.json` and set:

| Field | Meaning |
|---|---|
| `api_url` | HTTP `/tts` endpoint |
| `ref_audio_path` | Reference audio path readable by the synthesis server |
| `prompt_text` | Exact reference-audio transcript |
| `prompt_lang` | Reference-audio language |
| `text_lang` | Generated speech language |

The local file is ignored by Git. Configure an installed service separately or
set `GPT_SOVITS_ROOT` and run `tools/launchers/start_gpt_sovits.command` on macOS.
That launcher uses an existing environment and copies inference configuration
into `.data/`; it does not download a model or install the upstream application.

## Run the host

```sh
python -m pip install '.[voice-audio]'
python examples/integrations/live_voice_chat.py --text --tts-provider gpt-sovits --sovits-config config/gpt_sovits.local.json --tts-timeout 180 --model YOUR_MODEL
```

Configure the inference endpoint using `--backend` and `--base-url` as needed.
`--voice` controls system TTS, not the GPT-SoVITS reference voice. `/quit` exits the
chat without stopping a shared synthesis server.

## Behavior and acceptance

The adapter requests non-streaming synthesis, validates WAV responses, converts
them to PCM16 and preserves the returned sample rate. HTTP errors, invalid audio,
size limits and timeouts fail explicitly rather than switching voices silently.
Local cancellation closes client playback/HTTP work; it does not guarantee remote
inference cancellation or terminate a shared service.

Use `--check`, `--test-speaker` and the [hardware acceptance host](mac_live_voice.md)
on the target machine. Voice similarity and audible quality require human review.
No prior machine's benchmark is treated as acceptance for another host.
