# macOS voice host

This optional host connects audio devices, STT, the character runtime and TTS.
It is separate from the platform-neutral core. Device/model availability and
perceptual quality require acceptance on the actual host.

## Setup

Use a configured inference endpoint and install the matching extra:

```sh
python -m pip install '.[voice-mac]'
python examples/integrations/live_voice_chat.py --help
```

The launchers under `tools/launchers` use uv and select the relevant extra. Run
from the repository root. Select the model and endpoint explicitly:

```sh
./tools/launchers/run_mac_voice.command --text --no-audio --backend lmstudio --base-url http://127.0.0.1:1234/v1 --model YOUR_MODEL
./tools/launchers/run_mac_voice.command --check
./tools/launchers/run_mac_voice.command --list-devices
```

`--text` bypasses microphone/STT; add `--no-audio` to bypass TTS and speaker output.
`/quit` or Ctrl-D exits text input. Ctrl-C cancels the local turn. The interactive
console adapter uses macOS/Unix event-loop readiness and is not a Windows console
implementation.

## Speech configuration

Set `--stt-backend faster-whisper` or `--stt-backend sherpa-onnx` and the appropriate
`--stt-model`. CLI arguments take precedence over `VOICE_STT_BACKEND` and
`VOICE_STT_MODEL`. `--local-files-only` prevents STT model downloads. The host does
not discover personal model directories or copy another application's models.

Use `--input-device` and `--output-device` to select devices. `--output-sample-rate`
can probe a specific speaker format; GPT-SoVITS output is also validated at its
actual returned sample rate. `--test-speaker` synthesizes a short test phrase.

The default TTS provider uses macOS speech synthesis. See [GPT-SoVITS](gpt_sovits.md)
for the optional HTTP synthesis adapter.

## Hardware acceptance

```sh
./tools/launchers/run_mac_acceptance.command --backend lmstudio --model YOUR_MODEL
```

The harness requires at least five turns and records timing and tool evidence.
Human confirmation is required for actual microphone input and audible output;
a device API alone cannot certify perceptual quality. Without confirmation the
report remains pending. Reports are written under `benchmarks/`, outside tracked
source and excluded from candidate identity.

Benchmark JSONL contains timing metrics; console transcripts and upstream server
logs follow their own policies. Compare latency only with a matching host profile.
