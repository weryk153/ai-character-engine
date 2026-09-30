#!/bin/zsh
set -eu
cd "$(dirname "$0")/../.."
if ! command -v uv >/dev/null 2>&1; then
  print -u2 'uv not found. Install it as docs/mac_live_voice.md describes, or use the manual Python installation.'
  exit 2
fi
stt_backend="${VOICE_STT_BACKEND:-faster-whisper}"
i=1
while (( i <= $# )); do
  arg="${argv[$i]}"
  if [[ "$arg" == "--stt-backend" ]]; then
    (( i += 1 ))
    (( i <= $# )) || { print -u2 '--stt-backend requires a value'; exit 2; }
    stt_backend="${argv[$i]}"
  elif [[ "$arg" == --stt-backend=* ]]; then
    stt_backend="${arg#--stt-backend=}"
  fi
  (( i += 1 ))
done
extra=voice-mac
if [[ "$stt_backend" == sherpa-onnx ]]; then
  extra=voice-sherpa
fi
exec uv run --python 3.11 --extra "$extra" python examples/integrations/hardware_acceptance.py -- "$@"
