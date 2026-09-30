#!/bin/zsh
set -eu
cd "$(dirname "$0")/../.."
engine_dir="$PWD"
sovits_root="${GPT_SOVITS_ROOT:-$engine_dir/../GPT-SoVITS}"
if [[ ! -x "$sovits_root/.venv/bin/python" || ! -f "$sovits_root/api_v2.py" ]]; then
  print -u2 'GPT-SoVITS not found. Set GPT_SOVITS_ROOT to an existing installation.'
  exit 2
fi
if curl --silent --fail --max-time 3 http://127.0.0.1:9880/openapi.json >/dev/null; then
  print 'A service is already running on 9880; using it instead of starting another.'
  exit 0
fi
mkdir -p "$engine_dir/.data"
cp "$sovits_root/GPT_SoVITS/configs/tts_infer.yaml" "$engine_dir/.data/gpt-sovits-infer.yaml"
cd "$sovits_root"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
exec .venv/bin/python -u api_v2.py -a 127.0.0.1 -p 9880 -c "$engine_dir/.data/gpt-sovits-infer.yaml"
