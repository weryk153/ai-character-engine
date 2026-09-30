#!/bin/zsh
set -eu
cd "$(dirname "$0")/../.."
exec ./tools/launchers/run_mac_voice.command --text --tts-provider gpt-sovits --tts-timeout 180 "$@"
