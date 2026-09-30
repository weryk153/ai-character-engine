#!/usr/bin/env python3
"""Run the five-turn Mac microphone hardware acceptance sequence.

Usage:
  python examples/integrations/hardware_acceptance.py -- --backend lmstudio
  python examples/integrations/hardware_acceptance.py -- --stt-backend sherpa-onnx --stt-model /path/to/model

Everything after ``--`` is forwarded to ``examples/integrations/live_voice_chat.py``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from ai_character_engine.voice.acceptance import evaluate_hardware_acceptance

PROMPTS = (
    "我叫 Alex。",
    "我今天在做 AI Character Engine。",
    "我剛才說我在做什麼？",
    "你記得我的名字嗎？",
    "現在幾點？",
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--turns", type=int, default=5, help="must be at least 5")
    p.add_argument("--benchmark", type=Path, default=Path("benchmarks/hardware_acceptance.jsonl"))
    p.add_argument("--report", type=Path, default=Path("benchmarks/hardware_acceptance_report.json"))
    p.add_argument("--non-interactive", action="store_true", help="leave human confirmation pending")
    p.add_argument("live_args", nargs=argparse.REMAINDER, help="arguments forwarded after --")
    return p


def _read_new_rows(path: Path, offset: int) -> list[dict]:
    if not path.exists():
        return []
    with path.open("rb") as file:
        file.seek(offset)
        raw = file.read().decode("utf-8")
    rows = []
    for line in raw.splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _confirm(prompt: str) -> bool:
    answer = input(prompt + " [y/N] ").strip().lower()
    return answer in {"y", "yes", "是", "好"}


def main() -> int:
    args = parser().parse_args()
    if args.turns < 5:
        raise SystemExit("--turns must be at least 5")
    forwarded = list(args.live_args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    forbidden = {"--text", "--no-audio", "--turns", "--benchmark", "--test-agent", "--check"}
    for item in forwarded:
        name = item.split("=", 1)[0]
        if name in forbidden:
            raise SystemExit(f"hardware acceptance controls {name}; remove it from forwarded arguments")

    print("Hardware acceptance — use a real microphone and follow these five prompts in order:")
    for index, prompt in enumerate(PROMPTS, 1):
        print(f"  {index}. {prompt}")
    if args.turns > len(PROMPTS):
        print(f"  Then continue naturally for {args.turns - len(PROMPTS)} more turn(s).")
    print("Full automatic barge-in is NOT part of this milestone; Ctrl-C/cancel hook remains available.\n")

    args.benchmark.parent.mkdir(parents=True, exist_ok=True)
    offset = args.benchmark.stat().st_size if args.benchmark.exists() else 0
    live = Path(__file__).with_name("live_voice_chat.py")
    command = [sys.executable, str(live), "--turns", str(args.turns), "--benchmark", str(args.benchmark), *forwarded]
    completed = subprocess.run(command)
    rows = _read_new_rows(args.benchmark, offset)

    human = None
    if completed.returncode == 0 and not args.non_interactive and sys.stdin.isatty():
        human = {
            "live_microphone": _confirm("Were all five turns spoken by a person directly into the selected microphone?"),
            "heard_speaker_audio": _confirm("Did you actually hear the character from the selected speaker or headphones on every turn?"),
            "no_severe_audio_issue": _confirm("Was there no severe clipping, cut-off sentence or unintelligible recognition?"),
        }
    report = evaluate_hardware_acceptance(rows, expected_turns=args.turns, human_confirmation=human)
    report["child_exit_code"] = completed.returncode
    report["expected_script"] = list(PROMPTS)
    report["benchmark_file"] = str(args.benchmark)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Acceptance report: {args.report}")
    return 0 if report["status"] == "passed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
