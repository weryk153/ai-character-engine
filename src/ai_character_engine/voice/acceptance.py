"""Hardware acceptance helpers for the live Mac voice demo.

The evaluator only handles timing/tool evidence recorded by the engine. Human
confirmation is kept separate because software cannot prove that a person spoke
into the selected microphone or actually heard clean audio from the speaker.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

_REQUIRED_METRICS = (
    "stt_latency_ms",
    "llm_ttft_ms",
    "llm_total_latency_ms",
    "tts_ttfa_ms",
    "end_to_end_latency_ms",
)


def _valid_latency(value: Any) -> bool:
    return (type(value) in (int, float) and math.isfinite(value) and value >= 0)


def _nearest_rank(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    return ordered[max(0, math.ceil(len(ordered) * percentile) - 1)]


def evaluate_hardware_acceptance(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_turns: int = 5,
    human_confirmation: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    """Evaluate one live microphone acceptance run.

    Automatic evidence requires at least ``expected_turns`` microphone turns,
    complete latency metrics, no failed tools, and at least one successful tool
    call (the documented five-turn script uses ``get_current_time`` on turn 5).
    Human confirmation remains explicit rather than inferred from PortAudio.
    """
    if expected_turns < 5:
        raise ValueError("hardware acceptance requires at least 5 turns")
    selected = list(rows[-expected_turns:]) if len(rows) >= expected_turns else list(rows)
    microphone_turns = sum(row.get("input_source") == "microphone" for row in selected)
    complete_metrics = sum(
        all(_valid_latency(row.get(key)) for key in _REQUIRED_METRICS)
        for row in selected
    )
    successful_turns = sum(
        all(_valid_latency(row.get(key)) for key in _REQUIRED_METRICS)
        and int(row.get("tool_results_failed") or 0) == 0
        for row in selected
    )
    successful_tools = sum(int(row.get("tool_results_successful") or 0) for row in selected)
    failed_tools = sum(int(row.get("tool_results_failed") or 0) for row in selected)
    tool_calls = sum(int(row.get("tool_calls") or 0) for row in selected)
    auto_pass = (
        len(selected) == expected_turns
        and microphone_turns == expected_turns
        and complete_metrics == expected_turns
        and successful_tools >= 1
        and tool_calls >= 1
        and failed_tools == 0
    )
    human = dict(human_confirmation or {})
    human_required = ("live_microphone", "heard_speaker_audio", "no_severe_audio_issue")
    human_complete = all(key in human for key in human_required)
    human_pass = human_complete and all(bool(human[key]) for key in human_required)
    if not auto_pass:
        status = "failed"
    elif not human_complete:
        status = "pending_human_confirmation"
    elif human_pass:
        status = "passed"
    else:
        status = "failed"

    latency = {}
    for key in (*_REQUIRED_METRICS, "speech_end_to_stt_ms", "tts_provider_ttfa_ms", "first_playback_ms", "turn_complete_ms"):
        values = [float(row[key]) for row in selected if _valid_latency(row.get(key))]
        latency[key] = {
            "n": len(values),
            "p50": _nearest_rank(values, 0.50),
            "p95": _nearest_rank(values, 0.95),
        }
    return {
        "status": status,
        "expected_turns": expected_turns,
        "recorded_turns": len(selected),
        "successful_turns": successful_turns,
        "success_rate": successful_turns / expected_turns,
        "microphone_turns": microphone_turns,
        "complete_metric_turns": complete_metrics,
        "tool_calls": tool_calls,
        "successful_tool_results": successful_tools,
        "failed_tool_results": failed_tools,
        "automatic_checks_passed": auto_pass,
        "human_confirmation": human,
        "latency_ms": latency,
        "full_barge_in_supported": False,
        "acceptance_scope": "five_turn_half_duplex",
        "streaming_tts_verified": False,
        "memory_persistence_verified": False,
        "metric_definitions": {
            "stt_latency_ms": "STT processing only; excludes endpointing",
            "speech_end_to_stt_ms": "last speech buffer end to STT result",
            "tts_provider_ttfa_ms": "synthesis call to buffered PCM ready; not streaming",
            "first_playback_ms": "speech end to first PortAudio DAC estimate",
            "turn_complete_ms": "speech end to playback completion",
            "tts_ttfa_ms": "legacy: synthesis start to first PortAudio DAC estimate",
            "end_to_end_latency_ms": "legacy: speech end to first PortAudio DAC estimate",
        },
    }
