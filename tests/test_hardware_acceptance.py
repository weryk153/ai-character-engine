from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from ai_character_engine.voice.acceptance import evaluate_hardware_acceptance
from ai_character_engine.voice import devices


def _row(**overrides):
    row = {
        "input_source": "microphone",
        "stt_latency_ms": 100.0,
        "llm_ttft_ms": 120.0,
        "llm_total_latency_ms": 250.0,
        "tts_ttfa_ms": 80.0,
        "end_to_end_latency_ms": 550.0,
        "tool_calls": 0,
        "tool_results_successful": 0,
        "tool_results_failed": 0,
    }
    row.update(overrides)
    return row


def test_hardware_acceptance_requires_metrics_tool_and_human_confirmation():
    rows = [_row() for _ in range(4)] + [_row(tool_calls=1, tool_results_successful=1)]
    pending = evaluate_hardware_acceptance(rows)
    assert pending["automatic_checks_passed"] is True
    assert pending["status"] == "pending_human_confirmation"
    passed = evaluate_hardware_acceptance(rows, human_confirmation={
        "live_microphone": True,
        "heard_speaker_audio": True,
        "no_severe_audio_issue": True,
    })
    assert passed["status"] == "passed"
    assert passed["latency_ms"]["stt_latency_ms"]["p50"] == 100.0
    failed = evaluate_hardware_acceptance(rows[:-1] + [_row(input_source="text")], human_confirmation={
        "live_microphone": True,
        "heard_speaker_audio": True,
        "no_severe_audio_issue": True,
    })
    assert failed["status"] == "failed"


def test_device_report_uses_selected_output_rate_and_can_skip_dynamic_probe(monkeypatch):
    output_calls = []
    fake = NS(
        query_devices=lambda: [dict(name="Device", max_input_channels=1, max_output_channels=2, default_samplerate=48000)],
        default=NS(device=(0, 0)),
        check_input_settings=lambda **kwargs: None,
        check_output_settings=lambda **kwargs: output_calls.append(kwargs),
    )
    monkeypatch.setattr(devices, "sounddevice", lambda: fake)
    report = devices.device_report(output_sample_rate_hz=32000)
    assert output_calls[-1]["samplerate"] == 32000
    assert report["output_sample_rate_hz"] == 32000
    output_calls.clear()
    report = devices.device_report(output_sample_rate_hz=None, check_output=False)
    assert output_calls == []
    assert report["output_pcm16_mono"] is None


def test_generic_mac_launchers_have_no_product_specific_model_paths():
    root = Path(__file__).resolve().parents[1]
    voice = (root / "tools/launchers/run_mac_voice.command").read_text(encoding="utf-8")
    acceptance = (root / "tools/launchers/run_mac_acceptance.command").read_text(encoding="utf-8")
    combined = (voice + acceptance).lower()
    assert "/users/" not in combined and "/home/" not in combined
    assert "voice_stt_backend" in combined
    assert "--stt-backend" in combined


def test_hardware_acceptance_rejects_less_than_five_turns():
    with pytest.raises(ValueError, match="at least 5"):
        evaluate_hardware_acceptance([], expected_turns=4)
