from __future__ import annotations

import json
from pathlib import Path

from ai_character_engine import __version__
from ai_character_engine_vrm import (
    VRMCalibrationProfile,
    VRMModelManifest,
    VRMRendererBridge,
    VRMSpecFamily,
)
from ai_character_engine.live import LiveEventType, LiveRuntimeEvent

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "examples" / "vrm0_acceptance"


def test_v0291_version():
    assert __version__ == "1.0.0"


def test_real_vrm0_derived_fixture_is_ready_without_raw_asset():
    manifest = VRMModelManifest.load(FIXTURE / "manifest.json")
    profile = VRMCalibrationProfile.load(FIXTURE / "calibration.json")
    report = profile.validate(manifest)

    assert manifest.spec_family is VRMSpecFamily.VRM0
    assert manifest.spec_version == "0.0"
    assert manifest.sha256 == "b5fffb730d99b27ffdd64f6c1f039f1b6c5c4cfaa502d474a90374b226b8b48e"
    assert len(manifest.humanoid_bones) == 54
    assert len(manifest.expressions) == 18
    assert manifest.secondary_animation_groups == 9
    assert manifest.collider_groups == 12
    assert manifest.look_at_type == "bone"
    assert report.ready
    assert profile.expression_map["aa"] == "A"
    assert profile.expression_map["happy"] == "Joy"
    assert profile.expression_map["sad"] == "Sorrow"
    assert profile.expression_map["surprised"] == "Surprised"

    expected_report = json.loads((FIXTURE / "report.json").read_text(encoding="utf-8"))
    assert report.to_dict() == expected_report
    assert not any(path.suffix.lower() in {".vrm", ".vrma"} for path in FIXTURE.rglob("*"))


def test_real_vrm0_fixture_compiles_renderer_targets():
    manifest = VRMModelManifest.load(FIXTURE / "manifest.json")
    profile = VRMCalibrationProfile.load(FIXTURE / "calibration.json")
    bridge = VRMRendererBridge(manifest, profile)
    packet = bridge.compile_event(
        LiveRuntimeEvent(
            LiveEventType.AVATAR_CUE,
            input_id="fixture-turn",
            data={
                "vrm": [
                    {"expression": "aa", "channel": "mouth", "weight": 0.7},
                    {"expression": "happy", "channel": "expression:face", "weight": 0.5},
                    {"expression": "sad", "channel": "expression:face", "weight": 0.4},
                ]
            },
        )
    )
    assert packet is not None
    assert [command.target for command in packet.commands] == ["A", "Joy", "Sorrow"]
    assert packet.model_spec_family == "vrm0"
    assert packet.model_spec_version == "0.0"


def test_future_spec_string_is_not_silently_accepted_under_vrm1_extension():
    manifest = VRMModelManifest(
        model_name="Future",
        model_version=None,
        spec_version="2.0",
        generator=None,
        sha256="0" * 64,
        expressions=("aa", "blink"),
        custom_expressions=(),
        humanoid_bones=("head",),
        animation_names=(),
        extensions=("VRMC_vrm",),
    )
    report = VRMCalibrationProfile.for_manifest(manifest).validate(manifest)
    assert not report.ready
    assert any(issue.code == "unsupported_vrm_version" for issue in report.errors)
