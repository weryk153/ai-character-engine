from __future__ import annotations

from ai_character_engine_vrm import VRMCalibrationProfile, VRMModelManifest


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
