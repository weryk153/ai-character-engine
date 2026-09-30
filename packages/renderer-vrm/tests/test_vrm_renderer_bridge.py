import asyncio
import json
import struct

import pytest

from ai_character_engine_vrm import (
    JsonLineRendererTransport,
    RecordingRendererTransport,
    VRMCalibrationProfile,
    VRMInspectionError,
    VRMRendererBridge,
    VRMRendererCalibrationError,
    inspect_vrm_bytes,
)
from ai_character_engine.live import LiveEventType, LiveRuntimeEvent


def glb_bytes(*, spec="1.0", expressions=None, bones=None, animations=None, look_at=True):
    expressions = expressions or ["aa", "ih", "ou", "ee", "oh", "blink", "happy"]
    bones = bones or ["head", "leftEye", "rightEye"]
    animations = animations or []
    vrm = {
        "specVersion": spec,
        "meta": {"name": "Fixture", "version": "1", "authors": ["test"]},
        "humanoid": {"humanBones": {name: {"node": 0} for name in bones}},
        "expressions": {"preset": {name: {} for name in expressions}, "custom": {}},
    }
    if look_at:
        vrm["lookAt"] = {"type": "bone", "offsetFromHeadBone": [0, 0, 0]}
    obj = {
        "asset": {"version": "2.0", "generator": "pytest"},
        "extensionsUsed": ["VRMC_vrm"],
        "extensions": {"VRMC_vrm": vrm},
        "nodes": [{}],
        "animations": [{"name": name} for name in animations],
    }
    raw = json.dumps(obj, separators=(",", ":")).encode()
    raw += b" " * ((4 - len(raw) % 4) % 4)
    chunk = struct.pack("<II", len(raw), 0x4E4F534A) + raw
    return b"glTF" + struct.pack("<II", 2, 12 + len(chunk)) + chunk


def avatar_event():
    return LiveRuntimeEvent(
        LiveEventType.AVATAR_CUE,
        input_id="turn-1",
        data={
            "vrm": [
                {
                    "expression": "aa",
                    "start_offset_ms": 10,
                    "duration_ms": 120,
                    "weight": 0.8,
                    "channel": "mouth",
                    "source": "provider",
                },
                {
                    "expression": "happy",
                    "start_offset_ms": 0,
                    "duration_ms": 500,
                    "weight": 0.6,
                    "channel": "expression:face",
                    "source": "state",
                },
            ]
        },
    )


def behavior_event():
    return LiveRuntimeEvent(
        LiveEventType.AVATAR_BEHAVIOR_CUE,
        input_id="turn-1",
        data={
            "vrm": [
                {
                    "channel": "look_at",
                    "target": "user",
                    "duration_ms": 1000,
                    "values": {"yaw_deg": 80, "pitch_deg": -40, "eye_weight": 1, "head_weight": 0.2, "blend_ms": 100},
                    "source": "host",
                },
                {
                    "channel": "humanoid_rotation",
                    "target": "head",
                    "duration_ms": 500,
                    "values": {"yaw_delta_deg": 20, "pitch_delta_deg": -20, "roll_delta_deg": 10, "blend_ms": 100},
                    "source": "idle_micro_motion",
                },
                {
                    "channel": "expression",
                    "target": "blink",
                    "duration_ms": 120,
                    "values": {"weight": 1.0},
                    "source": "natural_blink",
                },
                {
                    "channel": "animation",
                    "target": "wave",
                    "duration_ms": 800,
                    "values": {"weight": 1.0},
                    "source": "host",
                },
            ]
        },
    )


def test_inspector_extracts_vrm1_manifest():
    manifest = inspect_vrm_bytes(glb_bytes(animations=["idle"]))
    assert manifest.model_name == "Fixture"
    assert manifest.spec_version == "1.0"
    assert "aa" in manifest.expressions
    assert "head" in manifest.humanoid_bones
    assert manifest.animation_names == ("idle",)
    assert manifest.look_at_type == "bone"
    assert len(manifest.sha256) == 64


def test_inspector_rejects_non_glb():
    with pytest.raises(VRMInspectionError):
        inspect_vrm_bytes(b"not a vrm")


def test_default_profile_is_bound_to_model_and_ready():
    manifest = inspect_vrm_bytes(glb_bytes())
    profile = VRMCalibrationProfile.for_manifest(manifest)
    report = profile.validate(manifest)
    assert report.ready
    assert profile.model_sha256 == manifest.sha256


def test_hash_mismatch_is_blocking():
    manifest = inspect_vrm_bytes(glb_bytes())
    profile = VRMCalibrationProfile(model_sha256="0" * 64)
    report = profile.validate(manifest)
    assert not report.ready
    assert any(issue.code == "model_hash_mismatch" for issue in report.errors)
    with pytest.raises(VRMRendererCalibrationError):
        VRMRendererBridge(manifest, profile)


def test_missing_animation_library_is_nonblocking_info():
    manifest = inspect_vrm_bytes(glb_bytes())
    report = VRMCalibrationProfile.for_manifest(manifest).validate(manifest)
    assert report.ready
    assert any(issue.code == "no_animation_library" for issue in report.issues)


def test_avatar_event_maps_expression_with_gain_and_timing():
    manifest = inspect_vrm_bytes(glb_bytes())
    profile = VRMCalibrationProfile.for_manifest(manifest)
    profile = VRMCalibrationProfile.from_dict({**profile.to_dict(), "mouth_gain": 0.5})
    packet = VRMRendererBridge(manifest, profile, monotonic=lambda: 1.25).compile_event(avatar_event())
    assert packet.sequence == 0
    assert packet.created_at_ms == 1250
    assert [cmd.op for cmd in packet.commands] == ["set_expression", "set_expression"]
    assert packet.commands[0].target == "aa"
    assert packet.commands[0].values["weight"] == pytest.approx(0.4)
    assert packet.commands[0].start_offset_ms == 10
    assert packet.commands[0].duration_ms == 120


def test_behavior_commands_are_calibrated_and_clamped():
    manifest = inspect_vrm_bytes(glb_bytes())
    profile = VRMCalibrationProfile.for_manifest(manifest)
    profile = VRMCalibrationProfile.from_dict({
        **profile.to_dict(),
        "max_gaze_yaw_deg": 30,
        "max_gaze_pitch_deg": 15,
        "max_head_yaw_delta_deg": 5,
        "max_head_pitch_delta_deg": 4,
        "max_head_roll_delta_deg": 3,
    })
    packet = VRMRendererBridge(manifest, profile).compile_event(behavior_event())
    assert [cmd.op for cmd in packet.commands] == ["look_at_angles", "rotate_humanoid_delta", "set_expression"]
    gaze, head, blink = packet.commands
    assert gaze.values["yaw_deg"] == 30
    assert gaze.values["pitch_deg"] == -15
    assert head.values["yaw_delta_deg"] == 5
    assert head.values["pitch_delta_deg"] == -4
    assert head.values["roll_delta_deg"] == 3
    assert blink.target == "blink"
    assert "dropped unavailable animation 'wave'" in packet.warnings


def test_declared_external_animation_is_forwarded():
    manifest = inspect_vrm_bytes(glb_bytes())
    base = VRMCalibrationProfile.for_manifest(manifest)
    profile = VRMCalibrationProfile.from_dict({**base.to_dict(), "external_animations": ["wave"]})
    packet = VRMRendererBridge(manifest, profile).compile_event(behavior_event())
    assert any(cmd.op == "play_animation" and cmd.target == "wave" for cmd in packet.commands)


def test_reset_events_map_engine_channels_to_renderer_channels():
    manifest = inspect_vrm_bytes(glb_bytes())
    bridge = VRMRendererBridge(manifest)
    avatar = bridge.compile_event(LiveRuntimeEvent(LiveEventType.AVATAR_RESET, data={"reset": ["mouth", "expressions"]}))
    behavior = bridge.compile_event(LiveRuntimeEvent(LiveEventType.AVATAR_BEHAVIOR_RESET, data={"reset": ["gaze", "blink", "head", "gestures"]}))
    assert avatar.reset == ("expressions:mouth", "expressions:face")
    assert behavior.reset == ("look_at", "expressions:blink", "humanoid:head", "animation:gestures")


def test_unrelated_live_event_does_not_create_renderer_packet():
    manifest = inspect_vrm_bytes(glb_bytes())
    bridge = VRMRendererBridge(manifest)
    assert bridge.compile_event(LiveRuntimeEvent(LiveEventType.REPLY, text="hi")) is None


async def test_recording_transport_receives_packets_in_sequence():
    manifest = inspect_vrm_bytes(glb_bytes())
    transport = RecordingRendererTransport()
    bridge = VRMRendererBridge(manifest, transport=transport)
    await bridge.dispatch(avatar_event())
    await bridge.dispatch(behavior_event())
    assert [packet.sequence for packet in transport.packets] == [0, 1]


class Writer:
    def __init__(self):
        self.data = bytearray()
        self.drained = 0

    def write(self, value):
        self.data.extend(value)

    async def drain(self):
        self.drained += 1


async def test_json_line_transport_emits_one_json_packet_per_line():
    manifest = inspect_vrm_bytes(glb_bytes())
    writer = Writer()
    bridge = VRMRendererBridge(manifest, transport=JsonLineRendererTransport(writer))
    await bridge.dispatch(avatar_event())
    lines = bytes(writer.data).decode().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["schema_version"] == 1
    assert payload["source_event"] == "avatar_cue"
    assert writer.drained == 1


def test_profile_roundtrip_preserves_calibration_data(tmp_path):
    manifest = inspect_vrm_bytes(glb_bytes())
    profile = VRMCalibrationProfile.for_manifest(manifest)
    path = tmp_path / "calibration.json"
    profile.save(path)
    loaded = VRMCalibrationProfile.load(path)
    assert loaded.to_dict() == profile.to_dict()


def test_minimal_vrm0_manifest_is_inspectable_but_not_ready():
    obj = {
        "asset": {"version": "2.0"},
        "extensions": {"VRM": {"meta": {"title": "Legacy"}}},
        "extensionsUsed": ["VRM"],
    }
    raw = json.dumps(obj).encode()
    raw += b" " * ((4 - len(raw) % 4) % 4)
    data = b"glTF" + struct.pack("<II", 2, 20 + len(raw)) + struct.pack("<II", len(raw), 0x4E4F534A) + raw
    manifest = inspect_vrm_bytes(data)
    assert manifest.spec_version == "0.x"
    report = VRMCalibrationProfile(model_sha256=manifest.sha256).validate(manifest)
    assert not report.ready
    assert not any(issue.code == "unsupported_vrm_version" for issue in report.errors)


def vrm0_glb_bytes(*, expressions=None, bones=None, look_at="Bone"):
    expressions = expressions or [
        ("Neutral", "neutral"), ("A", "a"), ("I", "i"), ("U", "u"),
        ("E", "e"), ("O", "o"), ("Blink", "blink"),
        ("Blink_L", "blink_l"), ("Blink_R", "blink_r"),
        ("Angry", "angry"), ("Fun", "fun"), ("Joy", "joy"),
        ("Sorrow", "sorrow"), ("Surprised", "unknown"),
    ]
    bones = bones or ["head", "leftEye", "rightEye"]
    legacy = {
        "specVersion": "0.0",
        "meta": {"title": "Legacy Fixture", "version": "1.0"},
        "humanoid": {"humanBones": [{"bone": name, "node": 0} for name in bones]},
        "blendShapeMaster": {
            "blendShapeGroups": [
                {"name": name, "presetName": preset, "binds": []}
                for name, preset in expressions
            ]
        },
        "firstPerson": {"lookAtTypeName": look_at},
        "secondaryAnimation": {"boneGroups": [{}, {}], "colliderGroups": [{}]},
    }
    obj = {
        "asset": {"version": "2.0", "generator": "UniGLTF-test"},
        "extensionsUsed": ["VRM"],
        "extensions": {"VRM": legacy},
        "nodes": [{}],
    }
    raw = json.dumps(obj, separators=(",", ":")).encode()
    raw += b" " * ((4 - len(raw) % 4) % 4)
    chunk = struct.pack("<II", len(raw), 0x4E4F534A) + raw
    return b"glTF" + struct.pack("<II", 2, 12 + len(chunk)) + chunk


def test_vrm0_inspector_normalizes_legacy_capabilities():
    from ai_character_engine_vrm import VRMSpecFamily

    manifest = inspect_vrm_bytes(vrm0_glb_bytes())
    assert manifest.spec_family is VRMSpecFamily.VRM0
    assert manifest.spec_version == "0.0"
    assert manifest.model_name == "Legacy Fixture"
    assert {"head", "leftEye", "rightEye"}.issubset(manifest.humanoid_bones)
    assert {"A", "Blink", "Joy", "Sorrow", "Surprised"}.issubset(manifest.expressions)
    assert manifest.expression_presets["A"] == "a"
    assert manifest.expression_presets["Blink_L"] == "blink_l"
    assert manifest.look_at_type == "bone"
    assert manifest.secondary_animation_groups == 2
    assert manifest.collider_groups == 1


def test_vrm0_default_calibration_maps_to_engine_logical_vocabulary():
    manifest = inspect_vrm_bytes(vrm0_glb_bytes())
    profile = VRMCalibrationProfile.for_manifest(manifest)
    assert profile.expression_map["aa"] == "A"
    assert profile.expression_map["ih"] == "I"
    assert profile.expression_map["ou"] == "U"
    assert profile.expression_map["ee"] == "E"
    assert profile.expression_map["oh"] == "O"
    assert profile.expression_map["blink"] == "Blink"
    assert profile.expression_map["blinkLeft"] == "Blink_L"
    assert profile.expression_map["blinkRight"] == "Blink_R"
    assert profile.expression_map["happy"] == "Joy"
    assert profile.expression_map["relaxed"] == "Fun"
    assert profile.expression_map["sad"] == "Sorrow"
    assert profile.expression_map["surprised"] == "Surprised"
    report = profile.validate(manifest)
    assert report.ready
    assert any(item.code == "legacy_vrm0_compatibility" for item in report.issues)
    assert not any(item.code == "unsupported_vrm_version" for item in report.issues)


def test_vrm0_renderer_compiles_normalized_cues_to_legacy_targets():
    manifest = inspect_vrm_bytes(vrm0_glb_bytes())
    bridge = VRMRendererBridge(manifest)
    packet = bridge.compile_event(avatar_event())
    assert packet is not None
    assert [cmd.target for cmd in packet.commands] == ["A", "Joy"]
    payload = packet.to_dict()
    assert payload["model"]["spec_version"] == "0.0"
    assert payload["model"]["spec_family"] == "vrm0"


def test_vrm0_incomplete_model_is_rejected_for_missing_capabilities_not_version():
    obj = {
        "asset": {"version": "2.0"},
        "extensions": {"VRM": {"specVersion": "0.0", "meta": {"title": "Legacy"}}},
        "extensionsUsed": ["VRM"],
    }
    raw = json.dumps(obj).encode()
    raw += b" " * ((4 - len(raw) % 4) % 4)
    data = b"glTF" + struct.pack("<II", 2, 20 + len(raw)) + struct.pack("<II", len(raw), 0x4E4F534A) + raw
    manifest = inspect_vrm_bytes(data)
    profile = VRMCalibrationProfile.for_manifest(manifest)
    report = profile.validate(manifest)
    assert not report.ready
    assert not any(issue.code == "unsupported_vrm_version" for issue in report.errors)
    assert any(issue.code == "missing_required_expression" for issue in report.errors)
    assert any(issue.code == "missing_required_bone" for issue in report.errors)


def test_renderer_bridge_compiles_renderer_neutral_avatar_event():
    from ai_character_engine.avatar import AvatarCueBundle, ExpressionCue, VisemeCue

    manifest = inspect_vrm_bytes(glb_bytes())
    neutral = AvatarCueBundle(
        "turn-neutral",
        0,
        0,
        0.0,
        120.0,
        visemes=(VisemeCue("a", 10.0, 80.0, 0.8, "provider"),),
        expressions=(ExpressionCue("happy", 0.0, 120.0, 0.6),),
    )
    assert "vrm" not in neutral.to_dict()
    packet = VRMRendererBridge(manifest).compile_event(
        LiveRuntimeEvent(LiveEventType.AVATAR_CUE, input_id="turn-neutral", data=neutral.to_dict())
    )
    assert [cmd.target for cmd in packet.commands] == ["aa", "happy"]


def test_renderer_bridge_compiles_renderer_neutral_behavior_event():
    from ai_character_engine.avatar import (
        AvatarBehaviorCueBundle,
        AvatarBehaviorPhase,
        BlinkCue,
        GazeCue,
        GazeTarget,
        GestureCue,
        HeadMotionCue,
    )

    manifest = inspect_vrm_bytes(glb_bytes())
    base = VRMCalibrationProfile.for_manifest(manifest)
    profile = VRMCalibrationProfile.from_dict({**base.to_dict(), "external_animations": ["wave"]})
    neutral = AvatarBehaviorCueBundle(
        0,
        AvatarBehaviorPhase.SPEAKING,
        10.0,
        turn_id="turn-neutral",
        gaze=GazeCue(GazeTarget("user", 10.0, -5.0), 500.0),
        blink=BlinkCue(),
        head_motion=HeadMotionCue(1.0, -1.0, 0.5, 300.0),
        gestures=(GestureCue("wave", 600.0),),
    )
    assert "vrm" not in neutral.to_dict()
    packet = VRMRendererBridge(manifest, profile).compile_event(
        LiveRuntimeEvent(LiveEventType.AVATAR_BEHAVIOR_CUE, input_id="turn-neutral", data=neutral.to_dict())
    )
    assert [cmd.op for cmd in packet.commands] == [
        "look_at_angles",
        "set_expression",
        "rotate_humanoid_delta",
        "play_animation",
    ]
