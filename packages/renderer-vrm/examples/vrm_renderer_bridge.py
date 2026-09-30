"""Compile engine avatar cues into renderer packets for a VRM model.

    python vrm_renderer_bridge.py            # a model with the standard VRM 1.0 vocabulary
    python vrm_renderer_bridge.py model.vrm  # your own VRM 0.x or 1.0 model

No model file ships with the engine; bring your own.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from ai_character_engine_vrm import (
    RecordingRendererTransport,
    VRMCalibrationProfile,
    VRMModelManifest,
    VRMRendererBridge,
    inspect_vrm_path,
)
from ai_character_engine.live import LiveEventType, LiveRuntimeEvent

# The preset expressions and humanoid bones of the VRM 1.0 specification.
STANDARD_EXPRESSIONS = (
    "aa", "ih", "ou", "ee", "oh", "blink", "blinkLeft", "blinkRight",
    "happy", "angry", "sad", "relaxed", "surprised", "neutral",
    "lookUp", "lookDown", "lookLeft", "lookRight",
)
STANDARD_BONES = (
    "hips", "spine", "chest", "upperChest", "neck", "head", "leftEye", "rightEye", "jaw",
    *(
        f"{side}{part}"
        for side in ("left", "right")
        for part in (
            "Shoulder", "UpperArm", "LowerArm", "Hand", "UpperLeg", "LowerLeg", "Foot", "Toes",
            "ThumbMetacarpal", "ThumbProximal", "ThumbDistal",
            *(f"{finger}{joint}" for finger in ("Index", "Middle", "Ring", "Little")
              for joint in ("Proximal", "Intermediate", "Distal")),
        )
    ),
)


def standard_manifest() -> VRMModelManifest:
    return VRMModelManifest(
        model_name="Standard VRM 1.0",
        model_version=None,
        spec_version="1.0",
        generator=None,
        sha256="0" * 64,
        expressions=STANDARD_EXPRESSIONS,
        custom_expressions=(),
        humanoid_bones=STANDARD_BONES,
        animation_names=(),
        extensions=("VRMC_vrm",),
        look_at_type="bone",
    )


async def main() -> None:
    manifest = inspect_vrm_path(Path(sys.argv[1])) if len(sys.argv) > 1 else standard_manifest()
    profile = VRMCalibrationProfile.for_manifest(manifest)
    transport = RecordingRendererTransport()
    bridge = VRMRendererBridge(manifest, profile, transport=transport)

    events = [
        LiveRuntimeEvent(
            LiveEventType.AVATAR_CUE,
            input_id="demo-turn",
            data={
                "vrm": [
                    {
                        "expression": "aa",
                        "start_offset_ms": 0,
                        "duration_ms": 120,
                        "weight": 0.72,
                        "channel": "mouth",
                        "source": "provider",
                    },
                    {
                        "expression": "happy",
                        "start_offset_ms": 0,
                        "duration_ms": 800,
                        "weight": 0.55,
                        "channel": "expression:face",
                        "source": "state",
                    },
                ]
            },
        ),
        LiveRuntimeEvent(
            LiveEventType.AVATAR_BEHAVIOR_CUE,
            input_id="demo-turn",
            data={
                "vrm": [
                    {
                        "channel": "look_at",
                        "target": "user",
                        "duration_ms": 1000,
                        "values": {"yaw_deg": 8, "pitch_deg": -3, "eye_weight": 1, "head_weight": 0.2, "blend_ms": 120},
                        "source": "host",
                    },
                    {
                        "channel": "expression",
                        "target": "blink",
                        "duration_ms": 195,
                        "values": {"weight": 1.0},
                        "source": "natural_blink",
                    },
                    {
                        "channel": "humanoid_rotation",
                        "target": "head",
                        "duration_ms": 700,
                        "values": {"yaw_delta_deg": 1.5, "pitch_delta_deg": -0.5, "roll_delta_deg": 0.25, "blend_ms": 180},
                        "source": "idle_micro_motion",
                    },
                    {
                        "channel": "animation",
                        "target": "small_wave",
                        "duration_ms": 900,
                        "values": {"weight": 1.0},
                        "source": "host",
                    },
                ]
            },
        ),
        LiveRuntimeEvent(
            LiveEventType.AVATAR_RESET,
            input_id="demo-turn",
            data={"reset": ["mouth", "expressions"]},
        ),
    ]

    for event in events:
        await bridge.dispatch(event)

    print(json.dumps(bridge.report.to_dict(), ensure_ascii=False, indent=2))
    for packet in transport.packets:
        print(packet.to_json())


if __name__ == "__main__":
    asyncio.run(main())
