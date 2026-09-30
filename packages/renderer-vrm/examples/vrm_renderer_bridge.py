"""Offline renderer bridge integration using the checked-in VRM manifest fixture."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from ai_character_engine_vrm import (
    RecordingRendererTransport,
    VRMCalibrationProfile,
    VRMModelManifest,
    VRMRendererBridge,
)
from ai_character_engine.live import LiveEventType, LiveRuntimeEvent

ROOT = Path(__file__).resolve().parent


async def main() -> None:
    manifest = VRMModelManifest.load(ROOT / "vroid_base_vrm_manifest.json")
    profile = VRMCalibrationProfile.load(ROOT / "vroid_base_vrm_calibration.json")
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
