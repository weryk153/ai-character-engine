from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ai_character_engine.avatar import AvatarBehaviorCueBundle

def _finite(name, value, *, minimum=None, maximum=None):
    value=float(value)
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return value

def _nonempty(name, value):
    value=str(value).strip()
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value

@dataclass(frozen=True, slots=True)
class VRMBehaviorCommand:
    channel: str
    target: str
    duration_ms: float
    values: dict[str, float] = field(default_factory=dict)
    source: str = "avatar_behavior"

    def __post_init__(self) -> None:
        object.__setattr__(self, "channel", _nonempty("channel", self.channel))
        object.__setattr__(self, "target", _nonempty("target", self.target))
        object.__setattr__(self, "source", _nonempty("source", self.source))
        object.__setattr__(self, "duration_ms", _finite("duration_ms", self.duration_ms, minimum=0.0))
        clean: dict[str, float] = {}
        for key, value in self.values.items():
            clean[_nonempty("value key", key)] = _finite(f"values[{key}]", value)
        object.__setattr__(self, "values", clean)

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "target": self.target,
            "duration_ms": self.duration_ms,
            "values": dict(self.values),
            "source": self.source,
        }


class VRM10BehaviorAdapter:
    """Translate neutral behavior cues to renderer-facing VRM host commands.

    VRM defines expressions and humanoid bones, but it does not standardize an
    application-level gesture library. Therefore animation/gesture commands are
    explicit host contracts rather than claims about a built-in VRM preset.
    """

    def commands(self, bundle: AvatarBehaviorCueBundle) -> tuple[VRMBehaviorCommand, ...]:
        commands: list[VRMBehaviorCommand] = []
        if bundle.gaze is not None:
            gaze = bundle.gaze
            commands.append(
                VRMBehaviorCommand(
                    "look_at",
                    gaze.target.name,
                    gaze.duration_ms,
                    {
                        "yaw_deg": gaze.target.yaw_deg,
                        "pitch_deg": gaze.target.pitch_deg,
                        "eye_weight": gaze.eye_weight,
                        "head_weight": gaze.head_weight,
                        "blend_ms": gaze.blend_ms,
                    },
                    gaze.source,
                )
            )
        if bundle.blink is not None:
            blink = bundle.blink
            commands.append(
                VRMBehaviorCommand(
                    "expression",
                    "blink",
                    blink.duration_ms,
                    {
                        "weight": blink.weight,
                        "close_ms": blink.close_ms,
                        "hold_ms": blink.hold_ms,
                        "open_ms": blink.open_ms,
                    },
                    blink.source,
                )
            )
        if bundle.head_motion is not None:
            head = bundle.head_motion
            commands.append(
                VRMBehaviorCommand(
                    "humanoid_rotation",
                    "head",
                    head.duration_ms,
                    {
                        "yaw_delta_deg": head.yaw_delta_deg,
                        "pitch_delta_deg": head.pitch_delta_deg,
                        "roll_delta_deg": head.roll_delta_deg,
                        "blend_ms": head.blend_ms,
                    },
                    head.source,
                )
            )
        if bundle.idle_motion is not None:
            idle = bundle.idle_motion
            commands.append(
                VRMBehaviorCommand(
                    "animation",
                    idle.motion,
                    idle.duration_ms,
                    {"intensity": idle.intensity, "loop": 1.0 if idle.loop else 0.0},
                    idle.source,
                )
            )
        for gesture in bundle.gestures:
            commands.append(
                VRMBehaviorCommand(
                    "animation",
                    gesture.gesture,
                    gesture.duration_ms,
                    {
                        "weight": gesture.weight,
                        "blend_in_ms": gesture.blend_in_ms,
                        "blend_out_ms": gesture.blend_out_ms,
                    },
                    gesture.source,
                )
            )
        return tuple(commands)


