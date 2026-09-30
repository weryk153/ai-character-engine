from __future__ import annotations

from dataclasses import dataclass, field

from ai_character_engine.avatar import AvatarCueBundle, Viseme
from .models import VRMExpressionKeyframe


@dataclass(frozen=True, slots=True)
class VRM10AvatarConfig:
    viseme_map: dict[str, str] = field(
        default_factory=lambda: {
            Viseme.A.value: "aa",
            Viseme.I.value: "ih",
            Viseme.U.value: "ou",
            Viseme.E.value: "ee",
            Viseme.O.value: "oh",
        }
    )
    expression_map: dict[str, str] = field(
        default_factory=lambda: {
            "happy": "happy",
            "angry": "angry",
            "sad": "sad",
            "relaxed": "relaxed",
            "surprised": "surprised",
        }
    )
    envelope_fallback_expression: str = "aa"
    envelope_gain: float = 1.0

    def __post_init__(self) -> None:
        if self.envelope_gain < 0:
            raise ValueError("envelope_gain must be >= 0")
        if not self.envelope_fallback_expression.strip():
            raise ValueError("envelope_fallback_expression must not be empty")


class VRM10AvatarAdapter:
    """Translate neutral engine cues to VRM 1.0 expression keyframes.

    The adapter emits data only. It never imports a renderer, writes a VRM file,
    or assumes ownership of a Unity/three-vrm scene.
    """

    def __init__(self, config: VRM10AvatarConfig | None = None) -> None:
        self.config = config or VRM10AvatarConfig()

    def commands(self, bundle: AvatarCueBundle) -> tuple[VRMExpressionKeyframe, ...]:
        commands: list[VRMExpressionKeyframe] = []
        if bundle.visemes:
            for cue in bundle.visemes:
                label = cue.viseme.value if isinstance(cue.viseme, Viseme) else str(cue.viseme)
                mapped = self.config.viseme_map.get(label)
                if mapped is None:
                    continue
                commands.append(
                    VRMExpressionKeyframe(
                        mapped,
                        cue.start_offset_ms,
                        cue.duration_ms,
                        cue.weight,
                        "mouth",
                        cue.source,
                    )
                )
        else:
            for point in bundle.envelope:
                weight = min(1.0, point.amplitude * self.config.envelope_gain)
                if weight <= 0:
                    continue
                commands.append(
                    VRMExpressionKeyframe(
                        self.config.envelope_fallback_expression,
                        point.start_offset_ms,
                        point.duration_ms,
                        weight,
                        "mouth",
                        "audio_envelope",
                    )
                )
        for cue in bundle.expressions:
            mapped = self.config.expression_map.get(cue.expression, cue.expression)
            commands.append(
                VRMExpressionKeyframe(
                    mapped,
                    cue.start_offset_ms,
                    cue.duration_ms,
                    cue.weight,
                    f"expression:{cue.group}",
                    cue.source,
                )
            )
        return tuple(commands)
