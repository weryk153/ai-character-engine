from __future__ import annotations

import inspect
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol

from ai_character_engine.live.models import LiveEventType, LiveRuntimeEvent

from .calibration import CalibrationReport, VRMCalibrationProfile, VRMModelManifest
from .avatar import VRM10AvatarAdapter
from .behavior import VRM10BehaviorAdapter


class VRMRendererCalibrationError(RuntimeError):
    pass


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


@dataclass(frozen=True, slots=True)
class VRMRendererCommand:
    op: str
    target: str
    channel: str
    duration_ms: float = 0.0
    start_offset_ms: float = 0.0
    values: dict[str, Any] = field(default_factory=dict)
    source: str = "engine"

    def __post_init__(self) -> None:
        if not self.op.strip() or not self.target.strip() or not self.channel.strip() or not self.source.strip():
            raise ValueError("op, target, channel and source must not be empty")
        for name in ("duration_ms", "start_offset_ms"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and >= 0")
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "target": self.target,
            "channel": self.channel,
            "duration_ms": self.duration_ms,
            "start_offset_ms": self.start_offset_ms,
            "values": dict(self.values),
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class VRMRendererPacket:
    sequence: int
    source_event: str
    model_name: str
    model_sha256: str
    created_at_ms: float
    model_spec_version: str | None = None
    model_spec_family: str | None = None
    turn_id: str | None = None
    commands: tuple[VRMRendererCommand, ...] = ()
    reset: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sequence": self.sequence,
            "source_event": self.source_event,
            "model": {
                "name": self.model_name,
                "sha256": self.model_sha256,
                "spec_version": self.model_spec_version,
                "spec_family": self.model_spec_family,
            },
            "created_at_ms": self.created_at_ms,
            "turn_id": self.turn_id,
            "commands": [item.to_dict() for item in self.commands],
            "reset": list(self.reset),
            "warnings": list(self.warnings),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))


class RendererTransport(Protocol):
    async def send(self, packet: VRMRendererPacket) -> None: ...


class RecordingRendererTransport:
    def __init__(self) -> None:
        self.packets: list[VRMRendererPacket] = []

    async def send(self, packet: VRMRendererPacket) -> None:
        self.packets.append(packet)


class CallableRendererTransport:
    def __init__(self, callback: Callable[[VRMRendererPacket], Awaitable[None] | None]) -> None:
        self.callback = callback

    async def send(self, packet: VRMRendererPacket) -> None:
        result = self.callback(packet)
        if inspect.isawaitable(result):
            await result


class JsonLineRendererTransport:
    """Newline-delimited JSON transport for a renderer subprocess/socket writer."""

    def __init__(self, writer: Any) -> None:
        if not hasattr(writer, "write"):
            raise TypeError("writer must provide write(bytes)")
        self.writer = writer

    async def send(self, packet: VRMRendererPacket) -> None:
        self.writer.write((packet.to_json() + "\n").encode("utf-8"))
        drain = getattr(self.writer, "drain", None)
        if drain is not None:
            result = drain()
            if inspect.isawaitable(result):
                await result


class VRMRendererBridge:
    """Compile live avatar events into calibrated renderer packets.

    The bridge intentionally consumes existing LiveRuntimeEvent objects so the
    character engine remains independent of three-vrm/Unity/Godot. A host can
    forward each live event through :meth:`dispatch` without changing
    the orchestrator event contract.
    """

    def __init__(
        self,
        manifest: VRMModelManifest,
        calibration: VRMCalibrationProfile | None = None,
        *,
        transport: RendererTransport | None = None,
        require_ready: bool = True,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.manifest = manifest
        self.calibration = calibration or VRMCalibrationProfile.for_manifest(manifest)
        self.report: CalibrationReport = self.calibration.validate(manifest)
        if require_ready and not self.report.ready:
            messages = "; ".join(item.message for item in self.report.errors)
            raise VRMRendererCalibrationError(messages or "VRM calibration is not ready")
        self.transport = transport
        self._monotonic = monotonic
        self._sequence = 0

    def _packet(
        self,
        event: LiveRuntimeEvent,
        commands: list[VRMRendererCommand],
        *,
        reset: tuple[str, ...] = (),
        warnings: list[str] | None = None,
    ) -> VRMRendererPacket:
        packet = VRMRendererPacket(
            sequence=self._sequence,
            source_event=event.type.value,
            model_name=self.manifest.model_name,
            model_sha256=self.manifest.sha256,
            created_at_ms=self._monotonic() * 1000.0,
            model_spec_version=self.manifest.spec_version,
            model_spec_family=self.manifest.spec_family.value,
            turn_id=event.input_id or event.data.get("turn_id"),
            commands=tuple(commands),
            reset=reset,
            warnings=tuple(warnings or ()),
        )
        self._sequence += 1
        return packet

    def _expression_command(self, raw: dict[str, Any], warnings: list[str]) -> VRMRendererCommand | None:
        logical = str(raw.get("expression") or raw.get("target") or "").strip()
        if not logical:
            warnings.append("dropped expression command without target")
            return None
        target = self.calibration.mapped_expression(logical)
        expressions = set(self.manifest.expressions) | set(self.manifest.custom_expressions)
        if target not in expressions:
            warnings.append(f"dropped unavailable expression '{target}'")
            return None
        channel = str(raw.get("channel") or "expression:face")
        source = str(raw.get("source") or "avatar")
        values = raw.get("values") or {}
        weight = raw.get("weight", values.get("weight", 1.0))
        gain = self.calibration.mouth_gain if channel == "mouth" else self.calibration.blink_gain if logical.startswith("blink") else self.calibration.expression_gain
        return VRMRendererCommand(
            "set_expression",
            target,
            channel,
            duration_ms=float(raw.get("duration_ms") or 0.0),
            start_offset_ms=float(raw.get("start_offset_ms") or 0.0),
            values={
                "weight": _clamp(float(weight) * gain, 0.0, 1.0),
                "logical_expression": logical,
            },
            source=source,
        )

    def _behavior_command(self, raw: dict[str, Any], warnings: list[str]) -> VRMRendererCommand | None:
        channel = str(raw.get("channel") or "")
        target = str(raw.get("target") or "")
        values = dict(raw.get("values") or {})
        source = str(raw.get("source") or "avatar_behavior")
        duration = float(raw.get("duration_ms") or 0.0)

        if channel == "expression":
            return self._expression_command(raw, warnings)
        if channel == "look_at":
            if self.manifest.look_at_type is None:
                warnings.append("dropped look-at command because model has no VRM lookAt")
                return None
            yaw = _clamp(float(values.get("yaw_deg", 0.0)) * self.calibration.gaze_yaw_scale, -self.calibration.max_gaze_yaw_deg, self.calibration.max_gaze_yaw_deg)
            pitch = _clamp(float(values.get("pitch_deg", 0.0)) * self.calibration.gaze_pitch_scale, -self.calibration.max_gaze_pitch_deg, self.calibration.max_gaze_pitch_deg)
            return VRMRendererCommand(
                "look_at_angles",
                target or "camera",
                "look_at",
                duration_ms=duration,
                values={
                    "yaw_deg": yaw,
                    "pitch_deg": pitch,
                    "eye_weight": _clamp(float(values.get("eye_weight", 1.0)), 0.0, 1.0),
                    "head_weight": _clamp(float(values.get("head_weight", 0.0)), 0.0, 1.0),
                    "blend_ms": max(0.0, float(values.get("blend_ms", 0.0))),
                },
                source=source,
            )
        if channel == "humanoid_rotation":
            bone = self.calibration.head_bone if target == "head" else target
            if bone not in self.manifest.humanoid_bones:
                warnings.append(f"dropped humanoid command for missing bone '{bone}'")
                return None
            scale = self.calibration.head_motion_scale
            return VRMRendererCommand(
                "rotate_humanoid_delta",
                bone,
                "humanoid",
                duration_ms=duration,
                values={
                    "yaw_delta_deg": _clamp(float(values.get("yaw_delta_deg", 0.0)) * scale, -self.calibration.max_head_yaw_delta_deg, self.calibration.max_head_yaw_delta_deg),
                    "pitch_delta_deg": _clamp(float(values.get("pitch_delta_deg", 0.0)) * scale, -self.calibration.max_head_pitch_delta_deg, self.calibration.max_head_pitch_delta_deg),
                    "roll_delta_deg": _clamp(float(values.get("roll_delta_deg", 0.0)) * scale, -self.calibration.max_head_roll_delta_deg, self.calibration.max_head_roll_delta_deg),
                    "blend_ms": max(0.0, float(values.get("blend_ms", 0.0))),
                },
                source=source,
            )
        if channel == "animation":
            mapped = self.calibration.mapped_animation(target)
            if mapped not in self.manifest.animation_names and mapped not in self.calibration.external_animations:
                warnings.append(f"dropped unavailable animation '{mapped}'")
                return None
            return VRMRendererCommand(
                "play_animation",
                mapped,
                "animation",
                duration_ms=duration,
                values=values,
                source=source,
            )
        warnings.append(f"dropped unsupported behavior channel '{channel}'")
        return None

    def compile_event(self, event: LiveRuntimeEvent) -> VRMRendererPacket | None:
        warnings: list[str] = []
        commands: list[VRMRendererCommand] = []
        if event.type is LiveEventType.AVATAR_CUE:
            from ai_character_engine.avatar import AudioEnvelopePoint, AvatarCueBundle, ExpressionCue, VisemeCue
            data = event.data
            if data.get("vrm"):
                for raw in data.get("vrm", ()):
                    command = self._expression_command(dict(raw), warnings)
                    if command is not None:
                        commands.append(command)
                return self._packet(event, commands, warnings=warnings)
            bundle = AvatarCueBundle(
                str(data.get("turn_id") or event.input_id or "renderer"),
                int(data.get("segment_sequence", 0)),
                int(data.get("chunk_index", 0)),
                float(data.get("start_offset_ms", 0.0)),
                float(data["duration_ms"]) if data.get("duration_ms") is not None else None,
                envelope=tuple(AudioEnvelopePoint(**item) for item in data.get("envelope", ())),
                visemes=tuple(VisemeCue(**item) for item in data.get("visemes", ())),
                expressions=tuple(ExpressionCue(**item) for item in data.get("expressions", ())),
                timing_source=str(data.get("timing_source") or "tts_audio"),
            )
            for keyframe in VRM10AvatarAdapter().commands(bundle):
                raw = keyframe.to_dict()
                command = self._expression_command(raw, warnings)
                if command is not None:
                    commands.append(command)
            return self._packet(event, commands, warnings=warnings)
        if event.type is LiveEventType.AVATAR_BEHAVIOR_CUE:
            if event.data.get("vrm"):
                for raw in event.data.get("vrm", ()):
                    command = self._behavior_command(dict(raw), warnings)
                    if command is not None:
                        commands.append(command)
                return self._packet(event, commands, warnings=warnings)
            from ai_character_engine.avatar import (
                AvatarBehaviorCueBundle, AvatarBehaviorPhase, BlinkCue, GazeCue, GazeTarget,
                GestureCue, HeadMotionCue, IdleMotionCue,
            )
            data = event.data
            gaze_data = data.get("gaze")
            gaze = None
            if gaze_data:
                target = GazeTarget(**gaze_data["target"])
                gaze = GazeCue(target=target, duration_ms=gaze_data["duration_ms"], eye_weight=gaze_data.get("eye_weight",1.0), head_weight=gaze_data.get("head_weight",0.2), blend_ms=gaze_data.get("blend_ms",120.0), priority=gaze_data.get("priority",0), source=gaze_data.get("source","behavior_default"))
            bundle = AvatarBehaviorCueBundle(
                sequence=int(data.get("sequence", 0)),
                phase=AvatarBehaviorPhase(str(data.get("phase") or "idle")),
                elapsed_ms=float(data.get("elapsed_ms", 0.0)),
                turn_id=data.get("turn_id"),
                gaze=gaze,
                blink=BlinkCue(**{k:v for k,v in data["blink"].items() if k != "duration_ms"}) if data.get("blink") else None,
                head_motion=HeadMotionCue(**data["head_motion"]) if data.get("head_motion") else None,
                idle_motion=IdleMotionCue(**data["idle_motion"]) if data.get("idle_motion") else None,
                gestures=tuple(GestureCue(**item) for item in data.get("gestures", ())),
            )
            for raw_cmd in VRM10BehaviorAdapter().commands(bundle):
                command = self._behavior_command(raw_cmd.to_dict(), warnings)
                if command is not None:
                    commands.append(command)
            return self._packet(event, commands, warnings=warnings)
        if event.type is LiveEventType.AVATAR_RESET:
            mapping = {"mouth": "expressions:mouth", "expressions": "expressions:face"}
            reset = tuple(mapping.get(str(item), str(item)) for item in event.data.get("reset", ()))
            return self._packet(event, [], reset=reset)
        if event.type is LiveEventType.AVATAR_BEHAVIOR_RESET:
            mapping = {"gaze": "look_at", "blink": "expressions:blink", "head": f"humanoid:{self.calibration.head_bone}", "idle_motion": "animation:idle", "gestures": "animation:gestures"}
            reset = tuple(mapping.get(str(item), str(item)) for item in event.data.get("reset", ()))
            return self._packet(event, [], reset=reset)
        return None
    async def dispatch(self, event: LiveRuntimeEvent) -> VRMRendererPacket | None:
        packet = self.compile_event(event)
        if packet is not None and self.transport is not None:
            await self.transport.send(packet)
        return packet

    async def dispatch_many(self, events: list[LiveRuntimeEvent] | tuple[LiveRuntimeEvent, ...]) -> tuple[VRMRendererPacket, ...]:
        packets: list[VRMRendererPacket] = []
        for event in events:
            packet = await self.dispatch(event)
            if packet is not None:
                packets.append(packet)
        return tuple(packets)
