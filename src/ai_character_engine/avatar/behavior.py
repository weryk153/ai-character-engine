from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any
from uuid import uuid4


def _finite(name: str, value: float, *, minimum: float | None = None, maximum: float | None = None) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return value


def _nonempty(name: str, value: str) -> str:
    value = str(value).strip()
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


class AvatarBehaviorPhase(StrEnum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True, slots=True)
class GazeTarget:
    name: str = "camera"
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonempty("name", self.name))
        object.__setattr__(self, "yaw_deg", _finite("yaw_deg", self.yaw_deg, minimum=-90.0, maximum=90.0))
        object.__setattr__(self, "pitch_deg", _finite("pitch_deg", self.pitch_deg, minimum=-60.0, maximum=60.0))

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "yaw_deg": self.yaw_deg, "pitch_deg": self.pitch_deg}


@dataclass(frozen=True, slots=True)
class GazeCue:
    target: GazeTarget
    duration_ms: float
    eye_weight: float = 1.0
    head_weight: float = 0.2
    blend_ms: float = 120.0
    priority: int = 0
    source: str = "behavior_default"

    def __post_init__(self) -> None:
        object.__setattr__(self, "duration_ms", _finite("duration_ms", self.duration_ms, minimum=0.0))
        object.__setattr__(self, "eye_weight", _finite("eye_weight", self.eye_weight, minimum=0.0, maximum=1.0))
        object.__setattr__(self, "head_weight", _finite("head_weight", self.head_weight, minimum=0.0, maximum=1.0))
        object.__setattr__(self, "blend_ms", _finite("blend_ms", self.blend_ms, minimum=0.0))
        if type(self.priority) is not int:
            raise ValueError("priority must be an integer")
        object.__setattr__(self, "source", _nonempty("source", self.source))

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target.to_dict(),
            "duration_ms": self.duration_ms,
            "eye_weight": self.eye_weight,
            "head_weight": self.head_weight,
            "blend_ms": self.blend_ms,
            "priority": self.priority,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class GazeRequest:
    target: GazeTarget
    duration_ms: float = 2000.0
    eye_weight: float = 1.0
    head_weight: float = 0.25
    blend_ms: float = 120.0
    priority: int = 100
    delay_ms: float = 0.0
    source: str = "host"
    turn_scoped: bool = False
    id: str = field(default_factory=lambda: uuid4().hex)

    def __post_init__(self) -> None:
        object.__setattr__(self, "duration_ms", _finite("duration_ms", self.duration_ms, minimum=0.0))
        object.__setattr__(self, "delay_ms", _finite("delay_ms", self.delay_ms, minimum=0.0))
        object.__setattr__(self, "eye_weight", _finite("eye_weight", self.eye_weight, minimum=0.0, maximum=1.0))
        object.__setattr__(self, "head_weight", _finite("head_weight", self.head_weight, minimum=0.0, maximum=1.0))
        object.__setattr__(self, "blend_ms", _finite("blend_ms", self.blend_ms, minimum=0.0))
        if type(self.priority) is not int:
            raise ValueError("priority must be an integer")
        object.__setattr__(self, "source", _nonempty("source", self.source))
        object.__setattr__(self, "id", _nonempty("id", self.id))


@dataclass(frozen=True, slots=True)
class BlinkCue:
    close_ms: float = 70.0
    hold_ms: float = 35.0
    open_ms: float = 90.0
    weight: float = 1.0
    source: str = "natural_blink"

    def __post_init__(self) -> None:
        for name in ("close_ms", "hold_ms", "open_ms"):
            object.__setattr__(self, name, _finite(name, getattr(self, name), minimum=0.0))
        object.__setattr__(self, "weight", _finite("weight", self.weight, minimum=0.0, maximum=1.0))
        object.__setattr__(self, "source", _nonempty("source", self.source))

    @property
    def duration_ms(self) -> float:
        return self.close_ms + self.hold_ms + self.open_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "close_ms": self.close_ms,
            "hold_ms": self.hold_ms,
            "open_ms": self.open_ms,
            "duration_ms": self.duration_ms,
            "weight": self.weight,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class HeadMotionCue:
    yaw_delta_deg: float
    pitch_delta_deg: float
    roll_delta_deg: float
    duration_ms: float
    blend_ms: float = 180.0
    source: str = "idle_micro_motion"

    def __post_init__(self) -> None:
        for name in ("yaw_delta_deg", "pitch_delta_deg", "roll_delta_deg"):
            object.__setattr__(self, name, _finite(name, getattr(self, name), minimum=-30.0, maximum=30.0))
        object.__setattr__(self, "duration_ms", _finite("duration_ms", self.duration_ms, minimum=0.0))
        object.__setattr__(self, "blend_ms", _finite("blend_ms", self.blend_ms, minimum=0.0))
        object.__setattr__(self, "source", _nonempty("source", self.source))

    def to_dict(self) -> dict[str, Any]:
        return {
            "yaw_delta_deg": self.yaw_delta_deg,
            "pitch_delta_deg": self.pitch_delta_deg,
            "roll_delta_deg": self.roll_delta_deg,
            "duration_ms": self.duration_ms,
            "blend_ms": self.blend_ms,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class IdleMotionCue:
    motion: str = "breathing"
    duration_ms: float = 5000.0
    intensity: float = 0.25
    loop: bool = True
    source: str = "behavior_default"

    def __post_init__(self) -> None:
        object.__setattr__(self, "motion", _nonempty("motion", self.motion))
        object.__setattr__(self, "duration_ms", _finite("duration_ms", self.duration_ms, minimum=0.0))
        object.__setattr__(self, "intensity", _finite("intensity", self.intensity, minimum=0.0, maximum=1.0))
        object.__setattr__(self, "source", _nonempty("source", self.source))

    def to_dict(self) -> dict[str, Any]:
        return {
            "motion": self.motion,
            "duration_ms": self.duration_ms,
            "intensity": self.intensity,
            "loop": self.loop,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class GestureCue:
    gesture: str
    duration_ms: float
    weight: float = 1.0
    priority: int = 100
    group: str = "upper_body"
    blend_in_ms: float = 140.0
    blend_out_ms: float = 180.0
    source: str = "host"

    def __post_init__(self) -> None:
        object.__setattr__(self, "gesture", _nonempty("gesture", self.gesture))
        object.__setattr__(self, "group", _nonempty("group", self.group))
        object.__setattr__(self, "source", _nonempty("source", self.source))
        object.__setattr__(self, "duration_ms", _finite("duration_ms", self.duration_ms, minimum=0.0))
        object.__setattr__(self, "weight", _finite("weight", self.weight, minimum=0.0, maximum=1.0))
        object.__setattr__(self, "blend_in_ms", _finite("blend_in_ms", self.blend_in_ms, minimum=0.0))
        object.__setattr__(self, "blend_out_ms", _finite("blend_out_ms", self.blend_out_ms, minimum=0.0))
        if type(self.priority) is not int:
            raise ValueError("priority must be an integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "gesture": self.gesture,
            "duration_ms": self.duration_ms,
            "weight": self.weight,
            "priority": self.priority,
            "group": self.group,
            "blend_in_ms": self.blend_in_ms,
            "blend_out_ms": self.blend_out_ms,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class GestureRequest:
    gesture: str
    duration_ms: float = 900.0
    weight: float = 1.0
    priority: int = 100
    delay_ms: float = 0.0
    group: str = "upper_body"
    blend_in_ms: float = 140.0
    blend_out_ms: float = 180.0
    source: str = "host"
    allowed_phases: tuple[AvatarBehaviorPhase, ...] = (
        AvatarBehaviorPhase.IDLE,
        AvatarBehaviorPhase.LISTENING,
        AvatarBehaviorPhase.THINKING,
        AvatarBehaviorPhase.SPEAKING,
    )
    turn_scoped: bool = True
    id: str = field(default_factory=lambda: uuid4().hex)

    def __post_init__(self) -> None:
        object.__setattr__(self, "gesture", _nonempty("gesture", self.gesture))
        object.__setattr__(self, "group", _nonempty("group", self.group))
        object.__setattr__(self, "source", _nonempty("source", self.source))
        object.__setattr__(self, "id", _nonempty("id", self.id))
        object.__setattr__(self, "duration_ms", _finite("duration_ms", self.duration_ms, minimum=0.0))
        object.__setattr__(self, "delay_ms", _finite("delay_ms", self.delay_ms, minimum=0.0))
        object.__setattr__(self, "weight", _finite("weight", self.weight, minimum=0.0, maximum=1.0))
        object.__setattr__(self, "blend_in_ms", _finite("blend_in_ms", self.blend_in_ms, minimum=0.0))
        object.__setattr__(self, "blend_out_ms", _finite("blend_out_ms", self.blend_out_ms, minimum=0.0))
        if type(self.priority) is not int:
            raise ValueError("priority must be an integer")
        phases: list[AvatarBehaviorPhase] = []
        for phase in self.allowed_phases:
            phases.append(phase if isinstance(phase, AvatarBehaviorPhase) else AvatarBehaviorPhase(str(phase)))
        if not phases:
            raise ValueError("allowed_phases must not be empty")
        object.__setattr__(self, "allowed_phases", tuple(phases))


@dataclass(frozen=True, slots=True)
class AvatarBehaviorCueBundle:
    sequence: int
    phase: AvatarBehaviorPhase
    elapsed_ms: float
    turn_id: str | None = None
    gaze: GazeCue | None = None
    blink: BlinkCue | None = None
    head_motion: HeadMotionCue | None = None
    idle_motion: IdleMotionCue | None = None
    gestures: tuple[GestureCue, ...] = ()

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("sequence must be >= 0")
        object.__setattr__(self, "elapsed_ms", _finite("elapsed_ms", self.elapsed_ms, minimum=0.0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "phase": self.phase.value,
            "elapsed_ms": self.elapsed_ms,
            "turn_id": self.turn_id,
            "gaze": self.gaze.to_dict() if self.gaze is not None else None,
            "blink": self.blink.to_dict() if self.blink is not None else None,
            "head_motion": self.head_motion.to_dict() if self.head_motion is not None else None,
            "idle_motion": self.idle_motion.to_dict() if self.idle_motion is not None else None,
            "gestures": [item.to_dict() for item in self.gestures],
        }


@dataclass(frozen=True, slots=True)
class AvatarBehaviorConfig:
    default_gaze: GazeTarget = field(default_factory=GazeTarget)
    default_gaze_duration_ms: float = 1500.0
    blink_interval_min_ms: float = 2800.0
    blink_interval_max_ms: float = 5200.0
    blink_close_ms: float = 70.0
    blink_hold_ms: float = 35.0
    blink_open_ms: float = 90.0
    head_interval_min_ms: float = 1800.0
    head_interval_max_ms: float = 4200.0
    head_motion_duration_ms: float = 900.0
    head_max_yaw_deg: float = 3.0
    head_max_pitch_deg: float = 2.0
    head_max_roll_deg: float = 1.0
    idle_motion: str = "breathing"
    idle_motion_duration_ms: float = 5000.0
    idle_motion_refresh_ms: float = 4500.0

    def __post_init__(self) -> None:
        positive = (
            "default_gaze_duration_ms",
            "blink_interval_min_ms",
            "blink_interval_max_ms",
            "head_interval_min_ms",
            "head_interval_max_ms",
            "head_motion_duration_ms",
            "idle_motion_duration_ms",
            "idle_motion_refresh_ms",
        )
        for name in positive:
            object.__setattr__(self, name, _finite(name, getattr(self, name), minimum=0.001))
        for name in ("blink_close_ms", "blink_hold_ms", "blink_open_ms"):
            object.__setattr__(self, name, _finite(name, getattr(self, name), minimum=0.0))
        for name in ("head_max_yaw_deg", "head_max_pitch_deg", "head_max_roll_deg"):
            object.__setattr__(self, name, _finite(name, getattr(self, name), minimum=0.0, maximum=30.0))
        if self.blink_interval_min_ms > self.blink_interval_max_ms:
            raise ValueError("blink interval min must be <= max")
        if self.head_interval_min_ms > self.head_interval_max_ms:
            raise ValueError("head interval min must be <= max")
        object.__setattr__(self, "idle_motion", _nonempty("idle_motion", self.idle_motion))


@dataclass(slots=True)
class _ScheduledGaze:
    request: GazeRequest
    start_ms: float
    end_ms: float
    order: int


@dataclass(slots=True)
class _ScheduledGesture:
    request: GestureRequest
    start_ms: float
    end_ms: float
    order: int


class AvatarBehaviorRuntime:
    """Session-scoped natural behavior scheduler for an avatar.

    The runtime owns *presentation timing*, not character truth. It never writes
    CharacterState, Memory, or conversation history. Hosts may poll it at render
    cadence, while LiveCharacterOrchestrator also polls opportunistically during
    idle, generation, and playback.
    """

    def __init__(
        self,
        config: AvatarBehaviorConfig | None = None,
        *,
        rng: random.Random | None = None,
    ) -> None:
        self.config = config or AvatarBehaviorConfig()
        self.rng = rng or random.Random()
        self._active = False
        self._started_ms = 0.0
        self._last_now_ms = 0.0
        self._phase = AvatarBehaviorPhase.IDLE
        self._phase_changed = True
        self._turn_id: str | None = None
        self._sequence = 0
        self._next_blink_ms = 0.0
        self._next_head_ms = 0.0
        self._next_idle_ms = 0.0
        self._scheduled_gaze: list[_ScheduledGaze] = []
        self._scheduled_gestures: list[_ScheduledGesture] = []
        self._order = 0
        self._last_gaze_signature: tuple[Any, ...] | None = None

    @property
    def active(self) -> bool:
        return self._active

    @property
    def phase(self) -> AvatarBehaviorPhase:
        return self._phase

    @property
    def turn_id(self) -> str | None:
        return self._turn_id

    @property
    def elapsed_ms(self) -> float:
        return max(0.0, self._last_now_ms - self._started_ms) if self._active else 0.0

    def _sample(self, low: float, high: float) -> float:
        if low == high:
            return low
        return self.rng.uniform(low, high)

    def _validate_now(self, now_ms: float) -> float:
        now = _finite("now_ms", now_ms, minimum=0.0)
        if self._active and now < self._last_now_ms:
            raise ValueError("now_ms must be monotonic")
        return now

    def start(self, *, now_ms: float = 0.0) -> None:
        now = _finite("now_ms", now_ms, minimum=0.0)
        self._active = True
        self._started_ms = now
        self._last_now_ms = now
        self._phase = AvatarBehaviorPhase.IDLE
        self._phase_changed = True
        self._turn_id = None
        self._sequence = 0
        self._scheduled_gaze.clear()
        self._scheduled_gestures.clear()
        self._order = 0
        self._last_gaze_signature = None
        self._next_blink_ms = now + self._sample(self.config.blink_interval_min_ms, self.config.blink_interval_max_ms)
        self._next_head_ms = now + self._sample(self.config.head_interval_min_ms, self.config.head_interval_max_ms)
        self._next_idle_ms = now

    def stop(self, *, now_ms: float | None = None, reason: str = "host") -> dict[str, Any]:
        if not self._active:
            return {"active": False, "reason": reason, "elapsed_ms": 0.0}
        now = self._last_now_ms if now_ms is None else self._validate_now(now_ms)
        elapsed = max(0.0, now - self._started_ms)
        turn_id = self._turn_id
        self._active = False
        self._turn_id = None
        self._scheduled_gaze.clear()
        self._scheduled_gestures.clear()
        self._last_gaze_signature = None
        return {
            "active": False,
            "reason": reason,
            "elapsed_ms": elapsed,
            "turn_id": turn_id,
            "reset": ("gaze", "blink", "head", "idle_motion", "gestures"),
        }

    def begin_turn(self, turn_id: str, *, now_ms: float) -> None:
        if not self._active:
            self.start(now_ms=now_ms)
        now = self._validate_now(now_ms)
        self._last_now_ms = now
        self._turn_id = _nonempty("turn_id", turn_id)
        self.set_phase(AvatarBehaviorPhase.THINKING, now_ms=now)

    def end_turn(self, *, now_ms: float, reason: str = "completed") -> None:
        if not self._active:
            return
        now = self._validate_now(now_ms)
        self._last_now_ms = now
        self._scheduled_gaze = [item for item in self._scheduled_gaze if not item.request.turn_scoped]
        self._scheduled_gestures = [item for item in self._scheduled_gestures if not item.request.turn_scoped]
        self._turn_id = None
        self.set_phase(AvatarBehaviorPhase.IDLE, now_ms=now)
        if reason == "interrupted":
            self._last_gaze_signature = None

    def interrupt_turn(self, *, now_ms: float) -> None:
        if not self._active:
            return
        now = self._validate_now(now_ms)
        self._last_now_ms = now
        self._scheduled_gestures = [item for item in self._scheduled_gestures if not item.request.turn_scoped]
        self._scheduled_gaze = [item for item in self._scheduled_gaze if not item.request.turn_scoped]
        self.set_phase(AvatarBehaviorPhase.INTERRUPTED, now_ms=now)

    def set_phase(self, phase: AvatarBehaviorPhase | str, *, now_ms: float) -> None:
        if not self._active:
            self.start(now_ms=now_ms)
        now = self._validate_now(now_ms)
        self._last_now_ms = now
        resolved = phase if isinstance(phase, AvatarBehaviorPhase) else AvatarBehaviorPhase(str(phase))
        if resolved is not self._phase:
            self._phase = resolved
            self._phase_changed = True
            self._next_idle_ms = min(self._next_idle_ms, now)

    def schedule_gaze(self, request: GazeRequest, *, now_ms: float) -> str:
        if not self._active:
            self.start(now_ms=now_ms)
        now = self._validate_now(now_ms)
        self._last_now_ms = now
        start = now + request.delay_ms
        self._scheduled_gaze.append(_ScheduledGaze(request, start, start + request.duration_ms, self._order))
        self._order += 1
        return request.id

    def schedule_gesture(self, request: GestureRequest, *, now_ms: float) -> str:
        if not self._active:
            self.start(now_ms=now_ms)
        now = self._validate_now(now_ms)
        self._last_now_ms = now
        start = now + request.delay_ms
        self._scheduled_gestures.append(_ScheduledGesture(request, start, start + request.duration_ms, self._order))
        self._order += 1
        return request.id

    @staticmethod
    def _gaze_signature(cue: GazeCue) -> tuple[Any, ...]:
        return (
            cue.target.name,
            round(cue.target.yaw_deg, 4),
            round(cue.target.pitch_deg, 4),
            round(cue.eye_weight, 4),
            round(cue.head_weight, 4),
            cue.priority,
            cue.source,
        )

    def _resolve_gaze(self, now: float, *, force: bool) -> GazeCue | None:
        kept: list[_ScheduledGaze] = []
        candidates: list[_ScheduledGaze] = []
        for item in self._scheduled_gaze:
            if item.end_ms > now:
                kept.append(item)
            if item.start_ms <= now < item.end_ms:
                candidates.append(item)
        self._scheduled_gaze = kept
        if candidates:
            winner = max(candidates, key=lambda item: (item.request.priority, item.order))
            request = winner.request
            cue = GazeCue(
                request.target,
                max(0.0, winner.end_ms - now),
                eye_weight=request.eye_weight,
                head_weight=request.head_weight,
                blend_ms=request.blend_ms,
                priority=request.priority,
                source=request.source,
            )
        else:
            cue = GazeCue(
                self.config.default_gaze,
                self.config.default_gaze_duration_ms,
                source="behavior_default",
            )
        signature = self._gaze_signature(cue)
        if force or signature != self._last_gaze_signature:
            self._last_gaze_signature = signature
            return cue
        return None

    def _resolve_gestures(self, now: float) -> tuple[GestureCue, ...]:
        due: list[_ScheduledGesture] = []
        kept: list[_ScheduledGesture] = []
        for item in self._scheduled_gestures:
            if item.end_ms <= now:
                continue
            if item.start_ms <= now:
                if self._phase in item.request.allowed_phases:
                    due.append(item)
                else:
                    kept.append(item)
            else:
                kept.append(item)

        winners: dict[str, _ScheduledGesture] = {}
        for item in due:
            current = winners.get(item.request.group)
            if current is None or (item.request.priority, item.order) > (current.request.priority, current.order):
                winners[item.request.group] = item

        # One-shot gesture events: once a group's due window is arbitrated, all
        # due requests in that group are consumed so a lower-priority request
        # cannot unexpectedly fire after the winner.
        due_groups = {item.request.group for item in due}
        self._scheduled_gestures = [item for item in kept if item.request.group not in due_groups or item.start_ms > now]

        cues: list[GestureCue] = []
        for group in sorted(winners):
            item = winners[group]
            request = item.request
            cues.append(
                GestureCue(
                    request.gesture,
                    min(request.duration_ms, max(0.0, item.end_ms - now)),
                    weight=request.weight,
                    priority=request.priority,
                    group=request.group,
                    blend_in_ms=request.blend_in_ms,
                    blend_out_ms=request.blend_out_ms,
                    source=request.source,
                )
            )
        return tuple(cues)

    def _phase_motion_scale(self) -> float:
        return {
            AvatarBehaviorPhase.IDLE: 1.0,
            AvatarBehaviorPhase.LISTENING: 0.45,
            AvatarBehaviorPhase.THINKING: 0.65,
            AvatarBehaviorPhase.SPEAKING: 0.55,
            AvatarBehaviorPhase.INTERRUPTED: 0.30,
        }[self._phase]

    def _idle_intensity(self) -> float:
        return {
            AvatarBehaviorPhase.IDLE: 0.25,
            AvatarBehaviorPhase.LISTENING: 0.12,
            AvatarBehaviorPhase.THINKING: 0.18,
            AvatarBehaviorPhase.SPEAKING: 0.14,
            AvatarBehaviorPhase.INTERRUPTED: 0.10,
        }[self._phase]

    def tick(self, *, now_ms: float, force: bool = False) -> AvatarBehaviorCueBundle | None:
        if not self._active:
            self.start(now_ms=now_ms)
        now = self._validate_now(now_ms)
        self._last_now_ms = now
        phase_changed = self._phase_changed

        gaze = self._resolve_gaze(now, force=force or phase_changed)

        blink: BlinkCue | None = None
        if now >= self._next_blink_ms:
            blink = BlinkCue(
                close_ms=self.config.blink_close_ms,
                hold_ms=self.config.blink_hold_ms,
                open_ms=self.config.blink_open_ms,
            )
            self._next_blink_ms = now + self._sample(self.config.blink_interval_min_ms, self.config.blink_interval_max_ms)

        head: HeadMotionCue | None = None
        if now >= self._next_head_ms:
            scale = self._phase_motion_scale()
            head = HeadMotionCue(
                self.rng.uniform(-self.config.head_max_yaw_deg, self.config.head_max_yaw_deg) * scale,
                self.rng.uniform(-self.config.head_max_pitch_deg, self.config.head_max_pitch_deg) * scale,
                self.rng.uniform(-self.config.head_max_roll_deg, self.config.head_max_roll_deg) * scale,
                self.config.head_motion_duration_ms,
            )
            self._next_head_ms = now + self._sample(self.config.head_interval_min_ms, self.config.head_interval_max_ms)

        idle: IdleMotionCue | None = None
        if force or phase_changed or now >= self._next_idle_ms:
            idle = IdleMotionCue(
                self.config.idle_motion,
                self.config.idle_motion_duration_ms,
                intensity=self._idle_intensity(),
            )
            self._next_idle_ms = now + self.config.idle_motion_refresh_ms

        gestures = self._resolve_gestures(now)
        self._phase_changed = False

        if gaze is None and blink is None and head is None and idle is None and not gestures:
            return None

        base = AvatarBehaviorCueBundle(
            self._sequence,
            self._phase,
            max(0.0, now - self._started_ms),
            turn_id=self._turn_id,
            gaze=gaze,
            blink=blink,
            head_motion=head,
            idle_motion=idle,
            gestures=gestures,
        )
        self._sequence += 1
        return base
