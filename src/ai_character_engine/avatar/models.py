from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from uuid import uuid4


def _finite_nonnegative(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and >= 0")
    return value


def _weight(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1")
    return value


class Viseme(StrEnum):
    NEUTRAL = "neutral"
    A = "a"
    I = "i"
    U = "u"
    E = "e"
    O = "o"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class AudioEnvelopePoint:
    start_offset_ms: float
    duration_ms: float
    amplitude: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "start_offset_ms", _finite_nonnegative("start_offset_ms", self.start_offset_ms))
        object.__setattr__(self, "duration_ms", _finite_nonnegative("duration_ms", self.duration_ms))
        object.__setattr__(self, "amplitude", _weight("amplitude", self.amplitude))

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_offset_ms": self.start_offset_ms,
            "duration_ms": self.duration_ms,
            "amplitude": self.amplitude,
        }


@dataclass(frozen=True, slots=True)
class VisemeCue:
    viseme: Viseme | str
    start_offset_ms: float
    duration_ms: float
    weight: float = 1.0
    source: str = "provider"

    def __post_init__(self) -> None:
        label = self.viseme.value if isinstance(self.viseme, Viseme) else str(self.viseme).strip().lower()
        if not label:
            raise ValueError("viseme must not be empty")
        try:
            label = Viseme(label)
        except ValueError:
            pass
        object.__setattr__(self, "viseme", label)
        object.__setattr__(self, "start_offset_ms", _finite_nonnegative("start_offset_ms", self.start_offset_ms))
        object.__setattr__(self, "duration_ms", _finite_nonnegative("duration_ms", self.duration_ms))
        object.__setattr__(self, "weight", _weight("weight", self.weight))
        if not self.source.strip():
            raise ValueError("source must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "viseme": self.viseme.value if isinstance(self.viseme, Viseme) else self.viseme,
            "start_offset_ms": self.start_offset_ms,
            "duration_ms": self.duration_ms,
            "weight": self.weight,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class ExpressionCue:
    expression: str
    start_offset_ms: float
    duration_ms: float
    weight: float = 1.0
    priority: int = 0
    fade_in_ms: float = 80.0
    fade_out_ms: float = 120.0
    source: str = "state"
    group: str = "face"

    def __post_init__(self) -> None:
        if not self.expression.strip():
            raise ValueError("expression must not be empty")
        if type(self.priority) is not int:
            raise ValueError("priority must be an integer")
        object.__setattr__(self, "start_offset_ms", _finite_nonnegative("start_offset_ms", self.start_offset_ms))
        object.__setattr__(self, "duration_ms", _finite_nonnegative("duration_ms", self.duration_ms))
        object.__setattr__(self, "weight", _weight("weight", self.weight))
        object.__setattr__(self, "fade_in_ms", _finite_nonnegative("fade_in_ms", self.fade_in_ms))
        object.__setattr__(self, "fade_out_ms", _finite_nonnegative("fade_out_ms", self.fade_out_ms))
        if not self.group.strip() or not self.source.strip():
            raise ValueError("group and source must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "expression": self.expression,
            "start_offset_ms": self.start_offset_ms,
            "duration_ms": self.duration_ms,
            "weight": self.weight,
            "priority": self.priority,
            "fade_in_ms": self.fade_in_ms,
            "fade_out_ms": self.fade_out_ms,
            "source": self.source,
            "group": self.group,
        }


@dataclass(frozen=True, slots=True)
class ExpressionRequest:
    expression: str
    duration_ms: float
    weight: float = 1.0
    priority: int = 100
    delay_ms: float = 0.0
    fade_in_ms: float = 80.0
    fade_out_ms: float = 120.0
    group: str = "face"
    source: str = "host"
    id: str = field(default_factory=lambda: uuid4().hex)

    def __post_init__(self) -> None:
        if not self.expression.strip():
            raise ValueError("expression must not be empty")
        if type(self.priority) is not int:
            raise ValueError("priority must be an integer")
        object.__setattr__(self, "duration_ms", _finite_nonnegative("duration_ms", self.duration_ms))
        object.__setattr__(self, "delay_ms", _finite_nonnegative("delay_ms", self.delay_ms))
        object.__setattr__(self, "fade_in_ms", _finite_nonnegative("fade_in_ms", self.fade_in_ms))
        object.__setattr__(self, "fade_out_ms", _finite_nonnegative("fade_out_ms", self.fade_out_ms))
        object.__setattr__(self, "weight", _weight("weight", self.weight))
        if not self.group.strip() or not self.source.strip():
            raise ValueError("group and source must not be empty")


@dataclass(frozen=True, slots=True)
class AvatarCueBundle:
    turn_id: str
    segment_sequence: int
    chunk_index: int
    start_offset_ms: float
    duration_ms: float | None
    envelope: tuple[AudioEnvelopePoint, ...] = ()
    visemes: tuple[VisemeCue, ...] = ()
    expressions: tuple[ExpressionCue, ...] = ()
    timing_source: str = "tts_audio"

    def __post_init__(self) -> None:
        if not self.turn_id.strip():
            raise ValueError("turn_id must not be empty")
        if self.segment_sequence < 0 or self.chunk_index < 0:
            raise ValueError("sequence values must be >= 0")
        object.__setattr__(self, "start_offset_ms", _finite_nonnegative("start_offset_ms", self.start_offset_ms))
        if self.duration_ms is not None:
            object.__setattr__(self, "duration_ms", _finite_nonnegative("duration_ms", self.duration_ms))
        if not self.timing_source.strip():
            raise ValueError("timing_source must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "segment_sequence": self.segment_sequence,
            "chunk_index": self.chunk_index,
            "start_offset_ms": self.start_offset_ms,
            "duration_ms": self.duration_ms,
            "timing_source": self.timing_source,
            "envelope": [item.to_dict() for item in self.envelope],
            "visemes": [item.to_dict() for item in self.visemes],
            "expressions": [item.to_dict() for item in self.expressions],
        }
