from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any

def _finite_nonnegative(name: str, value: float) -> float:
    value=float(value)
    if not math.isfinite(value) or value < 0: raise ValueError(f"{name} must be finite and >= 0")
    return value

def _weight(name: str, value: float) -> float:
    value=float(value)
    if not math.isfinite(value) or not 0 <= value <= 1: raise ValueError(f"{name} must be between 0 and 1")
    return value

@dataclass(frozen=True, slots=True)
class VRMExpressionKeyframe:
    expression: str
    start_offset_ms: float
    duration_ms: float
    weight: float
    channel: str
    source: str
    def __post_init__(self):
        if not self.expression.strip() or not self.channel.strip() or not self.source.strip():
            raise ValueError("expression, channel and source must not be empty")
        object.__setattr__(self, "start_offset_ms", _finite_nonnegative("start_offset_ms", self.start_offset_ms))
        object.__setattr__(self, "duration_ms", _finite_nonnegative("duration_ms", self.duration_ms))
        object.__setattr__(self, "weight", _weight("weight", self.weight))
    def to_dict(self) -> dict[str, Any]:
        return {"expression": self.expression, "start_offset_ms": self.start_offset_ms, "duration_ms": self.duration_ms, "weight": self.weight, "channel": self.channel, "source": self.source}
