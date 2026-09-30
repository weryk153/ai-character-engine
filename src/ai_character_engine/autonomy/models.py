from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from types import MappingProxyType
from uuid import uuid4


def _aware(value: datetime, name: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")


def _integer(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")


def _text(value, name, limit):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{name} must contain 1 to {limit} characters")


def _json_value(value, budget, depth=0):
    budget[0] -= 1
    if depth > 12 or budget[0] < 0:
        raise ValueError("payload exceeds structure limits")
    if value is None or type(value) in (bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, str) and len(value) <= 16384:
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) or len(key) > 256 for key in value):
            raise ValueError("payload keys must be strings of at most 256 characters")
        return {key: _json_value(item, budget, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item, budget, depth + 1) for item in value]
    raise ValueError("payload must contain bounded finite JSON values, not media bytes")


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class ProactiveCandidate:
    """A host observation that may become one proactive character turn."""

    content: str
    source: str
    event_type: str = "proactive_observation"
    priority: int = 50
    dedupe_key: str | None = None
    cooldown_key: str | None = None
    not_before: datetime | None = None
    expires_at: datetime | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid4().hex)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    _payload_json: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _text(self.content, "content", 8192)
        for name in ("source", "event_type", "id"):
            _text(getattr(self, name), name, 256)
        for name in ("dedupe_key", "cooldown_key"):
            if getattr(self, name) is not None:
                _text(getattr(self, name), name, 256)
        _integer(self.priority, "priority", 0, 100)
        _aware(self.created_at, "created_at")
        if self.not_before is not None:
            _aware(self.not_before, "not_before")
        if self.expires_at is not None:
            _aware(self.expires_at, "expires_at")
        if self.expires_at is not None and self.expires_at <= self.created_at:
            raise ValueError("expires_at must be later than created_at")
        if self.expires_at is not None and self.not_before is not None and self.expires_at <= self.not_before:
            raise ValueError("expires_at must be later than not_before")
        if not isinstance(self.payload, Mapping):
            raise ValueError("payload must be a JSON object")
        value = _json_value(self.payload, [4096])
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        if len(encoded.encode("utf-8")) > 16384:
            raise ValueError("payload must be at most 16384 UTF-8 bytes")
        if "memory_importance" in value:
            importance = value["memory_importance"]
            if type(importance) not in (int, float) or not 0 <= importance <= 1:
                raise ValueError("memory_importance must be a number between 0 and 1")
        object.__setattr__(self, "_payload_json", encoded)
        object.__setattr__(self, "payload", _freeze(value))

    def event_payload(self) -> dict[str, Any]:
        """Return an independent mutable JSON copy for the runtime event."""
        return json.loads(self._payload_json)


class AdmissionStatus(StrEnum):
    ACCEPTED = "accepted"
    REPLACED = "replaced"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class AdmissionResult:
    status: AdmissionStatus
    candidate_id: str
    reason: str
    replaced_candidate_id: str | None = None


class DispatchStatus(StrEnum):
    DELIVERED = "delivered"
    EMPTY = "empty"
    BLOCKED = "blocked"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class DispatchResult:
    status: DispatchStatus
    candidate: ProactiveCandidate | None = None
    run_result: Any | None = None
    reason: str = ""
    requires_review: bool = False


@dataclass(frozen=True, slots=True)
class RetryState:
    attempts: int = 0
    retry_at: datetime | None = None
    held_reason: str = ""
