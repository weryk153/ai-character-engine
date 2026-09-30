from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone, tzinfo

from .models import ProactiveCandidate, _aware, _integer


@dataclass(frozen=True, slots=True)
class QuietHours:
    """A daily local-time interval. Equal endpoints mean quiet all day."""

    start: time
    end: time
    tz: tzinfo = timezone.utc

    def __post_init__(self):
        if not isinstance(self.tz, tzinfo):
            raise ValueError("tz must be a timezone")
        if any(not isinstance(value, time) or value.tzinfo is not None for value in (self.start, self.end)):
            raise ValueError("quiet-hours endpoints must be local times without tzinfo; use tz")

    def contains(self, moment: datetime) -> bool:
        _aware(moment, "moment")
        local = moment.astimezone(self.tz).time().replace(tzinfo=None)
        start = self.start.replace(tzinfo=None)
        end = self.end.replace(tzinfo=None)
        if start == end:
            return True
        if start < end:
            return start <= local < end
        return local >= start or local < end


@dataclass(frozen=True, slots=True)
class AutonomyPolicy:
    min_priority: int = 1
    global_cooldown: timedelta = timedelta(seconds=30)
    per_key_cooldown: timedelta = timedelta(minutes=5)
    max_pending: int = 100
    quiet_hours: QuietHours | None = None
    allowed_event_types: frozenset[str] = frozenset({"proactive_observation"})
    max_cooldown_keys: int = 1000
    max_attempts: int = 3
    retry_backoff: timedelta = timedelta(seconds=1)
    max_retry_backoff: timedelta = timedelta(minutes=1)

    def __post_init__(self) -> None:
        _integer(self.min_priority, "min_priority", 0, 100)
        _integer(self.max_pending, "max_pending", 1, 100000)
        _integer(self.max_cooldown_keys, "max_cooldown_keys", 1, 100000)
        _integer(self.max_attempts, "max_attempts", 1, 100)
        if self.quiet_hours is not None and not isinstance(self.quiet_hours, QuietHours):
            raise ValueError("quiet_hours must be QuietHours or None")
        for value in (self.global_cooldown, self.per_key_cooldown, self.retry_backoff, self.max_retry_backoff):
            if not isinstance(value, timedelta) or value < timedelta(0):
                raise ValueError("cooldowns and backoff must be non-negative timedeltas")
        if self.max_retry_backoff < self.retry_backoff:
            raise ValueError("max_retry_backoff must be at least retry_backoff")
        if isinstance(self.allowed_event_types, str) or not self.allowed_event_types or any(
            not isinstance(value, str) or not value.strip() for value in self.allowed_event_types
        ):
            raise ValueError("allowed_event_types must not be empty")
        object.__setattr__(self, "allowed_event_types", frozenset(self.allowed_event_types))

    def admission_reason(self, candidate: ProactiveCandidate) -> str | None:
        if candidate.priority < self.min_priority:
            return "below_min_priority"
        if candidate.event_type not in self.allowed_event_types:
            return "event_type_not_allowed"
        return None
