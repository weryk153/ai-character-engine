from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import uuid4

from .models import AdmissionResult, AdmissionStatus, ProactiveCandidate, RetryState, _aware
from .policy import AutonomyPolicy


@dataclass
class _Entry:
    candidate: ProactiveCandidate
    sequence: int
    retry: RetryState = field(default_factory=RetryState)


@dataclass(frozen=True, eq=False)
class DispatchClaim:
    """Identity-checked reservation, valid only until ack or release."""
    candidate: ProactiveCandidate
    token: str = field(default_factory=lambda: uuid4().hex)


class AutonomyScheduler:
    """Bounded queue with one in-flight claim, confined to one asyncio loop.

    Synchronous transitions never yield. Thread/process/distributed safety is
    intentionally outside this in-memory implementation's contract.
    """

    def __init__(self, policy: AutonomyPolicy | None = None, *, clock: Callable[[], datetime] | None = None):
        if clock is not None and not callable(clock):
            raise ValueError("clock must be callable")
        self.policy = policy or AutonomyPolicy()
        self._clock = clock or (lambda: datetime.now(UTC))
        self._entries: dict[str, _Entry] = {}
        self._sequence = 0
        self._claim: DispatchClaim | None = None
        self._claimed_entry: _Entry | None = None
        self._last_dispatch: datetime | None = None
        self._last_by_key: dict[str, datetime] = {}

    def time(self, now: datetime | None = None) -> datetime:
        value = self._clock() if now is None else now
        _aware(value, "now")
        return value

    @property
    def busy(self) -> bool:
        return self._claim is not None

    @property
    def pending(self) -> tuple[ProactiveCandidate, ...]:
        return tuple(item.candidate for item in sorted(self._entries.values(), key=lambda item: item.sequence))

    @property
    def in_flight(self) -> ProactiveCandidate | None:
        return self._claim.candidate if self._claim else None

    @property
    def held(self) -> tuple[ProactiveCandidate, ...]:
        return tuple(item.candidate for item in self._entries.values() if item.retry.held_reason)

    def retry_state(self, candidate_id: str) -> RetryState:
        if self._claim and self._claim.candidate.id == candidate_id:
            return self._claimed_entry.retry
        return self._entries[candidate_id].retry

    def _prune(self, now: datetime):
        self._entries = {
            key: entry for key, entry in self._entries.items()
            if entry.retry.held_reason or entry.candidate.expires_at is None or entry.candidate.expires_at > now
        }
        self._last_by_key = {
            key: stamp for key, stamp in self._last_by_key.items()
            if now - stamp < self.policy.per_key_cooldown
        }

    def submit(self, candidate: ProactiveCandidate, *, now: datetime | None = None) -> AdmissionResult:
        now = self.time(now)
        self._prune(now)
        reason = self.policy.admission_reason(candidate)
        if candidate.expires_at is not None and candidate.expires_at <= now:
            reason = "expired"
        if reason:
            return AdmissionResult(AdmissionStatus.REJECTED, candidate.id, reason)
        if candidate.id in self._entries or (self._claim and candidate.id == self._claim.candidate.id):
            return AdmissionResult(AdmissionStatus.REJECTED, candidate.id, "duplicate_id")
        if candidate.dedupe_key and self._claim and candidate.dedupe_key == self._claim.candidate.dedupe_key:
            return AdmissionResult(AdmissionStatus.REJECTED, candidate.id, "duplicate_in_flight")
        replaced = None
        reason = "accepted"
        for entry in self._entries.values():
            current = entry.candidate
            if candidate.dedupe_key and current.dedupe_key == candidate.dedupe_key:
                if entry.retry.held_reason or candidate.priority <= current.priority:
                    return AdmissionResult(AdmissionStatus.REJECTED, candidate.id, "duplicate_held" if entry.retry.held_reason else "duplicate_pending")
                replaced, reason = current, "higher_priority_duplicate"
                break
        if replaced is None and len(self._entries) + int(self.busy) >= self.policy.max_pending:
            choices = [entry for entry in self._entries.values() if not entry.retry.held_reason]
            weakest = min(choices, key=lambda item: (item.candidate.priority, -item.sequence)) if choices else None
            if weakest is None or candidate.priority <= weakest.candidate.priority:
                return AdmissionResult(AdmissionStatus.REJECTED, candidate.id, "queue_full")
            replaced, reason = weakest.candidate, "queue_pressure"
        if replaced:
            del self._entries[replaced.id]
        self._sequence += 1
        self._entries[candidate.id] = _Entry(candidate, self._sequence)
        return AdmissionResult(AdmissionStatus.REPLACED if replaced else AdmissionStatus.ACCEPTED, candidate.id, reason, replaced.id if replaced else None)

    def next_ready(self, now: datetime | None = None) -> ProactiveCandidate | None:
        now = self.time(now)
        self._prune(now)
        if self.busy or (self.policy.quiet_hours and self.policy.quiet_hours.contains(now)):
            return None
        if self._last_dispatch is not None and now - self._last_dispatch < self.policy.global_cooldown:
            return None
        ready = []
        for entry in self._entries.values():
            item = entry.candidate
            if entry.retry.held_reason or (entry.retry.retry_at and entry.retry.retry_at > now):
                continue
            if item.not_before and item.not_before > now:
                continue
            key = item.cooldown_key
            if key and key in self._last_by_key:
                continue
            if key and len(self._last_by_key) >= self.policy.max_cooldown_keys:
                continue
            ready.append(entry)
        if not ready:
            return None
        return min(ready, key=lambda entry: (-entry.candidate.priority, entry.sequence)).candidate

    def claim_next(self, now: datetime | None = None) -> DispatchClaim | None:
        candidate = self.next_ready(now)
        if candidate is None:
            return None
        self._claimed_entry = self._entries.pop(candidate.id)
        self._claim = DispatchClaim(candidate)
        return self._claim

    def _check_claim(self, claim: DispatchClaim):
        if claim is not self._claim:
            raise ValueError("claim is not active for this scheduler")

    def _record_completion(self, candidate, now):
        self._prune(now)
        key = candidate.cooldown_key
        if key and key not in self._last_by_key and len(self._last_by_key) >= self.policy.max_cooldown_keys:
            raise ValueError("cooldown key capacity exhausted")
        self._last_dispatch = now
        if key and self.policy.per_key_cooldown.total_seconds() > 0:
            self._last_by_key[key] = now

    def ack(self, claim: DispatchClaim, *, now: datetime | None = None):
        now = self.time(now)
        self._check_claim(claim)
        self._record_completion(claim.candidate, now)
        self._claim = self._claimed_entry = None

    def release(self, claim: DispatchClaim, *, failed=False, requires_review=False, now: datetime | None = None):
        now = self.time(now)
        self._check_claim(claim)
        entry = self._claimed_entry
        attempts = entry.retry.attempts + int(failed)
        held = "partial_execution" if requires_review else "attempts_exhausted" if attempts >= self.policy.max_attempts else ""
        factor = 2 ** max(0, attempts - 1)
        base, cap = self.policy.retry_backoff, self.policy.max_retry_backoff
        delay = cap if base > cap / factor else base * factor
        retry_at = now + delay if failed and not held else None
        entry.retry = RetryState(attempts, retry_at, held)
        self._entries[entry.candidate.id] = entry
        self._claim = self._claimed_entry = None

    def resume(self, candidate_id: str):
        """Host has reconciled effects and explicitly authorizes another attempt."""
        entry = self._entries[candidate_id]
        if not entry.retry.held_reason:
            raise ValueError("candidate is not held")
        entry.retry = RetryState()

    def mark_dispatched(self, candidate: ProactiveCandidate, now: datetime | None = None):
        """Compatibility for synchronous hosts; controllers must use claim/ack."""
        now = self.time(now)
        if self.busy:
            raise ValueError("use ack for an in-flight dispatch")
        if candidate.id not in self._entries or self._entries[candidate.id].candidate is not candidate:
            raise ValueError("candidate is not pending")
        self._record_completion(candidate, now)
        self._entries.pop(candidate.id, None)

    def discard(self, candidate: ProactiveCandidate):
        if self._claim and self._claim.candidate.id == candidate.id:
            raise ValueError("cannot discard an in-flight candidate")
        if candidate.id not in self._entries:
            raise ValueError("candidate is not pending")
        del self._entries[candidate.id]
