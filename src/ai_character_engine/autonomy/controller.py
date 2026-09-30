from __future__ import annotations

import asyncio
import math
from datetime import datetime

from ai_character_engine.events import CharacterEvent
from ai_character_engine.runtime import CharacterRuntime
from ai_character_engine.runtime.coordination import PartialTurnError, RuntimeBusyError

from .models import DispatchResult, DispatchStatus
from .scheduler import AutonomyScheduler


class AutonomyController:
    """One host-driven attempt, with a deadline and conservative retry handling."""

    def __init__(self, runtime: CharacterRuntime, scheduler: AutonomyScheduler, *, turn_timeout_seconds: float = 90.0):
        if type(turn_timeout_seconds) not in (int, float) or not math.isfinite(turn_timeout_seconds) or turn_timeout_seconds <= 0:
            raise ValueError("turn_timeout_seconds must be finite and positive")
        self.runtime = runtime
        self.scheduler = scheduler
        self.turn_timeout_seconds = turn_timeout_seconds
        self._lock = asyncio.Lock()

    async def run_once(self, *, now: datetime | None = None) -> DispatchResult:
        now = self.scheduler.time(now)
        if self._lock.locked() or self.scheduler.busy:
            return DispatchResult(DispatchStatus.BLOCKED, reason="controller_busy" if self._lock.locked() else "scheduler_busy")
        if getattr(getattr(self.runtime, "turns", None), "busy", False):
            return DispatchResult(DispatchStatus.BLOCKED, reason="runtime_busy")
        async with self._lock:
            claim = self.scheduler.claim_next(now)
            if claim is None:
                return DispatchResult(DispatchStatus.EMPTY, reason="no_ready_candidate")
            candidate = claim.candidate
            payload = candidate.event_payload()
            payload["autonomy"] = {"candidate_id": candidate.id, "priority": candidate.priority, "dedupe_key": candidate.dedupe_key}
            payload.setdefault("memory_importance", 0.0)
            event = CharacterEvent(type=candidate.event_type, source=candidate.source, content=candidate.content, payload=payload, id=candidate.id, created_at=candidate.created_at)
            cancelled_with_effects = False

            async def invoke():
                nonlocal cancelled_with_effects
                try:
                    return await self.runtime.process_event(event)
                except asyncio.CancelledError as exc:
                    cancelled_with_effects = getattr(exc, "requires_review", False)
                    raise

            try:
                result = await asyncio.wait_for(invoke(), self.turn_timeout_seconds)
            except RuntimeBusyError:
                self.scheduler.release(claim)
                return DispatchResult(DispatchStatus.BLOCKED, candidate, reason="runtime_busy")
            except asyncio.CancelledError as exc:
                self.scheduler.release(claim, requires_review=cancelled_with_effects or getattr(exc, "requires_review", False))
                raise
            except Exception as exc:
                review = cancelled_with_effects or isinstance(exc, PartialTurnError) or getattr(exc.__cause__, "requires_review", False)
                self.scheduler.release(claim, failed=True, requires_review=review)
                state = self.scheduler.retry_state(candidate.id)
                reason = state.held_reason or ("runtime_timeout" if isinstance(exc, TimeoutError) else "runtime_failed")
                return DispatchResult(DispatchStatus.FAILED, candidate, reason=reason, requires_review=bool(state.held_reason))
            # Completion time always comes from the clock, not the tick snapshot.
            self.scheduler.ack(claim)
            return DispatchResult(DispatchStatus.DELIVERED, candidate, result, "delivered")
