from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

from .errors import CharacterSchedulerOverloadedError
from .models import FairSchedulerConfig, FairSchedulerSnapshot

T = TypeVar("T")


@dataclass(slots=True)
class _Waiter:
    future: asyncio.Future[None]
    cancelled: bool = False


class FairCharacterScheduler:
    """Bounded round-robin admission across character runtimes.

    At most one operation per character runs at a time. Global concurrency can
    be higher, so different characters may progress in parallel. Waiting work is
    queued per character and characters re-enter the ready rotation only after
    their running operation finishes, preventing one noisy character from
    monopolising every admission slot.
    """

    def __init__(self, config: FairSchedulerConfig | None = None) -> None:
        self.config = config or FairSchedulerConfig()
        self._lock = asyncio.Lock()
        self._queues: dict[str, deque[_Waiter]] = {}
        self._ready: deque[str] = deque()
        self._ready_set: set[str] = set()
        self._active: set[str] = set()
        self._running = 0

    async def run(self, character_id: str, operation: Callable[[], Awaitable[T]]) -> T:
        loop = asyncio.get_running_loop()
        waiter = _Waiter(loop.create_future())
        async with self._lock:
            queue = self._queues.setdefault(character_id, deque())
            total_pending = sum(len(items) for items in self._queues.values())
            free_slots = self.config.max_concurrent_characters - self._running
            can_admit_without_wait = (
                character_id not in self._active
                and character_id not in self._ready_set
                and free_slots > len(self._ready)
            )
            if not can_admit_without_wait:
                if len(queue) >= self.config.max_pending_per_character:
                    raise CharacterSchedulerOverloadedError(
                        f"pending limit reached for character {character_id}"
                    )
                if total_pending >= self.config.max_total_pending:
                    raise CharacterSchedulerOverloadedError("global pending limit reached")
            queue.append(waiter)
            if character_id not in self._active and character_id not in self._ready_set:
                self._ready.append(character_id)
                self._ready_set.add(character_id)
            self._dispatch_locked()

        try:
            await waiter.future
        except BaseException:
            async with self._lock:
                waiter.cancelled = True
                queue = self._queues.get(character_id)
                if queue is not None:
                    try:
                        queue.remove(waiter)
                    except ValueError:
                        pass
                    if not queue and character_id not in self._active:
                        self._queues.pop(character_id, None)
                        self._remove_ready_locked(character_id)
                self._dispatch_locked()
            raise

        try:
            return await operation()
        finally:
            async with self._lock:
                self._running -= 1
                self._active.discard(character_id)
                queue = self._queues.get(character_id)
                if queue:
                    if character_id not in self._ready_set:
                        self._ready.append(character_id)
                        self._ready_set.add(character_id)
                else:
                    self._queues.pop(character_id, None)
                self._dispatch_locked()

    def snapshot(self) -> FairSchedulerSnapshot:
        return FairSchedulerSnapshot(
            running_character_ids=tuple(sorted(self._active)),
            pending_by_character={
                key: len(value) for key, value in sorted(self._queues.items()) if value
            },
        )

    def _dispatch_locked(self) -> None:
        while self._running < self.config.max_concurrent_characters and self._ready:
            character_id = self._ready.popleft()
            self._ready_set.discard(character_id)
            if character_id in self._active:
                continue
            queue = self._queues.get(character_id)
            if not queue:
                continue
            waiter = queue.popleft()
            while waiter.cancelled or waiter.future.cancelled():
                if not queue:
                    waiter = None  # type: ignore[assignment]
                    break
                waiter = queue.popleft()
            if waiter is None:
                continue
            self._active.add(character_id)
            self._running += 1
            if not waiter.future.done():
                waiter.future.set_result(None)

    def _remove_ready_locked(self, character_id: str) -> None:
        if character_id not in self._ready_set:
            return
        self._ready = deque(item for item in self._ready if item != character_id)
        self._ready_set.discard(character_id)
