from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable

from ai_character_engine.events.models import CharacterEvent
from ai_character_engine.runtime.character_runtime import CharacterRuntime
from ai_character_engine.runtime.models import CharacterRunResult

ResultHandler = Callable[[CharacterRunResult], Awaitable[None] | None]


class CharacterEventLoop:
    """FIFO event loop for host applications that drive a character.

    The loop does not own I/O integrations. OBS, chat platforms, games, vision
    systems, or timers publish CharacterEvent objects into it. The runtime then
    observes the event, lets the model decide what to do, executes tools, and
    returns a CharacterRunResult to the host.
    """

    def __init__(
        self,
        runtime: CharacterRuntime,
        *,
        max_queue_size: int = 0,
    ) -> None:
        if max_queue_size < 0:
            raise ValueError("max_queue_size must be >= 0")
        self.runtime = runtime
        self._queue: asyncio.Queue[CharacterEvent] = asyncio.Queue(
            maxsize=max_queue_size
        )
        self._stopped = False

    @property
    def pending_count(self) -> int:
        return self._queue.qsize()

    async def publish(self, event: CharacterEvent) -> None:
        if self._stopped:
            raise RuntimeError("character event loop is stopped")
        await self._queue.put(event)

    def publish_nowait(self, event: CharacterEvent) -> None:
        if self._stopped:
            raise RuntimeError("character event loop is stopped")
        self._queue.put_nowait(event)

    async def run_once(self) -> CharacterRunResult:
        if self._stopped and self._queue.empty():
            raise RuntimeError("character event loop is stopped")
        event = await self._queue.get()
        try:
            return await self.runtime.process_event(event)
        finally:
            self._queue.task_done()

    async def run_forever(self, on_result: ResultHandler | None = None) -> None:
        while not self._stopped:
            result = await self.run_once()
            if on_result is not None:
                maybe_awaitable = on_result(result)
                if inspect.isawaitable(maybe_awaitable):
                    await maybe_awaitable

    def stop(self) -> None:
        self._stopped = True
