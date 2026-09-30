"""Who uses the model next, when foreground and background share one.

A host with a single local model runs the character's reply and the background
cognition on the same endpoint. Every rule here was measured on a local 9B
model over eight-turn conversations:

- Background calls overlapping the reply raised the median time to the first
  sentence from 2.6 s to 4.7 s, up to 12 s. So background gives way while the
  character is replying; a call already running is abandoned and redone.
- Emotion analysis has a lane of its own. It decides the mood of the next
  reply and is short. When it queued with the other workers, only 3 turns in 8
  produced a committed observation: the moment a reply ended, work abandoned
  in the previous turn took the model first.
- The other workers use the model one at a time, most urgent first, and only
  after the emotion analysis of the turn that just ended. Alone it took 4.6 s;
  started together with memory extraction, neither was done after 6 s and the
  next turn began with the old mood.
- A job gives way to the foreground once. Goal and reflection calls take longer
  than the pause between two turns; abandoned every time, they never finish.
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
import logging
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)


class ModelAccess:
    def __init__(self, patience_seconds: float) -> None:
        if patience_seconds <= 0:
            raise ValueError("patience_seconds must be > 0")
        self._patience = patience_seconds
        self._talking = 0
        self._generation = 0
        self._quiet = asyncio.Event()
        self._quiet.set()
        self._interrupted = asyncio.Event()
        self._busy = False
        self._waiting: list[tuple[int, int, asyncio.Future[None]]] = []
        self._ticket = itertools.count()
        self._patience_timer: asyncio.TimerHandle | None = None
        self._holds: dict[object, asyncio.TimerHandle] = {}
        self._calm = asyncio.Event()
        self._calm.set()

    @property
    def foreground_active(self) -> bool:
        return self._talking > 0

    async def quiet(self) -> None:
        """Wait until no reply is being generated."""
        await self._quiet.wait()

    def foreground_started(self) -> None:
        self._talking += 1
        self._generation += 1
        self._quiet.clear()
        self._interrupted.set()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        # Timed from here rather than by whoever is waiting: a waiting job
        # times out first and would never notice the missing end signal.
        if self._patience_timer is not None:
            self._patience_timer.cancel()
        self._patience_timer = loop.call_later(
            self._patience, self._give_up_waiting_for, self._generation
        )

    def foreground_finished(self) -> None:
        self._talking = max(0, self._talking - 1)
        if self._talking == 0:
            if self._patience_timer is not None:
                self._patience_timer.cancel()
                self._patience_timer = None
            self._quiet.set()
            self._interrupted.clear()

    def hold(self, key: object, seconds: float) -> None:
        """Keep the shared lane waiting until release(key).

        For a job in a lane of its own that should have the model to itself.
        The hold ends by itself after ``seconds``: a job that never gets to its
        call must not stop everything else.
        """
        self.release(key)
        self._calm.clear()
        self._holds[key] = asyncio.get_running_loop().call_later(seconds, self.release, key)

    def release(self, key: object) -> None:
        timer = self._holds.pop(key, None)
        if timer is not None:
            timer.cancel()
        if not self._holds:
            self._calm.set()

    def _give_up_waiting_for(self, generation: int) -> None:
        if self._talking and generation == self._generation:
            logger.warning(
                "no end-of-reply signal for %.0fs; resuming background work", self._patience
            )
            self._talking = 0
            self.foreground_finished()

    async def _take_turn(self, rank: int) -> None:
        if not self._busy and not self._waiting:
            self._busy = True
            return
        ready: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        entry = (rank, next(self._ticket), ready)
        heapq.heappush(self._waiting, entry)
        try:
            await ready
        except asyncio.CancelledError:
            if ready.done() and not ready.cancelled():
                self._pass_on()  # cancelled just as its turn came: hand it on
            elif entry in self._waiting:
                # It may already be gone: _pass_on discards cancelled entries.
                # Raising anything else here would replace the CancelledError
                # and the task runtime would count the job as merely failed.
                self._waiting.remove(entry)
                heapq.heapify(self._waiting)
            raise

    def _pass_on(self) -> None:
        while self._waiting:
            _, _, ready = heapq.heappop(self._waiting)
            if not ready.done():
                ready.set_result(None)
                return
        self._busy = False

    async def call(
        self,
        rank: int,
        make_call: Callable[[], Awaitable[Any]],
        timeout: float,
        *,
        own_lane: bool = False,
    ) -> Any:
        given_way = False
        while True:
            await self._quiet.wait()
            if not own_lane:
                await self._calm.wait()
                await self._take_turn(rank)
            try:
                if not self._quiet.is_set():
                    continue  # the character started replying while this waited
                if not own_lane and not self._calm.is_set():
                    continue  # a hold was placed while this waited for its turn
                call = asyncio.ensure_future(make_call())
                watching = (
                    [] if given_way else [asyncio.ensure_future(self._interrupted.wait())]
                )
                try:
                    done, _ = await asyncio.wait(
                        {call, *watching},
                        timeout=timeout,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                finally:
                    for watcher in watching:
                        watcher.cancel()
                    if not call.done():
                        call.cancel()
                        await asyncio.gather(call, return_exceptions=True)
                if call in done and not call.cancelled():
                    return call.result()
                if not done:
                    raise TimeoutError(f"model call exceeded {timeout:.0f}s")
                given_way = True
            finally:
                if not own_lane:
                    self._pass_on()


class PoliteClient:
    """An LLMClient whose every call goes through ModelAccess."""

    def __init__(
        self, inner: Any, access: ModelAccess, *, rank: int, timeout: float, own_lane: bool
    ) -> None:
        self._inner = inner
        self._access = access
        self._rank = rank
        self._timeout = timeout
        self._own_lane = own_lane

    async def generate(self, messages, *, tools=None):
        return await self._access.call(
            self._rank,
            lambda: self._inner.generate(messages, tools=tools),
            self._timeout,
            own_lane=self._own_lane,
        )
