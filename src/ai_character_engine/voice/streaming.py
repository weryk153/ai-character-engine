from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from ai_character_engine.events.models import CharacterEvent
from ai_character_engine.observability import TraceContext
from ai_character_engine.session.runtime import ManagedCharacterSession

from .models import TextDelta


class BufferedCharacterTextStream:
    """Provider-neutral fallback text stream.

    The character turn completes first; text is then emitted as transport-level
    deltas. This keeps the voice pipeline independent of a specific token-
    streaming LLM provider. A live adapter can implement the same interface.
    """

    def __init__(self, *, chunk_chars: int = 24) -> None:
        if chunk_chars <= 0:
            raise ValueError("chunk_chars must be > 0")
        self.chunk_chars = chunk_chars

    async def stream_text(
        self,
        session: ManagedCharacterSession,
        *,
        content: str,
        trace_context: TraceContext,
    ) -> AsyncIterator[TextDelta]:
        result = await session.process_event(
            CharacterEvent.user_message(content),
            trace_context=trace_context,
        )
        text = result.text
        for index in range(0, len(text), self.chunk_chars):
            yield TextDelta(text[index : index + self.chunk_chars])
            await asyncio.sleep(0)
        yield TextDelta("", final=True, metadata={"text": text})


class RuntimeCharacterTextStream:
    """True CharacterRuntime text stream backed by the optional LLM capability.

    The managed session still owns transactional commit/rollback. Text deltas are
    exposed as soon as CharacterRuntime receives them, while the final marker is
    emitted only after the whole turn commits successfully.
    """

    def __init__(self, *, queue_size: int = 32) -> None:
        if queue_size < 1:
            raise ValueError("queue_size must be >= 1")
        self.queue_size = queue_size

    async def stream_text(
        self,
        session: ManagedCharacterSession,
        *,
        content: str,
        trace_context: TraceContext,
    ) -> AsyncIterator[TextDelta]:
        queue: asyncio.Queue[object] = asyncio.Queue(self.queue_size)
        done = object()

        async def on_delta(text: str) -> None:
            await queue.put(TextDelta(text))

        async def produce() -> None:
            try:
                result = await session.process_event(
                    CharacterEvent.user_message(content),
                    trace_context=trace_context,
                    on_text_delta=on_delta,
                )
                await queue.put(
                    TextDelta("", final=True, metadata={"text": result.text})
                )
            except BaseException as exc:
                await queue.put(exc)
            finally:
                await queue.put(done)

        task = asyncio.create_task(produce())
        try:
            while True:
                item = await queue.get()
                try:
                    if item is done:
                        break
                    if isinstance(item, BaseException):
                        raise item
                    assert isinstance(item, TextDelta)
                    yield item
                finally:
                    queue.task_done()
            await task
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
