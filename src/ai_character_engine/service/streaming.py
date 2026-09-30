from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Literal, Protocol

from ai_character_engine.events.models import CharacterEvent
from ai_character_engine.observability import TraceContext
from ai_character_engine.session.runtime import ManagedCharacterSession


StreamEventType = Literal["delta", "trace", "final", "error"]


@dataclass(slots=True, frozen=True)
class ServiceStreamEvent:
    type: StreamEventType
    data: dict[str, Any] = field(default_factory=dict)

    def to_sse(self) -> str:
        payload = json.dumps(self.data, ensure_ascii=False, separators=(",", ":"))
        return f"event: {self.type}\ndata: {payload}\n\n"


class CharacterStreamSource(Protocol):
    async def stream_message(
        self,
        session: ManagedCharacterSession,
        *,
        content: str,
        request_id: str,
        trace_id: str,
        timeout_seconds: float,
    ) -> AsyncIterator[ServiceStreamEvent]: ...


class BufferedCharacterStreamSource:
    """Provider-neutral fallback stream source.

    It waits for the normal CharacterRuntime result, then emits transport-level
    deltas. This proves the SSE/WebSocket contract without coupling the engine
    to OpenAI/Claude token streaming. A future live provider can implement the
    same CharacterStreamSource protocol and emit deltas during inference.
    """

    def __init__(self, *, chunk_chars: int = 24) -> None:
        if chunk_chars <= 0:
            raise ValueError("chunk_chars must be > 0")
        self.chunk_chars = chunk_chars

    async def stream_message(
        self,
        session: ManagedCharacterSession,
        *,
        content: str,
        request_id: str,
        trace_id: str,
        timeout_seconds: float,
    ) -> AsyncIterator[ServiceStreamEvent]:
        context = TraceContext(
            trace_id=trace_id, request_id=request_id, session_id=session.record.id,
            user_id=session.record.user_id, character_id=session.record.character_id,
        )
        result = await asyncio.wait_for(
            session.process_event(CharacterEvent.user_message(content), trace_context=context),
            timeout=timeout_seconds,
        )
        text = result.text
        for index in range(0, len(text), self.chunk_chars):
            yield ServiceStreamEvent(
                "delta",
                {
                    "text": text[index : index + self.chunk_chars],
                    "request_id": request_id,
                    "trace_id": trace_id,
                },
            )
            await asyncio.sleep(0)

        yield ServiceStreamEvent(
            "trace",
            {
                "request_id": request_id,
                "trace_id": trace_id,
                "rounds": result.rounds,
                "tool_count": len(result.tool_results),
                "input_tokens": result.response.input_tokens,
                "output_tokens": result.response.output_tokens,
                "latency_ms": result.response.latency_ms,
            },
        )
        yield ServiceStreamEvent(
            "final",
            {
                "text": text,
                "session_id": session.record.id,
                "request_id": request_id,
                "trace_id": trace_id,
            },
        )
