from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from ai_character_engine.llm.models import LLMResponse, LLMStreamChunk, Message
from ai_character_engine.tools.models import ToolDefinition


class LLMClient(Protocol):
    async def generate(
        self,
        messages: list[Message],
        *,
        tools: list[ToolDefinition] | None = None,
    ) -> LLMResponse: ...


class StreamingLLMClient(LLMClient, Protocol):
    """Optional capability for true provider text streaming."""

    def stream_generate(
        self,
        messages: list[Message],
        *,
        tools: list[ToolDefinition] | None = None,
    ) -> AsyncIterator[LLMStreamChunk]: ...
