from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from ai_character_engine.tools.models import ToolCall, ToolResult

Role = Literal["system", "user", "assistant", "tool", "event"]


@dataclass(slots=True, frozen=True)
class Message:
    role: Role
    content: str
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)
    tool_result: ToolResult | None = None


@dataclass(slots=True, frozen=True)
class LLMResponse:
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class LLMStreamChunk:
    """One provider-neutral streamed generation update.

    ``text`` contains only user-visible assistant text. Tool-call fragments stay
    provider-internal until the final normalized ``response`` is available.
    This keeps callers from speaking incomplete JSON/function arguments.
    """

    text: str = ""
    final: bool = False
    response: LLMResponse | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.final and self.response is None:
            raise ValueError("final LLMStreamChunk requires response")
        if not self.final and self.response is not None:
            raise ValueError("non-final LLMStreamChunk must not include response")
