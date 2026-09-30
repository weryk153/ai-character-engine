from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Protocol

from ai_character_engine.llm.models import Message
from ai_character_engine.tools.models import ToolDefinition

_CJK = re.compile(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]")


class TokenEstimator(Protocol):
    """Provider-neutral token estimator used for context budgeting.

    Exact tokenization belongs to a provider/model adapter. The default engine
    estimator intentionally uses a deterministic approximation so the core
    package does not depend on a tokenizer library.
    """

    def estimate_text(self, text: str) -> int: ...

    def estimate_message(self, message: Message) -> int: ...

    def estimate_tools(self, tools: list[ToolDefinition] | None) -> int: ...


class HeuristicTokenEstimator:
    """Small deterministic approximation suitable for budgeting and tests.

    CJK characters are conservatively counted near one token each. Remaining
    text is approximated at four characters per token. Provider integrations
    may inject an exact tokenizer-specific estimator later.
    """

    message_overhead_tokens = 4

    def estimate_text(self, text: str) -> int:
        if not text:
            return 0
        cjk_count = len(_CJK.findall(text))
        non_cjk_count = max(0, len(text) - cjk_count)
        return max(1, cjk_count + math.ceil(non_cjk_count / 4))

    def estimate_message(self, message: Message) -> int:
        total = self.message_overhead_tokens + self.estimate_text(message.content)
        for call in message.tool_calls:
            total += self.estimate_text(call.name)
            total += self.estimate_text(
                json.dumps(call.arguments, ensure_ascii=False, sort_keys=True)
            )
        if message.tool_result is not None:
            total += self.estimate_text(message.tool_result.name)
            total += self.estimate_text(message.tool_result.output)
        return total

    def estimate_tools(self, tools: list[ToolDefinition] | None) -> int:
        if not tools:
            return 0
        payload = [tool.to_json_schema() for tool in tools]
        # Tool schemas have provider framing overhead beyond their raw JSON.
        return self.estimate_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True)
        ) + 8 * len(tools)


@dataclass(slots=True, frozen=True)
class ContextBudget:
    """Input budget policy for one model call.

    context_window_tokens is the model's total window. Output is reserved first,
    then tool schemas, mandatory prompt/event content, recent history, bounded active
    beliefs, memories, and finally older history.
    """

    context_window_tokens: int = 8192
    reserved_output_tokens: int = 1024
    recent_history_target_tokens: int = 2048
    max_memory_tokens: int = 2048
    max_belief_tokens: int = 768
    max_goal_tokens: int = 768

    def __post_init__(self) -> None:
        if self.context_window_tokens <= 0:
            raise ValueError("context_window_tokens must be > 0")
        if self.reserved_output_tokens < 0:
            raise ValueError("reserved_output_tokens must be >= 0")
        if self.reserved_output_tokens >= self.context_window_tokens:
            raise ValueError("reserved_output_tokens must be smaller than context window")
        if self.recent_history_target_tokens < 0:
            raise ValueError("recent_history_target_tokens must be >= 0")
        if self.max_memory_tokens < 0:
            raise ValueError("max_memory_tokens must be >= 0")
        if self.max_belief_tokens < 0:
            raise ValueError("max_belief_tokens must be >= 0")
        if self.max_goal_tokens < 0:
            raise ValueError("max_goal_tokens must be >= 0")

    @property
    def input_budget_tokens(self) -> int:
        return self.context_window_tokens - self.reserved_output_tokens


@dataclass(slots=True, frozen=True)
class ContextTrace:
    context_window_tokens: int
    reserved_output_tokens: int
    input_budget_tokens: int
    estimated_tool_tokens: int
    estimated_message_tokens: int
    system_tokens: int
    event_tokens: int
    history_tokens: int
    memory_tokens: int
    included_history_messages: int
    dropped_history_messages: int
    included_memories: int
    dropped_memories: int
    selected_memory_ids: tuple[str, ...] = field(default_factory=tuple)
    belief_tokens: int = 0
    included_beliefs: int = 0
    dropped_beliefs: int = 0
    selected_belief_ids: tuple[str, ...] = field(default_factory=tuple)
    goal_tokens: int = 0
    included_goals: int = 0
    dropped_goals: int = 0
    selected_goal_ids: tuple[str, ...] = field(default_factory=tuple)

    @property
    def estimated_total_tokens(self) -> int:
        return (
            self.estimated_tool_tokens
            + self.estimated_message_tokens
            + self.reserved_output_tokens
        )

    @property
    def remaining_input_tokens(self) -> int:
        return max(
            0,
            self.input_budget_tokens
            - self.estimated_tool_tokens
            - self.estimated_message_tokens,
        )


@dataclass(slots=True, frozen=True)
class ContextBuildResult:
    messages: tuple[Message, ...]
    trace: ContextTrace


class ContextBudgetExceededError(ValueError):
    """Mandatory system/event/tool content cannot fit the configured budget."""
