from __future__ import annotations

from collections.abc import Iterable

from ai_character_engine.llm.models import LLMResponse, Message
from ai_character_engine.tools.models import ToolDefinition


class FakeLLMClient:
    def __init__(self, text: str = "fake response") -> None:
        self.text = text
        self.calls: list[list[Message]] = []
        self.tools_seen: list[list[ToolDefinition] | None] = []

    async def generate(
        self,
        messages: list[Message],
        *,
        tools: list[ToolDefinition] | None = None,
    ) -> LLMResponse:
        self.calls.append(list(messages))
        self.tools_seen.append(tools)
        return LLMResponse(text=self.text, model="fake", input_tokens=10, output_tokens=2)


class ScriptedLLMClient:
    def __init__(self, responses: Iterable[LLMResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[list[Message]] = []
        self.tools_seen: list[list[ToolDefinition] | None] = []

    async def generate(
        self,
        messages: list[Message],
        *,
        tools: list[ToolDefinition] | None = None,
    ) -> LLMResponse:
        self.calls.append(list(messages))
        self.tools_seen.append(tools)
        if not self.responses:
            raise AssertionError("ScriptedLLMClient ran out of responses")
        return self.responses.pop(0)


def system_context(messages: Iterable[Message]) -> str:
    """Everything the model was given as instructions or context for a call.

    The character's turn context is a message of its own, placed just before
    the newest message, so tests about *what* reached the model must not look
    at the first message only.
    """
    from ai_character_engine.context.builder import is_turn_context

    return "\n\n".join(
        message.content
        for message in messages
        if message.role == "system" or is_turn_context(message)
    )
