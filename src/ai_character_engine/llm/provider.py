from __future__ import annotations

import json
import logging
import time
from typing import Any

try:
    from openai import AsyncOpenAI
except ImportError:  # optional until the OpenAI provider is instantiated
    AsyncOpenAI = None  # type: ignore[assignment]

from ai_character_engine.llm.errors import LLMError
from ai_character_engine.llm.models import LLMResponse, Message
from ai_character_engine.tools.models import ToolCall, ToolDefinition

logger = logging.getLogger(__name__)


class OpenAIResponsesClient:
    def __init__(self, *, model: str, client: Any | None = None) -> None:
        self.model = model
        if client is not None:
            self.client = client
        else:
            if AsyncOpenAI is None:
                raise RuntimeError(
                    "The OpenAI provider requires the optional 'openai' package. "
                    "Install project dependencies before using OpenAIResponsesClient."
                )
            self.client = AsyncOpenAI()

    async def generate(
        self,
        messages: list[Message],
        *,
        tools: list[ToolDefinition] | None = None,
    ) -> LLMResponse:
        start = time.perf_counter()
        try:
            instructions = "\n\n".join(
                message.content for message in messages if message.role == "system"
            ) or None
            input_items = self._to_response_input(messages)
            tool_payload = [tool.to_json_schema() for tool in tools] if tools else None

            request: dict[str, Any] = {
                "model": self.model,
                "instructions": instructions,
                "input": input_items,
            }
            if tool_payload:
                request["tools"] = tool_payload

            response = await self.client.responses.create(**request)
        except Exception as exc:  # provider-specific exceptions are normalized here
            logger.exception("LLM provider call failed")
            raise LLMError(str(exc)) from exc

        calls: list[ToolCall] = []
        for item in getattr(response, "output", []):
            if getattr(item, "type", None) != "function_call":
                continue
            try:
                arguments = json.loads(getattr(item, "arguments", "{}"))
            except json.JSONDecodeError as exc:
                raise LLMError(f"invalid tool arguments from provider: {exc}") from exc
            calls.append(
                ToolCall(
                    call_id=getattr(item, "call_id"),
                    name=getattr(item, "name"),
                    arguments=arguments,
                )
            )

        usage = getattr(response, "usage", None)
        return LLMResponse(
            text=getattr(response, "output_text", "") or "",
            tool_calls=tuple(calls),
            model=getattr(response, "model", self.model),
            input_tokens=getattr(usage, "input_tokens", None) if usage else None,
            output_tokens=getattr(usage, "output_tokens", None) if usage else None,
            latency_ms=(time.perf_counter() - start) * 1000,
        )

    @staticmethod
    def _to_response_input(messages: list[Message]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "system":
                continue

            if message.role in {"user", "assistant"} and message.content:
                items.append({"role": message.role, "content": message.content})

            if message.role == "event" and message.content:
                # Provider APIs generally do not have a first-class environment
                # event role. Keep the engine domain model explicit, then adapt
                # it at the provider boundary as contextual input.
                items.append(
                    {
                        "role": "user",
                        "content": "[Environment event]\n" + message.content,
                    }
                )

            if message.role == "assistant" and message.tool_calls:
                for call in message.tool_calls:
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": call.call_id,
                            "name": call.name,
                            "arguments": json.dumps(call.arguments, ensure_ascii=False),
                        }
                    )

            if message.role == "tool" and message.tool_result is not None:
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": message.tool_result.call_id,
                        "output": message.tool_result.output,
                    }
                )
        return items
