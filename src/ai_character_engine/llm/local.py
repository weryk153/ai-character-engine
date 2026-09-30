from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit, urlunsplit

try:
    from openai import AsyncOpenAI
except ImportError:  # optional until a compatible provider is instantiated
    AsyncOpenAI = None  # type: ignore[assignment]

from ai_character_engine.llm.errors import LLMError
from ai_character_engine.llm.models import LLMResponse, LLMStreamChunk, Message
from ai_character_engine.tools.models import ToolCall, ToolDefinition


def safe_base_url(value: str) -> str:
    """Return a log-safe URL without userinfo, query string, or fragment."""
    parts = urlsplit(value)
    hostname = parts.hostname or ""
    if parts.port is not None:
        hostname = f"{hostname}:{parts.port}"
    return urlunsplit((parts.scheme, hostname, parts.path.rstrip("/"), "", ""))


@dataclass(frozen=True, slots=True)
class InferenceRuntimeMetadata:
    """Optional descriptive metadata about a local/private inference runtime."""

    backend: str
    device: str | None = None
    quantization: str | None = None
    context_length: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"backend": self.backend}
        if self.device is not None:
            payload["device"] = self.device
        if self.quantization is not None:
            payload["quantization"] = self.quantization
        if self.context_length is not None:
            payload["context_length"] = self.context_length
        payload.update(self.extra)
        return payload


class OpenAICompatibleChatClient:
    """Provider-neutral Chat Completions client for OpenAI-compatible servers.

    The engine deliberately targets the common Chat Completions surface here
    because it is supported by local runtimes such as Ollama and vLLM. Provider
    specifics remain at this boundary; CharacterRuntime still sees LLMClient.
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str | None = None,
        backend: str = "openai_compatible",
        client: Any | None = None,
        timeout_seconds: float | None = 60.0,
        max_retries: int = 0,
        retry_backoff_seconds: float = 0.25,
        max_concurrency: int = 4,
        runtime_metadata: InferenceRuntimeMetadata | None = None,
        request_options: dict[str, Any] | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be empty")
        if not base_url.strip():
            raise ValueError("base_url must not be empty")
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0 when set")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must be >= 0")
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")

        self.model = model
        self.request_options = dict(request_options or {})
        if set(self.request_options) - {"temperature", "max_tokens", "extra_body"}:
            raise ValueError("request_options supports temperature, max_tokens and extra_body only")
        extra_body = self.request_options.get("extra_body") or {}
        if not isinstance(extra_body, dict) or set(extra_body) & {"model", "messages", "tools", "stream"}:
            raise ValueError("extra_body cannot override model, messages, tools or stream")
        self.base_url = base_url.rstrip("/")
        self.backend = backend
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self.runtime_metadata = runtime_metadata or InferenceRuntimeMetadata(backend=backend)

        if client is not None:
            self.client = client
        else:
            if AsyncOpenAI is None:
                raise RuntimeError(
                    "OpenAI-compatible local providers require the 'openai' package."
                )
            # The OpenAI SDK requires a non-empty key even when a local server ignores it.
            self.client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=api_key or "local-not-used",
                max_retries=0,  # engine-level retries stay explicit and observable
            )

    async def generate(
        self,
        messages: list[Message],
        *,
        tools: list[ToolDefinition] | None = None,
    ) -> LLMResponse:
        async with self._semaphore:
            started = time.perf_counter()
            last_error: Exception | None = None
            for attempt in range(self.max_retries + 1):
                try:
                    response = await self._call(messages, tools=tools)
                    latency_ms = (time.perf_counter() - started) * 1000
                    return self._normalize(response, latency_ms=latency_ms, attempt=attempt)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # provider/network exceptions normalized at boundary
                    last_error = exc
                    if attempt >= self.max_retries:
                        break
                    delay = self.retry_backoff_seconds * (2**attempt)
                    if delay:
                        await asyncio.sleep(delay)

            raise LLMError(
                f"{self.backend} provider call failed after {self.max_retries + 1} attempt(s): "
                f"{last_error}"
            ) from last_error

    async def stream_generate(
        self,
        messages: list[Message],
        *,
        tools: list[ToolDefinition] | None = None,
    ):
        """Yield normalized text deltas and one final ``LLMResponse``.

        Retries are allowed only before the first externally visible delta. Once
        text has been yielded, replaying a request could duplicate spoken output,
        so a later provider failure is surfaced immediately.
        """
        async with self._semaphore:
            last_error: Exception | None = None
            for attempt in range(self.max_retries + 1):
                emitted = False
                started = time.perf_counter()
                stream = None
                try:
                    request: dict[str, Any] = {
                        **self.request_options,
                        "model": self.model,
                        "messages": self._to_chat_messages(messages),
                        "stream": True,
                    }
                    if tools:
                        request["tools"] = [self._to_chat_tool(tool) for tool in tools]

                    async def consume():
                        nonlocal emitted, stream
                        stream = await self.client.chat.completions.create(**request)
                        text_parts: list[str] = []
                        raw_calls: dict[int, dict[str, str]] = {}
                        model = self.model
                        usage = None
                        finish_reason = None
                        async for chunk in stream:
                            model = getattr(chunk, "model", None) or model
                            usage = getattr(chunk, "usage", None) or usage
                            for choice in getattr(chunk, "choices", None) or []:
                                if getattr(choice, "index", 0) != 0:
                                    continue
                                finish_reason = (
                                    getattr(choice, "finish_reason", None) or finish_reason
                                )
                                delta = getattr(choice, "delta", None)
                                if delta is None:
                                    continue
                                content = getattr(delta, "content", None)
                                if content:
                                    emitted = True
                                    text_parts.append(content)
                                    yield LLMStreamChunk(text=content)
                                for raw in getattr(delta, "tool_calls", None) or []:
                                    index = int(getattr(raw, "index", 0))
                                    entry = raw_calls.setdefault(
                                        index, {"id": "", "name": "", "arguments": ""}
                                    )
                                    entry["id"] += getattr(raw, "id", None) or ""
                                    fn = getattr(raw, "function", None)
                                    if fn is not None:
                                        entry["name"] += getattr(fn, "name", None) or ""
                                        entry["arguments"] += getattr(fn, "arguments", None) or ""

                        calls: list[ToolCall] = []
                        for _, raw in sorted(raw_calls.items()):
                            try:
                                arguments = json.loads(raw["arguments"] or "{}")
                            except json.JSONDecodeError as exc:
                                raise LLMError(f"invalid tool arguments from {self.backend}: {exc}") from exc
                            if not raw["id"] or not raw["name"]:
                                raise LLMError(f"incomplete streamed tool call from {self.backend}")
                            calls.append(
                                ToolCall(
                                    call_id=raw["id"],
                                    name=raw["name"],
                                    arguments=arguments,
                                )
                            )
                        latency_ms = (time.perf_counter() - started) * 1000
                        deployment = {
                            "backend": self.backend,
                            "endpoint_type": "openai_compatible",
                            "base_url": safe_base_url(self.base_url),
                            "model": model,
                            "attempt": attempt + 1,
                            "runtime": self.runtime_metadata.to_dict(),
                        }
                        response = LLMResponse(
                            text="".join(text_parts),
                            tool_calls=tuple(calls),
                            model=model,
                            input_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
                            output_tokens=getattr(usage, "completion_tokens", None) if usage else None,
                            latency_ms=latency_ms,
                            metadata={
                                "deployment": deployment,
                                "streaming": {
                                    "provider_streaming": True,
                                    "attempt": attempt + 1,
                                },
                                **self._finish_metadata(finish_reason),
                            },
                        )
                        yield LLMStreamChunk(final=True, response=response)

                    iterator = consume()
                    if self.timeout_seconds is None:
                        async for update in iterator:
                            yield update
                    else:
                        async with asyncio.timeout(self.timeout_seconds):
                            async for update in iterator:
                                yield update
                    return
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    last_error = exc
                    if emitted or attempt >= self.max_retries:
                        break
                    delay = self.retry_backoff_seconds * (2**attempt)
                    if delay:
                        await asyncio.sleep(delay)
                finally:
                    if stream is not None:
                        close = getattr(stream, "close", None)
                        if callable(close):
                            result = close()
                            if hasattr(result, "__await__"):
                                await result

            raise LLMError(
                f"{self.backend} streaming provider call failed after {attempt + 1} attempt(s): {last_error}"
            ) from last_error

    async def _call(
        self,
        messages: list[Message],
        *,
        tools: list[ToolDefinition] | None,
    ) -> Any:
        request: dict[str, Any] = {
            **self.request_options,
            "model": self.model,
            "messages": self._to_chat_messages(messages),
        }
        if tools:
            request["tools"] = [self._to_chat_tool(tool) for tool in tools]

        call = self.client.chat.completions.create(**request)
        if self.timeout_seconds is None:
            return await call
        return await asyncio.wait_for(call, timeout=self.timeout_seconds)

    def _normalize(self, response: Any, *, latency_ms: float, attempt: int) -> LLMResponse:
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise LLMError(f"{self.backend} returned no choices")
        message = getattr(choices[0], "message", None)
        if message is None:
            raise LLMError(f"{self.backend} returned a choice without a message")

        calls: list[ToolCall] = []
        for raw_call in getattr(message, "tool_calls", None) or []:
            function = getattr(raw_call, "function", None)
            if function is None:
                continue
            raw_arguments = getattr(function, "arguments", "{}") or "{}"
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError as exc:
                raise LLMError(f"invalid tool arguments from {self.backend}: {exc}") from exc
            calls.append(
                ToolCall(
                    call_id=str(getattr(raw_call, "id", "")),
                    name=str(getattr(function, "name", "")),
                    arguments=arguments,
                )
            )

        usage = getattr(response, "usage", None)
        deployment = {
            "backend": self.backend,
            "endpoint_type": "openai_compatible",
            "base_url": safe_base_url(self.base_url),
            "model": getattr(response, "model", None) or self.model,
            "attempt": attempt + 1,
            "runtime": self.runtime_metadata.to_dict(),
        }
        return LLMResponse(
            text=(getattr(message, "content", None) or ""),
            tool_calls=tuple(calls),
            model=getattr(response, "model", None) or self.model,
            input_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
            output_tokens=getattr(usage, "completion_tokens", None) if usage else None,
            latency_ms=latency_ms,
            metadata={
                "deployment": deployment,
                **self._finish_metadata(getattr(choices[0], "finish_reason", None)),
            },
        )

    @staticmethod
    def _finish_metadata(finish_reason: Any) -> dict[str, str]:
        # Lets a host tell a truncated or reasoning-only reply from a silent one.
        return {"finish_reason": finish_reason} if isinstance(finish_reason, str) else {}

    @staticmethod
    def _to_chat_tool(tool: ToolDefinition) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }

    @staticmethod
    def _to_chat_messages(messages: list[Message]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for message in messages:
            if message.role in {"system", "user"}:
                items.append({"role": message.role, "content": message.content})
                continue
            if message.role == "event":
                items.append(
                    {
                        "role": "user",
                        "content": "[Environment event]\n" + message.content,
                    }
                )
                continue
            if message.role == "assistant":
                item: dict[str, Any] = {"role": "assistant", "content": message.content or None}
                if message.tool_calls:
                    item["tool_calls"] = [
                        {
                            "id": call.call_id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments, ensure_ascii=False),
                            },
                        }
                        for call in message.tool_calls
                    ]
                items.append(item)
                continue
            if message.role == "tool" and message.tool_result is not None:
                items.append(
                    {
                        "role": "tool",
                        "tool_call_id": message.tool_result.call_id,
                        "content": message.tool_result.output,
                    }
                )
        return items


class OllamaClient(OpenAICompatibleChatClient):
    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://127.0.0.1:11434/v1",
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            base_url=base_url,
            api_key=api_key or "ollama",
            backend="ollama",
            **kwargs,
        )


class VLLMClient(OpenAICompatibleChatClient):
    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://127.0.0.1:8000/v1",
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            base_url=base_url,
            api_key=api_key or "local-not-used",
            backend="vllm",
            **kwargs,
        )
