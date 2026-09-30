from __future__ import annotations

import time
from typing import Any, Awaitable, Callable

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None  # type: ignore[assignment]

from ai_character_engine.llm.local import safe_base_url

from .errors import VisionProviderError
from .models import ImageInput, VisionAnalysis


class CallableVisionProvider:
    def __init__(
        self,
        fn: Callable[[ImageInput, str | None], Awaitable[VisionAnalysis] | VisionAnalysis],
        *,
        provider: str = "callable",
    ) -> None:
        self.fn = fn
        self.provider = provider

    async def analyze(self, image: ImageInput, *, prompt: str | None = None) -> VisionAnalysis:
        result = self.fn(image, prompt)
        if hasattr(result, "__await__"):
            result = await result  # type: ignore[misc]
        if not isinstance(result, VisionAnalysis):
            raise VisionProviderError("callable vision provider must return VisionAnalysis")
        return result


class OpenAICompatibleVisionProvider:
    """Vision adapter for OpenAI-compatible Chat Completions servers.

    Works with cloud or local VLM servers that accept text + image_url content.
    CharacterRuntime remains unaware of image transport details.
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        provider: str = "openai_compatible_vision",
        client: Any | None = None,
        max_inline_bytes: int = 8 * 1024 * 1024,
        request_options: dict[str, Any] | None = None,
        default_prompt: str = "Describe the visual scene relevant to the character.",
    ) -> None:
        """``request_options`` takes temperature, max_tokens and extra_body,
        like OpenAICompatibleChatClient. A local model that reasons before it
        answers needs its reasoning switched off here, or one picture takes
        longer than the reply it belongs to."""
        if not model.strip():
            raise ValueError("model must not be empty")
        self.request_options = dict(request_options or {})
        if set(self.request_options) - {"temperature", "max_tokens", "extra_body"}:
            raise ValueError("request_options supports temperature, max_tokens and extra_body only")
        self.default_prompt = default_prompt
        self.model = model
        self.provider = provider
        self.max_inline_bytes = max_inline_bytes
        self.base_url = base_url
        if client is not None:
            self.client = client
        else:
            if AsyncOpenAI is None:
                raise RuntimeError("OpenAI-compatible vision requires the openai package")
            kwargs: dict[str, Any] = {"api_key": api_key or "local-not-used"}
            if base_url is not None:
                kwargs["base_url"] = base_url
            self.client = AsyncOpenAI(**kwargs)

    async def analyze(self, image: ImageInput, *, prompt: str | None = None) -> VisionAnalysis:
        image_ref = image.url if image.url is not None else image.to_data_url(max_bytes=self.max_inline_bytes)
        started = time.perf_counter()
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt or self.default_prompt},
                            {"type": "image_url", "image_url": {"url": image_ref}},
                        ],
                    }
                ],
                **self.request_options,
            )
        except Exception as exc:
            raise VisionProviderError(str(exc)) from exc
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise VisionProviderError("vision provider returned no choices")
        message = getattr(choices[0], "message", None)
        text = getattr(message, "content", None) if message is not None else None
        if not text:
            raise VisionProviderError("vision provider returned empty analysis")
        metadata = {"latency_ms": (time.perf_counter() - started) * 1000}
        if self.base_url is not None:
            metadata["base_url"] = safe_base_url(self.base_url)
        return VisionAnalysis(
            text=str(text), provider=self.provider,
            model=getattr(response, "model", None) or self.model,
            metadata=metadata,
        )
