from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .embedding import checked_vector


class OpenAIEmbeddingProvider:
    """Optional convenience adapter for OpenAI-compatible async embedding APIs.

    The core engine does not depend on this class. Pass any provider implementing
    aembed_many/model_id/dimensions to the production retrievers.
    """

    def __init__(
        self,
        *,
        model: str,
        dimensions: int,
        client: Any | None = None,
    ) -> None:
        if not model.strip() or dimensions < 1:
            raise ValueError("model and positive dimensions are required")
        if client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:  # pragma: no cover - project normally has openai
                raise RuntimeError("install openai to use OpenAIEmbeddingProvider") from exc
            client = AsyncOpenAI()
        self.model = model
        self._dimensions = dimensions
        self.client = client

    @property
    def model_id(self) -> str:
        return f"openai/{self.model}/{self.dimensions}"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def aembed_many(
        self, texts: Sequence[str]
    ) -> tuple[tuple[float, ...], ...]:
        if not texts:
            return ()
        response = await self.client.embeddings.create(
            model=self.model,
            input=list(texts),
            dimensions=self.dimensions,
        )
        vectors = tuple(
            checked_vector(item.embedding, self.dimensions)
            for item in sorted(response.data, key=lambda item: item.index)
        )
        if len(vectors) != len(texts):
            raise ValueError("embedding response size does not match input batch")
        return vectors
