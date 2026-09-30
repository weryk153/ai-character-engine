from __future__ import annotations

import asyncio
import hashlib
import inspect
import math
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from .models import MemoryRecord


class EmbeddingProvider(Protocol):
    """Backward-compatible synchronous embedding contract."""

    @property
    def model_id(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    def embed(self, text: str) -> Sequence[float]: ...


class AsyncEmbeddingProvider(Protocol):
    """Production embedding contract: async and batch-oriented."""

    @property
    def model_id(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    async def aembed_many(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


def checked_vector(values: Sequence[float], dimensions: int) -> tuple[float, ...]:
    vector = tuple(float(value) for value in values)
    if len(vector) != dimensions or dimensions < 1:
        raise ValueError("embedding dimensions do not match the vector space")
    if not all(math.isfinite(value) for value in vector):
        raise ValueError("embedding values must be finite")
    return vector


def unit_vector(values: Sequence[float], dimensions: int) -> tuple[float, ...]:
    vector = checked_vector(values, dimensions)
    scale = max((abs(value) for value in vector), default=0.0)
    if not scale:
        return vector
    scaled = tuple(value / scale for value in vector)
    norm = math.sqrt(sum(value * value for value in scaled))
    return tuple(value / norm for value in scaled)


def memory_text(record: MemoryRecord) -> str:
    return record.summary + " " + " ".join(record.tags)


def content_hash(record: MemoryRecord) -> str:
    return hashlib.sha256(memory_text(record).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class HashEmbeddingProvider:
    """Offline feature hashing for tests/demos, NOT a semantic language model."""

    dimensions: int = 256

    def __post_init__(self) -> None:
        if not isinstance(self.dimensions, int) or self.dimensions < 1:
            raise ValueError("dimensions must be a positive integer")

    @property
    def model_id(self) -> str:
        return f"local/hash-terms-v1/{self.dimensions}"

    def embed(self, text: str) -> tuple[float, ...]:
        from .retriever import _terms

        values = [0.0] * self.dimensions
        for term in sorted(_terms(text)):
            digest = hashlib.sha256(term.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:8], "big") % self.dimensions
            values[bucket] += 1.0 if digest[8] & 1 else -1.0
        return unit_vector(values, self.dimensions)

    def embed_many(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        return tuple(self.embed(text) for text in texts)

    async def aembed_many(
        self, texts: Sequence[str]
    ) -> tuple[tuple[float, ...], ...]:
        return self.embed_many(texts)


@dataclass(frozen=True, slots=True)
class CallableAsyncEmbeddingProvider:
    """Provider-neutral adapter for remote/local async embedding callables.

    The callable receives a batch of strings and may return either an awaitable
    or a concrete batch. This keeps the engine independent from any vendor SDK.
    """

    model_id: str
    dimensions: int
    embed_many_fn: Callable[
        [Sequence[str]],
        Sequence[Sequence[float]] | Awaitable[Sequence[Sequence[float]]],
    ]

    def __post_init__(self) -> None:
        if not self.model_id.strip() or self.dimensions < 1:
            raise ValueError("model_id and positive dimensions are required")

    async def aembed_many(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        result = self.embed_many_fn(tuple(texts))
        if inspect.isawaitable(result):
            result = await result
        vectors = tuple(checked_vector(v, self.dimensions) for v in result)
        if len(vectors) != len(texts):
            raise ValueError("embedding provider returned a different batch size")
        return vectors


async def embed_many_async(
    provider: EmbeddingProvider | AsyncEmbeddingProvider,
    texts: Sequence[str],
) -> tuple[tuple[float, ...], ...]:
    """Use the fastest available batch API without forcing one provider model.

    Priority: aembed_many -> embed_many -> embed. Synchronous fallbacks run in
    a worker thread so remote sync SDKs do not block an async character runtime.
    """

    texts = tuple(texts)
    if not texts:
        return ()

    async_batch = getattr(provider, "aembed_many", None)
    if callable(async_batch):
        values = await async_batch(texts)
    else:
        sync_batch = getattr(provider, "embed_many", None)
        if callable(sync_batch):
            values = await asyncio.to_thread(sync_batch, texts)
        else:
            sync_one = getattr(provider, "embed", None)
            if not callable(sync_one):
                raise TypeError("embedding provider must expose aembed_many, embed_many, or embed")
            values = await asyncio.gather(
                *(asyncio.to_thread(sync_one, text) for text in texts)
            )

    vectors = tuple(checked_vector(v, provider.dimensions) for v in values)
    if len(vectors) != len(texts):
        raise ValueError("embedding provider returned a different batch size")
    return vectors


def embed_memory(record: MemoryRecord, provider: EmbeddingProvider) -> MemoryRecord:
    """Return a new record; the caller chooses whether/how to persist it."""
    vector = checked_vector(provider.embed(memory_text(record)), provider.dimensions)
    return replace(
        record,
        embedding=vector,
        embedding_metadata={
            "model_id": provider.model_id,
            "dimensions": provider.dimensions,
            "content_hash": content_hash(record),
        },
    )


def stored_embedding(
    record: MemoryRecord, provider: EmbeddingProvider | AsyncEmbeddingProvider
) -> tuple[float, ...] | None:
    metadata = record.embedding_metadata
    if record.embedding is None or (
        metadata.get("model_id") != provider.model_id
        or metadata.get("dimensions") != provider.dimensions
        or metadata.get("content_hash") != content_hash(record)
    ):
        return None
    try:
        return checked_vector(record.embedding, provider.dimensions)
    except (ValueError, TypeError, OverflowError):
        return None
