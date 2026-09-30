import math
from dataclasses import dataclass

import pytest

from ai_character_engine.character.profile import CharacterProfile
from ai_character_engine.events import CharacterEvent
from ai_character_engine.memory import (
    AsyncHybridMemoryRetriever,
    CallableAsyncEmbeddingProvider,
    HybridMemoryRetriever,
    InMemoryMemoryStore,
    InMemoryVectorIndex,
    MemoryRecord,
    QdrantPoint,
    QdrantVectorIndex,
    QdrantVectorMemoryRetriever,
    ReciprocalRankFusion,
)
from ai_character_engine.memory.vector import VectorHit
from ai_character_engine.runtime import CharacterRuntime
from tests.fakes import FakeLLMClient, system_context


class CountingAsyncEmbedding:
    model_id = "test/async-v1"
    dimensions = 3

    def __init__(self):
        self.batches = []

    async def aembed_many(self, texts):
        self.batches.append(tuple(texts))
        result = []
        for text in texts:
            lowered = text.lower()
            if "coffee" in lowered or "espresso" in lowered or "咖啡" in lowered:
                result.append((1.0, 0.0, 0.0))
            elif "movie" in lowered or "film" in lowered or "電影" in lowered:
                result.append((0.0, 1.0, 0.0))
            else:
                result.append((0.0, 0.0, 1.0))
        return result


class FakeQdrantBackend:
    def __init__(self):
        self.dimensions = None
        self.points = {}
        self.search_calls = 0
        self.upsert_calls = 0

    async def ensure_collection(self, *, collection_name, dimensions):
        if self.dimensions is not None and self.dimensions != dimensions:
            raise ValueError("dimension mismatch")
        self.dimensions = dimensions

    async def fingerprints(self, *, collection_name, character_id, model_id):
        return {
            memory_id: payload["content_hash"]
            for memory_id, (vector, payload) in self.points.items()
            if payload["character_id"] == character_id
            and payload["model_id"] == model_id
        }

    async def upsert_points(self, *, collection_name, points):
        self.upsert_calls += 1
        for point in points:
            self.points[point.memory_id] = (tuple(point.vector), dict(point.payload))

    async def delete_memory_ids(
        self, *, collection_name, character_id, model_id, memory_ids
    ):
        for memory_id in memory_ids:
            self.points.pop(memory_id, None)

    async def search(
        self,
        *,
        collection_name,
        character_id,
        model_id,
        vector,
        limit,
        min_score,
        metadata_filter=None,
    ):
        self.search_calls += 1
        hits = []
        for memory_id, (values, payload) in self.points.items():
            if payload["character_id"] != character_id:
                continue
            if payload["model_id"] != model_id or payload["status"] != "active":
                continue
            if any(payload.get("metadata", {}).get(k) != v for k, v in (metadata_filter or {}).items()):
                continue
            score = sum(a * b for a, b in zip(values, vector))
            if score >= min_score:
                hits.append(VectorHit(memory_id, score))
        return sorted(hits, key=lambda hit: (-hit.score, hit.memory_id))[:limit]


def record(memory_id, summary, **kwargs):
    return MemoryRecord(
        id=memory_id,
        character_id="test",
        summary=summary,
        **kwargs,
    )


def make_qdrant_retriever(store, embedding, backend, **kwargs):
    index = QdrantVectorIndex(
        backend=backend,
        collection_name="memories",
        dimensions=embedding.dimensions,
        model_id=embedding.model_id,
    )
    return QdrantVectorMemoryRetriever(
        store,
        embedding=embedding,
        index=index,
        **kwargs,
    )


def test_min_score_one_is_inclusive_for_exact_vector():
    index = InMemoryVectorIndex(dimensions=2, model_id="test")
    index.upsert(character_id="c", memory_id="same", vector=[1, 0])
    assert index.search(
        character_id="c", vector=[1, 0], min_score=1.0
    )[0].memory_id == "same"


def test_vector_weight_zero_skips_sync_embedding_pipeline():
    class ExplodingEmbedding:
        model_id = "never"
        dimensions = 2

        def embed(self, text):
            raise AssertionError("vector branch must not execute")

    store = InMemoryMemoryStore([record("c", "coffee", importance=0.9)])
    result = HybridMemoryRetriever(
        store,
        embedding=ExplodingEmbedding(),
        fusion=ReciprocalRankFusion(lexical_weight=1, vector_weight=0),
    ).retrieve_with_trace(character_id="test", query="coffee")
    assert result.memories[0].record.id == "c"
    assert result.trace.lexical_executed is True
    assert result.trace.vector_executed is False


@pytest.mark.asyncio
async def test_callable_async_embedding_batches_and_validates():
    calls = []

    async def embed_batch(texts):
        calls.append(tuple(texts))
        return [(1, 0) for _ in texts]

    provider = CallableAsyncEmbeddingProvider(
        model_id="remote/test", dimensions=2, embed_many_fn=embed_batch
    )
    result = await provider.aembed_many(["a", "b"])
    assert result == ((1.0, 0.0), (1.0, 0.0))
    assert calls == [("a", "b")]


@pytest.mark.asyncio
async def test_qdrant_retriever_batches_documents_caches_and_filters_metadata():
    backend = FakeQdrantBackend()
    embedding = CountingAsyncEmbedding()
    store = InMemoryMemoryStore(
        [
            record("coffee", "coffee", metadata={"locale": "zh-TW"}),
            record("movie", "movie", metadata={"locale": "en"}),
            record("forgot", "coffee", status="forgotten", metadata={"locale": "zh-TW"}),
        ]
    )
    retriever = make_qdrant_retriever(
        store, embedding, backend, metadata_filter={"locale": "zh-TW"}
    )

    first = await retriever.retrieve_with_trace_async(
        character_id="test", query="espresso", limit=5
    )
    assert [item.record.id for item in first.memories] == ["coffee"]
    assert first.trace.embedded_records == 2
    assert first.trace.filtered_inactive == 1
    assert first.trace.index_backend == "qdrant"
    assert first.trace.vector_executed is True
    assert len(embedding.batches) == 2  # one document batch + one query batch

    second = await retriever.retrieve_with_trace_async(
        character_id="test", query="espresso", limit=5
    )
    assert [item.record.id for item in second.memories] == ["coffee"]
    assert second.trace.embedded_records == 0
    assert second.trace.cached_vectors == 2
    assert len(embedding.batches) == 3  # only the second query was embedded

    store.replace_for_character("test", [record("coffee", "movie", metadata={"locale": "zh-TW"})])
    third = await retriever.retrieve_with_trace_async(
        character_id="test", query="espresso", limit=5
    )
    assert third.memories == ()
    assert third.trace.removed_vectors == 1


@pytest.mark.asyncio
async def test_async_hybrid_skips_disabled_vector_branch():
    store = InMemoryMemoryStore([record("coffee", "coffee", importance=0.9)])

    class ExplodingVector:
        async def retrieve_with_trace_async(self, **kwargs):
            raise AssertionError("disabled vector branch was executed")

    retriever = AsyncHybridMemoryRetriever(
        store,
        vector_retriever=ExplodingVector(),
        fusion=ReciprocalRankFusion(lexical_weight=1, vector_weight=0),
    )
    result = await retriever.retrieve_with_trace_async(
        character_id="test", query="coffee"
    )
    assert result.memories[0].record.id == "coffee"
    assert result.trace.lexical_executed is True
    assert result.trace.vector_executed is False


@pytest.mark.asyncio
async def test_runtime_accepts_async_qdrant_hybrid_retriever():
    backend = FakeQdrantBackend()
    embedding = CountingAsyncEmbedding()
    store = InMemoryMemoryStore([record("coffee", "coffee")])
    vector = make_qdrant_retriever(store, embedding, backend)
    retriever = AsyncHybridMemoryRetriever(store, vector_retriever=vector)
    llm = FakeLLMClient("ok")
    runtime = CharacterRuntime(
        character=CharacterProfile(id="test", name="Test", description="Test"),
        llm=llm,
        memory_retriever=retriever,
    )
    result = await runtime.process_event(CharacterEvent.user_message("espresso"))
    assert result.retrieval_trace.strategy == "async-hybrid"
    assert result.retrieval_trace.selected_memory_ids == ("coffee",)
    assert "coffee" in system_context(llm.calls[0])
