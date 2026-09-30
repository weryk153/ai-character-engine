import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, replace
from datetime import UTC, datetime

import pytest

from ai_character_engine.memory import (
    HashEmbeddingProvider,
    HybridMemoryRetriever,
    ImportanceRecencyReranker,
    InMemoryMemoryStore,
    InMemoryVectorIndex,
    JsonlMemoryStore,
    LexicalMemoryRetriever,
    MemoryRecord,
    MemoryRetriever,
    ReciprocalRankFusion,
    RetrievedMemory,
    VectorMemoryRetriever,
    embed_memory,
)
from ai_character_engine.memory.retriever import retrieve_with_trace

NOW = datetime(2026, 9, 16, tzinfo=UTC)


def record(id, summary, **kwargs):
    return MemoryRecord(
        id=id,
        character_id=kwargs.pop("character_id", "test"),
        summary=summary,
        created_at=NOW,
        **kwargs,
    )


class SemanticTestEmbedding:
    """Tiny hand-authored test oracle; no claim of learned semantics."""

    model_id = "test/concepts-v1"
    dimensions = 3

    def __init__(self):
        self.calls = []

    def embed(self, text):
        self.calls.append(text)
        lowered = text.lower()
        if any(word in lowered for word in ("coffee", "espresso", "咖啡")):
            return (1.0, 0.0, 0.0)
        if any(word in lowered for word in ("cinema", "film", "movie", "電影")):
            return (0.0, 1.0, 0.0)
        if "opposite" in lowered:
            return (-1.0, 0.0, 0.0)
        return (0.0, 0.0, 1.0)


def test_hash_embedding_is_normalized_multilingual_and_stable_across_processes():
    provider = HashEmbeddingProvider(32)
    expected = provider.embed("喜歡咖啡 coffee 電影")
    assert len(expected) == 32
    assert math.isclose(sum(v * v for v in expected), 1)
    assert provider.embed("") == (0.0,) * 32
    assert provider.embed("COFFEE") == provider.embed("coffee")
    code = 'import json; from ai_character_engine.memory import HashEmbeddingProvider; print(json.dumps(HashEmbeddingProvider(32).embed("喜歡咖啡 coffee 電影")))'
    for seed in ("1", "23"):
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": "src"}
        output = subprocess.check_output(
            [sys.executable, "-c", code], env=env, text=True
        )
        assert tuple(json.loads(output)) == expected


@pytest.mark.parametrize("dimensions", [0, -1, 2.5])
def test_hash_rejects_invalid_dimensions(dimensions):
    with pytest.raises(ValueError):
        HashEmbeddingProvider(dimensions)


def test_index_cosine_sort_upsert_partition_and_delete():
    index = InMemoryVectorIndex(dimensions=2, model_id="test")
    for owner, id, values in (
        ("a", "x", [10, 0]),
        ("a", "b", [1, 1]),
        ("a", "negative", [-1, 0]),
        ("b", "x", [0, 1]),
        ("a", "zero", [0, 0]),
    ):
        index.upsert(character_id=owner, memory_id=id, vector=values)
    hits = index.search(character_id="a", vector=[1, 0])
    assert [hit.memory_id for hit in hits] == ["x", "b"]
    assert hits[0].score == 1
    assert hits[1].score == pytest.approx(1 / math.sqrt(2))
    assert index.search(character_id="a", vector=[0, 0]) == []
    assert index.search(character_id="a", vector=[1, 0], allowed_ids=[]) == []
    assert index.search(character_id="a", vector=[1, 0], limit=0) == []
    index.upsert(character_id="a", memory_id="x", vector=[0, 1])
    assert [h.memory_id for h in index.search(character_id="a", vector=[1, 0])] == ["b"]
    index.delete(character_id="a", memory_id="b")
    assert index.retain(character_id="a", memory_ids=[]) == 3
    assert len(index.search(character_id="b", vector=[0, 1])) == 1


@pytest.mark.parametrize(
    "vector", [[1], [1, 2, 3], [float("nan"), 0], [0, float("inf")]]
)
def test_index_rejects_invalid_vectors_atomically(vector):
    index = InMemoryVectorIndex(dimensions=2, model_id="test")
    index.upsert(character_id="a", memory_id="x", vector=[1, 0])
    with pytest.raises(ValueError):
        index.upsert(character_id="a", memory_id="x", vector=vector)
    assert index.search(character_id="a", vector=[1, 0])[0].score == 1
    with pytest.raises(ValueError):
        index.search(character_id="a", vector=vector)


def test_index_ties_and_large_values_are_deterministic():
    index = InMemoryVectorIndex(dimensions=2, model_id="test")
    for id in ("z", "a"):
        index.upsert(character_id="a", memory_id=id, vector=[1e308, 1e308])
    assert [
        h.memory_id for h in index.search(character_id="a", vector=[1e308, 1e308])
    ] == ["a", "z"]


def test_old_and_new_jsonl_round_trip(tmp_path):
    path = tmp_path / "memory.jsonl"
    old = record("old", "coffee", status="forgotten", forgotten_at=NOW)
    payload = JsonlMemoryStore._to_dict(old)
    for key in ("embedding", "embedding_metadata", "vector_metadata"):
        del payload[key]
    path.write_text(json.dumps(payload) + "\n")
    store = JsonlMemoryStore(path)
    assert store.list_for_character("test")[0] == old
    embedded = replace(
        embed_memory(record("new", "espresso"), SemanticTestEmbedding()),
        vector_metadata={"external_id": "opaque-123"},
    )
    store.add(embedded)
    assert JsonlMemoryStore(path).list_for_character("test") == [old, embedded]
    store.replace_for_character("test", [embedded])
    assert JsonlMemoryStore(path).list_for_character("test") == [embedded]


@pytest.mark.parametrize(
    "retriever_type",
    [
        MemoryRetriever,
        LexicalMemoryRetriever,
        VectorMemoryRetriever,
        HybridMemoryRetriever,
    ],
)
def test_scope_status_limit_and_empty_store(retriever_type):
    store = InMemoryMemoryStore(
        [
            record("ok", "coffee", importance=0.9),
            record("forgot", "coffee", status="forgotten"),
            record("old", "coffee", status="superseded"),
            record("other", "coffee", character_id="other"),
        ]
    )
    retriever = retriever_type(store)
    result = retriever.retrieve_with_trace(
        character_id="test", query="coffee", now=NOW, limit=1
    )
    assert [i.record.id for i in result.memories] == ["ok"]
    assert result.trace.selected_memory_ids == ("ok",)
    assert result.trace.filtered_inactive == 2
    assert result.trace.active_records == 1
    assert result.trace.total_records == 3
    assert result.trace.elapsed_ms >= 0
    json.dumps(asdict(result.trace))
    for limit in (0, -1):
        assert (
            retriever.retrieve(character_id="test", query="coffee", limit=limit) == []
        )
    assert retriever.retrieve(character_id="missing", query="coffee") == []


def test_lexical_preserves_original_weights_and_important_fallback():
    store = InMemoryMemoryStore(
        [
            record("coffee", "coffee", importance=0.5),
            record("important", "film", importance=0.9),
            record("unrelated", "toast", importance=0.4),
        ]
    )
    old = MemoryRetriever(store).retrieve(character_id="test", query="coffee", now=NOW)
    new = LexicalMemoryRetriever(store).retrieve(
        character_id="test", query="coffee", now=NOW
    )
    assert old == new
    assert new[0].score == pytest.approx(0.65 + 0.5 * 0.25 + 0.1)
    assert [r.record.id for r in new] == ["coffee", "important"]
    assert [
        r.record.id
        for r in MemoryRetriever(store).retrieve(character_id="test", query="", now=NOW)
    ] == ["important"]


def test_vector_finds_synonym_without_lexical_overlap():
    store = InMemoryMemoryStore([record("c", "coffee"), record("f", "film")])
    assert (
        LexicalMemoryRetriever(store).retrieve(character_id="test", query="espresso")
        == []
    )
    result = VectorMemoryRetriever(
        store, embedding=SemanticTestEmbedding()
    ).retrieve_with_trace(character_id="test", query="espresso")
    assert [i.record.id for i in result.memories] == ["c"]
    assert result.memories[0].score == 1
    assert result.trace.embedded_records == 2
    assert result.trace.embedding_model_id == "test/concepts-v1"


def test_vector_syncs_add_edit_forget_supersede_delete_and_character_scope():
    provider = SemanticTestEmbedding()
    original = record("same", "coffee")
    store = InMemoryMemoryStore(
        [original, record("same", "film", character_id="other")]
    )
    retriever = VectorMemoryRetriever(store, embedding=provider)
    args = dict(character_id="test", query="espresso")
    first = retriever.retrieve_with_trace(**args)
    assert first.trace.embedded_records == 1
    second = retriever.retrieve_with_trace(**args)
    assert second.trace.cached_vectors == 1 and second.trace.embedded_records == 0
    retriever.retrieve(character_id="other", query="film")
    store.replace_for_character("test", [replace(original, summary="film")])
    assert retriever.retrieve(**args) == []
    store.add(record("new", "coffee"))
    assert retriever.retrieve(**args)[0].record.id == "new"
    store.replace_for_character(
        "test",
        [
            replace(original, status="forgotten"),
            record("new", "coffee", status="superseded"),
        ],
    )
    result = retriever.retrieve_with_trace(**args)
    assert not result.memories and result.trace.removed_vectors == 2
    assert (
        retriever.retrieve(character_id="other", query="film")[0].record.character_id
        == "other"
    )
    store.replace_for_character("test", [original])
    assert retriever.retrieve(**args)
    store.replace_for_character("test", [])
    assert retriever.retrieve_with_trace(**args).trace.removed_vectors == 1


def test_persisted_vector_reuse_and_stale_content_or_model_recompute():
    provider = SemanticTestEmbedding()
    embedded = embed_memory(record("c", "coffee"), provider)
    assert embedded.embedding == (1, 0, 0)
    store = InMemoryMemoryStore([embedded])
    result = VectorMemoryRetriever(store, embedding=provider).retrieve_with_trace(
        character_id="test", query="espresso"
    )
    assert result.trace.reused_embeddings == 1 and result.trace.embedded_records == 0
    for changed in (
        replace(embedded, summary="film"),
        replace(embedded, tags=("cinema",)),
        replace(
            embedded,
            embedding_metadata={**embedded.embedding_metadata, "model_id": "stale"},
        ),
        replace(embedded, embedding=(1.0, 0.0)),
        replace(embedded, embedding_metadata={}),
    ):
        store.replace_for_character("test", [changed])
        result = VectorMemoryRetriever(store, embedding=provider).retrieve_with_trace(
            character_id="test", query="espresso"
        )
        assert (
            result.trace.embedded_records == 1 and result.trace.reused_embeddings == 0
        )


def test_incompatible_spaces_and_provider_errors_fail_explicitly():
    for dims, model in ((2, "test/concepts-v1"), (3, "different")):
        with pytest.raises(ValueError, match="same vector space"):
            VectorMemoryRetriever(
                InMemoryMemoryStore(),
                embedding=SemanticTestEmbedding(),
                index=InMemoryVectorIndex(dimensions=dims, model_id=model),
            )

    class Bad(SemanticTestEmbedding):
        def embed(self, text):
            return (float("nan"), 0, 0)

    with pytest.raises(ValueError, match="finite"):
        VectorMemoryRetriever(
            InMemoryMemoryStore([record("c", "coffee")]), embedding=Bad()
        ).retrieve(character_id="test", query="coffee")
    provider = SemanticTestEmbedding()
    retriever = VectorMemoryRetriever(InMemoryMemoryStore(), embedding=provider)
    provider.model_id = "changed"
    with pytest.raises(ValueError, match="same vector space"):
        retriever.retrieve(character_id="test", query="coffee")


def test_rrf_expected_scores_deduplicates_and_disables_channels():
    a, b, c = (record(id, id) for id in "abc")
    lex = [RetrievedMemory(a, 0.9), RetrievedMemory(b, 0.5)]
    vec = [RetrievedMemory(b, 0.8), RetrievedMemory(c, 0.7)]
    fusion = ReciprocalRankFusion(k=10)
    result = fusion.fuse(lex, vec)
    assert [i.record.id for i in result] == ["b", "a", "c"]
    assert result[0].score == pytest.approx(1 / 12 + 1 / 11)
    assert fusion.fuse([lex[0], lex[0]], [])[0].score == pytest.approx(1 / 11)
    assert [
        i.record.id for i in ReciprocalRankFusion(vector_weight=0).fuse(lex, vec)
    ] == ["a", "b"]


def test_hybrid_union_trace_and_custom_reranker():
    store = InMemoryMemoryStore(
        [
            record("lex", "espresso recipe", importance=0.9),
            record("semantic", "coffee", importance=0.3),
            record("none", "movie"),
        ]
    )
    retriever = HybridMemoryRetriever(
        store,
        embedding=SemanticTestEmbedding(),
        candidate_limit=1,
        reranker=ImportanceRecencyReranker(),
    )
    result = retriever.retrieve_with_trace(
        character_id="test", query="espresso", now=NOW, limit=2
    )
    assert {i.record.id for i in result.memories} == {"lex", "semantic"}
    assert result.trace.candidate_limit == 2
    assert result.trace.reranker == "ImportanceRecencyReranker"
    traces = {c.memory_id: c for c in result.trace.candidates}
    assert traces["lex"].lexical_score is not None
    assert traces["semantic"].lexical_score is None
    assert traces["semantic"].vector_rank is not None
    assert all(c.fusion_score > 0 and c.selected for c in traces.values())
    assert result.trace.selected_memory_ids == tuple(
        i.record.id for i in result.memories
    )


def test_reranker_weights_and_candidate_boundary():
    items = [
        RetrievedMemory(record("low", "coffee", importance=0.1), 0.5),
        RetrievedMemory(record("high", "coffee", importance=0.9), 0.5),
    ]
    ranked = ImportanceRecencyReranker().rerank(items, query="coffee", now=NOW)
    assert ranked[0].record.id == "high"
    assert ranked[0].score == pytest.approx(0.85 * 0.5 + 0.1 * 0.9 + 0.05)

    class InjectingReranker:
        def rerank(self, candidates, **kwargs):
            return [
                RetrievedMemory(record("outside", "coffee", character_id="other"), 1)
            ]

    with pytest.raises(ValueError, match="input candidates"):
        HybridMemoryRetriever(
            InMemoryMemoryStore([items[0].record]), reranker=InjectingReranker()
        ).retrieve(character_id="test", query="coffee")


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ReciprocalRankFusion(k=-1),
        lambda: ReciprocalRankFusion(lexical_weight=0, vector_weight=0),
        lambda: ReciprocalRankFusion(vector_weight=float("nan")),
        lambda: ImportanceRecencyReranker(recency_weight=-1),
        lambda: HybridMemoryRetriever(InMemoryMemoryStore(), candidate_limit=0),
        lambda: VectorMemoryRetriever(InMemoryMemoryStore(), min_score=2),
    ],
)
def test_invalid_configuration(factory):
    with pytest.raises(ValueError):
        factory()


def test_legacy_custom_retriever_and_subclass_keep_their_output():
    expected = [RetrievedMemory(record("custom", "coffee"), 0.123)]

    class OldDuck:
        def retrieve(self, *, character_id, query, limit=5):
            return expected

    class OldSubclass(MemoryRetriever):
        def retrieve(self, *, character_id, query, limit=5, now=None):
            return expected

    for retriever in (OldDuck(), OldSubclass(InMemoryMemoryStore())):
        result = retrieve_with_trace(retriever, character_id="test", query="coffee")
        assert list(result.memories) == expected
        assert result.trace.strategy == "custom"
