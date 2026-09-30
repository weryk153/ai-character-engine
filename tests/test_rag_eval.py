from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ai_character_engine.memory import (
    AsyncHybridMemoryRetriever,
    CallableAsyncMemoryReranker,
    CallableQueryRewriter,
    InMemoryMemoryStore,
    MemoryRecord,
    ReciprocalRankFusion,
    RetrievalEvalCase,
    RetrievalEvalCaseResult,
    compute_metrics,
)
from ai_character_engine.memory.models import RetrievedMemory
from ai_character_engine.memory.trace import RetrievalResult, RetrievalTrace


class FakeVectorRetriever:
    def __init__(self, memories):
        self.memories = tuple(memories)
        self.calls = []

    async def retrieve_with_trace_async(self, *, character_id, query, limit=5, now=None):
        self.calls.append(query)
        selected = self.memories[:limit]
        return RetrievalResult(
            tuple(selected),
            RetrievalTrace(
                strategy="fake-vector",
                character_id=character_id,
                query=query,
                requested_limit=limit,
                selected_memory_ids=tuple(x.record.id for x in selected),
                vector_executed=True,
            ),
        )


def memory(mid: str, text: str) -> MemoryRecord:
    return MemoryRecord(character_id="c", id=mid, summary=text, created_at=datetime.now(UTC))


@pytest.mark.asyncio
async def test_query_rewrite_changes_retrieval_query_and_trace():
    store = InMemoryMemoryStore()
    target = memory("m1", "Kurosawa Seven Samurai")
    store.add(target)
    vector = FakeVectorRetriever([RetrievedMemory(target, 0.9)])
    calls = []

    async def rewrite(query, **kwargs):
        calls.append(query)
        return "Kurosawa Seven Samurai"

    retriever = AsyncHybridMemoryRetriever(
        store,
        vector_retriever=vector,
        fusion=ReciprocalRankFusion(lexical_weight=0, vector_weight=1),
        query_rewriter=CallableQueryRewriter(rewrite),
    )
    result = await retriever.retrieve_with_trace_async(
        character_id="c", query="that director", limit=1
    )
    assert calls == ["that director"]
    assert vector.calls == ["Kurosawa Seven Samurai"]
    assert result.trace.query_original == "that director"
    assert result.trace.query_rewritten == "Kurosawa Seven Samurai"
    assert result.trace.rewrite_used is True


@pytest.mark.asyncio
async def test_disabled_rewrite_does_not_call_provider():
    store = InMemoryMemoryStore()
    target = memory("m1", "coffee")
    store.add(target)
    vector = FakeVectorRetriever([RetrievedMemory(target, 0.8)])
    retriever = AsyncHybridMemoryRetriever(
        store,
        vector_retriever=vector,
        fusion=ReciprocalRankFusion(lexical_weight=0, vector_weight=1),
    )
    result = await retriever.retrieve_with_trace_async(character_id="c", query="coffee", limit=1)
    assert vector.calls == ["coffee"]
    assert result.trace.rewrite_used is False
    assert result.trace.query_original == "coffee"
    assert result.trace.query_rewritten == "coffee"


@pytest.mark.asyncio
async def test_async_reranker_improves_ranking_and_trace():
    store = InMemoryMemoryStore()
    a = memory("a", "generic movie")
    b = memory("b", "Seven Samurai")
    store.add(a); store.add(b)
    vector = FakeVectorRetriever([RetrievedMemory(a, 0.9), RetrievedMemory(b, 0.8)])

    async def rerank(candidates, **kwargs):
        by_id = {x.record.id: x for x in candidates}
        return [RetrievedMemory(by_id["b"].record, 1.0), RetrievedMemory(by_id["a"].record, 0.2)]

    retriever = AsyncHybridMemoryRetriever(
        store,
        vector_retriever=vector,
        fusion=ReciprocalRankFusion(lexical_weight=0, vector_weight=1),
        reranker=CallableAsyncMemoryReranker(rerank),
    )
    result = await retriever.retrieve_with_trace_async(character_id="c", query="favorite film", limit=1)
    assert [x.record.id for x in result.memories] == ["b"]
    assert result.trace.reranker_used is True
    assert result.trace.candidate_count == 2
    assert result.trace.final_count == 1
    assert result.trace.reranker_elapsed_ms >= 0


@pytest.mark.asyncio
async def test_reranker_cannot_add_candidate():
    store = InMemoryMemoryStore()
    a = memory("a", "one")
    outsider = memory("x", "not retrieved")
    store.add(a)
    vector = FakeVectorRetriever([RetrievedMemory(a, 0.9)])

    async def bad(candidates, **kwargs):
        return [RetrievedMemory(outsider, 1.0)]

    retriever = AsyncHybridMemoryRetriever(
        store,
        vector_retriever=vector,
        fusion=ReciprocalRankFusion(lexical_weight=0, vector_weight=1),
        reranker=CallableAsyncMemoryReranker(bad),
    )
    with pytest.raises(ValueError):
        await retriever.retrieve_with_trace_async(character_id="c", query="one", limit=1)


def test_retrieval_metrics_are_correct():
    case1 = RetrievalEvalCase("c", "q1", ("a", "b"), "1")
    case2 = RetrievalEvalCase("c", "q2", ("z",), "2")
    results = (
        RetrievalEvalCaseResult(case1, ("a", "x", "b")),
        RetrievalEvalCaseResult(case2, ("x", "z", "y")),
    )
    metrics = compute_metrics(results, k=2)
    assert metrics.recall_at_k == pytest.approx((0.5 + 1.0) / 2)
    assert metrics.precision_at_k == pytest.approx((0.5 + 0.5) / 2)
    assert metrics.mrr == pytest.approx((1.0 + 0.5) / 2)
    # case1 DCG=1, IDCG=1+1/log2(3); case2 DCG=1/log2(3), IDCG=1
    expected_ndcg = (1/(1+1/1.584962500721156) + 1/1.584962500721156) / 2
    assert metrics.ndcg_at_k == pytest.approx(expected_ndcg)
    assert metrics.hit_rate_at_k == 1.0

@pytest.mark.asyncio
async def test_context_rewrite_can_improve_lexical_recall():
    from ai_character_engine.memory import ContextAppendingQueryRewriter

    store = InMemoryMemoryStore()
    target = memory("movie", "Kurosawa Seven Samurai favorite movie")
    store.add(target)
    vector = FakeVectorRetriever([])

    baseline = AsyncHybridMemoryRetriever(
        store,
        vector_retriever=vector,
        fusion=ReciprocalRankFusion(lexical_weight=1, vector_weight=0),
    )
    before = await baseline.retrieve_with_trace_async(
        character_id="c", query="that director", limit=1
    )
    assert before.memories == ()

    rewritten = AsyncHybridMemoryRetriever(
        store,
        vector_retriever=vector,
        fusion=ReciprocalRankFusion(lexical_weight=1, vector_weight=0),
        query_rewriter=ContextAppendingQueryRewriter(),
    )
    after = await rewritten.retrieve_with_trace_async(
        character_id="c",
        query="that director",
        limit=1,
        rewrite_context=("We were discussing Kurosawa and Seven Samurai.",),
    )
    assert [item.record.id for item in after.memories] == ["movie"]
    assert after.trace.vector_executed is False
    assert after.trace.rewrite_used is True

@pytest.mark.asyncio
async def test_compare_retrievers_uses_same_eval_cases():
    from ai_character_engine.memory import compare_retrievers, LexicalMemoryRetriever
    store = InMemoryMemoryStore()
    store.add(memory("a", "coffee preference"))
    cases = (RetrievalEvalCase("c", "coffee", ("a",), "one"),)
    report = await compare_retrievers(
        {"lexical-a": LexicalMemoryRetriever(store), "lexical-b": LexicalMemoryRetriever(store)},
        cases,
        k=1,
    )
    assert set(report.reports) == {"lexical-a", "lexical-b"}
    assert report.reports["lexical-a"].metrics.hit_rate_at_k == 1.0
