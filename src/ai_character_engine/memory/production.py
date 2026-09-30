from __future__ import annotations

import math
from dataclasses import replace
from datetime import UTC, datetime
from time import perf_counter
from collections.abc import Mapping

from .embedding import (
    AsyncEmbeddingProvider,
    EmbeddingProvider,
    content_hash,
    embed_many_async,
    memory_text,
    stored_embedding,
)
from .models import RetrievedMemory
from .qdrant import QdrantPoint, QdrantVectorIndex
from .ranking import AsyncMemoryReranker, MemoryReranker, ReciprocalRankFusion, rerank_async
from .retriever import LexicalMemoryRetriever
from .query import QueryRewriter
from .store import MemoryStore
from .trace import RetrievalCandidateTrace, RetrievalResult, RetrievalTrace


class QdrantVectorMemoryRetriever:
    """Async production vector retriever backed by Qdrant."""

    def __init__(
        self,
        store: MemoryStore,
        *,
        embedding: EmbeddingProvider | AsyncEmbeddingProvider,
        index: QdrantVectorIndex,
        min_score: float = math.nextafter(0.0, 1.0),
        metadata_filter: Mapping[str, object] | None = None,
    ) -> None:
        if (
            index.dimensions != embedding.dimensions
            or index.model_id != embedding.model_id
        ):
            raise ValueError("index and embedding provider must share vector space")
        if not math.isfinite(min_score) or not -1 <= min_score <= 1:
            raise ValueError("min_score must be finite and between -1 and 1")
        self.store = store
        self.embedding = embedding
        self.index = index
        self.min_score = min_score
        self.metadata_filter = dict(metadata_filter or {})

    async def retrieve_with_trace_async(
        self,
        *,
        character_id: str,
        query: str,
        limit: int = 5,
        now: datetime | None = None,
    ) -> RetrievalResult:
        start = perf_counter()
        trace = RetrievalTrace(
            strategy="qdrant-vector",
            character_id=character_id,
            query=query,
            requested_limit=limit,
            embedding_model_id=self.embedding.model_id,
            embedding_dimensions=self.embedding.dimensions,
            min_vector_score=self.min_score,
            vector_executed=True,
            index_backend="qdrant",
        )
        if limit <= 0:
            return RetrievalResult((), replace(trace, elapsed_ms=(perf_counter()-start)*1000))

        records = self.store.list_for_character(character_id)
        active = [r for r in records if r.character_id == character_id and r.is_active]

        qdrant_fingerprints = await self.index.fingerprints(character_id=character_id)
        existing_vectors: dict[str, tuple[float, ...]] = {}
        need_embed = []
        cached_ids: set[str] = set()
        for record in active:
            fingerprint = content_hash(record)
            if qdrant_fingerprints.get(record.id) == fingerprint:
                cached_ids.add(record.id)
                continue
            vector = stored_embedding(record, self.embedding)
            if vector is None:
                need_embed.append(record)
            else:
                existing_vectors[record.id] = vector

        embed_start = perf_counter()
        new_vectors = await embed_many_async(
            self.embedding, [memory_text(record) for record in need_embed]
        )
        embedding_ms = (perf_counter() - embed_start) * 1000
        generated = {record.id: vector for record, vector in zip(need_embed, new_vectors)}

        points = []
        for record in active:
            if record.id in cached_ids:
                continue
            vector = existing_vectors.get(record.id) or generated[record.id]
            points.append(
                QdrantPoint(
                    memory_id=record.id,
                    vector=tuple(vector),
                    payload={
                        "memory_id": record.id,
                        "character_id": character_id,
                        "status": record.status,
                        "model_id": self.embedding.model_id,
                        "content_hash": content_hash(record),
                        "kind": record.kind,
                        "tags": list(record.tags),
                        "importance": record.importance,
                        "created_at": record.created_at.isoformat(),
                        "metadata": record.metadata,
                    },
                )
            )

        stats = await self.index.sync_points(
            character_id=character_id,
            points=points,
            active_memory_ids=[record.id for record in active],
            existing_fingerprints=qdrant_fingerprints,
        )
        if not active or not query.strip():
            return RetrievalResult(
                (),
                replace(
                    trace,
                    total_records=len(records),
                    active_records=len(active),
                    filtered_inactive=sum(not r.is_active for r in records),
                    embedded_records=len(need_embed),
                    reused_embeddings=len(existing_vectors),
                    cached_vectors=stats.unchanged,
                    removed_vectors=stats.removed,
                    embedding_elapsed_ms=embedding_ms,
                    vector_elapsed_ms=(perf_counter()-start)*1000,
                    elapsed_ms=(perf_counter()-start)*1000,
                ),
            )

        query_embed_start = perf_counter()
        query_vector = (await embed_many_async(self.embedding, [query]))[0]
        embedding_ms += (perf_counter() - query_embed_start) * 1000
        search_start = perf_counter()
        hits = await self.index.search(
            character_id=character_id,
            vector=query_vector,
            limit=max(limit, len(active)),
            min_score=self.min_score,
            metadata_filter=self.metadata_filter,
        )
        search_ms = (perf_counter() - search_start) * 1000
        by_id = {r.id: r for r in active}
        selected = tuple(
            RetrievedMemory(by_id[hit.memory_id], hit.score)
            for hit in hits[:limit]
            if hit.memory_id in by_id
        )
        selected_ids = tuple(item.record.id for item in selected)
        elapsed = (perf_counter()-start)*1000
        return RetrievalResult(
            selected,
            replace(
                trace,
                total_records=len(records),
                active_records=len(active),
                filtered_inactive=sum(not r.is_active for r in records),
                embedded_records=len(need_embed),
                reused_embeddings=len(existing_vectors),
                cached_vectors=stats.unchanged,
                removed_vectors=stats.removed,
                selected_memory_ids=selected_ids,
                candidates=tuple(
                    RetrievalCandidateTrace(
                        memory_id=hit.memory_id,
                        vector_score=hit.score,
                        vector_rank=rank,
                        final_score=hit.score,
                        selected=hit.memory_id in selected_ids,
                    )
                    for rank, hit in enumerate(hits, 1)
                ),
                embedding_elapsed_ms=embedding_ms,
                vector_search_elapsed_ms=search_ms,
                vector_elapsed_ms=elapsed,
                elapsed_ms=elapsed,
            ),
        )


class AsyncHybridMemoryRetriever:
    """Hybrid lexical + async vector retrieval for production RAG/memory."""

    def __init__(
        self,
        store: MemoryStore,
        *,
        vector_retriever: QdrantVectorMemoryRetriever,
        candidate_limit: int = 20,
        fusion: ReciprocalRankFusion | None = None,
        reranker: MemoryReranker | AsyncMemoryReranker | None = None,
        query_rewriter: QueryRewriter | None = None,
    ) -> None:
        if candidate_limit < 1:
            raise ValueError("candidate_limit must be positive")
        self.store = store
        self.lexical = LexicalMemoryRetriever(store)
        self.vector = vector_retriever
        self.candidate_limit = candidate_limit
        self.fusion = fusion or ReciprocalRankFusion()
        self.reranker = reranker
        self.query_rewriter = query_rewriter

    async def retrieve_with_trace_async(
        self,
        *,
        character_id: str,
        query: str,
        limit: int = 5,
        now: datetime | None = None,
        rewrite_context: tuple[str, ...] = (),
    ) -> RetrievalResult:
        start = perf_counter()
        now = now or datetime.now(UTC)
        if limit <= 0:
            return RetrievalResult(
                (),
                RetrievalTrace(
                    strategy="async-hybrid",
                    character_id=character_id,
                    query=query,
                    requested_limit=limit,
                    lexical_weight=self.fusion.lexical_weight,
                    vector_weight=self.fusion.vector_weight,
                    elapsed_ms=(perf_counter()-start)*1000,
                ),
            )
        original_query = query
        rewrite_start = perf_counter()
        if self.query_rewriter is not None:
            query = await self.query_rewriter.rewrite(
                query, character_id=character_id, context=rewrite_context
            )
        rewrite_ms = (perf_counter() - rewrite_start) * 1000
        pool_limit = max(limit, self.candidate_limit)
        args = dict(character_id=character_id, query=query, limit=pool_limit, now=now)

        lexical = None
        if self.fusion.lexical_weight > 0:
            lexical = self.lexical.retrieve_with_trace(**args)
        vector = None
        if self.fusion.vector_weight > 0:
            vector = await self.vector.retrieve_with_trace_async(**args)

        lexical_memories = lexical.memories if lexical else ()
        vector_memories = vector.memories if vector else ()
        fused = self.fusion.fuse(lexical_memories, vector_memories)
        normalized = [
            RetrievedMemory(item.record, item.score / self.fusion.max_score)
            for item in fused
        ]
        rerank_start = perf_counter()
        ranked = await rerank_async(self.reranker, normalized, query=query, now=now)
        rerank_ms = (perf_counter() - rerank_start) * 1000
        allowed = {item.record.id: item.record for item in fused}
        if len({item.record.id for item in ranked}) != len(ranked) or any(
            item.record.id not in allowed
            or item.record != allowed[item.record.id]
            or not math.isfinite(item.score)
            for item in ranked
        ):
            raise ValueError("reranker must return unique, finite-scored input candidates")

        selected = tuple(ranked[:limit])
        selected_ids = tuple(item.record.id for item in selected)
        lex_scores = {
            item.record.id: (rank, item.score)
            for rank, item in enumerate(lexical_memories, 1)
        }
        vec_scores = {
            item.record.id: (rank, item.score)
            for rank, item in enumerate(vector_memories, 1)
        }
        fused_scores = {item.record.id: item.score for item in fused}
        candidates = []
        for item in ranked:
            mid = item.record.id
            lr, ls = lex_scores.get(mid, (None, None))
            vr, vs = vec_scores.get(mid, (None, None))
            candidates.append(
                RetrievalCandidateTrace(
                    memory_id=mid,
                    lexical_score=ls,
                    vector_score=vs,
                    lexical_rank=lr,
                    vector_rank=vr,
                    fusion_score=fused_scores[mid],
                    final_score=item.score,
                    selected=mid in selected_ids,
                )
            )

        base = vector.trace if vector else lexical.trace
        elapsed = (perf_counter()-start)*1000
        return RetrievalResult(
            selected,
            replace(
                base,
                strategy="async-hybrid",
                requested_limit=limit,
                candidates=tuple(candidates),
                selected_memory_ids=selected_ids,
                candidate_limit=pool_limit,
                fusion_k=self.fusion.k,
                lexical_weight=self.fusion.lexical_weight,
                vector_weight=self.fusion.vector_weight,
                reranker=type(self.reranker).__name__ if self.reranker else None,
                reranker_used=self.reranker is not None,
                candidate_count=len(ranked),
                final_count=len(selected),
                query_original=original_query,
                query_rewritten=query,
                rewrite_used=self.query_rewriter is not None and query != original_query,
                rewrite_elapsed_ms=rewrite_ms,
                reranker_elapsed_ms=rerank_ms,
                lexical_executed=lexical is not None,
                vector_executed=vector is not None,
                lexical_elapsed_ms=lexical.trace.elapsed_ms if lexical else 0.0,
                vector_elapsed_ms=vector.trace.elapsed_ms if vector else 0.0,
                embedding_elapsed_ms=vector.trace.embedding_elapsed_ms if vector else 0.0,
                vector_search_elapsed_ms=vector.trace.vector_search_elapsed_ms if vector else 0.0,
                index_backend=vector.trace.index_backend if vector else None,
                elapsed_ms=elapsed,
            ),
        )
